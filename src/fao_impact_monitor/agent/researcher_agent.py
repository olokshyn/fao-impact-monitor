"""Evidence-processing researcher agent with citation-preserving research loop."""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Literal

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

# Import data_source before Metric to avoid the metric ↔ data_source package cycle.
import fao_impact_monitor.data_source  # noqa: F401
from fao_impact_monitor.agent.query_generator_agent import (
    EvidenceGapInput,
    ResearchQuery,
    generate_research_queries,
    normalize_query,
)
from fao_impact_monitor.config import (
    AwsBedrockConfig,
    ResearcherConfig,
    get_config,
)
from fao_impact_monitor.data_lake.vectorstore import ChunkHit, VectorStore
from fao_impact_monitor.data_provider.web_scout_provider import (
    WebResearchFn,
    WebScoutProviderError,
    WebSource,
    run_web_scout_research,
)
from fao_impact_monitor.metric.metric import Metric
from fao_impact_monitor.utils.country import iso3_to_country_name

logger = logging.getLogger(__name__)

GenerateResearchQueriesFn = Callable[..., Awaitable[list[ResearchQuery]]]

CLAIM_EXTRACTION_SYSTEM = """\
You are a claim-extraction agent for evidence-based metric research.

Extract ONLY verbatim quotations from the provided source texts that are
relevant to the selected country and the metric or its documented gaps.

Critical rules:
1. Prefer claims that provide quantitative evidence (percentages, hectares,
   tonnes, heads of livestock, production change, area affected, people
   affected when tied to agricultural impact).
2. Prefer claims from newer / more recent sources over older ones when both
   are available (more recent publication year, report date, or data period).
3. quoted_text MUST be an exact contiguous substring of the source text.
   Do not rewrite, clean up, correct, paraphrase, or invent quotations.
4. Prefer the smallest self-contained quotation that preserves meaning.
   Include neighboring context only when needed for country, units, dates,
   or qualifiers.
5. Extract only claims clearly about the selected country unless the metric
   explicitly requires cross-country comparison. A country name elsewhere in
   a long chunk is not enough.
6. Useful evidence includes: affected cropland/pasture/rangeland area or
   share; production/yield loss; drought/flood severity by zone; El Niño
   event attribution and years; subnational geography named in the source.
7. If a source has no relevant claim, return no claims for that source.
8. Never use Metric.example or general knowledge as evidence.
9. Do not invent claim_id values that collide with existing ids; leave
   claim_id empty or temporary — the system assigns stable ids.
"""

GAP_ANALYSIS_SYSTEM = """\
You are an evidence-gap analyst for country-specific metric research.

Given validated claims only, identify what is known and what is still missing
to answer the metric for the selected country. Do not invent facts. Do not
treat Metric.example as evidence.

Rules:
1. Take Metric.name, Metric.description, Metric.unit, and Metric.example as
   given. Do NOT invent gaps that critique or redefine the metric wording.
2. Prefer quantitative completeness: missing percentages, hectares, tonnes,
   heads of livestock, production/yield change, or other Metric.unit values
   are high-priority gaps. Purely narrative weather descriptions ("heavy
   rains began") without magnitudes are not enough to close the metric.
3. Only mark blocking gaps that still prevent reporting useful national or
   subnational findings for El Niño-related agricultural impacts.
4. Missing coverage for some events/years is a remaining gap, not a reason to
   discard solid findings already established for other events.
5. When summarizing what is known, prefer findings from newer / more recent
   sources over older ones if they conflict or supersede earlier figures.
6. Keep established_facts concrete and claim-grounded; include numbers and
   units whenever the claims provide them.
"""

SUFFICIENCY_SYSTEM = """\
You are an evidence-sufficiency judge for country-specific metric research.

Decide whether the validated claims support a useful, citation-backed answer
for the selected country. The desired answer is QUANTITATIVE and aligned with
Metric.unit / Metric.example depth (percentages, area, production change,
livestock loss, etc.). A narrative-only finding ("heavy rains began",
"severe flooding occurred") is not sufficient by itself.

Set is_sufficient=true only when claims support at least one quantitative
finding for the metric (number + unit or clear magnitude of change), even if
not every event since 1997 is covered.

Do NOT require a complete inventory of all El Niño events, perfect hectare
numerators and denominators for every geography, or resolution of metric-
definition debates before drafting.

Prefer newer / more recent sources when claims conflict.

Return next_action:
- draft_answer if claims support one or more concrete findings (especially
  quantitative; also draft when only qualitative findings exist so a
  best-effort answer can still be written), even with remaining gaps
- generate_more_queries if claims are too thin / lack needed quantities but
  more research could help
- return_insufficient_evidence ONLY when there are essentially no usable
  country-specific findings and further research is unlikely to help

Never refuse to draft merely because coverage is incomplete. Always prefer
seeking or preserving quantitative evidence over qualitative narrative alone.
"""

ANSWER_SYSTEM = """\
You are an answer-statement generator for evidence-based metric research.

Write atomic factual statements that answer the metric for the selected
country using ONLY the validated claims provided. Never use general knowledge
or Metric.example as factual content. Metric.example is style/depth guidance
only.

Primary goal: answer with QUANTITATIVE data whenever the claims allow —
percentages, hectares/area, tonnes/production, yield change, livestock heads
lost, people/households affected when tied to the metric, and other values
in Metric.unit. Prefer statements that report magnitudes over purely
narrative descriptions of weather or events ("heavy rains began", "floods
occurred") when both are available.

When the claims only partially cover the metric, still emit every statement
that is supported. Incomplete event coverage is fine — report the events and
geographies you can support, and omit the rest without inventing numbers.
If only qualitative claims exist, still draft those statements (best-effort);
do not invent numbers to fill gaps.

Critical rules:
1. Preserve and foreground quantitative information from the claims (include
   the number and unit in the statement text).
2. Each statement makes one independently verifiable assertion.
3. Every factual statement must cite one or more supporting_claim_ids.
4. Preserve all material qualifiers from claims (country, date/period, unit,
   population, geography, uncertainty, observed vs estimated/projected,
   correlation vs causation).
5. Prefer newer / more recent sources over older ones when claims conflict,
   overlap, or offer alternative figures for the same aspect.
6. Accept best available evidence for agricultural impact: affected area or
   share of cultivated/pasture land, production/yield loss percentages,
   severity by zone, and El Niño-attributed event impacts. State clearly
   what quantity the source measured.
7. Do not calculate unless inputs and formula are supported by claims and
   required by the metric.
8. Do not invent facts absent from the claims.
"""

VERIFY_SYSTEM = """\
You are an entailment verifier. Use ONLY the provided statement and cited
claims. Do not use general knowledge.

Verdicts:
- entailed: the cited claims jointly support the statement
- partially_entailed: the core factual content is supported but some wording
  is slightly broader than the claims
- contradicted: claims conflict with the statement
- insufficient: claims do not support the statement

Accept statements that faithfully report source quantities (including
production loss, affected people/livestock, drought severity, or area
impacts) with preserved qualifiers. Reject only clear over-claims: wrong
country/period/unit, invented numbers, unsupported causation, or dropping
material uncertainty/estimate language. WebScout summaries are not evidence.
"""

REPAIR_SYSTEM = """\
You revise an answer statement so it is entailed by its cited claims.
Tighten wording to the claims; preserve material qualifiers and uncertainty.
Prefer keeping quantitative magnitudes (numbers and units) from the claims.
Do not invent facts. Prefer a narrower true statement over remove=true.
Set remove=true only when nothing claim-supported remains.
"""

ResearcherStatus = Literal["answered", "high_level_answer", "cannot_answer"]

STATUS_DISPLAY: dict[ResearcherStatus, str] = {
    "answered": "answered",
    "high_level_answer": "high level answer, lacking detailed evidence",
    "cannot_answer": "cannot answer with available evidence",
}

# Years/months alone are not metric quantities ("rains began in 1997").
_QUANTITATIVE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\d+(?:[.,]\d+)?\s*%",
        r"\d+(?:[.,]\d+)?\s*(?:percent|per\s*cent)\b",
        r"\d+(?:[.,]\d+)?\s*(?:ha\b|hectares?\b|acres?\b|km²|km2\b)",
        r"\d+(?:[.,]\d+)?\s*(?:tonnes?\b|tons?\b|kg\b|mt\b)",
        r"\d+(?:[.,]\d+)?\s*(?:million|billion)\b",
        (
            r"(?:fell|rose|increased|decreased|declined|dropped|grew|lost|"
            r"loss(?:es)?)\s+(?:by\s+)?\d"
        ),
        (
            r"\d+(?:[.,]\d+)?\s*(?:heads?\b|animals?\b|cattle\b|livestock\b|"
            r"people\b|farmers?\b|households?\b)"
        ),
    )
)


class EvidenceClaim(BaseModel):
    claim_id: str
    source_type: Literal["vectorstore", "web"]
    source_id: str
    quoted_text: str
    country: str
    relevance: str
    metric_aspects: list[str] = Field(default_factory=list)
    page_number: int | None = None
    section: str | None = None
    url: str
    match_kind: Literal["exact", "normalized"] | None = None


class ExtractedClaimCandidate(BaseModel):
    source_id: str
    quoted_text: str
    country: str
    relevance: str
    metric_aspects: list[str] = Field(default_factory=list)
    page_number: int | None = None
    section: str | None = None
    url: str | None = None


class ExtractedClaimList(BaseModel):
    claims: list[ExtractedClaimCandidate] = Field(default_factory=list)


class RejectedClaim(BaseModel):
    source_id: str
    quoted_text: str
    reason: str


class EvidenceGap(BaseModel):
    gap_id: str
    description: str
    why_required: str
    preferred_source_type: str | None = None
    suggested_terms: list[str] = Field(default_factory=list)
    status: Literal["open", "closed", "unresolvable"] = "open"


class EvidenceGapList(BaseModel):
    gaps: list[EvidenceGap] = Field(default_factory=list)
    established_facts: list[str] = Field(default_factory=list)


class EvidenceSufficiency(BaseModel):
    is_sufficient: bool
    supported_metric_aspects: list[str] = Field(default_factory=list)
    open_gap_ids: list[str] = Field(default_factory=list)
    reasoning: str
    next_action: Literal[
        "generate_more_queries",
        "draft_answer",
        "return_insufficient_evidence",
    ]
    needs_web: bool = False


class StatementCitation(BaseModel):
    document_name: str
    document_uri: str
    page_number: int | None = None
    origin: str | None = None


class AnswerStatement(BaseModel):
    statement_id: str
    text: str
    supporting_claim_ids: list[str] = Field(default_factory=list)
    metric_aspects: list[str] = Field(default_factory=list)
    citations: list[StatementCitation] = Field(default_factory=list)


class AnswerStatementList(BaseModel):
    statements: list[AnswerStatement] = Field(default_factory=list)


class StatementVerification(BaseModel):
    statement_id: str
    verdict: Literal[
        "entailed",
        "partially_entailed",
        "contradicted",
        "insufficient",
    ]
    unsupported_parts: list[str] = Field(default_factory=list)
    reasoning: str
    suggested_revision: str | None = None


class StatementRepair(BaseModel):
    statement_id: str
    text: str
    supporting_claim_ids: list[str] = Field(default_factory=list)
    remove: bool = False


class SourceReference(BaseModel):
    source_id: str
    source_type: Literal["vectorstore", "web"]
    document_id: str | None = None
    chunk_index: int | None = None
    document_uri: str
    document_name: str
    page_number: int | None = None
    document_source: str | None = None


class RetrievedChunk(BaseModel):
    source_id: str
    document_id: str
    chunk_index: int
    document_url: str
    document_title: str | None = None
    document_source: str | None = None
    chunk_text: str
    countries_iso3: list[str] = Field(default_factory=list)
    retrieval_query: str
    score: float | None = None
    page_number: int | None = None


class ResearcherOutput(BaseModel):
    status: ResearcherStatus
    country: str
    metric_name: str
    final_summary: str
    statements: list[AnswerStatement] = Field(default_factory=list)
    claims: list[EvidenceClaim] = Field(default_factory=list)
    sources: list[SourceReference] = Field(default_factory=list)
    open_gaps: list[EvidenceGap] = Field(default_factory=list)
    research_iterations: int


class ResearchState(BaseModel):
    metric: Metric
    country_iso3: str
    country_name: str
    research_iteration: int = 0
    current_queries: list[ResearchQuery] = Field(default_factory=list)
    all_queries: list[ResearchQuery] = Field(default_factory=list)
    executed_queries: list[str] = Field(default_factory=list)
    vector_chunks: list[RetrievedChunk] = Field(default_factory=list)
    web_sources: list[WebSource] = Field(default_factory=list)
    validated_claims: list[EvidenceClaim] = Field(default_factory=list)
    rejected_claims: list[RejectedClaim] = Field(default_factory=list)
    gaps: list[EvidenceGap] = Field(default_factory=list)
    established_facts: list[str] = Field(default_factory=list)
    sufficiency: EvidenceSufficiency | None = None
    draft_statements: list[AnswerStatement] = Field(default_factory=list)
    verified_statements: list[AnswerStatement] = Field(default_factory=list)
    verifications: list[StatementVerification] = Field(default_factory=list)
    claim_extraction_retries: int = 0
    answer_verification_retries: int = 0
    next_claim_seq: int = 1
    next_statement_seq: int = 1
    weak_terms: list[str] = Field(default_factory=list)
    needs_web: bool = False
    termination_reason: str | None = None
    output: ResearcherOutput | None = None


def build_chat_model(
    *,
    llm_model: str,
    aws_bedrock_config: AwsBedrockConfig | None = None,
) -> BaseChatModel:
    aws = aws_bedrock_config or get_config().aws_bedrock
    kwargs: dict[str, Any] = {
        "api_key": aws.api_key.get_secret_value(),
        "base_url": aws.base_url,
        "use_responses_api": True,
    }
    chat_model = init_chat_model(llm_model, **kwargs)
    if not isinstance(chat_model, BaseChatModel):
        raise TypeError(f"Expected BaseChatModel, got {type(chat_model)}")
    return chat_model


def normalize_for_quote_match(text: str) -> str:
    """Normalize harmless formatting differences for quotation matching."""
    text = unicodedata.normalize("NFKC", text)
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u00ad": "",  # soft hyphen
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    # Rejoin hyphenation caused by line wrapping: "agri-\nculture" -> "agriculture"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", " ", text)
    return text.strip()


def match_quoted_text(
    quoted_text: str,
    source_text: str,
) -> Literal["exact", "normalized"] | None:
    """Return match kind if quote exists in source, else None."""
    if not quoted_text or not source_text:
        return None
    if quoted_text in source_text:
        return "exact"
    if normalize_for_quote_match(quoted_text) in normalize_for_quote_match(source_text):
        return "normalized"
    return None


def vector_source_id(document_id: Any, chunk_index: int) -> str:
    return f"vs:{document_id}:{chunk_index}"


def chunk_from_hit(hit: ChunkHit, retrieval_query: str) -> RetrievedChunk:
    doc_id = str(hit.document_id)
    page_number = hit.chunk_index + 1
    return RetrievedChunk(
        source_id=vector_source_id(doc_id, hit.chunk_index),
        document_id=doc_id,
        chunk_index=hit.chunk_index,
        document_url=hit.document_url,
        document_title=hit.document_title,
        document_source=hit.document_source,
        chunk_text=hit.chunk_text,
        countries_iso3=list(hit.countries_iso3),
        retrieval_query=retrieval_query,
        score=hit.score,
        page_number=page_number,
    )


def _source_text_map(state: ResearchState) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for chunk in state.vector_chunks:
        mapping[chunk.source_id] = chunk.chunk_text
    for source in state.web_sources:
        mapping[source.source_id] = source.content
    return mapping


def _source_meta(state: ResearchState) -> dict[str, SourceReference]:
    refs: dict[str, SourceReference] = {}
    for chunk in state.vector_chunks:
        refs[chunk.source_id] = SourceReference(
            source_id=chunk.source_id,
            source_type="vectorstore",
            document_id=chunk.document_id,
            chunk_index=chunk.chunk_index,
            document_uri=chunk.document_url,
            document_name=chunk.document_title or chunk.document_url,
            page_number=chunk.page_number,
            document_source=chunk.document_source,
        )
    for source in state.web_sources:
        refs[source.source_id] = SourceReference(
            source_id=source.source_id,
            source_type="web",
            document_uri=source.url,
            document_name=source.title or source.url,
            page_number=source.page_number,
            document_source=None,
        )
    return refs


def format_source_origin(
    *,
    source_type: str,
    document_source: str | None = None,
) -> str:
    """Human-readable origin label for citations / references."""
    if source_type == "web":
        return "web search"
    raw = (document_source or "").strip().casefold().replace("_", "").replace(" ", "")
    if raw == "tellus":
        return "tellus"
    if raw in {"faorepository", "faoknowledgerepository"}:
        return "FAORepository"
    if document_source and document_source.strip():
        return document_source.strip()
    return "vectorstore"


def _claim_fingerprint(quoted_text: str, source_id: str) -> str:
    return f"{source_id}::{normalize_for_quote_match(quoted_text)}"


async def _structured_invoke(
    model: BaseChatModel,
    schema: type[BaseModel],
    *,
    system: str,
    user: str,
) -> BaseModel:
    structured = model.with_structured_output(schema)
    result = await structured.ainvoke(
        [SystemMessage(content=system), HumanMessage(content=user)]
    )
    if isinstance(result, schema):
        return result
    return schema.model_validate(result)


def _metric_prompt_block(metric: Metric, country_name: str, country_iso3: str) -> str:
    return (
        f"Metric.name:\n{metric.name}\n\n"
        f"Metric.description:\n{metric.description}\n\n"
        f"Metric.unit:\n{metric.unit}\n\n"
        f"Country: {country_name} ({country_iso3})\n"
    )


def _format_citation(citation: StatementCitation) -> str:
    if citation.page_number is not None:
        label = f"{citation.document_name}, p. {citation.page_number}"
    else:
        label = citation.document_name
    return f"([{label}]({citation.document_uri}))"


def resolve_statement_citations(
    statement: AnswerStatement,
    claims: Sequence[EvidenceClaim],
    sources: dict[str, SourceReference],
) -> list[StatementCitation]:
    claim_by_id = {c.claim_id: c for c in claims}
    citations: list[StatementCitation] = []
    seen: set[tuple[str, int | None]] = set()
    for claim_id in statement.supporting_claim_ids:
        claim = claim_by_id.get(claim_id)
        if claim is None:
            continue
        ref = sources.get(claim.source_id)
        if ref is None:
            continue
        key = (ref.document_uri, ref.page_number)
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            StatementCitation(
                document_name=ref.document_name,
                document_uri=ref.document_uri,
                page_number=ref.page_number,
                origin=format_source_origin(
                    source_type=ref.source_type,
                    document_source=ref.document_source,
                ),
            )
        )
    return citations


def build_final_summary(statements: Sequence[AnswerStatement]) -> str:
    parts: list[str] = []
    for statement in statements:
        cite_bits = [_format_citation(c) for c in statement.citations]
        if cite_bits:
            parts.append(f"{statement.text} {' '.join(cite_bits)}")
        else:
            parts.append(statement.text)
    return "\n\n".join(parts)


def statements_have_quantitative_evidence(
    statements: Sequence[AnswerStatement],
) -> bool:
    """True if any statement text reports a metric-relevant quantity."""
    return any(
        any(pattern.search(statement.text) for pattern in _QUANTITATIVE_PATTERNS)
        for statement in statements
    )


def classify_researcher_status(
    statements: Sequence[AnswerStatement],
) -> ResearcherStatus:
    """Map drafted findings to the report status label.

    - answered: at least one quantitative finding
    - high_level_answer: findings exist but lack detailed quantitative evidence
    - cannot_answer: no supported findings to answer the metric
    """
    if not statements:
        return "cannot_answer"
    if statements_have_quantitative_evidence(statements):
        return "answered"
    return "high_level_answer"


def build_answered_summary(
    *,
    statements: Sequence[AnswerStatement],
    gaps: Sequence[EvidenceGap],
) -> str:
    """Findings first; remaining open gaps second (if any)."""
    parts: list[str] = []
    if statements:
        parts.append(build_final_summary(statements))
    unresolved = [g for g in gaps if g.status in {"open", "unresolvable"}]
    if unresolved:
        parts.append("### Remaining evidence gaps")
        parts.append("\n".join(_format_gap_markdown(g) for g in unresolved))
    return "\n\n".join(parts) if parts else ""


def _format_gap_markdown(gap: EvidenceGap) -> str:
    """Format one open/unresolvable gap, including soft source references."""
    lines = [f"- **{gap.gap_id}**: {gap.description} ({gap.why_required})"]
    refs: list[str] = []
    if gap.preferred_source_type:
        refs.append(f"preferred source: `{gap.preferred_source_type}`")
    if gap.suggested_terms:
        terms = ", ".join(f"`{t}`" for t in gap.suggested_terms)
        refs.append(f"suggested terms: {terms}")
    for ref in refs:
        lines.append(f"  - {ref}")
    return "\n".join(lines)


def build_high_level_summary(
    *,
    statements: Sequence[AnswerStatement],
    gaps: Sequence[EvidenceGap],
) -> str:
    """Best-effort qualitative findings, then remaining gaps."""
    return build_answered_summary(statements=statements, gaps=gaps)


def build_cannot_answer_summary(
    *,
    country_name: str,
    gaps: Sequence[EvidenceGap],
    statements: Sequence[AnswerStatement],
) -> str:
    """Explain that evidence cannot answer; still include any findings/gaps."""
    parts: list[str] = [
        (f"The available evidence cannot answer this metric for {country_name}.")
    ]
    unresolved = [g for g in gaps if g.status in {"open", "unresolvable"}]
    if unresolved:
        parts.append("### Evidence gaps")
        parts.append("\n".join(_format_gap_markdown(g) for g in unresolved))
    if statements:
        parts.append("### Supported findings")
        parts.append(build_final_summary(statements))
    elif not unresolved:
        parts.append("No supported findings or unresolved gaps were recorded.")
    return "\n\n".join(parts)


def build_insufficient_summary(
    *,
    country_name: str,
    gaps: Sequence[EvidenceGap],
    statements: Sequence[AnswerStatement],
) -> str:
    """Backward-compatible alias for cannot-answer summaries."""
    return build_cannot_answer_summary(
        country_name=country_name,
        gaps=gaps,
        statements=statements,
    )


def build_status_summary(
    *,
    status: ResearcherStatus,
    country_name: str,
    statements: Sequence[AnswerStatement],
    gaps: Sequence[EvidenceGap],
) -> str:
    """Build the markdown body for a researcher status (without the Status line)."""
    if status == "answered":
        return build_answered_summary(statements=statements, gaps=gaps)
    if status == "high_level_answer":
        return build_high_level_summary(statements=statements, gaps=gaps)
    return build_cannot_answer_summary(
        country_name=country_name,
        gaps=gaps,
        statements=statements,
    )


def _example_leaked(text: str, example: str) -> bool:
    sample = example.strip()
    if len(sample) < 40:
        return False
    # Detect long contiguous reuse of the example as evidence/content.
    window = sample[:80]
    return window in text


async def research(
    *,
    metric: Metric,
    country_iso3: str,
    vector_store: VectorStore,
    config: ResearcherConfig | None = None,
    model: BaseChatModel | None = None,
    verifier_model: BaseChatModel | None = None,
    query_model: BaseChatModel | None = None,
    web_research_fn: WebResearchFn | None = None,
    generate_research_queries_fn: GenerateResearchQueriesFn | None = None,
) -> ResearcherOutput:
    """Run the bounded iterative researcher loop and return structured output."""
    cfg = config or get_config().researcher
    country_name = iso3_to_country_name(country_iso3)
    main_model = model or build_chat_model(llm_model=cfg.llm_model)
    verify_model = verifier_model or build_chat_model(
        llm_model=cfg.verifier_llm_model,
    )
    query_gen = generate_research_queries_fn or generate_research_queries

    state = ResearchState(
        metric=metric,
        country_iso3=country_iso3.upper(),
        country_name=country_name,
    )
    logger.info(
        "Researcher START country=%s metric=%r max_iterations=%s",
        state.country_iso3,
        metric.name,
        cfg.max_research_iterations,
    )

    while state.research_iteration < cfg.max_research_iterations:
        state.research_iteration += 1
        remaining = cfg.max_research_iterations - state.research_iteration
        logger.info(
            "Researcher STAGE=iteration %s/%s remaining=%s country=%s metric=%r",
            state.research_iteration,
            cfg.max_research_iterations,
            remaining,
            state.country_iso3,
            metric.name,
        )

        open_gaps = [g for g in state.gaps if g.status == "open"]
        preferred: list[Literal["vectorstore", "web", "both"]]
        if state.needs_web or (
            open_gaps
            and any(
                (g.preferred_source_type or "").lower() in {"web", "both"}
                for g in open_gaps
            )
        ):
            preferred = ["both", "web"]
        else:
            preferred = ["vectorstore", "both"]

        logger.info(
            "Researcher STAGE=queries open_gaps=%s preferred_destinations=%s",
            [g.gap_id for g in open_gaps],
            preferred,
        )
        try:
            queries = await query_gen(
                research_question=metric.name,
                explanation=metric.description,
                unit=metric.unit,
                country_name=country_name,
                country_iso3=state.country_iso3,
                example=metric.example,
                established_facts=state.established_facts,
                open_gaps=[
                    EvidenceGapInput(
                        gap_id=g.gap_id,
                        description=g.description,
                        why_required=g.why_required,
                        preferred_source_type=g.preferred_source_type,
                        suggested_terms=g.suggested_terms,
                    )
                    for g in open_gaps
                ],
                executed_queries=state.executed_queries,
                weak_terms=state.weak_terms,
                preferred_destinations=preferred,
                min_queries=1 if open_gaps else min(3, cfg.max_queries_per_iteration),
                max_queries=cfg.max_queries_per_iteration,
                model=query_model,
            )
        except Exception:
            logger.exception("Query generation failed")
            state.termination_reason = "query_generation_failure"
            break

        # Dedup against executed queries again at the researcher layer.
        fresh: list[ResearchQuery] = []
        executed_norm = {normalize_query(q) for q in state.executed_queries}
        for q in queries:
            key = normalize_query(q.query)
            if key in executed_norm:
                continue
            fresh.append(q)
            executed_norm.add(key)
        state.current_queries = fresh[: cfg.max_queries_per_iteration]
        state.all_queries.extend(state.current_queries)
        logger.info(
            "Researcher STAGE=queries done count=%s details=%s",
            len(state.current_queries),
            [(q.query, q.purpose, q.destination) for q in state.current_queries],
        )
        if not state.current_queries:
            state.termination_reason = "no_new_queries"
            break

        logger.info("Researcher STAGE=vector_retrieve")
        new_chunks: list[RetrievedChunk] = []
        known_chunk_ids = {c.source_id for c in state.vector_chunks}
        for q in state.current_queries:
            state.executed_queries.append(q.query)
            if not q.query.strip():
                logger.warning("Researcher skipping empty vector query")
                continue
            if q.destination in {"vectorstore", "both"}:
                try:
                    hits = await vector_store.search(
                        q.query,
                        countries_iso3=[state.country_iso3],
                        limit=cfg.vector_results_per_query,
                    )
                except Exception:
                    logger.exception("Vector store search failed for %r", q.query)
                    state.weak_terms.append(q.query)
                    continue
                if not hits:
                    state.weak_terms.append(q.query)
                for hit in hits:
                    chunk = chunk_from_hit(hit, q.query)
                    if chunk.source_id in known_chunk_ids:
                        continue
                    known_chunk_ids.add(chunk.source_id)
                    state.vector_chunks.append(chunk)
                    new_chunks.append(chunk)

        # Conditional web research only when a prior sufficiency decision (or an
        # explicit web-only follow-up) says material gaps need the open web.
        # Cap web-scout calls by both the configured limit and the number of
        # queries actually produced for this round (never more than generated).
        web_queries = [
            q for q in state.current_queries if q.destination in {"web", "both"}
        ]
        web_budget = min(
            cfg.max_web_queries_per_iteration,
            len(state.current_queries),
            len(web_queries),
        )
        run_web = web_budget > 0 and state.needs_web
        new_web: list[WebSource] = []
        if run_web:
            selected_web = web_queries[:web_budget]
            logger.info(
                "Researcher STAGE=web_retrieve queries=%s budget=%s generated=%s",
                [q.query for q in selected_web],
                web_budget,
                len(state.current_queries),
            )
            known_web = {s.source_id for s in state.web_sources}
            for q in selected_web:
                try:
                    mapped = await run_web_scout_research(
                        q.query,
                        domain_expertise="food and agriculture statistics",
                        include_domains=[
                            "fao.org",
                            "un.org",
                            "worldbank.org",
                            "data.worldbank.org",
                        ],
                        web_research_fn=web_research_fn,
                    )
                except WebScoutProviderError:
                    logger.warning(
                        "WebScout failed; continuing with vector evidence",
                        exc_info=True,
                    )
                    continue
                for source in mapped.sources:
                    if source.source_id in known_web:
                        continue
                    known_web.add(source.source_id)
                    state.web_sources.append(source)
                    new_web.append(source)
        else:
            logger.info(
                "Researcher STAGE=web_retrieve skipped needs_web=%s web_candidates=%s",
                state.needs_web,
                len(web_queries),
            )

        logger.info(
            "Researcher STAGE=retrieve done new_chunks=%s new_web=%s "
            "totals chunks=%s web=%s",
            len(new_chunks),
            len(new_web),
            len(state.vector_chunks),
            len(state.web_sources),
        )

        # Claim extraction + validation with retries.
        state.claim_extraction_retries = 0
        newly_validated: list[EvidenceClaim] = []
        while True:
            logger.info(
                "Researcher STAGE=extract_claims retry=%s/%s",
                state.claim_extraction_retries,
                cfg.max_claim_extraction_retries,
            )
            candidates = await _extract_claims(
                state,
                model=main_model,
                new_chunks=new_chunks,
                new_web=new_web,
            )
            logger.info(
                "Researcher STAGE=validate_claims candidates=%s", len(candidates)
            )
            accepted, rejected = _validate_claim_candidates(state, candidates)
            newly_validated.extend(accepted)
            state.rejected_claims.extend(rejected)
            reject_reasons: dict[str, int] = {}
            for item in rejected:
                reject_reasons[item.reason] = reject_reasons.get(item.reason, 0) + 1
            logger.info(
                "Researcher STAGE=validate_claims accepted=%s rejected=%s "
                "validated_total=%s reject_reasons=%s",
                len(accepted),
                len(rejected),
                len(state.validated_claims),
                reject_reasons,
            )
            if accepted or state.claim_extraction_retries >= (
                cfg.max_claim_extraction_retries
            ):
                break
            if not candidates:
                break
            state.claim_extraction_retries += 1

        logger.info("Researcher STAGE=gap_analysis")
        await _assess_gaps(state, model=main_model)
        logger.info(
            "Researcher STAGE=gap_analysis done open=%s closed=%s facts=%s",
            sum(1 for g in state.gaps if g.status == "open"),
            sum(1 for g in state.gaps if g.status == "closed"),
            len(state.established_facts),
        )
        logger.info("Researcher STAGE=sufficiency")
        await _assess_sufficiency(state, model=main_model, config=cfg)
        assert state.sufficiency is not None
        logger.info(
            "Researcher STAGE=sufficiency done sufficient=%s next_action=%s "
            "open_gaps=%s needs_web=%s",
            state.sufficiency.is_sufficient,
            state.sufficiency.next_action,
            state.sufficiency.open_gap_ids,
            state.sufficiency.needs_web,
        )
        state.needs_web = state.sufficiency.needs_web

        if state.sufficiency.next_action == "draft_answer":
            logger.info("Researcher STAGE=draft_answer")
            await _draft_and_verify(
                state,
                model=main_model,
                verifier_model=verify_model,
                config=cfg,
            )
            if state.verified_statements:
                status = classify_researcher_status(state.verified_statements)
                state.termination_reason = status
                output = _finalize(state, status=status)
                logger.info(
                    "Researcher END status=%s statements=%s claims=%s iterations=%s",
                    status,
                    len(output.statements),
                    len(output.claims),
                    output.research_iterations,
                )
                return output
            # Could not produce verified statements — try more research if possible.
            if state.research_iteration >= cfg.max_research_iterations:
                state.termination_reason = "verification_failed_limit"
                break
            logger.info(
                "Researcher STAGE=draft_answer produced no verified statements; "
                "continuing research"
            )
            continue

        if state.sufficiency.next_action == "return_insufficient_evidence":
            state.termination_reason = "cannot_answer"
            break

        # generate_more_queries
        if state.research_iteration >= cfg.max_research_iterations:
            state.termination_reason = "research_iteration_limit"
            break
        logger.info(
            "Researcher STAGE=generate_more_queries continuing to next iteration"
        )

    # Insufficient / exhausted — still draft every statement the claims support.
    if state.validated_claims and not state.verified_statements:
        logger.info(
            "Researcher STAGE=draft_answer (partial findings from available claims)"
        )
        state.sufficiency = EvidenceSufficiency(
            is_sufficient=False,
            supported_metric_aspects=list(
                state.sufficiency.supported_metric_aspects
                if state.sufficiency is not None
                else []
            ),
            open_gap_ids=[g.gap_id for g in state.gaps if g.status == "open"],
            reasoning=(
                state.sufficiency.reasoning
                if state.sufficiency is not None
                else "Research incomplete; drafting supported findings only."
            ),
            next_action="draft_answer",
        )
        await _draft_and_verify(
            state,
            model=main_model,
            verifier_model=verify_model,
            config=cfg,
        )

    if state.verified_statements:
        status = classify_researcher_status(state.verified_statements)
        state.termination_reason = state.termination_reason or status
        output = _finalize(state, status=status)
        logger.info(
            "Researcher END status=%s statements=%s claims=%s "
            "gaps=%s reason=%s iterations=%s",
            status,
            len(output.statements),
            len(output.claims),
            len(output.open_gaps),
            state.termination_reason,
            output.research_iterations,
        )
        return output

    state.termination_reason = state.termination_reason or "cannot_answer"
    output = _finalize(state, status="cannot_answer")
    logger.info(
        "Researcher END status=cannot_answer statements=%s claims=%s "
        "gaps=%s reason=%s iterations=%s",
        len(output.statements),
        len(output.claims),
        len(output.open_gaps),
        state.termination_reason,
        output.research_iterations,
    )
    return output


async def _extract_claims(
    state: ResearchState,
    *,
    model: BaseChatModel,
    new_chunks: Sequence[RetrievedChunk],
    new_web: Sequence[WebSource],
) -> list[ExtractedClaimCandidate]:
    if not new_chunks and not new_web:
        return []
    source_blocks: list[str] = []
    for chunk in new_chunks:
        source_blocks.append(
            f"[source_id={chunk.source_id} type=vectorstore "
            f"url={chunk.document_url} page={chunk.page_number}]\n"
            f"{chunk.chunk_text}"
        )
    for source in new_web:
        source_blocks.append(
            f"[source_id={source.source_id} type=web url={source.url}]\n"
            f"{source.content}"
        )
    gaps = (
        "\n".join(
            f"- {g.gap_id}: {g.description}" for g in state.gaps if g.status == "open"
        )
        or "(none)"
    )
    user = (
        f"{_metric_prompt_block(state.metric, state.country_name, state.country_iso3)}"
        f"\nOpen gaps:\n{gaps}\n\n"
        "Extract verbatim claims from these newly retrieved sources only.\n\n"
        + "\n\n---\n\n".join(source_blocks)
    )
    result = await _structured_invoke(
        model,
        ExtractedClaimList,
        system=CLAIM_EXTRACTION_SYSTEM,
        user=user,
    )
    assert isinstance(result, ExtractedClaimList)
    return result.claims


def _validate_claim_candidates(
    state: ResearchState,
    candidates: Sequence[ExtractedClaimCandidate],
) -> tuple[list[EvidenceClaim], list[RejectedClaim]]:
    texts = _source_text_map(state)
    metas = _source_meta(state)
    existing = {
        _claim_fingerprint(c.quoted_text, c.source_id) for c in state.validated_claims
    }
    accepted: list[EvidenceClaim] = []
    rejected: list[RejectedClaim] = []
    for cand in candidates:
        source_text = texts.get(cand.source_id)
        if source_text is None:
            rejected.append(
                RejectedClaim(
                    source_id=cand.source_id,
                    quoted_text=cand.quoted_text,
                    reason="unknown_source_id",
                )
            )
            continue
        match_kind = match_quoted_text(cand.quoted_text, source_text)
        if match_kind is None:
            rejected.append(
                RejectedClaim(
                    source_id=cand.source_id,
                    quoted_text=cand.quoted_text,
                    reason="substring_validation_failed",
                )
            )
            continue
        country_ok = (
            state.country_name.casefold() in cand.quoted_text.casefold()
            or state.country_iso3.casefold() in cand.quoted_text.casefold()
            or state.country_name.casefold() in cand.country.casefold()
            or state.country_iso3.casefold() in cand.country.casefold()
        )
        # Also accept if the quote clearly refers to the country via candidate
        # country field matching selected country.
        short = re.split(r"[,(\[]", state.country_name.casefold(), maxsplit=1)[0]
        if short and short in cand.quoted_text.casefold():
            country_ok = True
        if not country_ok:
            rejected.append(
                RejectedClaim(
                    source_id=cand.source_id,
                    quoted_text=cand.quoted_text,
                    reason="not_country_specific",
                )
            )
            continue
        fp = _claim_fingerprint(cand.quoted_text, cand.source_id)
        if fp in existing:
            rejected.append(
                RejectedClaim(
                    source_id=cand.source_id,
                    quoted_text=cand.quoted_text,
                    reason="duplicate_claim",
                )
            )
            continue
        meta = metas[cand.source_id]
        claim_id = f"claim_{state.next_claim_seq:03d}"
        state.next_claim_seq += 1
        claim = EvidenceClaim(
            claim_id=claim_id,
            source_type=meta.source_type,
            source_id=cand.source_id,
            quoted_text=cand.quoted_text,
            country=cand.country or state.country_name,
            relevance=cand.relevance,
            metric_aspects=list(cand.metric_aspects),
            page_number=cand.page_number
            if cand.page_number is not None
            else meta.page_number,
            section=cand.section,
            url=cand.url or meta.document_uri,
            match_kind=match_kind,
        )
        if state.metric.example and _example_leaked(
            claim.quoted_text, state.metric.example
        ):
            rejected.append(
                RejectedClaim(
                    source_id=cand.source_id,
                    quoted_text=cand.quoted_text,
                    reason="metric_example_leak",
                )
            )
            continue
        state.validated_claims.append(claim)
        existing.add(fp)
        accepted.append(claim)
    return accepted, rejected


async def _assess_gaps(state: ResearchState, *, model: BaseChatModel) -> None:
    claims_block = (
        "\n".join(f"- {c.claim_id}: {c.quoted_text}" for c in state.validated_claims)
        or "(no validated claims yet)"
    )
    prior = (
        "\n".join(f"- {g.gap_id} [{g.status}]: {g.description}" for g in state.gaps)
        or "(none)"
    )
    user = (
        f"{_metric_prompt_block(state.metric, state.country_name, state.country_iso3)}"
        f"\nValidated claims:\n{claims_block}\n\nPrior gaps:\n{prior}\n\n"
        "Return the full updated gap list and established_facts inventory."
    )
    result = await _structured_invoke(
        model,
        EvidenceGapList,
        system=GAP_ANALYSIS_SYSTEM,
        user=user,
    )
    assert isinstance(result, EvidenceGapList)
    # Preserve stable gap ids when possible; accept model list as current.
    if result.gaps:
        state.gaps = result.gaps
    elif not state.validated_claims and not state.gaps:
        state.gaps = [
            EvidenceGap(
                gap_id="gap_001",
                description="Missing country-specific evidence for the metric",
                why_required="No validated claims yet",
                preferred_source_type="vectorstore",
                suggested_terms=[state.metric.name, state.country_name],
                status="open",
            )
        ]
    if result.established_facts:
        state.established_facts = result.established_facts
    else:
        state.established_facts = [
            f"{c.claim_id}: {c.quoted_text}" for c in state.validated_claims
        ]


async def _assess_sufficiency(
    state: ResearchState,
    *,
    model: BaseChatModel,
    config: ResearcherConfig,
) -> None:
    claims_block = (
        "\n".join(f"- {c.claim_id}: {c.quoted_text}" for c in state.validated_claims)
        or "(none)"
    )
    gaps_block = (
        "\n".join(
            f"- {g.gap_id} [{g.status}]: {g.description} ({g.why_required})"
            for g in state.gaps
        )
        or "(none)"
    )
    iterations_left = config.max_research_iterations - state.research_iteration
    user = (
        f"{_metric_prompt_block(state.metric, state.country_name, state.country_iso3)}"
        f"\nValidated claims:\n{claims_block}\n\nGaps:\n{gaps_block}\n\n"
        f"Research iterations remaining after this one: {iterations_left}\n"
        "Set needs_web=true only if a material gap requires authoritative web "
        "sources beyond the vector store."
    )
    result = await _structured_invoke(
        model,
        EvidenceSufficiency,
        system=SUFFICIENCY_SYSTEM,
        user=user,
    )
    assert isinstance(result, EvidenceSufficiency)
    # Prefer drafting a best-effort answer whenever we already have claims.
    if state.validated_claims and result.next_action == "return_insufficient_evidence":
        result.next_action = "draft_answer"
    if iterations_left <= 0 and result.next_action == "generate_more_queries":
        if state.validated_claims:
            result.next_action = "draft_answer"
        else:
            result.next_action = "return_insufficient_evidence"
            result.is_sufficient = False
    if not state.validated_claims and result.next_action == "draft_answer":
        result.next_action = (
            "generate_more_queries"
            if iterations_left > 0
            else "return_insufficient_evidence"
        )
        result.is_sufficient = False
    state.sufficiency = result


async def _draft_and_verify(
    state: ResearchState,
    *,
    model: BaseChatModel,
    verifier_model: BaseChatModel,
    config: ResearcherConfig,
) -> None:
    claims_block = "\n".join(
        f"- {c.claim_id} (source={c.source_id}, url={c.url}, "
        f"page={c.page_number}): {c.quoted_text}"
        for c in state.validated_claims
    )
    style = (
        "Style/depth guidance ONLY from Metric.example (never copy facts):\n"
        f"{state.metric.example}"
    )
    user = (
        f"{_metric_prompt_block(state.metric, state.country_name, state.country_iso3)}"
        f"\n{style}\n\nValidated claims:\n{claims_block}\n\n"
        "Produce atomic answer statements with supporting_claim_ids. "
        "Prioritize quantitative findings (numbers + units matching the metric) "
        "over qualitative narrative alone. "
        "Prefer newer / more recent sources over older ones when claims "
        "conflict or overlap."
    )
    drafted = await _structured_invoke(
        model,
        AnswerStatementList,
        system=ANSWER_SYSTEM,
        user=user,
    )
    assert isinstance(drafted, AnswerStatementList)
    claim_ids = {c.claim_id for c in state.validated_claims}
    statements: list[AnswerStatement] = []
    for item in drafted.statements:
        supporting = [cid for cid in item.supporting_claim_ids if cid in claim_ids]
        if not supporting:
            continue
        if state.metric.example and _example_leaked(item.text, state.metric.example):
            continue
        sid = item.statement_id or f"stmt_{state.next_statement_seq:03d}"
        state.next_statement_seq += 1
        statements.append(
            AnswerStatement(
                statement_id=sid,
                text=item.text.strip(),
                supporting_claim_ids=supporting,
                metric_aspects=list(item.metric_aspects),
            )
        )
    state.draft_statements = statements
    logger.info(
        "Researcher STAGE=draft_answer done draft_statements=%s",
        len(statements),
    )

    verified: list[AnswerStatement] = []
    for statement in statements:
        current = statement
        for attempt in range(config.max_answer_verification_retries + 1):
            logger.info(
                "Researcher STAGE=verify_statements statement=%s attempt=%s/%s",
                current.statement_id,
                attempt,
                config.max_answer_verification_retries,
            )
            verification = await _verify_statement(
                state,
                current,
                model=verifier_model,
            )
            state.verifications.append(verification)
            logger.info(
                "Researcher STAGE=verify_statements statement=%s verdict=%s",
                current.statement_id,
                verification.verdict,
            )
            if verification.verdict == "entailed":
                verified.append(current)
                break
            if (
                verification.verdict == "partially_entailed"
                and attempt >= config.max_answer_verification_retries
            ):
                # Keep a narrowed statement rather than discarding usable evidence.
                logger.info(
                    "Researcher STAGE=verify_statements statement=%s "
                    "accepting partially_entailed after retries",
                    current.statement_id,
                )
                verified.append(current)
                break
            if attempt >= config.max_answer_verification_retries:
                logger.info(
                    "Researcher STAGE=verify_statements statement=%s dropped "
                    "verdict=%s",
                    current.statement_id,
                    verification.verdict,
                )
                break
            repaired = await _repair_statement(
                state,
                current,
                verification,
                model=model,
            )
            if repaired is None:
                logger.info(
                    "Researcher STAGE=verify_statements statement=%s "
                    "repair removed statement",
                    current.statement_id,
                )
                break
            current = repaired
    state.verified_statements = verified
    logger.info(
        "Researcher STAGE=verify_statements done verified=%s drafted=%s",
        len(verified),
        len(statements),
    )


async def _verify_statement(
    state: ResearchState,
    statement: AnswerStatement,
    *,
    model: BaseChatModel,
) -> StatementVerification:
    cited = [
        c
        for c in state.validated_claims
        if c.claim_id in statement.supporting_claim_ids
    ]
    claims_block = (
        "\n".join(f"- {c.claim_id}: {c.quoted_text}" for c in cited) or "(no claims)"
    )
    user = (
        f"Country: {state.country_name} ({state.country_iso3})\n"
        f"Metric: {state.metric.name}\n"
        f"Statement ({statement.statement_id}): {statement.text}\n\n"
        f"Cited claims:\n{claims_block}\n"
    )
    result = await _structured_invoke(
        model,
        StatementVerification,
        system=VERIFY_SYSTEM,
        user=user,
    )
    assert isinstance(result, StatementVerification)
    return StatementVerification(
        statement_id=statement.statement_id,
        verdict=result.verdict,
        unsupported_parts=list(result.unsupported_parts),
        reasoning=result.reasoning,
        suggested_revision=result.suggested_revision,
    )


async def _repair_statement(
    state: ResearchState,
    statement: AnswerStatement,
    verification: StatementVerification,
    *,
    model: BaseChatModel,
) -> AnswerStatement | None:
    cited = [
        c
        for c in state.validated_claims
        if c.claim_id in statement.supporting_claim_ids
    ]
    claims_block = "\n".join(f"- {c.claim_id}: {c.quoted_text}" for c in cited)
    user = (
        f"Statement: {statement.text}\n"
        f"Verification verdict: {verification.verdict}\n"
        f"Unsupported parts: {verification.unsupported_parts}\n"
        f"Suggested revision: {verification.suggested_revision}\n"
        f"Reasoning: {verification.reasoning}\n\n"
        f"Cited claims:\n{claims_block}\n"
    )
    result = await _structured_invoke(
        model,
        StatementRepair,
        system=REPAIR_SYSTEM,
        user=user,
    )
    assert isinstance(result, StatementRepair)
    if result.remove or not result.text.strip():
        return None
    claim_ids = {c.claim_id for c in state.validated_claims}
    supporting = [cid for cid in result.supporting_claim_ids if cid in claim_ids]
    if not supporting:
        supporting = list(statement.supporting_claim_ids)
    return AnswerStatement(
        statement_id=statement.statement_id,
        text=result.text.strip(),
        supporting_claim_ids=supporting,
        metric_aspects=list(statement.metric_aspects),
    )


def _finalize(
    state: ResearchState,
    *,
    status: ResearcherStatus,
) -> ResearcherOutput:
    sources = _source_meta(state)
    statements: list[AnswerStatement] = []
    for statement in state.verified_statements:
        citations = resolve_statement_citations(
            statement, state.validated_claims, sources
        )
        statements.append(statement.model_copy(update={"citations": citations}))

    cited_claim_ids = {cid for s in statements for cid in s.supporting_claim_ids}
    claims = [c for c in state.validated_claims if c.claim_id in cited_claim_ids]
    cited_source_ids = {c.source_id for c in claims}
    source_list = [sources[sid] for sid in cited_source_ids if sid in sources]

    # Re-classify from finalized statements so status matches report content.
    status = classify_researcher_status(statements) if statements else status
    if not statements:
        status = "cannot_answer"

    final_summary = build_status_summary(
        status=status,
        country_name=state.country_name,
        statements=statements,
        gaps=state.gaps,
    )

    output = ResearcherOutput(
        status=status,
        country=state.country_name,
        metric_name=state.metric.name,
        final_summary=final_summary,
        statements=statements,
        claims=claims,
        sources=source_list,
        open_gaps=[g for g in state.gaps if g.status != "closed"],
        research_iterations=state.research_iteration,
    )
    state.output = output
    return output
