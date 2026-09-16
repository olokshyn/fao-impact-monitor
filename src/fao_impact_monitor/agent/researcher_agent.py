"""Evidence-processing researcher agent with citation-preserving research loop."""

from __future__ import annotations

import base64
import hashlib
import logging
import re
import unicodedata
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

import pycountry
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

# Import data_source before Metric to avoid the metric ↔ data_source package cycle.
import fao_impact_monitor.data_source  # noqa: F401
from fao_impact_monitor.agent.query_generator_agent import (
    ResearchQuery,
    generate_research_queries,
    normalize_query,
)
from fao_impact_monitor.config import (
    AwsBedrockConfig,
    ResearcherConfig,
    get_config,
)
from fao_impact_monitor.data_lake.vectorstore import ChunkHit
from fao_impact_monitor.data_provider.web_scout_provider import (
    WebResearchFn,
    WebScoutProviderError,
    WebSource,
    run_web_scout_research,
)
from fao_impact_monitor.metric.metric import Metric
from fao_impact_monitor.utils.country import iso3_to_country_name
from fao_impact_monitor.utils.document_uri import markdown_document_target

logger = logging.getLogger(__name__)

GenerateResearchQueriesFn = Callable[..., Awaitable[list[ResearchQuery]]]


class ResearchVectorStore(Protocol):
    """Search interface accepted by the researcher agent."""

    async def search(
        self,
        query: str,
        *,
        countries_iso3: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> list[ChunkHit]: ...


CLAIM_EXTRACTION_SYSTEM = """\
You are a claim-extraction agent for evidence-based metric research.

Extract verbatim quotations from the provided source texts that help answer
the selected metric. Prefer quantitative evidence, but also keep qualitative
metric-subject findings and 2026-27 forecasts / current-condition outlooks.

Critical rules:
1. Extract quantitative evidence (percentages, hectares, tonnes, heads of
   livestock, production change, area affected, people affected when tied to
   agricultural impact) whenever present. Also extract a qualitative claim when
   it describes the metric subject itself (coping strategies, subsequent
   hazards, livelihood change, trade disruption, projected food insecurity) or
   a 2026-27 forecast / current agrifood starting point.
2. Classify answer_fit for every claim as one of:
   - direct_requested_unit: quantitatively measures the metric subject in the
     requested unit (or an explicitly convertible equivalent in the quotation)
   - direct_related_measure: quantitatively measures the SAME metric subject
     in a different unit or related measurement form. Prefer this over
     quantitative_proxy whenever the claim talks about the metric subject
     itself and reports numbers — exact unit match is NOT required
   - quantitative_proxy: a quantitative result that does NOT measure the
     metric subject itself but materially informs it (for example a hazard
     magnitude such as rainfall deficit when the metric is production impact,
     or a 2026 cereal/production outlook when the metric is yield or hunger)
   - direct_qualitative: describes the metric subject without a number
   - supporting_context: relevant hazard, forecast, response, or background
3. Prefer claims from newer / more recent sources over older ones when both
   are available (more recent publication year, report date, or data period).
   Keep 2026-27 outlook rows from tables and verified visual facts.
4. quoted_text MUST be an exact contiguous substring of the source text,
   including [verified visual facts] / [VERIFIED VISUAL FACT] lines.
   Do not rewrite, clean up, correct, paraphrase, or invent quotations.
5. source_id MUST be copied exactly from the source header (for example
   web:001 or vs:...). Never invent, truncate, or rewrite source_id values.
6. Prefer the smallest self-contained quotation that preserves meaning.
   Include neighboring context only when needed for country, units, dates,
   or qualifiers.
7. Vectorstore sources have already been filtered by trusted country metadata.
   Do not require the country name to appear in a vectorstore quotation.
   Web claims still need country context in the supplied web source.
8. Useful evidence includes the requested measurement, its numerator and
   denominator, directly convertible component measures, related quantitative
   measures of the metric subject (any unit), hazard magnitude, event
   attribution, time period, geography, and forward-looking production,
   price, trade, or food-security figures for the current/upcoming event.
9. Keep distinct figures or propositions as separate claims, including when
   they occur in the same source. Return no claim only for truly irrelevant
   sources.
10. Never use Metric.example or general knowledge as evidence. The example
   describes desired answer structure only.
11. Do not invent claim_id values that collide with existing ids; leave
   claim_id empty or temporary — the system assigns stable ids.
"""

CLAIM_USEFULNESS_SYSTEM = """\
You are a strict metric-answer judge. Decide whether each source-validated
claim can appear as a finding that answers the requested metric.

Classify each claim as:
- direct_answer: quantitatively measures the requested metric subject — either
  in the requested unit, a convertible equivalent, or another quantitative
  form that still reports a result for that same subject. Exact unit match
  with Metric.unit is NOT required when the quotation clearly talks about the
  metric subject and provides quantitative data (percentages, counts, area,
  production, yield, livestock, people/households affected when that is the
  subject, or similar magnitudes).
- context: materially helps interpret the answer but does NOT itself measure
  the metric subject. Examples include a quantitative change in a related
  hazard (rainfall deficit, drought duration), funding or response figures,
  or background that does not report a result for the metric subject.
- reject: irrelevant to the metric and to El Nino agrifood impacts (funding
  appeals, methodology, legends, missing-value notes, or a different topic).

For a quantitative metric, do not use numbers attached to the wrong subject as
direct_answer. A related hazard, production outlook, price, trade, or
food-security figure can be context when it materially helps interpret the
requested metric or the 2026-27 starting point. A quantitative result about
the metric subject itself is always direct_answer even when the unit differs
from Metric.unit. Reject funding, response targets, generic methodology,
legend categories, and statements that merely say a value is missing.

Context must still be specific, relevant evidence for the selected metric,
country El Nino event, or the current/upcoming (2026-27) outlook. Do not
classify something as context merely because it mentions El Nino or the
selected country. Do not reject a country or regional table row, chart fact,
or forecast solely because it is not in the exact requested unit.

Vectorstore claims have already been filtered by trusted country metadata, so
do not require the country name inside the quote. El Nino event context may also
be inherited from the source bundle. However, reject a claim whose explicit
date is outside the eligible event periods or whose measurement is only a
generic method rather than a result for the selected country.

Never treat a global / worldwide / aggregate-total agricultural figure as a
direct_answer for the selected country. Such claims may be context only, and
only when they clearly describe the global situation rather than the country.

Metric.example defines the desired answer shape only. Never treat it as
evidence. Return one verdict for every supplied claim_id and do not rewrite the
claim.
"""

ANSWER_SYSTEM = """\
You are an answer-statement generator for evidence-based metric research.

Write a comprehensive synthesis for the selected metric and country using ONLY
the validated claims provided. Never use general knowledge or Metric.example
as factual content. Metric.example is style/depth guidance only.

Each claim is labeled either answer or context. An answer quantitatively
measures the metric subject (including related quantitative forms in a
different unit). Context is relevant supporting evidence that must remain
clearly separate and must not be worded as though it answers the metric.

Primary goal: answer with QUANTITATIVE data for the selected country (or its
subnational units) whenever the claims allow — percentages, hectares/area,
tonnes/production, yield change, livestock heads lost, people/households
affected when tied to the metric, and other values that measure the metric
subject (requested unit or a related quantitative form). Prefer statements
that report magnitudes over purely narrative descriptions of weather or
events ("heavy rains began", "floods occurred") when both are available.
The answer must include quantitative country-level or subnational-level
figures from the claims when any such figures exist.

Temporal coverage: when quantitative national or subnational data for
different time periods is present, use all such data. Report each period's
figures and compare how the values changed year over year (or period over
period), citing the supporting claims. Do not drop older periods when newer
ones exist unless they are true duplicates of the same figure.

Geographic evidence scope:
- Use only evidence that applies to the selected country. A claim or source
  may mention other countries; ignore figures and facts for those other
  countries and never attribute them to the selected country.
- You may also use evidence for a broader geographic region that the
  selected country belongs to (e.g. East Africa, sub-Saharan Africa for
  Ethiopia), provided the claim states that regional scope. Always prefer
  more specific national or subnational data over broader regional data
  when both are available.
- If a claim reports a global, worldwide, or other aggregate that is not
  the selected country and not a region it belongs to, keep that scope
  only as context when needed; never reattribute such a figure to the
  selected country unless the claim itself names that country.

When the claims only partially cover the metric, still emit every statement
that is supported for the selected country (or an allowed broader region).
Incomplete event coverage is fine — report the events and geographies you
can support, and omit the rest without inventing numbers. If only
qualitative claims exist for the country, still draft those statements
(best-effort); do not invent numbers to fill gaps.

Critical rules:
1. Preserve and foreground quantitative information from the claims (include
   the number and unit in the statement text), especially country-level and
   subnational-level magnitudes.
2. Cover every supplied claim that is in scope for the selected country (or
   an allowed broader region it belongs to) in at least one independently
   verifiable statement. Skip claim content that only concerns other
   countries. Combine compatible claims when that makes the answer clearer.
3. Every factual statement must cite all claim_ids that support it. Never mix
   answer and context claims in the same statement.
4. Preserve all material qualifiers from claims (country, date/period, unit,
   population, geography, uncertainty, observed vs estimated/projected,
   correlation vs causation).
5. Geographic scope is material: never reattribute another country's figure,
   or a global/worldwide/unrelated-aggregate figure, to the selected
   country. When using regional evidence, keep the regional scope explicit.
6. Prefer newer / more recent sources over older ones when claims conflict,
   overlap, or offer alternative figures for the same aspect. Prefer
   national/subnational over broader regional figures for the same aspect.
7. Preserve the supplied answer/context distinction and state exactly what
   quantity or relationship the source measured.
8. Do not calculate unless inputs and formula are supported by claims and
   required by the metric. State any calculation transparently and preserve all
   stated causes and qualifiers.
9. Do not invent facts absent from the claims.
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

Accept statements that faithfully report source quantities with preserved
qualifiers. Reject only clear over-claims: wrong country/period/unit, invented
numbers, unsupported causation, or dropping material uncertainty/estimate
language. If a cited claim is global/worldwide/aggregate-total and the
statement attributes that quantity to the selected country, verdict is
contradicted (or insufficient). Accept transparent arithmetic when every input
is cited and all stated causes and qualifiers are preserved. WebScout summaries
are not evidence.
"""

REPAIR_SYSTEM = """\
You revise an answer statement so it is entailed by its cited claims.
Tighten wording to the claims; preserve material qualifiers and uncertainty.
Prefer keeping quantitative magnitudes (numbers and units) from the claims.
If the claim is global/worldwide/aggregate-total, restore that geographic scope
and remove any unsupported attribution to the selected country.
Do not invent facts. Prefer a narrower true statement over remove=true.
Set remove=true only when nothing claim-supported remains.
"""

VISUAL_ANALYSIS_SYSTEM = """\
You are a source-grounded visual evidence analyst. Analyze ONLY the attached
source image(s) and the accompanying canonical source text for the selected
country and metric.

Return atomic observations that add relevant information not already stated
in the canonical text. Preserve titles, axes, units, dates, geography,
categories, legends, uncertainty, and observed/forecast qualifiers. You may
describe direct comparisons or patterns visible in the image, but do not infer
causality, hidden values, or facts requiring outside knowledge. Every
observation must cite one or more supplied artifact_ids. Return no observation
when the visual is unreadable or adds nothing relevant.
"""

VISUAL_VERIFY_SYSTEM = """\
You are a visual entailment verifier. Use ONLY the attached source image(s).
For each proposed observation, return entailed only when the visible chart,
table, map, diagram, legend, caption, or labels directly support every material
part. Return insufficient for unreadable or missing support, and contradicted
for a visible conflict. Do not use the accompanying canonical text or outside
knowledge as visual support.
"""

ResearcherStatus = Literal["answered", "high_level_answer", "cannot_answer"]
StatementType = Literal["answer", "context"]
ClaimAnswerFit = Literal[
    "direct_requested_unit",
    "direct_related_measure",
    "quantitative_proxy",
    "direct_qualitative",
    "supporting_context",
]

STATUS_DISPLAY: dict[ResearcherStatus, str] = {
    "answered": "answered",
    "high_level_answer": "high level answer, lacking detailed evidence",
    "cannot_answer": "cannot answer with available evidence",
}

EL_NINO_EVENT_PERIODS = "1997-98, 2015-16, 2018-19, 2023-24, and 2026-27"
_ELIGIBLE_EVENT_YEARS = {
    1997,
    1998,
    2015,
    2016,
    2018,
    2019,
    2023,
    2024,
    2026,
    2027,
}
_YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
_PERCENT_PATTERN = re.compile(
    r"(?:\d+(?:[.,]\d+)?|zero|one|two|three|four|five|six|seven|eight|nine|"
    r"ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
    r"nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\s*"
    r"(?:%(?!\w)|percent\b|per\s*cent\b)",
    re.IGNORECASE,
)
_GLOBAL_SCOPE_PATTERN = re.compile(
    r"\b(?:at\s+(?:the\s+)?global\s+(?:agricultural\s+)?level|"
    r"global(?:ly)?\s+(?:(?:\w+)\s+){0,3}"
    r"(?:level|average|analysis|estimate|estimates|total|agriculture|"
    r"agricultural|cropping|cropland|crops?|areas?)|"
    r"worldwide(?:\s+(?:(?:\w+)\s+){0,3}"
    r"(?:level|average|analysis|estimate|estimates|total|agriculture|"
    r"agricultural|cropping|cropland|crops?|areas?))?|"
    r"globally)\b",
    re.IGNORECASE,
)
# Aggregate totals without a named country are treated as global-scope results.
_AGGREGATE_AREA_SCOPE_PATTERN = re.compile(
    r"\b(?:total\s+agricultural\s+(?:area|surface|land)|"
    r"agricultural\s+surface|"
    r"global\s+(?:agricultural|cropping|crop)\s+(?:area|areas|surface|land))\b",
    re.IGNORECASE,
)

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
            r"people\b|persons?\b|deaths?\b|farmers?\b|households?\b|"
            r"homes?\b|houses?\b|days?\b|months?\b)"
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
    statement_type: StatementType = "context"
    answer_fit: ClaimAnswerFit = "supporting_context"
    metric_aspects: list[str] = Field(default_factory=list)
    page_number: int | None = None
    section: str | None = None
    url: str
    match_kind: Literal["exact", "normalized"] | None = None
    evidence_modality: Literal["text", "verified_visual_fact"] = "text"
    visual_artifact_ids: list[str] = Field(default_factory=list)


class ExtractedClaimCandidate(BaseModel):
    source_id: str
    quoted_text: str
    country: str
    relevance: str
    answer_fit: ClaimAnswerFit = "supporting_context"
    metric_aspects: list[str] = Field(default_factory=list)
    page_number: int | None = None
    section: str | None = None
    url: str | None = None


class ExtractedClaimList(BaseModel):
    claims: list[ExtractedClaimCandidate] = Field(default_factory=list)


class ClaimUsefulnessVerdict(BaseModel):
    claim_id: str
    verdict: Literal["direct_answer", "context", "reject"]
    reason: str


class ClaimUsefulnessList(BaseModel):
    verdicts: list[ClaimUsefulnessVerdict] = Field(default_factory=list)


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


class StatementCitation(BaseModel):
    document_name: str
    document_uri: str
    page_number: int | None = None
    origin: str | None = None


class AnswerStatement(BaseModel):
    statement_id: str
    text: str
    statement_type: StatementType = "answer"
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


class VisualArtifact(BaseModel):
    artifact_id: str
    path: str
    sha256: str
    media_type: Literal["image/png", "image/jpeg", "image/webp", "image/gif"]
    physical_page: int


class VisualInsightCandidate(BaseModel):
    text: str
    artifact_ids: list[str] = Field(min_length=1)
    relevance: str


class VisualInsightList(BaseModel):
    insights: list[VisualInsightCandidate] = Field(default_factory=list)


class VisualInsightVerdict(BaseModel):
    insight_index: int
    verdict: Literal["entailed", "contradicted", "insufficient"]
    reasoning: str


class VisualInsightVerdictList(BaseModel):
    verdicts: list[VisualInsightVerdict] = Field(default_factory=list)


class VerifiedResearchVisualFact(BaseModel):
    text: str
    artifact_ids: list[str] = Field(min_length=1)
    verifier_verdict: Literal["entailed"] = "entailed"


class PdfEvidenceEvent(BaseModel):
    event_id: str
    relationship: str


class PdfVerifiedVisualFact(BaseModel):
    text: str


class SourceReference(BaseModel):
    source_id: str
    source_type: Literal["vectorstore", "web"]
    document_id: str | None = None
    chunk_index: int | None = None
    document_uri: str
    document_name: str
    page_number: int | None = None
    document_source: str | None = None
    source_text: str | None = None
    evidence_id: str | None = None
    physical_pages: list[int] = Field(default_factory=list)
    printed_pages: list[str] = Field(default_factory=list)
    events: list[PdfEvidenceEvent] = Field(default_factory=list)
    verified_visual_facts: list[PdfVerifiedVisualFact] = Field(default_factory=list)


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
    source_text: str | None = None
    evidence_id: str | None = None
    physical_pages: list[int] = Field(default_factory=list)
    printed_pages: list[str] = Field(default_factory=list)
    events: list[PdfEvidenceEvent] = Field(default_factory=list)
    verified_visual_facts: list[PdfVerifiedVisualFact] = Field(default_factory=list)
    visual_artifacts: list[VisualArtifact] = Field(default_factory=list)
    research_visual_facts: list[VerifiedResearchVisualFact] = Field(
        default_factory=list
    )


class QueryRunStat(BaseModel):
    """One executed retrieval query and how many results were kept."""

    query: str
    destination: Literal["vectorstore", "web"]
    results_returned: int = 0
    results_accepted: int = 0


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
    query_runs: list[QueryRunStat] = Field(default_factory=list)


class ResearchState(BaseModel):
    metric: Metric
    country_iso3: str
    country_name: str
    research_iteration: int = 0
    current_queries: list[ResearchQuery] = Field(default_factory=list)
    all_queries: list[ResearchQuery] = Field(default_factory=list)
    executed_queries: list[str] = Field(default_factory=list)
    query_runs: list[QueryRunStat] = Field(default_factory=list)
    vector_chunks: list[RetrievedChunk] = Field(default_factory=list)
    web_sources: list[WebSource] = Field(default_factory=list)
    validated_claims: list[EvidenceClaim] = Field(default_factory=list)
    rejected_claims: list[RejectedClaim] = Field(default_factory=list)
    gaps: list[EvidenceGap] = Field(default_factory=list)
    draft_statements: list[AnswerStatement] = Field(default_factory=list)
    verified_statements: list[AnswerStatement] = Field(default_factory=list)
    verifications: list[StatementVerification] = Field(default_factory=list)
    next_claim_seq: int = 1
    next_statement_seq: int = 1
    analyzed_visual_artifact_ids: set[str] = Field(default_factory=set)
    corroborating_claim_ids: dict[str, list[str]] = Field(default_factory=dict)
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


_HYPHEN_CHARS = {
    "\u2010",  # hyphen
    "\u2011",  # non-breaking hyphen
    "\u2012",  # figure dash
    "\u2013",  # en dash
    "\u2014",  # em dash
}


def normalize_for_quote_match(text: str) -> str:
    """Normalize harmless formatting differences for quotation matching."""
    text = unicodedata.normalize("NFKC", text)
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        **{char: "-" for char in _HYPHEN_CHARS},
        "\u00ad": "",  # soft hyphen
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    # Letter hyphenation across wraps: "agri-\nculture" -> "agriculture".
    # Keep digit ranges: "5-\n10" / "5–\n10" -> "5-10" (not "510").
    text = re.sub(r"(?<=[A-Za-z])-\s+(?=[A-Za-z])", "", text)
    text = re.sub(r"(?<=\d)-\s+(?=\d)", "-", text)
    # Collapse all whitespace (spaces, tabs, newlines) to a single space.
    return re.sub(r"\s+", " ", text).strip()


def quote_word_tokens(text: str) -> list[str]:
    """Tokenize text into alphanumeric words for punctuation-tolerant matching.

    Whitespace and punctuation (including hyphens) are dropped after the same
    hyphen/soft-wrap normalization used for substring matching, so PDF wraps
    like ``did \\nnot`` and ``5–\\n10`` align with cleaned LLM quotations.
    """
    normalized = normalize_for_quote_match(text).casefold()
    return re.findall(r"[a-z0-9]+", normalized)


def _words_are_contiguous_subsequence(
    needle: Sequence[str], haystack: Sequence[str]
) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    size = len(needle)
    for start in range(len(haystack) - size + 1):
        if haystack[start : start + size] == needle:
            return True
    return False


def match_quoted_text(
    quoted_text: str,
    source_text: str,
) -> Literal["exact", "normalized"] | None:
    """Return match kind if quote exists in source, else None.

    Matching order:
    1. exact contiguous substring
    2. whitespace/hyphen-normalized substring
    3. contiguous word-token list (ignores remaining punctuation)
    """
    if not quoted_text or not source_text:
        return None
    if quoted_text in source_text:
        return "exact"
    if normalize_for_quote_match(quoted_text) in normalize_for_quote_match(source_text):
        return "normalized"
    quote_words = quote_word_tokens(quoted_text)
    if quote_words and _words_are_contiguous_subsequence(
        quote_words, quote_word_tokens(source_text)
    ):
        return "normalized"
    return None


def vector_source_id(document_id: Any, chunk_index: int) -> str:
    return f"vs:{document_id}:{chunk_index}"


def chunk_from_hit(hit: ChunkHit, retrieval_query: str) -> RetrievedChunk:
    doc_id = str(hit.document_id)
    page_number = hit.chunk_index + 1
    raw_visual_artifacts = hit.document_meta.get("visual_artifacts", [])
    visual_artifacts = [
        VisualArtifact.model_validate(item)
        for item in raw_visual_artifacts
        if isinstance(item, dict)
    ]
    raw_events = hit.document_meta.get("events", [])
    events = [
        PdfEvidenceEvent.model_validate(item)
        for item in raw_events
        if isinstance(item, dict)
    ]
    raw_visual_facts = hit.document_meta.get("verified_visual_facts", [])
    verified_visual_facts = [
        PdfVerifiedVisualFact.model_validate(item)
        for item in raw_visual_facts
        if isinstance(item, dict)
    ]
    physical_pages = [
        int(page)
        for page in hit.document_meta.get("physical_pages", [])
        if isinstance(page, int)
    ]
    printed_pages = [
        str(page)
        for page in hit.document_meta.get("printed_pages", [])
        if isinstance(page, (str, int))
    ]
    evidence_id = hit.document_meta.get("evidence_id")
    source_text = hit.document_meta.get("source_text")
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
        source_text=source_text if isinstance(source_text, str) else None,
        evidence_id=evidence_id if isinstance(evidence_id, str) else None,
        physical_pages=physical_pages,
        printed_pages=printed_pages,
        events=events,
        verified_visual_facts=verified_visual_facts,
        visual_artifacts=visual_artifacts,
    )


def _visual_fact_texts(chunk: RetrievedChunk) -> list[str]:
    texts = [fact.text for fact in chunk.verified_visual_facts if fact.text]
    texts.extend(fact.text for fact in chunk.research_visual_facts if fact.text)
    return texts


def _source_text_map(state: ResearchState) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for chunk in state.vector_chunks:
        parts = [chunk.chunk_text]
        parts.extend(
            text for text in _visual_fact_texts(chunk) if text not in chunk.chunk_text
        )
        mapping[chunk.source_id] = "\n\n".join(part for part in parts if part)
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
            source_text=chunk.source_text,
            evidence_id=chunk.evidence_id,
            physical_pages=list(chunk.physical_pages),
            printed_pages=list(chunk.printed_pages),
            events=list(chunk.events),
            verified_visual_facts=list(chunk.verified_visual_facts),
        )
    for source in state.web_sources:
        refs[source.source_id] = SourceReference(
            source_id=source.source_id,
            source_type="web",
            document_uri=source.url,
            document_name=source.title or source.url,
            page_number=source.page_number,
            document_source=None,
            source_text=source.content,
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


def _visual_artifact_ids_for_quote(
    state: ResearchState, source_id: str, quoted_text: str
) -> list[str] | None:
    """Return artifact ids when the quote is a verified visual fact.

    An empty list is a valid match for ingestion-time facts whose source image
    was verified upstream but was not materialized as a research artifact.
    """
    chunk = next(
        (item for item in state.vector_chunks if item.source_id == source_id), None
    )
    if chunk is None:
        return None
    research_fact = next(
        (
            fact
            for fact in chunk.research_visual_facts
            if match_quoted_text(quoted_text, fact.text) is not None
        ),
        None,
    )
    if research_fact is not None:
        return list(research_fact.artifact_ids)
    if any(
        match_quoted_text(quoted_text, fact.text) is not None
        for fact in chunk.verified_visual_facts
    ):
        return [item.artifact_id for item in chunk.visual_artifacts]
    for match in re.finditer(
        r"\[(?:RESEARCH-TIME )?VERIFIED VISUAL FACT[^\]]*\]\s*"
        r"(?P<fact>.*?)(?=\n\n\[|\Z)",
        chunk.chunk_text,
        flags=re.DOTALL,
    ):
        fact_text = match.group("fact").strip()
        if match_quoted_text(quoted_text, fact_text) is not None:
            return [item.artifact_id for item in chunk.visual_artifacts]
    return None


async def _structured_invoke(
    model: BaseChatModel,
    schema: type[BaseModel],
    *,
    system: str,
    user: Any,
    method: Literal["function_calling", "json_mode", "json_schema"] | None = None,
) -> BaseModel:
    structured = (
        model.with_structured_output(schema, method=method)
        if method is not None
        else model.with_structured_output(schema)
    )
    result = await structured.ainvoke(
        [SystemMessage(content=system), HumanMessage(content=user)]
    )
    if isinstance(result, schema):
        return result
    return schema.model_validate(result)


def _visual_content_blocks(
    artifacts: Sequence[VisualArtifact],
) -> tuple[list[dict[str, Any]], list[VisualArtifact]]:
    """Load hash-verified PDF-pipeline images as multimodal message blocks."""
    root = get_config().pdf_pipeline.artifact_dir.resolve()
    blocks: list[dict[str, Any]] = []
    loaded: list[VisualArtifact] = []
    for artifact in artifacts:
        path = Path(artifact.path).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            logger.warning("Skipping visual artifact outside pipeline root: %s", path)
            continue
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != artifact.sha256:
            logger.warning("Skipping visual artifact with hash mismatch: %s", path)
            continue
        encoded = base64.b64encode(data).decode("ascii")
        blocks.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{artifact.media_type};base64,{encoded}",
                    "detail": "high",
                },
            }
        )
        loaded.append(artifact)
    return blocks, loaded


async def _enrich_visual_chunk(
    state: ResearchState,
    chunk: RetrievedChunk,
    *,
    model: BaseChatModel,
    verifier_model: BaseChatModel,
    max_artifacts: int,
) -> None:
    pending = [
        artifact
        for artifact in chunk.visual_artifacts
        if artifact.artifact_id not in state.analyzed_visual_artifact_ids
    ][:max_artifacts]
    image_blocks, loaded = _visual_content_blocks(pending)
    if not loaded:
        return
    artifact_manifest = "\n".join(
        f"- {artifact.artifact_id}: physical page {artifact.physical_page}"
        for artifact in loaded
    )
    analysis_text = (
        f"{_metric_prompt_block(state.metric, state.country_name, state.country_iso3)}"
        f"\nVisual artifacts:\n{artifact_manifest}\n\n"
        f"Canonical source text:\n{chunk.chunk_text}\n\n"
        "Return only additional metric-relevant observations directly visible in "
        "the attached source images."
    )
    analysis = await _structured_invoke(
        model,
        VisualInsightList,
        system=VISUAL_ANALYSIS_SYSTEM,
        user=[{"type": "text", "text": analysis_text}, *image_blocks],
        method="function_calling",
    )
    assert isinstance(analysis, VisualInsightList)
    loaded_ids = {artifact.artifact_id for artifact in loaded}
    candidates = [
        insight
        for insight in analysis.insights
        if insight.text.strip()
        and insight.artifact_ids
        and set(insight.artifact_ids).issubset(loaded_ids)
    ]
    state.analyzed_visual_artifact_ids.update(loaded_ids)
    if not candidates:
        return
    proposed = "\n".join(
        f"{index}. {insight.text}\n   artifact_ids={insight.artifact_ids}"
        for index, insight in enumerate(candidates)
    )
    verification = await _structured_invoke(
        verifier_model,
        VisualInsightVerdictList,
        system=VISUAL_VERIFY_SYSTEM,
        user=[
            {
                "type": "text",
                "text": f"Proposed observations:\n{proposed}",
            },
            *image_blocks,
        ],
        method="function_calling",
    )
    assert isinstance(verification, VisualInsightVerdictList)
    entailed = {
        verdict.insight_index
        for verdict in verification.verdicts
        if verdict.verdict == "entailed"
    }
    existing = {fact.text for fact in chunk.research_visual_facts}
    for index, insight in enumerate(candidates):
        if index not in entailed or insight.text in existing:
            continue
        fact = VerifiedResearchVisualFact(
            text=insight.text,
            artifact_ids=list(insight.artifact_ids),
        )
        chunk.research_visual_facts.append(fact)
        existing.add(fact.text)
        chunk.chunk_text += (
            "\n\n[RESEARCH-TIME VERIFIED VISUAL FACT | "
            f"physical page {chunk.page_number} | "
            f"artifacts {','.join(fact.artifact_ids)}]\n{fact.text}"
        )


async def _enrich_visual_chunks(
    state: ResearchState,
    chunks: Sequence[RetrievedChunk],
    *,
    model: BaseChatModel,
    verifier_model: BaseChatModel,
    max_chunks: int,
    max_artifacts_per_chunk: int,
) -> None:
    visual_chunks = [chunk for chunk in chunks if chunk.visual_artifacts][:max_chunks]
    for chunk in visual_chunks:
        try:
            await _enrich_visual_chunk(
                state,
                chunk,
                model=model,
                verifier_model=verifier_model,
                max_artifacts=max_artifacts_per_chunk,
            )
        except Exception:
            logger.exception(
                "Visual evidence analysis failed for source %s; using existing "
                "verified_visual_facts only",
                chunk.source_id,
            )


def _metric_prompt_block(metric: Metric, country_name: str, country_iso3: str) -> str:
    return (
        f"Metric.name:\n{metric.name}\n\n"
        f"Metric.description:\n{metric.description}\n\n"
        f"Metric.unit:\n{metric.unit}\n\n"
        f"Country: {country_name} ({country_iso3})\n"
        f"Eligible El Nino event periods: {EL_NINO_EVENT_PERIODS}\n"
    )


def _format_citation(citation: StatementCitation) -> str:
    if citation.page_number is not None:
        label = f"{citation.document_name}, p. {citation.page_number}"
    else:
        label = citation.document_name
    target = markdown_document_target(citation.document_uri)
    return f"([{label}]({target}))"


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


def _ensure_statement_citations(output: ResearcherOutput) -> list[AnswerStatement]:
    sources = {source.source_id: source for source in output.sources}
    filled: list[AnswerStatement] = []
    for statement in output.statements:
        if statement.citations:
            filled.append(statement)
            continue
        filled.append(
            statement.model_copy(
                update={
                    "citations": resolve_statement_citations(
                        statement, output.claims, sources
                    )
                }
            )
        )
    return filled


def format_human_markdown(
    output: ResearcherOutput,
    *,
    metric: Metric,
    section_number: int,
) -> str:
    """Render the human-readable twin of a researcher metric report.

    Uses already-validated statements, citations, and gaps. Does not call
    the model again.
    """
    statements = _ensure_statement_citations(output)
    answers = [
        statement for statement in statements if statement.statement_type == "answer"
    ]
    context = sorted(
        statements,
        key=lambda statement: (
            int(_contains_quantity(statement.text)),
            int(statement.statement_type == "answer"),
            -len(statement.text),
        ),
        reverse=True,
    )
    context_body = (
        build_final_summary(context)
        if context
        else "No relevant evidence was retained."
    )
    if answers:
        answer_body = build_final_summary(answers)
    else:
        answer_body = (
            f"The available evidence cannot answer this metric for {output.country}."
        )
    unresolved = [
        gap for gap in output.open_gaps if gap.status in {"open", "unresolvable"}
    ]
    gap_body = (
        "\n".join(_format_gap_markdown(gap) for gap in unresolved)
        if unresolved
        else "None."
    )
    unit = metric.unit or "(none)"
    return "\n".join(
        [
            f"# {section_number}. {metric.name}",
            "",
            f"**Description:** {metric.description}",
            "",
            f"**Example:** {metric.example}",
            "",
            f"**Unit:** {unit}",
            "",
            "## Context",
            "",
            context_body,
            "",
            "## Answer",
            "",
            answer_body,
            "",
            "## Gaps",
            "",
            gap_body,
            "",
        ]
    )


def statements_have_quantitative_evidence(
    statements: Sequence[AnswerStatement],
) -> bool:
    """True if any direct-answer statement reports a quantity."""
    return any(
        statement.statement_type == "answer" and _contains_quantity(statement.text)
        for statement in statements
    )


def classify_researcher_status(
    statements: Sequence[AnswerStatement],
) -> ResearcherStatus:
    """Map drafted findings to the report status label.

    Context statements never determine status.

    - answered: at least one quantitative answer statement
    - high_level_answer: answer statements exist but lack quantitative evidence
    - cannot_answer: no direct answer statement exists
    """
    answers = [
        statement for statement in statements if statement.statement_type == "answer"
    ]
    if not answers:
        return "cannot_answer"
    if statements_have_quantitative_evidence(answers):
        return "answered"
    return "high_level_answer"


def _build_typed_statement_summary(
    statements: Sequence[AnswerStatement],
) -> str:
    answers = [
        statement for statement in statements if statement.statement_type == "answer"
    ]
    context = [
        statement for statement in statements if statement.statement_type == "context"
    ]
    parts: list[str] = []
    if answers:
        parts.extend(["### Answer", build_final_summary(answers)])
    if context:
        parts.extend(["### Context", build_final_summary(context)])
    return "\n\n".join(parts)


def build_answered_summary(
    *,
    statements: Sequence[AnswerStatement],
    gaps: Sequence[EvidenceGap],
) -> str:
    """Findings first; remaining open gaps second (if any)."""
    parts: list[str] = []
    if statements:
        parts.append(_build_typed_statement_summary(statements))
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
        parts.append(_build_typed_statement_summary(statements))
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


_ANSWER_FIT_PRIORITY: dict[ClaimAnswerFit, int] = {
    "direct_requested_unit": 5,
    "direct_related_measure": 4,
    "quantitative_proxy": 3,
    "direct_qualitative": 2,
    "supporting_context": 1,
}


def _fallback_pdf_queries(
    metric: Metric,
    country_name: str,
    limit: int,
) -> list[ResearchQuery]:
    """Deterministic quantitative queries used if query generation fails."""
    raw = [
        f"{country_name} El Nino {metric.name} quantitative measurements",
        f"{country_name} El Nino {metric.name} quantitative numerical estimates",
        (
            f"{country_name} El Nino percentage hectares affected area "
            "production yield loss people households livestock"
        ),
        (
            f"{country_name} El Nino {metric.name} table figure chart "
            "assessment estimate statistics"
        ),
        (
            f"{country_name} El Nino {EL_NINO_EVENT_PERIODS} {metric.name} "
            "assessment data values 2026-27 forecast outlook"
        ),
    ]
    return [
        ResearchQuery(
            query=query,
            purpose="Retrieve quantitative PDF evidence for the metric",
            destination="vectorstore",
        )
        for query in raw[:limit]
    ]


def _bounded_web_depth(config: ResearcherConfig) -> dict[str, Any]:
    """Build a WebScout depth that cannot exceed the configured query budget."""
    budget = max(1, config.max_web_searches_per_metric)
    iterations = max(1, min(2, config.max_web_depth))
    if iterations == 1:
        first, followup = budget, 0
    else:
        first = min(3, budget)
        followup = budget - first
    return {
        "max_iterations": iterations,
        "queries_first": first,
        "queries_followup": followup,
        "urls_first": 3,
        "urls_followup": 2,
        "hub_deepening_cap": 5,
        "evaluator_extra_prompt": (
            "Prioritize numerical results that directly answer the metric: "
            "values, units, numerators, denominators, affected area or "
            "population, magnitude of change, geography, event and period."
        ),
    }


def _web_query(metric: Metric, country_name: str) -> str:
    return (
        f"Find authoritative quantitative and outlook evidence for {country_name} "
        f"that answers this El Nino impact metric: {metric.name}. "
        f"Required answer form or unit: {metric.unit}. "
        f"Analysis required: {metric.description}. Report numerical values, "
        "units, affected area or population, magnitude, geography, event and "
        "reporting period where available. Also seek the numerator, denominator, "
        "and directly convertible component measurements implied by the metric. "
        "Include 2026-27 forecasts, current-season production, prices, trade, "
        "and food-security outlooks even when they are not in the exact unit. "
        "Only use these El Nino event "
        f"periods: {EL_NINO_EVENT_PERIODS}."
    )


def _batched[T](items: Sequence[T], size: int) -> list[Sequence[T]]:
    safe_size = max(1, size)
    return [
        items[index : index + safe_size] for index in range(0, len(items), safe_size)
    ]


def _contains_quantity(text: str) -> bool:
    return bool(_PERCENT_PATTERN.search(text)) or any(
        pattern.search(text) for pattern in _QUANTITATIVE_PATTERNS
    )


def is_direct_evidence_claim(claim: EvidenceClaim) -> bool:
    """Return whether a claim answers the metric subject.

    Direct evidence includes quantitative results for the metric subject in the
    requested unit or a related quantitative form, and qualitative findings
    that still describe that same subject.
    """
    if claim.statement_type != "answer":
        return False
    return _contains_quantity(claim.quoted_text) or claim.answer_fit in {
        "direct_requested_unit",
        "direct_related_measure",
        "direct_qualitative",
    }


def _quantitative_focus_excerpts(metric: Metric, text: str) -> str:
    """Surface high-value source windows without replacing the full source."""
    windows: list[tuple[int, int, str]] = []
    seen_ranges: list[tuple[int, int]] = []
    matches = [
        match for pattern in _QUANTITATIVE_PATTERNS for match in pattern.finditer(text)
    ]
    matches.extend(_PERCENT_PATTERN.finditer(text))
    for match in matches:
        start = max(0, match.start() - 500)
        end = min(len(text), match.end() + 700)
        if any(
            start >= prior_start and end <= prior_end
            for prior_start, prior_end in seen_ranges
        ):
            continue
        excerpt = text[start:end].strip()
        score = 10 * _metric_term_overlap(metric, excerpt)
        windows.append((score, match.start(), excerpt))
        seen_ranges.append((start, end))
    windows.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return "\n\n[...]\n\n".join(excerpt for _, _, excerpt in windows[:6])


def _metric_term_overlap(metric: Metric, text: str) -> int:
    stop = {
        "after",
        "and",
        "answer",
        "country",
        "during",
        "from",
        "into",
        "metric",
        "selected",
        "that",
        "the",
        "this",
        "with",
    }
    metric_tokens = {
        token
        for token in re.findall(
            r"[a-z0-9]+",
            f"{metric.name} {metric.description} {metric.unit}".casefold(),
        )
        if len(token) > 2 and token not in stop
    }
    claim_tokens = set(re.findall(r"[a-z0-9]+", text.casefold()))
    return len(metric_tokens & claim_tokens)


def _has_only_ineligible_explicit_years(text: str) -> bool:
    """Reject measurements explicitly tied only to unsupported event years."""
    years = {int(value) for value in _YEAR_PATTERN.findall(text)}
    return bool(years) and years.isdisjoint(_ELIGIBLE_EVENT_YEARS)


def _country_name_variants(state: ResearchState) -> list[str]:
    """Official, common, and ISO labels used to detect country mentions."""
    variants = [state.country_name, state.country_iso3]
    country = pycountry.countries.get(alpha_3=state.country_iso3.upper())
    if country is not None:
        variants.append(str(country.name))
        official = getattr(country, "official_name", None)
        if isinstance(official, str) and official.strip():
            variants.append(official)
    # Keep longer names first so "Federal Republic of Somalia" matches before
    # a short substring search over the same text.
    deduped: list[str] = []
    seen: set[str] = set()
    for name in sorted(
        {item.strip() for item in variants if item.strip()}, key=len, reverse=True
    ):
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(name)
    return deduped


def _mentions_selected_country(state: ResearchState, text: str) -> bool:
    folded = normalize_for_quote_match(text).casefold()
    return any(name.casefold() in folded for name in _country_name_variants(state))


def _scope_window_for_quote(text: str, source_text: str | None) -> str:
    """Return quote text plus nearby source context when the quote can be located."""
    scope_text = normalize_for_quote_match(text)
    if not source_text:
        return scope_text
    normalized_source = normalize_for_quote_match(source_text)
    quote_position = normalized_source.find(scope_text)
    if quote_position < 0:
        return scope_text
    start = max(0, quote_position - 1_200)
    end = min(len(normalized_source), quote_position + len(scope_text) + 1_200)
    return normalized_source[start:end]


def _has_explicit_scope_conflict(
    state: ResearchState,
    text: str,
    *,
    source_text: str | None = None,
) -> bool:
    """True when the claim is a global/aggregate result, not a country result.

    Country metadata on a PDF is not evidence of geographic scope. A global or
    unscoped aggregate-total figure may still be kept as context, but must not
    be treated as a country answer.
    """
    quote = normalize_for_quote_match(text)
    if _mentions_selected_country(state, quote):
        return False
    if _GLOBAL_SCOPE_PATTERN.search(quote):
        return True
    if _AGGREGATE_AREA_SCOPE_PATTERN.search(quote):
        return True
    window = _scope_window_for_quote(text, source_text)
    if window == quote:
        return False
    # Nearby global framing only conflicts when the quotation itself looks like
    # an agricultural-area magnitude rather than an unrelated country sentence.
    return bool(
        _GLOBAL_SCOPE_PATTERN.search(window)
        and (
            _AGGREGATE_AREA_SCOPE_PATTERN.search(quote)
            or (
                _contains_quantity(quote)
                and re.search(
                    r"\b(?:agricultural|agriculture|cropland|cropping|crops?|area)\b",
                    quote,
                    flags=re.IGNORECASE,
                )
            )
        )
    )


def _statement_misattributes_global_scope(
    state: ResearchState,
    statement: AnswerStatement,
    claims: Sequence[EvidenceClaim],
    *,
    source_texts: dict[str, str],
) -> bool:
    """True when a statement assigns a global/aggregate claim to the country."""
    if not _mentions_selected_country(state, statement.text):
        return False
    for claim in claims:
        if claim.claim_id not in statement.supporting_claim_ids:
            continue
        if _mentions_selected_country(state, claim.quoted_text):
            continue
        if _has_explicit_scope_conflict(
            state,
            claim.quoted_text,
            source_text=source_texts.get(claim.source_id),
        ):
            return True
    return False


def _as_global_context_claim(claim: EvidenceClaim) -> EvidenceClaim:
    return claim.model_copy(
        update={
            "statement_type": "context",
            "answer_fit": (
                "quantitative_proxy"
                if _contains_quantity(claim.quoted_text)
                else "supporting_context"
            ),
        }
    )


async def _judge_claim_usefulness(
    state: ResearchState,
    claims: Sequence[EvidenceClaim],
    *,
    model: BaseChatModel,
) -> list[EvidenceClaim]:
    """Keep only claims that independently answer or quantify the metric."""
    source_texts = _source_text_map(state)
    eligible = []
    accepted: list[EvidenceClaim] = []
    rejected_reasons: dict[str, int] = {}
    for claim in claims:
        if _has_only_ineligible_explicit_years(claim.quoted_text):
            reason = "unsupported_explicit_event_year"
            rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1
            continue
        if _has_explicit_scope_conflict(
            state,
            claim.quoted_text,
            source_text=source_texts.get(claim.source_id),
        ):
            # Global/aggregate evidence may inform context, but never answers the
            # country metric and must not be localized to the selected country.
            if _contains_quantity(claim.quoted_text):
                accepted.append(_as_global_context_claim(claim))
            continue
        eligible.append(claim)

    for batch in _batched(eligible, 20):
        claims_block = "\n\n".join(
            f"claim_id: {claim.claim_id}\n"
            f"source_type: {claim.source_type}\n"
            f"extractor_answer_fit: {claim.answer_fit}\n"
            f"quotation: {claim.quoted_text}"
            for claim in batch
        )
        user = (
            f"{_metric_prompt_block(state.metric, state.country_name, state.country_iso3)}"
            "\nDesired answer shape from Metric.example (structure only; never "
            f"copy its facts):\n{state.metric.example}\n\n"
            f"Source-validated claims to judge:\n{claims_block}"
        )
        try:
            result = await _structured_invoke(
                model,
                ClaimUsefulnessList,
                system=CLAIM_USEFULNESS_SYSTEM,
                user=user,
            )
            assert isinstance(result, ClaimUsefulnessList)
        except Exception:
            logger.exception("Metric-answer usefulness judging failed; rejecting batch")
            rejected_reasons["answerability_judge_failed"] = rejected_reasons.get(
                "answerability_judge_failed", 0
            ) + len(batch)
            continue

        verdict_by_id = {verdict.claim_id: verdict for verdict in result.verdicts}
        for claim in batch:
            verdict = verdict_by_id.get(claim.claim_id)
            if verdict is None:
                reason = "missing_answerability_verdict"
                rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1
                continue
            if verdict.verdict == "reject":
                if _contains_quantity(claim.quoted_text):
                    # Related El Nino agrifood quantities still inform expected
                    # impacts even when they are not the exact metric subject.
                    accepted.append(
                        claim.model_copy(
                            update={
                                "statement_type": "context",
                                "answer_fit": "quantitative_proxy",
                            }
                        )
                    )
                    continue
                if claim.answer_fit in {
                    "supporting_context",
                    "direct_qualitative",
                    "direct_related_measure",
                }:
                    accepted.append(
                        claim.model_copy(
                            update={
                                "statement_type": "context",
                                "answer_fit": (
                                    claim.answer_fit
                                    if claim.answer_fit != "direct_related_measure"
                                    else "supporting_context"
                                ),
                            }
                        )
                    )
                    continue
                reason = verdict.reason.strip() or "does_not_answer_metric"
                rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1
                continue
            if not _contains_quantity(claim.quoted_text):
                accepted.append(
                    claim.model_copy(
                        update={
                            "statement_type": (
                                "answer"
                                if verdict.verdict == "direct_answer"
                                else "context"
                            ),
                            "answer_fit": (
                                "direct_qualitative"
                                if verdict.verdict == "direct_answer"
                                else "supporting_context"
                            ),
                        }
                    )
                )
                continue
            extractor_says_direct = claim.answer_fit in {
                "direct_requested_unit",
                "direct_related_measure",
            }
            # Metric-subject quantitative claims are direct answers even when the
            # unit differs from Metric.unit. Promote extractor-confirmed direct
            # fits if the judge was overly unit-strict and returned context.
            if verdict.verdict == "direct_answer" or (
                verdict.verdict == "context" and extractor_says_direct
            ):
                accepted.append(
                    claim.model_copy(
                        update={
                            "statement_type": "answer",
                            "answer_fit": (
                                claim.answer_fit
                                if extractor_says_direct
                                else "direct_requested_unit"
                            ),
                        }
                    )
                )
                continue
            accepted.append(
                claim.model_copy(
                    update={
                        "statement_type": "context",
                        "answer_fit": "quantitative_proxy",
                    }
                )
            )

    logger.info(
        "Researcher STAGE=answerability candidates=%s accepted=%s rejected=%s "
        "reasons=%s",
        len(claims),
        len(accepted),
        len(claims) - len(accepted),
        rejected_reasons,
    )
    return accepted


def _claim_rank_key(state: ResearchState, claim: EvidenceClaim) -> tuple[Any, ...]:
    retrieval_score = 0.0
    if claim.source_type == "vectorstore":
        chunk = next(
            (item for item in state.vector_chunks if item.source_id == claim.source_id),
            None,
        )
        if chunk is not None and chunk.score is not None:
            retrieval_score = chunk.score
    return (
        int(claim.statement_type == "answer"),
        _ANSWER_FIT_PRIORITY[claim.answer_fit],
        int(_contains_quantity(claim.quoted_text)),
        _metric_term_overlap(state.metric, claim.quoted_text),
        retrieval_score,
        -len(claim.quoted_text),
    )


def _normalized_proposition(text: str) -> str:
    return " ".join(
        re.findall(r"[a-z0-9]+", normalize_for_quote_match(text).casefold())
    )


def _claims_duplicate(left: EvidenceClaim, right: EvidenceClaim) -> bool:
    """Conservatively identify duplicate propositions without merging facts."""
    left_text = _normalized_proposition(left.quoted_text)
    right_text = _normalized_proposition(right.quoted_text)
    if left_text == right_text:
        return True
    shorter, longer = sorted((left_text, right_text), key=len)
    if len(shorter) >= 50 and shorter in longer and len(shorter) / len(longer) >= 0.8:
        return True
    left_tokens = set(left_text.split())
    right_tokens = set(right_text.split())
    left_numbers = set(re.findall(r"\d+(?:[.,]\d+)?", left_text))
    right_numbers = set(re.findall(r"\d+(?:[.,]\d+)?", right_text))
    if not left_numbers or left_numbers != right_numbers:
        return False
    union = left_tokens | right_tokens
    return bool(union) and len(left_tokens & right_tokens) / len(union) >= 0.8


def _select_claims(
    state: ResearchState,
    candidates: Sequence[EvidenceClaim],
    *,
    limit: int,
    existing: Sequence[EvidenceClaim] = (),
) -> list[EvidenceClaim]:
    selected: list[EvidenceClaim] = []
    for candidate in sorted(
        candidates,
        key=lambda claim: _claim_rank_key(state, claim),
        reverse=True,
    ):
        if any(_claims_duplicate(candidate, prior) for prior in [*existing, *selected]):
            continue
        selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected


def _select_web_claims(
    state: ResearchState,
    candidates: Sequence[EvidenceClaim],
    *,
    pdf_claims: Sequence[EvidenceClaim],
    limit: int,
) -> list[EvidenceClaim]:
    selected: list[EvidenceClaim] = []
    for candidate in sorted(
        candidates,
        key=lambda claim: _claim_rank_key(state, claim),
        reverse=True,
    ):
        pdf_duplicate = next(
            (claim for claim in pdf_claims if _claims_duplicate(candidate, claim)),
            None,
        )
        if pdf_duplicate is not None:
            state.corroborating_claim_ids.setdefault(pdf_duplicate.claim_id, []).append(
                candidate.claim_id
            )
            selected.append(candidate)
        elif not any(_claims_duplicate(candidate, prior) for prior in selected):
            selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected


async def research(
    *,
    metric: Metric,
    country_iso3: str,
    vector_store: ResearchVectorStore,
    config: ResearcherConfig | None = None,
    model: BaseChatModel | None = None,
    verifier_model: BaseChatModel | None = None,
    visual_model: BaseChatModel | None = None,
    visual_verifier_model: BaseChatModel | None = None,
    query_model: BaseChatModel | None = None,
    web_research_fn: WebResearchFn | None = None,
    web_research_enabled: bool = True,
    generate_research_queries_fn: GenerateResearchQueriesFn | None = None,
) -> ResearcherOutput:
    """Mine country-filtered PDF evidence, then append bounded web evidence."""
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
        research_iteration=1,
    )
    logger.info(
        "Researcher START country=%s metric=%r pdf_queries=%s pdf_claim_target=%s "
        "web_enabled=%s",
        state.country_iso3,
        metric.name,
        cfg.max_pdf_queries_per_metric,
        cfg.target_pdf_claims_per_metric,
        web_research_enabled,
    )

    try:
        generated = await query_gen(
            research_question=metric.name,
            explanation=(
                f"{metric.description}\nPrioritize quantitative evidence that "
                "matches the requested answer form and the structural depth of "
                "the example. Use only these eligible El Nino event periods: "
                f"{EL_NINO_EVENT_PERIODS}."
            ),
            country_name=country_name,
            country_iso3=state.country_iso3,
            example=metric.example,
            established_facts=[],
            open_gaps=[],
            executed_queries=[],
            weak_terms=[],
            preferred_destinations=["vectorstore"],
            min_queries=min(3, cfg.max_pdf_queries_per_metric),
            max_queries=cfg.max_pdf_queries_per_metric,
            model=query_model,
        )
    except Exception:
        logger.exception("Quantitative PDF query generation failed; using fallbacks")
        generated = _fallback_pdf_queries(
            metric,
            country_name,
            cfg.max_pdf_queries_per_metric,
        )

    seen_queries: set[str] = set()
    pdf_queries: list[ResearchQuery] = []
    for query in generated:
        if not query.query.strip():
            continue
        key = normalize_query(query.query)
        if key in seen_queries:
            continue
        seen_queries.add(key)
        pdf_queries.append(query.model_copy(update={"destination": "vectorstore"}))
        if len(pdf_queries) >= cfg.max_pdf_queries_per_metric:
            break
    if not pdf_queries:
        pdf_queries = _fallback_pdf_queries(
            metric,
            country_name,
            cfg.max_pdf_queries_per_metric,
        )
    state.current_queries = pdf_queries
    state.all_queries.extend(pdf_queries)
    state.executed_queries.extend(query.query for query in pdf_queries)
    logger.info(
        "Researcher STAGE=pdf_queries count=%s queries=%s",
        len(pdf_queries),
        [query.query for query in pdf_queries],
    )

    chunk_by_id: dict[str, RetrievedChunk] = {}
    hits_by_query: dict[str, list[str]] = {}
    for query in pdf_queries:
        try:
            hits = await vector_store.search(
                query.query,
                countries_iso3=[state.country_iso3],
                limit=cfg.pdf_results_per_query,
            )
        except Exception:
            logger.exception("PDF vector search failed for %r", query.query)
            hits = []
        logger.info(
            "Researcher STAGE=pdf_retrieve query=%r hits=%s",
            query.query,
            len(hits),
        )
        hit_ids: list[str] = []
        for hit in hits:
            chunk = chunk_from_hit(hit, query.query)
            hit_ids.append(chunk.source_id)
            previous = chunk_by_id.get(chunk.source_id)
            if previous is None or (chunk.score or 0.0) > (previous.score or 0.0):
                chunk_by_id[chunk.source_id] = chunk
        hits_by_query[query.query] = hit_ids
        state.query_runs.append(
            QueryRunStat(
                query=query.query,
                destination="vectorstore",
                results_returned=len(hit_ids),
                results_accepted=0,
            )
        )
    state.vector_chunks = sorted(
        chunk_by_id.values(),
        key=lambda chunk: chunk.score or 0.0,
        reverse=True,
    )[: cfg.max_pdf_evidence_to_analyze]
    logger.info(
        "Researcher STAGE=pdf_retrieve done unique=%s analyze=%s",
        len(chunk_by_id),
        len(state.vector_chunks),
    )

    if cfg.use_visual_evidence and any(
        chunk.visual_artifacts for chunk in state.vector_chunks
    ):
        visual_reader = visual_model or build_chat_model(llm_model=cfg.visual_llm_model)
        visual_verifier = visual_verifier_model or build_chat_model(
            llm_model=cfg.visual_verifier_llm_model
        )
        await _enrich_visual_chunks(
            state,
            state.vector_chunks,
            model=visual_reader,
            verifier_model=visual_verifier,
            max_chunks=cfg.max_visual_chunks_per_iteration,
            max_artifacts_per_chunk=cfg.max_visual_artifacts_per_chunk,
        )

    pdf_candidates = await _extract_and_validate_batches(
        state,
        model=main_model,
        chunks=state.vector_chunks,
        web_sources=[],
        batch_size=cfg.claim_extraction_batch_size,
        max_claims_per_source=cfg.max_claims_per_evidence,
    )
    answerable_pdf = await _judge_claim_usefulness(
        state,
        pdf_candidates,
        model=main_model,
    )
    selected_pdf = _select_claims(
        state,
        answerable_pdf,
        limit=cfg.target_pdf_claims_per_metric,
    )
    state.validated_claims = list(selected_pdf)
    accepted_pdf_sources = {claim.source_id for claim in selected_pdf}
    for run in state.query_runs:
        if run.destination != "vectorstore":
            continue
        hit_ids = hits_by_query.get(run.query, [])
        run.results_accepted = sum(
            source_id in accepted_pdf_sources for source_id in hit_ids
        )
    logger.info(
        "Researcher STAGE=pdf_claims candidates=%s answerable=%s selected=%s "
        "quantitative=%s",
        len(pdf_candidates),
        len(answerable_pdf),
        len(selected_pdf),
        sum(_contains_quantity(claim.quoted_text) for claim in selected_pdf),
    )

    selected_web: list[EvidenceClaim] = []
    if web_research_enabled:
        state.research_iteration = 2
        web_query = _web_query(metric, country_name)
        try:
            mapped = await run_web_scout_research(
                web_query,
                domain_expertise=(
                    "authoritative FAO, FEWS NET, FSNAU, UN, World Bank, "
                    "government, food and agriculture statistics"
                ),
                research_depth=_bounded_web_depth(cfg),
                web_research_fn=web_research_fn,
            )
            # Compact sequential ids (web:001, ...) are much more reliably
            # copied by the claim extractor than long URL/hash ids.
            state.web_sources = [
                source.model_copy(update={"source_id": f"web:{index:03d}"})
                for index, source in enumerate(mapped.sources, start=1)
            ]
            # web-scout does not attribute scraped URLs to individual search
            # queries; accepted evidence-source counts are filled after selection.
            for item in mapped.query_stats:
                state.query_runs.append(
                    QueryRunStat(
                        query=item.query,
                        destination="web",
                        results_returned=item.results_returned,
                        results_accepted=0,
                    )
                )
            logger.info(
                "Researcher STAGE=web_retrieve sources=%s searches=%s",
                len(state.web_sources),
                len(mapped.queries),
            )
        except WebScoutProviderError:
            logger.warning(
                "WebScout failed; continuing with PDF evidence",
                exc_info=True,
            )
        web_candidates = await _extract_and_validate_batches(
            state,
            model=main_model,
            chunks=[],
            web_sources=state.web_sources,
            batch_size=cfg.claim_extraction_batch_size,
            max_claims_per_source=cfg.max_claims_per_evidence,
        )
        answerable_web = await _judge_claim_usefulness(
            state,
            web_candidates,
            model=main_model,
        )
        selected_web = _select_web_claims(
            state,
            answerable_web,
            pdf_claims=selected_pdf,
            limit=cfg.max_web_claims_per_metric,
        )
        state.validated_claims = [*selected_pdf, *selected_web]
        accepted_web_sources = len({claim.source_id for claim in selected_web})
        web_runs = [run for run in state.query_runs if run.destination == "web"]
        if web_runs:
            # Scrapes are not linked to individual search queries; record the
            # accepted source count once so the section does not double-count.
            web_runs[0].results_accepted = accepted_web_sources
        logger.info(
            "Researcher STAGE=web_claims candidates=%s answerable=%s selected=%s",
            len(web_candidates),
            len(answerable_web),
            len(selected_web),
        )
    else:
        logger.info("Researcher STAGE=web_retrieve skipped explicitly disabled")

    if not state.validated_claims:
        state.gaps = [
            EvidenceGap(
                gap_id="gap_001",
                description=(
                    "No country-specific evidence directly answering the metric "
                    "was found in the PDF collection or permitted web research."
                ),
                why_required="At least one supported finding is required to answer.",
                preferred_source_type="vectorstore",
                suggested_terms=[metric.name, country_name, metric.unit],
                status="open",
            )
        ]
        state.termination_reason = "no_supported_claims"
        return _finalize(state, status="cannot_answer")

    if any(claim.statement_type == "answer" for claim in state.validated_claims):
        state.gaps = []
    else:
        state.gaps = [
            EvidenceGap(
                gap_id="gap_001",
                description=(
                    "Relevant contextual evidence was found, but no evidence "
                    "directly answers the requested metric."
                ),
                why_required=(
                    "A quantitative measurement of the metric subject is "
                    "required to answer (requested unit or a related "
                    "quantitative form)."
                ),
                preferred_source_type="vectorstore",
                suggested_terms=[metric.name, country_name, metric.unit],
                status="open",
            )
        ]
    await _draft_and_verify(
        state,
        model=main_model,
        verifier_model=verify_model,
        config=cfg,
    )
    status = classify_researcher_status(state.verified_statements)
    state.termination_reason = status
    output = _finalize(state, status=status)
    logger.info(
        "Researcher END status=%s pdf_claims=%s web_claims=%s statements=%s",
        output.status,
        len(selected_pdf),
        len(selected_web),
        len(output.statements),
    )
    return output


async def _extract_claims(
    state: ResearchState,
    *,
    model: BaseChatModel,
    new_chunks: Sequence[RetrievedChunk],
    new_web: Sequence[WebSource],
    max_claims_per_source: int,
    retry_exact: bool = False,
) -> list[ExtractedClaimCandidate]:
    if not new_chunks and not new_web:
        return []
    source_blocks: list[str] = []
    for chunk in new_chunks:
        fact_texts = _visual_fact_texts(chunk)
        searchable = "\n".join(
            part for part in (chunk.chunk_text, "\n".join(fact_texts)) if part
        )
        focus = _quantitative_focus_excerpts(state.metric, searchable)
        focus_block = (
            f"\n[high-priority quantitative excerpts]\n{focus}\n" if focus else ""
        )
        visual_block = ""
        if fact_texts:
            visual_block = "\n[verified visual facts]\n" + "\n".join(
                f"- {fact}" for fact in fact_texts
            )
        source_blocks.append(
            f"[source_id={chunk.source_id} type=vectorstore "
            f"country_scope={state.country_iso3} url={chunk.document_url} "
            f"page={chunk.page_number}]"
            f"{focus_block}{visual_block}\n[full source]\n"
            f"{chunk.chunk_text}"
        )
    for source in new_web:
        focus = _quantitative_focus_excerpts(state.metric, source.content)
        focus_block = (
            f"\n[high-priority quantitative excerpts]\n{focus}\n" if focus else ""
        )
        source_blocks.append(
            f"[source_id={source.source_id} type=web url={source.url}]"
            f"{focus_block}\n[full source]\n"
            f"{source.content}"
        )
    example = state.metric.example or "(none)"
    retry_instruction = (
        "\nA previous attempt produced no source-verifiable claim. Copy each "
        "quoted_text character-for-character from one contiguous source span; "
        "do not join separated sentences or clean up PDF formatting.\n"
        if retry_exact
        else ""
    )
    user = (
        f"{_metric_prompt_block(state.metric, state.country_name, state.country_iso3)}"
        "\nDesired answer shape from Metric.example (structure only; never copy "
        f"its facts):\n{example}\n\n"
        f"Extract at most {max_claims_per_source} distinct atomic verbatim claims "
        "from each source. Prefer quantities that answer the metric in the "
        f"requested form.{retry_instruction}\n" + "\n\n---\n\n".join(source_blocks)
    )
    result = await _structured_invoke(
        model,
        ExtractedClaimList,
        system=CLAIM_EXTRACTION_SYSTEM,
        user=user,
    )
    assert isinstance(result, ExtractedClaimList)
    return result.claims


async def _extract_and_validate_batches(
    state: ResearchState,
    *,
    model: BaseChatModel,
    chunks: Sequence[RetrievedChunk],
    web_sources: Sequence[WebSource],
    batch_size: int,
    max_claims_per_source: int,
) -> list[EvidenceClaim]:
    """Mine small evidence batches so one sparse response cannot erase coverage."""
    sources: list[RetrievedChunk | WebSource] = [*chunks, *web_sources]
    accepted_all: list[EvidenceClaim] = []
    for batch_number, batch in enumerate(_batched(sources, batch_size), start=1):
        batch_chunks = [item for item in batch if isinstance(item, RetrievedChunk)]
        batch_web = [item for item in batch if isinstance(item, WebSource)]
        for attempt in range(2):
            try:
                candidates = await _extract_claims(
                    state,
                    model=model,
                    new_chunks=batch_chunks,
                    new_web=batch_web,
                    max_claims_per_source=max_claims_per_source,
                    retry_exact=attempt > 0,
                )
            except Exception:
                logger.exception(
                    "Claim extraction failed for batch %s attempt %s; continuing",
                    batch_number,
                    attempt + 1,
                )
                candidates = []
            counts: dict[str, int] = {}
            capped: list[ExtractedClaimCandidate] = []
            for candidate in candidates:
                count = counts.get(candidate.source_id, 0)
                if count >= max_claims_per_source:
                    continue
                counts[candidate.source_id] = count + 1
                capped.append(candidate)
            accepted, rejected = _validate_claim_candidates(state, capped)
            accepted_all.extend(accepted)
            state.rejected_claims.extend(rejected)
            reasons: dict[str, int] = {}
            for item in rejected:
                reasons[item.reason] = reasons.get(item.reason, 0) + 1
            logger.info(
                "Researcher STAGE=claim_batch batch=%s attempt=%s sources=%s "
                "candidates=%s accepted=%s rejected=%s reasons=%s",
                batch_number,
                attempt + 1,
                len(batch),
                len(capped),
                len(accepted),
                len(rejected),
                reasons,
            )
            if accepted or attempt == 1:
                break
            logger.info(
                "Researcher STAGE=claim_batch retrying batch=%s with exact-copy "
                "instruction",
                batch_number,
            )
    return accepted_all


def _resolve_candidate_source_id(
    candidate: ExtractedClaimCandidate,
    texts: dict[str, str],
    metas: dict[str, SourceReference],
) -> str | None:
    """Resolve extractor source_id mistakes via URL or unique quote match."""
    if candidate.source_id in texts:
        return candidate.source_id
    if candidate.url:
        url = candidate.url.strip()
        url_matches = [
            source_id for source_id, meta in metas.items() if meta.document_uri == url
        ]
        if len(url_matches) == 1:
            return url_matches[0]
    quote_matches = [
        source_id
        for source_id, source_text in texts.items()
        if match_quoted_text(candidate.quoted_text, source_text) is not None
    ]
    if len(quote_matches) == 1:
        return quote_matches[0]
    web_matches = [
        source_id
        for source_id in quote_matches
        if metas[source_id].source_type == "web"
    ]
    if len(web_matches) == 1:
        return web_matches[0]
    return None


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
        source_id = _resolve_candidate_source_id(cand, texts, metas)
        if source_id is None:
            rejected.append(
                RejectedClaim(
                    source_id=cand.source_id,
                    quoted_text=cand.quoted_text,
                    reason="unknown_source_id",
                )
            )
            continue
        source_text = texts[source_id]
        match_kind = match_quoted_text(cand.quoted_text, source_text)
        if match_kind is None:
            rejected.append(
                RejectedClaim(
                    source_id=source_id,
                    quoted_text=cand.quoted_text,
                    reason="substring_validation_failed",
                )
            )
            continue
        meta = metas[source_id]
        # Vector results are already country-filtered by countries_iso3. Web
        # pages have no trusted country metadata, so require country context in
        # the scraped page rather than in every minimal quotation.
        source_country_ok = True
        if meta.source_type == "web":
            source_country_ok = _mentions_selected_country(state, source_text)
        if not source_country_ok:
            rejected.append(
                RejectedClaim(
                    source_id=source_id,
                    quoted_text=cand.quoted_text,
                    reason="not_country_specific",
                )
            )
            continue
        fp = _claim_fingerprint(cand.quoted_text, source_id)
        if fp in existing:
            rejected.append(
                RejectedClaim(
                    source_id=source_id,
                    quoted_text=cand.quoted_text,
                    reason="duplicate_claim",
                )
            )
            continue
        visual_artifact_ids = _visual_artifact_ids_for_quote(
            state, source_id, cand.quoted_text
        )
        claim_id = f"claim_{state.next_claim_seq:03d}"
        state.next_claim_seq += 1
        claim = EvidenceClaim(
            claim_id=claim_id,
            source_type=meta.source_type,
            source_id=source_id,
            quoted_text=cand.quoted_text,
            country=cand.country or state.country_name,
            relevance=cand.relevance,
            answer_fit=cand.answer_fit,
            metric_aspects=list(cand.metric_aspects),
            page_number=meta.page_number,
            section=cand.section,
            url=meta.document_uri,
            match_kind=match_kind,
            evidence_modality=(
                "verified_visual_fact" if visual_artifact_ids is not None else "text"
            ),
            visual_artifact_ids=visual_artifact_ids or [],
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


async def _draft_and_verify(
    state: ResearchState,
    *,
    model: BaseChatModel,
    verifier_model: BaseChatModel,
    config: ResearcherConfig,
) -> None:
    corroborating_ids = {
        claim_id
        for claim_ids in state.corroborating_claim_ids.values()
        for claim_id in claim_ids
    }
    primary_claims = [
        claim
        for claim in state.validated_claims
        if claim.claim_id not in corroborating_ids
    ]
    primary_claims.sort(
        key=lambda claim: _claim_rank_key(state, claim),
        reverse=True,
    )
    claims_block = "\n".join(
        f"- {c.claim_id} (statement_type={c.statement_type}, "
        f"source={c.source_id}, url={c.url}, "
        f"page={c.page_number}): {c.quoted_text}"
        for c in primary_claims
    )
    style = (
        "Style/depth guidance ONLY from Metric.example (never copy facts):\n"
        f"{state.metric.example}"
    )
    user = (
        f"{_metric_prompt_block(state.metric, state.country_name, state.country_iso3)}"
        f"\n{style}\n\nValidated claims:\n{claims_block}\n\n"
        "Produce a comprehensive synthesis that covers every listed claim. "
        "Combine compatible claims when useful and cite every supporting "
        "claim_id. Never mix answer and context claims in one statement. "
        "Prioritize quantitative findings that measure the metric subject "
        "(requested unit or a related quantitative form) over qualitative "
        "narrative alone. "
        "Prefer newer / more recent sources over older ones when claims "
        "conflict or overlap."
    )
    try:
        drafted = await _structured_invoke(
            model,
            AnswerStatementList,
            system=ANSWER_SYSTEM,
            user=user,
        )
        assert isinstance(drafted, AnswerStatementList)
    except Exception:
        logger.exception("Answer drafting failed; using validated quotations")
        drafted = AnswerStatementList()

    primary_ids = {claim.claim_id for claim in primary_claims}
    claim_by_id = {claim.claim_id: claim for claim in primary_claims}
    covered_claim_ids: set[str] = set()
    statements: list[AnswerStatement] = []
    generated_statement_ids: set[str] = set()
    for item in drafted.statements:
        supported = list(
            dict.fromkeys(
                claim_id
                for claim_id in item.supporting_claim_ids
                if claim_id in primary_ids
            )
        )
        if (
            not supported
            or not item.text.strip()
            or any(claim_id in covered_claim_ids for claim_id in supported)
        ):
            continue
        statement_types = {
            claim_by_id[claim_id].statement_type for claim_id in supported
        }
        if len(statement_types) != 1:
            continue
        if state.metric.example and _example_leaked(item.text, state.metric.example):
            continue
        statement_id = f"stmt_{state.next_statement_seq:03d}"
        state.next_statement_seq += 1
        supporting = list(
            dict.fromkeys(
                claim_id
                for primary_id in supported
                for claim_id in [
                    primary_id,
                    *state.corroborating_claim_ids.get(primary_id, []),
                ]
            )
        )
        statement = AnswerStatement(
            statement_id=statement_id,
            text=item.text.strip(),
            statement_type=next(iter(statement_types)),
            supporting_claim_ids=supporting,
            metric_aspects=list(
                item.metric_aspects
                or dict.fromkeys(
                    aspect
                    for claim_id in supported
                    for aspect in claim_by_id[claim_id].metric_aspects
                )
            ),
        )
        statements.append(statement)
        generated_statement_ids.add(statement_id)
        covered_claim_ids.update(supported)

    for claim in primary_claims:
        if claim.claim_id in covered_claim_ids:
            continue
        statement_id = f"stmt_{state.next_statement_seq:03d}"
        state.next_statement_seq += 1
        statements.append(
            AnswerStatement(
                statement_id=statement_id,
                text=claim.quoted_text.strip(),
                statement_type=claim.statement_type,
                supporting_claim_ids=[
                    claim.claim_id,
                    *state.corroborating_claim_ids.get(claim.claim_id, []),
                ],
                metric_aspects=list(claim.metric_aspects),
            )
        )
    state.draft_statements = statements
    logger.info(
        "Researcher STAGE=draft_answer done draft_statements=%s",
        len(statements),
    )

    verified: list[AnswerStatement] = []
    for statement in statements:
        primary_statement_claims = [
            claim_by_id[claim_id]
            for claim_id in statement.supporting_claim_ids
            if claim_id in claim_by_id
        ]
        primary_id = primary_statement_claims[0].claim_id
        fallback = AnswerStatement(
            statement_id=statement.statement_id,
            text=" ".join(
                claim.quoted_text.strip() for claim in primary_statement_claims
            ),
            statement_type=claim_by_id[primary_id].statement_type,
            supporting_claim_ids=list(statement.supporting_claim_ids),
            metric_aspects=list(
                dict.fromkeys(
                    aspect
                    for claim in primary_statement_claims
                    for aspect in claim.metric_aspects
                )
            ),
        )
        if statement.statement_id not in generated_statement_ids:
            verified.append(fallback)
            continue
        current = statement
        for attempt in range(config.max_answer_verification_retries + 1):
            logger.info(
                "Researcher STAGE=verify_statements statement=%s attempt=%s/%s",
                current.statement_id,
                attempt,
                config.max_answer_verification_retries,
            )
            try:
                verification = await _verify_statement(
                    state,
                    current,
                    model=verifier_model,
                )
            except Exception:
                logger.exception(
                    "Statement verification failed for %s; using quotation",
                    current.statement_id,
                )
                verified.append(fallback)
                break
            state.verifications.append(verification)
            logger.info(
                "Researcher STAGE=verify_statements statement=%s verdict=%s",
                current.statement_id,
                verification.verdict,
            )
            if verification.verdict == "entailed":
                verified.append(current)
                break
            if attempt >= config.max_answer_verification_retries:
                logger.info(
                    "Researcher STAGE=verify_statements statement=%s falling back "
                    "to validated quotation verdict=%s",
                    current.statement_id,
                    verification.verdict,
                )
                verified.append(fallback)
                break
            try:
                repaired = await _repair_statement(
                    state,
                    current,
                    verification,
                    model=model,
                )
            except Exception:
                logger.exception(
                    "Statement repair failed for %s; using quotation",
                    current.statement_id,
                )
                verified.append(fallback)
                break
            if repaired is None:
                logger.info(
                    "Researcher STAGE=verify_statements statement=%s "
                    "repair removed statement; using quotation",
                    current.statement_id,
                )
                verified.append(fallback)
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
    if _statement_misattributes_global_scope(
        state,
        statement,
        cited,
        source_texts=_source_text_map(state),
    ):
        return StatementVerification(
            statement_id=statement.statement_id,
            verdict="contradicted",
            unsupported_parts=[
                (
                    "statement attributes a global/aggregate-total figure to "
                    "the selected country"
                )
            ],
            reasoning=(
                "Cited claim reports a global or unscoped aggregate agricultural "
                "result; the statement incorrectly localizes it to the selected "
                "country."
            ),
            suggested_revision=(
                "Restate the quantity with its global/worldwide scope; do not "
                f"attribute it to {state.country_name}."
            ),
        )
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
    return AnswerStatement(
        statement_id=statement.statement_id,
        text=result.text.strip(),
        statement_type=statement.statement_type,
        supporting_claim_ids=list(statement.supporting_claim_ids),
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

    claims = list(state.validated_claims)
    cited_source_ids = {c.source_id for c in claims}
    source_list = [
        source for source_id, source in sources.items() if source_id in cited_source_ids
    ]

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
        query_runs=list(state.query_runs),
    )
    state.output = output
    return output
