"""UNDRR summarizer: cited per-metric answers from EM-DAT / DesInventar reports."""

from __future__ import annotations

import asyncio
import base64
import logging
import re
import time
from collections.abc import Sequence
from typing import Any, Literal

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from fao_impact_monitor.config import (
    AwsBedrockConfig,
    UndrrSummarizerConfig,
    get_config,
)
from fao_impact_monitor.impact_report import (
    ParsedEvidence,
    ParsedMetricReport,
    construct_reference_line,
    ensure_citations_present,
    load_plot_image_bytes,
    render_undrr_markdown,
)
from fao_impact_monitor.utils.country import iso3_to_country_name
from fao_impact_monitor.utils.document_uri import ascii_relative_path

logger = logging.getLogger(__name__)

_INLINE_CITE_RE = re.compile(r"\[@([^\s\[\]]+)\]")

DRAFT_SYSTEM = """\
You are an FAO disaster-loss analyst preparing UNDRR-style answers for one
country metric at a time. Use ONLY the supplied evidence items (DesInventar
and/or EM-DAT tables, plots, and source text). Never invent figures from
general knowledge.

Answer the metric Description in a quantitative, hazard-linked, source-aware
tone like the Example. Impacts are linked to hazards (floods, drought, storms,
etc.), not to El Nino attribution.

Required analytical shape (keep concise; typically a short paragraph or a few
tight sentences, not a dump of every row):
1. Report a country-level total across all disaster types present in the
   evidence for this metric. You MAY sum or aggregate rows from the supplied
   tables to produce national totals; do not list every contributing row.
2. Separately report totals for each region / admin unit that appears in the
   evidence, again across all disaster types (do not skip regions that have
   values). Summing rows by region is expected and preferred over picking a
   single peak row.
3. Analyze trends and dynamics across the El Nino periods listed in
   data_filter. Do NOT just highlight the single biggest number. Compare how
   the metric moves across those periods when those years appear in the
   evidence.
4. Prefer newer observations for the more recent El Nino periods over older
   data when characterizing the latest situation.

Other requirements:
- Prefer the primary dataset order already present in the evidence (Direct
  first; earlier Direct blocks first).
- Place [@evidence_id] immediately after each sentence that uses that source.
- Do not claim El Nino causation. Do not invent years, places, or magnitudes
  that are absent from the evidence; aggregation of present rows is fine.
- Write percentages with the % symbol when applicable.
"""

VERIFY_SYSTEM = """\
You score how well a UNDRR metric answer is supported by the cited evidence
(tables, source text, and/or plots). Use ONLY that evidence. Do not use
general knowledge.

Score from 1 to 10:
- 9-10: figures and geography clearly follow from the cited evidence
  (including reasonable sums/aggregates of table rows)
- 6-8: mostly supported; minor aggregation or wording uncertainty remains
- 4-5: partially supported, but important parts look weak or incomplete
- 1-3: contradicted, largely invented, wrong country/unit, or unsupported

National and regional totals that sum rows in the cited tables should score
high when the arithmetic is plausible from the visible evidence. Do not
penalize merely because every addend is not restated. Penalize invented
magnitudes, wrong country/unit, and unsupported El Nino attribution.
"""

REPAIR_SYSTEM = """\
Repair the UNDRR metric answer so it better matches the cited evidence.
Keep a concise quantitative answer covering: country total across disaster
types; per-region totals across disaster types; and El Nino-period trends from
data_filter when supported; prefer newer data for the latest periods.
Summing table rows for national/regional totals is allowed. Do not only
restore the single largest figure. Preserve or fix [@evidence_id] citations.
If the claim cannot be made evidence-based, set remove=true.
Do not invent figures. Do not attribute impacts to El Nino itself.
"""

_MIN_KEEP_SCORE = 5


class UndrrDraftAnswer(BaseModel):
    text: str
    supporting_evidence_ids: list[str] = Field(default_factory=list)


class StatementVerification(BaseModel):
    statement_id: str
    score: int = Field(
        ge=1, le=10, description="Support score from 1 (weak) to 10 (strong)."
    )
    unsupported_parts: list[str] = Field(default_factory=list)
    reasoning: str
    suggested_revision: str | None = None


class StatementRepair(BaseModel):
    statement_id: str
    text: str
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    remove: bool = False


class VerifiedUndrrStatement(BaseModel):
    statement_id: str
    metric_seq: int
    metric_name: str
    text: str
    supporting_evidence_ids: list[str] = Field(default_factory=list)


class UndrrSummarizerOutput(BaseModel):
    country: str
    country_iso3: str
    use_case_ascii: str
    markdown: str
    statements: list[VerifiedUndrrStatement] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)


def build_chat_model(
    *,
    llm_model: str | None = None,
    reasoning_effort: str | None = None,
    aws_bedrock_config: AwsBedrockConfig | None = None,
    config: UndrrSummarizerConfig | None = None,
) -> BaseChatModel:
    cfg = config or get_config().undrr_summarizer
    aws = aws_bedrock_config or get_config().aws_bedrock
    kwargs: dict[str, Any] = {
        "api_key": aws.api_key.get_secret_value(),
        "base_url": aws.base_url,
        "use_responses_api": True,
        "reasoning_effort": reasoning_effort or cfg.reasoning_effort,
    }
    chat_model = init_chat_model(llm_model or cfg.llm_model, **kwargs)
    if not isinstance(chat_model, BaseChatModel):
        raise TypeError(f"Expected BaseChatModel, got {type(chat_model)}")
    return chat_model


async def summarize_undrr(
    *,
    country_iso3: str,
    reports: Sequence[ParsedMetricReport],
    use_case_name: str,
    data_filter: str | None = None,
    config: UndrrSummarizerConfig | None = None,
    model: BaseChatModel | None = None,
    verifier_model: BaseChatModel | None = None,
) -> UndrrSummarizerOutput:
    """Draft, verify, and render a cited UNDRR summary for one country."""
    cfg = config or get_config().undrr_summarizer
    country_name = iso3_to_country_name(country_iso3)
    iso3 = country_iso3.upper()
    use_case_ascii = ascii_relative_path(use_case_name)
    main_model = model or build_chat_model(config=cfg)
    verify_model = verifier_model or build_chat_model(
        llm_model=cfg.verifier_llm_model,
        reasoning_effort="low",
        config=cfg,
    )

    ordered_reports = sorted(reports, key=lambda item: item.meta.seq_number)
    if not ordered_reports:
        raise ValueError(f"No UNDRR metric reports available for {iso3}")

    started = time.perf_counter()
    for report in ordered_reports:
        for item in [*report.direct, *report.indirect]:
            if item.plot_path is not None:
                load_plot_image_bytes(item)

    stage = time.perf_counter()
    drafted = await _draft_all_metrics(
        reports=ordered_reports,
        country_name=country_name,
        country_iso3=iso3,
        data_filter=data_filter,
        model=main_model,
        config=cfg,
    )
    logger.info(
        "UndrrSummarizer STAGE=draft country=%s statements=%s elapsed=%.1fs",
        iso3,
        len(drafted),
        time.perf_counter() - stage,
    )

    stage = time.perf_counter()
    evidence = [item for report in ordered_reports for item in report.direct]
    evidence.extend(item for report in ordered_reports for item in report.indirect)
    verified = await _verify_and_repair(
        drafted,
        evidence=evidence,
        country_name=country_name,
        country_iso3=iso3,
        model=main_model,
        verifier_model=verify_model,
        config=cfg,
    )
    logger.info(
        "UndrrSummarizer STAGE=verify country=%s verified=%s drafted=%s elapsed=%.1fs",
        iso3,
        len(verified),
        len(drafted),
        time.perf_counter() - stage,
    )

    markdown, references = _render_output(
        country_name=country_name,
        country_iso3=iso3,
        use_case_ascii=use_case_ascii,
        statements=verified,
        evidence=evidence,
        reports=ordered_reports,
    )
    logger.info(
        "UndrrSummarizer END country=%s total_elapsed=%.1fs refs=%s",
        iso3,
        time.perf_counter() - started,
        len(references),
    )
    return UndrrSummarizerOutput(
        country=country_name,
        country_iso3=iso3,
        use_case_ascii=use_case_ascii,
        markdown=markdown,
        statements=verified,
        references=references,
    )


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


def _evidence_by_id(evidence: Sequence[ParsedEvidence]) -> dict[str, ParsedEvidence]:
    return {item.evidence_id: item for item in evidence}


def _inline_citation_ids(text: str) -> list[str]:
    return _INLINE_CITE_RE.findall(text)


def _normalize_inline_citations(
    text: str,
    supporting_evidence_ids: Sequence[str],
    known: dict[str, ParsedEvidence],
) -> tuple[str, list[str]]:
    inline_ids = _inline_citation_ids(text)
    if inline_ids:

        def _keep(match: re.Match[str]) -> str:
            evidence_id = match.group(1)
            return f"[@{evidence_id}]" if evidence_id in known else ""

        cleaned = _INLINE_CITE_RE.sub(_keep, text)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r" +([.,;:])", r"\1", cleaned).strip()
        supported = list(
            dict.fromkeys(
                evidence_id for evidence_id in inline_ids if evidence_id in known
            )
        )
        return cleaned, supported
    supported = [
        evidence_id
        for evidence_id in dict.fromkeys(supporting_evidence_ids)
        if evidence_id in known
    ]
    return text.strip(), supported


def _render_statement_text(
    text: str,
    supporting_evidence_ids: Sequence[str],
    number_by_id: dict[str, int],
) -> str | None:
    if _INLINE_CITE_RE.search(text):

        def _replace(match: re.Match[str]) -> str:
            evidence_id = match.group(1)
            number = number_by_id.get(evidence_id)
            return f"[{number}]" if number is not None else ""

        rendered = _INLINE_CITE_RE.sub(_replace, text)
        rendered = re.sub(r"[ \t]{2,}", " ", rendered)
        rendered = re.sub(r" +([.,;:])", r"\1", rendered).strip()
        if not re.search(r"\[\d+\]", rendered):
            return None
        return rendered
    cites = " ".join(
        f"[{number_by_id[evidence_id]}]"
        for evidence_id in dict.fromkeys(supporting_evidence_ids)
        if evidence_id in number_by_id
    )
    if not cites:
        return None
    return f"{text} {cites}".rstrip()


def _plot_image_block(
    evidence: ParsedEvidence,
    *,
    detail: str = "low",
) -> dict[str, Any]:
    data = load_plot_image_bytes(evidence)
    encoded = base64.b64encode(data).decode("ascii")
    suffix = evidence.plot_path.suffix.lower() if evidence.plot_path else ".png"
    media = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(suffix, "image/png")
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{media};base64,{encoded}",
            "detail": detail,
        },
    }


def _format_evidence_for_prompt(
    item: ParsedEvidence,
    *,
    max_source_text_chars: int = 4_000,
) -> str:
    lines = [
        f"evidence_id: {item.evidence_id}",
        f"kind: {item.kind}",
        f"source_type: {item.source_type}",
        f"metric: {item.metric_name}",
        f"title: {item.title}",
    ]
    if item.dataset:
        lines.append(f"dataset: {item.dataset}")
    if item.indicator:
        lines.append(f"indicator: {item.indicator}")
    if item.source_url:
        lines.append(f"source_url: {item.source_url}")
    if item.latest_value is not None:
        year = f" ({item.latest_year})" if item.latest_year is not None else ""
        lines.append(f"latest_value: {item.latest_value}{year}")
    if item.plot_path is not None:
        lines.append(f"plot: attached image for {item.plot_path.name}")
    if item.source_data:
        text = item.source_data
        if len(text) > max_source_text_chars:
            # Silent trim: do not tell the model the table was truncated.
            text = text[:max_source_text_chars].rstrip()
        lines.append(f"source_data:\n{text}")
    if item.source_text:
        text = item.source_text
        if len(text) > max_source_text_chars:
            text = text[:max_source_text_chars].rstrip()
        lines.append(f"source_text:\n{text}")
    if item.verified_visual_facts:
        facts = "\n".join(f"- {fact}" for fact in item.verified_visual_facts)
        lines.append(f"verified_visual_facts:\n{facts}")
    return "\n".join(lines)


def _metric_evidence(report: ParsedMetricReport) -> list[ParsedEvidence]:
    """Prefer Direct evidence; fall back to Indirect when Direct is empty."""
    if report.direct:
        return list(report.direct)
    return list(report.indirect)


async def _draft_all_metrics(
    *,
    reports: Sequence[ParsedMetricReport],
    country_name: str,
    country_iso3: str,
    data_filter: str | None,
    model: BaseChatModel,
    config: UndrrSummarizerConfig,
) -> list[VerifiedUndrrStatement]:
    semaphore = asyncio.Semaphore(max(1, config.verify_concurrency))

    async def _draft_one(
        report: ParsedMetricReport,
    ) -> VerifiedUndrrStatement | None:
        async with semaphore:
            evidence = _metric_evidence(report)
            known = _evidence_by_id(evidence)
            text_parts = [
                f"Country: {country_name} ({country_iso3})",
                f"Metric seq: {report.meta.seq_number}",
                f"Metric name: {report.meta.name}",
                f"Description: {report.meta.description}",
                f"Example: {report.meta.example}",
                f"Unit: {report.meta.unit or '(none)'}",
                f"Data filter: {data_filter or '(none)'}",
                "",
                "Evidence items (cite by evidence_id; Direct preferred):",
            ]
            image_blocks: list[dict[str, Any]] = []
            for item in evidence:
                text_parts.append(
                    _format_evidence_for_prompt(
                        item, max_source_text_chars=config.max_source_text_chars
                    )
                )
                text_parts.append("---")
                if item.plot_path is not None:
                    text_parts.append(
                        f"[IMAGE FOLLOWS for evidence_id={item.evidence_id}]"
                    )
                    image_blocks.append(
                        _plot_image_block(item, detail=config.default_plot_detail)
                    )
            user_content: Any
            if image_blocks:
                user_content = [
                    {"type": "text", "text": "\n".join(text_parts)},
                    *image_blocks,
                ]
            else:
                user_content = "\n".join(text_parts)
            try:
                result = await _structured_invoke(
                    model,
                    UndrrDraftAnswer,
                    system=DRAFT_SYSTEM,
                    user=user_content,
                )
            except Exception:
                logger.exception(
                    "Draft failed for metric %s (%s)",
                    report.meta.seq_number,
                    report.meta.name,
                )
                return None
            assert isinstance(result, UndrrDraftAnswer)
            text, supported = _normalize_inline_citations(
                result.text,
                result.supporting_evidence_ids,
                known,
            )
            if not supported or not text:
                return None
            return VerifiedUndrrStatement(
                statement_id=f"m{report.meta.seq_number:04d}",
                metric_seq=report.meta.seq_number,
                metric_name=report.meta.name,
                text=text,
                supporting_evidence_ids=supported,
            )

    results = await asyncio.gather(*[_draft_one(report) for report in reports])
    return [item for item in results if item is not None]


async def _verify_and_repair(
    statements: Sequence[VerifiedUndrrStatement],
    *,
    evidence: Sequence[ParsedEvidence],
    country_name: str,
    country_iso3: str,
    model: BaseChatModel,
    verifier_model: BaseChatModel,
    config: UndrrSummarizerConfig,
) -> list[VerifiedUndrrStatement]:
    known = _evidence_by_id(evidence)
    semaphore = asyncio.Semaphore(max(1, config.verify_concurrency))

    async def _verify_one(
        statement: VerifiedUndrrStatement,
    ) -> VerifiedUndrrStatement | None:
        async with semaphore:
            current = statement
            for attempt in range(config.max_answer_verification_retries + 1):
                try:
                    verification = await _verify_statement(
                        current,
                        evidence_by_id=known,
                        country_name=country_name,
                        country_iso3=country_iso3,
                        model=verifier_model,
                        config=config,
                    )
                except Exception:
                    logger.exception(
                        "Verification failed for %s; dropping statement",
                        current.statement_id,
                    )
                    return None
                if verification.score > _MIN_KEEP_SCORE:
                    return current
                if attempt >= config.max_answer_verification_retries:
                    logger.info(
                        "Dropping statement %s after retries score=%s",
                        current.statement_id,
                        verification.score,
                    )
                    return None
                try:
                    repaired = await _repair_statement(
                        current,
                        verification,
                        evidence_by_id=known,
                        model=model,
                        config=config,
                    )
                except Exception:
                    logger.exception(
                        "Repair failed for %s; dropping statement",
                        current.statement_id,
                    )
                    return None
                if repaired is None:
                    return None
                current = repaired
            return None

    results = await asyncio.gather(
        *[_verify_one(statement) for statement in statements]
    )
    return [item for item in results if item is not None]


async def _verify_statement(
    statement: VerifiedUndrrStatement,
    *,
    evidence_by_id: dict[str, ParsedEvidence],
    country_name: str,
    country_iso3: str,
    model: BaseChatModel,
    config: UndrrSummarizerConfig,
) -> StatementVerification:
    cited = [
        evidence_by_id[evidence_id]
        for evidence_id in statement.supporting_evidence_ids
        if evidence_id in evidence_by_id
    ]
    text_block = (
        "\n\n".join(
            _format_evidence_for_prompt(
                item, max_source_text_chars=config.max_source_text_chars
            )
            for item in cited
        )
        or "(no evidence)"
    )
    user_text = (
        f"Country: {country_name} ({country_iso3})\n"
        f"Metric: {statement.metric_name}\n"
        f"Statement ({statement.statement_id}): {statement.text}\n\n"
        f"Cited evidence:\n{text_block}"
    )
    image_blocks = [
        _plot_image_block(item, detail=config.default_plot_detail)
        for item in cited
        if item.plot_path is not None
    ]
    user: Any = (
        [{"type": "text", "text": user_text}, *image_blocks]
        if image_blocks
        else user_text
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
        score=result.score,
        unsupported_parts=list(result.unsupported_parts),
        reasoning=result.reasoning,
        suggested_revision=result.suggested_revision,
    )


async def _repair_statement(
    statement: VerifiedUndrrStatement,
    verification: StatementVerification,
    *,
    evidence_by_id: dict[str, ParsedEvidence],
    model: BaseChatModel,
    config: UndrrSummarizerConfig,
) -> VerifiedUndrrStatement | None:
    cited = [
        evidence_by_id[evidence_id]
        for evidence_id in statement.supporting_evidence_ids
        if evidence_id in evidence_by_id
    ]
    claims_block = "\n\n".join(
        _format_evidence_for_prompt(
            item, max_source_text_chars=config.max_source_text_chars
        )
        for item in cited
    )
    user = (
        f"Metric: {statement.metric_name}\n"
        f"Statement: {statement.text}\n"
        f"Verification score: {verification.score}/10\n"
        f"Unsupported parts: {verification.unsupported_parts}\n"
        f"Suggested revision: {verification.suggested_revision}\n"
        f"Reasoning: {verification.reasoning}\n\n"
        f"Cited evidence:\n{claims_block}"
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
    text, supported = _normalize_inline_citations(
        result.text,
        result.supporting_evidence_ids or statement.supporting_evidence_ids,
        evidence_by_id,
    )
    if not supported or not text:
        return None
    return VerifiedUndrrStatement(
        statement_id=statement.statement_id,
        metric_seq=statement.metric_seq,
        metric_name=statement.metric_name,
        text=text,
        supporting_evidence_ids=supported,
    )


def _render_output(
    *,
    country_name: str,
    country_iso3: str,
    use_case_ascii: str,
    statements: Sequence[VerifiedUndrrStatement],
    evidence: Sequence[ParsedEvidence],
    reports: Sequence[ParsedMetricReport],
) -> tuple[str, list[str]]:
    known = _evidence_by_id(evidence)
    by_seq = {item.metric_seq: item for item in statements}
    used_ids: list[str] = []
    for report in reports:
        statement = by_seq.get(report.meta.seq_number)
        if statement is None:
            continue
        inline_ids = _inline_citation_ids(statement.text)
        ordered_ids = inline_ids or list(statement.supporting_evidence_ids)
        for evidence_id in ordered_ids:
            if evidence_id in known and evidence_id not in used_ids:
                used_ids.append(evidence_id)

    number_by_id = {evidence_id: index for index, evidence_id in enumerate(used_ids, 1)}
    references = [
        construct_reference_line(known[evidence_id], country_iso3=country_iso3)
        for evidence_id in used_ids
    ]

    sections: list[tuple[str, str]] = []
    for report in reports:
        statement = by_seq.get(report.meta.seq_number)
        if statement is None:
            sections.append((report.meta.name, "_No supported evidence._"))
            continue
        text = _render_statement_text(
            statement.text,
            statement.supporting_evidence_ids,
            number_by_id,
        )
        sections.append((report.meta.name, text or "_No supported evidence._"))

    markdown = render_undrr_markdown(
        country_name=country_name,
        use_case_ascii=use_case_ascii,
        sections=sections,
        references=references,
    )
    markdown = ensure_citations_present(markdown)
    return markdown, references
