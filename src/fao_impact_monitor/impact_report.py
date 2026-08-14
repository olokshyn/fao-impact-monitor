"""Parse metric markdown reports and build impact-analysis outputs."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from fao_impact_monitor.data_source import get_data_source
from fao_impact_monitor.data_source.world_bank import world_bank_indicator_url
from fao_impact_monitor.metric.metric import Metric
from fao_impact_monitor.research_report import (
    default_research_dir,
    list_metric_report_files,
    markdown_to_pdf,
    resolve_use_case_path,
    use_case_display_name,
)
from fao_impact_monitor.utils.document_uri import (
    ascii_relative_path,
    markdown_document_target,
)

logger = logging.getLogger(__name__)

EvidenceKind = Literal["direct", "indirect"]
EvidenceSourceType = Literal["pdf", "web", "worldbank", "faostat", "structured"]

_SECTION_SPLIT = re.compile(r"(?=^## )", re.MULTILINE)
_EVIDENCE_SPLIT = re.compile(
    r"(?=^### (?:Direct|Indirect) Evidence \d+\s*$)", re.MULTILINE
)
_FIELD_SPLIT = re.compile(r"\n\n+")
_SOURCE_TEXT_FENCE = re.compile(
    r"Source text:\s*\n+```(?:\n)?(.*?)```",
    re.DOTALL,
)
_PLOT_MARKDOWN = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_LATEST_VALUE = re.compile(
    r"^Latest value:\s*(.+?)\s*\((\d{4})\)\s*$",
    re.IGNORECASE,
)
_PHYSICAL_PAGES = re.compile(r"[\d]+")
_HTTP_URL = re.compile(r"^https?://", re.IGNORECASE)


@dataclass(frozen=True)
class MetricReportMeta:
    seq_number: int
    name: str
    description: str
    example: str
    unit: str
    path: Path


@dataclass
class ParsedEvidence:
    evidence_id: str
    kind: EvidenceKind
    source_type: EvidenceSourceType
    metric_name: str
    metric_seq: int
    title: str
    source_url: str | None = None
    physical_pages: list[int] = field(default_factory=list)
    printed_pages: list[str] = field(default_factory=list)
    events: str | None = None
    indicator: str | None = None
    source_text: str | None = None
    verified_visual_facts: list[str] = field(default_factory=list)
    plot_path: Path | None = None
    latest_value: str | None = None
    latest_year: int | None = None
    provenance_evidence_id: str | None = None


@dataclass
class ParsedMetricReport:
    meta: MetricReportMeta
    direct: list[ParsedEvidence]
    indirect: list[ParsedEvidence]
    answer: str | None = None


def impact_analysis_stem(
    country_iso3: str,
    *,
    use_case: Path | str = "el-nino",
) -> str:
    """Filename stem, e.g. ``ETH impact analysis El Nino``."""
    use_case_path = resolve_use_case_path(use_case)
    name = use_case_display_name(use_case_path)
    return ascii_relative_path(f"{country_iso3.upper()} impact analysis {name}")


def default_impact_analysis_md_path(
    country_iso3: str,
    *,
    use_case: Path | str = "el-nino",
    output_root: Path | None = None,
) -> Path:
    iso3 = country_iso3.upper()
    use_case_path = resolve_use_case_path(use_case)
    base = (
        (output_root / use_case_path.stem / iso3)
        if output_root is not None
        else default_research_dir(iso3, use_case=use_case_path)
    )
    return base / f"{impact_analysis_stem(iso3, use_case=use_case_path)}.md"


def default_impact_analysis_pdf_path(
    country_iso3: str,
    *,
    use_case: Path | str = "el-nino",
    output_root: Path | None = None,
) -> Path:
    return default_impact_analysis_md_path(
        country_iso3, use_case=use_case, output_root=output_root
    ).with_suffix(".pdf")


def write_impact_analysis_files(
    *,
    markdown_text: str,
    output_md: Path,
    output_pdf: Path | None = None,
) -> tuple[Path, Path]:
    """Write impact-analysis markdown and PDF next to metric reports."""
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(markdown_text, encoding="utf-8")
    pdf_path = output_pdf or output_md.with_suffix(".pdf")
    markdown_to_pdf(markdown_text, pdf_path, base_dir=output_md.parent)
    return output_md, pdf_path


def parse_metric_report_directory(directory: Path) -> list[ParsedMetricReport]:
    """Parse every ``NNNN.md`` metric report under ``directory``."""
    files = list_metric_report_files(directory)
    return [parse_metric_report_file(path) for path in files]


def parse_metric_report_file(path: Path) -> ParsedMetricReport:
    text = path.read_text(encoding="utf-8")
    sections = _split_sections(text)
    meta = _parse_metric_info(sections.get("Metric info", ""), path)
    answer = sections.get("Answer")
    if answer is not None:
        answer = answer.strip() or None

    direct: list[ParsedEvidence] = []
    indirect: list[ParsedEvidence] = []
    for kind, heading in (
        ("direct", "Direct evidence"),
        ("indirect", "Indirect evidence"),
    ):
        body = sections.get(heading, "None.")
        blocks = _split_evidence_blocks(body)
        target = direct if kind == "direct" else indirect
        for index, block in enumerate(blocks, start=1):
            target.append(
                _parse_evidence_block(
                    block,
                    kind=kind,  # type: ignore[arg-type]
                    index=index,
                    meta=meta,
                    report_dir=path.parent,
                )
            )
    return ParsedMetricReport(
        meta=meta,
        direct=direct,
        indirect=indirect,
        answer=answer,
    )


def all_evidence(reports: list[ParsedMetricReport]) -> list[ParsedEvidence]:
    items: list[ParsedEvidence] = []
    for report in reports:
        items.extend(report.direct)
        items.extend(report.indirect)
    return items


def construct_reference_line(
    evidence: ParsedEvidence,
    *,
    country_iso3: str | None = None,
) -> str:
    """Build a bibliography line from Direct/Indirect fields only."""
    if evidence.source_type == "pdf":
        label = evidence.title
        if evidence.physical_pages:
            pages = ", ".join(str(page) for page in evidence.physical_pages)
            label = f"{label}, p. {pages}"
        if evidence.source_url:
            target = markdown_document_target(evidence.source_url)
            return f"[{label}]({target})"
        return label

    if evidence.source_type == "web":
        web_url = evidence.source_url or evidence.title
        title = (
            evidence.title
            if evidence.title and not _HTTP_URL.match(evidence.title)
            else web_url
        )
        if title and title != web_url:
            return f"[{title}]({web_url})"
        return f"[{web_url}]({web_url})"

    if evidence.source_type in {"worldbank", "faostat", "structured"}:
        title = evidence.title
        indicator_url = evidence.source_url
        indicator = evidence.indicator or ""
        if (
            not indicator_url
            and evidence.source_type == "worldbank"
            and indicator
            and country_iso3
        ):
            indicator_url = world_bank_indicator_url(indicator, country_iso3)
        if evidence.source_type == "worldbank":
            suffix = f" (World Bank indicator `{indicator}`)" if indicator else ""
        elif evidence.source_type == "faostat":
            suffix = f" (FAOSTAT `{indicator}`)" if indicator else ""
        else:
            suffix = ""
        if indicator_url:
            return f"[{title}]({indicator_url}){suffix}"
        return f"{title}{suffix}"

    return evidence.title


def load_plot_image_bytes(evidence: ParsedEvidence) -> bytes:
    """Load a plot PNG; raise if the linked plot is missing."""
    if evidence.plot_path is None:
        raise FileNotFoundError(
            f"Evidence {evidence.evidence_id} is a structured indicator without a plot path"
        )
    if not evidence.plot_path.is_file():
        raise FileNotFoundError(
            f"Linked plot missing for evidence {evidence.evidence_id}: "
            f"{evidence.plot_path}"
        )
    return evidence.plot_path.read_bytes()


async def enrich_structured_latest_values(
    reports: list[ParsedMetricReport],
    *,
    country_iso3: str,
    use_case_path: Path,
) -> None:
    """Fetch latest values for WB/FAOSTAT items that lack a Latest value line."""
    metrics = Metric.from_use_case(use_case_path)
    for report in reports:
        for evidence in [*report.direct, *report.indirect]:
            if evidence.source_type not in {"worldbank", "faostat"}:
                continue
            if evidence.latest_value is not None and evidence.latest_year is not None:
                continue
            metric_index = report.meta.seq_number - 1
            if metric_index < 0 or metric_index >= len(metrics):
                continue
            metric = metrics[metric_index]
            await _fill_latest_from_source(
                evidence,
                metric=metric,
                country_iso3=country_iso3,
            )


async def _fill_latest_from_source(
    evidence: ParsedEvidence,
    *,
    metric: Metric,
    country_iso3: str,
) -> None:
    for config in metric.data_sources:
        if evidence.source_type == "worldbank" and config.source != "WorldBank":
            continue
        if evidence.source_type == "faostat" and config.source != "FAOSTAT":
            continue
        if evidence.indicator:
            indicator_name = str(getattr(config, "indicator", "") or "")
            if (
                indicator_name
                and evidence.indicator not in indicator_name
                and (not evidence.title or evidence.title not in indicator_name)
            ):
                continue
        try:
            source = get_data_source(config.source)
            results = await source.get_data(metric, config, country_iso3)
        except Exception:
            logger.exception(
                "Failed to fetch latest value for %s / %s",
                evidence.evidence_id,
                evidence.indicator,
            )
            continue
        for result in results:
            data = getattr(result, "data", None)
            if data is None or getattr(data, "empty", True):
                continue
            if "year" not in data.columns or "value" not in data.columns:
                continue
            ordered = data.dropna(subset=["value"]).sort_values("year")
            if ordered.empty:
                continue
            row = ordered.iloc[-1]
            year = int(row["year"])
            value = row["value"]
            unit = str(getattr(result, "metadata", {}).get("unit") or metric.unit or "")
            if isinstance(value, float) and value.is_integer():
                value_text = str(int(value))
            else:
                value_text = f"{value}"
            unit_text = f" {unit}" if unit else ""
            evidence.latest_value = f"{value_text}{unit_text}"
            evidence.latest_year = year
            if not evidence.source_url and getattr(result, "url", None):
                evidence.source_url = str(result.url)
            return


def _split_sections(text: str) -> dict[str, str]:
    parts = [part.strip() for part in _SECTION_SPLIT.split(text) if part.strip()]
    sections: dict[str, str] = {}
    for part in parts:
        first_line, _, rest = part.partition("\n")
        heading = first_line.removeprefix("## ").strip()
        sections[heading] = rest.strip()
    return sections


def _parse_metric_info(body: str, path: Path) -> MetricReportMeta:
    fields = _parse_simple_fields(body)
    seq_raw = fields.get("Seq Number") or path.stem
    try:
        seq_number = int(seq_raw)
    except ValueError:
        seq_number = int(path.stem)
    return MetricReportMeta(
        seq_number=seq_number,
        name=fields.get("Name", path.stem),
        description=fields.get("Description", ""),
        example=fields.get("Example", ""),
        unit=fields.get("Unit", ""),
        path=path,
    )


def _parse_simple_fields(body: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for chunk in _FIELD_SPLIT.split(body.strip()):
        chunk = chunk.strip()
        if not chunk or ":" not in chunk:
            continue
        key, _, value = chunk.partition(":")
        fields[key.strip()] = value.strip()
    return fields


def _split_evidence_blocks(body: str) -> list[str]:
    stripped = body.strip()
    if not stripped or stripped == "None.":
        return []
    return [
        block.strip()
        for block in _EVIDENCE_SPLIT.split(stripped)
        if block.strip() and block.strip().startswith("### ")
    ]


def _parse_evidence_block(
    block: str,
    *,
    kind: EvidenceKind,
    index: int,
    meta: MetricReportMeta,
    report_dir: Path,
) -> ParsedEvidence:
    heading_match = re.match(
        r"^### (Direct|Indirect) Evidence (\d+)\s*",
        block,
    )
    block_index = int(heading_match.group(2)) if heading_match else index
    body = block[heading_match.end() :].strip() if heading_match else block

    source_text_match = _SOURCE_TEXT_FENCE.search(body)
    source_text = source_text_match.group(1).strip() if source_text_match else None
    body_without_text = (
        body[: source_text_match.start()] + body[source_text_match.end() :]
        if source_text_match
        else body
    )

    fields = _parse_simple_fields(body_without_text)
    plot_match = _PLOT_MARKDOWN.search(body_without_text)
    plot_path: Path | None = None
    if plot_match is not None:
        relative = plot_match.group(2).strip()
        plot_path = (report_dir / relative).resolve()

    source_field = fields.get("Source", "").strip()
    source_url = fields.get("Source url") or None
    source_title = fields.get("Source title") or None
    indicator = fields.get("Indicator") or None
    provenance = fields.get("Evidence id") or None
    events = fields.get("Events")
    if events in {None, "None.", "None"}:
        events = None

    physical_pages = _parse_pages(fields.get("Source physical pages"))
    printed_pages = _parse_printed_pages(fields.get("Source printed pages"))

    latest_value: str | None = None
    latest_year: int | None = None
    for key, value in fields.items():
        if key.casefold() == "latest value":
            match = _LATEST_VALUE.match(f"Latest value: {value}")
            if match:
                latest_value = match.group(1).strip()
                latest_year = int(match.group(2))
            else:
                latest_value = value
            break
    # Also accept full-line form when field parser kept year in value.
    for chunk in _FIELD_SPLIT.split(body_without_text):
        match = _LATEST_VALUE.match(chunk.strip())
        if match:
            latest_value = match.group(1).strip()
            latest_year = int(match.group(2))
            break

    visual_facts: list[str] = []
    if "Verified visual facts:" in body_without_text:
        facts_part = body_without_text.split("Verified visual facts:", 1)[1]
        for line in facts_part.splitlines():
            line = line.strip()
            if line.startswith("- "):
                visual_facts.append(line[2:].strip())

    source_type = _infer_source_type(
        source_field=source_field,
        source_url=source_url,
        indicator=indicator,
        plot_path=plot_path,
        physical_pages=physical_pages,
        provenance=provenance,
    )

    title = source_title or source_field
    url = source_url
    if source_type == "web" and _HTTP_URL.match(source_field):
        url = source_field
        if not source_title:
            title = source_field

    evidence_id = f"m{meta.seq_number:04d}-{kind[0]}{block_index:02d}"
    return ParsedEvidence(
        evidence_id=evidence_id,
        kind=kind,
        source_type=source_type,
        metric_name=meta.name,
        metric_seq=meta.seq_number,
        title=title or f"Evidence {evidence_id}",
        source_url=url,
        physical_pages=physical_pages,
        printed_pages=printed_pages,
        events=events,
        indicator=indicator,
        source_text=source_text,
        verified_visual_facts=visual_facts,
        plot_path=plot_path,
        latest_value=latest_value,
        latest_year=latest_year,
        provenance_evidence_id=provenance,
    )


def _infer_source_type(
    *,
    source_field: str,
    source_url: str | None,
    indicator: str | None,
    plot_path: Path | None,
    physical_pages: list[int],
    provenance: str | None,
) -> EvidenceSourceType:
    if plot_path is not None or (indicator and not provenance and not physical_pages):
        if indicator and re.fullmatch(r"[A-Z]{2}\.[A-Z0-9.]+", indicator or ""):
            return "worldbank"
        if indicator:
            return "faostat"
        return "structured"
    if _HTTP_URL.match(source_field) or (
        source_url is not None and _HTTP_URL.match(source_url)
    ):
        if provenance or physical_pages:
            return "pdf"
        return "web"
    if (
        provenance
        or physical_pages
        or (source_url is not None and source_url.startswith("file://"))
    ):
        return "pdf"
    if source_url is not None and _HTTP_URL.match(source_url):
        return "web"
    return "pdf"


def _parse_pages(raw: str | None) -> list[int]:
    if not raw or raw.strip() in {"None.", "None"}:
        return []
    return [int(match) for match in _PHYSICAL_PAGES.findall(raw)]


def _parse_printed_pages(raw: str | None) -> list[str]:
    if not raw or raw.strip() in {"None.", "None"}:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def render_impact_markdown(
    *,
    country_name: str,
    sections: dict[str, str],
    references: list[str],
) -> str:
    """Assemble the final impact-analysis markdown document."""
    parts = [f"# {country_name}: El Nino Risk Outlook", ""]
    for heading in ("Past impacts", "Expected impacts"):
        body = sections.get(heading, "").strip()
        parts.extend([f"## {heading}", "", body or "_No supported evidence._", ""])
    parts.extend(["## References", ""])
    if references:
        parts.extend(
            f"{index}. {line}" for index, line in enumerate(references, start=1)
        )
        parts.append("")
    else:
        parts.extend(["None.", ""])
    return "\n".join(parts).rstrip() + "\n"


def ensure_citations_present(markdown_text: str) -> str:
    """Drop body paragraphs that lack ``[n]`` citations (keep headings)."""
    citation = re.compile(r"\[\d+\]")
    lines = markdown_text.splitlines()
    kept: list[str] = []
    in_references = False
    for line in lines:
        if line.startswith("## References"):
            in_references = True
            kept.append(line)
            continue
        if in_references:
            kept.append(line)
            continue
        stripped = line.strip()
        is_heading = stripped.startswith("#")
        is_placeholder = stripped.startswith("_") and stripped.endswith("_")
        if not stripped or is_heading or is_placeholder or citation.search(stripped):
            kept.append(line)
            continue
        # Uncited narrative paragraphs are dropped.
        logger.info("Dropping uncited paragraph: %s", stripped[:120])
    return "\n".join(kept).rstrip() + "\n"


def collect_used_reference_numbers(markdown_text: str) -> list[int]:
    return sorted({int(match) for match in re.findall(r"\[(\d+)\]", markdown_text)})
