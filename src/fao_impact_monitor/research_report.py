"""Helpers for research CLI routing and markdown report generation."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote, urlparse

import markdown
from pypdf import PdfWriter
from pypdf.generic import ByteStringObject, DictionaryObject, NameObject
from xhtml2pdf import pisa

from fao_impact_monitor.agent.researcher_agent import (
    STATUS_DISPLAY,
    ResearcherOutput,
    StatementCitation,
    build_status_summary,
    format_source_origin,
)
from fao_impact_monitor.data_plot import plot_time_series
from fao_impact_monitor.data_source.faostat import FAOSTATDataResult
from fao_impact_monitor.data_source.world_bank import (
    WorldBankDataResult,
    world_bank_indicator_url,
)
from fao_impact_monitor.metric.metric import Metric
from fao_impact_monitor.utils.document_uri import markdown_document_target

MetricPath = Literal["worldbank", "faostat", "researcher"]

_STRUCTURED_DATA_SOURCES = {"FAOSTAT", "WorldBank"}

_METRIC_REPORT_FILENAME = re.compile(r"^\d{4}\.md$")
_SECTION_HEADING = re.compile(r"^##\s+(\d+)\.\s+(.*\S)\s*$")
_HTML_HREF = re.compile(r'href="([^"]+)"')
_DEFAULT_USE_CASE = Path("use-cases/el-nino.json")
_DEFAULT_REPORT_PDF_TEMPLATE = "{name} - {country}.pdf"

_PDF_HTML_STYLE = """
@page {
  size: a4;
  margin: 1.5cm;
}
body {
  font-family: Helvetica, Arial, sans-serif;
  font-size: 11pt;
  line-height: 1.35;
  color: #222;
}
h1 { font-size: 18pt; margin-top: 0; margin-bottom: 0.8em; }
h2 {
  font-size: 14pt;
  margin-top: 0;
  margin-bottom: 0.6em;
  page-break-after: avoid;
}
h3 {
  font-size: 12pt;
  margin-top: 1em;
  page-break-after: avoid;
}
.metric-section {
  page-break-before: always;
}
table { border-collapse: collapse; width: 100%; margin: 0.6em 0; }
th, td { border: 1px solid #ccc; padding: 4px 8px; text-align: left; }
th { background: #f3f3f3; }
a { color: #0645ad; text-decoration: underline; }
code { font-family: Courier, monospace; font-size: 10pt; }
"""


def metric_report_path(output_dir: Path, metric_index: int) -> Path:
    """Per-metric markdown path: ``<output_dir>/{metric_index:04d}.md``."""
    return output_dir / f"{metric_index:04d}.md"


def default_research_dir(
    country_iso3: str,
    *,
    use_case: Path | str = "el-nino",
) -> Path:
    """Default directory for one use-case and country report set."""
    return Path("reports") / Path(use_case).stem / country_iso3.upper()


def resolve_use_case_path(use_case: Path | str) -> Path:
    """Resolve a use-case path or stem to a JSON file when possible."""
    path = Path(use_case)
    if path.is_file():
        return path
    candidate = Path("use-cases") / f"{path.stem}.json"
    if candidate.is_file():
        return candidate
    return path


def report_pdf_filename(
    country_iso3: str,
    *,
    use_case: Path | str = _DEFAULT_USE_CASE,
) -> str:
    """Build the combined report PDF filename from the use-case template.

    The use-case may define ``report_pdf_template`` with ``{name}`` and
    ``{country}`` placeholders (default: ``"{name} - {country}.pdf"``).
    """
    use_case_path = resolve_use_case_path(use_case)
    name = use_case_path.stem
    template = _DEFAULT_REPORT_PDF_TEMPLATE
    try:
        payload = json.loads(use_case_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        payload = {}
    if isinstance(payload, dict):
        raw_name = payload.get("name")
        if isinstance(raw_name, str) and raw_name.strip():
            name = raw_name.strip()
        raw_template = payload.get("report_pdf_template")
        if isinstance(raw_template, str) and raw_template.strip():
            template = raw_template.strip()
    filename = template.format(name=name, country=country_iso3.upper())
    if not filename.lower().endswith(".pdf"):
        filename = f"{filename}.pdf"
    return filename


def default_research_pdf_path(
    country_iso3: str,
    *,
    use_case: Path | str = _DEFAULT_USE_CASE,
) -> Path:
    """Default combined PDF path for a use case and country."""
    use_case_path = resolve_use_case_path(use_case)
    return default_research_dir(
        country_iso3, use_case=use_case_path
    ) / report_pdf_filename(country_iso3, use_case=use_case_path)


def ensure_research_output_dir(output_dir: Path) -> Path:
    """Create ``output_dir`` (and parents) if missing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_metric_report(report_path: Path, content: str) -> Path:
    """Write a metric markdown report (parent dir must already exist)."""
    report_path.write_text(content, encoding="utf-8")
    return report_path


def list_metric_report_files(directory: Path) -> list[Path]:
    """Return ``NNNN.md`` metric report files sorted by numeric section order."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Research report directory not found: {directory}")
    files = [
        path
        for path in directory.iterdir()
        if path.is_file() and _METRIC_REPORT_FILENAME.match(path.name)
    ]
    if not files:
        raise FileNotFoundError(
            f"No metric markdown files (NNNN.md) found in {directory}"
        )
    return sorted(files, key=lambda path: int(path.stem))


def _parse_metric_section(path: Path) -> tuple[int, str, str]:
    """Return ``(section_number, section_title_line, body_markdown)``.

    ``body_markdown`` starts with the ``## N. Title`` heading and keeps the
    metric's own References block. Extra H1 titles are stripped.
    """
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Empty metric report: {path}")

    lines = text.splitlines()
    body_start = 0
    if lines[0].startswith("# "):
        body_start = 1
        while body_start < len(lines) and not lines[body_start].strip():
            body_start += 1

    body_lines: list[str] = []
    for line in lines[body_start:]:
        # Keep ## / ### ; drop leftover document-level H1 lines only.
        if line.startswith("# ") and not line.startswith("##"):
            continue
        body_lines.append(line)

    # Find the metric section heading.
    heading_idx = next(
        (i for i, line in enumerate(body_lines) if _SECTION_HEADING.match(line)),
        None,
    )
    file_number = int(path.stem)
    if heading_idx is None:
        # Recover a missing ## title from the filename number.
        heading = f"## {file_number}. Metric {file_number}"
        body_lines.insert(0, heading)
        heading_idx = 0
    else:
        heading = body_lines[heading_idx]

    match = _SECTION_HEADING.match(heading)
    assert match is not None
    section_number = int(match.group(1))
    # Ensure heading is first so it is never lost after content reshuffling.
    if heading_idx != 0:
        body_lines = [
            heading,
            *body_lines[:heading_idx],
            *body_lines[heading_idx + 1 :],
        ]
    body = "\n".join(body_lines).strip()
    return section_number, heading, body


def combine_metric_reports(files: list[Path]) -> str:
    """Combine per-metric markdown files into one document.

    - Uses a single top-level ``#`` header (from the first file).
    - Orders sections by their ``## N.`` number (fallback: filename).
    - Keeps each metric section intact, including its own ``### References``.
    """
    if not files:
        raise ValueError("No markdown files to combine")

    title: str | None = None
    parsed: list[tuple[int, str]] = []
    for path in files:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        lines = text.splitlines()
        if title is None and lines and lines[0].startswith("# "):
            title = lines[0].rstrip()
        section_number, _heading, body = _parse_metric_section(path)
        if body:
            parsed.append((section_number, body))

    if not parsed:
        raise ValueError("No markdown sections to combine")

    parsed.sort(key=lambda item: item[0])
    header = title or "# Research report"
    return header + "\n\n" + "\n\n".join(body for _, body in parsed).rstrip() + "\n"


def _make_html_hrefs_clickable(html: str, base_dir: Path | None) -> str:
    """Prefix relative ``href`` values so xhtml2pdf emits link annotations.

    xhtml2pdf only creates annotations for hrefs matching ``^(#|[a-z]+:)``.
    Local PDFs use the ``pdf:`` scheme so the path stays relative to the
    generated report (``fao_data/...`` beside ``report.pdf``). GoToR actions
    are rewritten to portable Launch actions afterward.
    """
    del base_dir

    def replace(match: re.Match[str]) -> str:
        href = match.group(1)
        if href.startswith("#") or urlparse(href).scheme:
            return match.group(0)
        relative = unquote(href)
        if relative.lower().endswith(".pdf"):
            return f'href="pdf:{relative}"'
        return f'href="file:{href}"'

    return _HTML_HREF.sub(replace, html)


def _launch_filespec(relative_path: str) -> DictionaryObject:
    """Build a relative Launch filespec that macOS Preview can open.

    Preview interprets ``/F`` byte strings as MacRoman. NFC + MacRoman preserves
    characters like ``ñ`` and ``’``; UTF-8 filespecs are mojibaked and break.
    """
    relative_nfc = unicodedata.normalize("NFC", relative_path)
    try:
        mac_roman = relative_nfc.encode("mac_roman")
    except UnicodeEncodeError as exc:
        raise ValueError(
            f"Local PDF path is not MacRoman-encodable (Preview-safe): {relative_path!r}"
        ) from exc
    return DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Filespec"),
            NameObject("/F"): ByteStringObject(mac_roman),
            NameObject("/UF"): ByteStringObject(
                b"\xfe\xff" + relative_nfc.encode("utf-16-be")
            ),
        }
    )


def _filespec_path(value: object) -> str:
    if isinstance(value, DictionaryObject):
        raw = value.get("/UF") or value.get("/F")
        if raw is None:
            raise ValueError("Filespec is missing /F and /UF")
        value = raw
    if isinstance(value, bytes):
        for encoding in ("utf-8", "mac_roman", "latin-1"):
            try:
                return unquote(value.decode(encoding))
            except UnicodeDecodeError:
                continue
        return unquote(value.decode("utf-8", errors="replace"))
    return unquote(str(value))


def _as_pdf_object(value: object) -> object:
    get_object = getattr(value, "get_object", None)
    return get_object() if callable(get_object) else value


def _rewrite_local_pdf_links(pdf_path: Path) -> None:
    """Convert relative GoToR links to Preview-safe relative Launch actions."""
    writer = PdfWriter(clone_from=str(pdf_path))
    changed = False
    for page in writer.pages:
        annots = page.get("/Annots")
        if not annots:
            continue
        for annot_ref in annots:
            annot = _as_pdf_object(annot_ref)
            if not isinstance(annot, DictionaryObject):
                continue
            action = annot.get("/A")
            if action is None:
                continue
            action_obj = _as_pdf_object(action)
            if not isinstance(action_obj, DictionaryObject):
                continue
            if str(action_obj.get("/S")) != "/GoToR":
                continue
            target = action_obj.get("/F")
            if target is None:
                continue
            relative = _filespec_path(_as_pdf_object(target))
            annot[NameObject("/A")] = DictionaryObject(
                {
                    NameObject("/Type"): NameObject("/Action"),
                    NameObject("/S"): NameObject("/Launch"),
                    NameObject("/F"): _launch_filespec(relative),
                }
            )
            changed = True
    if not changed:
        return
    tmp_path = pdf_path.with_name(pdf_path.name + ".tmp")
    with tmp_path.open("wb") as handle:
        writer.write(handle)
    tmp_path.replace(pdf_path)


def markdown_to_pdf(
    markdown_text: str,
    output_path: Path,
    *,
    base_dir: Path | None = None,
) -> Path:
    """Render markdown to a PDF file via HTML intermediate.

    Each ``##`` metric section is wrapped so it starts on a new page and keeps
    its heading with the following content.
    """
    lines = markdown_text.splitlines()
    header_lines: list[str] = []
    sections: list[list[str]] = []
    current: list[str] | None = None
    for line in lines:
        if _SECTION_HEADING.match(line):
            if current is not None:
                sections.append(current)
            current = [line]
        elif current is None:
            header_lines.append(line)
        else:
            current.append(line)
    if current is not None:
        sections.append(current)

    header_md = "\n".join(header_lines).strip()
    header_html = (
        markdown.markdown(
            header_md,
            extensions=["tables", "fenced_code", "sane_lists", "nl2br"],
            output_format="html5",
        )
        if header_md
        else ""
    )

    section_html_parts: list[str] = []
    for section_lines in sections:
        section_md = "\n".join(section_lines).strip()
        if not section_md:
            continue
        inner = markdown.markdown(
            section_md,
            extensions=["tables", "fenced_code", "sane_lists", "nl2br"],
            output_format="html5",
        )
        section_html_parts.append(f'<div class="metric-section">{inner}</div>')

    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'/>"
        f"<style>{_PDF_HTML_STYLE}</style></head><body>"
        f"{header_html}{''.join(section_html_parts)}"
        "</body></html>"
    )
    html = _make_html_hrefs_clickable(html, base_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def resolve_asset(uri: str, _relative_uri: str | None) -> str:
        if base_dir is None or urlparse(uri).scheme:
            return uri
        return str((base_dir / unquote(uri)).resolve())

    with output_path.open("wb") as handle:
        result = pisa.CreatePDF(
            html,
            dest=handle,
            encoding="utf-8",
            path=str(base_dir.resolve()) if base_dir is not None else "",
            link_callback=resolve_asset,
        )
    if result.err:
        raise RuntimeError(
            f"Failed to create PDF at {output_path} (errors={result.err})"
        )
    _rewrite_local_pdf_links(output_path)
    return output_path


def build_research_pdf(
    *,
    input_dir: Path,
    output_path: Path,
) -> Path:
    """Combine metric markdown under ``input_dir`` and write a PDF."""
    files = list_metric_report_files(input_dir)
    combined = combine_metric_reports(files)
    return markdown_to_pdf(combined, output_path, base_dir=input_dir)


def is_structured_data_only(metric: Metric) -> bool:
    """Return True when all resolved sources return structured time series."""
    if not metric.data_sources:
        return False
    return all(s.source in _STRUCTURED_DATA_SOURCES for s in metric.data_sources)


def is_worldbank_only(metric: Metric) -> bool:
    """Compatibility helper for callers that specifically inspect World Bank."""
    if not metric.data_sources:
        return False
    return all(s.source == "WorldBank" for s in metric.data_sources)


def metric_path(metric: Metric) -> MetricPath:
    if is_worldbank_only(metric):
        return "worldbank"
    if metric.data_sources and all(s.source == "FAOSTAT" for s in metric.data_sources):
        return "faostat"
    return "researcher"


def select_metrics(
    metrics: list[Metric],
    indices: list[int] | None,
) -> list[tuple[int, Metric]]:
    """Select metrics by 1-based indices; omit indices to select all.

    Returns ``(metric_number, metric)`` pairs where ``metric_number`` starts
    at 1 (used for filenames and markdown section headings).
    """
    if indices is None:
        return [(i + 1, metric) for i, metric in enumerate(metrics)]
    selected: list[tuple[int, Metric]] = []
    n = len(metrics)
    for idx in indices:
        if idx < 1 or idx > n:
            raise ValueError(
                f"Metric index {idx} is out of range; valid range is 1..{n}"
            )
        selected.append((idx, metrics[idx - 1]))
    return selected


def _format_citation(citation: StatementCitation) -> str:
    if citation.page_number is not None:
        label = f"{citation.document_name}, p. {citation.page_number}"
    else:
        label = citation.document_name
    link = f"[{label}]({markdown_document_target(citation.document_uri)})"
    if citation.origin:
        return f"{link} ({citation.origin})"
    return link


def format_structured_result(
    results: list[Any],
    *,
    plot_dir: Path,
    plot_stem: str,
) -> tuple[str, list[str]]:
    """Return plot-only result markdown for World Bank and FAOSTAT data."""
    if not results:
        return ("No structured data returned for this metric.", [])

    sections: list[str] = []
    references: list[str] = []
    for result_index, result in enumerate(results, start=1):
        if isinstance(result, WorldBankDataResult):
            section, reference = _format_worldbank_plot(
                result,
                plot_dir=plot_dir,
                plot_stem=f"{plot_stem}-worldbank-{result_index}",
            )
        elif isinstance(result, FAOSTATDataResult):
            section, reference = _format_faostat_plot(
                result,
                plot_dir=plot_dir,
                plot_stem=f"{plot_stem}-faostat-{result_index}",
            )
        else:
            title = getattr(result, "title", None) or "Structured data result"
            url = getattr(result, "url", None) or ""
            section = str(title)
            reference = f"- [{title}]({url})" if url else ""
        sections.append(section)
        if reference:
            references.append(reference)
    return ("\n\n".join(sections), references)


def format_worldbank_result(
    results: list[Any],
    *,
    plot_dir: Path,
    plot_stem: str,
) -> tuple[str, list[str]]:
    """Compatibility wrapper for World Bank-only callers."""
    return format_structured_result(
        results,
        plot_dir=plot_dir,
        plot_stem=plot_stem,
    )


def _format_worldbank_plot(
    result: WorldBankDataResult,
    *,
    plot_dir: Path,
    plot_stem: str,
) -> tuple[str, str]:
    indicator = result.metadata.get("indicator", "")
    country_iso3 = str(result.metadata.get("country_iso3") or "")
    unit = str(result.metadata.get("unit") or "")
    title = result.title or indicator or "World Bank indicator"
    url = result.url or ""
    if not url and indicator and country_iso3:
        url = world_bank_indicator_url(str(indicator), country_iso3)
    elif not url and indicator:
        url = f"https://data.worldbank.org/indicator/{indicator}"

    plot_path = plot_time_series(
        result.data,
        title=title,
        output_path=plot_dir / f"{plot_stem}.png",
        default_unit=unit,
    )
    section = _plot_markdown(plot_path, title, plot_dir)
    reference = (
        f"- [{title}]({url}) (indicator `{indicator}`)"
        if url
        else f"- World Bank indicator `{indicator}`"
    )
    return section, reference


def _format_faostat_plot(
    result: FAOSTATDataResult,
    *,
    plot_dir: Path,
    plot_stem: str,
) -> tuple[str, str]:
    indicator = str(result.metadata.get("indicator") or "")
    title = result.title or indicator or "FAOSTAT indicator"
    if "qualifier" in result.data:
        qualifiers = result.data["qualifier"].dropna().astype(str).unique()
        if len(qualifiers) == 1:
            title = f"{title} — {qualifiers[0]}"
    plot_path = plot_time_series(
        result.data,
        title=title,
        output_path=plot_dir / f"{plot_stem}.png",
        series_columns=("item", "element", "unit"),
        default_unit=str(result.metadata.get("unit") or ""),
    )
    section = _plot_markdown(plot_path, title, plot_dir)
    url = result.url or ""
    reference = (
        f"- [{title}]({url}) (FAOSTAT `{indicator}`)"
        if url
        else f"- FAOSTAT `{indicator}`"
    )
    return section, reference


def _plot_markdown(plot_path: Path | None, title: str, plot_dir: Path) -> str:
    if plot_path is None:
        return (
            f"No readable plot could be generated for **{title}** because the "
            "result contains too many distinct time series or no numeric values."
        )
    relative_path = plot_path.relative_to(plot_dir.parent).as_posix()
    return f"![{title}]({relative_path})"


def format_researcher_result(
    output: ResearcherOutput,
) -> tuple[str, list[str]]:
    """Return (result markdown, reference markdown lines) for ResearcherAgent.

    Always includes a Status line and best-effort findings. Status values:
    answered; high level answer, lacking detailed evidence; cannot answer
    with available evidence.
    """
    body = build_status_summary(
        status=output.status,
        country_name=output.country,
        statements=output.statements,
        gaps=output.open_gaps,
    )
    if not body.strip():
        body = output.final_summary.strip() or "(empty researcher summary)"
    status_label = STATUS_DISPLAY[output.status]
    result_body = f"**Status:** {status_label}\n\n{body}"

    refs: list[str] = []
    seen: set[tuple[str, int | None]] = set()
    for statement in output.statements:
        for citation in statement.citations:
            key = (citation.document_uri, citation.page_number)
            if key in seen:
                continue
            seen.add(key)
            refs.append(f"- {_format_citation(citation)}")
    # Fall back to sources if statements lack citations.
    if not refs:
        for source in output.sources:
            key = (source.document_uri, source.page_number)
            if key in seen:
                continue
            seen.add(key)
            if source.page_number is not None:
                label = f"{source.document_name}, p. {source.page_number}"
            else:
                label = source.document_name
            origin = format_source_origin(
                source_type=source.source_type,
                document_source=source.document_source,
            )
            target = markdown_document_target(source.document_uri)
            refs.append(f"- [{label}]({target}) ({origin})")
    return result_body, refs


def format_metric_section(
    *,
    section_number: int,
    metric: Metric,
    result_markdown: str,
    reference_lines: list[str],
) -> str:
    """Build one markdown section for a metric."""
    unit = metric.unit or "(none)"
    refs = "\n".join(reference_lines) if reference_lines else "- (none)"
    return "\n".join(
        [
            f"## {section_number}. {metric.name}",
            "",
            f"**Description:** {metric.description}",
            "",
            f"**Example:** {metric.example}",
            "",
            f"**Unit:** {unit}",
            "",
            "### Result",
            "",
            result_markdown,
            "",
            "### References",
            "",
            refs,
            "",
        ]
    )


def build_report(
    *,
    title: str,
    country_iso3: str,
    sections: list[str],
) -> str:
    header = f"# {title} - {country_iso3.upper()}\n\n"
    return header + "\n".join(sections).rstrip() + "\n"
