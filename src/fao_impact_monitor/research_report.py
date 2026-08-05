"""Helpers for research CLI routing and markdown report generation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import markdown
from xhtml2pdf import pisa

from fao_impact_monitor.agent.researcher_agent import (
    STATUS_DISPLAY,
    ResearcherOutput,
    StatementCitation,
    build_status_summary,
    format_source_origin,
)
from fao_impact_monitor.data_source.world_bank import (
    WorldBankDataResult,
    world_bank_indicator_url,
)
from fao_impact_monitor.metric.metric import Metric

MetricPath = Literal["worldbank", "researcher"]

_METRIC_REPORT_FILENAME = re.compile(r"^\d{4}\.md$")
_SECTION_HEADING = re.compile(r"^##\s+(\d+)\.\s+(.*\S)\s*$")

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
a { color: #0645ad; text-decoration: none; }
code { font-family: Courier, monospace; font-size: 10pt; }
"""


def metric_report_path(output_dir: Path, metric_index: int) -> Path:
    """Per-metric markdown path: ``<output_dir>/{metric_index:04d}.md``."""
    return output_dir / f"{metric_index:04d}.md"


def default_research_dir(country_iso3: str) -> Path:
    """Default directory for per-metric research markdown."""
    return Path(f"reports/el-nino-{country_iso3.upper()}")


def default_research_pdf_path(country_iso3: str) -> Path:
    """Default combined PDF path for a country research report."""
    return Path(f"reports/el-nino-{country_iso3.upper()}.pdf")


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


def markdown_to_pdf(markdown_text: str, output_path: Path) -> Path:
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        result = pisa.CreatePDF(html, dest=handle, encoding="utf-8")
    if result.err:
        raise RuntimeError(
            f"Failed to create PDF at {output_path} (errors={result.err})"
        )
    return output_path


def build_research_pdf(
    *,
    input_dir: Path,
    output_path: Path,
) -> Path:
    """Combine metric markdown under ``input_dir`` and write a PDF."""
    files = list_metric_report_files(input_dir)
    combined = combine_metric_reports(files)
    return markdown_to_pdf(combined, output_path)


def is_worldbank_only(metric: Metric) -> bool:
    """Return True when every resolved data source is WorldBank."""
    if not metric.data_sources:
        return False
    return all(s.source == "WorldBank" for s in metric.data_sources)


def metric_path(metric: Metric) -> MetricPath:
    return "worldbank" if is_worldbank_only(metric) else "researcher"


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
    link = f"[{label}]({citation.document_uri})"
    if citation.origin:
        return f"{link} ({citation.origin})"
    return link


def format_worldbank_result(results: list[Any]) -> tuple[str, list[str]]:
    """Return (result markdown, reference markdown lines) for WorldBank data."""
    if not results:
        return ("No World Bank data returned for this metric.", [])

    sections: list[str] = []
    references: list[str] = []
    for result in results:
        if not isinstance(result, WorldBankDataResult):
            title = getattr(result, "title", None) or "World Bank result"
            url = getattr(result, "url", None) or ""
            sections.append(str(title))
            if url:
                references.append(f"- [{title}]({url})")
            continue

        indicator = result.metadata.get("indicator", "")
        country_iso3 = str(result.metadata.get("country_iso3") or "")
        unit = result.metadata.get("unit") or ""
        title = result.title or indicator or "World Bank indicator"
        url = result.url or ""
        if not url and indicator and country_iso3:
            url = world_bank_indicator_url(str(indicator), country_iso3)
        elif not url and indicator:
            url = f"https://data.worldbank.org/indicator/{indicator}"
        df = result.data
        if df is None or df.empty:
            sections.append(f"No time-series values for **{title}**.")
        else:
            lines = [
                f"**{title}**" + (f" ({unit})" if unit else ""),
                "",
                "| Year | Value |",
                "| --- | --- |",
            ]
            # Show most recent years last for readability; include all rows.
            ordered = df.sort_values("year")
            for _, row in ordered.iterrows():
                year = int(row["year"])
                value = row["value"]
                if isinstance(value, float):
                    value_s = f"{value:.4g}"
                else:
                    value_s = str(value)
                lines.append(f"| {year} | {value_s} |")
            sections.append("\n".join(lines))
        if url:
            references.append(f"- [{title}]({url}) (indicator `{indicator}`)")
        elif indicator:
            references.append(f"- World Bank indicator `{indicator}`")
    return ("\n\n".join(sections), references)


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
            refs.append(f"- [{label}]({source.document_uri}) ({origin})")
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
