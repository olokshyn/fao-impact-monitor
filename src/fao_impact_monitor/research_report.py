"""Helpers for research CLI routing and markdown report generation."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote, urlparse

import markdown
import pandas as pd
from pypdf import PdfWriter
from pypdf.generic import ByteStringObject, DictionaryObject, NameObject
from xhtml2pdf import pisa

from fao_impact_monitor.agent.researcher_agent import (
    EvidenceClaim,
    QueryRunStat,
    ResearcherOutput,
    SourceReference,
    format_source_origin,
    is_direct_evidence_claim,
)
from fao_impact_monitor.data_plot import plot_comparison_time_series, plot_time_series
from fao_impact_monitor.data_source.desinventar import (
    DesInventarDataResult,
)
from fao_impact_monitor.data_source.desinventar import (
    national_totals_by_year as desinventar_national_totals_by_year,
)
from fao_impact_monitor.data_source.emdat import (
    EmDatDataResult,
    national_totals_by_year,
)
from fao_impact_monitor.data_source.faostat import FAOSTATDataResult
from fao_impact_monitor.data_source.world_bank import (
    WorldBankDataResult,
    world_bank_indicator_url,
)
from fao_impact_monitor.metric.metric import Metric
from fao_impact_monitor.utils.document_uri import (
    ascii_relative_path,
    markdown_document_target,
)

MetricPath = Literal[
    "worldbank", "faostat", "emdat", "desinventar", "structured", "researcher"
]

_STRUCTURED_DATA_SOURCES = {"FAOSTAT", "WorldBank", "EMDAT", "DesInventar"}

_METRIC_REPORT_FILENAME = re.compile(r"^\d{4}\.md$")
_SECTION_HEADING = re.compile(r"^##\s+(\d+)\.\s+(.*\S)\s*$")
_NUMBERED_H1_HEADING = re.compile(r"^#\s+(\d+)\.\s+(.*\S)\s*$")
_SEQ_NUMBER = re.compile(r"^Seq Number:\s*(\d+)\s*$")
_METRIC_INFO_HEADING = "## Metric info"
_METRIC_INFO_FIELD = re.compile(r"^(Seq Number|Name|Description|Example|Unit):\s*(.*)$")
_MARKDOWN_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_QUERIES_HEADING = "## Queries"
_REFERENCES_HEADING = "## References"
_HTML_HREF = re.compile(r'href="([^"]+)"')
_HTML_TABLE = re.compile(r"<table>.*?</table>", re.DOTALL)
_HTML_TABLE_ROW = re.compile(r"<tr(?:\s[^>]*)?>(.*?)</tr>", re.DOTALL)
_HTML_TABLE_HEADER = re.compile(r"<th(?:\s[^>]*)?>(.*?)</th>", re.DOTALL)
_HTML_TABLE_CELL = re.compile(r"<td(?:\s[^>]*)?>(.*?)</td>", re.DOTALL)
_HTML_TAG = re.compile(r"<[^>]+>")
_DEFAULT_USE_CASE = Path("use-cases/el-nino.json")
_DEFAULT_REPORT_PDF_TEMPLATE = "{country} metrics {name}.pdf"

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
th, td {
  border: 1px solid #ccc;
  padding: 4px 8px;
  text-align: left;
  vertical-align: top;
  word-wrap: break-word;
  overflow-wrap: break-word;
}
th { background: #f3f3f3; }
table.wide-table { font-size: 7pt; }
table.wide-table th, table.wide-table td { padding: 2px; }
table.very-wide-table { font-size: 6pt; }
table.very-wide-table th, table.very-wide-table td { padding: 1px; }
table.emdat-table { font-size: 8pt; }
table.emdat-table th, table.emdat-table td { padding: 3px; }
a { color: #0645ad; text-decoration: underline; }
code { font-family: Courier, monospace; font-size: 10pt; }
pre {
  font-family: Courier, monospace;
  font-size: 9pt;
  white-space: pre-wrap;
  word-wrap: break-word;
}
"""


def metric_report_path(output_dir: Path, metric_index: int) -> Path:
    """Per-metric markdown path: ``<output_dir>/{metric_index:04d}.md``."""
    return output_dir / f"{metric_index:04d}.md"


def metric_human_report_path(output_dir: Path, metric_index: int) -> Path:
    """Human-readable twin: ``<output_dir>/{metric_index:04d}-H.md``."""
    return output_dir / f"{metric_index:04d}-H.md"


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


def use_case_display_name(use_case: Path | str = _DEFAULT_USE_CASE) -> str:
    """Return the use-case ``name`` field, falling back to the file stem."""
    use_case_path = resolve_use_case_path(use_case)
    try:
        payload = json.loads(use_case_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return use_case_path.stem
    if isinstance(payload, dict):
        raw_name = payload.get("name")
        if isinstance(raw_name, str) and raw_name.strip():
            return raw_name.strip()
    return use_case_path.stem


def use_case_data_filter(use_case: Path | str = _DEFAULT_USE_CASE) -> str | None:
    """Return the use-case ``data_filter`` field when present."""
    use_case_path = resolve_use_case_path(use_case)
    try:
        payload = json.loads(use_case_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return None
    if not isinstance(payload, dict):
        return None
    raw = payload.get("data_filter")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


def report_pdf_filename(
    country_iso3: str,
    *,
    use_case: Path | str = _DEFAULT_USE_CASE,
    human: bool = False,
) -> str:
    """Build the combined metrics PDF filename from the use-case template.

    The use-case may define ``report_pdf_template`` with ``{name}`` and
    ``{country}`` placeholders (default: ``"{country} metrics {name}.pdf"``).
    With ``human=True``, `` - human`` is inserted before the extension
    (e.g. ``FJI metrics El Nino - human.pdf``).
    """
    use_case_path = resolve_use_case_path(use_case)
    name = use_case_display_name(use_case_path)
    template = _DEFAULT_REPORT_PDF_TEMPLATE
    try:
        payload = json.loads(use_case_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        payload = {}
    if isinstance(payload, dict):
        raw_template = payload.get("report_pdf_template")
        if isinstance(raw_template, str) and raw_template.strip():
            template = raw_template.strip()
    filename = template.format(name=name, country=country_iso3.upper())
    if not filename.lower().endswith(".pdf"):
        filename = f"{filename}.pdf"
    if human:
        filename = f"{Path(filename).stem} - human.pdf"
    return ascii_relative_path(filename)


def default_research_pdf_path(
    country_iso3: str,
    *,
    use_case: Path | str = _DEFAULT_USE_CASE,
    human: bool = False,
) -> Path:
    """Default combined PDF path for a use case and country."""
    use_case_path = resolve_use_case_path(use_case)
    return default_research_dir(
        country_iso3, use_case=use_case_path
    ) / report_pdf_filename(country_iso3, use_case=use_case_path, human=human)


def ensure_research_output_dir(output_dir: Path) -> Path:
    """Create ``output_dir`` (and parents) if missing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_metric_report(report_path: Path, content: str) -> Path:
    """Write a metric markdown report (parent dir must already exist)."""
    report_path.write_text(content, encoding="utf-8")
    return report_path


def list_metric_report_files(directory: Path, *, human: bool = False) -> list[Path]:
    """Return metric report files sorted by numeric section order.

    Machine files match ``NNNN.md``. With ``human=True``, each researcher
    twin ``NNNN-H.md`` is used when present; structured World Bank / FAOSTAT /
    EM-DAT reports stay on ``NNNN.md`` (so plots remain in the PDF).
    """
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
    files = sorted(files, key=lambda path: int(path.stem))
    if not human:
        return files
    selected: list[Path] = []
    for path in files:
        human_path = metric_human_report_path(path.parent, int(path.stem))
        selected.append(human_path if human_path.is_file() else path)
    return selected


def _parse_metric_section(path: Path) -> tuple[int, str, str]:
    """Return ``(section_number, section_title_line, body_markdown)``.

    New files are ordered by ``Seq Number``. The legacy ``## N.`` heading is
    still accepted so older report directories remain renderable.
    """
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Empty metric report: {path}")

    lines = text.splitlines()
    sequence = next(
        (
            int(match.group(1))
            for line in lines
            if (match := _SEQ_NUMBER.match(line)) is not None
        ),
        None,
    )
    if sequence is not None:
        return sequence, f"Metric {sequence}", text

    if lines:
        h1_match = _NUMBERED_H1_HEADING.match(lines[0])
        if h1_match is not None:
            return int(h1_match.group(1)), lines[0], text

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
    file_number = int(path.stem.removesuffix("-H"))
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


def _section_body_after(text: str, heading: str) -> str:
    """Return the body of ``heading`` up to the next ``## `` section."""
    lines = text.splitlines()
    start = next(
        (index for index, line in enumerate(lines) if line.strip() == heading),
        None,
    )
    if start is None:
        return ""
    end = start + 1
    while end < len(lines) and not (
        lines[end].startswith("## ") and lines[end].strip() != heading
    ):
        end += 1
    return "\n".join(lines[start + 1 : end]).strip()


def _humanize_structured_metric_markdown(text: str, *, section_number: int) -> str:
    """Rewrite World Bank / FAOSTAT Metric info into a short human layout.

    Researcher ``NNNN-H.md`` files and researcher machine reports (which
    include ``## Queries``) are left unchanged.
    """
    stripped = text.strip()
    if not stripped.startswith(_METRIC_INFO_HEADING):
        return text
    if any(line.strip() == _QUERIES_HEADING for line in stripped.splitlines()):
        return text
    lines = stripped.splitlines()
    fields: dict[str, str] = {}
    index = 1
    while index < len(lines):
        line = lines[index]
        if line.startswith("## ") and line.strip() != _METRIC_INFO_HEADING:
            break
        match = _METRIC_INFO_FIELD.match(line)
        if match is not None:
            fields[match.group(1)] = match.group(2).strip()
        index += 1
    images = _MARKDOWN_IMAGE.findall(stripped)
    plots = "\n\n".join(images) if images else "None."
    references = _section_body_after(stripped, _REFERENCES_HEADING) or "None."
    name = fields.get("Name") or f"Metric {section_number}"
    return "\n".join(
        [
            f"# {section_number}. {name}",
            "",
            f"Description: {fields.get('Description', '')}",
            "",
            f"Example: {fields.get('Example', '')}",
            "",
            "## Plots",
            "",
            plots,
            "",
            "## References",
            "",
            references,
            "",
        ]
    )


def combine_metric_reports(
    files: list[Path],
    *,
    title: str = "Research report",
    human: bool = False,
) -> str:
    """Combine per-metric markdown files into one document.

    - Adds the combined-document title only here, never to metric files.
    - Orders new sections by ``Seq Number`` (legacy fallback: ``## N.``).
    - With ``human=True``, World Bank / FAOSTAT Metric info becomes
      ``# N. Title`` plus Description, Example, Plots, and References.
    - Keeps every metric file otherwise intact.
    """
    if not files:
        raise ValueError("No markdown files to combine")

    parsed: list[tuple[int, str]] = []
    for path in files:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        section_number, _heading, body = _parse_metric_section(path)
        if human:
            body = _humanize_structured_metric_markdown(
                body, section_number=section_number
            )
        if body:
            parsed.append((section_number, body))

    if not parsed:
        raise ValueError("No markdown sections to combine")

    parsed.sort(key=lambda item: item[0])
    header = f"# {title.strip() or 'Research report'}"
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

    Preview interprets ``/F`` byte strings as MacRoman. ASCII-fold names like
    ``El Niño`` and unicode dashes so the filespec is always Preview-safe.
    """
    relative_ascii = ascii_relative_path(relative_path)
    mac_roman = relative_ascii.encode("mac_roman")
    return DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Filespec"),
            NameObject("/F"): ByteStringObject(mac_roman),
            NameObject("/UF"): ByteStringObject(
                b"\xfe\xff" + relative_ascii.encode("utf-16-be")
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
    break_on_h2: bool = False,
) -> Path:
    """Render markdown to a PDF file via HTML intermediate.

    Each metric section is wrapped so it starts on a new page and keeps
    its heading with the following content.

    By default only numbered metric headings (``## N. …`` / ``# N. …``) and
    ``## Metric info`` start a section. Pass ``break_on_h2=True`` to start a
    new page on every ``##`` heading (used for UNDRR reports).
    """
    lines = markdown_text.splitlines()
    header_lines: list[str] = []
    sections: list[list[str]] = []
    current: list[str] | None = None
    for line in lines:
        if break_on_h2:
            is_section = line.startswith("## ")
        else:
            is_section = (
                line == _METRIC_INFO_HEADING
                or _SECTION_HEADING.match(line) is not None
                or _NUMBERED_H1_HEADING.match(line) is not None
            )
        if is_section:
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
    html = _classify_wide_html_tables(html)
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


def _classify_wide_html_tables(html: str) -> str:
    """Apply compact PDF styling to tables that cannot fit normal cell padding."""

    def classify(match: re.Match[str]) -> str:
        table = match.group(0)
        headers = _HTML_TABLE_HEADER.findall(table)
        column_count = len(headers)
        if column_count >= 13:
            collapsed = _collapse_emdat_html_table(table, headers)
            if collapsed is not None:
                return collapsed
            css_class = "very-wide-table"
        elif column_count >= 8:
            collapsed = _collapse_desinventar_html_table(table, headers)
            if collapsed is not None:
                return collapsed
            css_class = "wide-table"
        else:
            return table
        return table.replace("<table>", f'<table class="{css_class}">', 1)

    return _HTML_TABLE.sub(classify, html)


def _collapse_desinventar_html_table(table: str, headers: list[str]) -> str | None:
    """Collapse DesInventar fields into five readable PDF columns.

    Location and Regions are stacked with ``<br />`` so xhtml2pdf does not
    paint both cells on top of each other in a cramped 9-column layout.
    """
    names = [_HTML_TAG.sub("", header).strip().casefold() for header in headers]
    index = {name: position for position, name in enumerate(names)}
    required = {
        "event id",
        "start year",
        "start month",
        "start day",
        "disaster type",
        "location",
        "regions",
        "value",
        "unit",
    }
    if not required.issubset(index):
        return None

    rendered_rows: list[str] = []
    for row_html in _HTML_TABLE_ROW.findall(table):
        cells = _HTML_TABLE_CELL.findall(row_html)
        if len(cells) != len(headers):
            continue
        row = {name: cells[position].strip() for name, position in index.items()}
        cell = row.__getitem__

        def joined(*values: str) -> str:
            return "<br />".join(value for value in values if value)

        period = "-".join(
            value
            for value in (
                cell("start year"),
                cell("start month"),
                cell("start day"),
            )
            if value
        )
        area = joined(cell("location"), cell("regions"))
        reported_value = " ".join(
            value for value in (cell("value"), cell("unit")) if value
        )
        rendered_rows.append(
            "<tr>"
            f"<td>{cell('event id') or '&mdash;'}</td>"
            f"<td>{period or '&mdash;'}</td>"
            f"<td>{cell('disaster type') or '&mdash;'}</td>"
            f"<td>{area or '&mdash;'}</td>"
            f"<td>{reported_value or '&mdash;'}</td>"
            "</tr>"
        )

    if not rendered_rows:
        return None
    return (
        '<table class="emdat-table"><thead><tr>'
        '<th width="12%">Event</th>'
        '<th width="14%">Period</th>'
        '<th width="16%">Hazard</th>'
        '<th width="42%">Area</th>'
        '<th width="16%">Reported value</th>'
        "</tr></thead><tbody>" + "".join(rendered_rows) + "</tbody></table>"
    )


def _collapse_emdat_html_table(table: str, headers: list[str]) -> str | None:
    """Collapse the 14 EM-DAT fields into five readable PDF columns."""
    names = [_HTML_TAG.sub("", header).strip().casefold() for header in headers]
    index = {name: position for position, name in enumerate(names)}
    required = {
        "disno.",
        "start year",
        "start month",
        "start day",
        "end year",
        "end month",
        "end day",
        "disaster type",
        "disaster subtype",
        "event name",
        "location",
        "regions",
        "value",
        "unit",
    }
    if not required.issubset(index):
        return None

    rendered_rows: list[str] = []
    for row_html in _HTML_TABLE_ROW.findall(table):
        cells = _HTML_TABLE_CELL.findall(row_html)
        if len(cells) != len(headers):
            continue
        row = {name: cells[position].strip() for name, position in index.items()}
        cell = row.__getitem__

        def joined(*values: str) -> str:
            return "<br />".join(value for value in values if value)

        start_date = "-".join(
            value
            for value in (
                cell("start year"),
                cell("start month"),
                cell("start day"),
            )
            if value
        )
        end_date = "-".join(
            value
            for value in (
                cell("end year"),
                cell("end month"),
                cell("end day"),
            )
            if value
        )
        period = start_date if start_date == end_date else f"{start_date} to {end_date}"
        event = cell("disno.")
        hazard = joined(
            cell("disaster type"),
            cell("disaster subtype"),
            cell("event name"),
        )
        area = joined(cell("location"), cell("regions"))
        reported_value = " ".join(
            value for value in (cell("value"), cell("unit")) if value
        )
        rendered_rows.append(
            "<tr>"
            f"<td>{event or '&mdash;'}</td>"
            f"<td>{period or '&mdash;'}</td>"
            f"<td>{hazard or '&mdash;'}</td>"
            f"<td>{area or '&mdash;'}</td>"
            f"<td>{reported_value or '&mdash;'}</td>"
            "</tr>"
        )

    if not rendered_rows:
        return None
    return (
        '<table class="emdat-table"><thead><tr>'
        '<th width="15%">Event</th>'
        '<th width="15%">Period</th>'
        '<th width="20%">Hazard</th>'
        '<th width="38%">Area</th>'
        '<th width="12%">Reported value</th>'
        "</tr></thead><tbody>" + "".join(rendered_rows) + "</tbody></table>"
    )


def build_research_pdf(
    *,
    input_dir: Path,
    output_path: Path,
    human: bool = False,
) -> Path:
    """Combine metric markdown under ``input_dir`` and write a PDF."""
    files = list_metric_report_files(input_dir, human=human)
    combined = combine_metric_reports(files, title=output_path.stem, human=human)
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
    if metric.data_sources and all(s.source == "EMDAT" for s in metric.data_sources):
        return "emdat"
    if metric.data_sources and all(
        s.source == "DesInventar" for s in metric.data_sources
    ):
        return "desinventar"
    if is_structured_data_only(metric):
        return "structured"
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


def undrr_metric_indices(metrics: Sequence[Metric]) -> list[int]:
    """1-based indices for metrics tagged ``undrr``."""
    return [
        index
        for index, metric in enumerate(metrics, start=1)
        if any(tag.casefold() == "undrr" for tag in metric.tags)
    ]


def parse_metric_option(
    metrics: Sequence[Metric],
    specs: Sequence[str] | None,
) -> list[int] | None:
    """Parse CLI ``--metric`` values as 1-based integers.

    Returns ``None`` when ``specs`` is omitted (run all).
    """
    if specs is None:
        return None
    if not specs:
        raise ValueError("At least one --metric value is required when provided")

    indices: list[int] = []
    seen: set[int] = set()
    n = len(metrics)
    for raw in specs:
        spec = raw.strip()
        if not spec:
            raise ValueError("Metric selector must not be empty")
        try:
            value = int(spec)
        except ValueError as exc:
            raise ValueError(
                f"Invalid metric selector {raw!r}; use a 1-based number"
            ) from exc
        if value < 1 or value > n:
            raise ValueError(
                f"Metric index {value} is out of range; valid range is 1..{n}"
            )
        if value not in seen:
            seen.add(value)
            indices.append(value)
    if not indices:
        raise ValueError("No metrics matched the given --metric selectors")
    return indices


def parse_tags_option(raw: str | None) -> list[str] | None:
    """Parse comma/whitespace ``--tags``; return ``None`` when omitted."""
    if raw is None:
        return None
    tokens = [token.strip() for token in re.split(r"[\s,]+", raw) if token.strip()]
    if not tokens:
        raise ValueError("At least one tag is required when --tags is provided")
    seen: set[str] = set()
    tags: list[str] = []
    for token in tokens:
        key = token.casefold()
        if key in seen:
            continue
        seen.add(key)
        tags.append(token)
    return tags


def metric_has_any_tag(metric: Metric, tags: Sequence[str]) -> bool:
    """True when the metric has at least one of ``tags`` (case-insensitive)."""
    metric_tags = {tag.casefold() for tag in metric.tags}
    return any(tag.casefold() in metric_tags for tag in tags)


def select_metrics_by_tags(
    metrics: Sequence[Metric],
    tags: Sequence[str] | None,
) -> list[int] | None:
    """Return 1-based indices matching any tag, or ``None`` when tags omitted."""
    if tags is None:
        return None
    selected = [
        index
        for index, metric in enumerate(metrics, start=1)
        if metric_has_any_tag(metric, tags)
    ]
    if not selected:
        raise ValueError(f"No metrics matched tags {list(tags)!r}")
    return selected


def resolve_metric_indices(
    metrics: Sequence[Metric],
    *,
    metric_specs: Sequence[str] | None = None,
    tags: Sequence[str] | None = None,
) -> list[int] | None:
    """Intersect ``--metric`` and ``--tags`` selections; ``None`` means all."""
    by_metric = parse_metric_option(metrics, metric_specs)
    by_tags = select_metrics_by_tags(metrics, tags)
    if by_metric is None and by_tags is None:
        return None
    if by_metric is None:
        return by_tags
    if by_tags is None:
        return by_metric
    allowed = set(by_tags)
    selected = [index for index in by_metric if index in allowed]
    if not selected:
        raise ValueError("No metrics matched both --metric and --tags selectors")
    return selected


def parse_countries_iso3(raw: str) -> list[str]:
    """Parse a comma/whitespace-separated ISO3 list; drop empties and duplicates."""
    tokens = [
        token.strip().upper() for token in re.split(r"[\s,]+", raw) if token.strip()
    ]
    if not tokens:
        raise ValueError("At least one ISO3 country code is required")
    seen: set[str] = set()
    countries: list[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        countries.append(token)
    return countries


def parse_countries_file(path: Path) -> list[str]:
    """Parse ISO3 codes from a file (comma/whitespace/newline separated)."""
    text = path.read_text(encoding="utf-8")
    return parse_countries_iso3(text)


def merge_countries_iso3(
    *,
    countries: str | None = None,
    countries_file: Path | None = None,
) -> list[str]:
    """Union ``--countries`` and ``--countries-file`` (file order, then CLI)."""
    merged: list[str] = []
    seen: set[str] = set()
    for source in (
        parse_countries_file(countries_file) if countries_file is not None else [],
        parse_countries_iso3(countries) if countries is not None else [],
    ):
        for iso3 in source:
            if iso3 in seen:
                continue
            seen.add(iso3)
            merged.append(iso3)
    if not merged:
        raise ValueError("Provide --countries and/or --countries-file")
    return merged


def metric_report_is_generated(report_path: Path) -> bool:
    """True when a non-empty metric markdown file already exists."""
    try:
        return report_path.is_file() and report_path.stat().st_size > 0
    except OSError:
        return False


def filter_missing_metric_indices(
    selected: list[tuple[int, Metric]],
    output_dir: Path,
) -> list[tuple[int, Metric]]:
    """Keep only metrics whose ``NNNN.md`` report is not yet generated."""
    return [
        (index, metric)
        for index, metric in selected
        if not metric_report_is_generated(metric_report_path(output_dir, index))
    ]


def format_structured_result(
    results: list[Any],
    *,
    plot_dir: Path,
    plot_stem: str,
) -> tuple[str, list[str]]:
    """Return parseable evidence blocks for structured data results."""
    if not results:
        return ("## Direct evidence\n\nNone.\n\n## Indirect evidence\n\nNone.", [])

    sections: list[str] = []
    references: list[str] = []
    for result_index, result in enumerate(results, start=1):
        if isinstance(result, WorldBankDataResult):
            section, reference = _format_worldbank_evidence(
                result,
                plot_dir=plot_dir,
                plot_stem=f"{plot_stem}-worldbank-{result_index}",
            )
        elif isinstance(result, FAOSTATDataResult):
            section, reference = _format_faostat_evidence(
                result,
                plot_dir=plot_dir,
                plot_stem=f"{plot_stem}-faostat-{result_index}",
            )
        elif isinstance(result, EmDatDataResult):
            section, reference = _format_emdat_evidence(
                result,
                plot_dir=plot_dir,
                plot_stem=f"{plot_stem}-emdat-{result_index}",
            )
        elif isinstance(result, DesInventarDataResult):
            section, reference = _format_desinventar_evidence(
                result,
                plot_dir=plot_dir,
                plot_stem=f"{plot_stem}-desinventar-{result_index}",
            )
        else:
            title = getattr(result, "title", None) or "Structured data result"
            url = getattr(result, "url", None) or ""
            section = f"Source: {title}"
            reference = f"[{title}]({url})" if url else str(title)
        sections.append(f"### Direct Evidence {result_index}\n\n{section}")
        if reference:
            references.append(reference)

    comparison = _format_desinventar_emdat_comparison(
        results,
        plot_dir=plot_dir,
        plot_stem=f"{plot_stem}-comparison",
    )
    if comparison is not None:
        sections.append(f"### Source comparison\n\n{comparison}")

    body = (
        "## Direct evidence\n\n"
        + "\n\n".join(sections)
        + "\n\n## Indirect evidence\n\nNone."
    )
    return body, references


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


def _latest_value_line(data: pd.DataFrame, *, unit: str) -> str | None:
    """Return a Latest value line for the last non-null year in a time series."""
    if data.empty or "year" not in data.columns or "value" not in data.columns:
        return None
    ordered = data.dropna(subset=["value"]).sort_values("year")
    if ordered.empty:
        return None
    row = ordered.iloc[-1]
    year = int(row["year"])
    value = row["value"]
    if isinstance(value, float) and value.is_integer():
        value_text = str(int(value))
    else:
        value_text = f"{value}"
    unit_text = f" {unit}" if unit else ""
    return f"Latest value: {value_text}{unit_text} ({year})"


def _format_worldbank_evidence(
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
    section_parts = [
        f"Source: {title}",
        f"Indicator: {indicator}",
    ]
    if url:
        section_parts.append(f"Source url: {url}")
    latest = _latest_value_line(result.data, unit=unit)
    if latest is not None:
        section_parts.append(latest)
    if plot_path is not None:
        section_parts.append(f"Plot: {_plot_markdown(plot_path, title, plot_dir)}")
    section = "\n\n".join(section_parts)
    reference = (
        f"[{title}]({url}) (World Bank indicator `{indicator}`)"
        if url
        else f"World Bank indicator `{indicator}`"
    )
    return section, reference


def _format_faostat_evidence(
    result: FAOSTATDataResult,
    *,
    plot_dir: Path,
    plot_stem: str,
) -> tuple[str, str]:
    indicator = str(result.metadata.get("indicator") or "")
    title = result.title or indicator or "FAOSTAT indicator"
    unit = str(result.metadata.get("unit") or "")
    if "qualifier" in result.data:
        qualifiers = result.data["qualifier"].dropna().astype(str).unique()
        if len(qualifiers) == 1:
            title = f"{title} — {qualifiers[0]}"
    plot_path = plot_time_series(
        result.data,
        title=title,
        output_path=plot_dir / f"{plot_stem}.png",
        series_columns=("item", "element", "unit"),
        default_unit=unit,
    )
    section_parts = [
        f"Source: {title}",
        f"Indicator: {indicator}",
    ]
    url = result.url or ""
    if url:
        section_parts.append(f"Source url: {url}")
    latest = _latest_value_line(result.data, unit=unit)
    if latest is not None:
        section_parts.append(latest)
    if plot_path is not None:
        section_parts.append(f"Plot: {_plot_markdown(plot_path, title, plot_dir)}")
    section = "\n\n".join(section_parts)
    reference = (
        f"[{title}]({url}) (FAOSTAT `{indicator}`)" if url else f"FAOSTAT `{indicator}`"
    )
    return section, reference


def _format_emdat_evidence(
    result: EmDatDataResult,
    *,
    plot_dir: Path,
    plot_stem: str,
) -> tuple[str, str]:
    indicator = str(result.metadata.get("indicator") or "")
    title = result.title or indicator or "EM-DAT indicator"
    unit = str(result.metadata.get("unit") or "")
    if result.data.empty:
        table = "None."
        plot_path = None
    else:
        national = national_totals_by_year(result.data)
        plot_path = plot_time_series(
            national,
            title=title,
            output_path=plot_dir / f"{plot_stem}.png",
            default_unit=unit,
        )
        table_data = result.data.copy()
        if "unit" not in table_data:
            table_data["unit"] = unit
        table = _dataframe_markdown_table(
            table_data,
            columns=(
                "dis_no",
                "start_year",
                "start_month",
                "start_day",
                "end_year",
                "end_month",
                "end_day",
                "disaster_type",
                "disaster_subtype",
                "event_name",
                "location",
                "regions",
                "value",
                "unit",
            ),
        )

    section_parts = [
        f"Source: {title}",
        "Dataset: EM-DAT",
        f"Indicator: {indicator}",
        f"Source data:\n\n{table}",
    ]
    if plot_path is not None:
        section_parts.append(f"Plot: {_plot_markdown(plot_path, title, plot_dir)}")
    section = "\n\n".join(section_parts)

    url = result.url or ""
    reference = (
        f"[{title}]({url}) (EM-DAT `{indicator}`)" if url else f"EM-DAT `{indicator}`"
    )
    return section, reference


def _format_desinventar_evidence(
    result: DesInventarDataResult,
    *,
    plot_dir: Path,
    plot_stem: str,
) -> tuple[str, str]:
    indicator = str(result.metadata.get("indicator") or "")
    title = result.title or indicator or "DesInventar indicator"
    unit = str(result.metadata.get("unit") or "")
    if result.data.empty:
        table = "None."
        plot_path = None
    else:
        national = desinventar_national_totals_by_year(result.data)
        plot_path = plot_time_series(
            national,
            title=title,
            output_path=plot_dir / f"{plot_stem}.png",
            default_unit=unit,
        )
        table_data = result.data.copy()
        if "unit" not in table_data:
            table_data["unit"] = unit
        table = _dataframe_markdown_table(
            table_data,
            columns=(
                "event_id",
                "start_year",
                "start_month",
                "start_day",
                "disaster_type",
                "location",
                "regions",
                "value",
                "unit",
            ),
        )

    section_parts = [
        f"Source: {title}",
        "Dataset: DesInventar",
        f"Indicator: {indicator}",
        f"Source data:\n\n{table}",
    ]
    if plot_path is not None:
        section_parts.append(f"Plot: {_plot_markdown(plot_path, title, plot_dir)}")
    section = "\n\n".join(section_parts)

    url = result.url or ""
    reference = (
        f"[{title}]({url}) (DesInventar `{indicator}`)"
        if url
        else f"DesInventar `{indicator}`"
    )
    return section, reference


def _format_desinventar_emdat_comparison(
    results: list[Any],
    *,
    plot_dir: Path,
    plot_stem: str,
) -> str | None:
    """Overlay DesInventar and EM-DAT national yearly totals when both are present."""
    rows: list[dict[str, Any]] = []
    has_desinventar = False
    has_emdat = False
    for result in results:
        if isinstance(result, DesInventarDataResult):
            has_desinventar = True
            totals = desinventar_national_totals_by_year(result.data)
            source_label = "DesInventar"
        elif isinstance(result, EmDatDataResult):
            has_emdat = True
            totals = national_totals_by_year(result.data)
            source_label = "EM-DAT"
        else:
            continue
        indicator = str(result.metadata.get("indicator") or result.title or "indicator")
        unit = str(result.metadata.get("unit") or "")
        series = f"{source_label}: {indicator}"
        for year, value in totals[["year", "value"]].itertuples(index=False, name=None):
            if year is None or pd.isna(year) or value is None or pd.isna(value):
                continue
            rows.append(
                {
                    "year": int(year),
                    "value": float(value),
                    "series": series,
                    "unit": unit,
                }
            )
    if not has_desinventar or not has_emdat or not rows:
        return None

    frame = pd.DataFrame(rows)
    title = "DesInventar vs EM-DAT"
    plot_path = plot_comparison_time_series(
        frame,
        title=title,
        output_path=plot_dir / f"{plot_stem}.png",
        series_column="series",
    )
    if plot_path is None:
        return None
    return f"Plot: {_plot_markdown(plot_path, title, plot_dir)}"


def _dataframe_markdown_table(
    data: pd.DataFrame,
    *,
    columns: tuple[str, ...],
) -> str:
    selected = [column for column in columns if column in data.columns]
    if not selected:
        return "None."
    headers = [_column_heading(column) for column in selected]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for values in data[selected].itertuples(index=False, name=None):
        cells = [_escape_table_cell(_table_value(value)) for value in values]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _column_heading(column: str) -> str:
    special = {"dis_no": "DisNo.", "event_id": "Event Id", "value_raw": "Value raw"}
    return special.get(column, column.replace("_", " ").title())


def _table_value(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        return "; ".join(str(item) for item in value)
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).replace("\n", " ")


def _escape_table_cell(text: str) -> str:
    return text.replace("|", "\\|")


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
    """Return parseable direct/indirect evidence and a numbered-citation answer.

    Direct evidence is any quantitative claim that answers the metric subject
    (requested unit or a related quantitative form). Indirect evidence is
    supporting/context evidence that does not itself measure that subject.
    """
    source_by_id = {source.source_id: source for source in output.sources}
    direct_ids: list[str] = []
    indirect_ids: list[str] = []
    for claim in output.claims:
        target = direct_ids if is_direct_evidence_claim(claim) else indirect_ids
        if claim.source_id not in target:
            target.append(claim.source_id)
    indirect_ids = [
        source_id for source_id in indirect_ids if source_id not in direct_ids
    ]
    ordered_ids = [*direct_ids, *indirect_ids]
    sources: list[SourceReference] = []
    for source_id in ordered_ids:
        source = source_by_id.get(source_id)
        if source is None:
            raise ValueError(f"Research evidence source is missing: {source_id}")
        _validate_report_source(source)
        sources.append(source)

    reference_numbers = {
        source.source_id: number for number, source in enumerate(sources, start=1)
    }
    direct_sources = [source_by_id[source_id] for source_id in direct_ids]
    indirect_sources = [source_by_id[source_id] for source_id in indirect_ids]
    direct = _format_evidence_section("Direct", direct_sources)
    indirect = _format_evidence_section("Indirect", indirect_sources)
    answer = _format_numbered_answer(
        output,
        reference_numbers=reference_numbers,
    )
    references = [_format_source_reference(source) for source in sources]
    return f"{direct}\n\n{indirect}\n\n## Answer\n\n{answer}", references


def _validate_report_source(source: SourceReference) -> None:
    if source.source_type == "web":
        if source.source_text is None:
            raise ValueError(f"Web evidence has no source text: {source.source_id}")
        return
    if source.evidence_id is None or not source.physical_pages:
        raise ValueError(
            "Research reports require PDF evidence provenance; rerun with "
            f"`research` (source {source.source_id})."
        )


def _format_evidence_section(
    kind: Literal["Direct", "Indirect"],
    sources: list[SourceReference],
) -> str:
    heading = f"## {kind} evidence"
    if not sources:
        return f"{heading}\n\nNone."
    blocks = [
        _format_evidence_block(kind, index, source)
        for index, source in enumerate(sources, start=1)
    ]
    return f"{heading}\n\n" + "\n\n".join(blocks)


def _clean_source_text(text: str) -> str:
    """Strip each line and collapse consecutive blank lines to one."""
    cleaned: list[str] = []
    previous_blank = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if cleaned and not previous_blank:
                cleaned.append("")
                previous_blank = True
            continue
        cleaned.append(stripped)
        previous_blank = False
    return "\n".join(cleaned)


def _format_source_text(text: str) -> str:
    """Wrap source text in a fenced code block so Markdown is not rendered."""
    cleaned = _clean_source_text(text)
    return f"Source text:\n\n```\n{cleaned}\n```"


def _format_evidence_block(
    kind: Literal["Direct", "Indirect"],
    index: int,
    source: SourceReference,
) -> str:
    # Blank lines between fields so Markdown renders each on its own line.
    # Source text is always last before verified visual facts.
    lines = [f"### {kind} Evidence {index}"]
    if source.source_type == "web":
        lines.extend(
            [
                f"Source: {source.document_uri}",
            ]
        )
        if source.document_name and source.document_name != source.document_uri:
            lines.append(f"Source title: {source.document_name}")
        lines.append(_format_source_text(source.source_text or ""))
        return "\n\n".join(lines)

    events = "; ".join(
        f"{event.event_id}, {event.relationship}" for event in source.events
    )
    lines.extend(
        [
            f"Evidence id: {source.evidence_id}",
            f"Source: {source.document_name}",
            f"Source url: {source.document_uri}",
            "Source physical pages: "
            + ", ".join(str(page) for page in source.physical_pages),
            "Source printed pages: " + (", ".join(source.printed_pages) or "None."),
            f"Events: {events or 'None.'}",
        ]
    )
    lines.append(_format_source_text(source.source_text or "None."))
    if source.verified_visual_facts:
        facts = "\n".join(f"- {fact.text}" for fact in source.verified_visual_facts)
        lines.append(f"Verified visual facts:\n{facts}")
    return "\n\n".join(lines)


def _format_numbered_answer(
    output: ResearcherOutput,
    *,
    reference_numbers: dict[str, int],
) -> str:
    claim_by_id: dict[str, EvidenceClaim] = {
        claim.claim_id: claim for claim in output.claims
    }
    paragraphs: list[str] = []
    for statement in output.statements:
        direct_claims = [
            claim_by_id[claim_id]
            for claim_id in statement.supporting_claim_ids
            if claim_id in claim_by_id
            and is_direct_evidence_claim(claim_by_id[claim_id])
        ]
        if not direct_claims:
            continue
        numbers = list(
            dict.fromkeys(
                reference_numbers[claim.source_id]
                for claim in direct_claims
                if claim.source_id in reference_numbers
            )
        )
        citations = " ".join(f"[{number}]" for number in numbers)
        text = statement.text.strip()
        paragraphs.append(f"{text} {citations}".rstrip())
    if paragraphs:
        return "\n\n".join(paragraphs)
    return (
        "The available evidence cannot answer this metric quantitatively for "
        f"{output.country}."
    )


def _format_source_reference(source: SourceReference) -> str:
    label = source.document_name
    if source.physical_pages:
        page_label = ", ".join(str(page) for page in source.physical_pages)
        label = f"{label}, physical pages {page_label}"
    target = markdown_document_target(source.document_uri)
    origin = format_source_origin(
        source_type=source.source_type,
        document_source=source.document_source,
    )
    return f"[{label}]({target}) ({origin})"


def format_queries_section(query_runs: list[QueryRunStat]) -> str:
    """Render the trailing ## Queries section for a researcher report."""
    lines = ["## Queries", ""]
    if not query_runs:
        lines.append("None.")
        return "\n".join(lines)
    for index, run in enumerate(query_runs, start=1):
        lines.append(
            f"{index}. [{run.destination}] {run.query} — "
            f"returned: {run.results_returned}, accepted: {run.results_accepted}"
        )
    return "\n".join(lines)


def format_metric_section(
    *,
    section_number: int,
    metric: Metric,
    result_markdown: str,
    reference_lines: list[str],
    queries_markdown: str | None = None,
) -> str:
    """Build one markdown section for a metric."""
    unit = metric.unit or "(none)"
    refs = (
        "\n".join(
            f"{number}. {line.removeprefix('- ').strip()}"
            for number, line in enumerate(reference_lines, start=1)
        )
        if reference_lines
        else "None."
    )
    parts = [
        "## Metric info",
        "",
        f"Seq Number: {section_number}",
        "",
        f"Name: {metric.name}",
        "",
        f"Description: {metric.description}",
        "",
        f"Example: {metric.example}",
        "",
        f"Unit: {unit}",
        "",
        result_markdown,
        "",
        "## References",
        "",
        refs,
        "",
    ]
    if queries_markdown:
        parts.extend([queries_markdown.rstrip(), ""])
    return "\n".join(parts)


def build_report(
    *,
    title: str,
    country_iso3: str,
    sections: list[str],
) -> str:
    del title, country_iso3
    return "\n".join(sections).rstrip() + "\n"
