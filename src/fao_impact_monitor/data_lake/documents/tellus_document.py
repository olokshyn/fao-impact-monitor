from typing import Any

from pydantic import Field, computed_field

from fao_impact_monitor.data_lake.document import Document, DocumentType

TELLUS_DOCUMENT_TYPE = DocumentType.TELLUS

MAX_TITLE_LENGTH = 80


class TellusDocument(Document):
    matched_pages: list[int] = Field(default_factory=list)
    page_paths: list[str] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def citation(self) -> str:
        meta_citation = self.metadata.get("citation")
        if isinstance(meta_citation, str) and meta_citation.strip():
            return meta_citation
        title = self.title or "Untitled Tellus document"
        return f"{title} ({self.url})"

    class Settings:
        class_id_value = TELLUS_DOCUMENT_TYPE


def format_page_ranges(pages: list[int]) -> str:
    """Collapse sorted page numbers into compact inclusive ranges."""
    if not pages:
        return ""

    ranges: list[str] = []
    start = previous = pages[0]
    for page in pages[1:]:
        if page == previous + 1:
            previous = page
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = page
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ", ".join(ranges)


def build_tellus_full_cite_as(
    *,
    publisher: str,
    title: str,
    year: str,
    link: str,
    pages: list[int] | None = None,
) -> str:
    """Build a full Tellus citation, optionally including page ranges."""
    pages_text = ""
    if pages:
        page_ranges = format_page_ranges(sorted(set(pages)))
        if page_ranges:
            pages_text = f". Pages {page_ranges}."
    return f"{publisher}, {title}. - {year}{pages_text} {link}".strip()


def build_tellus_metadata(
    document_payload: dict[str, Any],
    *,
    external_id: str,
) -> dict[str, Any]:
    """Build Tellus document metadata without page-specific citations.

    Page numbers are selected later in the pipeline; citations here omit pages.
    """
    title = str(document_payload.get("title") or "Untitled Tellus document")
    source_raw = document_payload.get("source") or {}
    source = source_raw if isinstance(source_raw, dict) else {}
    link = str(source.get("handle_url") or "")
    short_title = (
        f"{title[:MAX_TITLE_LENGTH]}..." if len(title) > MAX_TITLE_LENGTH else title
    ).strip()
    publisher = str(source.get("publisher", "FAO")).replace(";", "").strip()
    year = str(source.get("publication_year", ""))
    full_cite_as = build_tellus_full_cite_as(
        publisher=publisher,
        title=title,
        year=year,
        link=link,
    )
    short_cite_as = f"[{short_title}]({link})"

    return {
        "document_id": str(document_payload.get("document_id") or external_id),
        "title": title,
        "year": year,
        "source": source,
        "url": link,
        "citation": full_cite_as,
        "short_cite_as": short_cite_as,
    }
