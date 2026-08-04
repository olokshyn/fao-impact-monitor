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
