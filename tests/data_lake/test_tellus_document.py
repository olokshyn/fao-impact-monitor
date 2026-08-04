"""Unit tests for TellusDocument."""

from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

from fao_impact_monitor.data_lake.document import Document, DocumentType
from fao_impact_monitor.data_lake.documents.tellus_document import (
    TELLUS_DOCUMENT_TYPE,
    TellusDocument,
    build_tellus_full_cite_as,
    format_page_ranges,
)

T = TypeVar("T")
RunAsync = Callable[[Coroutine[Any, Any, T]], T]


def test_tellus_document_type_and_fields(
    document_store: dict[str, Document],
) -> None:
    del document_store
    doc = TellusDocument(
        url="tellus://abc",
        external_id="abc",
        matched_pages=[1, 3],
        page_paths=["/tmp/page_0001.md"],
        title="Sample",
    )
    assert doc.type == DocumentType.TELLUS
    assert TELLUS_DOCUMENT_TYPE == DocumentType.TELLUS
    assert doc.matched_pages == [1, 3]
    assert doc.page_paths == ["/tmp/page_0001.md"]
    assert doc.citation == "Sample (tellus://abc)"


def test_citation_prefers_metadata(
    document_store: dict[str, Document],
) -> None:
    del document_store
    doc = TellusDocument(
        url="tellus://abc",
        external_id="abc",
        metadata={"citation": "FAO, Title. - 2020 https://example.org"},
    )
    assert doc.citation == "FAO, Title. - 2020 https://example.org"


def test_format_page_ranges() -> None:
    assert format_page_ranges([]) == ""
    assert format_page_ranges([1, 2, 3, 5, 7, 8]) == "1-3, 5, 7-8"


def test_build_tellus_full_cite_as_without_pages() -> None:
    assert (
        build_tellus_full_cite_as(
            publisher="FAO",
            title="Water Report",
            year="2021",
            link="https://example.org/d1",
        )
        == "FAO, Water Report. - 2021 https://example.org/d1"
    )


def test_build_tellus_full_cite_as_with_pages() -> None:
    assert (
        build_tellus_full_cite_as(
            publisher="FAO",
            title="Water Report",
            year="2021",
            link="https://example.org/d1",
            pages=[2, 3, 5],
        )
        == "FAO, Water Report. - 2021. Pages 2-3, 5. https://example.org/d1"
    )
