from __future__ import annotations

import asyncio
from typing import Any

import pytest

from fao_impact_monitor.pdf_pipeline.models import (
    EvidenceUnit,
    PdfDocumentRecord,
    PdfEmbeddingRecord,
    SectionContextCard,
)
from fao_impact_monitor.pdf_pipeline.mongo import purge_pdf_document


class _Collection:
    def __init__(self, *, remaining: int = 0) -> None:
        self.remaining = remaining
        self.delete_filters: list[dict[str, Any]] = []

    async def delete_many(self, query: dict[str, Any]) -> None:
        self.delete_filters.append(query)

    async def count_documents(self, _query: dict[str, Any]) -> int:
        return self.remaining


def _install_collections(
    monkeypatch: pytest.MonkeyPatch, *, remaining_sections: int = 0
) -> dict[str, _Collection]:
    collections = {
        "documents": _Collection(),
        "sections": _Collection(remaining=remaining_sections),
        "evidence": _Collection(),
        "embeddings": _Collection(),
    }
    for model, name in (
        (PdfDocumentRecord, "documents"),
        (SectionContextCard, "sections"),
        (EvidenceUnit, "evidence"),
        (PdfEmbeddingRecord, "embeddings"),
    ):
        monkeypatch.setattr(
            model,
            "get_pymongo_collection",
            staticmethod(lambda name=name: collections[name]),
        )
    return collections


def test_pdf_cleanup_targets_every_stable_hash_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collections = _install_collections(monkeypatch)

    asyncio.run(purge_pdf_document("abc123"))

    section_filter = collections["sections"].delete_filters[0]
    evidence_filter = collections["evidence"].delete_filters[0]
    embedding_filter = collections["embeddings"].delete_filters[0]
    assert {"document_sha256": "abc123"} in section_filter["$or"]
    assert any("evidence_id" in item for item in evidence_filter["$or"])
    assert {"pipeline": "pdf_pipeline"}.items() <= embedding_filter.items()
    assert {next(iter(item)) for item in embedding_filter["$or"]} == {
        "section_id",
        "owner_id",
        "evidence_id",
    }


def test_pdf_cleanup_fails_closed_when_any_old_record_remains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_collections(monkeypatch, remaining_sections=1)

    with pytest.raises(RuntimeError, match="cleanup incomplete"):
        asyncio.run(purge_pdf_document("abc123"))
