from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pymupdf
import pytest
from beanie import PydanticObjectId
from bson import BSON

from fao_impact_monitor.pdf_pipeline.artifacts import (
    PdfArtifacts,
    RenderedPage,
    sha256_file,
)
from fao_impact_monitor.pdf_pipeline.ingest import PdfEvidenceIngestor
from fao_impact_monitor.pdf_pipeline.models import (
    ArtifactRef,
    BoundingBox,
    EvidenceUnit,
    PdfDocumentRecord,
    SectionContextCard,
)


def test_artifacts_render_source_pages_and_preserve_exact_text(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    document = pymupdf.open()  # type: ignore[no-untyped-call]
    page = document.new_page()
    page.insert_text((72, 72), "El Nino caused drought in Zambia.")
    document.save(source)  # type: ignore[no-untyped-call]
    document.close()  # type: ignore[no-untyped-call]
    artifacts = PdfArtifacts(tmp_path / "artifacts", sha256_file(source), 72)
    pdf_artifact, pages = artifacts.prepare(source)
    assert pdf_artifact.relative_path.endswith("/source.pdf")
    assert len(pages) == 1
    assert pages[0].physical_page == 1
    assert "El Nino caused drought" in pages[0].extracted_text
    assert (tmp_path / "artifacts" / pages[0].artifact.relative_path).exists()
    assert (
        PdfEvidenceIngestor()._exact_source_text(
            "El Nino\ncaused drought in Zambia.", pages
        )
        == "El Nino caused drought in Zambia."
    )
    assert artifacts.crop(1, BoundingBox(x0=900, y0=900, x1=1000, y1=1000)) is None


def test_section_pdf_contains_only_requested_source_pages(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    document = pymupdf.open()  # type: ignore[no-untyped-call]
    for page_number in range(1, 6):
        page = document.new_page()
        page.insert_text((72, 72), f"Original physical page {page_number}")
    document.save(source)  # type: ignore[no-untyped-call]
    document.close()  # type: ignore[no-untyped-call]
    artifacts = PdfArtifacts(tmp_path / "artifacts", sha256_file(source), 72)
    artifacts.prepare(source)

    excerpt_path = artifacts.section_pdf([2, 3, 4])
    excerpt = pymupdf.open(excerpt_path)  # type: ignore[no-untyped-call]
    try:
        assert len(excerpt) == 3
        extracted_text = [
            excerpt[index].get_text().strip()  # type: ignore[no-untyped-call]
            for index in range(len(excerpt))
        ]
        assert extracted_text == [
            "Original physical page 2",
            "Original physical page 3",
            "Original physical page 4",
        ]
    finally:
        excerpt.close()  # type: ignore[no-untyped-call]


def test_section_request_pages_add_only_available_neighbours() -> None:
    assert PdfEvidenceIngestor._with_adjacent_pages([3, 4], 6) == [2, 3, 4, 5]
    assert PdfEvidenceIngestor._with_adjacent_pages([1], 6) == [1, 2]
    assert PdfEvidenceIngestor._with_adjacent_pages([6], 6) == [5, 6]


def test_publication_date_is_bson_compatible() -> None:
    publication_date = PdfEvidenceIngestor()._mongo_date_or_none("2026-06-29")

    assert publication_date == datetime(2026, 6, 29, tzinfo=UTC)
    BSON.encode({"publication_date": publication_date})


def test_section_boundary_correction_is_limited_to_adjacent_pages() -> None:
    accepted = PdfEvidenceIngestor._validated_section_range(
        {
            "corrected_page_start": 2,
            "corrected_page_end": 5,
            "boundary_rationale": "The heading begins on page 2.",
        },
        page_start=3,
        page_end=4,
        included_pages=[2, 3, 4, 5],
    )
    rejected = PdfEvidenceIngestor._validated_section_range(
        {
            "corrected_page_start": 1,
            "corrected_page_end": 5,
            "boundary_rationale": "Unsupported expansion.",
        },
        page_start=3,
        page_end=4,
        included_pages=[1, 2, 3, 4, 5],
    )

    assert accepted == (2, 5, "The heading begins on page 2.")
    assert rejected[:2] == (3, 4)


def test_section_context_page_evidence_is_rejected_unless_boundary_expands() -> None:
    adjacent_unit = {
        "pages": [2],
        "regions": [{"page": 2, "bbox": None}],
    }

    assert not PdfEvidenceIngestor._draft_within_pages(adjacent_unit, {3, 4})
    assert PdfEvidenceIngestor._draft_within_pages(adjacent_unit, {2, 3, 4})


def test_next_unprocessed_selection_reclaims_processing_and_skips_duplicate_hashes(
    tmp_path: Path,
) -> None:
    first = tmp_path / "a.pdf"
    duplicate = tmp_path / "b.pdf"
    third = tmp_path / "c.pdf"
    for path, content in ((first, b"one"), (duplicate, b"one"), (third, b"three")):
        path.write_bytes(content)
    hashes = {path: sha256_file(path) for path in (first, duplicate, third)}
    records = {
        hashes[first]: {"status": "completed"},
        hashes[third]: {"status": "processing"},
    }
    assert PdfEvidenceIngestor._select_unprocessed(
        [first, duplicate, third], hashes, records, limit=1
    ) == [third]


def test_next_unprocessed_selection_returns_next_failed_document(
    tmp_path: Path,
) -> None:
    first = tmp_path / "a.pdf"
    second = tmp_path / "b.pdf"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    hashes = {path: sha256_file(path) for path in (first, second)}
    selected = PdfEvidenceIngestor._select_unprocessed(
        [first, second], hashes, {hashes[first]: {"status": "completed"}}, limit=1
    )
    assert selected == [second]


def test_explicit_pdf_uses_force_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class RecordingIngestor(PdfEvidenceIngestor):
        called: tuple[Path, bool] | None = None

        async def ingest_pdf(self, pdf: Path, *, force: bool = False) -> bool:
            self.called = (pdf, force)
            return True

    source = tmp_path / "one.pdf"
    source.write_bytes(b"test")
    collection = AsyncMock()
    collection.find_one.return_value = None
    collection.count_documents.return_value = 0
    monkeypatch.setattr(
        PdfDocumentRecord,
        "get_pymongo_collection",
        staticmethod(lambda: collection),
    )
    monkeypatch.setattr(
        SectionContextCard,
        "get_pymongo_collection",
        staticmethod(lambda: collection),
    )
    ingestor = RecordingIngestor()
    assert asyncio.run(ingestor.ingest_single_pdf(source, force=True)) == {
        "processed": 1,
        "skipped": 0,
        "failed": 0,
    }
    assert ingestor.called == (source, True)


def test_explicit_failed_pdf_resumes_when_checkpoint_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class RecordingIngestor(PdfEvidenceIngestor):
        called: tuple[Path, bool] | None = None

        async def ingest_pdf(self, pdf: Path, *, force: bool = False) -> bool:
            self.called = (pdf, force)
            return True

    source = tmp_path / "one.pdf"
    source.write_bytes(b"test")
    document_collection = AsyncMock()
    document_collection.find_one.return_value = {
        "status": "failed",
        "structure_analysis": {"sections": [{"ordinal": 1}]},
    }
    section_collection = AsyncMock()
    section_collection.count_documents.return_value = 1
    monkeypatch.setattr(
        PdfDocumentRecord,
        "get_pymongo_collection",
        staticmethod(lambda: document_collection),
    )
    monkeypatch.setattr(
        SectionContextCard,
        "get_pymongo_collection",
        staticmethod(lambda: section_collection),
    )
    ingestor = RecordingIngestor()

    asyncio.run(ingestor.ingest_single_pdf(source))

    assert ingestor.called == (source, False)


def test_explicit_resume_reclaims_interrupted_processing_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class RecordingIngestor(PdfEvidenceIngestor):
        called: tuple[Path, bool] | None = None

        async def ingest_pdf(self, pdf: Path, *, force: bool = False) -> bool:
            self.called = (pdf, force)
            return True

    source = tmp_path / "one.pdf"
    source.write_bytes(b"test")
    document_collection = AsyncMock()
    document_collection.find_one.return_value = {
        "_id": PydanticObjectId(),
        "status": "processing",
        "structure_analysis": {"sections": [{"ordinal": 1}]},
    }
    section_collection = AsyncMock()
    section_collection.count_documents.return_value = 1
    monkeypatch.setattr(
        PdfDocumentRecord,
        "get_pymongo_collection",
        staticmethod(lambda: document_collection),
    )
    monkeypatch.setattr(
        SectionContextCard,
        "get_pymongo_collection",
        staticmethod(lambda: section_collection),
    )
    ingestor = RecordingIngestor()

    asyncio.run(ingestor.ingest_single_pdf(source))

    assert ingestor.called == (source, False)
    document_collection.update_one.assert_not_awaited()


def test_directory_ingest_logs_document_progress(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    class RecordingIngestor(PdfEvidenceIngestor):
        async def ingest_pdf(self, pdf: Path, *, force: bool = False) -> bool:
            return True

    (tmp_path / "a.pdf").write_bytes(b"one")
    (tmp_path / "b.pdf").write_bytes(b"two")
    with caplog.at_level(logging.INFO):
        result = asyncio.run(RecordingIngestor().ingest_directory(tmp_path, force=True))

    assert result == {"processed": 2, "skipped": 0, "failed": 0}
    messages = caplog.text
    assert "Discovered 2 PDF document(s)" in messages
    assert "[1/2] Starting" in messages
    assert "[2/2] Completed" in messages
    assert "Folder ingestion finished" in messages


def test_directory_ingest_continues_after_one_document_fails(tmp_path: Path) -> None:
    class PartlyFailingIngestor(PdfEvidenceIngestor):
        def __init__(self) -> None:
            super().__init__()
            self.completed: list[str] = []

        async def ingest_pdf(self, pdf: Path, *, force: bool = False) -> bool:
            del force
            if pdf.name == "b.pdf":
                raise RuntimeError("bad PDF")
            self.completed.append(pdf.name)
            return True

    for name in ("a.pdf", "b.pdf", "c.pdf"):
        (tmp_path / name).write_bytes(name.encode())
    ingestor = PartlyFailingIngestor()

    result = asyncio.run(ingestor.ingest_directory(tmp_path, force=True))

    assert result == {"processed": 2, "skipped": 0, "failed": 1}
    assert ingestor.completed == ["a.pdf", "c.pdf"]


def test_document_failure_preserves_section_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import fao_impact_monitor.pdf_pipeline.ingest as ingest_module

    class RecordingCollection:
        def __init__(self) -> None:
            self.updates: list[dict[str, Any]] = []

        async def update_one(
            self, _query: dict[str, object], update: dict[str, Any]
        ) -> None:
            self.updates.append(update)

    collection = RecordingCollection()
    document_id = PydanticObjectId()
    source = tmp_path / "failed.pdf"
    source.write_bytes(b"pdf")
    source_artifact = ArtifactRef(
        relative_path="source.pdf",
        sha256="a" * 64,
        media_type="application/pdf",
    )

    async def claim(**_kwargs: object) -> tuple[dict[str, object], bool]:
        return {"_id": document_id}, True

    async def fail_process(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("analysis failed")

    async def cleanup(*_args: object, **_kwargs: object) -> None:
        assert collection.updates[0]["$set"]["status"] == "failed"

    monkeypatch.setattr(ingest_module, "claim_document", claim)
    monkeypatch.setattr(ingest_module, "delete_document_embeddings", cleanup)
    monkeypatch.setattr(
        PdfDocumentRecord,
        "get_pymongo_collection",
        staticmethod(lambda: collection),
    )
    monkeypatch.setattr(
        PdfArtifacts,
        "prepare",
        lambda _self, _pdf: (source_artifact, []),
    )
    ingestor = PdfEvidenceIngestor()
    monkeypatch.setattr(ingestor, "_process", fail_process)

    with pytest.raises(RuntimeError, match="analysis failed"):
        asyncio.run(ingestor.ingest_pdf(source))

    assert len(collection.updates) == 1
    assert collection.updates[0]["$unset"] == {"completed_at": ""}
    assert collection.updates[0]["$set"]["status"] == "failed"


def test_section_checkpoint_writes_evidence_before_completion_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes: list[str] = []

    class Card:
        section_id = "hash:s001"

        async def insert(self) -> None:
            writes.append("section")

    class DocumentCollection:
        async def update_one(
            self, _query: dict[str, object], _update: dict[str, Any]
        ) -> None:
            writes.append("document_marker")

    async def insert_evidence(_evidence: list[object]) -> None:
        writes.append("evidence")

    monkeypatch.setattr(EvidenceUnit, "insert_many", insert_evidence)
    monkeypatch.setattr(
        PdfDocumentRecord,
        "get_pymongo_collection",
        staticmethod(lambda: DocumentCollection()),
    )

    asyncio.run(
        PdfEvidenceIngestor._save_section_checkpoint(
            document_id=PydanticObjectId(),
            card=Card(),  # type: ignore[arg-type]
            evidence=[object()],  # type: ignore[list-item]
        )
    )

    assert writes == ["evidence", "section", "document_marker"]


def test_empty_gemini_section_recovers_exact_non_overlapping_el_nino_passages(
    tmp_path: Path,
) -> None:
    text = (
        "El Niño conditions emerged in June 2026. Associated dry weather may "
        "reduce cereal yields. Other conditions remained stable. The most recent "
        "El Niño occurred in 2023/24. It caused a record dry spell."
    )
    artifact = ArtifactRef(
        relative_path="pages/page-0001.png",
        sha256="a" * 64,
        media_type="image/png",
    )
    page = RenderedPage(
        physical_page=1,
        width_points=600,
        height_points=800,
        rotation_degrees=0,
        extracted_text=text,
        artifact=artifact,
    )
    ingestor = PdfEvidenceIngestor()

    drafts = ingestor._fallback_scope_drafts(
        pages=[page],
        countries=["ZMB"],
        document_events=[],
        section_events=[],
    )

    assert len(drafts) == 2
    first = drafts[0]["source_text"]
    second = drafts[1]["source_text"]
    assert first in text and second in text
    assert "reduce cereal yields" in first
    assert "record dry spell" in second
    assert first not in second and second not in first
    assert drafts[0]["events"] == [
        {"event_id": "el_nino_2026_27", "relationship": "associated"}
    ]
    assert drafts[1]["events"] == [
        {"event_id": "el_nino_2023_24", "relationship": "attributed"}
    ]


def test_gemini_bounding_box_coordinates_are_normalized() -> None:
    bbox = PdfEvidenceIngestor._normalized_bounding_box(
        [48, 610, 500, 110], prefix="e1"
    )

    assert bbox == BoundingBox(x0=48, y0=110, x1=500, y1=610)
    assert (
        PdfEvidenceIngestor._normalized_bounding_box([48, 110, 48, 610], prefix="e2")
        is None
    )
