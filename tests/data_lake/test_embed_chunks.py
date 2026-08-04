"""Unit tests for the embed_chunks stage."""

from __future__ import annotations

from collections.abc import Callable, Coroutine, Iterator, Sequence
from pathlib import Path
from typing import Any, TypeVar

from fao_impact_monitor.config import VectorStoreConfig
from fao_impact_monitor.data_lake.common import Status
from fao_impact_monitor.data_lake.document import Document
from fao_impact_monitor.data_lake.documents.pdf_document import PdfDocument
from fao_impact_monitor.data_lake.pipeline import (
    PdfProcessPipeline,
    TellusProcessPipeline,
    extracted_pdf_chunk_iterator,
    extracted_tellus_chunk_iterator,
)
from fao_impact_monitor.data_lake.stage import (
    StageResult,
    get_stage,
    get_stage_result_class,
)
from fao_impact_monitor.data_lake.stages.country_detect_stage import (
    CHUNK_ITERATOR_PARAM,
    COUNTRY_DETECT_STAGE_NAME,
    CountryDetection,
    CountryDetectStageResult,
)
from fao_impact_monitor.data_lake.stages.embed_chunks_stage import (
    EMBED_CHUNKS_STAGE_NAME,
    EmbedChunksStage,
    EmbedChunksStageResult,
)
from fao_impact_monitor.data_lake.stages.pdf_extract_stage import (
    PDF_EXTRACT_STAGE_NAME,
    PdfExtractStageResult,
)
from fao_impact_monitor.data_lake.stages.tellus_document_fetch_stage import (
    TELLUS_DOCUMENT_FETCH_STAGE_NAME,
)
from fao_impact_monitor.data_lake.vectorstore import ChunkEmbedding

T = TypeVar("T")
RunAsync = Callable[[Coroutine[Any, Any, T]], T]


def _params(
    chunk_iterator: Callable[[list[StageResult]], Iterator[str]],
) -> dict[str, Any]:
    return {CHUNK_ITERATOR_PARAM: chunk_iterator}


def test_embed_chunks_registration() -> None:
    assert isinstance(get_stage(EMBED_CHUNKS_STAGE_NAME), EmbedChunksStage)
    assert get_stage_result_class(EMBED_CHUNKS_STAGE_NAME) is EmbedChunksStageResult


def test_pdf_process_pipeline_includes_embed_chunks() -> None:
    steps_field = PdfProcessPipeline.model_fields["steps"]
    assert steps_field.default_factory is not None
    steps = steps_field.default_factory()
    assert [step.stage_name for step in steps] == [
        PDF_EXTRACT_STAGE_NAME,
        COUNTRY_DETECT_STAGE_NAME,
        EMBED_CHUNKS_STAGE_NAME,
    ]
    assert steps[2].params[CHUNK_ITERATOR_PARAM] is extracted_pdf_chunk_iterator


def test_tellus_process_pipeline_includes_embed_chunks() -> None:
    steps_field = TellusProcessPipeline.model_fields["steps"]
    assert steps_field.default_factory is not None
    steps = steps_field.default_factory()
    assert [step.stage_name for step in steps] == [
        TELLUS_DOCUMENT_FETCH_STAGE_NAME,
        COUNTRY_DETECT_STAGE_NAME,
        EMBED_CHUNKS_STAGE_NAME,
    ]
    assert steps[2].params[CHUNK_ITERATOR_PARAM] is extracted_tellus_chunk_iterator


def _extract_result(page_paths: list[str]) -> PdfExtractStageResult:
    return PdfExtractStageResult(
        version_id="extract-v1",
        status=Status.COMPLETED,
        title="Sample",
        num_pages=len(page_paths),
        page_paths=page_paths,
    )


def _country_result(
    detections: list[CountryDetection],
) -> CountryDetectStageResult:
    return CountryDetectStageResult(
        version_id="country-v1",
        status=Status.COMPLETED,
        detections=detections,
    )


def test_embed_chunks_writes_embeddings_with_countries(
    document_store: dict[str, Document],
    run_async: RunAsync[Any],
    tmp_path: Path,
) -> None:
    del document_store
    page1 = tmp_path / "page-1.md"
    page2 = tmp_path / "page-2.md"
    page1.write_text("Farmers in Kenya.", encoding="utf-8")
    page2.write_text("Markets in Uganda.", encoding="utf-8")

    doc = PdfDocument(
        url="https://example.com/report.pdf",
        source="FaoRepository",
        title="Report",
        metadata={"year": "2024"},
        pipeline_statuses={"pdf_process": Status.PENDING},
    )
    run_async(doc.insert())
    assert doc.id is not None

    async def embed_fn(texts: Sequence[str]) -> list[list[float]]:
        return [[float(i), 0.0] for i in range(len(texts))]

    stage = EmbedChunksStage(embed_fn=embed_fn, config=VectorStoreConfig())
    prev: list[StageResult] = [
        _extract_result([str(page1), str(page2)]),
        _country_result(
            [
                CountryDetection(
                    countries_iso3=["KEN"],
                    detections=["Kenya"],
                    error=None,
                ),
                CountryDetection(
                    countries_iso3=["UGA"],
                    detections=["Uganda"],
                    error=None,
                ),
            ]
        ),
    ]
    result = run_async(stage.run(doc, _params(extracted_pdf_chunk_iterator), prev))

    assert result.status == Status.COMPLETED
    assert isinstance(result, EmbedChunksStageResult)
    assert result.chunk_count == 2

    chunks = run_async(ChunkEmbedding.find_all().to_list())
    assert len(chunks) == 2
    by_index = {c.chunk_index: c for c in chunks}
    assert by_index[0].chunk_text == "Farmers in Kenya."
    assert by_index[0].countries_iso3 == ["KEN"]
    assert by_index[0].document_source == "FaoRepository"
    assert by_index[0].document_id == doc.id
    assert by_index[0].document_title == "Report"
    assert by_index[0].document_meta == {"year": "2024"}
    assert by_index[0].embedding == [0.0, 0.0]
    assert by_index[1].countries_iso3 == ["UGA"]
    assert by_index[1].embedding == [1.0, 0.0]


def test_embed_chunks_deletes_existing_rows_before_insert(
    document_store: dict[str, Document],
    run_async: RunAsync[Any],
    tmp_path: Path,
) -> None:
    del document_store
    page = tmp_path / "page-1.md"
    page.write_text("Only one chunk.", encoding="utf-8")

    doc = PdfDocument(
        url="https://example.com/old.pdf",
        source="FaoRepository",
        pipeline_statuses={"pdf_process": Status.PENDING},
    )
    run_async(doc.insert())
    assert doc.id is not None

    stale = ChunkEmbedding(
        document_id=doc.id,
        document_url=doc.url,
        document_type=doc.type,
        document_source=doc.source,
        chunk_index=99,
        chunk_text="stale",
        countries_iso3=["XXX"],
        embedding=[9.0],
    )
    run_async(stale.insert())

    async def embed_fn(texts: Sequence[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]

    stage = EmbedChunksStage(embed_fn=embed_fn, config=VectorStoreConfig())
    prev: list[StageResult] = [
        _extract_result([str(page)]),
        _country_result(
            [
                CountryDetection(
                    countries_iso3=["KEN"],
                    detections=["Kenya"],
                    error=None,
                )
            ]
        ),
    ]
    result = run_async(stage.run(doc, _params(extracted_pdf_chunk_iterator), prev))
    assert result.status == Status.COMPLETED

    chunks = run_async(ChunkEmbedding.find_all().to_list())
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].chunk_text == "Only one chunk."
    assert chunks[0].countries_iso3 == ["KEN"]


def test_embed_chunks_fails_without_country_detect(
    document_store: dict[str, Document],
    run_async: RunAsync[Any],
    tmp_path: Path,
) -> None:
    del document_store
    page = tmp_path / "page-1.md"
    page.write_text("text", encoding="utf-8")
    doc = PdfDocument(
        url="https://example.com/missing-country.pdf",
        pipeline_statuses={"pdf_process": Status.PENDING},
    )
    run_async(doc.insert())

    async def embed_fn(texts: Sequence[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    stage = EmbedChunksStage(embed_fn=embed_fn, config=VectorStoreConfig())
    result = run_async(
        stage.run(
            doc,
            _params(extracted_pdf_chunk_iterator),
            [_extract_result([str(page)])],
        )
    )
    assert result.status == Status.FAILED
    assert result.error is not None
    assert COUNTRY_DETECT_STAGE_NAME in result.error


def test_embed_chunks_fails_on_misaligned_detections(
    document_store: dict[str, Document],
    run_async: RunAsync[Any],
    tmp_path: Path,
) -> None:
    del document_store
    page1 = tmp_path / "page-1.md"
    page2 = tmp_path / "page-2.md"
    page1.write_text("a", encoding="utf-8")
    page2.write_text("b", encoding="utf-8")
    doc = PdfDocument(
        url="https://example.com/misaligned.pdf",
        pipeline_statuses={"pdf_process": Status.PENDING},
    )
    run_async(doc.insert())

    async def embed_fn(texts: Sequence[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    stage = EmbedChunksStage(embed_fn=embed_fn, config=VectorStoreConfig())
    prev: list[StageResult] = [
        _extract_result([str(page1), str(page2)]),
        _country_result(
            [
                CountryDetection(
                    countries_iso3=["KEN"],
                    detections=["Kenya"],
                    error=None,
                )
            ]
        ),
    ]
    result = run_async(stage.run(doc, _params(extracted_pdf_chunk_iterator), prev))
    assert result.status == Status.FAILED
    assert result.error is not None
    assert "does not match" in result.error


def test_embed_chunks_uses_empty_countries_on_detection_error(
    document_store: dict[str, Document],
    run_async: RunAsync[Any],
    tmp_path: Path,
) -> None:
    del document_store
    page = tmp_path / "page-1.md"
    page.write_text("unknown place", encoding="utf-8")
    doc = PdfDocument(
        url="https://example.com/error-country.pdf",
        pipeline_statuses={"pdf_process": Status.PENDING},
    )
    run_async(doc.insert())

    async def embed_fn(texts: Sequence[str]) -> list[list[float]]:
        return [[0.5] for _ in texts]

    stage = EmbedChunksStage(embed_fn=embed_fn, config=VectorStoreConfig())
    prev: list[StageResult] = [
        _extract_result([str(page)]),
        _country_result(
            [
                CountryDetection(
                    countries_iso3=None,
                    detections=None,
                    error="boom",
                )
            ]
        ),
    ]
    result = run_async(stage.run(doc, _params(extracted_pdf_chunk_iterator), prev))
    assert result.status == Status.COMPLETED
    chunks = run_async(ChunkEmbedding.find_all().to_list())
    assert len(chunks) == 1
    assert chunks[0].countries_iso3 == []
