"""Unit tests for the tellus_document_fetch stage."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any, TypeVar, cast

from fao_impact_monitor.config import TellusConfig
from fao_impact_monitor.data_lake.common import Status
from fao_impact_monitor.data_lake.document import Document, DocumentType
from fao_impact_monitor.data_lake.documents.tellus_document import TellusDocument
from fao_impact_monitor.data_lake.documents.web_page_document import WebPageDocument
from fao_impact_monitor.data_lake.pipeline import (
    PIPELINE_TELLUS_PROCESS,
    TellusProcessPipeline,
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
)
from fao_impact_monitor.data_lake.stages.embed_chunks_stage import (
    EMBED_CHUNKS_STAGE_NAME,
)
from fao_impact_monitor.data_lake.stages.tellus_document_fetch_stage import (
    TELLUS_DOCUMENT_FETCH_STAGE_NAME,
    TellusDocumentFetchStage,
    TellusDocumentFetchStageResult,
)

T = TypeVar("T")
RunAsync = Callable[[Coroutine[Any, Any, T]], T]


def test_tellus_document_fetch_registration() -> None:
    stage = get_stage(TELLUS_DOCUMENT_FETCH_STAGE_NAME)
    assert isinstance(stage, TellusDocumentFetchStage)
    assert (
        get_stage_result_class(TELLUS_DOCUMENT_FETCH_STAGE_NAME)
        is TellusDocumentFetchStageResult
    )


def test_tellus_process_pipeline_steps() -> None:
    steps_field = TellusProcessPipeline.model_fields["steps"]
    assert steps_field.default_factory is not None
    steps = steps_field.default_factory()
    assert [step.stage_name for step in steps] == [
        TELLUS_DOCUMENT_FETCH_STAGE_NAME,
        COUNTRY_DETECT_STAGE_NAME,
        EMBED_CHUNKS_STAGE_NAME,
    ]
    assert steps[1].params[CHUNK_ITERATOR_PARAM] is extracted_tellus_chunk_iterator
    assert steps[2].params[CHUNK_ITERATOR_PARAM] is extracted_tellus_chunk_iterator
    assert TellusProcessPipeline.model_fields["name"].default == PIPELINE_TELLUS_PROCESS


def test_fetch_writes_chunks_and_metadata(
    document_store: dict[str, Document],
    tellus_dirs: TellusConfig,
    run_async: RunAsync[Any],
) -> None:
    del document_store
    doc = TellusDocument(
        url="tellus://doc-1",
        external_id="doc-1",
        matched_pages=[2],
        pipeline_statuses={PIPELINE_TELLUS_PROCESS: Status.PENDING},
    )

    async def fetch_fn(document_id: str) -> dict[str, Any]:
        assert document_id == "doc-1"
        return {
            "document": {
                "document_id": "doc-1",
                "title": "Drought Outlook",
                "source": {
                    "publisher": "FAO",
                    "publication_year": 2022,
                    "handle_url": "https://example.org/doc-1",
                },
            },
            "chunks": [
                {"content": "Chunk A", "page_num": 1},
                {"content": "Chunk B", "page_num": 2},
            ],
        }

    stage = TellusDocumentFetchStage(config=tellus_dirs, fetch_fn=fetch_fn)
    result = run_async(stage.run(doc, {}, []))
    assert isinstance(result, TellusDocumentFetchStageResult)
    assert result.status == Status.COMPLETED
    assert result.num_pages == 2
    assert doc.title == "Drought Outlook"
    assert doc.matched_pages == [2]
    assert doc.metadata["citation"] == (
        "FAO, Drought Outlook. - 2022 https://example.org/doc-1"
    )
    assert "Pages" not in doc.metadata["citation"]
    assert doc.metadata["url"] == "https://example.org/doc-1"
    assert len(doc.page_paths) == 2
    assert Path(doc.page_paths[0]).read_text(encoding="utf-8") == "Chunk A"
    assert Path(doc.page_paths[1]).read_text(encoding="utf-8") == "Chunk B"


def test_fetch_rejects_non_tellus_document(
    document_store: dict[str, Document],
    tellus_dirs: TellusConfig,
    run_async: RunAsync[Any],
) -> None:
    del document_store
    page = WebPageDocument(url="https://example.com/page")
    stage = TellusDocumentFetchStage(config=tellus_dirs)
    result = run_async(stage.run(page, {}, []))
    assert result.status == Status.FAILED
    assert result.error is not None
    assert "TellusDocument" in result.error


def test_extracted_tellus_chunk_iterator(
    tmp_path: Path,
    document_store: dict[str, Document],
) -> None:
    del document_store
    page1 = tmp_path / "page_0001.md"
    page2 = tmp_path / "page_0002.md"
    page1.write_text("one", encoding="utf-8")
    page2.write_text("two", encoding="utf-8")
    prev = [
        TellusDocumentFetchStageResult(
            version_id="v1",
            status=Status.COMPLETED,
            num_pages=2,
            page_paths=[str(page1), str(page2)],
        )
    ]
    assert list(extracted_tellus_chunk_iterator(cast(list[StageResult], prev))) == [
        "one",
        "two",
    ]


def test_fetch_requires_external_id(
    document_store: dict[str, Document],
    tellus_dirs: TellusConfig,
    run_async: RunAsync[Any],
) -> None:
    del document_store
    doc = TellusDocument(url="tellus://missing", external_id=None)
    assert doc.type == DocumentType.TELLUS
    stage = TellusDocumentFetchStage(config=tellus_dirs)
    result = run_async(stage.run(doc, {}, []))
    assert result.status == Status.FAILED
    assert result.error is not None
    assert "external_id" in result.error
