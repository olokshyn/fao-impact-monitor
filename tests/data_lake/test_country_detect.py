"""Unit tests for the country_detect stage."""

from __future__ import annotations

from collections.abc import Callable, Coroutine, Iterator
from pathlib import Path
from typing import Any, TypeVar

import pytest
from pydantic import ValidationError

from fao_impact_monitor.config import CountryDetectConfig
from fao_impact_monitor.data_lake.common import Status
from fao_impact_monitor.data_lake.document import Document
from fao_impact_monitor.data_lake.documents.pdf_document import PdfDocument
from fao_impact_monitor.data_lake.pipeline import (
    PdfProcessPipeline,
    extracted_pdf_chunk_iterator,
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
    CountryDetectStage,
    CountryDetectStageResult,
)
from fao_impact_monitor.data_lake.stages.embed_chunks_stage import (
    EMBED_CHUNKS_STAGE_NAME,
)
from fao_impact_monitor.data_lake.stages.pdf_extract_stage import (
    PDF_EXTRACT_STAGE_NAME,
    PdfExtractStageResult,
)


def _pdf(url: str) -> PdfDocument:
    return PdfDocument(
        url=url,
        pipeline_statuses={"pdf_process": Status.PENDING},
    )


def _params(
    chunk_iterator: Callable[[list[StageResult]], Iterator[str]],
) -> dict[str, Any]:
    return {CHUNK_ITERATOR_PARAM: chunk_iterator}


T = TypeVar("T")
RunAsync = Callable[[Coroutine[Any, Any, T]], T]


def test_country_detect_registration() -> None:
    assert isinstance(get_stage(COUNTRY_DETECT_STAGE_NAME), CountryDetectStage)
    assert get_stage_result_class(COUNTRY_DETECT_STAGE_NAME) is CountryDetectStageResult


def test_pdf_process_pipeline_includes_country_detect_after_extract() -> None:
    steps_field = PdfProcessPipeline.model_fields["steps"]
    assert steps_field.default_factory is not None
    steps = steps_field.default_factory()
    assert [step.stage_name for step in steps] == [
        PDF_EXTRACT_STAGE_NAME,
        COUNTRY_DETECT_STAGE_NAME,
        EMBED_CHUNKS_STAGE_NAME,
    ]
    assert steps[1].params[CHUNK_ITERATOR_PARAM] is extracted_pdf_chunk_iterator


def test_country_detection_xor_invariant() -> None:
    CountryDetection(countries_iso3=["KEN"], detections=["Kenya"], error=None)
    CountryDetection(countries_iso3=None, detections=None, error="boom")
    with pytest.raises(ValidationError):
        CountryDetection(countries_iso3=["KEN"], detections=["Kenya"], error="boom")
    with pytest.raises(ValidationError):
        CountryDetection(countries_iso3=None, detections=None, error=None)
    with pytest.raises(ValidationError):
        CountryDetection(countries_iso3=["KEN"], detections=["Kenya", "x"], error=None)


def _extract_result(page_paths: list[str]) -> PdfExtractStageResult:
    return PdfExtractStageResult(
        version_id="extract-v1",
        status=Status.COMPLETED,
        title="Sample",
        num_pages=len(page_paths),
        page_paths=page_paths,
    )


def test_extracted_pdf_chunk_iterator_reads_page_paths(
    tmp_path: Path,
    document_store: dict[str, Document],
    run_async: RunAsync[Any],
) -> None:
    del document_store
    page1 = tmp_path / "page-1.md"
    page2 = tmp_path / "page-2.md"
    page1.write_text("Farmers in Kenya.", encoding="utf-8")
    page2.write_text("Markets in Uganda.", encoding="utf-8")

    seen: list[str] = []

    async def detect_fn(text: str, *, max_retries: int) -> tuple[list[str], list[str]]:
        del max_retries
        seen.append(text)
        if "Kenya" in text:
            return ["KEN"], ["Kenya"]
        return ["UGA"], ["Uganda"]

    doc = _pdf("https://example.com/a.pdf")
    stage = CountryDetectStage(
        detect_fn=detect_fn,
        config=CountryDetectConfig(max_agent_retries=1),
    )
    result = run_async(
        stage.run(
            doc,
            _params(extracted_pdf_chunk_iterator),
            [_extract_result([str(page1), str(page2)])],
        )
    )
    assert isinstance(result, CountryDetectStageResult)
    assert result.status == Status.COMPLETED
    assert seen == ["Farmers in Kenya.", "Markets in Uganda."]
    assert len(result.detections) == 2
    assert result.detections[0].countries_iso3 == ["KEN"]
    assert result.detections[0].detections == ["Kenya"]
    assert result.detections[0].error is None
    assert result.detections[1].countries_iso3 == ["UGA"]
    assert result.detections[1].detections == ["Uganda"]


def test_result_length_matches_chunk_iterator(
    document_store: dict[str, Document],
    run_async: RunAsync[Any],
) -> None:
    del document_store

    def chunk_iterator(_prev: list[StageResult]) -> Iterator[str]:
        yield "Kenya"
        yield "Uganda"
        yield "no countries here"

    async def detect_fn(text: str, *, max_retries: int) -> tuple[list[str], list[str]]:
        del max_retries
        if text == "Kenya":
            return ["KEN"], ["Kenya"]
        if text == "Uganda":
            return ["UGA"], ["Uganda"]
        return [], []

    doc = _pdf("https://example.com/b.pdf")
    stage = CountryDetectStage(
        detect_fn=detect_fn,
        config=CountryDetectConfig(),
    )
    result = run_async(stage.run(doc, _params(chunk_iterator), []))
    assert isinstance(result, CountryDetectStageResult)
    assert result.status == Status.COMPLETED
    assert len(result.detections) == 3
    assert result.detections[2].countries_iso3 == []
    assert result.detections[2].detections == []
    assert result.detections[2].error is None


def test_per_chunk_error_shape(
    document_store: dict[str, Document],
    run_async: RunAsync[Any],
) -> None:
    del document_store

    def chunk_iterator(_prev: list[StageResult]) -> Iterator[str]:
        yield "ok"
        yield "bad"

    async def detect_fn(text: str, *, max_retries: int) -> tuple[list[str], list[str]]:
        del max_retries
        if text == "bad":
            raise RuntimeError("model failed")
        return ["KEN"], ["Kenya"]

    doc = _pdf("https://example.com/c.pdf")
    stage = CountryDetectStage(
        detect_fn=detect_fn,
        config=CountryDetectConfig(),
    )
    result = run_async(stage.run(doc, _params(chunk_iterator), []))
    assert isinstance(result, CountryDetectStageResult)
    assert result.status == Status.COMPLETED
    assert len(result.detections) == 2
    assert result.detections[0].error is None
    assert result.detections[0].countries_iso3 == ["KEN"]
    assert result.detections[1].countries_iso3 is None
    assert result.detections[1].detections is None
    assert result.detections[1].error == "model failed"


def test_missing_chunk_iterator_param_raises(
    document_store: dict[str, Document],
    run_async: RunAsync[Any],
) -> None:
    del document_store

    async def detect_fn(text: str, *, max_retries: int) -> tuple[list[str], list[str]]:
        raise AssertionError(f"should not detect: {text!r} retries={max_retries}")

    doc = _pdf("https://example.com/d.pdf")
    stage = CountryDetectStage(
        detect_fn=detect_fn,
        config=CountryDetectConfig(),
    )
    with pytest.raises(ValueError, match=CHUNK_ITERATOR_PARAM):
        run_async(stage.run(doc, {}, []))


def test_missing_extract_returns_failed(
    document_store: dict[str, Document],
    run_async: RunAsync[Any],
) -> None:
    del document_store

    async def detect_fn(text: str, *, max_retries: int) -> tuple[list[str], list[str]]:
        raise AssertionError(f"should not detect: {text!r} retries={max_retries}")

    doc = _pdf("https://example.com/e.pdf")
    stage = CountryDetectStage(
        detect_fn=detect_fn,
        config=CountryDetectConfig(),
    )
    result = run_async(stage.run(doc, _params(extracted_pdf_chunk_iterator), []))
    assert isinstance(result, CountryDetectStageResult)
    assert result.status == Status.FAILED
    assert result.error is not None
    assert PDF_EXTRACT_STAGE_NAME in result.error
    assert result.detections == []


def test_failed_extract_returns_failed(
    document_store: dict[str, Document],
    run_async: RunAsync[Any],
) -> None:
    del document_store
    doc = _pdf("https://example.com/f.pdf")
    prev = PdfExtractStageResult(
        version_id="extract-v1",
        status=Status.FAILED,
        error="docling exploded",
    )
    stage = CountryDetectStage(config=CountryDetectConfig())
    result = run_async(stage.run(doc, _params(extracted_pdf_chunk_iterator), [prev]))
    assert isinstance(result, CountryDetectStageResult)
    assert result.status == Status.FAILED
    assert result.error is not None
    assert PDF_EXTRACT_STAGE_NAME in result.error


def test_missing_page_file_returns_failed(
    tmp_path: Path,
    document_store: dict[str, Document],
    run_async: RunAsync[Any],
) -> None:
    del document_store
    missing = tmp_path / "missing.md"
    doc = _pdf("https://example.com/g.pdf")
    stage = CountryDetectStage(config=CountryDetectConfig())
    result = run_async(
        stage.run(
            doc,
            _params(extracted_pdf_chunk_iterator),
            [_extract_result([str(missing)])],
        )
    )
    assert isinstance(result, CountryDetectStageResult)
    assert result.status == Status.FAILED
    assert result.error is not None
    assert "not found" in result.error
