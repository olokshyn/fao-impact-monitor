import asyncio
from collections.abc import Iterator
from typing import Any, ClassVar, cast
from unittest.mock import AsyncMock

import pytest

from fao_impact_monitor.data_lake.document import Document
from fao_impact_monitor.data_lake.documents.web_page_document import WebPageDocument
from fao_impact_monitor.data_lake.pipeline import Pipeline, PipelineStep
from fao_impact_monitor.data_lake.stage import (
    _STAGE_REGISTRY,
    _STAGE_RESULT_REGISTRY,
    Stage,
    StageResult,
    StageStatus,
    StageVersion,
)


class MockAResult(StageResult):
    name: str = "mock_a"
    value: str = ""
    result_status: StageStatus = StageStatus.COMPLETED

    @property
    def status(self) -> StageStatus:
        return self.result_status


class MockBResult(StageResult):
    name: str = "mock_b"
    value: str = ""
    result_status: StageStatus = StageStatus.COMPLETED

    @property
    def status(self) -> StageStatus:
        return self.result_status


class MockFailResult(StageResult):
    name: str = "mock_fail"
    result_status: StageStatus = StageStatus.FAILED

    @property
    def status(self) -> StageStatus:
        return self.result_status


class MockAStage(Stage):
    name = "mock_a"
    calls: ClassVar[list[tuple[dict[str, Any], list[StageResult]]]] = []

    async def get_version(self) -> StageVersion:
        raise NotImplementedError

    async def run(
        self,
        document: Document,
        stage_params: dict[str, Any],
        prev_stages: list[StageResult],
    ) -> StageResult:
        MockAStage.calls.append((stage_params, list(prev_stages)))
        return MockAResult(version_id="a-v1", value=str(stage_params.get("tag", "")))


class MockBStage(Stage):
    name = "mock_b"
    calls: ClassVar[list[tuple[dict[str, Any], list[StageResult]]]] = []

    async def get_version(self) -> StageVersion:
        raise NotImplementedError

    async def run(
        self,
        document: Document,
        stage_params: dict[str, Any],
        prev_stages: list[StageResult],
    ) -> StageResult:
        MockBStage.calls.append((stage_params, list(prev_stages)))
        prev = prev_stages[-1] if prev_stages else None
        prev_value = prev.value if isinstance(prev, MockAResult) else ""
        return MockBResult(
            version_id="b-v1",
            value=f"{prev_value}:{stage_params.get('tag', '')}",
        )


class MockFailStage(Stage):
    name = "mock_fail"
    calls: ClassVar[list[tuple[dict[str, Any], list[StageResult]]]] = []

    async def get_version(self) -> StageVersion:
        raise NotImplementedError

    async def run(
        self,
        document: Document,
        stage_params: dict[str, Any],
        prev_stages: list[StageResult],
    ) -> StageResult:
        MockFailStage.calls.append((stage_params, list(prev_stages)))
        return MockFailResult(version_id="fail-v1", error="stage failed")


@pytest.fixture(autouse=True)
def _reset_mock_stage_calls() -> Iterator[None]:
    MockAStage.calls = []
    MockBStage.calls = []
    MockFailStage.calls = []
    yield


def _make_document(
    stage_results: dict[str, list[StageResult]] | None = None,
) -> tuple[WebPageDocument, AsyncMock]:
    doc = WebPageDocument.model_construct(
        title="Example",
        url="https://example.com/doc",
        pipeline_name="test-pipeline",
        stage_results=stage_results if stage_results is not None else {},
        relations=[],
    )
    save = AsyncMock()
    object.__setattr__(doc, "save", save)
    return doc, save


def _make_pipeline(*steps: PipelineStep) -> Pipeline:
    return cast(
        Pipeline,
        Pipeline.model_construct(name="test-pipeline", steps=list(steps)),
    )


def test_metaclass_registers_mock_stages() -> None:
    assert _STAGE_REGISTRY["mock_a"] is MockAStage
    assert _STAGE_REGISTRY["mock_b"] is MockBStage
    assert _STAGE_REGISTRY["mock_fail"] is MockFailStage
    assert _STAGE_RESULT_REGISTRY["mock_a"] is MockAResult
    assert _STAGE_RESULT_REGISTRY["mock_b"] is MockBResult
    assert _STAGE_RESULT_REGISTRY["mock_fail"] is MockFailResult


def test_is_completed_empty_pipeline() -> None:
    pipeline = _make_pipeline()
    document, _ = _make_document()

    assert pipeline.is_completed(document) is True


def test_is_completed_missing_stage_result() -> None:
    pipeline = _make_pipeline(PipelineStep(stage_name="mock_a", params={}))
    document, _ = _make_document()

    assert pipeline.is_completed(document) is False


def test_is_completed_failed_stage_result() -> None:
    pipeline = _make_pipeline(PipelineStep(stage_name="mock_fail", params={}))
    document, _ = _make_document(
        {
            "mock_fail": [
                MockFailResult(version_id="fail-v1", error="stage failed"),
            ]
        }
    )

    assert pipeline.is_completed(document) is False


def test_is_completed_all_stages_completed() -> None:
    pipeline = _make_pipeline(
        PipelineStep(stage_name="mock_a", params={}),
        PipelineStep(stage_name="mock_b", params={}),
    )
    document, _ = _make_document(
        {
            "mock_a": [MockAResult(version_id="a-v1", value="x")],
            "mock_b": [MockBResult(version_id="b-v1", value="x:y")],
        }
    )

    assert pipeline.is_completed(document) is True


def test_is_completed_uses_latest_result_only() -> None:
    pipeline = _make_pipeline(PipelineStep(stage_name="mock_a", params={}))
    document, _ = _make_document(
        {
            "mock_a": [
                MockAResult(version_id="a-v1", value="ok"),
                MockAResult(
                    version_id="a-v2",
                    value="bad",
                    result_status=StageStatus.FAILED,
                    error="later failure",
                ),
            ]
        }
    )

    assert pipeline.is_completed(document) is False


def test_run_executes_stages_in_order_and_saves() -> None:
    pipeline = _make_pipeline(
        PipelineStep(stage_name="mock_a", params={"tag": "first"}),
        PipelineStep(stage_name="mock_b", params={"tag": "second"}),
    )
    document, save = _make_document()

    asyncio.run(pipeline.run(document))

    assert list(document.stage_results) == ["mock_a", "mock_b"]
    assert cast(MockAResult, document.stage_results["mock_a"][0]).value == "first"
    assert (
        cast(MockBResult, document.stage_results["mock_b"][0]).value == "first:second"
    )
    assert save.await_count == 2
    assert pipeline.is_completed(document) is True

    assert len(MockAStage.calls) == 1
    assert MockAStage.calls[0][0] == {"tag": "first"}
    assert MockAStage.calls[0][1] == []

    assert len(MockBStage.calls) == 1
    assert MockBStage.calls[0][0] == {"tag": "second"}
    assert len(MockBStage.calls[0][1]) == 1
    assert MockBStage.calls[0][1][0].name == "mock_a"


def test_run_skips_completed_stages() -> None:
    existing = MockAResult(version_id="a-v1", value="cached")
    pipeline = _make_pipeline(
        PipelineStep(stage_name="mock_a", params={"tag": "first"}),
        PipelineStep(stage_name="mock_b", params={"tag": "second"}),
    )
    document, save = _make_document({"mock_a": [existing]})

    asyncio.run(pipeline.run(document))

    assert MockAStage.calls == []
    assert len(MockBStage.calls) == 1
    assert MockBStage.calls[0][1] == [existing]
    assert document.stage_results["mock_a"] == [existing]
    assert (
        cast(MockBResult, document.stage_results["mock_b"][0]).value == "cached:second"
    )
    assert save.await_count == 1


def test_run_retries_failed_stage() -> None:
    pipeline = _make_pipeline(
        PipelineStep(stage_name="mock_a", params={"tag": "retry"})
    )
    document, _ = _make_document(
        {
            "mock_a": [
                MockAResult(
                    version_id="a-v0",
                    value="old",
                    result_status=StageStatus.FAILED,
                    error="boom",
                )
            ]
        }
    )

    asyncio.run(pipeline.run(document))

    assert len(MockAStage.calls) == 1
    assert len(document.stage_results["mock_a"]) == 2
    latest = cast(MockAResult, document.stage_results["mock_a"][-1])
    assert latest.status == StageStatus.COMPLETED
    assert latest.value == "retry"


def test_run_continues_after_failed_stage_with_truncated_prev_results() -> None:
    pipeline = _make_pipeline(
        PipelineStep(stage_name="mock_fail", params={}),
        PipelineStep(stage_name="mock_b", params={"tag": "after-fail"}),
    )
    document, _ = _make_document()

    asyncio.run(pipeline.run(document))

    assert len(MockFailStage.calls) == 1
    assert len(MockBStage.calls) == 1
    assert MockBStage.calls[0][1] == []
    assert document.stage_results["mock_fail"][-1].status == StageStatus.FAILED
    assert (
        cast(MockBResult, document.stage_results["mock_b"][-1]).value == ":after-fail"
    )
    assert pipeline.is_completed(document) is False
