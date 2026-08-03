import asyncio
from collections.abc import Iterator
from typing import Any, ClassVar, cast
from unittest.mock import AsyncMock

import pytest

from fao_impact_monitor.data_lake.common import Status
from fao_impact_monitor.data_lake.document import Document, RelationSide, RelationType
from fao_impact_monitor.data_lake.documents.web_page_document import WebPageDocument
from fao_impact_monitor.data_lake.pipeline import Pipeline, PipelineStep
from fao_impact_monitor.data_lake.stage import (
    _STAGE_REGISTRY,
    _STAGE_RESULT_REGISTRY,
    Stage,
    StageResult,
    StageVersion,
)


class MockAResult(StageResult):
    name: str = "mock_a"
    value: str = ""
    status: Status = Status.COMPLETED


class MockBResult(StageResult):
    name: str = "mock_b"
    value: str = ""
    status: Status = Status.COMPLETED


class MockFailResult(StageResult):
    name: str = "mock_fail"
    status: Status = Status.FAILED


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


class MockChildResult(StageResult):
    name: str = "mock_child"
    status: Status = Status.COMPLETED


class MockChildStage(Stage):
    name = "mock_child"
    calls: ClassVar[list[str]] = []

    async def get_version(self) -> StageVersion:
        raise NotImplementedError

    async def run(
        self,
        document: Document,
        stage_params: dict[str, Any],
        prev_stages: list[StageResult],
    ) -> StageResult:
        del stage_params, prev_stages
        MockChildStage.calls.append(document.url)
        return MockChildResult(version_id="child-v1")


@pytest.fixture(autouse=True)
def _reset_mock_stage_calls() -> Iterator[None]:
    MockAStage.calls = []
    MockBStage.calls = []
    MockFailStage.calls = []
    MockChildStage.calls = []
    yield


@pytest.fixture(autouse=True)
def _stub_mongo_lookups(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _no_doc_by_url(_url: str) -> Document | None:
        return None

    async def _no_doc_by_id(_document_id: Any, _document_type: Any) -> Document | None:
        return None

    monkeypatch.setattr(
        "fao_impact_monitor.data_lake.pipeline._find_document_by_url",
        _no_doc_by_url,
    )
    monkeypatch.setattr(
        "fao_impact_monitor.data_lake.pipeline._find_document_by_id",
        _no_doc_by_id,
    )


def _make_document(
    stage_results: dict[str, list[StageResult]] | None = None,
    *,
    pipeline_statuses: dict[str, Status] | None = None,
) -> tuple[WebPageDocument, AsyncMock]:
    doc = WebPageDocument.model_construct(
        title="Example",
        url="https://example.com/doc",
        pipeline_statuses=pipeline_statuses if pipeline_statuses is not None else {},
        stage_results=stage_results if stage_results is not None else {},
        relations=[],
    )
    save = AsyncMock()
    object.__setattr__(doc, "save", save)
    return doc, save


def _make_pipeline(*steps: PipelineStep, name: str = "test-pipeline") -> Pipeline:
    return cast(
        Pipeline,
        Pipeline.model_construct(name=name, steps=list(steps)),
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
                    status=Status.FAILED,
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
    assert document.pipeline_status("test-pipeline") == Status.COMPLETED
    assert save.await_count >= 2
    assert pipeline.is_completed(document) is True

    assert len(MockAStage.calls) == 1
    assert MockAStage.calls[0][0] == {"tag": "first"}
    assert MockAStage.calls[0][1] == []

    assert len(MockBStage.calls) == 1
    assert MockBStage.calls[0][0] == {"tag": "second"}
    assert len(MockBStage.calls[0][1]) == 1
    assert MockBStage.calls[0][1][0].name == "mock_a"


def test_run_skips_when_pipeline_completed(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = _make_pipeline(PipelineStep(stage_name="mock_a", params={"tag": "x"}))
    document, save = _make_document(
        pipeline_statuses={"test-pipeline": Status.COMPLETED}
    )

    async def _find(url: str) -> Document | None:
        assert url == document.url
        return document

    monkeypatch.setattr(
        "fao_impact_monitor.data_lake.pipeline._find_document_by_url",
        _find,
    )

    asyncio.run(pipeline.run(document))

    assert MockAStage.calls == []
    assert save.await_count == 0


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
    assert document.pipeline_status("test-pipeline") == Status.COMPLETED
    assert save.await_count >= 1


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
                    status=Status.FAILED,
                    error="boom",
                )
            ]
        }
    )

    asyncio.run(pipeline.run(document))

    assert len(MockAStage.calls) == 1
    assert len(document.stage_results["mock_a"]) == 2
    latest = cast(MockAResult, document.stage_results["mock_a"][-1])
    assert latest.status == Status.COMPLETED
    assert latest.value == "retry"
    assert document.pipeline_status("test-pipeline") == Status.COMPLETED


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
    assert document.stage_results["mock_fail"][-1].status == Status.FAILED
    assert (
        cast(MockBResult, document.stage_results["mock_b"][-1]).value == ":after-fail"
    )
    assert pipeline.is_completed(document) is False
    assert document.pipeline_status("test-pipeline") == Status.FAILED


def test_run_cascades_to_child_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    parent_pipeline = _make_pipeline(
        PipelineStep(stage_name="mock_a", params={"tag": "parent"}),
        name="parent-pipeline",
    )
    child_pipeline = _make_pipeline(
        PipelineStep(stage_name="mock_child", params={}),
        name="child-pipeline",
    )

    def _get_pipeline(name: str) -> Pipeline:
        assert name == "child-pipeline"
        return child_pipeline

    monkeypatch.setattr(
        "fao_impact_monitor.data_lake.pipeline.get_pipeline",
        _get_pipeline,
    )

    parent, parent_save = _make_document()
    child, child_save = _make_document(
        pipeline_statuses={"child-pipeline": Status.PENDING},
    )
    child.url = "https://example.com/child"
    from beanie import PydanticObjectId

    from fao_impact_monitor.data_lake.document import DocumentType, Relation

    child_id = PydanticObjectId()
    object.__setattr__(child, "id", child_id)
    parent.relations = [
        Relation(
            type=RelationType.URL_LINK,
            side=RelationSide.TO,
            d_id=child_id,
            d_type=DocumentType.WEB_PAGE,
        )
    ]

    async def _find_url(url: str) -> Document | None:
        if url == parent.url:
            return parent
        if url == child.url:
            return child
        return None

    async def _find_id(document_id: Any, _document_type: Any) -> Document | None:
        if document_id == child_id:
            return child
        return None

    monkeypatch.setattr(
        "fao_impact_monitor.data_lake.pipeline._find_document_by_url",
        _find_url,
    )
    monkeypatch.setattr(
        "fao_impact_monitor.data_lake.pipeline._find_document_by_id",
        _find_id,
    )

    asyncio.run(parent_pipeline.run(parent))

    assert MockAStage.calls
    assert MockChildStage.calls == [child.url]
    assert parent.pipeline_status("parent-pipeline") == Status.COMPLETED
    assert child.pipeline_status("child-pipeline") == Status.COMPLETED
    assert parent_save.await_count >= 1
    assert child_save.await_count >= 1
