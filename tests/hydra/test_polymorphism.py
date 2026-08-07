"""Polymorphic restore of WorkflowBranch and StageResult from Mongo."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

from fao_impact_monitor.hydra import Status, Workflow, WorkflowNode
from tests.hydra.conftest import (
    AlwaysNextBranch,
    CounterDocument,
    FanOutBranch,
    IncrementStageResult,
)


def test_workflow_branch_polymorphic_restore(
    hydra_db: None,
    run_async: Callable[[Coroutine[Any, Any, Any]], Any],
) -> None:
    async def _test() -> None:
        wf = Workflow(
            name="poly_wf",
            nodes=[
                WorkflowNode(
                    name="a",
                    stage_name="noop",
                    branches=[
                        AlwaysNextBranch(next_node_names=["b"]),
                        FanOutBranch(next_node_names=["b"]),
                    ],
                ),
                WorkflowNode(name="b", stage_name="noop"),
            ],
            entrypoints=["a"],
        )
        await wf.insert()
        loaded = await Workflow.get(wf.id)
        assert loaded is not None
        branches = loaded.get_node("a").branches
        assert len(branches) == 2
        assert type(branches[0]) is AlwaysNextBranch
        assert type(branches[1]) is FanOutBranch
        assert hasattr(branches[0], "route")

    run_async(_test())


def test_stage_result_polymorphic_restore(
    hydra_db: None,
    run_async: Callable[[Coroutine[Any, Any, Any]], Any],
) -> None:
    async def _test() -> None:
        doc = CounterDocument(url="http://poly", source="t")
        await doc.insert()
        result = IncrementStageResult(
            name="increment",
            status=Status.COMPLETED,
            counter=7,
        )
        await doc.push_stage_result("poly_wf", "a", result)
        loaded = await CounterDocument.get(doc.id)
        assert loaded is not None
        latest = loaded.latest_stage_result("poly_wf", "a")
        assert latest is not None
        assert type(latest) is IncrementStageResult
        assert latest.counter == 7

    run_async(_test())
