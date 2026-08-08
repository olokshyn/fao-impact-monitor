"""Workflow.submit creates Run and schedules entrypoint Tasks."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

import pytest

from fao_impact_monitor.hydra import Status, Task, Workflow, WorkflowNode
from tests.hydra.conftest import AlwaysNextBranch


def test_submit_single_entrypoint(
    hydra_db: None,
    run_async: Callable[[Coroutine[Any, Any, Any]], Any],
) -> None:
    async def _test() -> None:
        wf = Workflow(
            name="submit_one",
            nodes=[WorkflowNode(name="start", stage_name="noop")],
            entrypoints=["start"],
        )
        await wf.insert()
        task = Task(status=Status.CREATED, url="http://s", source="src")
        run = await wf.submit(task)
        assert run.workflow_id == wf.id
        assert run.status == Status.SCHEDULED
        assert len(run.task_ids) == 1
        scheduled = await Task.get(run.task_ids[0])
        assert scheduled is not None
        assert scheduled.status == Status.SCHEDULED
        assert scheduled.workflow_id == wf.id
        assert scheduled.run_id == run.id
        assert scheduled.workflow_node_name == "start"
        assert scheduled.stage_name == "noop"
        assert scheduled.parent_task_id is None

    run_async(_test())


def test_submit_multi_entrypoint_copies_task(
    hydra_db: None,
    run_async: Callable[[Coroutine[Any, Any, Any]], Any],
) -> None:
    async def _test() -> None:
        wf = Workflow(
            name="submit_multi",
            nodes=[
                WorkflowNode(name="a", stage_name="noop"),
                WorkflowNode(name="b", stage_name="increment"),
            ],
            entrypoints=["a", "b"],
        )
        await wf.insert()
        task = Task(status=Status.CREATED, url="http://m", source="src")
        run = await wf.submit(task)
        assert len(run.task_ids) == 2
        tasks = [await Task.get(tid) for tid in run.task_ids]
        assert all(t is not None for t in tasks)
        names = {t.workflow_node_name for t in tasks if t is not None}
        assert names == {"a", "b"}
        ids = {t.id for t in tasks if t is not None}
        assert len(ids) == 2

    run_async(_test())


def test_submit_rejects_non_created(
    hydra_db: None,
    run_async: Callable[[Coroutine[Any, Any, Any]], Any],
) -> None:
    async def _test() -> None:
        wf = Workflow(
            name="submit_bad",
            nodes=[WorkflowNode(name="a", stage_name="noop")],
            entrypoints=["a"],
        )
        await wf.insert()
        task = Task(status=Status.SCHEDULED, url="http://x")
        with pytest.raises(ValueError, match="CREATED"):
            await wf.submit(task)

    run_async(_test())


def test_submit_rejects_missing_context_for_non_entrypoint_stage(
    hydra_db: None,
    run_async: Callable[[Coroutine[Any, Any, Any]], Any],
) -> None:
    """Missing keys required by a downstream stage still fail submit."""

    async def _test() -> None:
        from fao_impact_monitor.hydra.run import Run

        wf = Workflow(
            name="submit_ctx_missing",
            nodes=[
                WorkflowNode(
                    name="root",
                    stage_name="noop",
                    branches=[AlwaysNextBranch(next_node_names=["later"])],
                ),
                WorkflowNode(name="later", stage_name="requires_query"),
            ],
            entrypoints=["root"],
        )
        await wf.insert()
        task = Task(status=Status.CREATED, url="http://ctx-missing")
        with pytest.raises(ValueError, match="missing required keys") as exc_info:
            await wf.submit(task)
        assert "query" in str(exc_info.value)
        assert "requires_query" in str(exc_info.value)
        assert "later" in str(exc_info.value)
        assert await Run.find_all().count() == 0
        assert await Task.find_all().count() == 0

        ok = await wf.submit(
            Task(
                status=Status.CREATED,
                url="http://ctx-ok",
                context={"query": "el nino"},
            )
        )
        assert len(ok.task_ids) == 1

    run_async(_test())


def test_branch_roundtrip_on_workflow(
    hydra_db: None,
    run_async: Callable[[Coroutine[Any, Any, Any]], Any],
) -> None:
    """Ensure AlwaysNextBranch is available for other tests (registry smoke)."""

    async def _test() -> None:
        wf = Workflow(
            name="with_branch",
            nodes=[
                WorkflowNode(
                    name="a",
                    stage_name="noop",
                    branches=[
                        AlwaysNextBranch(next_node_names=["b"]),
                    ],
                ),
                WorkflowNode(name="b", stage_name="noop"),
            ],
            entrypoints=["a"],
        )
        await wf.insert()
        loaded = await Workflow.get(wf.id)
        assert loaded is not None
        assert isinstance(loaded.get_node("a").branches[0], AlwaysNextBranch)

    run_async(_test())
