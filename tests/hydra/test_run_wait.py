"""Run.wait aggregate status rules."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

from fao_impact_monitor.hydra import Status, Task, Workflow, WorkflowNode


def test_wait_completed_when_entrypoint_ok_even_if_child_failed(
    hydra_db: None,
    run_async: Callable[[Coroutine[Any, Any, Any]], Any],
) -> None:
    async def _test() -> None:
        wf = Workflow(
            name="wait_ok",
            nodes=[
                WorkflowNode(name="entry", stage_name="noop"),
                WorkflowNode(name="child", stage_name="noop"),
            ],
            entrypoints=["entry"],
        )
        await wf.insert()
        run = await wf.submit(Task(status=Status.CREATED, url="http://w1"))
        entry = await Task.get(run.task_ids[0])
        assert entry is not None
        await entry.set_fields(status=Status.COMPLETED)

        child = Task(
            status=Status.FAILED,
            run_id=run.id,
            workflow_id=wf.id,
            workflow_node_name="child",
            stage_name="noop",
            parent_task_id=entry.id,
            url="http://w1",
            error="child failed",
        )
        await child.insert()
        assert child.id is not None
        await run.push_task_id(child.id)
        await entry.push_child_task_id(child.id)

        await run.wait(poll_interval_seconds=0.01)
        assert run.status == Status.COMPLETED

    run_async(_test())


def test_wait_failed_when_entrypoint_failed(
    hydra_db: None,
    run_async: Callable[[Coroutine[Any, Any, Any]], Any],
) -> None:
    async def _test() -> None:
        wf = Workflow(
            name="wait_fail",
            nodes=[WorkflowNode(name="entry", stage_name="noop")],
            entrypoints=["entry"],
        )
        await wf.insert()
        run = await wf.submit(Task(status=Status.CREATED, url="http://w2"))
        entry = await Task.get(run.task_ids[0])
        assert entry is not None
        await entry.set_fields(status=Status.FAILED, error="boom")
        await run.wait(poll_interval_seconds=0.01)
        assert run.status == Status.FAILED

    run_async(_test())
