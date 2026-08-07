"""Atomic $push on Run.task_ids and Task.child_task_ids."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

from beanie import PydanticObjectId

from fao_impact_monitor.hydra import Run, Status, Task, Workflow, WorkflowNode


def test_concurrent_run_task_id_pushes(
    hydra_db: None,
    run_async: Callable[[Coroutine[Any, Any, Any]], Any],
) -> None:
    async def _test() -> None:
        wf = Workflow(
            name="push_wf",
            nodes=[WorkflowNode(name="n1", stage_name="noop")],
            entrypoints=["n1"],
        )
        await wf.insert()
        assert wf.id is not None
        run = Run(workflow_id=wf.id, status=Status.SCHEDULED)
        await run.insert()
        assert run.id is not None

        ids = [PydanticObjectId() for _ in range(40)]

        async def push_half(chunk: list[PydanticObjectId]) -> None:
            for tid in chunk:
                r = await Run.get(run.id)
                assert r is not None
                await r.push_task_id(tid)

        mid = len(ids) // 2
        await asyncio.gather(push_half(ids[:mid]), push_half(ids[mid:]))

        final = await Run.get(run.id)
        assert final is not None
        assert set(final.task_ids) == set(ids)
        assert len(final.task_ids) == len(ids)

    run_async(_test())


def test_concurrent_child_task_id_pushes(
    hydra_db: None,
    run_async: Callable[[Coroutine[Any, Any, Any]], Any],
) -> None:
    async def _test() -> None:
        parent = Task(status=Status.RUNNING, stage_name="noop", url="http://p")
        await parent.insert()
        assert parent.id is not None
        ids = [PydanticObjectId() for _ in range(40)]

        async def push_half(chunk: list[PydanticObjectId]) -> None:
            for cid in chunk:
                t = await Task.get(parent.id)
                assert t is not None
                await t.push_child_task_id(cid)

        mid = len(ids) // 2
        await asyncio.gather(push_half(ids[:mid]), push_half(ids[mid:]))

        final = await Task.get(parent.id)
        assert final is not None
        assert set(final.child_task_ids) == set(ids)
        assert len(final.child_task_ids) == len(ids)

    run_async(_test())
