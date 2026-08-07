"""WorkflowBranch routing and illegal target rejection."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

import pytest

from fao_impact_monitor.hydra import (
    Executor,
    Status,
    Task,
    Workflow,
    WorkflowNode,
)
from tests.hydra.conftest import (
    FanOutBranch,
    IllegalTargetBranch,
    run_executor_until,
)


def test_fan_out_creates_multiple_children(
    hydra_db: None,
    run_async: Callable[[Coroutine[Any, Any, Any]], Any],
) -> None:
    async def _test() -> None:
        wf = Workflow(
            name="fan",
            nodes=[
                WorkflowNode(
                    name="root",
                    stage_name="noop",
                    branches=[
                        FanOutBranch(next_node_names=["c1", "c2"]),
                    ],
                ),
                WorkflowNode(name="c1", stage_name="noop"),
                WorkflowNode(name="c2", stage_name="noop"),
            ],
            entrypoints=["root"],
        )
        await wf.insert()
        run = await wf.submit(Task(status=Status.CREATED, url="http://fan"))
        ex = Executor(
            concurrency={"noop": 4},
            claim_idle_sleep_seconds=0.01,
            heartbeat_interval_minutes=60,
        )

        async def _done() -> bool:
            tasks = await Task.find(Task.run_id == run.id).to_list()
            return len(tasks) >= 3 and all(t.status == Status.COMPLETED for t in tasks)

        await run_executor_until(ex, predicate=_done, timeout=5.0)
        tasks = await Task.find(Task.run_id == run.id).to_list()
        assert len(tasks) == 3
        root = next(t for t in tasks if t.workflow_node_name == "root")
        assert len(root.child_task_ids) == 2
        child_names = {
            t.workflow_node_name for t in tasks if t.id in root.child_task_ids
        }
        assert child_names == {"c1", "c2"}

    run_async(_test())


def test_illegal_branch_target_rejected(
    hydra_db: None,
    run_async: Callable[[Coroutine[Any, Any, Any]], Any],
) -> None:
    async def _test() -> None:
        wf = Workflow(
            name="bad_route",
            nodes=[
                WorkflowNode(
                    name="root",
                    stage_name="noop",
                    branches=[
                        IllegalTargetBranch(
                            next_node_names=["ok"],
                            params={"bad_node": "evil"},
                        ),
                    ],
                ),
                WorkflowNode(name="ok", stage_name="noop"),
                WorkflowNode(name="evil", stage_name="noop"),
            ],
            entrypoints=["root"],
        )
        await wf.insert()
        run = await wf.submit(Task(status=Status.CREATED, url="http://bad"))
        ex = Executor(
            concurrency={"noop": 1},
            claim_idle_sleep_seconds=0.01,
            heartbeat_interval_minutes=60,
        )

        # Directly exercise execute to assert the ValueError path
        task = await Task.find_one(Task.run_id == run.id)
        assert task is not None
        task.status = Status.RUNNING
        task.attempts = 1
        await task.save()
        with pytest.raises(ValueError, match="not in"):
            await ex.execute_task(task)

        tasks = await Task.find(Task.run_id == run.id).to_list()
        assert not any(t.workflow_node_name == "evil" for t in tasks)

    run_async(_test())
