"""Stage failure retries up to max_attempts then FAILED."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

from fao_impact_monitor.hydra import (
    Document,
    Executor,
    Status,
    Task,
    Workflow,
    WorkflowNode,
)
from tests.hydra.conftest import (
    AlwaysNextBranch,
    FailNTimesStage,
    run_executor_until,
)


def test_failing_task_retried_then_failed(
    hydra_db: None,
    run_async: Callable[[Coroutine[Any, Any, Any]], Any],
) -> None:
    async def _test() -> None:
        FailNTimesStage.reset("r1", n=5)  # always fail within max_attempts=3
        wf = Workflow(
            name="retry_wf",
            nodes=[
                WorkflowNode(
                    name="n1",
                    stage_name="fail_n_times",
                    stage_params={"key": "r1"},
                )
            ],
            entrypoints=["n1"],
        )
        await wf.insert()
        task = Task(status=Status.CREATED, url="http://retry")
        run = await wf.submit(task)
        ex = Executor(
            concurrency={"fail_n_times": 1},
            max_attempts=3,
            claim_idle_sleep_seconds=0.01,
            heartbeat_interval_minutes=60,
        )

        async def _done() -> bool:
            t = await Task.find_one(Task.run_id == run.id)
            return t is not None and t.status == Status.FAILED

        await run_executor_until(ex, predicate=_done, timeout=5.0)
        final = await Task.find_one(Task.run_id == run.id)
        assert final is not None
        assert final.status == Status.FAILED
        assert final.attempts == 3

    run_async(_test())


def test_stage_raise_marks_task_failed_without_retry_or_document(
    hydra_db: None,
    run_async: Callable[[Coroutine[Any, Any, Any]], Any],
) -> None:
    """Stage.process exceptions → Task FAILED; no RETRYING; no Document write."""

    async def _test() -> None:
        wf = Workflow(
            name="raise_wf",
            nodes=[
                WorkflowNode(
                    name="n1",
                    stage_name="raise_prerequisite",
                    stage_params={"message": "missing prior fetch"},
                    branches=[AlwaysNextBranch(next_node_names=["n2"])],
                ),
                WorkflowNode(name="n2", stage_name="noop"),
            ],
            entrypoints=["n1"],
        )
        await wf.insert()
        task = Task(status=Status.CREATED, url="http://raise-prereq")
        run = await wf.submit(task)
        ex = Executor(
            concurrency={"raise_prerequisite": 1, "noop": 1},
            max_attempts=3,
            claim_idle_sleep_seconds=0.01,
            heartbeat_interval_minutes=60,
        )

        async def _done() -> bool:
            t = await Task.find_one(
                Task.run_id == run.id,
                Task.workflow_node_name == "n1",
            )
            return t is not None and t.status == Status.FAILED

        await run_executor_until(ex, predicate=_done, timeout=5.0)
        final = await Task.find_one(
            Task.run_id == run.id,
            Task.workflow_node_name == "n1",
        )
        assert final is not None
        assert final.status == Status.FAILED
        assert final.error == "missing prior fetch"
        assert final.attempts == 1
        assert final.stage_result is None

        children = await Task.find(
            Task.run_id == run.id,
            Task.workflow_node_name == "n2",
        ).to_list()
        assert children == []

        docs = await Document.find(Document.url == "http://raise-prereq").to_list()
        assert docs == []

    run_async(_test())


def test_failing_task_succeeds_after_retries(
    hydra_db: None,
    run_async: Callable[[Coroutine[Any, Any, Any]], Any],
) -> None:
    async def _test() -> None:
        FailNTimesStage.reset("r2", n=2)  # fail twice, succeed on 3rd claim
        wf = Workflow(
            name="retry_ok_wf",
            nodes=[
                WorkflowNode(
                    name="n1",
                    stage_name="fail_n_times",
                    stage_params={"key": "r2"},
                )
            ],
            entrypoints=["n1"],
        )
        await wf.insert()
        task = Task(status=Status.CREATED, url="http://retry-ok")
        run = await wf.submit(task)
        ex = Executor(
            concurrency={"fail_n_times": 1},
            max_attempts=3,
            claim_idle_sleep_seconds=0.01,
            heartbeat_interval_minutes=60,
        )

        async def _done() -> bool:
            t = await Task.find_one(Task.run_id == run.id)
            return t is not None and t.status == Status.COMPLETED

        await run_executor_until(ex, predicate=_done, timeout=5.0)
        final = await Task.find_one(Task.run_id == run.id)
        assert final is not None
        assert final.status == Status.COMPLETED
        assert final.attempts == 3

    run_async(_test())
