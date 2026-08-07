"""Kill executor mid-flight, sweep stale claims, reclaim with a new executor."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from datetime import UTC, datetime, timedelta
from typing import Any

from fao_impact_monitor.hydra import (
    Executor,
    Status,
    Task,
    Workflow,
    WorkflowNode,
    sweep_stale_executors,
)
from fao_impact_monitor.hydra.executor import ExecutorHeartbeat
from tests.hydra.conftest import HangStage, run_executor_until


def test_sweep_reclaims_interrupted_running_task(
    hydra_db: None,
    run_async: Callable[[Coroutine[Any, Any, Any]], Any],
) -> None:
    async def _test() -> None:
        gate = asyncio.Event()
        HangStage.gate = gate

        wf = Workflow(
            name="sweep_wf",
            nodes=[WorkflowNode(name="hang", stage_name="hang")],
            entrypoints=["hang"],
        )
        await wf.insert()
        run = await wf.submit(Task(status=Status.CREATED, url="http://hang"))

        ex1 = Executor(
            concurrency={"hang": 1},
            claim_idle_sleep_seconds=0.01,
            heartbeat_interval_minutes=1.0,
            stale_multiplier=3.0,
            max_attempts=3,
            executor_id="victim-executor",
        )

        main = asyncio.create_task(ex1.run(), name="victim-run")

        # Wait until the task is RUNNING
        deadline = asyncio.get_event_loop().time() + 5.0
        task: Task | None = None
        while asyncio.get_event_loop().time() < deadline:
            task = await Task.find_one(Task.run_id == run.id)
            if task is not None and task.status == Status.RUNNING:
                break
            await asyncio.sleep(0.01)
        assert task is not None
        assert task.status == Status.RUNNING
        assert task.executor_id == "victim-executor"
        assert task.attempts == 1

        # Hard-kill: cancel without graceful stop/drain
        main.cancel()
        try:
            await main
        except asyncio.CancelledError:
            pass
        # Release any leftover HangStage waiters from the cancelled task.
        gate.set()

        # Age the heartbeat so the sweeper treats the executor as dead
        hb = await ExecutorHeartbeat.get("victim-executor")
        assert hb is not None
        old = datetime.now(UTC) - timedelta(minutes=10)
        await hb.update({"$set": {"updated_at": old}})

        swept = await sweep_stale_executors(
            heartbeat_interval_minutes=1.0,
            stale_multiplier=3.0,
        )
        assert "victim-executor" in swept

        refreshed = await Task.get(task.id)
        assert refreshed is not None
        assert refreshed.status == Status.SCHEDULED
        assert refreshed.attempts == 0  # claim attempt refunded
        assert refreshed.executor_id is None

        # Second executor completes the work
        HangStage.gate = asyncio.Event()
        HangStage.gate.set()  # don't hang this time — or use noop

        # Replace hang with immediate success by setting gate already set
        ex2 = Executor(
            concurrency={"hang": 1},
            claim_idle_sleep_seconds=0.01,
            heartbeat_interval_minutes=60,
            max_attempts=3,
            executor_id="rescuer",
        )

        async def _done() -> bool:
            t = await Task.get(task.id)
            return t is not None and t.status == Status.COMPLETED

        await run_executor_until(ex2, predicate=_done, timeout=5.0)
        final = await Task.get(task.id)
        assert final is not None
        assert final.status == Status.COMPLETED
        assert final.executor_id == "rescuer"
        assert final.attempts == 1

    run_async(_test())
