"""Atomic task claiming via Executor.claim_task."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

from fao_impact_monitor.hydra import Executor, Status, Task


def test_claim_scheduled_to_running(
    hydra_db: None,
    run_async: Callable[[Coroutine[Any, Any, Any]], Any],
) -> None:
    async def _test() -> None:
        task = Task(status=Status.SCHEDULED, stage_name="noop", url="http://a")
        await task.insert()
        ex = Executor(concurrency={"noop": 1}, max_attempts=3)
        claimed = await ex.claim_task("noop")
        assert claimed is not None
        assert claimed.id == task.id
        assert claimed.status == Status.RUNNING
        assert claimed.executor_id == ex.id
        assert claimed.attempts == 1
        assert claimed.executor_started_at is not None

    run_async(_test())


def test_concurrent_claimers_one_wins(
    hydra_db: None,
    run_async: Callable[[Coroutine[Any, Any, Any]], Any],
) -> None:
    async def _test() -> None:
        task = Task(status=Status.SCHEDULED, stage_name="noop", url="http://a")
        await task.insert()
        ex1 = Executor(concurrency={"noop": 1}, executor_id="ex-1")
        ex2 = Executor(concurrency={"noop": 1}, executor_id="ex-2")
        c1, c2 = await asyncio.gather(ex1.claim_task("noop"), ex2.claim_task("noop"))
        winners = [c for c in (c1, c2) if c is not None]
        losers = [c for c in (c1, c2) if c is None]
        assert len(winners) == 1
        assert len(losers) == 1
        assert winners[0].attempts == 1
        refreshed = await Task.get(task.id)
        assert refreshed is not None
        assert refreshed.status == Status.RUNNING
        assert refreshed.executor_id in {"ex-1", "ex-2"}

    run_async(_test())


def test_claim_retrying_under_max_attempts(
    hydra_db: None,
    run_async: Callable[[Coroutine[Any, Any, Any]], Any],
) -> None:
    async def _test() -> None:
        task = Task(
            status=Status.RETRYING,
            stage_name="noop",
            attempts=1,
            url="http://a",
        )
        await task.insert()
        ex = Executor(concurrency={"noop": 1}, max_attempts=3)
        claimed = await ex.claim_task("noop")
        assert claimed is not None
        assert claimed.status == Status.RUNNING
        assert claimed.attempts == 2

    run_async(_test())


def test_claim_retrying_at_max_attempts_skipped(
    hydra_db: None,
    run_async: Callable[[Coroutine[Any, Any, Any]], Any],
) -> None:
    async def _test() -> None:
        task = Task(
            status=Status.RETRYING,
            stage_name="noop",
            attempts=3,
            url="http://a",
        )
        await task.insert()
        ex = Executor(concurrency={"noop": 1}, max_attempts=3)
        claimed = await ex.claim_task("noop")
        assert claimed is None
        refreshed = await Task.get(task.id)
        assert refreshed is not None
        assert refreshed.status == Status.RETRYING
        assert refreshed.attempts == 3

    run_async(_test())


def test_claim_ignores_other_stage(
    hydra_db: None,
    run_async: Callable[[Coroutine[Any, Any, Any]], Any],
) -> None:
    async def _test() -> None:
        await Task(
            status=Status.SCHEDULED, stage_name="increment", url="http://a"
        ).insert()
        ex = Executor(concurrency={"noop": 1})
        assert await ex.claim_task("noop") is None

    run_async(_test())


def test_claim_prefers_oldest_updated_at(
    hydra_db: None,
    run_async: Callable[[Coroutine[Any, Any, Any]], Any],
) -> None:
    async def _test() -> None:
        from datetime import UTC, datetime, timedelta

        older = datetime.now(UTC) - timedelta(minutes=10)
        newer = datetime.now(UTC) - timedelta(minutes=1)
        t_new = Task(
            status=Status.SCHEDULED,
            stage_name="noop",
            url="http://new",
            updated_at=newer,
        )
        t_old = Task(
            status=Status.RETRYING,
            stage_name="noop",
            attempts=1,
            url="http://old",
            updated_at=older,
        )
        await t_new.insert()
        await t_old.insert()
        # Ensure ordering fields stick (mongomock/beanie may rewrite on insert)
        await t_new.update({"$set": {"updated_at": newer}})
        await t_old.update({"$set": {"updated_at": older}})

        ex = Executor(concurrency={"noop": 1}, max_attempts=3)
        claimed = await ex.claim_task("noop")
        assert claimed is not None
        assert claimed.id == t_old.id

    run_async(_test())
