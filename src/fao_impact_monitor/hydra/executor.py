from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any, cast
from uuid import uuid4

from beanie import Document as BeanieDocument
from beanie import Indexed
from beanie.odm.queries.update import UpdateResponse
from beanie.operators import And, Or
from pydantic import Field
from pymongo import ASCENDING

from fao_impact_monitor.hydra.stage.stage import StageResult, get_stage
from fao_impact_monitor.hydra.status import Status
from fao_impact_monitor.hydra.task.task import Task
from fao_impact_monitor.hydra.workflow.workflow import Workflow
from fao_impact_monitor.hydra.workflow.workflow_node import WorkflowNode

logger = logging.getLogger(__name__)


class ExecutorHeartbeat(BeanieDocument):
    """Heartbeat row in the ``executors`` collection."""

    id: Annotated[str, Indexed(unique=True)]  # type: ignore[assignment]
    updated_at: Annotated[datetime, Indexed()] = Field(
        default_factory=lambda: datetime.now(UTC)
    )

    class Settings:
        name = "executors"


async def sweep_stale_executors(
    *,
    heartbeat_interval_minutes: float,
    stale_multiplier: float = 3.0,
) -> list[str]:
    """Reset RUNNING tasks owned by executors whose heartbeat is stale.

    An executor is stale when ``updated_at`` is older than
    ``stale_multiplier * heartbeat_interval_minutes``. Each of its RUNNING
    tasks is set back to SCHEDULED and ``attempts`` is decreased by one.

    Returns the list of stale executor ids that were swept.
    """
    cutoff = datetime.now(UTC) - timedelta(
        minutes=stale_multiplier * heartbeat_interval_minutes
    )
    stale = await ExecutorHeartbeat.find(
        ExecutorHeartbeat.updated_at < cutoff
    ).to_list()
    stale_ids: list[str] = []
    for heartbeat in stale:
        executor_id = str(heartbeat.id)
        stale_ids.append(executor_id)
        running = await Task.find(
            Task.status == Status.RUNNING,
            Task.executor_id == executor_id,
        ).to_list()
        for task in running:
            new_attempts = max(0, task.attempts - 1)
            await task.update(
                {
                    "$set": {
                        "status": Status.SCHEDULED,
                        "executor_id": None,
                        "executor_started_at": None,
                        "attempts": new_attempts,
                        "updated_at": datetime.now(UTC),
                    }
                }
            )
        await heartbeat.delete()
    return stale_ids


class Executor:
    """Process-local worker. Safe to run many instances across machines."""

    def __init__(
        self,
        *,
        concurrency: dict[str, int],
        max_attempts: int = 3,
        heartbeat_interval_minutes: float = 5.0,
        stale_multiplier: float = 3.0,
        claim_idle_sleep_seconds: float = 0.05,
        executor_id: str | None = None,
    ) -> None:
        self.id = executor_id or str(uuid4())
        self.concurrency = dict(concurrency)
        self.max_attempts = max_attempts
        self.heartbeat_interval_minutes = heartbeat_interval_minutes
        self.stale_multiplier = stale_multiplier
        self.claim_idle_sleep_seconds = claim_idle_sleep_seconds
        self._in_flight: dict[str, set[asyncio.Task[None]]] = {
            stage: set() for stage in concurrency
        }
        self._stop = asyncio.Event()
        self._wake = asyncio.Event()
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._main_task: asyncio.Task[None] | None = None

    async def run(self) -> None:
        """Main loop: heartbeat, wait on capacity/completions, claim and execute."""
        self._stop.clear()
        await self.heartbeat()
        self._heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(), name=f"hydra-heartbeat-{self.id}"
        )
        try:
            while not self._stop.is_set():
                claimed_any = False
                for stage_name, limit in self.concurrency.items():
                    while len(self._in_flight[stage_name]) < limit:
                        task = await self.claim_task(stage_name)
                        if task is None:
                            break
                        claimed_any = True
                        async_task = asyncio.create_task(
                            self._run_claimed(task, stage_name),
                            name=f"hydra-task-{task.id}",
                        )
                        self._in_flight[stage_name].add(async_task)

                        def _discard(
                            done_task: asyncio.Task[None],
                            s: str = stage_name,
                        ) -> None:
                            self._in_flight[s].discard(done_task)

                        def _wake(_done_task: asyncio.Task[None]) -> None:
                            self._wake.set()

                        async_task.add_done_callback(_discard)
                        async_task.add_done_callback(_wake)

                in_flight = [t for tasks in self._in_flight.values() for t in tasks]
                if self._stop.is_set():
                    break
                if in_flight:
                    self._wake.clear()
                    wait_stop = asyncio.create_task(self._stop.wait())
                    wait_wake = asyncio.create_task(self._wake.wait())
                    try:
                        done, _pending = await asyncio.wait(
                            [*in_flight, wait_stop, wait_wake],
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                    finally:
                        for helper in (wait_stop, wait_wake):
                            if not helper.done():
                                helper.cancel()
                        await asyncio.gather(
                            wait_stop, wait_wake, return_exceptions=True
                        )
                    for d in done:
                        if d in (wait_stop, wait_wake):
                            continue
                        exc = d.exception() if not d.cancelled() else None
                        if exc is not None:
                            logger.exception(
                                "In-flight task failed",
                                exc_info=exc,
                            )
                elif not claimed_any:
                    try:
                        await asyncio.wait_for(
                            self._stop.wait(),
                            timeout=self.claim_idle_sleep_seconds,
                        )
                    except TimeoutError:
                        pass
        finally:
            if self._heartbeat_task is not None:
                self._heartbeat_task.cancel()
                try:
                    await self._heartbeat_task
                except asyncio.CancelledError:
                    pass
                self._heartbeat_task = None
            remaining = [t for tasks in self._in_flight.values() for t in tasks]
            if remaining:
                if self._stop.is_set():
                    # Graceful stop: drain in-flight work.
                    await asyncio.gather(*remaining, return_exceptions=True)
                else:
                    # Aborted (e.g. task cancelled): do not wait on hung stages.
                    for t in remaining:
                        t.cancel()
                    await asyncio.gather(*remaining, return_exceptions=True)

    async def stop(self) -> None:
        """Signal the loop to drain in-flight work and exit."""
        self._stop.set()
        self._wake.set()

    async def heartbeat(self) -> None:
        """Upsert {id, updated_at} into the executors collection."""
        now = datetime.now(UTC)
        await ExecutorHeartbeat.get_pymongo_collection().update_one(
            {"_id": self.id},
            {"$set": {"updated_at": now}},
            upsert=True,
        )

    async def _heartbeat_loop(self) -> None:
        interval = self.heartbeat_interval_minutes * 60.0
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=interval)
                break
            except TimeoutError:
                await self.heartbeat()

    async def claim_task(self, stage_name: str) -> Task | None:
        """Atomic findOneAndUpdate: eligible + stage_name → RUNNING.

        Prefers the oldest eligible task (``updated_at`` ascending).
        Eligible means ``SCHEDULED``, or ``RETRYING`` with attempts remaining.
        """
        now = datetime.now(UTC)
        claimed = await Task.find_one(
            Task.stage_name == stage_name,
            Or(
                Task.status == Status.SCHEDULED,
                And(
                    Task.status == Status.RETRYING,
                    Task.attempts < self.max_attempts,
                ),
            ),
        ).update(
            {
                "$set": {
                    "status": Status.RUNNING,
                    "executor_id": self.id,
                    "executor_started_at": now,
                    "updated_at": now,
                },
                "$inc": {"attempts": 1},
            },
            response_type=UpdateResponse.NEW_DOCUMENT,
            sort=[("updated_at", ASCENDING)],
        )
        return cast(Task | None, claimed)

    async def _run_claimed(self, task: Task, stage_name: str) -> None:
        try:
            await self.execute_task(task)
        except Exception:
            logger.exception("Unhandled error executing task %s", task.id)
            await task.set_fields(
                status=Status.FAILED,
                error="Unhandled executor exception",
            )

    async def execute_task(self, task: Task) -> None:
        """Resolve WorkflowNode → Stage; process; persist; route or retry/fail."""
        if task.workflow_id is None or task.workflow_node_name is None:
            await task.set_fields(
                status=Status.FAILED,
                error="Task missing workflow_id or workflow_node_name",
            )
            return

        workflow = await Workflow.get(task.workflow_id)
        if workflow is None:
            await task.set_fields(
                status=Status.FAILED,
                error=f"Workflow {task.workflow_id} not found",
            )
            return

        try:
            node = workflow.get_node(task.workflow_node_name)
        except KeyError as exc:
            await task.set_fields(status=Status.FAILED, error=str(exc))
            return

        stage = get_stage(node.stage_name)
        result, child_context = await stage.process(
            task,
            node.stage_params,
            workflow.name,
            task.workflow_node_name,
        )

        if result.status == Status.COMPLETED:
            status = Status.COMPLETED
            error = result.error
        elif result.status == Status.FAILED:
            if task.attempts < self.max_attempts:
                status = Status.RETRYING
                error = result.error or "Stage failed"
            else:
                status = Status.FAILED
                error = result.error or "Stage failed; max attempts reached"
        else:
            raise ValueError(f"Unexpected stage result status: {result.status}")

        await task.set_fields(
            stage_result=result,
            status=status,
            error=error,
        )

        if status == Status.COMPLETED:
            await self._route_children(task, workflow, node, result, child_context)

    async def _route_children(
        self,
        task: Task,
        workflow: Workflow,
        node: WorkflowNode,
        result: StageResult,
        child_context: dict[str, Any] | None,
    ) -> None:
        from fao_impact_monitor.hydra.run import Run

        for branch in node.branches:
            next_nodes = [workflow.get_node(n) for n in branch.next_node_names]
            route_params = {
                **branch.params,
                "workflow_name": workflow.name,
                "url": task.url,
                "source": task.source,
                "document_id": task.document_id,
            }
            children = await branch.route(
                result,
                node,
                next_nodes,
                route_params,
            )
            allowed = set(branch.next_node_names)
            for child in children:
                if child.workflow_node_name not in allowed:
                    raise ValueError(
                        f"Branch {branch.name!r} produced Task targeting "
                        f"{child.workflow_node_name!r}, not in {allowed}"
                    )
                child.parent_task_id = task.id
                child.run_id = task.run_id
                child.workflow_id = workflow.id
                if child.url is None:
                    child.url = task.url
                if child.source is None:
                    child.source = task.source
                if child.document_id is None:
                    child.document_id = task.document_id
                if child_context is not None:
                    child.context = dict(child_context)
                elif task.context is not None:
                    child.context = dict(task.context)
                else:
                    child.context = None
                if child.status == Status.CREATED:
                    child.status = Status.SCHEDULED
                await child.insert()
                assert task.id is not None and child.id is not None
                await task.push_child_task_id(child.id)
                if task.run_id is not None:
                    run = await Run.get(task.run_id)
                    if run is not None:
                        await run.push_task_id(child.id)
