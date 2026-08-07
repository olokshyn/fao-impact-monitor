from __future__ import annotations

import asyncio
import logging
from collections import Counter
from datetime import UTC, datetime
from typing import Annotated

from beanie import Document as BeanieDocument
from beanie import Indexed, PydanticObjectId
from pydantic import Field

from fao_impact_monitor.hydra.status import Status
from fao_impact_monitor.hydra.task.task import Task

logger = logging.getLogger(__name__)


class Run(BeanieDocument):
    """Group of Tasks for one execution of a Workflow (collection ``runs``)."""

    workflow_id: Annotated[PydanticObjectId, Indexed()]
    task_ids: list[PydanticObjectId] = Field(default_factory=list)
    status: Annotated[Status, Indexed()] = Status.SCHEDULED
    created_at: Annotated[datetime, Indexed()] = Field(
        default_factory=lambda: datetime.now(UTC)
    )
    updated_at: Annotated[datetime, Indexed()] = Field(
        default_factory=lambda: datetime.now(UTC)
    )

    class Settings:
        name = "runs"

    async def push_task_id(self, task_id: PydanticObjectId) -> None:
        """Atomically append ``task_id`` to ``task_ids``."""
        await self.update(
            {"$push": {"task_ids": task_id}},
            {"$set": {"updated_at": datetime.now(UTC)}},
        )
        # Beanie applies the operator to the local instance; avoid double-append.
        if task_id not in self.task_ids:
            self.task_ids.append(task_id)

    async def wait(self, *, poll_interval_seconds: float = 1.0) -> None:
        """Block until every Task with this run_id is COMPLETED or FAILED.

        Then set Run.status:
        - COMPLETED if every top-level (Workflow entrypoint) Task is COMPLETED,
          even if some non-entrypoint Tasks in the run FAILED;
        - FAILED otherwise.

        ``Run.status`` is updated only by this method (not by Executors).
        """
        from fao_impact_monitor.hydra.workflow.workflow import Workflow

        if self.id is None:
            raise RuntimeError("Run must be saved before wait()")

        terminal = {Status.COMPLETED, Status.FAILED}
        while True:
            tasks = await Task.find(Task.run_id == self.id).to_list()
            completed = sum(1 for t in tasks if t.status == Status.COMPLETED)
            failed = sum(1 for t in tasks if t.status == Status.FAILED)
            active = [t for t in tasks if t.status not in terminal]
            by_stage: Counter[str] = Counter(
                (t.workflow_node_name or t.stage_name or "?") for t in active
            )
            by_status: Counter[str] = Counter(t.status.value for t in active)
            logger.info(
                "Run %s: %d/%d completed, %d failed, %d active "
                "(active by status=%s, by stage=%s)",
                self.id,
                completed,
                len(tasks),
                failed,
                len(active),
                dict(by_status),
                dict(by_stage),
            )
            if tasks and all(t.status in terminal for t in tasks):
                break
            await asyncio.sleep(poll_interval_seconds)

        workflow = await Workflow.get(self.workflow_id)
        if workflow is None:
            raise RuntimeError(
                f"Workflow {self.workflow_id} not found for Run {self.id}"
            )

        entrypoint_names = set(workflow.entrypoints)
        entrypoint_tasks = [
            t
            for t in tasks
            if t.parent_task_id is None and t.workflow_node_name in entrypoint_names
        ]
        if entrypoint_tasks and all(
            t.status == Status.COMPLETED for t in entrypoint_tasks
        ):
            new_status = Status.COMPLETED
        else:
            new_status = Status.FAILED

        now = datetime.now(UTC)
        await self.update({"$set": {"status": new_status, "updated_at": now}})
        self.status = new_status
        self.updated_at = now
