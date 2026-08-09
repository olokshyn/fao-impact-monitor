from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any

from beanie import Document as BeanieDocument
from beanie import Indexed, PydanticObjectId
from pydantic import BaseModel, Field

from fao_impact_monitor.hydra.stage.stage import StageResult
from fao_impact_monitor.hydra.status import Status


class TaskState(BaseModel):
    """Partial overrides a Stage may apply to descendant Tasks.

    Any field left as ``None`` means "do not override"; the Executor keeps the
    parent Task's value (and still preserves ``url`` / ``source`` /
    ``document_id`` if a WorkflowBranch already set them on the child).
    """

    context: dict[str, Any] | None = None
    priority: int | None = None
    url: str | None = None
    source: str | None = None
    document_id: PydanticObjectId | None = None


class Task(BeanieDocument):
    """Unit of work in the Hydra task queue (collection ``tasks``)."""

    run_id: Annotated[PydanticObjectId | None, Indexed()] = None
    workflow_id: Annotated[PydanticObjectId | None, Indexed()] = None
    workflow_node_name: Annotated[str | None, Indexed()] = None
    parent_task_id: Annotated[PydanticObjectId | None, Indexed()] = None
    child_task_ids: list[PydanticObjectId] = Field(default_factory=list)
    status: Annotated[Status, Indexed()] = Status.CREATED
    stage_name: Annotated[str | None, Indexed()] = None
    context: dict[str, Any] | None = None
    priority: int = 0
    url: Annotated[str | None, Indexed()] = None
    source: Annotated[str | None, Indexed()] = None
    document_id: Annotated[PydanticObjectId | None, Indexed()] = None
    attempts: int = 0
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    executor_id: Annotated[str | None, Indexed()] = None
    executor_started_at: Annotated[datetime | None, Indexed()] = None
    stage_result: StageResult | None = None
    error: str | dict[str, Any] | None = None

    class Settings:
        name = "tasks"
        is_root = True

    async def push_child_task_id(self, child_id: PydanticObjectId) -> None:
        """Atomically append ``child_id`` to ``child_task_ids``."""
        await self.update(
            {"$push": {"child_task_ids": child_id}},
            {"$set": {"updated_at": datetime.now(UTC)}},
        )
        # Beanie applies the operator to the local instance; avoid double-append.
        if child_id not in self.child_task_ids:
            self.child_task_ids.append(child_id)

    async def set_fields(self, **fields: Any) -> None:
        """Atomically ``$set`` selected fields on this task."""
        fields.setdefault("updated_at", datetime.now(UTC))
        await self.update({"$set": fields})
        for key, value in fields.items():
            setattr(self, key, value)
