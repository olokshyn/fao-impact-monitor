from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, Field

from fao_impact_monitor.hydra.status import Status
from fao_impact_monitor.utils.meta_magic import RegistryMeta, RegistryModelMeta

if TYPE_CHECKING:
    from fao_impact_monitor.hydra.task.task import Task

_STAGE_RESULT_REGISTRY: dict[str, type[StageResult]] = {}


class StageResultMeta(RegistryModelMeta):
    registry = _STAGE_RESULT_REGISTRY
    attr = "name"


class StageResult(ABC, BaseModel, metaclass=StageResultMeta):
    """Result of running a Stage on a Task.

    Subclasses register with ``RegistryModelMeta`` and may add custom fields.
    """

    name: str
    status: Status
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


_STAGE_REGISTRY: dict[str, type[Stage]] = {}


class StageMeta(RegistryMeta):
    registry = _STAGE_REGISTRY
    attr = "name"


class Stage(ABC, metaclass=StageMeta):
    """Worker that executes a unit of work for a Task.

    Must not create Tasks or consult WorkflowBranches.
    """

    name: str
    context_required: ClassVar[dict[str, str] | None] = None
    # Keys this stage needs in Task.context → why each key is required.
    # Workflow.submit checks every stage in the workflow against the root task.

    @abstractmethod
    async def process(
        self,
        task: Task,
        params: dict[str, Any],
        workflow_name: str,
        workflow_node_name: str,
    ) -> tuple[StageResult, dict[str, Any] | None]:
        """Run this stage for ``task``.

        ``params`` are ``WorkflowNode.stage_params``.
        ``workflow_name`` and ``workflow_node_name`` identify where this stage
        reads/writes ``Document.stage_results[workflow_name][workflow_node_name]``.
        When prior results are needed, ``params`` should carry the
        ``Workflow.name`` / ``WorkflowNode.name`` pairs to look up.

        Persist (and optionally merge) this run's ``StageResult`` on the
        document under ``[workflow_name][workflow_node_name]`` using an atomic
        partial update. Never ``.save()`` the whole document.

        Return ``(StageResult, context)``. A non-``None`` ``context`` replaces
        ``Task.context`` on all child tasks created after this completion;
        ``None`` means passthrough: children inherit a copy of the parent ``Task.context``.
        """
        ...


def get_stage(name: str) -> Stage:
    """Instantiate the registered Stage for ``name``."""
    return _STAGE_REGISTRY[name]()


def get_stage_result_class(name: str) -> type[StageResult]:
    """Return the registered ``StageResult`` subclass for ``name``."""
    return _STAGE_RESULT_REGISTRY[name]
