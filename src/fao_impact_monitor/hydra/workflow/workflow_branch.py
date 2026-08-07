from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from fao_impact_monitor.utils.meta_magic import RegistryModelMeta

if TYPE_CHECKING:
    from fao_impact_monitor.hydra.stage.stage import StageResult
    from fao_impact_monitor.hydra.task.task import Task
    from fao_impact_monitor.hydra.workflow.workflow_node import WorkflowNode

_WORKFLOW_BRANCH_REGISTRY: dict[str, type[WorkflowBranch]] = {}


class WorkflowBranchMeta(RegistryModelMeta):
    registry = _WORKFLOW_BRANCH_REGISTRY
    attr = "name"


class WorkflowBranch(ABC, BaseModel, metaclass=WorkflowBranchMeta):
    """Router that builds child Tasks from a StageResult.

    ``name`` is the globally unique registry key for this branch algorithm.
    ``next_node_names`` are ``WorkflowNode.name`` values within the Workflow.
    """

    name: str
    next_node_names: list[str]
    params: dict[str, Any] = Field(default_factory=dict)

    @abstractmethod
    async def route(
        self,
        result: StageResult,
        current_node: WorkflowNode,
        next_nodes: list[WorkflowNode],
        params: dict[str, Any],
    ) -> list[Task]:
        """Build child Tasks from ``result``.

        Each returned Task must set ``workflow_id``, ``workflow_node_name``
        (plain ``WorkflowNode.name``), ``stage_name``, ``run_id``,
        ``parent_task_id``, etc.
        """
        ...


def get_workflow_branch_class(name: str) -> type[WorkflowBranch]:
    """Return the registered WorkflowBranch subclass for ``name``."""
    return _WORKFLOW_BRANCH_REGISTRY[name]


def hydrate_workflow_branch(data: Any) -> WorkflowBranch:
    """Restore a concrete WorkflowBranch subclass from a dict or instance."""
    if isinstance(data, WorkflowBranch) and type(data) is not WorkflowBranch:
        return data
    raw = data.model_dump() if isinstance(data, WorkflowBranch) else data
    if not isinstance(raw, dict):
        raise TypeError(f"Cannot hydrate WorkflowBranch from {type(data)}")
    branch_name = raw.get("name")
    if not isinstance(branch_name, str):
        raise TypeError("WorkflowBranch payload must include a string 'name'")
    try:
        cls = get_workflow_branch_class(branch_name)
    except KeyError as exc:
        raise ValueError(
            f"Unknown WorkflowBranch name {branch_name!r}; is it imported?"
        ) from exc
    return cls.model_validate(raw)


def hydrate_workflow_branches(value: list[Any] | None) -> list[WorkflowBranch]:
    """Hydrate a list of branch payloads via the registry."""
    if not value:
        return []
    return [hydrate_workflow_branch(item) for item in value]
