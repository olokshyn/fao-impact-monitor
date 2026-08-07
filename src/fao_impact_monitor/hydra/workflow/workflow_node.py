from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from fao_impact_monitor.hydra.workflow.workflow_branch import (
    WorkflowBranch,
    hydrate_workflow_branches,
)


def _reject_dots(name: str, label: str) -> str:
    if "." in name:
        raise ValueError(f"{label} must not contain dots, got {name!r}")
    if not name:
        raise ValueError(f"{label} must be a non-empty string")
    return name


class WorkflowNode(BaseModel):
    """One step in a Workflow plan (embedded in Workflow.nodes).

    ``stage_params`` are passed to ``Stage.process`` as ``params``. When a
    stage needs prior ``Document.stage_results``, pass the ``Workflow.name``
    and ``WorkflowNode.name`` to look up under
    ``stage_results[workflow_name][workflow_node_name]`` (for example
    ``{"prior_workflow": "ingest", "prior_node": "extract"}``).
    """

    name: str
    stage_name: str
    stage_params: dict[str, Any] = Field(default_factory=dict)
    branches: list[WorkflowBranch] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        return _reject_dots(value, "WorkflowNode.name")

    @field_validator("branches", mode="before")
    @classmethod
    def _hydrate_branches(cls, value: list[Any] | None) -> list[WorkflowBranch]:
        return hydrate_workflow_branches(value)
