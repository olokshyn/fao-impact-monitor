from __future__ import annotations

from typing import Annotated

from beanie import Document as BeanieDocument
from beanie import Indexed
from pydantic import Field, PrivateAttr, field_validator, model_validator

from fao_impact_monitor.hydra.run import Run
from fao_impact_monitor.hydra.stage.stage import get_stage
from fao_impact_monitor.hydra.status import Status
from fao_impact_monitor.hydra.task.task import Task
from fao_impact_monitor.hydra.workflow.workflow_node import WorkflowNode, _reject_dots


class Workflow(BeanieDocument):
    """Static plan for a pipeline (collection ``workflows``)."""

    name: Annotated[str, Indexed(unique=True)]
    nodes: list[WorkflowNode] = Field(default_factory=list)
    entrypoints: list[str] = Field(default_factory=list)

    _node_registry: dict[str, WorkflowNode] = PrivateAttr(default_factory=dict)

    class Settings:
        name = "workflows"

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        return _reject_dots(value, "Workflow.name")

    @model_validator(mode="after")
    def _build_registry_and_validate(self) -> Workflow:
        registry: dict[str, WorkflowNode] = {}
        for node in self.nodes:
            if node.name in registry:
                raise ValueError(
                    f"Duplicate WorkflowNode.name {node.name!r} in Workflow "
                    f"{self.name!r}"
                )
            registry[node.name] = node
        for entry in self.entrypoints:
            if entry not in registry:
                raise ValueError(
                    f"Entrypoint {entry!r} is not a node in Workflow {self.name!r}"
                )
        self._node_registry = registry
        return self

    def get_node(self, name: str) -> WorkflowNode:
        """Lookup by ``WorkflowNode.name`` via the in-memory registry."""
        try:
            return self._node_registry[name]
        except KeyError as exc:
            raise KeyError(
                f"No WorkflowNode named {name!r} in Workflow {self.name!r}"
            ) from exc

    def _validate_task_context(self, task: Task) -> None:
        """Ensure ``task.context`` has every key required by stages in this workflow."""
        context = task.context or {}
        missing: list[str] = []
        for node in self.nodes:
            stage = get_stage(node.stage_name)
            required = type(stage).context_required
            if not required:
                continue
            for key, reason in required.items():
                if key not in context:
                    missing.append(
                        f"{key!r} (required by stage {stage.name!r} "
                        f"on node {node.name!r}: {reason})"
                    )
        if missing:
            raise ValueError(
                "Task.context is missing required keys for Workflow "
                f"{self.name!r}: " + "; ".join(missing)
            )

    async def submit(self, task: Task) -> Run:
        """Accept a Task in CREATED state and enqueue work at entrypoints."""
        if self.id is None:
            raise RuntimeError("Workflow must be saved before submit()")
        if task.status != Status.CREATED:
            raise ValueError(
                f"Task must be in CREATED status to submit, got {task.status}"
            )
        if not self.entrypoints:
            raise ValueError(f"Workflow {self.name!r} has no entrypoints")

        self._validate_task_context(task)

        run = Run(workflow_id=self.id, status=Status.SCHEDULED)
        await run.insert()

        for index, entry_name in enumerate(self.entrypoints):
            node = self.get_node(entry_name)
            if index == 0:
                entry_task = task
            else:
                entry_task = task.model_copy(deep=True)
                entry_task.id = None

            entry_task.run_id = run.id
            entry_task.workflow_id = self.id
            entry_task.workflow_node_name = node.name
            entry_task.stage_name = node.stage_name
            entry_task.status = Status.SCHEDULED
            entry_task.parent_task_id = None
            await entry_task.insert()
            assert run.id is not None and entry_task.id is not None
            await run.push_task_id(entry_task.id)

        return run
