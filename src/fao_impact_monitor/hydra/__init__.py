"""Hydra: distributed data processing engine."""

from __future__ import annotations

from typing import Any

from beanie import init_beanie

from fao_impact_monitor.hydra.config import FetchConfig, HydraConfig
from fao_impact_monitor.hydra.document.document import (
    Document,
    DocumentType,
    Relation,
    RelationSide,
    RelationType,
)
from fao_impact_monitor.hydra.executor import (
    Executor,
    ExecutorHeartbeat,
    sweep_stale_executors,
)
from fao_impact_monitor.hydra.run import Run
from fao_impact_monitor.hydra.stage.fetch_stage import (
    ContentType,
    Fetch,
    FetchRequest,
    FetchResponse,
    FetchStage,
    FetchStageResult,
)
from fao_impact_monitor.hydra.stage.stage import (
    Stage,
    StageResult,
    get_stage,
    get_stage_result_class,
)
from fao_impact_monitor.hydra.status import Status
from fao_impact_monitor.hydra.task.task import Task
from fao_impact_monitor.hydra.workflow.workflow import Workflow
from fao_impact_monitor.hydra.workflow.workflow_branch import (
    WorkflowBranch,
    get_workflow_branch_class,
)
from fao_impact_monitor.hydra.workflow.workflow_node import WorkflowNode

HYDRA_DOCUMENT_MODELS: list[type[Any]] = [
    Task,
    Document,
    Workflow,
    Run,
    ExecutorHeartbeat,
    Fetch,
]


async def init_hydra_beanie(
    database: Any,
    *,
    skip_indexes: bool = False,
    extra_models: list[type[Any]] | None = None,
) -> None:
    """Initialize Beanie with Hydra document models.

    ``extra_models`` lets tests register Document subclasses (and any other
    Beanie models) without modifying production registration.
    """
    models = list(HYDRA_DOCUMENT_MODELS)
    if extra_models:
        models.extend(extra_models)
    await init_beanie(
        database=database,
        document_models=models,
        skip_indexes=skip_indexes,
    )


__all__ = [
    "HYDRA_DOCUMENT_MODELS",
    "ContentType",
    "Document",
    "DocumentType",
    "Executor",
    "ExecutorHeartbeat",
    "Fetch",
    "FetchConfig",
    "FetchRequest",
    "FetchResponse",
    "FetchStage",
    "FetchStageResult",
    "HydraConfig",
    "Relation",
    "RelationSide",
    "RelationType",
    "Run",
    "Stage",
    "StageResult",
    "Status",
    "Task",
    "Workflow",
    "WorkflowBranch",
    "WorkflowNode",
    "get_stage",
    "get_stage_result_class",
    "get_workflow_branch_class",
    "init_hydra_beanie",
    "sweep_stale_executors",
]
