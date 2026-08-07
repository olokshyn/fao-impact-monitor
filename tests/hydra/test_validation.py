"""Validation of Workflow / WorkflowNode names and structure."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

import pytest
from pydantic import ValidationError

from fao_impact_monitor.hydra import Workflow, WorkflowNode


def test_workflow_node_name_rejects_dots() -> None:
    with pytest.raises(ValidationError, match="must not contain dots"):
        WorkflowNode(name="a.b", stage_name="noop")


def test_workflow_name_rejects_dots(hydra_db: None) -> None:
    with pytest.raises(ValidationError, match="must not contain dots"):
        Workflow(
            name="bad.name",
            nodes=[WorkflowNode(name="n1", stage_name="noop")],
            entrypoints=["n1"],
        )


def test_duplicate_node_names_rejected(hydra_db: None) -> None:
    with pytest.raises(ValidationError, match="Duplicate WorkflowNode.name"):
        Workflow(
            name="wf",
            nodes=[
                WorkflowNode(name="n1", stage_name="noop"),
                WorkflowNode(name="n1", stage_name="increment"),
            ],
            entrypoints=["n1"],
        )


def test_invalid_entrypoint_rejected(hydra_db: None) -> None:
    with pytest.raises(ValidationError, match="Entrypoint"):
        Workflow(
            name="wf",
            nodes=[WorkflowNode(name="n1", stage_name="noop")],
            entrypoints=["missing"],
        )


def test_get_node(
    hydra_db: None,
    run_async: Callable[[Coroutine[Any, Any, Any]], Any],
) -> None:
    async def _test() -> None:
        wf = Workflow(
            name="pipe",
            nodes=[WorkflowNode(name="crawl", stage_name="noop")],
            entrypoints=["crawl"],
        )
        await wf.insert()
        assert wf.get_node("crawl").stage_name == "noop"

    run_async(_test())
