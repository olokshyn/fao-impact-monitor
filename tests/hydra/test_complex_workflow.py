"""Complex workflow with cycles and parallel fan-out (Node3 || Node4)."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

from fao_impact_monitor.hydra import (
    Executor,
    Status,
    Task,
    Workflow,
    WorkflowNode,
)
from tests.hydra.conftest import (
    AddValueStage,
    CounterDocument,
    CounterLessThanBranch,
    CounterOddBranch,
    IncrementStage,
    run_executor_until,
)


def test_complex_counter_workflow_parallel_fanout(
    hydra_db: None,
    run_async: Callable[[Coroutine[Any, Any, Any]], Any],
) -> None:
    """Task → Node1[+1] ⇄ Node2[+2] → parallel Node3[+4] & Node4[+9].

    Document.value starts at 0; after fan-out value must be exactly 13.
    Node3 and Node4 are siblings (Node4 does not follow Node3).
    """

    async def _test() -> None:
        # Ensure test stages are registered
        assert IncrementStage.name == "increment"
        assert AddValueStage.name == "add_value"

        doc = CounterDocument(
            url="http://complex",
            source="test",
            counter=0,
            value=0,
        )
        await doc.insert()
        assert doc.id is not None

        wf = Workflow(
            name="complex",
            nodes=[
                WorkflowNode(
                    name="node1",
                    stage_name="increment",
                    stage_params={"amount": 1},
                    branches=[
                        CounterLessThanBranch(
                            next_node_names=["node1", "node2"],
                            params={"threshold": 3},
                        ),
                    ],
                ),
                WorkflowNode(
                    name="node2",
                    stage_name="increment",
                    stage_params={"amount": 2},
                    branches=[
                        CounterOddBranch(
                            next_node_names=["node1", "node3", "node4"],
                        ),
                    ],
                ),
                WorkflowNode(
                    name="node3",
                    stage_name="add_value",
                    stage_params={"amount": 4},
                ),
                WorkflowNode(
                    name="node4",
                    stage_name="add_value",
                    stage_params={"amount": 9},
                ),
            ],
            entrypoints=["node1"],
        )
        await wf.insert()

        task = Task(
            status=Status.CREATED,
            url=doc.url,
            source=doc.source,
            document_id=doc.id,
        )
        run = await wf.submit(task)

        ex = Executor(
            concurrency={"increment": 2, "add_value": 2},
            claim_idle_sleep_seconds=0.01,
            heartbeat_interval_minutes=60,
            max_attempts=3,
        )

        async def _done() -> bool:
            tasks = await Task.find(Task.run_id == run.id).to_list()
            if not tasks:
                return False
            return all(t.status in {Status.COMPLETED, Status.FAILED} for t in tasks)

        await run_executor_until(ex, predicate=_done, timeout=10.0)
        await run.wait(poll_interval_seconds=0.01)
        assert run.status == Status.COMPLETED

        tasks = await Task.find(Task.run_id == run.id).to_list()
        assert all(t.status == Status.COMPLETED for t in tasks)

        node3_tasks = [t for t in tasks if t.workflow_node_name == "node3"]
        node4_tasks = [t for t in tasks if t.workflow_node_name == "node4"]
        assert len(node3_tasks) == 1
        assert len(node4_tasks) == 1

        t3, t4 = node3_tasks[0], node4_tasks[0]
        # Siblings: same parent, neither is the other's parent
        assert t3.parent_task_id is not None
        assert t3.parent_task_id == t4.parent_task_id
        assert t3.id not in (t4.child_task_ids or [])
        assert t4.id not in (t3.child_task_ids or [])

        parent = await Task.get(t3.parent_task_id)
        assert parent is not None
        assert parent.workflow_node_name == "node2"
        assert set(parent.child_task_ids) == {t3.id, t4.id}

        final_doc = await CounterDocument.get(doc.id)
        assert final_doc is not None
        assert final_doc.value == 13
        assert final_doc.counter == 8

    run_async(_test())
