"""Shared fixtures and test doubles for Hydra (in-memory mongomock only)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine, Iterator
from typing import Any, ClassVar, TypeVar

import mongomock
import pytest
from mongomock_motor import AsyncMongoMockClient

from fao_impact_monitor.hydra import (
    Document,
    Executor,
    Stage,
    StageResult,
    Status,
    Task,
    WorkflowBranch,
    WorkflowNode,
    init_hydra_beanie,
)
from fao_impact_monitor.hydra.document.document import DocumentType
from fao_impact_monitor.utils.meta_magic import get_class_id_value

T = TypeVar("T")

# Beanie passes authorizedCollections=; mongomock does not accept it.
_orig_list_collection_names = mongomock.Database.list_collection_names


def _patched_list_collection_names(
    self: Any,
    filter: dict[str, Any] | None = None,
    **kwargs: Any,
) -> list[str]:
    kwargs.pop("authorizedCollections", None)
    return _orig_list_collection_names(self, filter=filter)


mongomock.Database.list_collection_names = _patched_list_collection_names  # type: ignore[assignment]


class CounterDocument(Document):
    """Test Document with routing counter and accumulated value."""

    counter: int = 0
    value: int = 0

    class Settings:
        class_id_value = DocumentType.WEB_PAGE

    @property
    def type(self) -> DocumentType:
        return DocumentType(get_class_id_value(type(self)))

    async def inc_counter(self, amount: int = 1) -> int:
        await self.update({"$inc": {"counter": amount}})
        refreshed = await type(self).get(self.id)
        if refreshed is None:
            raise RuntimeError("Document disappeared during inc_counter")
        self.counter = refreshed.counter
        return self.counter

    async def inc_value(self, amount: int) -> int:
        await self.update({"$inc": {"value": amount}})
        refreshed = await type(self).get(self.id)
        if refreshed is None:
            raise RuntimeError("Document disappeared during inc_value")
        self.value = refreshed.value
        return self.value


class IncrementStageResult(StageResult):
    name: str = "increment"
    status: Status = Status.COMPLETED
    counter: int | None = None
    value: int | None = None


class AddValueStageResult(StageResult):
    name: str = "add_value"
    status: Status = Status.COMPLETED
    counter: int | None = None
    value: int | None = None


class NoopStageResult(StageResult):
    name: str = "noop"
    status: Status = Status.COMPLETED


class FailNTimesStageResult(StageResult):
    name: str = "fail_n_times"
    status: Status = Status.COMPLETED


class HangStageResult(StageResult):
    name: str = "hang"
    status: Status = Status.COMPLETED


class IncrementStage(Stage):
    """Increment CounterDocument.counter by params['amount'] (default 1)."""

    name = "increment"

    async def process(
        self,
        task: Task,
        params: dict[str, Any],
        workflow_name: str,
        workflow_node_name: str,
    ) -> StageResult:
        amount = int(params.get("amount", 1))
        doc = await _load_counter_doc(task)
        new_counter = await doc.inc_counter(amount)
        result = IncrementStageResult(
            name=self.name,
            status=Status.COMPLETED,
            counter=new_counter,
        )
        await doc.push_stage_result(workflow_name, workflow_node_name, result)
        if task.document_id is None and doc.id is not None:
            await task.set_fields(document_id=doc.id)
        return result


class AddValueStage(Stage):
    """Atomically add params['amount'] to CounterDocument.value."""

    name = "add_value"

    async def process(
        self,
        task: Task,
        params: dict[str, Any],
        workflow_name: str,
        workflow_node_name: str,
    ) -> StageResult:
        amount = int(params["amount"])
        doc = await _load_counter_doc(task)
        new_value = await doc.inc_value(amount)
        result = AddValueStageResult(
            name=self.name,
            status=Status.COMPLETED,
            value=new_value,
            counter=doc.counter,
        )
        await doc.push_stage_result(workflow_name, workflow_node_name, result)
        if task.document_id is None and doc.id is not None:
            await task.set_fields(document_id=doc.id)
        return result


class NoopStage(Stage):
    name = "noop"

    async def process(
        self,
        task: Task,
        params: dict[str, Any],
        workflow_name: str,
        workflow_node_name: str,
    ) -> StageResult:
        return NoopStageResult(name=self.name, status=Status.COMPLETED)


class FailNTimesStage(Stage):
    """Fail the first N process calls globally, then succeed."""

    name = "fail_n_times"
    _remaining: ClassVar[dict[str, int]] = {}

    @classmethod
    def reset(cls, key: str, n: int) -> None:
        cls._remaining[key] = n

    async def process(
        self,
        task: Task,
        params: dict[str, Any],
        workflow_name: str,
        workflow_node_name: str,
    ) -> StageResult:
        key = str(params.get("key", "default"))
        remaining = self._remaining.get(key, 0)
        if remaining > 0:
            self._remaining[key] = remaining - 1
            return FailNTimesStageResult(
                name=self.name,
                status=Status.FAILED,
                error=f"fail_n_times:{remaining}",
            )
        return FailNTimesStageResult(name=self.name, status=Status.COMPLETED)


class HangStage(Stage):
    """Block until the shared asyncio.Event is set (or cancelled)."""

    name = "hang"
    gate: asyncio.Event | None = None

    async def process(
        self,
        task: Task,
        params: dict[str, Any],
        workflow_name: str,
        workflow_node_name: str,
    ) -> StageResult:
        if self.gate is None:
            raise RuntimeError("HangStage.gate not set")
        await self.gate.wait()
        return HangStageResult(name=self.name, status=Status.COMPLETED)


class AlwaysNextBranch(WorkflowBranch):
    """Route to every next_node_names entry (one Task each)."""

    name: str = "always_next"

    async def route(
        self,
        result: StageResult,
        current_node: WorkflowNode,
        next_nodes: list[WorkflowNode],
        params: dict[str, Any],
    ) -> list[Task]:
        return [
            Task(
                status=Status.SCHEDULED,
                workflow_node_name=node.name,
                stage_name=node.stage_name,
                url=params.get("url"),
                source=params.get("source"),
                document_id=params.get("document_id"),
            )
            for node in next_nodes
        ]


class CounterLessThanBranch(WorkflowBranch):
    """If result.counter < threshold → first next node; else → second."""

    name: str = "counter_lt"

    async def route(
        self,
        result: StageResult,
        current_node: WorkflowNode,
        next_nodes: list[WorkflowNode],
        params: dict[str, Any],
    ) -> list[Task]:
        threshold = int(params.get("threshold", 3))
        counter = getattr(result, "counter", None)
        if counter is None:
            counter = 0
        if counter < threshold:
            target = next_nodes[0]
        else:
            target = next_nodes[1]
        return [
            Task(
                status=Status.SCHEDULED,
                workflow_node_name=target.name,
                stage_name=target.stage_name,
                url=params.get("url"),
                source=params.get("source"),
                document_id=params.get("document_id"),
            )
        ]


class CounterOddBranch(WorkflowBranch):
    """If counter odd → first next; else fan-out all remaining next nodes."""

    name: str = "counter_odd"

    async def route(
        self,
        result: StageResult,
        current_node: WorkflowNode,
        next_nodes: list[WorkflowNode],
        params: dict[str, Any],
    ) -> list[Task]:
        counter = getattr(result, "counter", None)
        if counter is None:
            counter = 0
        if counter % 2 == 1:
            targets = [next_nodes[0]]
        else:
            targets = list(next_nodes[1:])
        return [
            Task(
                status=Status.SCHEDULED,
                workflow_node_name=n.name,
                stage_name=n.stage_name,
                url=params.get("url"),
                source=params.get("source"),
                document_id=params.get("document_id"),
            )
            for n in targets
        ]


class FanOutBranch(WorkflowBranch):
    """Create one child Task for every next_node_names entry."""

    name: str = "fan_out"

    async def route(
        self,
        result: StageResult,
        current_node: WorkflowNode,
        next_nodes: list[WorkflowNode],
        params: dict[str, Any],
    ) -> list[Task]:
        return [
            Task(
                status=Status.SCHEDULED,
                workflow_node_name=n.name,
                stage_name=n.stage_name,
                url=params.get("url"),
                source=params.get("source"),
                document_id=params.get("document_id"),
            )
            for n in next_nodes
        ]


class IllegalTargetBranch(WorkflowBranch):
    """Deliberately targets a node outside next_node_names (for rejection tests)."""

    name: str = "illegal_target"

    async def route(
        self,
        result: StageResult,
        current_node: WorkflowNode,
        next_nodes: list[WorkflowNode],
        params: dict[str, Any],
    ) -> list[Task]:
        bad = str(params.get("bad_node", "not_allowed"))
        return [
            Task(
                status=Status.SCHEDULED,
                workflow_node_name=bad,
                stage_name="noop",
            )
        ]


async def _load_counter_doc(task: Task) -> CounterDocument:
    if task.document_id is not None:
        doc = await CounterDocument.get(task.document_id)
        if doc is None:
            raise RuntimeError(f"Document {task.document_id} not found")
        return doc
    if not task.url:
        raise RuntimeError("Task has neither document_id nor url")
    existing = await CounterDocument.find_one(
        CounterDocument.url == task.url,
        CounterDocument.source == task.source,
    )
    if existing is not None:
        return existing
    doc = CounterDocument(url=task.url, source=task.source, counter=0, value=0)
    await doc.insert()
    return doc


@pytest.fixture
def event_loop() -> Iterator[asyncio.AbstractEventLoop]:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()
    asyncio.set_event_loop(None)


@pytest.fixture
def run_async(
    event_loop: asyncio.AbstractEventLoop,
) -> Callable[[Coroutine[Any, Any, T]], T]:
    def _run(coro: Coroutine[Any, Any, T]) -> T:
        return event_loop.run_until_complete(coro)

    return _run


@pytest.fixture
def hydra_db(event_loop: asyncio.AbstractEventLoop) -> Iterator[None]:
    """Initialize Beanie on in-memory mongomock (no external Mongo)."""

    async def _setup() -> None:
        client: Any = AsyncMongoMockClient()
        await init_hydra_beanie(
            client["hydra_test"],
            skip_indexes=True,
            extra_models=[CounterDocument],
        )

    event_loop.run_until_complete(_setup())
    FailNTimesStage._remaining.clear()
    HangStage.gate = None
    yield


async def run_executor_until(
    executor: Executor,
    *,
    predicate: Callable[[], Coroutine[Any, Any, bool]],
    timeout: float = 5.0,
) -> None:
    """Run executor in the background until predicate is true, then stop it."""
    main = asyncio.create_task(executor.run(), name="test-executor")
    deadline = asyncio.get_event_loop().time() + timeout
    try:
        while asyncio.get_event_loop().time() < deadline:
            if await predicate():
                break
            await asyncio.sleep(0.01)
        else:
            raise TimeoutError("Executor predicate timed out")
    finally:
        await executor.stop()
        try:
            await asyncio.wait_for(main, timeout=2.0)
        except (TimeoutError, asyncio.CancelledError):
            main.cancel()
            try:
                await main
            except asyncio.CancelledError:
                pass
