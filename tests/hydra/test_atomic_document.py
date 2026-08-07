"""Concurrent atomic updates on Document fields must not overwrite each other."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

from fao_impact_monitor.hydra import Status
from tests.hydra.conftest import CounterDocument, IncrementStageResult


def test_concurrent_stage_result_writes_no_lost_update(
    hydra_db: None,
    run_async: Callable[[Coroutine[Any, Any, Any]], Any],
) -> None:
    async def _test() -> None:
        doc = CounterDocument(url="http://doc", source="test", counter=0, value=0)
        await doc.insert()
        assert doc.id is not None

        async def writer_a() -> None:
            # Simulate stage A writing its stage_results key + metadata path
            for _ in range(20):
                d = await CounterDocument.get(doc.id)
                assert d is not None
                result = IncrementStageResult(
                    name="increment",
                    status=Status.COMPLETED,
                    counter=1,
                )
                await d.push_stage_result("wf", "node_a", result)
                await d.atomic_set(**{"metadata.a": "from_a"})

        async def writer_b() -> None:
            from tests.hydra.conftest import AddValueStageResult

            for _ in range(20):
                d = await CounterDocument.get(doc.id)
                assert d is not None
                result = AddValueStageResult(
                    name="add_value",
                    status=Status.COMPLETED,
                    value=1,
                )
                await d.push_stage_result("wf", "node_b", result)
                await d.atomic_set(**{"metadata.b": "from_b"})

        await asyncio.gather(writer_a(), writer_b())

        final = await CounterDocument.get(doc.id)
        assert final is not None
        assert len(final.stage_results["wf"]["node_a"]) == 20
        assert len(final.stage_results["wf"]["node_b"]) == 20
        assert final.metadata.get("a") == "from_a"
        assert final.metadata.get("b") == "from_b"

    run_async(_test())


def test_concurrent_counter_and_value_incs(
    hydra_db: None,
    run_async: Callable[[Coroutine[Any, Any, Any]], Any],
) -> None:
    async def _test() -> None:
        doc = CounterDocument(url="http://doc2", source="test", counter=0, value=0)
        await doc.insert()

        async def bump_counter() -> None:
            for _ in range(50):
                d = await CounterDocument.get(doc.id)
                assert d is not None
                await d.inc_counter(1)

        async def bump_value() -> None:
            for _ in range(50):
                d = await CounterDocument.get(doc.id)
                assert d is not None
                await d.inc_value(1)

        await asyncio.gather(bump_counter(), bump_value())
        final = await CounterDocument.get(doc.id)
        assert final is not None
        assert final.counter == 50
        assert final.value == 50

    run_async(_test())
