from __future__ import annotations

import asyncio

import httpx
import pytest
from openai import APITimeoutError

from fao_impact_monitor.athena.agents.runner import _run_with_retry


def test_connect_timeout_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", sleep)
    calls = 0

    async def operation() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise APITimeoutError(httpx.Request("POST", "https://bedrock.example/v1"))
        return "ok"

    assert asyncio.run(_run_with_retry(operation, attempts=4)) == "ok"
    assert calls == 3
    assert sleeps == [2.0, 4.0]


def test_connect_timeout_raises_after_the_last_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def sleep(delay: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", sleep)

    async def operation() -> str:
        raise APITimeoutError(httpx.Request("POST", "https://bedrock.example/v1"))

    with pytest.raises(APITimeoutError):
        asyncio.run(_run_with_retry(operation, attempts=2))
