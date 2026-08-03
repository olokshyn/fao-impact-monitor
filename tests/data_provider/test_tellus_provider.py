"""Unit tests for the Tellus API provider."""

from __future__ import annotations

import asyncio
from typing import Any, Self

import httpx
import pytest
from pydantic import SecretStr

from fao_impact_monitor.config import TellusConfig
from fao_impact_monitor.data_provider.tellus_provider import (
    TellusAuthenticationError,
    tellus_get_all_document_chunks,
    tellus_search_chunks,
)


def _config(**kwargs: Any) -> TellusConfig:
    bearer = kwargs.pop("bearer_token", "secret-token")
    if isinstance(bearer, str):
        bearer = SecretStr(bearer)
    return TellusConfig(
        bearer_token=bearer,
        api_base=kwargs.pop("api_base", "https://tellus.test"),
        **kwargs,
    )


def test_search_raises_when_token_missing() -> None:
    with pytest.raises(TellusAuthenticationError, match="TELLUS_BEARER_TOKEN"):
        asyncio.run(tellus_search_chunks("water", config=_config(bearer_token="")))


def test_search_raises_on_401(monkeypatch: pytest.MonkeyPatch) -> None:
    response = httpx.Response(401, request=httpx.Request("POST", "https://tellus.test"))

    class FakeClient:
        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def post(self, *args: object, **kwargs: object) -> httpx.Response:
            del args, kwargs
            return response

    monkeypatch.setattr(
        "fao_impact_monitor.data_provider.tellus_provider.httpx.AsyncClient",
        FakeClient,
    )
    with pytest.raises(TellusAuthenticationError, match="HTTP 401"):
        asyncio.run(tellus_search_chunks("water", config=_config()))


def test_search_posts_expected_body(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    payload = {"chunks": [{"document_id": "d1"}], "documents": [{"document_id": "d1"}]}

    class FakeClient:
        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def post(
            self,
            url: str,
            *,
            headers: dict[str, str],
            json: dict[str, Any],
        ) -> httpx.Response:
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            return httpx.Response(
                200,
                json=payload,
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr(
        "fao_impact_monitor.data_provider.tellus_provider.httpx.AsyncClient",
        FakeClient,
    )
    result = asyncio.run(
        tellus_search_chunks(
            "water availability",
            ["KEN"],
            config=_config(min_year=2015, max_results=10),
        )
    )
    assert result == payload
    assert captured["url"] == "https://tellus.test/api/v1/search"
    assert captured["headers"]["Authorization"] == "Bearer secret-token"
    assert captured["json"]["semantic_query"] == "water availability"
    assert captured["json"]["min_year"] == 2015
    assert captured["json"]["max_results"] == 10
    assert captured["json"]["keyword_queries"] == ["Republic of Kenya"]


def test_get_all_document_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "document": {"document_id": "doc-1", "title": "Report"},
        "chunks": [{"content": "hello", "page_num": 1}],
    }
    captured: dict[str, Any] = {}

    class FakeClient:
        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def get(
            self,
            url: str,
            *,
            headers: dict[str, str],
        ) -> httpx.Response:
            captured["url"] = url
            captured["headers"] = headers
            return httpx.Response(
                200,
                json=payload,
                request=httpx.Request("GET", url),
            )

    monkeypatch.setattr(
        "fao_impact_monitor.data_provider.tellus_provider.httpx.AsyncClient",
        FakeClient,
    )
    result = asyncio.run(tellus_get_all_document_chunks("doc-1", config=_config()))
    assert result == payload
    assert captured["url"].endswith("/api/v1/sources/document/doc-1/chunks")
    assert captured["headers"]["Authorization"] == "Bearer secret-token"
