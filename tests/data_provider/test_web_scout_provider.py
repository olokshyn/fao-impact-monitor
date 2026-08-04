"""Unit tests for the web-scout data provider mapping."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from fao_impact_monitor.data_provider.web_scout_provider import (
    WebScoutProviderError,
    map_web_research_result,
    run_web_scout_research,
)


def test_map_accepts_scraped_and_ignores_snippets() -> None:
    result = SimpleNamespace(
        scraped=[
            SimpleNamespace(
                url="https://fao.org/report.pdf",
                title="FAO Report",
                content="Kenya maize production declined by 12%.",
            )
        ],
        snippet_only=[
            SimpleNamespace(
                url="https://example.com/snippet",
                title="Snippet",
                content="Kenya maize...",
            )
        ],
        scrape_failed=[],
        blocked_by_policy=[],
        source_http_error=[],
        scraped_irrelevant=[],
        bot_detected=[],
        queries=[SimpleNamespace(query="Kenya maize", num_results_returned=3)],
        synthesis="Model summary must not be evidence.",
    )
    mapped = map_web_research_result(result, query="Kenya maize")
    assert len(mapped.sources) == 1
    assert mapped.sources[0].url == "https://fao.org/report.pdf"
    assert mapped.sources[0].content.startswith("Kenya maize")
    assert mapped.snippet_only_count == 1
    assert "Model summary" not in mapped.sources[0].content


def test_map_skips_empty_scraped_content() -> None:
    result = SimpleNamespace(
        scraped=[
            SimpleNamespace(url="https://a.org", title="A", content=""),
            SimpleNamespace(url="", title="B", content="text"),
        ],
        snippet_only=[],
        scrape_failed=[],
        blocked_by_policy=[],
        source_http_error=[],
        scraped_irrelevant=[],
        bot_detected=[],
        queries=[],
        synthesis="",
    )
    mapped = map_web_research_result(result, query="q")
    assert mapped.sources == []


def test_run_web_scout_research_uses_injected_fn() -> None:
    async def fake_fn(query: str, **kwargs: Any) -> Any:
        del kwargs
        return SimpleNamespace(
            scraped=[
                SimpleNamespace(
                    url="https://example.org/doc",
                    title="Doc",
                    content=f"Body for {query}",
                )
            ],
            snippet_only=[],
            scrape_failed=[],
            blocked_by_policy=[],
            source_http_error=[],
            scraped_irrelevant=[],
            bot_detected=[],
            queries=[SimpleNamespace(query=query)],
            synthesis="ignore",
        )

    mapped = asyncio.run(
        run_web_scout_research("Kenya drought", web_research_fn=fake_fn)
    )
    assert len(mapped.sources) == 1
    assert "Kenya drought" in mapped.sources[0].content


def test_run_web_scout_research_wraps_failures() -> None:
    async def boom(query: str, **kwargs: Any) -> Any:
        del query, kwargs
        raise RuntimeError("network down")

    with pytest.raises(WebScoutProviderError, match="network down"):
        asyncio.run(run_web_scout_research("q", web_research_fn=boom))
