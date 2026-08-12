"""Unit tests for the web-scout data provider mapping."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel, Field

from fao_impact_monitor.data_provider import web_scout_provider as wsp
from fao_impact_monitor.data_provider.web_scout_provider import (
    WebScoutProviderError,
    map_web_research_result,
    run_web_scout_research,
    unwrap_schema_shaped_instance,
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
    assert mapped.sources[0].source_id.startswith("web:")
    assert "https://" not in mapped.sources[0].source_id
    assert mapped.sources[0].content.startswith("Kenya maize")
    assert mapped.snippet_only_count == 1
    assert "Model summary" not in mapped.sources[0].content
    assert mapped.queries == ["Kenya maize"]
    assert len(mapped.query_stats) == 1
    assert mapped.query_stats[0].query == "Kenya maize"
    assert mapped.query_stats[0].results_returned == 3


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


def test_unwrap_schema_shaped_instance_recovers_values() -> None:
    payload = {
        "description": "LLM output for evaluating coverage",
        "properties": {
            "fully_answered": False,
            "gaps": "Missing cropland percentages",
            "promising_unscraped_urls": ["https://example.org/a"],
            "needs_new_searches": False,
        },
        "required": [
            "fully_answered",
            "gaps",
            "promising_unscraped_urls",
            "needs_new_searches",
        ],
        "title": "CoverageEvaluation",
        "type": "object",
    }
    unwrapped = unwrap_schema_shaped_instance(payload)
    assert unwrapped == {
        "fully_answered": False,
        "gaps": "Missing cropland percentages",
        "promising_unscraped_urls": ["https://example.org/a"],
        "needs_new_searches": False,
    }


def test_unwrap_schema_shaped_instance_ignores_pure_schema() -> None:
    payload = {
        "description": "LLM output for evaluating coverage",
        "properties": {
            "fully_answered": {
                "description": "True if answered",
                "title": "Fully Answered",
                "type": "boolean",
            },
            "gaps": {
                "description": "Missing info",
                "title": "Gaps",
                "type": "string",
            },
        },
        "required": ["fully_answered", "gaps"],
        "title": "CoverageEvaluation",
        "type": "object",
    }
    assert unwrap_schema_shaped_instance(payload) is None


def test_schema_envelope_patch_recovers_coverage_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CoverageEvaluation(BaseModel):
        fully_answered: bool
        gaps: str
        promising_unscraped_urls: list[str] = Field(default_factory=list)
        needs_new_searches: bool = True

    monkeypatch.setattr(wsp, "_SCHEMA_ENVELOPE_PATCHED", False)
    wsp._patch_agents_schema_envelope_validation()

    from agents.agent_output import AgentOutputSchema

    schema = AgentOutputSchema(CoverageEvaluation)
    envelope = {
        "description": "LLM output for evaluating coverage",
        "properties": {
            "fully_answered": False,
            "gaps": "Need more data",
            "promising_unscraped_urls": ["https://example.org/doc"],
            "needs_new_searches": False,
        },
        "required": [
            "fully_answered",
            "gaps",
            "promising_unscraped_urls",
            "needs_new_searches",
        ],
        "title": "CoverageEvaluation",
        "type": "object",
    }
    parsed = schema.validate_json(json.dumps(envelope))
    assert parsed.fully_answered is False
    assert parsed.gaps == "Need more data"
    assert parsed.promising_unscraped_urls == ["https://example.org/doc"]
    assert parsed.needs_new_searches is False
