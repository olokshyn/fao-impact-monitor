"""Unit and integration tests for LinkExtractAgent."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import dspy
import pytest

from fao_impact_monitor.agent.link_extract_agent import (
    LinkExtractAgent,
    extract_page_urls,
    validate_urls_in_body,
)
from fao_impact_monitor.agent.link_extract_train_data import load_link_extract_examples
from fao_impact_monitor.config import get_config


def test_gold_urls_are_substrings_of_train_html() -> None:
    for example in load_link_extract_examples():
        missing = validate_urls_in_body(example["urls"], example["page_body"])
        assert missing == [], f"{example['id']}: missing {missing[:3]}"


def test_validate_urls_in_body_detects_missing() -> None:
    body = "hello https://example.com/a world"
    assert validate_urls_in_body(
        ["https://example.com/a", "https://example.com/b"],
        body,
    ) == ["https://example.com/b"]


def test_extract_retries_when_url_missing_from_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = LinkExtractAgent()
    calls: list[str] = []

    def fake_extract(**kwargs: Any) -> dspy.Prediction:
        calls.append(str(kwargs.get("correction", "")))
        if len(calls) == 1:
            return dspy.Prediction(urls=["https://example.com/good", "https://missing"])
        return dspy.Prediction(urls=["https://example.com/good"])

    monkeypatch.setattr(agent, "extract", MagicMock(side_effect=fake_extract))
    result = agent(
        page_url="https://example.com/page",
        page_body="link https://example.com/good here",
        topic="t",
        data_filter="d",
        max_urls=10,
        max_retries=2,
    )
    assert list(result.urls) == ["https://example.com/good"]
    assert len(calls) == 2
    assert calls[0] == ""
    assert "https://missing" in calls[1]


def test_extract_page_urls_returns_list(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = LinkExtractAgent()

    def fake_extract(**kwargs: Any) -> dspy.Prediction:
        del kwargs
        return dspy.Prediction(urls=["https://example.com/a"])

    monkeypatch.setattr(agent, "extract", MagicMock(side_effect=fake_extract))
    urls = asyncio.run(
        extract_page_urls(
            page_url="https://example.com/page",
            page_body="https://example.com/a",
            topic="t",
            data_filter="d",
            agent=agent,
        )
    )
    assert urls == ["https://example.com/a"]


def _require_bedrock_and_state() -> Path:
    config = get_config()
    if not config.aws_bedrock.api_key.get_secret_value():
        pytest.skip("AWS_BEDROCK_API_KEY not configured")
    state_path = Path(config.link_extract.dspy_state_path)
    if not state_path.is_file():
        pytest.skip(f"Trained LinkExtractAgent state missing: {state_path}")
    return state_path


@pytest.mark.integration
@pytest.mark.parametrize(
    "example_id",
    [
        "emergencies_el_nino_page1",
        "giews_el_nino_la_nina_collection",
    ],
)
def test_link_extract_agent_matches_gold(example_id: str) -> None:
    state_path = _require_bedrock_and_state()
    example = next(
        item for item in load_link_extract_examples() if item["id"] == example_id
    )
    urls = asyncio.run(
        extract_page_urls(
            page_url=example["page_url"],
            page_body=example["page_body"],
            topic=example["topic"],
            data_filter=example["data_filter"],
            max_urls=max(len(example["urls"]), 50),
            state_path=state_path,
        )
    )
    assert urls == example["urls"]
