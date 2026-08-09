"""Tests for LinkExtractStage (train_data HTML via mocked fetch body_path)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from fao_impact_monitor.agent.link_extract_train_data import (
    TRAIN_DATA_DIR,
    load_link_extract_examples,
)
from fao_impact_monitor.config import get_config
from fao_impact_monitor.hydra.document.document import Document
from fao_impact_monitor.hydra.stage.fetch_stage import ContentType, FetchStageResult
from fao_impact_monitor.hydra.stage.link_extract_stage import (
    LinkExtractStage,
    LinkExtractStageResult,
)
from fao_impact_monitor.hydra.status import Status
from fao_impact_monitor.hydra.task.task import Task

TOPIC = (
    "Droughts and floods caused by El Niño and their impact on agriculture "
    "and livelihoods"
)
DATA_FILTER = (
    "evidence related to these specific El Niño events in 1997-98, 2015-16, "
    "2018-19, 2023-24"
)


async def _seed_document_with_fetch(
    *,
    page_url: str,
    html_path: Path,
    workflow_name: str = "crawl",
    fetch_node: str = "fetch",
) -> Document:
    document = Document(url=page_url, source="FAORepository")
    await document.insert()
    fetch_result = FetchStageResult(
        name="fetch",
        status=Status.COMPLETED,
        status_code=200,
        requested_url=page_url,
        fetched_url=page_url,
        content_type=ContentType.HTML,
        body_path=str(html_path),
    )
    await document.push_stage_result(workflow_name, fetch_node, fetch_result)
    return document


def test_stage_reports_gold_links_for_both_train_pages(
    hydra_db: None,
    run_async: Any,
) -> None:
    """E2E stage: body_path points at train_data HTML; extract returns gold."""

    async def _test() -> None:
        examples = load_link_extract_examples()
        assert len(examples) == 2
        for example in examples:
            html_path = TRAIN_DATA_DIR / example["html_file"]
            document = await _seed_document_with_fetch(
                page_url=example["page_url"],
                html_path=html_path,
            )
            gold = list(example["urls"])

            def make_extract_fn(expected: list[str]) -> Any:
                async def extract_fn(**kwargs: Any) -> list[str]:
                    del kwargs
                    return list(expected)

                return extract_fn

            stage = LinkExtractStage(extract_fn=make_extract_fn(gold))
            task = Task(
                url=example["page_url"],
                source="FAORepository",
                document_id=document.id,
                priority=3,
                context={
                    "topic": TOPIC,
                    "data_filter": DATA_FILTER,
                    "depth": 0,
                },
            )
            result, task_state = await stage.process(
                task,
                {"fetch_node_name": "fetch"},
                "crawl",
                "link_extract",
            )
            assert isinstance(result, LinkExtractStageResult)
            assert result.status == Status.COMPLETED
            assert result.urls == gold
            assert task_state is not None
            assert task_state.priority == 4
            assert task_state.context is not None
            assert task_state.context["topic"] == TOPIC
            assert task_state.context["data_filter"] == DATA_FILTER
            assert task_state.context["depth"] == 1

            refreshed = await Document.get(document.id)
            assert refreshed is not None
            stored = refreshed.latest_stage_result("crawl", "link_extract")
            assert isinstance(stored, LinkExtractStageResult)
            assert stored.urls == gold

    run_async(_test())


def test_stage_raises_without_fetch_result(
    hydra_db: None,
    run_async: Any,
) -> None:
    async def _extract(**kwargs: Any) -> list[str]:
        del kwargs
        return []

    async def _test() -> None:
        document = Document(url="https://example.com/x", source="test")
        await document.insert()
        stage = LinkExtractStage(extract_fn=_extract)
        task = Task(
            url="https://example.com/x",
            source="test",
            document_id=document.id,
            context={"topic": TOPIC, "data_filter": DATA_FILTER, "depth": 0},
        )
        with pytest.raises(RuntimeError, match="No FetchStageResult"):
            await stage.process(
                task,
                {"fetch_node_name": "fetch"},
                "crawl",
                "link_extract",
            )

    run_async(_test())


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
def test_stage_e2e_live_agent_matches_gold(
    hydra_db: None,
    run_async: Any,
    example_id: str,
) -> None:
    """Stage e2e with train_data HTML body_path and the trained live agent."""
    _require_bedrock_and_state()
    example = next(
        item for item in load_link_extract_examples() if item["id"] == example_id
    )

    async def _test() -> None:
        html_path = TRAIN_DATA_DIR / example["html_file"]
        document = await _seed_document_with_fetch(
            page_url=example["page_url"],
            html_path=html_path,
        )
        stage = LinkExtractStage()
        task = Task(
            url=example["page_url"],
            source="FAORepository",
            document_id=document.id,
            priority=0,
            context={
                "topic": example["topic"],
                "data_filter": example["data_filter"],
                "depth": 2,
            },
        )
        result, task_state = await stage.process(
            task,
            {"fetch_node_name": "fetch"},
            "crawl",
            "link_extract",
        )
        assert isinstance(result, LinkExtractStageResult)
        assert result.status == Status.COMPLETED
        assert result.urls == example["urls"]
        assert task_state is not None
        assert task_state.priority == 1
        assert task_state.context is not None
        assert task_state.context["depth"] == 3
        assert task_state.context["topic"] == example["topic"]
        assert task_state.context["data_filter"] == example["data_filter"]

    run_async(_test())
