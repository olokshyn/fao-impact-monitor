from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import SecretStr

from fao_impact_monitor.config import GeminiConfig
from fao_impact_monitor.pdf_pipeline.gemini import (
    _DOCUMENT_STRUCTURE_SCHEMA,
    _SECTION_EVIDENCE_SCHEMA,
    DOCUMENT_STRUCTURE_PROMPT,
    SECTION_EVIDENCE_PROMPT,
    VISUAL_DESCRIBE_PROMPT,
    VISUAL_VERIFY_PROMPT,
    GeminiPdfClient,
)


class _Models:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses
        self.configs: list[Any] = []
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        self.configs.append(kwargs["config"])
        return self.responses.pop(0)


def _client(responses: list[Any]) -> tuple[GeminiPdfClient, _Models]:
    models = _Models(responses)
    client = GeminiPdfClient.__new__(GeminiPdfClient)
    client.config = GeminiConfig(api_key=SecretStr("test"), model="test-model")
    client.client = cast(Any, SimpleNamespace(models=models))
    return client, models


def test_visual_json_normalizes_top_level_list() -> None:
    client, models = _client(
        [SimpleNamespace(parsed=None, text='[{"text":"Map fact","entailed":true}]')]
    )

    result = asyncio.run(
        client._json(
            [],
            response_json_schema={"type": "object"},
            list_key="facts",
        )
    )

    assert result == {"facts": [{"text": "Map fact", "entailed": True}]}
    assert models.configs[0].response_json_schema == {"type": "object"}


def test_json_retries_malformed_response(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _ = _client(
        [
            SimpleNamespace(parsed=None, text="not json"),
            SimpleNamespace(parsed={"facts": []}, text='{"facts":[]}'),
        ]
    )

    async def no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    result = asyncio.run(client._json([], response_json_schema={"type": "object"}))

    assert result == {"facts": []}


def test_json_accepts_fenced_object() -> None:
    client, _ = _client(
        [SimpleNamespace(parsed=None, text='```json\n{"facts": []}\n```')]
    )

    result = asyncio.run(client._json([], response_json_schema={"type": "object"}))

    assert result == {"facts": []}


def test_structured_schemas_preserve_required_hierarchy_and_evidence_fields() -> None:
    section_properties = _DOCUMENT_STRUCTURE_SCHEMA["properties"]["sections"]["items"][
        "properties"
    ]
    unit_properties = _SECTION_EVIDENCE_SCHEMA["properties"]["evidence_units"]["items"][
        "properties"
    ]

    assert {"title", "page_start", "page_end", "events"} <= section_properties.keys()
    assert {
        "kind",
        "pages",
        "source_text",
        "events",
        "visual_facts",
    } <= unit_properties.keys()
    assert {
        "corrected_page_start",
        "corrected_page_end",
        "boundary_rationale",
    } <= _SECTION_EVIDENCE_SCHEMA["properties"].keys()


def test_section_evidence_prompt_requires_multicolumn_page_break_merge() -> None:
    assert "MULTI-COLUMN CONTINUATIONS" in SECTION_EVIDENCE_PROMPT
    assert "Same-page wraps" in SECTION_EVIDENCE_PROMPT
    assert "rightmost column" in SECTION_EVIDENCE_PROMPT
    assert "leftmost column on the next page" in SECTION_EVIDENCE_PROMPT
    assert "Never emit an orphan" in SECTION_EVIDENCE_PROMPT
    assert "PARENT REGIONAL SECTION HEADINGS" in SECTION_EVIDENCE_PROMPT
    assert "LATIN AMERICA AND THE CARIBBEAN" in SECTION_EVIDENCE_PROMPT
    assert "TABLES:" in SECTION_EVIDENCE_PROMPT
    assert "every column header" in SECTION_EVIDENCE_PROMPT
    assert "Do not stop after the first value column" in SECTION_EVIDENCE_PROMPT
    assert "CHARTS / INFOGRAPHICS:" in SECTION_EVIDENCE_PROMPT
    assert "one atomic fact per labelled bar" in SECTION_EVIDENCE_PROMPT
    assert "MULTI-COLUMN PAGE FLOW" in DOCUMENT_STRUCTURE_PROMPT
    assert "parent_ordinal" in DOCUMENT_STRUCTURE_PROMPT


def test_visual_prompts_require_citable_chart_facts_not_ocr_line_dumps() -> None:
    assert "CITABLE" in VISUAL_DESCRIBE_PROMPT
    assert "NEVER emit facts of the form" in VISUAL_DESCRIBE_PROMPT
    assert "one atomic fact per labelled category" in VISUAL_DESCRIBE_PROMPT
    assert "125.0 million tonnes" in VISUAL_DESCRIBE_PROMPT
    assert "OCR line-dump" in VISUAL_VERIFY_PROMPT
    assert "mid-word truncations" in VISUAL_VERIFY_PROMPT


def test_section_request_maps_excerpt_pages_to_original_pages(tmp_path: Path) -> None:
    response = {
        "corrected_page_start": 3,
        "corrected_page_end": 4,
        "boundary_rationale": "Candidate boundaries confirmed.",
        "scope_units": [],
        "evidence_units": [],
    }
    client, models = _client([SimpleNamespace(parsed=response, text="")])
    excerpt = tmp_path / "section.pdf"
    excerpt.write_bytes(b"PDF excerpt")

    result = asyncio.run(
        client.section_evidence(
            excerpt,
            section_title="Kenya",
            pages=[3, 4],
            included_pages=[2, 3, 4, 5],
        )
    )

    prompt = models.calls[0]["contents"][1]
    assert result == response
    assert "Candidate original physical pages: [3, 4]" in prompt
    assert "Adjacent context pages: [2, 5]" in prompt
    assert "excerpt page 1 = original physical page 2" in prompt
    assert "excerpt page 4 = original physical page 5" in prompt
