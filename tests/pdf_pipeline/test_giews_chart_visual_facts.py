"""Coverage for chart visual-fact extraction on GIEWS cover graphics."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

import pytest

from fao_impact_monitor.config import get_config
from fao_impact_monitor.pdf_pipeline.artifacts import PdfArtifacts, sha256_file
from fao_impact_monitor.pdf_pipeline.gemini import GeminiPdfClient

_TEST_PDF = (
    Path(__file__).resolve().parents[1] / "data" / "giews_crop_prospects_july_2026.pdf"
)
_COVER_PAGE = 1
_LIFDC_TITLE = "Low-Income Food-Deficit Countries cereal production 2026 over 2025"
_LIFDC_CONTEXT = (
    "Cover infographic bar chart titled "
    f"'{_LIFDC_TITLE}' with yearly percentage change callout."
)


def _require_gemini() -> None:
    if not get_config().gemini.api_key.get_secret_value():
        pytest.skip("GEMINI_API_KEY not configured")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip().lower()


def _fact_texts(response: dict[str, Any]) -> list[str]:
    raw = response.get("facts")
    if not isinstance(raw, list):
        return []
    return [
        fact["text"]
        for fact in raw
        if isinstance(fact, dict) and isinstance(fact.get("text"), str)
    ]


def _blob(texts: list[str]) -> str:
    return _normalize("\n".join(texts))


@pytest.mark.integration
def test_giews_page1_lifdc_chart_describe_visual_extracts_bar_values(
    tmp_path: Path,
) -> None:
    """describe_visual on physical page 1 must cite the LIFDC bar chart values."""
    _require_gemini()
    assert _TEST_PDF.is_file(), f"missing test PDF: {_TEST_PDF}"

    artifacts = PdfArtifacts(tmp_path / "artifacts", sha256_file(_TEST_PDF), 144)
    _, pages = artifacts.prepare(_TEST_PDF)
    page = pages[_COVER_PAGE - 1]
    image_path = artifacts.root / page.artifact.relative_path
    assert image_path.is_file()

    client = GeminiPdfClient()
    described = asyncio.run(client.describe_visual(image_path, context=_LIFDC_CONTEXT))
    candidates = _fact_texts(described)
    assert candidates, f"expected visual facts from describe_visual, got {described!r}"

    # Reject the OCR line-dump failure mode that motivated this coverage.
    for text in candidates:
        lowered = text.lower()
        assert "visible line of text" not in lowered, text
        assert not re.search(r"\b(gr|co)\s*$", text.strip(), flags=re.IGNORECASE), text

    verified = asyncio.run(client.verify_visual(image_path, candidates))
    entailed = [
        fact["text"]
        for fact in verified.get("facts", [])
        if isinstance(fact, dict)
        and isinstance(fact.get("text"), str)
        and fact.get("entailed") is True
    ]
    assert entailed, (
        "expected at least one entailed LIFDC chart fact after verify_visual; "
        f"candidates={candidates!r} verified={verified!r}"
    )

    blob = _blob(entailed)
    assert _normalize(_LIFDC_TITLE) in blob, (
        f"missing chart title in verified facts: {entailed!r}"
    )
    for label, value in (
        ("5-year average", "125.0"),
        ("2025", "127.2"),
        ("2026 forecast", "129.0"),
    ):
        assert _normalize(value) in blob, (
            f"missing {label} value {value!r} in verified facts: {entailed!r}"
        )
        assert _normalize(label) in blob, (
            f"missing category label {label!r} in verified facts: {entailed!r}"
        )


@pytest.mark.integration
def test_giews_page1_lifdc_chart_section_evidence_visual_facts(
    tmp_path: Path,
) -> None:
    """section_evidence on page 1 must surface the LIFDC chart as citable visual_facts."""
    _require_gemini()
    assert _TEST_PDF.is_file(), f"missing test PDF: {_TEST_PDF}"

    artifacts = PdfArtifacts(tmp_path / "artifacts", sha256_file(_TEST_PDF), 72)
    artifacts.prepare(_TEST_PDF)
    excerpt = artifacts.section_pdf([_COVER_PAGE])

    analysis = asyncio.run(
        GeminiPdfClient().section_evidence(
            excerpt,
            section_title="Crop Prospects and Food Situation - cover highlights",
            pages=[_COVER_PAGE],
            included_pages=[_COVER_PAGE],
        )
    )
    units = [
        *(analysis.get("scope_units") or []),
        *(analysis.get("evidence_units") or []),
    ]
    chart_units = [
        unit
        for unit in units
        if isinstance(unit, dict)
        and (
            unit.get("modality") in {"chart", "mixed", "diagram"}
            or _normalize(_LIFDC_TITLE)
            in _normalize(
                "\n".join(
                    part
                    for part in (
                        unit.get("source_text"),
                        unit.get("unit_description"),
                        unit.get("retrieval_text"),
                        *(
                            fact.get("text")
                            for fact in (unit.get("visual_facts") or [])
                            if isinstance(fact, dict)
                        ),
                    )
                    if isinstance(part, str)
                )
            )
        )
    ]
    assert chart_units, f"expected LIFDC chart evidence on page 1, got {units!r}"

    blob = _blob(
        [
            text
            for unit in chart_units
            for fact in (unit.get("visual_facts") or [])
            if isinstance(fact, dict)
            for text in [fact.get("text")]
            if isinstance(text, str)
        ]
    )
    if not blob:
        # Fall back to unit text if the model put values only in retrieval fields.
        blob = _blob(
            [
                part
                for unit in chart_units
                for part in (
                    unit.get("source_text"),
                    unit.get("unit_description"),
                    unit.get("retrieval_text"),
                )
                if isinstance(part, str)
            ]
        )

    assert _normalize(_LIFDC_TITLE) in blob, (
        f"missing LIFDC title in chart evidence: {chart_units!r}"
    )
    for value in ("125.0", "127.2", "129.0"):
        assert _normalize(value) in blob, (
            f"missing bar value {value!r} in chart evidence: {chart_units!r}"
        )
