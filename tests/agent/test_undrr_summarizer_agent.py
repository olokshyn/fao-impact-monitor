"""Unit tests for the UNDRR summarizer agent."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from fao_impact_monitor.agent.undrr_summarizer_agent import (
    StatementRepair,
    StatementVerification,
    UndrrDraftAnswer,
    summarize_undrr,
)
from fao_impact_monitor.config import UndrrSummarizerConfig
from fao_impact_monitor.impact_report import (
    ParsedMetricReport,
    parse_metric_report_file,
)


def _test_config(**overrides: Any) -> UndrrSummarizerConfig:
    return UndrrSummarizerConfig(verify_concurrency=1, **overrides)


def _tiny_png(path: Path) -> Path:
    data = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f"
        b"\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


class ScriptedModel:
    def __init__(self, scripts: dict[str, list[Any]]) -> None:
        self.scripts = {key: list(values) for key, values in scripts.items()}
        self.calls: list[str] = []
        self.messages: list[Any] = []
        self._lock = asyncio.Lock()

    def with_structured_output(self, schema: Any, **_kwargs: Any) -> Any:
        name = getattr(schema, "__name__", str(schema))
        parent = self

        class Structured:
            async def ainvoke(self, messages: Any) -> Any:
                async with parent._lock:
                    parent.calls.append(name)
                    parent.messages.append(messages)
                    queue = parent.scripts.get(name, [])
                    if not queue:
                        raise AssertionError(f"No scripted response for {name}")
                    return queue.pop(0)

        return Structured()


def _desinventar_report(tmp_path: Path) -> ParsedMetricReport:
    plot = _tiny_png(tmp_path / "plots" / "0030-desinventar-1.png")
    path = tmp_path / "0030.md"
    path.write_text(
        f"""## Metric info

Seq Number: 30

Name: Education and health facilities affected

Description: How many education and health facilities were affected.

Example: Landslides damaged schools recorded in DesInventar.

Unit: facilities

## Direct evidence

### Direct Evidence 1

Source: nescuelas

Dataset: DesInventar

Indicator: nescuelas

Source data:

| Event Id | Start Year | Disaster Type | Regions | Value | Unit |
| --- | --- | --- | --- | --- | --- |
| 1 | 2022 | FLOOD | Somali | 85 | facilities |
| 2 | 2024 | FLOOD | Oromiya | 31 | facilities |

Plot: ![{plot.stem}](plots/{plot.name})

## Indirect evidence

None.
""",
        encoding="utf-8",
    )
    return parse_metric_report_file(path)


def _emdat_report(tmp_path: Path) -> ParsedMetricReport:
    plot = _tiny_png(tmp_path / "plots" / "0026-emdat-1.png")
    path = tmp_path / "0026.md"
    path.write_text(
        f"""## Metric info

Seq Number: 26

Name: Deaths and missing persons

Description: How many people died or went missing due to climate hazards.

Example: Floods killed 120 people according to EM-DAT.

Unit: persons

## Direct evidence

### Direct Evidence 1

Source: Total Deaths

Dataset: EM-DAT

Indicator: Total Deaths

Source data:

| Year | Disaster Type | Total Deaths |
| --- | --- | --- |
| 2023 | Flood | 120 |
| 2024 | Drought | 40 |

Plot: ![{plot.stem}](plots/{plot.name})

## Indirect evidence

None.
""",
        encoding="utf-8",
    )
    return parse_metric_report_file(path)


def test_summarize_undrr_scripted_draft_and_verify(tmp_path: Path) -> None:
    deaths = _emdat_report(tmp_path)
    facilities = _desinventar_report(tmp_path)
    death_id = deaths.direct[0].evidence_id
    facility_id = facilities.direct[0].evidence_id

    assert deaths.direct[0].source_type == "structured"
    assert deaths.direct[0].dataset == "EM-DAT"
    assert deaths.direct[0].source_data is not None
    assert "120" in deaths.direct[0].source_data
    assert facilities.direct[0].dataset == "DesInventar"

    scripts: dict[str, list[Any]] = {
        "UndrrDraftAnswer": [
            UndrrDraftAnswer(
                text=(
                    "Floods killed 120 people in 2023 according to EM-DAT "
                    f"records. [@{death_id}]"
                ),
                supporting_evidence_ids=[death_id],
            ),
            UndrrDraftAnswer(
                text=(
                    "Floods damaged 85 schools in Somali in 2022 according to "
                    f"DesInventar. [@{facility_id}]"
                ),
                supporting_evidence_ids=[facility_id],
            ),
        ],
        "StatementVerification": [
            StatementVerification(
                statement_id="m0026",
                score=8,
                unsupported_parts=[],
                reasoning="Supported by EM-DAT table.",
            ),
            StatementVerification(
                statement_id="m0030",
                score=7,
                unsupported_parts=[],
                reasoning="Supported by DesInventar table.",
            ),
        ],
    }
    model = ScriptedModel(scripts)
    output = asyncio.run(
        summarize_undrr(
            country_iso3="ETH",
            reports=[deaths, facilities],
            use_case_name="El Nino",
            config=_test_config(),
            model=model,  # type: ignore[arg-type]
            verifier_model=model,  # type: ignore[arg-type]
        )
    )

    assert "UNDRR El Nino" in output.markdown
    assert output.markdown.startswith("# ")
    assert "## Deaths and missing persons" in output.markdown
    assert "## Education and health facilities affected" in output.markdown
    assert "## References" in output.markdown
    assert "120" in output.markdown
    assert "85" in output.markdown
    assert "[1]" in output.markdown
    assert "[2]" in output.markdown
    assert "Uncited claim without evidence" not in output.markdown
    assert any("EM-DAT" in ref for ref in output.references)
    assert any("DesInventar" in ref for ref in output.references)

    # No uncited narrative body lines outside placeholders/headings.
    for line in output.markdown.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("_") and stripped.endswith("_"):
            continue
        if stripped.startswith("## References") or stripped[0].isdigit():
            continue
        if stripped == "None.":
            continue
        assert "[" in stripped and "]" in stripped, stripped


def test_summarize_undrr_keeps_scores_above_five_only(tmp_path: Path) -> None:
    deaths = _emdat_report(tmp_path)
    facilities = _desinventar_report(tmp_path)
    death_id = deaths.direct[0].evidence_id
    facility_id = facilities.direct[0].evidence_id

    scripts: dict[str, list[Any]] = {
        "UndrrDraftAnswer": [
            UndrrDraftAnswer(
                text=f"Floods killed 120 people. [@{death_id}]",
                supporting_evidence_ids=[death_id],
            ),
            UndrrDraftAnswer(
                text=f"Floods damaged 85 schools. [@{facility_id}]",
                supporting_evidence_ids=[facility_id],
            ),
        ],
        "StatementVerification": [
            StatementVerification(
                statement_id="m0026",
                score=5,
                unsupported_parts=["totals unclear"],
                reasoning="Borderline support.",
            ),
            StatementVerification(
                statement_id="m0026",
                score=5,
                unsupported_parts=["still weak"],
                reasoning="Still borderline after repair.",
            ),
            StatementVerification(
                statement_id="m0030",
                score=6,
                unsupported_parts=[],
                reasoning="Adequate support.",
            ),
        ],
        "StatementRepair": [
            StatementRepair(
                statement_id="m0026",
                text=f"Floods killed about 120 people. [@{death_id}]",
                supporting_evidence_ids=[death_id],
            ),
        ],
    }
    model = ScriptedModel(scripts)
    output = asyncio.run(
        summarize_undrr(
            country_iso3="ETH",
            reports=[deaths, facilities],
            use_case_name="El Nino",
            config=_test_config(max_answer_verification_retries=1),
            model=model,  # type: ignore[arg-type]
            verifier_model=model,  # type: ignore[arg-type]
        )
    )
    assert "85" in output.markdown
    assert "120" not in output.markdown
    assert "_No supported evidence._" in output.markdown


def test_summarize_undrr_requires_reports() -> None:
    with pytest.raises(ValueError, match="No UNDRR metric reports"):
        asyncio.run(
            summarize_undrr(
                country_iso3="ETH",
                reports=[],
                use_case_name="El Nino",
                config=_test_config(),
                model=ScriptedModel({}),  # type: ignore[arg-type]
                verifier_model=ScriptedModel({}),  # type: ignore[arg-type]
            )
        )
