"""Unit and integration tests for the Impact Analyzer Agent."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from fao_impact_monitor.agent.impact_analyzer_agent import (
    CountryFilterList,
    CountryFilterVerdict,
    DraftStatement,
    DraftStatementList,
    StatementVerification,
    _normalize_inline_citations,
    _render_statement_text,
    analyze_impact,
    build_chat_model,
)
from fao_impact_monitor.config import ImpactAnalyzerConfig, get_config
from fao_impact_monitor.impact_report import (
    ParsedMetricReport,
    parse_metric_report_file,
)


def _test_config(**overrides: Any) -> ImpactAnalyzerConfig:
    """Deterministic config for scripted unit tests (no parallel verify races)."""
    return ImpactAnalyzerConfig(verify_concurrency=1, **overrides)


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


def _structured_report(tmp_path: Path) -> ParsedMetricReport:
    plot = _tiny_png(tmp_path / "plots" / "0001-worldbank-1.png")
    path = tmp_path / "0001.md"
    path.write_text(
        f"""## Metric info

Seq Number: 1

Name: Agriculture share of GDP

Description: Share of agriculture in GDP

Example: Agriculture contributed 24.3% of GDP in 2023.

Unit: %

## Direct evidence

### Direct Evidence 1

Source: Agriculture, forestry, and fishing, value added (% of GDP)

Indicator: NV.AGR.TOTL.ZS

Source url: https://data.worldbank.org/indicator/NV.AGR.TOTL.ZS?locations=MW

Latest value: 30.0 % (2025)

Plot: ![{plot.stem}](plots/{plot.name})

## Indirect evidence

None.
""",
        encoding="utf-8",
    )
    return parse_metric_report_file(path)


def _pdf_report(
    tmp_path: Path, *, country_in_text: str = "Malawi"
) -> ParsedMetricReport:
    path = tmp_path / "0016.md"
    path.write_text(
        f"""## Metric info

Seq Number: 16

Name: Crop yield loss

Description: Crop yield loss

Example: Example

Unit: %

## Direct evidence

### Direct Evidence 1

Evidence id: e1

Source: {country_in_text} El Nino impact assessment

Source url: file://fao_data/{country_in_text}-report.pdf

Source physical pages: 2

Source printed pages: 2

Events: el_nino_2023_24, attributed

Source text:

```
In {country_in_text}, maize production recorded a yield loss of 17% below
the five-year average. Current IPC Phase 3 caseload is about 1.9 million people.
```

## Indirect evidence

None.

## Answer

Maize yield fell 17% below the five-year average in {country_in_text}. [1]
""",
        encoding="utf-8",
    )
    return parse_metric_report_file(path)


def test_inline_citations_normalize_and_render() -> None:
    from fao_impact_monitor.impact_report import ParsedEvidence

    known = {
        "m0001-d01": ParsedEvidence(
            evidence_id="m0001-d01",
            kind="direct",
            source_type="worldbank",
            metric_name="GDP",
            metric_seq=1,
            title="GDP",
        ),
        "m0002-d01": ParsedEvidence(
            evidence_id="m0002-d01",
            kind="direct",
            source_type="worldbank",
            metric_name="Labour",
            metric_seq=2,
            title="Labour",
        ),
    }
    text, ids = _normalize_inline_citations(
        "GDP was 30%. [@m0001-d01] Employment was 65%. [@m0002-d01] "
        "GDP share still matters. [@m0001-d01] Junk. [@missing]",
        [],
        known,
    )
    assert "[@missing]" not in text
    assert text.count("[@m0001-d01]") == 2
    assert ids == ["m0001-d01", "m0002-d01"]
    rendered = _render_statement_text(text, ids, {"m0001-d01": 1, "m0002-d01": 2})
    assert rendered is not None
    assert "30%. [1]" in rendered
    assert "65%. [2]" in rendered
    assert "matters. [1]" in rendered


def test_impact_analyzer_config_defaults() -> None:
    cfg = get_config().impact_analyzer
    assert cfg.llm_model == "openai:openai.gpt-5.6-sol"
    assert cfg.filter_llm_model == "openai:openai.gpt-5.6-luna"
    assert cfg.verifier_llm_model == "openai:openai.gpt-5.6-luna"
    assert cfg.reasoning_effort == "high"
    assert cfg.max_answer_verification_retries == 1
    assert cfg.verify_concurrency == 6
    assert cfg.default_plot_detail == "low"
    assert cfg.risk_plot_detail == "high"


def test_build_chat_model_passes_reasoning_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    from langchain_core.language_models.chat_models import BaseChatModel

    class StubChat(BaseChatModel):
        @property
        def _llm_type(self) -> str:
            return "stub"

        def _generate(self, *args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError

    stub = StubChat()

    def fake_init(model_name: str, **kwargs: Any) -> Any:
        captured["model"] = model_name
        captured["kwargs"] = kwargs
        return stub

    monkeypatch.setattr(
        "fao_impact_monitor.agent.impact_analyzer_agent.init_chat_model",
        fake_init,
    )
    model = build_chat_model()
    assert captured["model"] == "openai:openai.gpt-5.6-sol"
    assert captured["kwargs"]["reasoning_effort"] == "high"
    assert captured["kwargs"]["use_responses_api"] is True
    assert model is stub


def test_analyze_impact_scripted_entailed_and_drops_unentailed(
    tmp_path: Path,
) -> None:
    gdp = _structured_report(tmp_path)
    crop = _pdf_report(tmp_path)
    gdp_id = gdp.direct[0].evidence_id
    crop_id = crop.direct[0].evidence_id

    scripts: dict[str, list[Any]] = {
        "CountryFilterList": [
            CountryFilterList(
                verdicts=[
                    CountryFilterVerdict(
                        evidence_id=crop_id,
                        discard=False,
                        reason="about Malawi",
                    )
                ]
            )
        ],
        "DraftStatementList": [
            DraftStatementList(
                statements=[
                    DraftStatement(
                        section="past_impacts",
                        subsection_title=None,
                        text=(
                            "Agriculture accounted for about 30.0% of GDP in 2025 "
                            f"and the series remains elevated. [@{gdp_id}]"
                        ),
                        supporting_evidence_ids=[gdp_id],
                    ),
                    DraftStatement(
                        section="past_impacts",
                        subsection_title="Agriculture and food production",
                        text=(
                            "In 2023-24, maize production recorded a yield loss "
                            f"of 17% below the five-year average. [@{crop_id}]"
                        ),
                        supporting_evidence_ids=[crop_id],
                    ),
                    DraftStatement(
                        section="expected_impacts",
                        subsection_title="Projected risks to agricultural production",
                        text=(
                            "A new drought shock would hit households already facing "
                            "Crisis (IPC Phase 3) for about 1.9 million people. "
                            f"[@{crop_id}] High agricultural GDP share leaves limited "
                            f"buffers. [@{gdp_id}]"
                        ),
                        supporting_evidence_ids=[crop_id, gdp_id],
                    ),
                ]
            )
        ],
        "StatementVerification": [
            StatementVerification(
                statement_id="stmt_001",
                verdict="entailed",
                unsupported_parts=[],
                reasoning="Supported by latest value and plot.",
            ),
            StatementVerification(
                statement_id="stmt_002",
                verdict="entailed",
                unsupported_parts=[],
                reasoning="Supported by source text.",
            ),
            StatementVerification(
                statement_id="stmt_003",
                verdict="entailed",
                unsupported_parts=[],
                reasoning="Cites past and current conditions.",
            ),
        ],
    }
    model = ScriptedModel(scripts)
    output = asyncio.run(
        analyze_impact(
            country_iso3="MWI",
            reports=[gdp, crop],
            config=_test_config(),
            model=model,  # type: ignore[arg-type]
            verifier_model=model,  # type: ignore[arg-type]
            filter_model=model,  # type: ignore[arg-type]
        )
    )
    assert "## Past impacts" in output.markdown
    assert "## Expected impacts" in output.markdown
    assert "30.0%" in output.markdown or "30.0 %" in output.markdown
    assert "17%" in output.markdown
    assert "1.9 million" in output.markdown
    assert "Unsupported preparedness claim" not in output.markdown
    # Citations sit after the sentences that use them, not only as a trailing dump.
    assert "elevated. [1]" in output.markdown
    assert "average. [2]" in output.markdown
    assert "1.9 million people. [2]" in output.markdown
    assert "buffers. [1]" in output.markdown
    assert any("p. 2" in ref for ref in output.references)
    assert any("fao_data/" in ref for ref in output.references)
    assert any("World Bank indicator" in ref for ref in output.references)

    # Plot was attached in draft/verify multimodal messages.
    def _has_image(content: Any) -> bool:
        if isinstance(content, list):
            return any(
                isinstance(block, dict) and block.get("type") == "image_url"
                for block in content
            )
        return False

    assert any(
        _has_image(getattr(message[1], "content", message))
        for message in model.messages
        if isinstance(message, list) and len(message) > 1
    )


def test_analyze_impact_discards_other_country_evidence(tmp_path: Path) -> None:
    malawi = _pdf_report(tmp_path, country_in_text="Malawi")
    kenya_path = tmp_path / "0017.md"
    kenya_path.write_text(
        """## Metric info

Seq Number: 17

Name: Livestock loss

Description: Livestock loss

Example: Example

Unit: %

## Direct evidence

### Direct Evidence 1

Evidence id: ken1

Source: Kenya livestock bulletin

Source url: file://fao_data/kenya.pdf

Source physical pages: 1

Source printed pages: 1

Events: el_nino_2023_24, attributed

Source text:

```
In Kenya, more than 200,000 livestock died during the drought.
```

## Indirect evidence

None.
""",
        encoding="utf-8",
    )
    kenya = parse_metric_report_file(kenya_path)
    kenya_id = kenya.direct[0].evidence_id
    malawi_id = malawi.direct[0].evidence_id

    scripts: dict[str, list[Any]] = {
        "CountryFilterList": [
            CountryFilterList(
                verdicts=[
                    CountryFilterVerdict(
                        evidence_id=malawi_id,
                        discard=False,
                        reason="Malawi",
                    ),
                    CountryFilterVerdict(
                        evidence_id=kenya_id,
                        discard=True,
                        reason="Explicitly about Kenya",
                    ),
                ]
            )
        ],
        "DraftStatementList": [
            DraftStatementList(
                statements=[
                    DraftStatement(
                        section="past_impacts",
                        subsection_title="Agriculture and food production",
                        text=(
                            "Maize yield fell 17% below the five-year average. "
                            f"[@{malawi_id}]"
                        ),
                        supporting_evidence_ids=[malawi_id],
                    ),
                    DraftStatement(
                        section="past_impacts",
                        subsection_title="Agriculture and food production",
                        text=f"More than 200,000 livestock died. [@{kenya_id}]",
                        supporting_evidence_ids=[kenya_id],
                    ),
                ]
            )
        ],
        "StatementVerification": [
            StatementVerification(
                statement_id="stmt_001",
                verdict="entailed",
                unsupported_parts=[],
                reasoning="ok",
            )
        ],
    }
    model = ScriptedModel(scripts)
    output = asyncio.run(
        analyze_impact(
            country_iso3="MWI",
            reports=[malawi, kenya],
            config=_test_config(),
            model=model,  # type: ignore[arg-type]
            verifier_model=model,  # type: ignore[arg-type]
            filter_model=model,  # type: ignore[arg-type]
        )
    )
    assert kenya_id in output.discarded_evidence_ids
    assert "200,000" not in output.markdown
    assert "17%" in output.markdown


def test_missing_plot_fails_analyze(tmp_path: Path) -> None:
    path = tmp_path / "0001.md"
    path.write_text(
        """## Metric info

Seq Number: 1

Name: Agriculture share of GDP

Description: d

Example: e

Unit: %

## Direct evidence

### Direct Evidence 1

Source: Agriculture share of GDP

Indicator: NV.AGR.TOTL.ZS

Plot: ![Agriculture share of GDP](plots/missing.png)

## Indirect evidence

None.
""",
        encoding="utf-8",
    )
    report = parse_metric_report_file(path)
    with pytest.raises(FileNotFoundError, match="Linked plot missing"):
        asyncio.run(
            analyze_impact(
                country_iso3="MWI",
                reports=[report],
                config=_test_config(),
                model=ScriptedModel({}),  # type: ignore[arg-type]
                verifier_model=ScriptedModel({}),  # type: ignore[arg-type]
                filter_model=ScriptedModel({}),  # type: ignore[arg-type]
            )
        )


def test_answer_is_orientation_not_citation(tmp_path: Path) -> None:
    report = _pdf_report(tmp_path)
    evidence_id = report.direct[0].evidence_id
    scripts: dict[str, list[Any]] = {
        "CountryFilterList": [
            CountryFilterList(
                verdicts=[
                    CountryFilterVerdict(
                        evidence_id=evidence_id,
                        discard=False,
                        reason="ok",
                    )
                ]
            )
        ],
        "DraftStatementList": [
            DraftStatementList(
                statements=[
                    DraftStatement(
                        section="past_impacts",
                        subsection_title="Agriculture and food production",
                        text=(
                            "Maize yield fell 17% below the five-year average. "
                            f"[@{evidence_id}]"
                        ),
                        supporting_evidence_ids=[evidence_id],
                    )
                ]
            )
        ],
        "StatementVerification": [
            StatementVerification(
                statement_id="stmt_001",
                verdict="entailed",
                unsupported_parts=[],
                reasoning="from source text",
            )
        ],
    }
    model = ScriptedModel(scripts)
    output = asyncio.run(
        analyze_impact(
            country_iso3="MWI",
            reports=[report],
            config=_test_config(),
            model=model,  # type: ignore[arg-type]
            verifier_model=model,  # type: ignore[arg-type]
            filter_model=model,  # type: ignore[arg-type]
        )
    )
    # Draft prompt should include Answer as orientation.
    draft_messages = [
        message
        for call, message in zip(model.calls, model.messages, strict=False)
        if call == "DraftStatementList"
    ]
    assert draft_messages
    content = draft_messages[0][1].content
    text = content if isinstance(content, str) else str(content)
    assert "direction ONLY" in text or "Answer sections" in text
    assert "Answer" not in "\n".join(output.references)


def _require_bedrock() -> None:
    if not get_config().aws_bedrock.api_key.get_secret_value():
        pytest.skip("AWS_BEDROCK_API_KEY not configured")


@pytest.mark.integration
def test_live_impact_analyzer_fixture_corpus(tmp_path: Path) -> None:
    _require_bedrock()
    gdp = _structured_report(tmp_path)
    labour_plot = _tiny_png(tmp_path / "plots" / "0002-worldbank-1.png")
    labour_path = tmp_path / "0002.md"
    labour_path.write_text(
        f"""## Metric info

Seq Number: 2

Name: Agriculture share of Labour

Description: Employment in agriculture

Example: Example

Unit: %

## Direct evidence

### Direct Evidence 1

Source: Employment in agriculture (% of total employment)

Indicator: SL.AGR.EMPL.ZS

Source url: https://data.worldbank.org/indicator/SL.AGR.EMPL.ZS?locations=MW

Latest value: 65.0 % (2025)

Plot: ![Employment](plots/{labour_plot.name})

## Indirect evidence

None.
""",
        encoding="utf-8",
    )
    irrig_plot = _tiny_png(tmp_path / "plots" / "0003-faostat-1.png")
    irrig_path = tmp_path / "0003.md"
    irrig_path.write_text(
        f"""## Metric info

Seq Number: 3

Name: Irrigated cropland

Description: Irrigation share

Example: Example

Unit: %

## Direct evidence

### Direct Evidence 1

Source: Land area equipped for irrigation; share in cropland (%)

Indicator: Land area equipped for irrigation; share in cropland (%)

Source url: https://www.fao.org/faostat/en/#data/RL

Latest value: 3.67 % (2024)

Plot: ![Irrigation](plots/{irrig_plot.name})

## Indirect evidence

None.
""",
        encoding="utf-8",
    )
    crop = _pdf_report(tmp_path)
    labour = parse_metric_report_file(labour_path)
    irrig = parse_metric_report_file(irrig_path)

    output = asyncio.run(
        analyze_impact(
            country_iso3="MWI",
            reports=[gdp, labour, irrig, crop],
        )
    )
    assert "## Past impacts" in output.markdown
    assert "## Expected impacts" in output.markdown
    assert "## Preparedness Considerations" in output.markdown
    assert "## References" in output.markdown
    # Every body paragraph should be cited.
    for line in output.markdown.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "_")) or stripped[0].isdigit():
            continue
        assert "[" in stripped and "]" in stripped
    assert any("p. " in ref for ref in output.references)


@pytest.mark.integration
def test_live_impact_analyzer_discards_kenya_evidence_for_malawi(
    tmp_path: Path,
) -> None:
    _require_bedrock()
    malawi = _pdf_report(tmp_path, country_in_text="Malawi")
    kenya_path = tmp_path / "0018.md"
    kenya_path.write_text(
        """## Metric info

Seq Number: 18

Name: Livestock loss

Description: Livestock loss

Example: Example

Unit: %

## Direct evidence

### Direct Evidence 1

Evidence id: ken-live

Source: Kenya livestock bulletin

Source url: file://fao_data/kenya-live.pdf

Source physical pages: 4

Source printed pages: 4

Events: el_nino_2015_16, attributed

Source text:

```
In Kenya specifically, more than 200,000 livestock died in southern pastoral areas.
This figure is attributed only to Kenya, not to Malawi.
```

## Indirect evidence

None.
""",
        encoding="utf-8",
    )
    kenya = parse_metric_report_file(kenya_path)
    output = asyncio.run(
        analyze_impact(
            country_iso3="MWI",
            reports=[malawi, kenya],
        )
    )
    assert kenya.direct[0].evidence_id in output.discarded_evidence_ids or (
        "200,000" not in output.markdown
    )
    assert "Kenya specifically" not in output.markdown
