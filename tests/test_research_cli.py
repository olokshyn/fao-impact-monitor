"""Unit tests for research / report / vs-search CLI."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest
from typer.testing import CliRunner

from fao_impact_monitor import pipeline
from fao_impact_monitor.agent.researcher_agent import ResearcherOutput
from fao_impact_monitor.data_source.data_source_config import DataSourceConfig
from fao_impact_monitor.data_source.emdat import EmDatDataResult
from fao_impact_monitor.data_source.faostat import FAOSTATDataResult
from fao_impact_monitor.metric.metric import Metric
from fao_impact_monitor.pipeline import _run_research


def _metric(
    name: str,
    *,
    worldbank: bool = False,
    tags: list[str] | None = None,
) -> Metric:
    if worldbank:
        sources = [
            DataSourceConfig.model_validate(
                {
                    "source": "WorldBank",
                    "indicator": "NV.AGR.TOTL.ZS",
                    "exclusive": True,
                }
            )
        ]
        default_tags = ["structured", "worldbank"]
    else:
        sources = [
            DataSourceConfig.model_validate(
                {"source": "FAORepository", "root_url": "https://example.org"}
            )
        ]
        default_tags = ["fao_repo"]
    return Metric(
        name=name,
        description="Desc",
        example="Example",
        unit="%",
        tags=tags if tags is not None else default_tags,
        data_sources=sources,
    )


def _patch_fao_repo_store(
    monkeypatch: pytest.MonkeyPatch, store: Any | None = None
) -> MagicMock:
    client = MagicMock(close=AsyncMock())
    vector_store = store if store is not None else MagicMock(name="pdf_vector_store")
    monkeypatch.setattr(
        pipeline,
        "_connect_research_mongo",
        AsyncMock(return_value=(client, vector_store)),
    )
    return client


def test_run_research_writes_per_metric_files_in_parallel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = [_metric("A"), _metric("B"), _metric("C", worldbank=True)]
    monkeypatch.setattr(
        "fao_impact_monitor.pipeline.Metric.from_use_case",
        lambda _path: metrics,
    )
    _patch_fao_repo_store(monkeypatch)

    started = asyncio.Event()
    release = asyncio.Event()
    concurrent = {"n": 0, "max": 0}

    async def fake_research(**kwargs: Any) -> ResearcherOutput:
        del kwargs
        concurrent["n"] += 1
        concurrent["max"] = max(concurrent["max"], concurrent["n"])
        started.set()
        await release.wait()
        concurrent["n"] -= 1
        return ResearcherOutput(
            status="answered",
            country="Kenya",
            metric_name="x",
            final_summary="ok",
            statements=[],
            claims=[],
            sources=[],
            open_gaps=[],
            research_iterations=1,
        )

    monkeypatch.setattr("fao_impact_monitor.pipeline.research", fake_research)

    async def fake_wb_get_data(*_args: Any, **_kwargs: Any) -> list[Any]:
        return []

    world_bank = MagicMock(get_data=fake_wb_get_data)
    monkeypatch.setattr(
        "fao_impact_monitor.pipeline.get_data_source",
        lambda _source: world_bank,
    )

    output_dir = tmp_path / "el-nino-KEN.md"

    async def run() -> Path:
        task = asyncio.create_task(
            _run_research(
                use_case_path=Path("use-cases/el-nino.json"),
                country_iso3="KEN",
                metric_indices=[1, 2, 3],
                output_dir=output_dir,
                max_parallel=2,
            )
        )
        await started.wait()
        await asyncio.sleep(0.05)
        assert concurrent["max"] >= 2
        release.set()
        return await task

    result_dir = asyncio.run(run())
    assert result_dir == output_dir
    assert (output_dir / "0001.md").is_file()
    assert (output_dir / "0002.md").is_file()
    assert (output_dir / "0003.md").is_file()
    assert (output_dir / "0001-H.md").is_file()
    assert (output_dir / "0002-H.md").is_file()
    assert not (output_dir / "0003-H.md").is_file()
    assert "A" in (output_dir / "0001.md").read_text(encoding="utf-8")
    assert "# 1. A" in (output_dir / "0001-H.md").read_text(encoding="utf-8")
    assert "C" in (output_dir / "0003.md").read_text(encoding="utf-8")


def test_run_research_loads_faostat_and_writes_plot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metric = Metric(
        name="Maize production",
        description="Annual maize production.",
        example="Production reached 4.2 million tonnes.",
        unit="t",
        tags=["structured", "faostat"],
        data_sources=[
            DataSourceConfig.model_validate(
                {
                    "source": "FAOSTAT",
                    "dataset": "Crops and livestock products",
                    "indicator": "Maize production",
                    "exclusive": True,
                }
            )
        ],
    )
    monkeypatch.setattr(
        "fao_impact_monitor.pipeline.Metric.from_use_case",
        lambda _path: [metric],
    )

    async def fake_get_data(*_args: Any, **_kwargs: Any) -> list[Any]:
        return [
            FAOSTATDataResult(
                source="FAOSTAT",
                title="Maize production",
                url="https://www.fao.org/faostat/en/#data/QCL",
                citation="cite",
                metadata={"indicator": "Maize production", "country_iso3": "KEN"},
                data=pd.DataFrame(
                    {
                        "year": [2021, 2022, 2023],
                        "value": [1.0, 2.0, 3.0],
                        "item": ["Maize"] * 3,
                        "element": ["Production"] * 3,
                        "unit": ["t"] * 3,
                    }
                ),
            )
        ]

    monkeypatch.setattr(
        "fao_impact_monitor.pipeline.get_data_source",
        lambda _source: MagicMock(get_data=fake_get_data),
    )

    output_dir = tmp_path / "reports"
    asyncio.run(
        _run_research(
            use_case_path=tmp_path / "case.json",
            country_iso3="KEN",
            metric_indices=None,
            output_dir=output_dir,
            max_parallel=1,
            use_fao_repo=False,
        )
    )
    report = (output_dir / "0001.md").read_text(encoding="utf-8")
    assert "Maize production" in report
    assert (output_dir / "plots" / "0001-faostat-1.png").is_file()


def test_run_research_loads_emdat_and_writes_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metric = Metric(
        name="Deaths and missing persons",
        description="Deaths from hazards.",
        example="59 people died.",
        unit="persons",
        tags=["structured", "undrr", "emdat"],
        data_sources=[
            DataSourceConfig.model_validate(
                {
                    "source": "EMDAT",
                    "indicator": "Total Deaths",
                    "unit": "persons",
                    "exclusive": True,
                }
            )
        ],
    )
    result = EmDatDataResult(
        source="EMDAT",
        title="Total Deaths",
        url="https://www.emdat.be",
        citation="cite",
        metadata={
            "indicator": "Total Deaths",
            "country_iso3": "KEN",
            "unit": "persons",
        },
        data=pd.DataFrame(
            {
                "start_year": [2023],
                "disaster_type": ["Flood"],
                "location": ["Nairobi"],
                "value": [178.0],
                "regions": [["Nairobi"]],
            }
        ),
    )
    source = MagicMock(get_data=AsyncMock(return_value=[result]))
    monkeypatch.setattr(
        "fao_impact_monitor.pipeline.Metric.from_use_case",
        lambda _path: [metric],
    )
    monkeypatch.setattr(
        "fao_impact_monitor.pipeline.get_data_source",
        lambda source_name: source if source_name == "EMDAT" else None,
    )

    output_dir = tmp_path / "reports"
    asyncio.run(
        _run_research(
            use_case_path=tmp_path / "case.json",
            country_iso3="KEN",
            metric_indices=None,
            output_dir=output_dir,
            max_parallel=1,
            use_fao_repo=False,
        )
    )

    source.get_data.assert_awaited_once()
    report = (output_dir / "0001.md").read_text(encoding="utf-8")
    assert "## Metric info" in report
    assert "Source: Total Deaths" in report
    assert "Dataset: EM-DAT" in report
    assert (
        "| Start Year | Disaster Type | Location | Regions | Value | Unit |" in report
    )
    assert "| 2023 | Flood | Nairobi | Nairobi | 178 | persons |" in report
    assert "![Total Deaths](plots/0001-emdat-1.png)" in report
    assert "## Answer" not in report
    assert "## References" in report


def test_run_research_continue_skips_existing_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = [
        _metric("A", worldbank=True),
        _metric("B", worldbank=True),
        _metric("C", worldbank=True),
    ]
    monkeypatch.setattr(
        "fao_impact_monitor.pipeline.Metric.from_use_case",
        lambda _path: metrics,
    )
    selected: list[int] = []

    async def fake_run_one_metric(**kwargs: Any) -> Path:
        selected.append(kwargs["index"])
        return cast(Path, kwargs["output_dir"]) / f"{kwargs['index']:04d}.md"

    monkeypatch.setattr(pipeline, "_run_one_metric", fake_run_one_metric)

    output_dir = tmp_path / "reports"
    output_dir.mkdir()
    (output_dir / "0001.md").write_text("done\n", encoding="utf-8")

    asyncio.run(
        _run_research(
            use_case_path=tmp_path / "case.json",
            country_iso3="KEN",
            metric_indices=None,
            output_dir=output_dir,
            max_parallel=1,
            use_fao_repo=False,
            continue_incomplete=True,
        )
    )
    assert selected == [2, 3]


def test_research_cli_forwards_tags_and_countries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_case = Path("use-cases/el-nino.json")
    calls: list[dict[str, Any]] = []

    async def fake_run_research(**kwargs: Any) -> Path:
        calls.append(dict(kwargs))
        return cast(Path, kwargs["output_dir"])

    monkeypatch.setattr(pipeline, "_run_research", fake_run_research)

    result = CliRunner().invoke(
        pipeline.app,
        [
            "research",
            "--countries",
            "ETH,KEN",
            "--use-case",
            str(use_case),
            "--tags",
            "undrr",
        ],
    )

    assert result.exit_code == 0, result.output
    assert [call["country_iso3"] for call in calls] == ["ETH", "KEN"]
    assert all(call["metric_indices"] == [26, 27, 28, 29, 30, 31] for call in calls)
    assert all(call["use_fao_repo"] is True for call in calls)
    assert all(call["use_tellus"] is False for call in calls)


def test_research_cli_forwards_faostat_tag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_case = Path("use-cases/el-nino.json")
    captured: dict[str, Any] = {}

    async def fake_run_research(**kwargs: Any) -> Path:
        captured.update(kwargs)
        return cast(Path, kwargs["output_dir"])

    monkeypatch.setattr(pipeline, "_run_research", fake_run_research)

    result = CliRunner().invoke(
        pipeline.app,
        [
            "research",
            "--countries",
            "ken",
            "--use-case",
            str(use_case),
            "--tags",
            "faostat",
            "--no-fao-repo",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["country_iso3"] == "KEN"
    assert captured["metric_indices"] == list(range(3, 14))
    assert captured["use_fao_repo"] is False
    assert captured["output_dir"] == Path("reports/el-nino/KEN")


def test_research_cli_dry_run_prints_plan_without_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    async def fake_run_research(**kwargs: Any) -> Path:
        del kwargs
        nonlocal called
        called = True
        return Path("reports")

    monkeypatch.setattr(pipeline, "_run_research", fake_run_research)
    result = CliRunner().invoke(
        pipeline.app,
        [
            "research",
            "--countries",
            "KEN",
            "--use-case",
            "use-cases/el-nino.json",
            "--tags",
            "worldbank",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Dry-run research plan" in result.output
    assert "Agriculture share of GDP" in result.output
    assert called is False


def test_research_cli_countries_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_case = tmp_path / "case.json"
    use_case.write_text(
        '{"metrics": [{"name": "A", "description": "d", "example": "e"}]}',
        encoding="utf-8",
    )
    countries_file = tmp_path / "countries.txt"
    countries_file.write_text("ETH\nKEN,MWI\n", encoding="utf-8")
    calls: list[str] = []

    async def fake_run_research(**kwargs: Any) -> Path:
        calls.append(str(kwargs["country_iso3"]))
        return cast(Path, kwargs["output_dir"])

    monkeypatch.setattr(pipeline, "_run_research", fake_run_research)
    result = CliRunner().invoke(
        pipeline.app,
        [
            "research",
            "--countries-file",
            str(countries_file),
            "--countries",
            "FJI",
            "--use-case",
            str(use_case),
            "--output-root",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls == ["ETH", "KEN", "MWI", "FJI"]


def test_research_cli_continue_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_case = tmp_path / "case.json"
    use_case.write_text(
        '{"metrics": [{"name": "A", "description": "d", "example": "e"}]}',
        encoding="utf-8",
    )
    captured: dict[str, Any] = {}

    async def fake_run_research(**kwargs: Any) -> Path:
        captured.update(kwargs)
        return cast(Path, kwargs["output_dir"])

    monkeypatch.setattr(pipeline, "_run_research", fake_run_research)
    result = CliRunner().invoke(
        pipeline.app,
        [
            "research",
            "--countries",
            "eth",
            "--use-case",
            str(use_case),
            "--continue",
            "--tellus",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["continue_incomplete"] is True
    assert captured["use_tellus"] is True


def test_run_research_uses_fao_repo_vector_store_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metric = _metric("PDF evidence metric")
    store = MagicMock(name="pdf_vector_store")
    client = _patch_fao_repo_store(monkeypatch, store)
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        "fao_impact_monitor.pipeline.Metric.from_use_case",
        lambda _path: [metric],
    )

    async def fake_research(**kwargs: Any) -> ResearcherOutput:
        captured.update(kwargs)
        return ResearcherOutput(
            status="answered",
            country="Kenya",
            metric_name=metric.name,
            final_summary="PDF result",
            statements=[],
            claims=[],
            sources=[],
            open_gaps=[],
            research_iterations=1,
        )

    monkeypatch.setattr("fao_impact_monitor.pipeline.research", fake_research)

    asyncio.run(
        _run_research(
            use_case_path=tmp_path / "case.json",
            country_iso3="KEN",
            metric_indices=None,
            output_dir=tmp_path / "reports",
            max_parallel=1,
            web_research_enabled=False,
        )
    )

    assert captured["vector_store"] is store
    assert captured["web_research_enabled"] is False
    client.close.assert_awaited_once_with()


def test_run_research_ensures_indexes_when_flag_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENSURE_INDEXES", "true")
    metric = _metric("PDF evidence metric")
    ensure_indexes = AsyncMock()
    client = MagicMock(close=AsyncMock())
    store = MagicMock()

    async def fake_connect(**kwargs: Any) -> tuple[Any, Any]:
        if kwargs.get("ensure_indexes"):
            await ensure_indexes()
        return client, store

    monkeypatch.setattr(
        "fao_impact_monitor.pipeline.Metric.from_use_case",
        lambda _path: [metric],
    )
    monkeypatch.setattr(pipeline, "_connect_research_mongo", fake_connect)
    monkeypatch.setattr(
        "fao_impact_monitor.pipeline.research",
        AsyncMock(
            return_value=ResearcherOutput(
                status="answered",
                country="Kenya",
                metric_name=metric.name,
                final_summary="PDF result",
                statements=[],
                claims=[],
                sources=[],
                open_gaps=[],
                research_iterations=1,
            )
        ),
    )

    asyncio.run(
        _run_research(
            use_case_path=tmp_path / "case.json",
            country_iso3="KEN",
            metric_indices=None,
            output_dir=tmp_path / "reports",
            max_parallel=1,
        )
    )
    ensure_indexes.assert_awaited_once_with()


def test_report_cli_default_impact_and_human(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_case = tmp_path / "el-nino.json"
    use_case.write_text('{"name": "El Niño", "metrics": []}', encoding="utf-8")
    calls: list[str] = []

    async def fake_impact(**kwargs: Any) -> list[Path]:
        del kwargs
        calls.append("impact")
        return []

    async def fake_undrr(**kwargs: Any) -> list[Path]:
        del kwargs
        calls.append("undrr")
        return []

    async def fake_pdf(**kwargs: Any) -> Path:
        calls.append("human" if kwargs.get("human") else "technical")
        return tmp_path / "out.pdf"

    monkeypatch.setattr(pipeline, "_run_impact_for_country", fake_impact)
    monkeypatch.setattr(pipeline, "_run_undrr_for_country", fake_undrr)
    monkeypatch.setattr(pipeline, "_run_metrics_pdf_for_country", fake_pdf)

    result = CliRunner().invoke(
        pipeline.app,
        ["report", "--countries", "ETH", "--use-case", str(use_case)],
    )
    assert result.exit_code == 0, result.output
    assert calls == ["impact", "human"]


def test_report_cli_explicit_flags_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_case = tmp_path / "el-nino.json"
    use_case.write_text('{"name": "El Niño", "metrics": []}', encoding="utf-8")
    calls: list[str] = []

    async def fake_impact(**kwargs: Any) -> list[Path]:
        del kwargs
        calls.append("impact")
        return []

    async def fake_undrr(**kwargs: Any) -> list[Path]:
        del kwargs
        calls.append("undrr")
        return []

    async def fake_pdf(**kwargs: Any) -> Path:
        calls.append("human" if kwargs.get("human") else "technical")
        return tmp_path / "out.pdf"

    monkeypatch.setattr(pipeline, "_run_impact_for_country", fake_impact)
    monkeypatch.setattr(pipeline, "_run_undrr_for_country", fake_undrr)
    monkeypatch.setattr(pipeline, "_run_metrics_pdf_for_country", fake_pdf)

    result = CliRunner().invoke(
        pipeline.app,
        [
            "report",
            "--countries",
            "ETH",
            "--use-case",
            str(use_case),
            "--technical",
            "--undrr",
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls == ["undrr", "technical"]


def test_report_cli_impact_writes_markdown_and_pdf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_case = tmp_path / "el-nino.json"
    use_case.write_text('{"name": "El Niño", "metrics": []}', encoding="utf-8")
    input_root = tmp_path / "reports"
    country_dir = input_root / "el-nino" / "ETH"
    country_dir.mkdir(parents=True)
    (country_dir / "0001.md").write_text(
        "## Metric info\n\nSeq Number: 1\n\nName: GDP\n\n"
        "Description: d\n\nExample: e\n\nUnit: %\n\n"
        "## Direct evidence\n\nNone.\n\n## Indirect evidence\n\nNone.\n",
        encoding="utf-8",
    )

    from fao_impact_monitor.agent.impact_analyzer_agent import ImpactAnalyzerOutput

    async def fake_analyze(**kwargs: Any) -> ImpactAnalyzerOutput:
        del kwargs
        return ImpactAnalyzerOutput(
            country="Ethiopia",
            country_iso3="ETH",
            markdown=(
                "# Ethiopia: El Niño Risk Outlook\n\n"
                "## Past impacts\n\nRisk is high. [1]\n\n"
                "## Expected impacts\n\nOutlook is negative. [1]\n\n"
                "## Preparedness Considerations\n\nActions exist. [1]\n\n"
                "## References\n\n1. Example ref\n"
            ),
            statements=[],
            references=["Example ref"],
            discarded_evidence_ids=[],
        )

    monkeypatch.setattr(pipeline, "analyze_impact", fake_analyze)
    monkeypatch.setattr(pipeline, "enrich_structured_latest_values", AsyncMock())
    monkeypatch.setattr(
        pipeline,
        "write_impact_analysis_files",
        lambda **kwargs: (
            kwargs["output_md"],
            kwargs["output_md"].with_suffix(".pdf"),
        ),
    )

    result = CliRunner().invoke(
        pipeline.app,
        [
            "report",
            "--countries",
            "ETH",
            "--use-case",
            str(use_case),
            "--input-root",
            str(input_root),
            "--output-root",
            str(input_root),
            "--impact",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "ETH impact analysis El Nino.md" in result.output
    assert "ETH impact analysis El Nino.pdf" in result.output


def test_report_cli_undrr_writes_markdown_and_pdf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_case = tmp_path / "el-nino.json"
    use_case.write_text(
        """{
          "name": "El Niño",
          "metrics": [
            {
              "name": "Deaths and missing persons",
              "description": "d",
              "example": "e",
              "unit": "persons",
              "tags": ["structured", "undrr", "emdat"],
              "data_sources": [
                {"source": "EMDAT", "exclusive": true, "config": {}}
              ]
            }
          ]
        }""",
        encoding="utf-8",
    )
    input_root = tmp_path / "reports"
    country_dir = input_root / "el-nino" / "ETH"
    country_dir.mkdir(parents=True)
    (country_dir / "0001.md").write_text(
        "## Metric info\n\nSeq Number: 1\n\nName: Deaths and missing persons\n\n"
        "Description: d\n\nExample: e\n\nUnit: persons\n\n"
        "## Direct evidence\n\nNone.\n\n## Indirect evidence\n\nNone.\n",
        encoding="utf-8",
    )

    from fao_impact_monitor.agent.undrr_summarizer_agent import UndrrSummarizerOutput

    async def fake_summarize(**kwargs: Any) -> UndrrSummarizerOutput:
        assert "data_filter" in kwargs
        return UndrrSummarizerOutput(
            country="Ethiopia",
            country_iso3="ETH",
            use_case_ascii="El Nino",
            markdown=(
                "# Ethiopia: UNDRR El Nino\n\n"
                "## Deaths and missing persons\n\n"
                "Floods killed 120 people. [1]\n\n"
                "## References\n\n1. EM-DAT Total Deaths\n"
            ),
            statements=[],
            references=["EM-DAT Total Deaths"],
        )

    monkeypatch.setattr(pipeline, "summarize_undrr", fake_summarize)
    monkeypatch.setattr(
        pipeline,
        "write_undrr_report_files",
        lambda **kwargs: (
            kwargs["output_md"],
            kwargs["output_md"].with_suffix(".pdf"),
        ),
    )

    result = CliRunner().invoke(
        pipeline.app,
        [
            "report",
            "--countries",
            "ETH",
            "--use-case",
            str(use_case),
            "--input-root",
            str(input_root),
            "--output-root",
            str(input_root),
            "--undrr",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "ETH UNDRR El Nino.md" in result.output
    assert "ETH UNDRR El Nino.pdf" in result.output


def test_report_cli_human_forwards_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_case = tmp_path / "el-nino.json"
    use_case.write_text('{"name": "El Niño", "metrics": []}', encoding="utf-8")
    captured: dict[str, Any] = {}

    async def fake_pdf(**kwargs: Any) -> Path:
        captured.update(kwargs)
        return tmp_path / "out.pdf"

    monkeypatch.setattr(pipeline, "_run_metrics_pdf_for_country", fake_pdf)
    result = CliRunner().invoke(
        pipeline.app,
        [
            "report",
            "--countries",
            "ETH",
            "--use-case",
            str(use_case),
            "--input-root",
            str(tmp_path),
            "--human",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["human"] is True
    assert captured["country_iso3"] == "ETH"


def test_vs_search_cli_forwards_query_country_and_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_search(**kwargs: Any) -> list[dict[str, Any]]:
        captured.update(kwargs)
        return []

    monkeypatch.setattr(pipeline, "_run_vs_search", fake_search)

    result = CliRunner().invoke(
        pipeline.app,
        [
            "vs-search",
            "drought Kenya",
            "--countries",
            "ken",
            "--limit",
            "7",
            "--tellus",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["query"] == "drought Kenya"
    assert captured["countries_iso3"] == ["KEN"]
    assert captured["limit"] == 7
    assert captured["use_fao_repo"] is True
    assert captured["use_tellus"] is True
