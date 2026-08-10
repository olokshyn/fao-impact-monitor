"""Unit tests for research CLI parallel per-metric report writing."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest
from typer.testing import CliRunner

from fao_impact_monitor import pipeline
from fao_impact_monitor.agent.researcher_agent import ResearcherOutput
from fao_impact_monitor.data_source.data_source_config import DataSourceConfig
from fao_impact_monitor.data_source.faostat import FAOSTATDataResult
from fao_impact_monitor.metric.metric import Metric
from fao_impact_monitor.pipeline import (
    _run_pdf_research_all_use_cases,
    _run_research,
)


def _metric(name: str, *, worldbank: bool = False) -> Metric:
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
    else:
        sources = [
            DataSourceConfig.model_validate(
                {"source": "FAORepository", "root_url": "https://example.org"}
            )
        ]
    return Metric(
        name=name,
        description="Desc",
        example="Example",
        unit="%",
        data_sources=sources,
    )


def test_run_research_writes_per_metric_files_in_parallel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = [_metric("A"), _metric("B"), _metric("C", worldbank=True)]
    monkeypatch.setattr(
        "fao_impact_monitor.pipeline.Metric.from_use_case",
        lambda _path: metrics,
    )

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
    monkeypatch.setattr(
        "fao_impact_monitor.pipeline.connect_data_lake",
        AsyncMock(return_value=MagicMock(close=AsyncMock())),
    )
    monkeypatch.setattr(
        "fao_impact_monitor.pipeline.VectorStore",
        lambda: MagicMock(),
    )

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
        # Allow the second researcher task to enter the semaphore.
        await asyncio.sleep(0.05)
        assert concurrent["max"] >= 2
        release.set()
        return await task

    result_dir = asyncio.run(run())
    assert result_dir == output_dir
    assert (output_dir / "0001.md").is_file()
    assert (output_dir / "0002.md").is_file()
    assert (output_dir / "0003.md").is_file()
    assert "A" in (output_dir / "0001.md").read_text(encoding="utf-8")
    assert "B" in (output_dir / "0002.md").read_text(encoding="utf-8")
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
        data_sources=[
            DataSourceConfig.model_validate(
                {
                    "source": "FAOSTAT",
                    "indicator": "Crop and livestock products",
                    "exclusive": True,
                }
            )
        ],
    )
    result = FAOSTATDataResult(
        source="FAOSTAT",
        title="Crop and livestock products",
        url="https://www.fao.org/faostat/en/#data/QCL",
        citation="cite",
        metadata={"indicator": "Crop and livestock products", "country_iso3": "KEN"},
        data=pd.DataFrame(
            {
                "year": [2021, 2022, 2023],
                "value": [3_800_000.0, 4_200_000.0, 4_050_000.0],
                "item": ["Maize (corn)"] * 3,
                "element": ["Production"] * 3,
                "unit": ["t"] * 3,
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
        lambda source_name: source if source_name == "FAOSTAT" else None,
    )

    output_dir = tmp_path / "reports"
    asyncio.run(
        _run_research(
            use_case_path=tmp_path / "case.json",
            country_iso3="KEN",
            metric_indices=None,
            output_dir=output_dir,
            max_parallel=1,
        )
    )

    source.get_data.assert_awaited_once()
    report = (output_dir / "0001.md").read_text(encoding="utf-8")
    assert "![Crop and livestock products](plots/0001-faostat-1.png)" in report
    assert "| Year | Value |" not in report
    assert (output_dir / "plots" / "0001-faostat-1.png").is_file()


def test_run_research_filters_metrics_by_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    faostat_metric = Metric(
        name="FAOSTAT metric",
        description="Desc",
        example="Example",
        data_sources=[DataSourceConfig(source="FAOSTAT", exclusive=True)],
    )
    metrics = [
        _metric("Repository metric"),
        _metric("World Bank", worldbank=True),
        faostat_metric,
    ]
    selected: list[tuple[int, str]] = []
    monkeypatch.setattr(
        "fao_impact_monitor.pipeline.Metric.from_use_case",
        lambda _path: metrics,
    )

    async def fake_run_one_metric(**kwargs: Any) -> Path:
        selected.append((kwargs["index"], kwargs["metric"].name))
        return kwargs["output_dir"] / f"{kwargs['index']:04d}.md"

    monkeypatch.setattr(
        "fao_impact_monitor.pipeline._run_one_metric",
        fake_run_one_metric,
    )

    asyncio.run(
        _run_research(
            use_case_path=tmp_path / "case.json",
            country_iso3="KEN",
            metric_indices=None,
            output_dir=tmp_path / "reports",
            max_parallel=1,
            data_source="faostat",
        )
    )

    assert selected == [(3, "FAOSTAT metric")]


def test_research_cli_forwards_source_filter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_case = tmp_path / "case.json"
    use_case.write_text("{}", encoding="utf-8")
    captured: dict[str, Any] = {}

    async def fake_run_research(**kwargs: Any) -> Path:
        captured.update(kwargs)
        return kwargs["output_dir"]

    monkeypatch.setattr(pipeline, "_run_research", fake_run_research)

    result = CliRunner().invoke(
        pipeline.app,
        [
            "research",
            "--country",
            "ken",
            "--use-case",
            str(use_case),
            "--source",
            "FAOSTAT",
        ],
    )

    assert result.exit_code == 0
    assert captured["country_iso3"] == "KEN"
    assert captured["data_source"] == "FAOSTAT"
    assert captured["output_dir"] == Path("reports/case/KEN")


def test_run_research_uses_pdf_vector_store_when_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metric = _metric("PDF evidence metric")
    store = MagicMock(name="pdf_vector_store")
    client = MagicMock(close=AsyncMock())
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        "fao_impact_monitor.pipeline.Metric.from_use_case",
        lambda _path: [metric],
    )
    monkeypatch.setattr(
        "fao_impact_monitor.pipeline.connect_pdf_pipeline",
        AsyncMock(return_value=client),
    )
    ensure_indexes = AsyncMock()
    monkeypatch.setattr(
        "fao_impact_monitor.pipeline.ensure_pdf_pipeline_indexes",
        ensure_indexes,
    )
    monkeypatch.setattr(
        "fao_impact_monitor.pipeline.PdfEvidenceVectorStore",
        lambda: store,
    )

    async def legacy_connection_must_not_run() -> None:
        raise AssertionError("legacy data-lake connection was used")

    monkeypatch.setattr(
        "fao_impact_monitor.pipeline.connect_data_lake",
        legacy_connection_must_not_run,
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
            use_pdf_vector_store=True,
            web_research_enabled=False,
        )
    )

    assert captured["vector_store"] is store
    assert captured["web_research_enabled"] is False
    ensure_indexes.assert_awaited_once_with()
    client.close.assert_awaited_once_with()


def test_pdf_research_discovers_all_use_cases_and_uses_separate_output_dirs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_cases = tmp_path / "use-cases"
    (use_cases / "nested").mkdir(parents=True)
    first = use_cases / "a.json"
    second = use_cases / "nested" / "b.json"
    first.write_text("{}", encoding="utf-8")
    second.write_text("{}", encoding="utf-8")
    calls: list[dict[str, Any]] = []

    async def fake_run_research(**kwargs: Any) -> Path:
        calls.append(kwargs)
        output_dir = kwargs["output_dir"]
        assert isinstance(output_dir, Path)
        return output_dir

    monkeypatch.setattr(
        "fao_impact_monitor.pipeline._run_research",
        fake_run_research,
    )
    output_root = tmp_path / "reports"

    outputs = asyncio.run(
        _run_pdf_research_all_use_cases(
            use_cases_dir=use_cases,
            country_iso3="KEN",
            output_root=output_root,
            max_parallel=3,
            metric_indices=[3],
            web_research_enabled=False,
        )
    )

    assert outputs == [
        output_root / "a" / "KEN",
        output_root / "b" / "KEN",
    ]
    assert [call["use_case_path"] for call in calls] == [first, second]
    assert all(call["metric_indices"] == [3] for call in calls)
    assert all(call["use_pdf_vector_store"] is True for call in calls)
    assert all(call["web_research_enabled"] is False for call in calls)
    assert all(call["max_parallel"] == 3 for call in calls)


def test_pdf_research_cli_forwards_repeatable_metric_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_run(**kwargs: Any) -> list[Path]:
        captured.update(kwargs)
        return []

    monkeypatch.setattr(pipeline, "_run_pdf_research_all_use_cases", fake_run)

    result = CliRunner().invoke(
        pipeline.app,
        [
            "pdf-research",
            "--country",
            "ken",
            "--use-cases-dir",
            str(tmp_path),
            "--metric",
            "3",
            "--metric",
            "5",
            "--no-web-research",
        ],
    )

    assert result.exit_code == 0
    assert captured["country_iso3"] == "KEN"
    assert captured["metric_indices"] == [3, 5]
    assert captured["web_research_enabled"] is False
    assert captured["output_root"] == Path("reports")


def test_pdf_search_cli_forwards_query_country_and_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_search(**kwargs: Any) -> list[dict[str, Any]]:
        captured.update(kwargs)
        return []

    monkeypatch.setattr(pipeline, "_run_pdf_embedding_search", fake_search)

    result = CliRunner().invoke(
        pipeline.app,
        [
            "pdf-search",
            "El Niño maize losses",
            "--country",
            "ken",
            "--limit",
            "7",
        ],
    )

    assert result.exit_code == 0
    assert captured == {
        "query": "El Niño maize losses",
        "country_iso3": "KEN",
        "limit": 7,
    }
