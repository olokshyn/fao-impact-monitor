"""Unit tests for research CLI parallel per-metric report writing."""

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
from fao_impact_monitor.pipeline import (
    _run_pdf_research_all_use_cases,
    _run_research,
    _run_research_parallel,
)
from fao_impact_monitor.research_report import MetricProcessJob


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
        "fao_impact_monitor.pipeline.connect_pdf_pipeline",
        AsyncMock(return_value=MagicMock(close=AsyncMock())),
    )
    monkeypatch.setattr(
        "fao_impact_monitor.pipeline.ensure_pdf_pipeline_indexes",
        AsyncMock(),
    )
    monkeypatch.setattr(
        "fao_impact_monitor.pipeline.PdfEvidenceVectorStore",
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
    assert "## Metric info" in report
    assert "# El Niño research - KEN" not in report
    assert "## Direct evidence" in report
    assert "Source data:" not in report
    assert "| Year | Item | Element | Unit | Value |" not in report
    assert "## Answer" not in report
    assert "## References" in report
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
        )
    )

    source.get_data.assert_awaited_once()
    report = (output_dir / "0001.md").read_text(encoding="utf-8")
    assert "## Metric info" in report
    assert "# El Niño research - KEN" not in report
    assert "Source: Total Deaths" in report
    assert "Dataset: EM-DAT" in report
    assert (
        "| Start Year | Disaster Type | Location | Regions | Value | Unit |" in report
    )
    assert "| 2023 | Flood | Nairobi | Nairobi | 178 | persons |" in report
    assert "![Total Deaths](plots/0001-emdat-1.png)" in report
    assert "## Answer" not in report
    assert "## References" in report


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
        return cast(Path, kwargs["output_dir"]) / f"{kwargs['index']:04d}.md"

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


def test_run_research_filters_metrics_by_emdat_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emdat_metric = Metric(
        name="EMDAT metric",
        description="Desc",
        example="Example",
        data_sources=[DataSourceConfig(source="EMDAT", exclusive=True)],
    )
    mixed_metric = Metric(
        name="Mixed impact",
        description="Desc",
        example="Example",
        data_sources=[
            DataSourceConfig.model_validate(
                {
                    "source": "DesInventar",
                    "indicator": "muertos",
                    "exclusive": True,
                }
            ),
            DataSourceConfig.model_validate(
                {
                    "source": "EMDAT",
                    "indicator": "Total Deaths",
                    "exclusive": True,
                }
            ),
        ],
    )
    metrics = [
        _metric("Repository metric"),
        _metric("World Bank", worldbank=True),
        Metric(
            name="FAOSTAT metric",
            description="Desc",
            example="Example",
            data_sources=[DataSourceConfig(source="FAOSTAT", exclusive=True)],
        ),
        emdat_metric,
        mixed_metric,
    ]
    selected: list[tuple[int, str]] = []
    monkeypatch.setattr(
        "fao_impact_monitor.pipeline.Metric.from_use_case",
        lambda _path: metrics,
    )

    async def fake_run_one_metric(**kwargs: Any) -> Path:
        selected.append((kwargs["index"], kwargs["metric"].name))
        return cast(Path, kwargs["output_dir"]) / f"{kwargs['index']:04d}.md"

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
            data_source="emdat",
        )
    )

    assert selected == [(4, "EMDAT metric"), (5, "Mixed impact")]


def test_run_research_filters_metrics_by_desinventar_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mixed_metric = Metric(
        name="Mixed impact",
        description="Desc",
        example="Example",
        data_sources=[
            DataSourceConfig.model_validate(
                {
                    "source": "DesInventar",
                    "indicator": "muertos",
                    "exclusive": True,
                }
            ),
            DataSourceConfig.model_validate(
                {
                    "source": "EMDAT",
                    "indicator": "Total Deaths",
                    "exclusive": True,
                }
            ),
        ],
    )
    metrics = [
        _metric("Repository metric"),
        Metric(
            name="DesInventar only",
            description="Desc",
            example="Example",
            data_sources=[
                DataSourceConfig.model_validate(
                    {
                        "source": "DesInventar",
                        "indicator": "nescuelas",
                        "exclusive": True,
                    }
                )
            ],
        ),
        mixed_metric,
    ]
    selected: list[tuple[int, str]] = []
    fetched_sources: list[list[str]] = []
    monkeypatch.setattr(
        "fao_impact_monitor.pipeline.Metric.from_use_case",
        lambda _path: metrics,
    )

    async def fake_run_one_metric(**kwargs: Any) -> Path:
        selected.append((kwargs["index"], kwargs["metric"].name))
        data_source = kwargs.get("data_source")
        configs = list(kwargs["metric"].data_sources)
        if data_source is not None:
            source_key = str(data_source).casefold()
            configs = [c for c in configs if c.source.casefold() == source_key]
        fetched_sources.append([c.source for c in configs])
        return cast(Path, kwargs["output_dir"]) / f"{kwargs['index']:04d}.md"

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
            data_source="desinventar",
        )
    )

    assert selected == [(2, "DesInventar only"), (3, "Mixed impact")]
    assert fetched_sources == [["DesInventar"], ["DesInventar"]]


def test_research_cli_forwards_source_filter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_case = tmp_path / "case.json"
    use_case.write_text('{"metrics": []}', encoding="utf-8")
    captured: dict[str, Any] = {}

    async def fake_run_research(**kwargs: Any) -> Path:
        captured.update(kwargs)
        return cast(Path, kwargs["output_dir"])

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
    assert captured["use_pdf_vector_store"] is True


def test_research_cli_undrr_metric_alias_and_countries(
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
            "--metric",
            "undrr",
        ],
    )

    assert result.exit_code == 0, result.output
    assert [call["country_iso3"] for call in calls] == ["ETH", "KEN"]
    assert all(call["metric_indices"] == [26, 27, 28, 29, 30, 31] for call in calls)


def test_research_parallel_cli_submits_country_job_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_case = tmp_path / "case.json"
    use_case.write_text('{"metrics": []}', encoding="utf-8")
    captured: dict[str, Any] = {}

    def fake_run_parallel(**kwargs: Any) -> list[dict[str, Any]]:
        captured.update(kwargs)
        return [
            {
                "country_iso3": "ETH",
                "kind": "worldbank",
                "metric_indices": [1, 2],
                "ok": True,
                "error": None,
                "pid": 1,
                "elapsed_seconds": 0.1,
            }
        ]

    monkeypatch.setattr(pipeline, "_run_research_parallel", fake_run_parallel)

    result = CliRunner().invoke(
        pipeline.app,
        [
            "research-parallel",
            "--countries",
            "eth,ken",
            "--use-case",
            str(use_case),
            "--output-root",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 0
    assert captured["countries_iso3"] == ["ETH", "KEN"]
    assert captured["use_case_path"] == use_case
    assert captured["output_root"] == tmp_path / "out"
    assert captured["continue_incomplete"] is False


def test_run_research_parallel_builds_payloads_and_submits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_case = tmp_path / "case.json"
    use_case.write_text("{}", encoding="utf-8")
    jobs = [
        MetricProcessJob(kind="worldbank", metric_indices=[1], data_source="WorldBank"),
        MetricProcessJob(kind="researcher", metric_indices=[3]),
    ]
    submitted: list[dict[str, Any]] = []

    def fake_submit(_fn: Any, payload: dict[str, Any]) -> MagicMock:
        submitted.append(payload)
        future = MagicMock()
        future.result.return_value = {
            "country_iso3": payload["country_iso3"],
            "kind": payload["kind"],
            "metric_indices": payload["metric_indices"],
            "ok": True,
            "error": None,
            "pid": 42,
            "elapsed_seconds": 0.01,
        }
        return future

    executor = MagicMock()
    executor.__enter__.return_value = executor
    executor.__exit__.return_value = None
    executor.submit.side_effect = fake_submit

    monkeypatch.setattr(
        "fao_impact_monitor.pipeline.Metric.from_use_case",
        lambda _path: [],
    )
    monkeypatch.setattr(
        "fao_impact_monitor.pipeline.ProcessPoolExecutor",
        lambda *args, **kwargs: executor,
    )
    monkeypatch.setattr(
        "fao_impact_monitor.pipeline.as_completed",
        lambda futures: list(futures),
    )
    ensure_once = AsyncMock()
    monkeypatch.setattr(
        "fao_impact_monitor.pipeline._ensure_pdf_pipeline_indexes_once",
        ensure_once,
    )

    results = _run_research_parallel(
        use_case_path=use_case,
        countries_iso3=["ETH", "KEN"],
        output_root=tmp_path / "reports",
        jobs=jobs,
    )

    assert len(submitted) == 4
    assert len(results) == 4
    assert {
        (item["country_iso3"], item["kind"], tuple(item["metric_indices"]))
        for item in submitted
    } == {
        ("ETH", "worldbank", (1,)),
        ("ETH", "researcher", (3,)),
        ("KEN", "worldbank", (1,)),
        ("KEN", "researcher", (3,)),
    }
    assert all(
        item["output_dir"] == str(tmp_path / "reports" / "case" / item["country_iso3"])
        for item in submitted
    )
    assert submitted[0]["data_source"] == "WorldBank"
    assert submitted[1]["data_source"] is None
    ensure_once.assert_not_called()


def test_run_research_parallel_ensures_indexes_when_flag_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENSURE_INDEXES", "true")
    use_case = tmp_path / "case.json"
    use_case.write_text("{}", encoding="utf-8")
    future = MagicMock()
    future.result.return_value = {
        "country_iso3": "KEN",
        "kind": "researcher",
        "metric_indices": [3],
        "ok": True,
        "error": None,
        "pid": 42,
        "elapsed_seconds": 0.01,
    }
    executor = MagicMock()
    executor.__enter__.return_value = executor
    executor.__exit__.return_value = None
    executor.submit.return_value = future
    ensure_once = AsyncMock()
    monkeypatch.setattr(
        "fao_impact_monitor.pipeline.Metric.from_use_case",
        lambda _path: [],
    )
    monkeypatch.setattr(
        "fao_impact_monitor.pipeline.ProcessPoolExecutor",
        lambda *args, **kwargs: executor,
    )
    monkeypatch.setattr(
        "fao_impact_monitor.pipeline.as_completed",
        lambda futures: list(futures),
    )
    monkeypatch.setattr(
        "fao_impact_monitor.pipeline._ensure_pdf_pipeline_indexes_once",
        ensure_once,
    )

    _run_research_parallel(
        use_case_path=use_case,
        countries_iso3=["KEN"],
        output_root=tmp_path / "reports",
        jobs=[MetricProcessJob(kind="researcher", metric_indices=[3])],
    )

    ensure_once.assert_called_once_with()


def _continue_metrics() -> list[Metric]:
    return [
        _metric("WB", worldbank=True),
        _metric("Text A"),
        _metric("Text B"),
        Metric(
            name="FAO",
            description="Desc",
            example="Example",
            unit="t",
            data_sources=[DataSourceConfig(source="FAOSTAT", exclusive=True)],
        ),
        Metric(
            name="EM",
            description="Desc",
            example="Example",
            unit="persons",
            data_sources=[
                DataSourceConfig.model_validate(
                    {"source": "EMDAT", "indicator": "Total Deaths", "exclusive": True}
                )
            ],
        ),
    ]


def test_research_parallel_cli_forwards_continue_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_case = tmp_path / "case.json"
    use_case.write_text('{"metrics": []}', encoding="utf-8")
    captured: dict[str, Any] = {}

    def fake_run_parallel(**kwargs: Any) -> list[dict[str, Any]]:
        captured.update(kwargs)
        return []

    monkeypatch.setattr(pipeline, "_run_research_parallel", fake_run_parallel)

    result = CliRunner().invoke(
        pipeline.app,
        [
            "research-parallel",
            "--countries",
            "eth",
            "--use-case",
            str(use_case),
            "--continue",
        ],
    )

    assert result.exit_code == 0
    assert captured["continue_incomplete"] is True
    assert captured["countries_iso3"] == ["ETH"]


def test_run_research_parallel_continue_spawns_only_missing_text_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_case = tmp_path / "case.json"
    use_case.write_text("{}", encoding="utf-8")
    reports = tmp_path / "reports"
    eth = reports / "case" / "ETH"
    ken = reports / "case" / "KEN"
    mwi = reports / "case" / "MWI"
    eth.mkdir(parents=True)
    ken.mkdir(parents=True)
    mwi.mkdir(parents=True)
    (eth / "0001.md").write_text("worldbank\n", encoding="utf-8")
    (eth / "0002.md").write_text("text A\n", encoding="utf-8")
    (eth / "0003.md").write_text("", encoding="utf-8")
    (mwi / "0002.md").write_text("text A\n", encoding="utf-8")
    (mwi / "0003.md").write_text("text B\n", encoding="utf-8")

    submitted: list[dict[str, Any]] = []

    def fake_submit(_fn: Any, payload: dict[str, Any]) -> MagicMock:
        submitted.append(payload)
        future = MagicMock()
        future.result.return_value = {
            "country_iso3": payload["country_iso3"],
            "kind": payload["kind"],
            "metric_indices": payload["metric_indices"],
            "ok": True,
            "error": None,
            "pid": 42,
            "elapsed_seconds": 0.01,
        }
        return future

    executor = MagicMock()
    executor.__enter__.return_value = executor
    executor.__exit__.return_value = None
    executor.submit.side_effect = fake_submit
    executor_cls = MagicMock(return_value=executor)
    ensure_once = AsyncMock()

    monkeypatch.setattr(
        "fao_impact_monitor.pipeline.Metric.from_use_case",
        lambda _path: _continue_metrics(),
    )
    monkeypatch.setattr(
        "fao_impact_monitor.pipeline.ProcessPoolExecutor",
        executor_cls,
    )
    monkeypatch.setattr(
        "fao_impact_monitor.pipeline.as_completed",
        lambda futures: list(futures),
    )
    monkeypatch.setattr(
        "fao_impact_monitor.pipeline._ensure_pdf_pipeline_indexes_once",
        ensure_once,
    )

    results = _run_research_parallel(
        use_case_path=use_case,
        countries_iso3=["ETH", "KEN", "MWI"],
        output_root=reports,
        continue_incomplete=True,
    )

    assert len(results) == 3
    assert {
        (item["country_iso3"], item["kind"], tuple(item["metric_indices"]))
        for item in submitted
    } == {
        ("ETH", "researcher", (3,)),
        ("KEN", "researcher", (2,)),
        ("KEN", "researcher", (3,)),
    }
    assert all(item["kind"] == "researcher" for item in submitted)
    assert all(item["data_source"] is None for item in submitted)
    executor_cls.assert_called_once()
    assert executor_cls.call_args.kwargs["max_workers"] == 3
    ensure_once.assert_not_called()


def test_run_research_parallel_continue_skips_when_all_text_metrics_exist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_case = tmp_path / "case.json"
    use_case.write_text("{}", encoding="utf-8")
    eth = tmp_path / "reports" / "case" / "ETH"
    eth.mkdir(parents=True)
    (eth / "0002.md").write_text("text A\n", encoding="utf-8")
    (eth / "0003.md").write_text("text B\n", encoding="utf-8")

    executor_cls = MagicMock()
    ensure_once = AsyncMock()
    monkeypatch.setattr(
        "fao_impact_monitor.pipeline.Metric.from_use_case",
        lambda _path: _continue_metrics(),
    )
    monkeypatch.setattr(
        "fao_impact_monitor.pipeline.ProcessPoolExecutor",
        executor_cls,
    )
    monkeypatch.setattr(
        "fao_impact_monitor.pipeline._ensure_pdf_pipeline_indexes_once",
        ensure_once,
    )

    results = _run_research_parallel(
        use_case_path=use_case,
        countries_iso3=["ETH"],
        output_root=tmp_path / "reports",
        continue_incomplete=True,
    )

    assert results == []
    executor_cls.assert_not_called()
    ensure_once.assert_not_called()


def test_research_process_worker_uses_pdf_vector_store_for_researcher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_run_research(**kwargs: Any) -> Path:
        captured.update(kwargs)
        return cast(Path, kwargs["output_dir"])

    monkeypatch.setattr(pipeline, "_run_research", fake_run_research)

    result = pipeline._research_process_worker(
        {
            "use_case_path": str(tmp_path / "case.json"),
            "country_iso3": "GTM",
            "kind": "researcher",
            "metric_indices": [7],
            "data_source": None,
            "output_dir": str(tmp_path / "reports"),
        }
    )

    assert result["ok"] is True
    assert captured["use_pdf_vector_store"] is True
    assert captured["ensure_indexes"] is False


def test_run_research_uses_pdf_vector_store_by_default(
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
            web_research_enabled=False,
        )
    )

    assert captured["vector_store"] is store
    assert captured["web_research_enabled"] is False
    ensure_indexes.assert_not_awaited()
    client.close.assert_awaited_once_with()


def test_run_research_skips_index_ensure_when_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENSURE_INDEXES", "true")
    metric = _metric("PDF evidence metric")
    ensure_indexes = AsyncMock()
    monkeypatch.setattr(
        "fao_impact_monitor.pipeline.Metric.from_use_case",
        lambda _path: [metric],
    )
    monkeypatch.setattr(
        "fao_impact_monitor.pipeline.connect_pdf_pipeline",
        AsyncMock(return_value=MagicMock(close=AsyncMock())),
    )
    monkeypatch.setattr(
        "fao_impact_monitor.pipeline.ensure_pdf_pipeline_indexes",
        ensure_indexes,
    )
    monkeypatch.setattr(
        "fao_impact_monitor.pipeline.PdfEvidenceVectorStore",
        lambda: MagicMock(),
    )
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
            ensure_indexes=False,
        )
    )

    ensure_indexes.assert_not_awaited()


def test_run_research_ensures_indexes_when_flag_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENSURE_INDEXES", "true")
    metric = _metric("PDF evidence metric")
    ensure_indexes = AsyncMock()
    monkeypatch.setattr(
        "fao_impact_monitor.pipeline.Metric.from_use_case",
        lambda _path: [metric],
    )
    monkeypatch.setattr(
        "fao_impact_monitor.pipeline.connect_pdf_pipeline",
        AsyncMock(return_value=MagicMock(close=AsyncMock())),
    )
    monkeypatch.setattr(
        "fao_impact_monitor.pipeline.ensure_pdf_pipeline_indexes",
        ensure_indexes,
    )
    monkeypatch.setattr(
        "fao_impact_monitor.pipeline.PdfEvidenceVectorStore",
        lambda: MagicMock(),
    )
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


def test_run_research_parallel_skips_indexes_without_researcher_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_case = tmp_path / "case.json"
    use_case.write_text("{}", encoding="utf-8")
    future = MagicMock()
    future.result.return_value = {
        "country_iso3": "KEN",
        "kind": "worldbank",
        "metric_indices": [1],
        "ok": True,
        "error": None,
        "pid": 42,
        "elapsed_seconds": 0.01,
    }
    executor = MagicMock()
    executor.__enter__.return_value = executor
    executor.__exit__.return_value = None
    executor.submit.return_value = future
    ensure_once = AsyncMock()
    monkeypatch.setattr(
        "fao_impact_monitor.pipeline.Metric.from_use_case",
        lambda _path: [],
    )
    monkeypatch.setattr(
        "fao_impact_monitor.pipeline.ProcessPoolExecutor",
        lambda *args, **kwargs: executor,
    )
    monkeypatch.setattr(
        "fao_impact_monitor.pipeline.as_completed",
        lambda futures: list(futures),
    )
    monkeypatch.setattr(
        "fao_impact_monitor.pipeline._ensure_pdf_pipeline_indexes_once",
        ensure_once,
    )

    _run_research_parallel(
        use_case_path=use_case,
        countries_iso3=["KEN"],
        output_root=tmp_path / "reports",
        jobs=[
            MetricProcessJob(
                kind="worldbank", metric_indices=[1], data_source="WorldBank"
            )
        ],
    )

    ensure_once.assert_not_called()


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


def test_impact_report_cli_writes_markdown_and_pdf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_case = tmp_path / "el-nino.json"
    use_case.write_text(
        '{"name": "El Niño", "metrics": []}',
        encoding="utf-8",
    )
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
    monkeypatch.setattr(
        pipeline,
        "enrich_structured_latest_values",
        AsyncMock(),
    )
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
            "impact-report",
            "--countries",
            "ETH",
            "--use-case",
            str(use_case),
            "--input-root",
            str(input_root),
            "--output-root",
            str(input_root),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "ETH impact analysis El Nino.md" in result.output
    assert "ETH impact analysis El Nino.pdf" in result.output


def test_undrr_report_cli_writes_markdown_and_pdf(
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
            "undrr-report",
            "--countries",
            "ETH",
            "--use-case",
            str(use_case),
            "--input-root",
            str(input_root),
            "--output-root",
            str(input_root),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "ETH UNDRR El Nino.md" in result.output
    assert "ETH UNDRR El Nino.pdf" in result.output


def test_report_pdf_cli_forwards_human_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_build(**kwargs: Any) -> Path:
        captured.update(kwargs)
        output = cast(Path, kwargs["output_path"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"%PDF-1.4\n")
        return output

    monkeypatch.setattr(pipeline, "build_research_pdf", fake_build)
    result = CliRunner().invoke(
        pipeline.app,
        [
            "report-pdf",
            "--country",
            "ETH",
            "--input",
            str(tmp_path),
            "--output",
            str(tmp_path / "out.pdf"),
            "--human",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["human"] is True
    assert captured["input_dir"] == tmp_path


def test_report_pdf_cli_human_defaults_to_human_filename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_build(**kwargs: Any) -> Path:
        captured.update(kwargs)
        output = cast(Path, kwargs["output_path"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"%PDF-1.4\n")
        return output

    monkeypatch.setattr(pipeline, "build_research_pdf", fake_build)
    result = CliRunner().invoke(
        pipeline.app,
        ["report-pdf", "--country", "FJI", "--input", str(tmp_path), "--human"],
    )
    assert result.exit_code == 0, result.output
    assert captured["human"] is True
    assert captured["output_path"] == Path(
        "reports/el-nino/FJI/FJI metrics El Nino - human.pdf"
    )


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
