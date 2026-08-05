"""Unit tests for research CLI parallel per-metric report writing."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from fao_impact_monitor.agent.researcher_agent import ResearcherOutput
from fao_impact_monitor.data_source.data_source_config import DataSourceConfig
from fao_impact_monitor.metric.metric import Metric
from fao_impact_monitor.pipeline import _run_research


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

    monkeypatch.setattr(
        "fao_impact_monitor.pipeline.WorldBank.get_data",
        fake_wb_get_data,
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
