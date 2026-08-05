import asyncio
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import pytest
import wbgapi as wb

from fao_impact_monitor.data_source import (
    WorldBank,
    WorldBankDataResult,
    WorldBankDataSourceConfig,
    get_data_source,
)
from fao_impact_monitor.data_source.data_source import _DATA_SOURCE_CLS_REGISTRY
from fao_impact_monitor.metric import Metric


def _metric_with_world_bank(
    indicator: str = "NV.AGR.TOTL.ZS",
    unit: str | None = "%",
) -> tuple[Metric, WorldBankDataSourceConfig]:
    config = WorldBankDataSourceConfig(
        source="WorldBank",
        indicator=indicator,
        unit=unit,
    )
    metric = Metric(
        name="Agriculture share of GDP",
        description="The share of agriculture in the total GDP of a country.",
        example="Agriculture contributed 24.3% of GDP in 2023.",
        unit="%",
        data_sources=[config],
    )
    return metric, config


def test_world_bank_is_registered() -> None:
    assert _DATA_SOURCE_CLS_REGISTRY["WorldBank"] is WorldBank
    source = get_data_source("WorldBank")
    assert isinstance(source, WorldBank)
    assert source.source == "WorldBank"


def test_time_range_all_years() -> None:
    assert WorldBank._time_range(None, None) == "all"


def test_time_range_both_bounds() -> None:
    assert WorldBank._time_range(2020, 2023) == range(2020, 2024)


def test_time_range_start_only(monkeypatch: pytest.MonkeyPatch) -> None:
    class FixedDateTime:
        @staticmethod
        def now(*, tz: Any = None) -> datetime:
            return datetime(2024, 6, 1, tzinfo=UTC)

    monkeypatch.setattr(
        "fao_impact_monitor.data_source.world_bank.datetime",
        FixedDateTime,
    )
    assert WorldBank._time_range(2020, None) == range(2020, 2025)


def test_time_range_end_only() -> None:
    assert WorldBank._time_range(None, 1970) == range(1960, 1971)


def test_time_range_invalid() -> None:
    with pytest.raises(ValueError, match="year_start .* must be <="):
        WorldBank._time_range(2025, 2020)


def test_get_data_returns_tidy_dataframe(monkeypatch: pytest.MonkeyPatch) -> None:
    wide = pd.DataFrame(
        {2020: [22.7], 2021: [21.5]},
        index=pd.Index(["KEN"], name="economy"),
    )
    captured: dict[str, Any] = {}

    def fake_dataframe(
        series: str,
        economy: str,
        *,
        time: str | range,
        numericTimeKeys: bool,
        skipBlanks: bool,
    ) -> pd.DataFrame:
        captured["series"] = series
        captured["economy"] = economy
        captured["time"] = time
        captured["numericTimeKeys"] = numericTimeKeys
        captured["skipBlanks"] = skipBlanks
        return wide

    monkeypatch.setattr(wb.data, "DataFrame", fake_dataframe)
    monkeypatch.setattr(
        wb.series,
        "get",
        lambda indicator: {
            "id": indicator,
            "value": "Agriculture, forestry, and fishing, value added (% of GDP)",
        },
    )

    source = WorldBank()
    metric, config = _metric_with_world_bank()
    results = asyncio.run(
        source.get_data(metric, config, "KEN", year_start=2020, year_end=2021)
    )

    assert captured == {
        "series": "NV.AGR.TOTL.ZS",
        "economy": "KEN",
        "time": range(2020, 2022),
        "numericTimeKeys": True,
        "skipBlanks": True,
    }
    assert len(results) == 1
    result = results[0]
    assert isinstance(result, WorldBankDataResult)
    assert result.source == "WorldBank"
    assert result.title == "Agriculture, forestry, and fishing, value added (% of GDP)"
    assert (
        result.url == "https://data.worldbank.org/indicator/NV.AGR.TOTL.ZS?locations=KE"
    )
    assert "World Development Indicators" in result.citation
    assert result.metadata == {
        "indicator": "NV.AGR.TOTL.ZS",
        "country_iso3": "KEN",
        "year_start": 2020,
        "year_end": 2021,
        "unit": "%",
    }
    pd.testing.assert_frame_equal(
        result.data,
        pd.DataFrame({"year": [2020, 2021], "value": [22.7, 21.5]}),
    )


def test_get_data_empty_returns_no_results(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wb.data, "DataFrame", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(
        wb.series, "get", lambda indicator: {"id": indicator, "value": "x"}
    )

    source = WorldBank()
    metric, config = _metric_with_world_bank()
    results = asyncio.run(
        source.get_data(metric, config, "KEN", year_start=2020, year_end=2021)
    )

    assert results == []


@pytest.mark.integration
def test_get_data_fetches_ken_agriculture_share_of_gdp() -> None:
    source = WorldBank()
    metric, config = _metric_with_world_bank()
    results = asyncio.run(
        source.get_data(metric, config, "KEN", year_start=2020, year_end=2023)
    )

    assert len(results) == 1
    result = results[0]
    assert isinstance(result, WorldBankDataResult)
    assert result.source == "WorldBank"
    assert result.metadata["indicator"] == "NV.AGR.TOTL.ZS"
    assert result.metadata["country_iso3"] == "KEN"
    assert (
        result.url == "https://data.worldbank.org/indicator/NV.AGR.TOTL.ZS?locations=KE"
    )
    assert "Agriculture" in (result.title or "")
    assert list(result.data.columns) == ["year", "value"]
    assert set(result.data["year"]) <= {2020, 2021, 2022, 2023}
    assert len(result.data) > 0
    assert result.data["value"].notna().all()
    assert (result.data["value"] > 0).all()
