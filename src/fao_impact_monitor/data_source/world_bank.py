import asyncio
from datetime import UTC, datetime

import pandas as pd
import wbgapi as wb
from pydantic import ConfigDict

from fao_impact_monitor.metric.metric import Metric
from fao_impact_monitor.utils.country import iso3_to_iso2

from .data_source import DataResult, DataSource
from .data_source_config import DataSourceConfig


class WorldBankDataResult(DataResult):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: pd.DataFrame


class WorldBankDataSourceConfig(DataSourceConfig):
    indicator: str


class WorldBank(DataSource):
    source: str = "WorldBank"

    async def get_data(
        self,
        metric: Metric,
        data_source_config: DataSourceConfig,
        country_iso3: str,
        *,
        year_start: int | None = None,
        year_end: int | None = None,
    ) -> list[DataResult]:
        config = WorldBankDataSourceConfig.model_validate(
            data_source_config.model_dump()
        )
        return await asyncio.to_thread(
            self._get_data_sync,
            config,
            country_iso3,
            year_start,
            year_end,
        )

    def _get_data_sync(
        self,
        config: WorldBankDataSourceConfig,
        country_iso3: str,
        year_start: int | None,
        year_end: int | None,
    ) -> list[DataResult]:
        wide = wb.data.DataFrame(
            config.indicator,
            country_iso3,
            time=self._time_range(year_start, year_end),
            numericTimeKeys=True,
            skipBlanks=True,
        )
        if wide.empty:
            return []

        row = wide.loc[country_iso3]
        data = row.rename("value").rename_axis("year").reset_index()
        data["year"] = data["year"].astype(int)

        series_info = wb.series.get(config.indicator)
        indicator_name = series_info.get("value", config.indicator)
        url = world_bank_indicator_url(config.indicator, country_iso3)
        citation = (
            f'World Bank. "{indicator_name}". World Development Indicators. {url}'
        )

        return [
            WorldBankDataResult(
                source=self.source,
                title=indicator_name,
                url=url,
                citation=citation,
                metadata={
                    "indicator": config.indicator,
                    "country_iso3": country_iso3,
                    "year_start": year_start,
                    "year_end": year_end,
                    "unit": config.unit,
                },
                data=data,
            )
        ]

    @staticmethod
    def _time_range(
        year_start: int | None,
        year_end: int | None,
    ) -> str | range:
        if year_start is None and year_end is None:
            return "all"

        start = year_start if year_start is not None else 1960
        end = year_end if year_end is not None else datetime.now(tz=UTC).year
        if start > end:
            raise ValueError(f"year_start ({start}) must be <= year_end ({end})")
        return range(start, end + 1)


def world_bank_indicator_url(indicator: str, country_iso3: str) -> str:
    """Country-specific World Bank indicator URL (ISO2 ``locations`` param)."""
    iso2 = iso3_to_iso2(country_iso3)
    return f"https://data.worldbank.org/indicator/{indicator}?locations={iso2}"
