import asyncio
from datetime import UTC, datetime

import pandas as pd
import wbgapi as wb
from pydantic import ConfigDict

from .data_source import DataResult, DataSource


class WorldBankDataResult(DataResult):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: pd.DataFrame


class WorldBank(DataSource):
    source: str = "WorldBank"
    indicator: str

    async def get_data(
        self,
        country_iso3: str,
        *,
        year_start: int | None = None,
        year_end: int | None = None,
    ) -> list[DataResult]:
        return await asyncio.to_thread(
            self._get_data_sync,
            country_iso3,
            year_start,
            year_end,
        )

    def _get_data_sync(
        self,
        country_iso3: str,
        year_start: int | None,
        year_end: int | None,
    ) -> list[DataResult]:
        wide = wb.data.DataFrame(
            self.indicator,
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

        series_info = wb.series.get(self.indicator)
        indicator_name = series_info.get("value", self.indicator)
        url = f"https://data.worldbank.org/indicator/{self.indicator}"
        citation = (
            f'World Bank. "{indicator_name}". World Development Indicators. {url}'
        )

        return [
            WorldBankDataResult(
                source=self.source,
                document=indicator_name,
                url=url,
                citation=citation,
                metadata={
                    "indicator": self.indicator,
                    "country_iso3": country_iso3,
                    "year_start": year_start,
                    "year_end": year_end,
                    "unit": self.unit,
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
