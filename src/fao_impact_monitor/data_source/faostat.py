from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import IO, Final, Literal
from zipfile import ZipFile

import pandas as pd
import pycountry
from pydantic import ConfigDict

from fao_impact_monitor.metric.metric import Metric

from .data_source import DataResult, DataSource
from .data_source_config import DataSourceConfig

FAOSTAT_INPUTS_LAND_USE: Final = "Inputs: Land use"
FAOSTAT_RURAL_LIVELIHOODS: Final = "Rural Livelihoods Indicators"
FAOSTAT_INVESTMENT_PROFILE: Final = "Investment: Country Investment Statistics Profile"
FAOSTAT_MACRO_INDICATORS: Final = "Macro-Statistics: Key Indicators"
FAOSTAT_PRODUCTION_CROPS_LIVESTOCK: Final = "Production: Crops and livestock products"
FAOSTAT_PRODUCTION_INDICES: Final = "Production: Production indices"
FAOSTAT_VALUE_OF_PRODUCTION: Final = "Value of Production"
FAOSTAT_FOOD_SECURITY: Final = "Food Security and Nutrition Indicators"
FAOSTAT_CROPS_LIVESTOCK_TRADE: Final = "Trade: Crops and livestock products"

FAOSTATDataset = Literal[
    "Inputs: Land use",
    "Rural Livelihoods Indicators",
    "Investment: Country Investment Statistics Profile",
    "Macro-Statistics: Key Indicators",
    "Production: Crops and livestock products",
    "Production: Production indices",
    "Value of Production",
    "Food Security and Nutrition Indicators",
    "Trade: Crops and livestock products",
]

_DATASET_ARCHIVES: dict[str, tuple[Path, str, str]] = {
    FAOSTAT_INPUTS_LAND_USE: (
        Path("FAOSTAT_A-S_E/Inputs_LandUse_E_All_Data_(Normalized).zip"),
        "Inputs_LandUse_E_All_Data_(Normalized).csv",
        "https://www.fao.org/faostat/en/#data/RL",
    ),
    FAOSTAT_RURAL_LIVELIHOODS: (
        Path("FAOSTAT_A-S_E/Rural_Livelihoods_Indicators_E_All_Data_(Normalized).zip"),
        "Rural_Livelihoods_Indicators_E_All_Data_(Normalized).csv",
        "https://www.fao.org/faostat/en/#data/OER",
    ),
    FAOSTAT_INVESTMENT_PROFILE: (
        Path(
            "FAOSTAT_A-S_E/"
            "Investment_CountryInvestmentStatisticsProfile_E_All_Data_(Normalized).zip"
        ),
        "Investment_CountryInvestmentStatisticsProfile_E_All_Data_(Normalized).csv",
        "https://www.fao.org/faostat/en/#data",
    ),
    FAOSTAT_MACRO_INDICATORS: (
        Path(
            "FAOSTAT_A-S_E/Macro-Statistics_Key_Indicators_E_All_Data_(Normalized).zip"
        ),
        "Macro-Statistics_Key_Indicators_E_All_Data_(Normalized).csv",
        "https://www.fao.org/faostat/en/#data/MK",
    ),
    FAOSTAT_PRODUCTION_CROPS_LIVESTOCK: (
        Path("FAOSTAT_A-S_E/Production_Crops_Livestock_E_All_Data_(Normalized).zip"),
        "Production_Crops_Livestock_E_All_Data_(Normalized).csv",
        "https://www.fao.org/faostat/en/#data/QCL",
    ),
    FAOSTAT_PRODUCTION_INDICES: (
        Path("FAOSTAT_A-S_E/Production_Indices_E_All_Data_(Normalized).zip"),
        "Production_Indices_E_All_Data_(Normalized).csv",
        "https://www.fao.org/faostat/en/#data/QI",
    ),
    FAOSTAT_VALUE_OF_PRODUCTION: (
        Path("FAOSTAT_T-Z_E/Value_of_Production_E_All_Data_(Normalized).zip"),
        "Value_of_Production_E_All_Data_(Normalized).csv",
        "https://www.fao.org/faostat/en/#data/QV",
    ),
    FAOSTAT_FOOD_SECURITY: (
        Path("FAOSTAT_A-S_E/Food_Security_Data_E_All_Data_(Normalized).zip"),
        "Food_Security_Data_E_All_Data_(Normalized).csv",
        "https://www.fao.org/faostat/en/#data/FS",
    ),
    FAOSTAT_CROPS_LIVESTOCK_TRADE: (
        Path("FAOSTAT_T-Z_E/Trade_CropsLivestock_E_All_Data_(Normalized).zip"),
        "Trade_CropsLivestock_E_All_Data_(Normalized).csv",
        "https://www.fao.org/faostat/en/#data/TCL",
    ),
}

_SURVEY_DATASETS = {FAOSTAT_RURAL_LIVELIHOODS}

_RURAL_COLUMNS = {
    "Survey Code": "survey_code",
    "Survey": "survey",
    "Indicator Code": "indicator_code",
    "Indicator": "indicator",
    "Element Code": "element_code",
    "Element": "element",
    "Qualifier Code": "qualifier_code",
    "Qualifier": "qualifier",
    "Source Code": "source_code",
    "Source": "observation_source",
    "Unit": "unit",
    "Value": "value",
    "Flag": "flag",
    "Note": "note",
}

_STANDARD_COLUMNS = {
    "Area Code": "area_code",
    "Area Code (M49)": "area_m49",
    "Area": "area",
    "Item Code": "item_code",
    "Item": "item",
    "Element Code": "element_code",
    "Element": "element",
    "Year": "year",
    "Unit": "unit",
    "Value": "value",
    "Flag": "flag",
}

_CHUNK_SIZE = 100_000


class FAOSTATDataResult(DataResult):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: pd.DataFrame


class FAOSTATDataSourceConfig(DataSourceConfig):
    dataset: FAOSTATDataset
    indicator: str
    item_code: str | None = None
    indicator_code: str | None = None
    element: str
    element_code: str
    qualifier_code: str | None = None
    source_code: str | None = None
    data_dir: Path = Path("faostat_data")


class FAOSTAT(DataSource):
    source: str = "FAOSTAT"

    def __init__(self) -> None:
        self._country_cache: dict[tuple[object, ...], pd.DataFrame] = {}
        self._cache_lock = Lock()

    async def get_data(
        self,
        metric: Metric,
        data_source_config: DataSourceConfig,
        country_iso3: str,
        *,
        year_start: int | None = None,
        year_end: int | None = None,
    ) -> list[DataResult]:
        del metric
        config = FAOSTATDataSourceConfig.model_validate(data_source_config.model_dump())
        return await asyncio.to_thread(
            self._get_data_sync,
            config,
            country_iso3,
            year_start,
            year_end,
        )

    def _get_data_sync(
        self,
        config: FAOSTATDataSourceConfig,
        country_iso3: str,
        year_start: int | None,
        year_end: int | None,
    ) -> list[DataResult]:
        if year_start is not None and year_end is not None and year_start > year_end:
            raise ValueError(
                f"year_start ({year_start}) must be <= year_end ({year_end})"
            )
        self._validate_config(config)
        country_m49 = _iso3_to_m49(country_iso3)
        relative_archive, csv_member, url = _DATASET_ARCHIVES[config.dataset]
        archive_path = config.data_dir / relative_archive
        if not archive_path.is_file():
            raise FileNotFoundError(f"FAOSTAT archive not found: {archive_path}")

        country_data = self._get_country_data(
            config.dataset,
            archive_path,
            csv_member,
            country_m49,
            year_start,
            year_end,
        )
        data = self._filter_sub_indicator(country_data, config)
        if data.empty:
            return []

        data = data.reset_index(drop=True).copy()
        data.insert(0, "country_iso3", country_iso3.upper())
        return [
            FAOSTATDataResult(
                source=self.source,
                title=config.indicator,
                url=url,
                citation=(
                    f'FAOSTAT. "{config.indicator}". {config.dataset}. '
                    f"Food and Agriculture Organization of the United Nations. {url}"
                ),
                metadata={
                    "dataset": config.dataset,
                    "indicator": config.indicator,
                    "item_code": config.item_code,
                    "indicator_code": config.indicator_code,
                    "element": config.element,
                    "element_code": config.element_code,
                    "qualifier_code": config.qualifier_code,
                    "source_code": config.source_code,
                    "country_iso3": country_iso3.upper(),
                    "year_start": year_start,
                    "year_end": year_end,
                    "unit": config.unit,
                    "archive_path": str(archive_path),
                    "row_count": len(data),
                },
                data=data,
            )
        ]

    @staticmethod
    def _validate_config(config: FAOSTATDataSourceConfig) -> None:
        if config.dataset in _SURVEY_DATASETS:
            if config.indicator_code is None:
                raise ValueError(
                    f"{config.dataset} requires indicator_code and element_code"
                )
            return
        if config.item_code is None:
            raise ValueError(f"{config.dataset} requires item_code and element_code")

    def _get_country_data(
        self,
        dataset: FAOSTATDataset,
        archive_path: Path,
        csv_member: str,
        country_m49: str,
        year_start: int | None,
        year_end: int | None,
    ) -> pd.DataFrame:
        cache_key = (
            dataset,
            str(archive_path.resolve()),
            archive_path.stat().st_mtime_ns,
            country_m49,
            year_start,
            year_end,
        )
        with self._cache_lock:
            cached = self._country_cache.get(cache_key)
            if cached is not None:
                return cached
            if dataset in _SURVEY_DATASETS:
                data = self._load_rural_country(
                    archive_path,
                    csv_member,
                    country_m49,
                    year_start,
                    year_end,
                )
            else:
                data = self._load_country(
                    archive_path,
                    csv_member,
                    country_m49,
                    year_start,
                    year_end,
                )
            self._country_cache[cache_key] = data
            return data

    @staticmethod
    def _load_country(
        archive_path: Path,
        csv_member: str,
        country_m49: str,
        year_start: int | None,
        year_end: int | None,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        with _open_csv_member(archive_path, csv_member) as csv_file:
            chunks = pd.read_csv(
                csv_file,
                usecols=list(_STANDARD_COLUMNS),
                dtype={column: "string" for column in _STANDARD_COLUMNS},
                chunksize=_CHUNK_SIZE,
            )
            for chunk in chunks:
                selected = chunk[chunk["Area Code (M49)"] == f"'{country_m49}"].copy()
                selected = _parse_values(selected)
                selected["Period"] = selected["Year"]
                selected["Year"] = selected["Year"].str.extract(
                    r"(\d{4})$", expand=False
                )
                selected = selected.dropna(subset=["Year"])
                selected["Year"] = selected["Year"].astype("int32")
                selected = _filter_years(selected, "Year", year_start, year_end)
                if not selected.empty:
                    frames.append(selected)
        if not frames:
            return pd.DataFrame(
                columns=[*_STANDARD_COLUMNS.values(), "value_raw", "period"]
            )
        data = pd.concat(frames, ignore_index=True).rename(
            columns=_STANDARD_COLUMNS | {"Value Raw": "value_raw", "Period": "period"}
        )
        data["area_m49"] = data["area_m49"].str.removeprefix("'")
        return data

    @staticmethod
    def _load_rural_country(
        archive_path: Path,
        csv_member: str,
        country_m49: str,
        year_start: int | None,
        year_end: int | None,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        with _open_csv_member(archive_path, csv_member) as csv_file:
            chunks = pd.read_csv(
                csv_file,
                usecols=list(_RURAL_COLUMNS),
                dtype={column: "string" for column in _RURAL_COLUMNS},
                chunksize=_CHUNK_SIZE,
            )
            for chunk in chunks:
                selected = chunk[
                    chunk["Survey Code"].str.startswith(f"{country_m49}_", na=False)
                ].copy()
                selected = _parse_values(selected)
                selected["Year"] = selected["Survey Code"].str.extract(
                    r"_(\d{4})$", expand=False
                )
                selected = selected.dropna(subset=["Year"])
                selected["Year"] = selected["Year"].astype("int32")
                selected["Period"] = selected["Note"]
                selected = _filter_years(selected, "Year", year_start, year_end)
                if not selected.empty:
                    frames.append(selected)
        if not frames:
            return pd.DataFrame(
                columns=[*_RURAL_COLUMNS.values(), "value_raw", "year", "period"]
            )
        return pd.concat(frames, ignore_index=True).rename(
            columns=_RURAL_COLUMNS
            | {"Value Raw": "value_raw", "Year": "year", "Period": "period"}
        )

    @staticmethod
    def _filter_sub_indicator(
        country_data: pd.DataFrame,
        config: FAOSTATDataSourceConfig,
    ) -> pd.DataFrame:
        selected = country_data
        filters = {
            "item_code": config.item_code,
            "indicator_code": config.indicator_code,
            "element": config.element,
            "element_code": config.element_code,
            "qualifier_code": config.qualifier_code,
            "source_code": config.source_code,
        }
        for column, value in filters.items():
            if value is not None and column in selected.columns:
                selected = selected[selected[column] == value]
        return selected


@contextmanager
def _open_csv_member(
    archive_path: Path,
    csv_member: str,
) -> Iterator[IO[bytes]]:
    with ZipFile(archive_path) as archive:
        try:
            csv_file = archive.open(csv_member)
        except KeyError as exc:
            raise ValueError(
                f"FAOSTAT archive {archive_path} does not contain {csv_member}"
            ) from exc
        with csv_file:
            yield csv_file


def _filter_years(
    data: pd.DataFrame,
    year_column: str,
    year_start: int | None,
    year_end: int | None,
) -> pd.DataFrame:
    selected = data
    if year_start is not None:
        selected = selected[selected[year_column] >= year_start]
    if year_end is not None:
        selected = selected[selected[year_column] <= year_end]
    return selected


def _parse_values(data: pd.DataFrame) -> pd.DataFrame:
    data["Value Raw"] = data["Value"]
    data["Value"] = pd.to_numeric(
        data["Value"].str.removeprefix("<"),
        errors="coerce",
    )
    return data.dropna(subset=["Value"])


def _iso3_to_m49(country_iso3: str) -> str:
    country = pycountry.countries.get(alpha_3=country_iso3.upper())
    if country is None:
        raise ValueError(f"Unknown ISO3 country code: {country_iso3!r}")
    numeric = getattr(country, "numeric", None)
    if not isinstance(numeric, str) or not numeric:
        raise ValueError(f"No M49 code for ISO3 country code: {country_iso3!r}")
    return numeric
