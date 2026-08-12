from __future__ import annotations

import asyncio
import json
from pathlib import Path
from threading import Lock
from typing import Any, Final

import pandas as pd
from pydantic import ConfigDict, Field

from fao_impact_monitor.metric.metric import Metric

from .data_source import DataResult, DataSource
from .data_source_config import DataSourceConfig

EMDAT_URL: Final = "https://www.emdat.be"
EMDAT_DATA_SHEET: Final = "EM-DAT Data"
DEFAULT_DISASTER_SUBGROUPS: Final[tuple[str, ...]] = (
    "Climatological",
    "Hydrological",
    "Meteorological",
)
UNSPECIFIED_REGION: Final = "Unspecified"

_KEEP_COLUMNS: Final[tuple[str, ...]] = (
    "DisNo.",
    "Disaster Subgroup",
    "Disaster Type",
    "Disaster Subtype",
    "Event Name",
    "ISO",
    "Country",
    "Location",
    "Admin Units",
    "Start Year",
    "Start Month",
    "Start Day",
    "End Year",
    "End Month",
    "End Day",
)

_OUTPUT_COLUMNS: Final[dict[str, str]] = {
    "DisNo.": "dis_no",
    "Disaster Subgroup": "disaster_subgroup",
    "Disaster Type": "disaster_type",
    "Disaster Subtype": "disaster_subtype",
    "Event Name": "event_name",
    "ISO": "iso",
    "Country": "country",
    "Location": "location",
    "Admin Units": "admin_units",
    "Start Year": "start_year",
    "Start Month": "start_month",
    "Start Day": "start_day",
    "End Year": "end_year",
    "End Month": "end_month",
    "End Day": "end_day",
}


class EmDatDataResult(DataResult):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: pd.DataFrame


class EmDatDataSourceConfig(DataSourceConfig):
    indicator: str
    data_path: Path = Path("datasets/emdat.xlsx")
    disaster_subgroups: list[str] = Field(
        default_factory=lambda: list(DEFAULT_DISASTER_SUBGROUPS)
    )


class EMDAT(DataSource):
    source: str = "EMDAT"

    def __init__(self) -> None:
        self._workbook_cache: dict[tuple[str, int], pd.DataFrame] = {}
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
        config = EmDatDataSourceConfig.model_validate(data_source_config.model_dump())
        return await asyncio.to_thread(
            self._get_data_sync,
            config,
            country_iso3,
            year_start,
            year_end,
        )

    def _get_data_sync(
        self,
        config: EmDatDataSourceConfig,
        country_iso3: str,
        year_start: int | None,
        year_end: int | None,
    ) -> list[DataResult]:
        if year_start is not None and year_end is not None and year_start > year_end:
            raise ValueError(
                f"year_start ({year_start}) must be <= year_end ({year_end})"
            )
        data_path = config.data_path
        if not data_path.is_file():
            raise FileNotFoundError(f"EM-DAT workbook not found: {data_path}")

        workbook = self._load_workbook(data_path)
        if config.indicator not in workbook.columns:
            raise ValueError(
                f"Unknown EM-DAT indicator {config.indicator!r}; "
                f"available columns include impact fields such as "
                f"'Total Deaths', 'No. Affected', 'No. Homeless'"
            )

        selected = workbook[
            workbook["ISO"].astype(str).str.upper() == country_iso3.upper()
        ]
        if config.disaster_subgroups:
            selected = selected[
                selected["Disaster Subgroup"].isin(config.disaster_subgroups)
            ]
        selected = _filter_years(selected, year_start, year_end)
        selected = selected[selected[config.indicator].notna()].copy()
        if selected.empty:
            return []

        keep = [column for column in _KEEP_COLUMNS if column in selected.columns]
        data = selected[keep].rename(
            columns={
                key: value for key, value in _OUTPUT_COLUMNS.items() if key in keep
            }
        )
        data["value"] = pd.to_numeric(selected[config.indicator], errors="coerce")
        data = data.dropna(subset=["value"]).reset_index(drop=True)
        if data.empty:
            return []
        if "admin_units" in data.columns:
            data["regions"] = data["admin_units"].map(parse_emdat_regions)
        else:
            data["regions"] = [[UNSPECIFIED_REGION] for _ in range(len(data))]
        data.insert(0, "country_iso3", country_iso3.upper())
        data = data.sort_values(
            by=["start_year", "start_month", "start_day", "dis_no"],
            ascending=[False, False, False, True],
            kind="mergesort",
        ).reset_index(drop=True)

        return [
            EmDatDataResult(
                source=self.source,
                title=config.indicator,
                url=EMDAT_URL,
                citation=(
                    f'EM-DAT. "{config.indicator}". '
                    f"CRED / UCLouvain, Brussels, Belgium. {EMDAT_URL}"
                ),
                metadata={
                    "indicator": config.indicator,
                    "country_iso3": country_iso3.upper(),
                    "year_start": year_start,
                    "year_end": year_end,
                    "unit": config.unit,
                    "disaster_subgroups": list(config.disaster_subgroups),
                    "data_path": str(data_path),
                    "row_count": len(data),
                },
                data=data,
            )
        ]

    def _load_workbook(self, data_path: Path) -> pd.DataFrame:
        resolved = data_path.resolve()
        cache_key = (str(resolved), resolved.stat().st_mtime_ns)
        with self._cache_lock:
            cached = self._workbook_cache.get(cache_key)
            if cached is not None:
                return cached
            data = pd.read_excel(resolved, sheet_name=EMDAT_DATA_SHEET)
            self._workbook_cache[cache_key] = data
            return data


def parse_emdat_regions(admin_units: Any) -> list[str]:
    """Return unique region names from an EM-DAT Admin Units JSON payload."""
    if admin_units is None or (isinstance(admin_units, float) and pd.isna(admin_units)):
        return [UNSPECIFIED_REGION]
    if isinstance(admin_units, str):
        text = admin_units.strip()
        if not text:
            return [UNSPECIFIED_REGION]
        try:
            payload: Any = json.loads(text)
        except json.JSONDecodeError:
            return [UNSPECIFIED_REGION]
    else:
        payload = admin_units
    if not isinstance(payload, list) or not payload:
        return [UNSPECIFIED_REGION]

    regions: list[str] = []
    seen: set[str] = set()
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        name = (
            entry.get("adm1_name")
            or entry.get("adm2_name")
            or entry.get("adm3_name")
            or entry.get("adm0_name")
        )
        if not isinstance(name, str):
            continue
        cleaned = " ".join(name.split())
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        regions.append(cleaned)
    return regions or [UNSPECIFIED_REGION]


def national_totals_by_year(data: pd.DataFrame) -> pd.DataFrame:
    """Sum event values by start year."""
    if data.empty:
        return pd.DataFrame(columns=["year", "value"])
    totals = (
        data.groupby("start_year", dropna=False, sort=False)["value"]
        .sum()
        .rename_axis("year")
        .reset_index(name="value")
    )
    totals["year"] = totals["year"].astype("Int64")
    return totals.sort_values("year", ascending=False, kind="mergesort").reset_index(
        drop=True
    )


def subnational_totals_by_year(data: pd.DataFrame) -> pd.DataFrame:
    """Sum values by year and region set.

    EM-DAT impact fields are event-level totals. When an event lists multiple admin
    units, those regions are kept together as one key and the event value is counted
    once - never divided across regions and never duplicated per region.
    """
    if data.empty:
        return pd.DataFrame(columns=["year", "region", "value"])

    rows: list[dict[str, Any]] = []
    for row in data.itertuples(index=False):
        year = getattr(row, "start_year", None)
        value = getattr(row, "value", None)
        if year is None or pd.isna(year) or value is None or pd.isna(value):
            continue
        regions = getattr(row, "regions", None)
        if not isinstance(regions, list) or not regions:
            regions = [UNSPECIFIED_REGION]
        rows.append(
            {
                "year": int(year),
                "region": format_region_set(regions),
                "value": float(value),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["year", "region", "value"])

    totals = (
        pd.DataFrame(rows)
        .groupby(["year", "region"], as_index=False, sort=False)
        .agg(value=("value", "sum"))
    )
    sorted_totals = totals.sort_values(
        by=["year", "region"],
        ascending=[False, True],
        kind="mergesort",
    )
    return sorted_totals.reset_index(drop=True)


def format_region_set(regions: list[str]) -> str:
    """Stable display key for one or more admin units belonging to the same total."""
    unique = list(dict.fromkeys(regions))
    if not unique:
        return UNSPECIFIED_REGION
    return "; ".join(unique)


def _filter_years(
    data: pd.DataFrame,
    year_start: int | None,
    year_end: int | None,
) -> pd.DataFrame:
    selected = data
    if year_start is not None:
        selected = selected[selected["Start Year"] >= year_start]
    if year_end is not None:
        selected = selected[selected["Start Year"] <= year_end]
    return selected
