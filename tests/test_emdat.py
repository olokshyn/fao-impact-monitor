from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd
import pytest

from fao_impact_monitor.data_source import (
    EMDAT,
    EmDatDataResult,
    EmDatDataSourceConfig,
    get_data_source,
)
from fao_impact_monitor.data_source.data_source import _DATA_SOURCE_CLS_REGISTRY
from fao_impact_monitor.metric import Metric
from fao_impact_monitor.research_report import metric_path


def _metric(config: EmDatDataSourceConfig) -> Metric:
    return Metric(
        name="EM-DAT metric",
        description="A configured EM-DAT indicator.",
        example="A value from EM-DAT.",
        data_sources=[config],
    )


def _config(data_path: Path, indicator: str = "Total Deaths") -> EmDatDataSourceConfig:
    return EmDatDataSourceConfig(
        source="EMDAT",
        indicator=indicator,
        unit="persons",
        exclusive=True,
        data_path=data_path,
    )


def _write_workbook(path: Path) -> Path:
    data = pd.DataFrame(
        [
            {
                "DisNo.": "2023-0001-KEN",
                "Historic": "No",
                "Classification Key": "nat-hyd-flo-flo",
                "Disaster Group": "Natural",
                "Disaster Subgroup": "Hydrological",
                "Disaster Type": "Flood",
                "Disaster Subtype": "Flood (General)",
                "Event Name": None,
                "ISO": "KEN",
                "Country": "Kenya",
                "Location": "Nairobi",
                "Admin Units": (
                    '[{"adm1_code":1,"adm1_name":"Nairobi"},'
                    '{"adm1_code":2,"adm1_name":"Kiambu"}]'
                ),
                "Start Year": 2023,
                "Start Month": 4,
                "Start Day": 10,
                "End Year": 2023,
                "End Month": 4,
                "End Day": 20,
                "Total Deaths": 178,
                "No. Injured": 242,
                "No. Affected": 695255,
                "No. Homeless": None,
                "Total Affected": 695497,
                "Reconstruction Costs, Adjusted ('000 US$)": None,
                "Insured Damage, Adjusted ('000 US$)": None,
                "Total Damage, Adjusted ('000 US$)": 1200,
            },
            {
                "DisNo.": "2016-0002-KEN",
                "Historic": "No",
                "Classification Key": "nat-hyd-flo-flo",
                "Disaster Group": "Natural",
                "Disaster Subgroup": "Hydrological",
                "Disaster Type": "Flood",
                "Disaster Subtype": "Flood (General)",
                "Event Name": None,
                "ISO": "KEN",
                "Country": "Kenya",
                "Location": "Coast",
                "Admin Units": '[{"adm1_code":3,"adm1_name":"Coast"}]',
                "Start Year": 2016,
                "Start Month": 5,
                "Start Day": 1,
                "End Year": 2016,
                "End Month": 5,
                "End Day": 15,
                "Total Deaths": 3,
                "No. Injured": None,
                "No. Affected": None,
                "No. Homeless": 1000,
                "Total Affected": 1000,
                "Reconstruction Costs, Adjusted ('000 US$)": 50,
                "Insured Damage, Adjusted ('000 US$)": 10,
                "Total Damage, Adjusted ('000 US$)": 80,
            },
            {
                "DisNo.": "2020-0003-UGA",
                "Historic": "No",
                "Classification Key": "nat-cli-dro-dro",
                "Disaster Group": "Natural",
                "Disaster Subgroup": "Climatological",
                "Disaster Type": "Drought",
                "Disaster Subtype": "Drought",
                "Event Name": None,
                "ISO": "UGA",
                "Country": "Uganda",
                "Location": "North",
                "Admin Units": '[{"adm1_code":4,"adm1_name":"Northern"}]',
                "Start Year": 2020,
                "Start Month": 1,
                "Start Day": 1,
                "End Year": 2020,
                "End Month": 12,
                "End Day": 31,
                "Total Deaths": 12,
                "No. Injured": None,
                "No. Affected": 50000,
                "No. Homeless": None,
                "Total Affected": 50000,
                "Reconstruction Costs, Adjusted ('000 US$)": None,
                "Insured Damage, Adjusted ('000 US$)": None,
                "Total Damage, Adjusted ('000 US$)": None,
            },
            {
                "DisNo.": "2010-0004-KEN",
                "Historic": "No",
                "Classification Key": "nat-tec-ind-ind",
                "Disaster Group": "Technological",
                "Disaster Subgroup": "Industrial accident",
                "Disaster Type": "Industrial accident",
                "Disaster Subtype": "Explosion",
                "Event Name": None,
                "ISO": "KEN",
                "Country": "Kenya",
                "Location": "Nairobi",
                "Start Year": 2010,
                "Start Month": 1,
                "Start Day": 1,
                "End Year": 2010,
                "End Month": 1,
                "End Day": 1,
                "Total Deaths": 99,
                "No. Injured": None,
                "No. Affected": None,
                "No. Homeless": None,
                "Total Affected": None,
                "Reconstruction Costs, Adjusted ('000 US$)": None,
                "Insured Damage, Adjusted ('000 US$)": None,
                "Total Damage, Adjusted ('000 US$)": None,
            },
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        data.to_excel(writer, sheet_name="EM-DAT Data", index=False)
        pd.DataFrame(
            {
                "Source:": ["EM-DAT, CRED / UCLouvain, Brussels, Belgium"],
                "EM-DAT, CRED / UCLouvain, Brussels, Belgium": ["https://www.emdat.be"],
            }
        ).to_excel(writer, sheet_name="EM-DAT Info", index=False)
    return path


def test_emdat_registry() -> None:
    assert _DATA_SOURCE_CLS_REGISTRY["EMDAT"] is EMDAT
    source = get_data_source("EMDAT")
    assert isinstance(source, EMDAT)
    assert source.source == "EMDAT"


def test_emdat_filters_country_indicator_and_subgroup(tmp_path: Path) -> None:
    workbook = _write_workbook(tmp_path / "emdat.xlsx")
    config = _config(workbook, "Total Deaths")
    results = asyncio.run(EMDAT().get_data(_metric(config), config, "KEN"))
    assert len(results) == 1
    result = results[0]
    assert isinstance(result, EmDatDataResult)
    assert result.metadata["indicator"] == "Total Deaths"
    assert list(result.data["dis_no"]) == ["2023-0001-KEN", "2016-0002-KEN"]
    assert list(result.data["value"]) == [178.0, 3.0]
    assert list(result.data["regions"]) == [["Nairobi", "Kiambu"], ["Coast"]]


def test_emdat_national_and_subnational_totals(tmp_path: Path) -> None:
    from fao_impact_monitor.data_source.emdat import (
        national_totals_by_year,
        subnational_totals_by_year,
    )

    workbook = _write_workbook(tmp_path / "emdat.xlsx")
    config = _config(workbook, "Total Deaths")
    result = asyncio.run(EMDAT().get_data(_metric(config), config, "KEN"))[0]
    assert isinstance(result, EmDatDataResult)
    national = national_totals_by_year(result.data)
    subnational = subnational_totals_by_year(result.data)

    assert list(national["year"]) == [2023, 2016]
    assert list(national["value"]) == [178.0, 3.0]
    assert list(subnational.itertuples(index=False, name=None)) == [
        (2023, "Nairobi; Kiambu", 178.0),
        (2016, "Coast", 3.0),
    ]
    assert subnational["value"].sum() == national["value"].sum()


def test_emdat_year_filter_and_empty_country(tmp_path: Path) -> None:
    workbook = _write_workbook(tmp_path / "emdat.xlsx")
    config = _config(workbook, "No. Homeless")
    source = EMDAT()
    results = asyncio.run(
        source.get_data(_metric(config), config, "KEN", year_start=2015, year_end=2017)
    )
    assert len(results) == 1
    assert isinstance(results[0], EmDatDataResult)
    assert list(results[0].data["value"]) == [1000.0]

    empty = asyncio.run(source.get_data(_metric(config), config, "TZA"))
    assert empty == []


def test_emdat_unknown_indicator_and_missing_file(tmp_path: Path) -> None:
    workbook = _write_workbook(tmp_path / "emdat.xlsx")
    bad = _config(workbook, "Not A Column")
    with pytest.raises(ValueError, match="Unknown EM-DAT indicator"):
        asyncio.run(EMDAT().get_data(_metric(bad), bad, "KEN"))

    missing = _config(tmp_path / "missing.xlsx")
    assert asyncio.run(EMDAT().get_data(_metric(missing), missing, "KEN")) == []


def test_emdat_metric_path() -> None:
    config = EmDatDataSourceConfig(
        source="EMDAT",
        indicator="Total Deaths",
        exclusive=True,
    )
    assert metric_path(_metric(config)) == "emdat"


def test_el_nino_emdat_metrics_configured() -> None:
    metrics = Metric.from_use_case(Path("use-cases/el-nino.json"))
    by_name = {metric.name: metric for metric in metrics}

    deaths = by_name["Deaths and missing persons"]
    assert any(config.source == "EMDAT" for config in deaths.data_sources)
    assert deaths.data_sources[-1].model_dump()["indicator"] == "Total Deaths"

    affected = by_name["People affected, injured and requiring assistance"]
    assert [config.source for config in affected.data_sources[:3]] == [
        "EMDAT",
        "EMDAT",
        "EMDAT",
    ]
    assert [
        config.model_dump()["indicator"] for config in affected.data_sources[:3]
    ] == [
        "No. Injured",
        "No. Affected",
        "Total Affected",
    ]

    displaced = by_name["Displaced, evacuated and relocated people"]
    assert any(
        config.source == "EMDAT"
        and config.model_dump().get("indicator") == "No. Homeless"
        for config in displaced.data_sources
    )

    housing = by_name["Housing damaged and destroyed"]
    emdat_indicators = [
        config.model_dump()["indicator"]
        for config in housing.data_sources
        if config.source == "EMDAT"
    ]
    assert emdat_indicators == [
        "No. Homeless",
        "Reconstruction Costs, Adjusted ('000 US$)",
        "Insured Damage, Adjusted ('000 US$)",
        "Total Damage, Adjusted ('000 US$)",
    ]
