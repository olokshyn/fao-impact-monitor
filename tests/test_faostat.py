from __future__ import annotations

import asyncio
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from pydantic import ValidationError

from fao_impact_monitor.data_source import (
    FAOSTAT,
    FAOSTATDataResult,
    FAOSTATDataSourceConfig,
    get_data_source,
)
from fao_impact_monitor.data_source.faostat import (
    FAOSTAT_CROPS_LIVESTOCK_TRADE,
    FAOSTAT_FOOD_SECURITY,
    FAOSTAT_INPUTS_LAND_USE,
    FAOSTAT_INVESTMENT_PROFILE,
    FAOSTAT_MACRO_INDICATORS,
    FAOSTAT_PRODUCTION_CROPS_LIVESTOCK,
    FAOSTAT_PRODUCTION_INDICES,
    FAOSTAT_RURAL_LIVELIHOODS,
    FAOSTAT_VALUE_OF_PRODUCTION,
)
from fao_impact_monitor.metric import Metric
from fao_impact_monitor.research_report import metric_path

LAND_INDICATOR = "Land area equipped for irrigation; share in cropland (%)"
LAND_MEMBER = "Inputs_LandUse_E_All_Data_(Normalized).csv"
LAND_HEADER = (
    "Area Code,Area Code (M49),Area,Item Code,Item,Element Code,Element,"
    "Year Code,Year,Unit,Value,Flag,Note\n"
)
TRADE_MEMBER = "Trade_CropsLivestock_E_All_Data_(Normalized).csv"
TRADE_HEADER = (
    "Area Code,Area Code (M49),Area,Item Code,Item Code (CPC),Item,"
    "Element Code,Element,Year,Unit,Value,Flag,Note\n"
)
FOOD_MEMBER = "Food_Security_Data_E_All_Data_(Normalized).csv"
FOOD_HEADER = (
    "Area Code,Area Code (M49),Area,Item Code,Item,Element Code,Element,"
    "Year Code,Year,Unit,Value,Flag\n"
)
RURAL_MEMBER = "Rural_Livelihoods_Indicators_E_All_Data_(Normalized).csv"
RURAL_HEADER = (
    "Survey Code,Survey,Indicator Code,Indicator,Element Code,Element,"
    "Qualifier Code,Qualifier,Source Code,Source,Unit,Value,Flag,Note\n"
)


def _metric(config: FAOSTATDataSourceConfig) -> Metric:
    return Metric(
        name="FAOSTAT metric",
        description="A configured FAOSTAT sub-indicator.",
        example="A value from FAOSTAT.",
        data_sources=[config],
    )


def _land_config(data_dir: Path) -> FAOSTATDataSourceConfig:
    return FAOSTATDataSourceConfig(
        source="FAOSTAT",
        dataset=FAOSTAT_INPUTS_LAND_USE,
        indicator=LAND_INDICATOR,
        item_code="6690",
        element="Share in Cropland",
        element_code="7252",
        unit="%",
        data_dir=data_dir,
    )


def _trade_config(data_dir: Path) -> FAOSTATDataSourceConfig:
    return FAOSTATDataSourceConfig(
        source="FAOSTAT",
        dataset=FAOSTAT_CROPS_LIVESTOCK_TRADE,
        indicator="Export value of crops and livestock products",
        item_code="1882",
        element="Export value",
        element_code="5922",
        unit="1000 USD",
        data_dir=data_dir,
    )


def _rural_config(data_dir: Path) -> FAOSTATDataSourceConfig:
    return FAOSTATDataSourceConfig(
        source="FAOSTAT",
        dataset=FAOSTAT_RURAL_LIVELIHOODS,
        indicator=(
            "Crop farm households with irrigation systems; "
            "share of total crop farm households (%)"
        ),
        indicator_code="24273",
        element="Value",
        element_code="6121",
        qualifier_code="N",
        source_code="3054",
        unit="%",
        data_dir=data_dir,
    )


def _food_security_config(data_dir: Path) -> FAOSTATDataSourceConfig:
    return FAOSTATDataSourceConfig(
        source="FAOSTAT",
        dataset=FAOSTAT_FOOD_SECURITY,
        indicator=(
            "Prevalence of moderate or severe food insecurity in the total "
            "population (3-year average)"
        ),
        item_code="210091",
        element="Value",
        element_code="6121",
        unit="%",
        data_dir=data_dir,
    )


def _write_land_archive(data_dir: Path) -> None:
    archive_path = (
        data_dir / "FAOSTAT_A-S_E" / "Inputs_LandUse_E_All_Data_(Normalized).zip"
    )
    archive_path.parent.mkdir(parents=True)
    rows = (
        "114,'404,Kenya,6690,Land area equipped for irrigation,7252,"
        "Share in Cropland,2022,2022,%,3.70,E,\n"
        "114,'404,Kenya,6690,Land area equipped for irrigation,7252,"
        "Share in Cropland,2023,2023,%,3.74,E,\n"
        "114,'404,Kenya,6690,Land area equipped for irrigation,5110,"
        "Area,2023,2023,1000 ha,240.0,E,\n"
        "100,'356,India,6690,Land area equipped for irrigation,7252,"
        "Share in Cropland,2023,2023,%,44.85,E,\n"
    )
    with ZipFile(archive_path, "w", ZIP_DEFLATED) as archive:
        archive.writestr(LAND_MEMBER, LAND_HEADER + rows)


def _write_trade_archive(data_dir: Path) -> None:
    archive_path = (
        data_dir / "FAOSTAT_T-Z_E" / "Trade_CropsLivestock_E_All_Data_(Normalized).zip"
    )
    archive_path.parent.mkdir(parents=True)
    rows = (
        "114,'404,Kenya,1882,'F1882,Crops and livestock products,"
        "5922,Export value,2021,1000 USD,1947000.0,A,\n"
        "114,'404,Kenya,1882,'F1882,Crops and livestock products,"
        "5622,Import value,2021,1000 USD,3100000.0,A,\n"
        "114,'404,Kenya,15,'0111,Wheat,5922,Export value,2021,"
        "1000 USD,100.0,A,\n"
        "2,'004,Afghanistan,1882,'F1882,Crops and livestock products,"
        "5922,Export value,2021,1000 USD,90000.0,A,\n"
    )
    with ZipFile(archive_path, "w", ZIP_DEFLATED) as archive:
        archive.writestr(TRADE_MEMBER, TRADE_HEADER + rows)


def _write_rural_archive(data_dir: Path) -> None:
    archive_path = (
        data_dir
        / "FAOSTAT_A-S_E"
        / "Rural_Livelihoods_Indicators_E_All_Data_(Normalized).zip"
    )
    archive_path.parent.mkdir(parents=True)
    rows = (
        '"404_2015","Kenya - 2015","24273","Crop farm households with '
        'irrigation systems; share of total crop farm households (%)","6121",'
        '"Value","N","National","3054","Household level","%","8.4",'
        '"E","2015-16"\n'
        '"404_2015","Kenya - 2015","24273","Crop farm households with '
        'irrigation systems; share of total crop farm households (%)","6121",'
        '"Value","R","Rural","3054","Household level","%","9.1",'
        '"E","2015-16"\n'
        '"356_2015","India - 2015","24273","Crop farm households with '
        'irrigation systems; share of total crop farm households (%)","6121",'
        '"Value","N","National","3054","Household level","%","42.0",'
        '"E","2015-16"\n'
    )
    with ZipFile(archive_path, "w", ZIP_DEFLATED) as archive:
        archive.writestr(RURAL_MEMBER, RURAL_HEADER + rows)


def _write_food_security_archive(data_dir: Path) -> None:
    archive_path = (
        data_dir / "FAOSTAT_A-S_E" / "Food_Security_Data_E_All_Data_(Normalized).zip"
    )
    archive_path.parent.mkdir(parents=True)
    rows = (
        "114,'404,Kenya,210091,Prevalence of moderate or severe food "
        "insecurity,6121,Value,20212023,2021-2023,%,36.8,E\n"
        "29,'108,Burundi,210091,Prevalence of moderate or severe food "
        "insecurity,6121,Value,20212023,2021-2023,%,<0.1,E\n"
    )
    with ZipFile(archive_path, "w", ZIP_DEFLATED) as archive:
        archive.writestr(FOOD_MEMBER, FOOD_HEADER + rows)


def test_faostat_is_registered() -> None:
    source = get_data_source("FAOSTAT")

    assert isinstance(source, FAOSTAT)
    assert source.source == "FAOSTAT"


def test_land_use_data_uses_exact_item_element_country_and_year(
    tmp_path: Path,
) -> None:
    _write_land_archive(tmp_path)
    config = _land_config(tmp_path)

    results = asyncio.run(
        FAOSTAT().get_data(
            _metric(config), config, "ken", year_start=2023, year_end=2023
        )
    )

    assert len(results) == 1
    result = results[0]
    assert isinstance(result, FAOSTATDataResult)
    assert result.title == LAND_INDICATOR
    assert result.metadata["dataset"] == FAOSTAT_INPUTS_LAND_USE
    assert list(result.data["area_m49"]) == ["404"]
    assert list(result.data["item_code"]) == ["6690"]
    assert list(result.data["element_code"]) == ["7252"]
    assert list(result.data["year"]) == [2023]
    assert list(result.data["value"]) == [3.74]


def test_trade_data_uses_aggregate_item_and_exact_element(tmp_path: Path) -> None:
    _write_trade_archive(tmp_path)
    config = _trade_config(tmp_path)

    results = asyncio.run(FAOSTAT().get_data(_metric(config), config, "KEN"))

    assert len(results) == 1
    result = results[0]
    assert isinstance(result, FAOSTATDataResult)
    assert result.metadata["dataset"] == FAOSTAT_CROPS_LIVESTOCK_TRADE
    assert list(result.data["area_m49"]) == ["404"]
    assert list(result.data["item_code"]) == ["1882"]
    assert list(result.data["element_code"]) == ["5922"]
    assert list(result.data["value"]) == [1_947_000.0]


def test_rural_data_uses_exact_indicator_national_value_and_survey_year(
    tmp_path: Path,
) -> None:
    _write_rural_archive(tmp_path)
    config = _rural_config(tmp_path)

    results = asyncio.run(FAOSTAT().get_data(_metric(config), config, "KEN"))

    assert len(results) == 1
    result = results[0]
    assert isinstance(result, FAOSTATDataResult)
    assert list(result.data["indicator_code"]) == ["24273"]
    assert list(result.data["element_code"]) == ["6121"]
    assert list(result.data["qualifier_code"]) == ["N"]
    assert list(result.data["source_code"]) == ["3054"]
    assert list(result.data["year"]) == [2015]
    assert list(result.data["period"]) == ["2015-16"]
    assert list(result.data["value"]) == [8.4]


def test_standard_data_supports_period_ranges_and_censored_values(
    tmp_path: Path,
) -> None:
    _write_food_security_archive(tmp_path)
    config = _food_security_config(tmp_path)

    results = asyncio.run(FAOSTAT().get_data(_metric(config), config, "KEN"))

    assert len(results) == 1
    result = results[0]
    assert isinstance(result, FAOSTATDataResult)
    assert list(result.data["year"]) == [2023]
    assert list(result.data["period"]) == ["2021-2023"]
    assert list(result.data["value"]) == [36.8]
    assert list(result.data["value_raw"]) == ["36.8"]

    censored_results = asyncio.run(FAOSTAT().get_data(_metric(config), config, "BDI"))
    assert len(censored_results) == 1
    censored = censored_results[0]
    assert isinstance(censored, FAOSTATDataResult)
    assert list(censored.data["value"]) == [0.1]
    assert list(censored.data["value_raw"]) == ["<0.1"]


def test_get_data_returns_empty_for_country_without_rows(tmp_path: Path) -> None:
    _write_land_archive(tmp_path)
    config = _land_config(tmp_path)

    results = asyncio.run(FAOSTAT().get_data(_metric(config), config, "ZWE"))

    assert results == []


def test_config_requires_item_and_element_codes() -> None:
    with pytest.raises(ValidationError):
        FAOSTATDataSourceConfig.model_validate(
            {
                "source": "FAOSTAT",
                "dataset": FAOSTAT_INPUTS_LAND_USE,
                "indicator": LAND_INDICATOR,
                "element": "Share in Cropland",
            }
        )


def test_get_data_rejects_invalid_year_range(tmp_path: Path) -> None:
    config = _land_config(tmp_path)

    with pytest.raises(ValueError, match="year_start .* must be <="):
        asyncio.run(
            FAOSTAT().get_data(
                _metric(config), config, "KEN", year_start=2022, year_end=2021
            )
        )


def test_el_nino_uses_best_available_faostat_series_for_all_final_metrics() -> None:
    expected_codes = {
        "Irrigated cropland": [
            (FAOSTAT_RURAL_LIVELIHOODS, "24280", "6121"),
            (FAOSTAT_INPUTS_LAND_USE, "6694", "5110"),
            (FAOSTAT_INPUTS_LAND_USE, "6690", "7252"),
            (FAOSTAT_INPUTS_LAND_USE, "6616", "5110"),
            (FAOSTAT_INPUTS_LAND_USE, "6611", "5110"),
        ],
        "Cropfarm households with irrigation system": [
            (FAOSTAT_RURAL_LIVELIHOODS, "24273", "6121"),
            (FAOSTAT_INPUTS_LAND_USE, "6611", "5110"),
            (FAOSTAT_INPUTS_LAND_USE, "6616", "5110"),
        ],
        "Credit obtained by households": [
            (FAOSTAT_RURAL_LIVELIHOODS, "24272", "6244"),
            (FAOSTAT_INVESTMENT_PROFILE, "23068", "61840"),
            (FAOSTAT_INVESTMENT_PROFILE, "23068", "61390"),
        ],
        "Agricultural income": [
            (FAOSTAT_RURAL_LIVELIHOODS, "24239", "6121"),
            (FAOSTAT_MACRO_INDICATORS, "22016", "6103"),
        ],
        "Farm income": [
            (FAOSTAT_RURAL_LIVELIHOODS, "24251", "6121"),
            (FAOSTAT_VALUE_OF_PRODUCTION, "2051", "152"),
        ],
        "Livestock farm households with one ruminant": [
            (FAOSTAT_RURAL_LIVELIHOODS, "24286", "6121"),
            (FAOSTAT_RURAL_LIVELIHOODS, "24289", "6121"),
            (FAOSTAT_PRODUCTION_CROPS_LIVESTOCK, "1746", "5111"),
            (FAOSTAT_PRODUCTION_CROPS_LIVESTOCK, "1749", "5111"),
        ],
        "Households with crop or livestock disease-related shocks": [
            (FAOSTAT_RURAL_LIVELIHOODS, "24322", "6121"),
            (FAOSTAT_RURAL_LIVELIHOODS, "24290", "6121"),
            (FAOSTAT_RURAL_LIVELIHOODS, "24284", "6121"),
            (FAOSTAT_PRODUCTION_INDICES, "2041", "432"),
            (FAOSTAT_PRODUCTION_INDICES, "2044", "432"),
        ],
        "Households with changed dietary patterns": [
            (FAOSTAT_RURAL_LIVELIHOODS, "24317", "6121"),
            (FAOSTAT_FOOD_SECURITY, "210091", "6121"),
        ],
        "Households with migrated members": [
            (FAOSTAT_RURAL_LIVELIHOODS, "24318", "6121")
        ],
        "Export value of crop and livestock products": [
            (FAOSTAT_CROPS_LIVESTOCK_TRADE, "1882", "5922")
        ],
        "Import value of crop and livestock products": [
            (FAOSTAT_CROPS_LIVESTOCK_TRADE, "1882", "5622")
        ],
    }
    metrics = Metric.from_use_case("use-cases/el-nino.json")
    configured = {
        metric.name: [
            FAOSTATDataSourceConfig.model_validate(config.model_dump())
            for config in metric.data_sources
        ]
        for metric in metrics
        if metric.name in expected_codes
    }

    assert set(configured) == set(expected_codes)
    for name, configs in configured.items():
        assert all(config.source == "FAOSTAT" for config in configs)
        assert [
            (
                config.dataset,
                config.item_code or config.indicator_code,
                config.element_code,
            )
            for config in configs
        ] == expected_codes[name]
        assert metric_path(
            next(metric for metric in metrics if metric.name == name)
        ) == ("faostat")


@pytest.mark.integration
@pytest.mark.parametrize("country_iso3", ["BRA", "IND", "KEN"])
def test_get_data_loads_downloaded_land_use_data(country_iso3: str) -> None:
    data_dir = Path("faostat_data")
    archive_path = (
        data_dir / "FAOSTAT_A-S_E" / "Inputs_LandUse_E_All_Data_(Normalized).zip"
    )
    if not archive_path.is_file():
        pytest.skip("Downloaded FAOSTAT Inputs: Land use archive is unavailable")

    config = _land_config(data_dir)
    results = asyncio.run(
        FAOSTAT().get_data(
            _metric(config), config, country_iso3, year_start=2023, year_end=2023
        )
    )

    assert len(results) == 1
    result = results[0]
    assert isinstance(result, FAOSTATDataResult)
    assert set(result.data["item_code"]) == {"6690"}
    assert set(result.data["element_code"]) == {"7252"}
    assert set(result.data["year"]) == {2023}


@pytest.mark.integration
def test_get_data_loads_downloaded_trade_data() -> None:
    data_dir = Path("faostat_data")
    archive_path = (
        data_dir / "FAOSTAT_T-Z_E" / "Trade_CropsLivestock_E_All_Data_(Normalized).zip"
    )
    if not archive_path.is_file():
        pytest.skip("Downloaded FAOSTAT trade archive is unavailable")

    config = _trade_config(data_dir)
    results = asyncio.run(
        FAOSTAT().get_data(
            _metric(config), config, "KEN", year_start=2023, year_end=2023
        )
    )

    assert len(results) == 1
    result = results[0]
    assert isinstance(result, FAOSTATDataResult)
    assert set(result.data["item_code"]) == {"1882"}
    assert set(result.data["element_code"]) == {"5922"}
    assert set(result.data["year"]) == {2023}


@pytest.mark.integration
def test_downloaded_el_nino_final_metrics_return_faostat_data_for_kenya() -> None:
    metrics = Metric.from_use_case("use-cases/el-nino.json")
    irrigated_index = next(
        index
        for index, metric in enumerate(metrics)
        if metric.name == "Irrigated cropland"
    )
    cropland_index = next(
        index for index, metric in enumerate(metrics) if metric.name == "Cropland"
    )
    source = FAOSTAT()

    for metric in metrics[irrigated_index:cropland_index]:
        assert metric.data_sources
        assert all(config.source == "FAOSTAT" for config in metric.data_sources)
        results: list[FAOSTATDataResult] = []
        for config in metric.data_sources:
            for result in asyncio.run(source.get_data(metric, config, "KEN")):
                assert isinstance(result, FAOSTATDataResult)
                results.append(result)

        assert results, f"No FAOSTAT data returned for {metric.name}"
        if metric.name != "Households with migrated members":
            assert sum(len(result.data) for result in results) > 1
