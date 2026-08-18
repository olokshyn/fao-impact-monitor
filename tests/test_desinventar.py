from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from fao_impact_monitor.data_source import (
    DesInventar,
    DesInventarDataResult,
    DesInventarDataSourceConfig,
    get_data_source,
)
from fao_impact_monitor.data_source.data_source import _DATA_SOURCE_CLS_REGISTRY
from fao_impact_monitor.data_source.desinventar import (
    filter_fichas_by_event_keywords,
    local_event_keyword_synonyms,
    national_totals_by_year,
    resolve_country_xml_path,
    subnational_totals_by_year,
)
from fao_impact_monitor.metric import Metric
from fao_impact_monitor.research_report import metric_path


def _metric(config: DesInventarDataSourceConfig) -> Metric:
    return Metric(
        name="DesInventar metric",
        description="A configured DesInventar indicator.",
        example="A value from DesInventar.",
        data_sources=[config],
    )


def _config(
    data_path: Path,
    indicator: str = "muertos",
    *,
    event_keywords: list[str] | None = None,
) -> DesInventarDataSourceConfig:
    kwargs: dict[str, object] = {
        "source": "DesInventar",
        "indicator": indicator,
        "unit": "persons",
        "exclusive": True,
        "data_path": data_path,
    }
    if event_keywords is not None:
        kwargs["event_keywords"] = event_keywords
    return DesInventarDataSourceConfig.model_validate(kwargs)


def _write_fixture_xml(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """<?xml version="1.0" encoding="UTF-8" ?>
<DESINVENTAR>
  <eventos><TR><serial>1</serial><nombre>FLOOD</nombre></TR></eventos>
  <fichas>
    <TR>
      <clave>1</clave><evento>FLOOD</evento>
      <fechano>2023</fechano><fechames>4</fechames><fechadia>10</fechadia>
      <name0>Nairobi</name0><lugar>Nairobi</lugar>
      <muertos>10</muertos><desaparece></desaparece>
      <heridos>0</heridos><afectados>100</afectados>
      <damnificados>0</damnificados><evacuados>100</evacuados>
      <reubicados>0</reubicados><vivdest>5</vivdest><vivafec>0</vivafec>
      <nescuelas>0</nescuelas><nhospitales></nhospitales>
      <kmvias>0</kmvias><transporte>0</transporte>
      <energia>-1</energia><acueducto>0</acueducto>
      <alcantarillado>0</alcantarillado><comunicaciones>1</comunicaciones>
    </TR>
    <TR>
      <clave>2</clave><evento>DROUGHT</evento>
      <fechano>2016</fechano><fechames>5</fechames><fechadia>1</fechadia>
      <name0>Turkana</name0><lugar></lugar>
      <muertos>0</muertos><desaparece>2</desaparece>
      <heridos>0</heridos><afectados>0</afectados>
      <damnificados>50</damnificados><evacuados>0</evacuados>
      <reubicados>650</reubicados><vivdest>0</vivdest><vivafec>3</vivafec>
      <nescuelas>1</nescuelas><nhospitales>0</nhospitales>
      <kmvias>12</kmvias><transporte>1</transporte>
      <energia>0</energia><acueducto>-1</acueducto>
      <alcantarillado>0</alcantarillado><comunicaciones>0</comunicaciones>
    </TR>
    <TR>
      <clave>3</clave><evento>ROAD ACCIDENT</evento>
      <fechano>2020</fechano><fechames>1</fechames><fechadia>1</fechadia>
      <name0>Kisumu</name0><lugar>Holo</lugar>
      <muertos>99</muertos><desaparece>0</desaparece>
      <heridos>5</heridos><afectados>0</afectados>
      <damnificados>0</damnificados><evacuados>0</evacuados>
      <reubicados>0</reubicados><vivdest>0</vivdest><vivafec>0</vivafec>
      <nescuelas>0</nescuelas><nhospitales>0</nhospitales>
      <kmvias>0</kmvias><transporte>1</transporte>
      <energia>0</energia><acueducto>0</acueducto>
      <alcantarillado>0</alcantarillado><comunicaciones>0</comunicaciones>
    </TR>
    <TR>
      <clave>4</clave><evento></evento>
      <fechano>2021</fechano><fechames>2</fechames><fechadia>2</fechadia>
      <name0></name0><lugar></lugar>
      <muertos>7</muertos>
    </TR>
  </fichas>
</DESINVENTAR>
""",
        encoding="utf-8",
    )
    return path


def test_desinventar_registry() -> None:
    assert _DATA_SOURCE_CLS_REGISTRY["DesInventar"] is DesInventar
    source = get_data_source("DesInventar")
    assert isinstance(source, DesInventar)
    assert source.source == "DesInventar"


def test_resolve_country_xml_path() -> None:
    assert resolve_country_xml_path("KEN") == Path(
        "datasets/DesInventar/DI_export_ken/DI_export_ken.xml"
    )
    custom = Path("/tmp/custom.xml")
    assert resolve_country_xml_path("ETH", data_path=custom) == custom


def _write_spanish_fixture_xml(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """<?xml version="1.0" encoding="UTF-8" ?>
<DESINVENTAR>
  <fichas>
    <TR>
      <clave>10</clave><evento>Inundación</evento>
      <fechano>2022</fechano><fechames>6</fechames><fechadia>1</fechadia>
      <name0>Escuintla</name0><lugar>Escuintla</lugar>
      <muertos>5</muertos><desaparece></desaparece>
      <heridos>0</heridos><afectados>20</afectados>
      <damnificados>0</damnificados><evacuados>0</evacuados>
      <reubicados>0</reubicados><vivdest>0</vivdest><vivafec>0</vivafec>
      <nescuelas>0</nescuelas><nhospitales>0</nhospitales>
      <kmvias>0</kmvias><transporte>0</transporte>
      <energia>0</energia><acueducto>0</acueducto>
      <alcantarillado>0</alcantarillado><comunicaciones>0</comunicaciones>
    </TR>
    <TR>
      <clave>11</clave><evento>Accidente</evento>
      <fechano>2021</fechano><fechames>3</fechames><fechadia>2</fechadia>
      <name0>Guatemala</name0><lugar></lugar>
      <muertos>9</muertos><desaparece>0</desaparece>
      <heridos>0</heridos><afectados>0</afectados>
      <damnificados>0</damnificados><evacuados>0</evacuados>
      <reubicados>0</reubicados><vivdest>0</vivdest><vivafec>0</vivafec>
      <nescuelas>0</nescuelas><nhospitales>0</nhospitales>
      <kmvias>0</kmvias><transporte>0</transporte>
      <energia>0</energia><acueducto>0</acueducto>
      <alcantarillado>0</alcantarillado><comunicaciones>0</comunicaciones>
    </TR>
  </fichas>
</DESINVENTAR>
""",
        encoding="utf-8",
    )
    return path


def test_local_event_keyword_synonyms_cover_observed_languages() -> None:
    synonyms = local_event_keyword_synonyms(["flood", "drought"])
    folded = {value.casefold() for value in synonyms}
    assert "inundación" in folded
    assert "inundação" in folded or "inundacao" in folded
    assert "inondation" in folded
    assert "inundasaun" in folded
    assert "sequía" in folded or "sequia" in folded
    assert "seca" in folded
    assert "sécheresse" in folded or "secheresse" in folded


def test_matches_event_keywords_avoids_rain_in_terrain() -> None:
    from fao_impact_monitor.data_source.desinventar import matches_event_keywords

    assert not matches_event_keywords("GLISSEMENT DE TERRAIN", ["rain"])
    assert matches_event_keywords("THUNDERSTORM", ["storm"])
    assert matches_event_keywords("FLOODS", ["flood"])
    assert matches_event_keywords("PLUIES EXTREME", ["pluies"])


def test_filter_fichas_english_first_then_local_synonyms(tmp_path: Path) -> None:
    from fao_impact_monitor.data_source.desinventar import (
        DEFAULT_EVENT_KEYWORDS,
        parse_fichas_xml,
    )

    english = parse_fichas_xml(_write_fixture_xml(tmp_path / "en.xml"))
    matched_en = filter_fichas_by_event_keywords(english, list(DEFAULT_EVENT_KEYWORDS))
    assert list(matched_en["clave"]) == ["1", "2"]

    spanish = parse_fichas_xml(_write_spanish_fixture_xml(tmp_path / "es.xml"))
    # Direct English keywords miss Spanish event names.
    assert filter_fichas_by_event_keywords(spanish, list(DEFAULT_EVENT_KEYWORDS)).empty
    local = local_event_keyword_synonyms(list(DEFAULT_EVENT_KEYWORDS))
    matched_es = filter_fichas_by_event_keywords(spanish, local)
    assert list(matched_es["clave"]) == ["10"]


def test_desinventar_spanish_evento_fallback(tmp_path: Path) -> None:
    xml_path = _write_spanish_fixture_xml(tmp_path / "DI_export_gtm.xml")
    config = _config(xml_path, "muertos")
    results = asyncio.run(DesInventar().get_data(_metric(config), config, "GTM"))
    assert len(results) == 1
    result = results[0]
    assert isinstance(result, DesInventarDataResult)
    assert list(result.data["event_id"]) == ["10"]
    assert list(result.data["disaster_type"]) == ["Inundación"]
    assert (
        result.metadata["event_keywords_applied"] != result.metadata["event_keywords"]
    )
    assert "Accidente" not in set(result.data["disaster_type"])


def test_desinventar_filters_indicator_and_keywords(tmp_path: Path) -> None:
    xml_path = _write_fixture_xml(tmp_path / "DI_export_ken.xml")
    config = _config(xml_path, "muertos")
    results = asyncio.run(DesInventar().get_data(_metric(config), config, "KEN"))
    assert len(results) == 1
    result = results[0]
    assert isinstance(result, DesInventarDataResult)
    assert result.metadata["indicator"] == "muertos"
    # ROAD ACCIDENT / empty evento excluded; DROUGHT zero dropped by default.
    assert list(result.data["event_id"]) == ["1"]
    assert list(result.data["value"]) == [10.0]
    assert list(result.data["regions"]) == [["Nairobi"]]
    assert "ROAD ACCIDENT" not in set(result.data["disaster_type"])


def test_desinventar_drop_zero_values_default_and_override(tmp_path: Path) -> None:
    xml_path = _write_fixture_xml(tmp_path / "fixture.xml")
    source = DesInventar()
    assert source.DROP_ZERO_VALUES is True

    deaths = asyncio.run(
        source.get_data(_metric(_config(xml_path, "muertos")), _config(xml_path), "KEN")
    )[0]
    assert isinstance(deaths, DesInventarDataResult)
    assert 0.0 not in list(deaths.data["value"])
    assert list(deaths.data["value"]) == [10.0]

    missing = asyncio.run(
        source.get_data(
            _metric(_config(xml_path, "desaparece")),
            _config(xml_path, "desaparece"),
            "KEN",
        )
    )[0]
    assert isinstance(missing, DesInventarDataResult)
    # Empty desaparece on FLOOD is N/A and dropped; DROUGHT has 2.
    assert list(missing.data["value"]) == [2.0]

    # Explicit zeros only (nhospitales) yield no rows when zeros are dropped.
    hospitals = asyncio.run(
        source.get_data(
            _metric(_config(xml_path, "nhospitales")),
            _config(xml_path, "nhospitales"),
            "KEN",
        )
    )
    assert hospitals == []

    source.DROP_ZERO_VALUES = False
    kept = asyncio.run(
        source.get_data(_metric(_config(xml_path, "muertos")), _config(xml_path), "KEN")
    )[0]
    assert isinstance(kept, DesInventarDataResult)
    assert list(kept.data["value"]) == [10.0, 0.0]


def test_desinventar_flag_indicators(tmp_path: Path) -> None:
    xml_path = _write_fixture_xml(tmp_path / "fixture.xml")
    config = _config(xml_path, "energia", event_keywords=["flood", "drought"])
    result = asyncio.run(DesInventar().get_data(_metric(config), config, "KEN"))[0]
    assert isinstance(result, DesInventarDataResult)
    # FLOOD: -1 -> 1; DROUGHT: 0 dropped by default
    by_event = dict(zip(result.data["event_id"], result.data["value"], strict=True))
    assert by_event == {"1": 1.0}


def test_desinventar_national_and_subnational_totals(tmp_path: Path) -> None:
    xml_path = _write_fixture_xml(tmp_path / "fixture.xml")
    config = _config(xml_path, "muertos")
    result = asyncio.run(DesInventar().get_data(_metric(config), config, "KEN"))[0]
    assert isinstance(result, DesInventarDataResult)
    national = national_totals_by_year(result.data)
    subnational = subnational_totals_by_year(result.data)

    assert list(national["year"]) == [2023]
    assert list(national["value"]) == [10.0]
    assert list(subnational.itertuples(index=False, name=None)) == [
        (2023, "Nairobi", 10.0),
    ]
    assert subnational["value"].sum() == national["value"].sum()


def test_desinventar_year_filter_and_empty_country(tmp_path: Path) -> None:
    xml_path = _write_fixture_xml(tmp_path / "fixture.xml")
    config = _config(xml_path, "reubicados")
    source = DesInventar()
    results = asyncio.run(
        source.get_data(_metric(config), config, "KEN", year_start=2015, year_end=2017)
    )
    assert len(results) == 1
    assert isinstance(results[0], DesInventarDataResult)
    assert list(results[0].data["value"]) == [650.0]

    missing_path = tmp_path / "missing.xml"
    missing = _config(missing_path)
    assert asyncio.run(source.get_data(_metric(missing), missing, "TZA")) == []


def test_desinventar_unknown_indicator(tmp_path: Path) -> None:
    xml_path = _write_fixture_xml(tmp_path / "fixture.xml")
    bad = _config(xml_path, "not_a_field")
    with pytest.raises(ValueError, match="Unknown DesInventar indicator"):
        asyncio.run(DesInventar().get_data(_metric(bad), bad, "KEN"))


def test_desinventar_metric_path() -> None:
    config = DesInventarDataSourceConfig(
        source="DesInventar",
        indicator="muertos",
        exclusive=True,
    )
    assert metric_path(_metric(config)) == "desinventar"


def test_el_nino_desinventar_and_emdat_metrics_configured() -> None:
    metrics = Metric.from_use_case(Path("use-cases/el-nino.json"))
    by_name = {metric.name: metric for metric in metrics}

    deaths = by_name["Deaths and missing persons"]
    assert metric_path(deaths) == "structured"
    assert [c.source for c in deaths.data_sources] == [
        "DesInventar",
        "DesInventar",
        "EMDAT",
    ]
    assert [c.model_dump()["indicator"] for c in deaths.data_sources] == [
        "muertos",
        "desaparece",
        "Total Deaths",
    ]

    affected = by_name["People affected, injured and requiring assistance"]
    assert [c.source for c in affected.data_sources[:3]] == ["EMDAT", "EMDAT", "EMDAT"]
    assert [c.source for c in affected.data_sources[3:]] == [
        "DesInventar",
        "DesInventar",
        "DesInventar",
    ]

    displaced = by_name["Displaced, evacuated and relocated people"]
    assert [c.model_dump()["indicator"] for c in displaced.data_sources] == [
        "evacuados",
        "reubicados",
        "No. Homeless",
    ]

    housing = by_name["Housing damaged and destroyed"]
    assert [c.model_dump()["indicator"] for c in housing.data_sources] == [
        "vivdest",
        "vivafec",
        "No. Homeless",
        "Reconstruction Costs, Adjusted ('000 US$)",
        "Insured Damage, Adjusted ('000 US$)",
        "Total Damage, Adjusted ('000 US$)",
    ]

    education = by_name["Education and health facilities affected"]
    assert metric_path(education) == "desinventar"
    assert [c.model_dump()["indicator"] for c in education.data_sources] == [
        "nescuelas",
        "nhospitales",
    ]

    infrastructure = by_name[
        "Roads, bridges, energy, water and communications affected"
    ]
    assert metric_path(infrastructure) == "desinventar"
    assert [c.model_dump()["indicator"] for c in infrastructure.data_sources] == [
        "kmvias",
        "transporte",
        "energia",
        "acueducto",
        "alcantarillado",
        "comunicaciones",
    ]


def _real_xml_path(country_iso3: str) -> Path:
    return resolve_country_xml_path(country_iso3)


@pytest.mark.integration
@pytest.mark.parametrize("country_iso3", ["KEN", "ETH", "GTM", "MOZ", "DJI", "TLS"])
def test_get_data_loads_real_desinventar_deaths(country_iso3: str) -> None:
    xml_path = _real_xml_path(country_iso3)
    if not xml_path.is_file():
        pytest.skip(f"DesInventar XML unavailable: {xml_path}")

    config = DesInventarDataSourceConfig(
        source="DesInventar",
        indicator="muertos",
        unit="persons",
        exclusive=True,
    )
    results = asyncio.run(DesInventar().get_data(_metric(config), config, country_iso3))
    assert len(results) == 1
    result = results[0]
    assert isinstance(result, DesInventarDataResult)
    assert result.metadata["country_iso3"] == country_iso3
    assert not result.data.empty
    assert result.data["value"].notna().all()
    assert {
        "event_id",
        "disaster_type",
        "start_year",
        "value",
        "regions",
    }.issubset(result.data.columns)
    disaster_types = {str(value) for value in result.data["disaster_type"]}
    assert "" not in disaster_types
    assert "ROAD ACCIDENT" not in disaster_types
