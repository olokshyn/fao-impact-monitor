from pathlib import Path

import pytest

from fao_impact_monitor.data_source.data_source_config import DataSourceConfig
from fao_impact_monitor.metric import Metric
from fao_impact_monitor.metric.metric import resolve_data_sources


def test_metric_preserves_provider_config_fields() -> None:
    metric = Metric.model_validate(
        {
            "name": "Agriculture share of GDP",
            "description": "The share of agriculture in the total GDP of a country.",
            "example": "Agriculture contributed 24.3% of GDP in 2023.",
            "unit": "%",
            "data_sources": [
                {
                    "source": "WorldBank",
                    "indicator": "NV.AGR.TOTL.ZS",
                    "unit": "%",
                }
            ],
        }
    )

    assert metric.name == "Agriculture share of GDP"
    assert len(metric.data_sources) == 1
    assert metric.data_sources[0].model_dump() == {
        "source": "WorldBank",
        "indicator": "NV.AGR.TOTL.ZS",
        "unit": "%",
        "exclusive": False,
    }


def test_resolve_data_sources_merges_defaults() -> None:
    defaults = [
        DataSourceConfig.model_validate(
            {"source": "FAORepository", "root_url": "https://a"}
        )
    ]
    extras = [DataSourceConfig(source="Tellus")]
    resolved = resolve_data_sources(defaults, extras)
    assert [s.source for s in resolved] == ["FAORepository", "Tellus"]


def test_resolve_data_sources_exclusive_drops_defaults_and_non_exclusive() -> None:
    defaults = [
        DataSourceConfig.model_validate(
            {"source": "FAORepository", "root_url": "https://a"}
        )
    ]
    extras = [
        DataSourceConfig(source="Tellus"),
        DataSourceConfig.model_validate(
            {
                "source": "WorldBank",
                "indicator": "NV.AGR.TOTL.ZS",
                "exclusive": True,
            }
        ),
    ]
    resolved = resolve_data_sources(defaults, extras)
    assert len(resolved) == 1
    assert resolved[0].source == "WorldBank"
    assert resolved[0].exclusive is True


def test_from_use_case_loads_el_nino_metrics() -> None:
    metrics = Metric.from_use_case(Path("use-cases/el-nino.json"))
    assert len(metrics) >= 1
    assert metrics[0].name == "Agriculture share of GDP"

    by_name = {m.name: m for m in metrics}
    gdp = by_name["Agriculture share of GDP"]
    assert len(gdp.data_sources) == 1
    assert gdp.data_sources[0].source == "WorldBank"
    assert gdp.data_sources[0].exclusive is True

    labour = by_name["Agriculture share of Labour"]
    assert [s.source for s in labour.data_sources] == ["WorldBank"]

    cropland = by_name["Cropland"]
    assert len(cropland.data_sources) == 3
    assert all(s.source == "FAORepository" for s in cropland.data_sources)

    hazards = by_name["Subsequent hazards"]
    assert hazards.unit == ""
    assert len(hazards.data_sources) == 3


def test_from_use_case_requires_metrics_list(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"name": "x"}', encoding="utf-8")
    with pytest.raises(TypeError, match="must contain a 'metrics' list"):
        Metric.from_use_case(path)


def test_from_use_case_temp_exclusive_and_defaults(tmp_path: Path) -> None:
    path = tmp_path / "uc.json"
    path.write_text(
        """
        {
          "data_sources": [{"source": "FAORepository", "root_url": "https://d"}],
          "metrics": [
            {
              "name": "A",
              "description": "d",
              "example": "e",
              "unit": "%",
              "data_sources": [
                {"source": "WorldBank", "indicator": "X", "exclusive": true}
              ]
            },
            {
              "name": "B",
              "description": "d",
              "example": "e",
              "unit": "%"
            }
          ]
        }
        """,
        encoding="utf-8",
    )
    metrics = Metric.from_use_case(path)
    assert [s.source for s in metrics[0].data_sources] == ["WorldBank"]
    assert [s.source for s in metrics[1].data_sources] == ["FAORepository"]
