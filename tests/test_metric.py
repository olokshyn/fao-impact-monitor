from pathlib import Path

import pytest

from fao_impact_monitor.metric import Metric


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
    }


def test_from_use_case_loads_el_nino_metrics() -> None:
    metrics = Metric.from_use_case(Path("use-cases/el-nino.json"))
    assert len(metrics) >= 1
    assert metrics[0].name == "Agriculture share of GDP"
    # Entries without unit / data_sources still parse.
    by_name = {m.name: m for m in metrics}
    assert by_name["Subsequent hazards"].unit == ""
    assert by_name["Subsequent hazards"].data_sources == []


def test_from_use_case_requires_metrics_list(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"name": "x"}', encoding="utf-8")
    with pytest.raises(TypeError, match="must contain a 'metrics' list"):
        Metric.from_use_case(path)
