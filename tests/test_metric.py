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
