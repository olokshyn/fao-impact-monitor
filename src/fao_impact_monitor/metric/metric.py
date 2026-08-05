from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, Field

from fao_impact_monitor.data_source.data_source_config import DataSourceConfig


def resolve_data_sources(
    defaults: list[DataSourceConfig],
    metric_sources: list[DataSourceConfig],
) -> list[DataSourceConfig]:
    """Merge use-case default sources with per-metric sources.

    If any metric-level source has ``exclusive=True``, only those exclusive
    sources are kept (defaults and non-exclusive metric sources are dropped).
    Otherwise the result is ``defaults + metric_sources``.
    """
    exclusive = [s for s in metric_sources if s.exclusive]
    if exclusive:
        return list(exclusive)
    return list(defaults) + list(metric_sources)


class Metric(BaseModel):
    name: str
    description: str
    example: str
    unit: str = ""
    data_sources: list[DataSourceConfig] = Field(default_factory=list)

    @classmethod
    def from_use_case(cls, use_case_path: Path | str) -> list[Self]:
        path = Path(use_case_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_metrics = payload.get("metrics")
        if not isinstance(raw_metrics, list):
            raise TypeError(f"Use-case file {path} must contain a 'metrics' list")

        raw_defaults = payload.get("data_sources") or []
        if not isinstance(raw_defaults, list):
            raise TypeError(
                f"Use-case file {path} 'data_sources' must be a list when present"
            )
        defaults = [
            DataSourceConfig.model_validate(item)
            for item in raw_defaults
            if isinstance(item, dict)
        ]

        metrics: list[Self] = []
        for entry in raw_metrics:
            if not isinstance(entry, dict):
                raise TypeError(f"Each metric in {path} must be an object")
            data: dict[str, Any] = dict(entry)
            raw_metric_sources = data.get("data_sources") or []
            if not isinstance(raw_metric_sources, list):
                raise TypeError(
                    f"Metric data_sources in {path} must be a list when present"
                )
            metric_sources = [
                DataSourceConfig.model_validate(item)
                for item in raw_metric_sources
                if isinstance(item, dict)
            ]
            data["data_sources"] = resolve_data_sources(defaults, metric_sources)
            metrics.append(cls.model_validate(data))
        return metrics
