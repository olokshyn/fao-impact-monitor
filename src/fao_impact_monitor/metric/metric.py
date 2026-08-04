from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, Field

from fao_impact_monitor.data_source.data_source_config import DataSourceConfig


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

        metrics: list[Self] = []
        for entry in raw_metrics:
            if not isinstance(entry, dict):
                raise TypeError(f"Each metric in {path} must be an object")
            data: dict[str, Any] = dict(entry)
            metrics.append(cls.model_validate(data))
        return metrics
