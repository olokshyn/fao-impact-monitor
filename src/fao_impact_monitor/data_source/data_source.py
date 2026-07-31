from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from fao_impact_monitor.data_source.data_source_config import DataSourceConfig
from fao_impact_monitor.metric.metric import Metric
from fao_impact_monitor.utils.meta_magic import RegistryMeta

_DATA_SOURCE_CLS_REGISTRY: dict[str, type["DataSource"]] = {}
_DATA_SOURCE_INSTANCE_REGISTRY: dict[str, "DataSource"] = {}


class DataResult(BaseModel):
    source: str
    title: str | None = None
    url: str | None = None
    citation: str
    metadata: dict[str, Any]


class DataSourceMeta(RegistryMeta):
    registry = _DATA_SOURCE_CLS_REGISTRY
    attr = "source"


class DataSource(ABC, metaclass=DataSourceMeta):
    source: str

    @abstractmethod
    async def get_data(
        self,
        metric: Metric,
        data_source_config: DataSourceConfig,
        country_iso3: str,
        *,
        year_start: int | None = None,
        year_end: int | None = None,
    ) -> list[DataResult]: ...


def get_data_source(source: str) -> DataSource:
    if source in _DATA_SOURCE_INSTANCE_REGISTRY:
        return _DATA_SOURCE_INSTANCE_REGISTRY[source]
    cls = _DATA_SOURCE_CLS_REGISTRY.get(source)
    if cls is None:
        raise ValueError(f"Unknown data source: {source}")
    instance = cls()
    _DATA_SOURCE_INSTANCE_REGISTRY[source] = instance
    return instance
