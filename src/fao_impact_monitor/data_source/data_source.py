from abc import ABC, ABCMeta, abstractmethod
from typing import Any, cast

from pydantic import BaseModel

from fao_impact_monitor.data_source.data_source_config import DataSourceConfig
from fao_impact_monitor.metric.metric import Metric

_DATA_SOURCE_CLS_REGISTRY: dict[str, type["DataSource"]] = {}
_DATA_SOURCE_INSTANCE_REGISTRY: dict[str, "DataSource"] = {}


class DataSourceMeta(ABCMeta):
    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> type["DataSource"]:
        cls = cast(
            "type[DataSource]",
            super().__new__(mcs, name, bases, namespace, **kwargs),
        )
        if cls.__dict__.get("__abstractmethods__"):
            return cls

        source = namespace.get("source")
        if not isinstance(source, str):
            raise TypeError(f"Source must be a string, got {type(source)} for {name}")

        _DATA_SOURCE_CLS_REGISTRY[source] = cls
        return cls


class DataResult(BaseModel):
    source: str
    title: str | None = None
    url: str | None = None
    citation: str
    metadata: dict[str, Any]


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
