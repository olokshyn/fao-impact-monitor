from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from fao_impact_monitor.data_source.data_source_config import DataSourceConfig
from fao_impact_monitor.utils.meta_magic import RegistryMeta

if TYPE_CHECKING:
    from fao_impact_monitor.metric.metric import Metric

_DATA_SOURCE_CLS_REGISTRY: dict[str, type[DataSource]] = {}
_DATA_SOURCE_INSTANCE_REGISTRY: dict[str, DataSource] = {}


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


def _ensure_builtin_sources_registered() -> None:
    """Import concrete sources so RegistryMeta has registered them."""
    # Local imports avoid a circular import with Metric → DataSourceConfig.
    from fao_impact_monitor.data_source import fao_repository as _fao_repository
    from fao_impact_monitor.data_source import tellus as _tellus
    from fao_impact_monitor.data_source import world_bank as _world_bank

    del _fao_repository, _tellus, _world_bank


def get_data_source(source: str) -> DataSource:
    if source not in _DATA_SOURCE_CLS_REGISTRY:
        _ensure_builtin_sources_registered()
    if source in _DATA_SOURCE_INSTANCE_REGISTRY:
        return _DATA_SOURCE_INSTANCE_REGISTRY[source]
    cls = _DATA_SOURCE_CLS_REGISTRY.get(source)
    if cls is None:
        raise ValueError(f"Unknown data source: {source}")
    instance = cls()
    _DATA_SOURCE_INSTANCE_REGISTRY[source] = instance
    return instance
