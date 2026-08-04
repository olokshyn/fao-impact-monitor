from typing import TYPE_CHECKING, Any

from .data_source import DataResult, DataSource, get_data_source
from .data_source_config import DataSourceConfig

if TYPE_CHECKING:
    from .fao_repository import (
        FaoRepository,
        FaoRepositoryDataResult,
        FaoRepositoryDataSourceConfig,
    )
    from .tellus import TellusDataSource
    from .world_bank import WorldBank, WorldBankDataResult, WorldBankDataSourceConfig

__all__ = [
    "DataResult",
    "DataSource",
    "DataSourceConfig",
    "FaoRepository",
    "FaoRepositoryDataResult",
    "FaoRepositoryDataSourceConfig",
    "TellusDataSource",
    "WorldBank",
    "WorldBankDataResult",
    "WorldBankDataSourceConfig",
    "get_data_source",
]

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "FaoRepository": (".fao_repository", "FaoRepository"),
    "FaoRepositoryDataResult": (".fao_repository", "FaoRepositoryDataResult"),
    "FaoRepositoryDataSourceConfig": (
        ".fao_repository",
        "FaoRepositoryDataSourceConfig",
    ),
    "TellusDataSource": (".tellus", "TellusDataSource"),
    "WorldBank": (".world_bank", "WorldBank"),
    "WorldBankDataResult": (".world_bank", "WorldBankDataResult"),
    "WorldBankDataSourceConfig": (".world_bank", "WorldBankDataSourceConfig"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    from importlib import import_module

    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
