from abc import ABC, ABCMeta, abstractmethod
from typing import Any, cast

from pydantic import BaseModel
from pydantic._internal._model_construction import ModelMetaclass
from pydantic_core import PydanticUndefined

_DATA_SOURCE_REGISTRY: dict[str, type["DataSource"]] = {}


class DataSourceMeta(ModelMetaclass, ABCMeta):
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

        source_field = cls.model_fields.get("source")
        if source_field is None or source_field.default is PydanticUndefined:
            return cls

        source = source_field.default
        if not isinstance(source, str):
            raise TypeError(f"Source must be a string, got {type(source)} for {name}")

        _DATA_SOURCE_REGISTRY[source] = cls
        return cls


class DataResult(BaseModel):
    source: str
    document: str | None = None
    url: str | None = None
    citation: str
    metadata: dict[str, Any]


class DataSource(ABC, BaseModel, metaclass=DataSourceMeta):
    source: str
    unit: str | None = None

    @abstractmethod
    async def get_data(
        self,
        country_iso3: str,
        *,
        year_start: int | None = None,
        year_end: int | None = None,
    ) -> list[DataResult]: ...


def build_data_source(source: str, **kwargs: Any) -> DataSource:
    cls = _DATA_SOURCE_REGISTRY.get(source)
    if cls is None:
        raise ValueError(f"Unknown data source: {source}")
    return cls(**kwargs)
