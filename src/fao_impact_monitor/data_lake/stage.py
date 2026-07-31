from abc import ABC, abstractmethod
from datetime import UTC, datetime
from enum import StrEnum, auto
from typing import TYPE_CHECKING, Annotated, Any

from beanie import Document as BeanieDocument
from beanie import Indexed
from pydantic import BaseModel, Field

from fao_impact_monitor.utils.meta_magic import (
    RegistryMeta,
    RegistryModelMeta,
    get_class_id_value,
)

if TYPE_CHECKING:
    from fao_impact_monitor.data_lake.document import Document


class StageStatus(StrEnum):
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()


# _STAGE_VERSION_REGISTRY: dict[str, type["StageVersion"]] = {}


# class StageVersionMeta(RegistryModelMeta):
#     registry = _STAGE_VERSION_REGISTRY
#     use_class_id_value = True


class StageVersion(BeanieDocument):  # , metaclass=StageVersionMeta):
    """Immutable stage version.

    Records important parameters of a stage for provenance.
    Derive from this class to store custom stage parameters.

    ``version_id`` is a caller-computed hash of stage parameters / prompt (not
    Mongo's ``_id``). Concrete subclasses set ``Settings.class_id_value``; the
    ``name`` property exposes it on instances.
    """

    version_id: Annotated[str, Indexed(unique=True)]
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def name(self) -> str:
        return str(get_class_id_value(type(self)))

    class Settings:
        name = "StageVersion"
        is_root = True
        class_id = "name"


_STAGE_RESULT_REGISTRY: dict[str, type["StageResult"]] = {}


class StageResultMeta(RegistryModelMeta):
    registry = _STAGE_RESULT_REGISTRY
    attr = "name"


class StageResult(ABC, BaseModel, metaclass=StageResultMeta):
    """Result produced by a single stage run.

    Records the version of the stage that produced it.

    Derive from this class to add custom fields that store actual results.
    """

    name: str
    version_id: str  # StageVersion.version_id
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    @abstractmethod
    def status(self) -> StageStatus: ...


_STAGE_REGISTRY: dict[str, type["Stage"]] = {}


class StageMeta(RegistryMeta):
    registry = _STAGE_REGISTRY
    attr = "name"


class Stage(ABC, metaclass=StageMeta):
    name: str

    @abstractmethod
    async def get_version(self) -> StageVersion: ...

    @abstractmethod
    async def run(
        self,
        document: "Document",
        stage_params: dict[str, Any],
        prev_stages: list[StageResult],
    ) -> StageResult: ...


def get_stage(name: str) -> Stage:
    return _STAGE_REGISTRY[name]()
