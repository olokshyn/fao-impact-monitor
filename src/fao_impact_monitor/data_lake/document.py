from enum import StrEnum, auto
from typing import Annotated, Any

from beanie import Document as BeanieDocument
from beanie import Indexed, PydanticObjectId
from pydantic import BaseModel, Field, computed_field, field_validator

from fao_impact_monitor.data_lake.common import Status
from fao_impact_monitor.data_lake.stage import StageResult, get_stage_result_class
from fao_impact_monitor.utils.meta_magic import get_class_id_value


class DocumentType(StrEnum):
    WEB_PAGE = auto()
    PDF = auto()


class RelationType(StrEnum):
    URL_LINK = auto()
    CITATION = auto()


class RelationSide(StrEnum):
    FROM = auto()
    TO = auto()


class Relation(BaseModel):
    type: RelationType
    side: RelationSide
    d_id: PydanticObjectId
    d_type: DocumentType


class Document(BeanieDocument):
    url: Annotated[str, Indexed(unique=True)]
    external_id: Annotated[
        str | None,
        Indexed(
            unique=True,
            partialFilterExpression={"external_id": {"$type": "string"}},
        ),
    ] = None
    title: str | None = None
    relations: list[Relation] = Field(default_factory=list)
    pipeline_statuses: dict[str, Status] = Field(default_factory=dict)
    stage_results: dict[str, list[StageResult]] = Field(default_factory=dict)

    @field_validator("stage_results", mode="before")
    @classmethod
    def _hydrate_stage_results(
        cls, value: dict[str, list[Any]] | None
    ) -> dict[str, list[Any]] | None:
        """Rehydrate concrete StageResult subclasses from the registry by name."""
        if not value:
            return value
        hydrated: dict[str, list[Any]] = {}
        for key, results in value.items():
            items: list[Any] = []
            for item in results:
                if isinstance(item, StageResult) and type(item) is not StageResult:
                    items.append(item)
                    continue
                data = item.model_dump() if isinstance(item, StageResult) else item
                if not isinstance(data, dict):
                    items.append(item)
                    continue
                name = data.get("name", key)
                try:
                    model_cls = get_stage_result_class(str(name))
                except KeyError:
                    model_cls = StageResult
                items.append(model_cls.model_validate(data))
            hydrated[key] = items
        return hydrated

    def pipeline_status(self, pipeline_name: str) -> Status | None:
        return self.pipeline_statuses.get(pipeline_name)

    def is_pipeline_completed(self, pipeline_name: str) -> bool:
        return self.pipeline_statuses.get(pipeline_name) == Status.COMPLETED

    def set_pipeline_status(self, pipeline_name: str, status: Status) -> None:
        self.pipeline_statuses[pipeline_name] = status

    @property
    def type(self) -> DocumentType:
        return DocumentType(get_class_id_value(type(self)))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def citation(self) -> str:
        raise NotImplementedError

    class Settings:
        name = "Document"
        is_root = True
        class_id = "type"
