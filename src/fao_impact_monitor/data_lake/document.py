from enum import StrEnum, auto
from typing import Annotated

from beanie import Document as BeanieDocument
from beanie import Indexed, PydanticObjectId
from pydantic import BaseModel, Field, computed_field

from fao_impact_monitor.data_lake.stage import StageResult
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
    title: str
    url: Annotated[str, Indexed(unique=True)]
    external_id: Annotated[
        str | None,
        Indexed(
            unique=True,
            partialFilterExpression={"external_id": {"$type": "string"}},
        ),
    ] = None
    relations: list[Relation] = Field(default_factory=list)
    pipeline_name: str
    pipeline_completed: bool = False
    stage_results: dict[str, list[StageResult]] = Field(default_factory=dict)

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
