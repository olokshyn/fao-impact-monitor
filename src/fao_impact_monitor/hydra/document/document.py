from __future__ import annotations

from enum import StrEnum, auto
from typing import Annotated, Any, ClassVar

from beanie import Document as BeanieDocument
from beanie import Indexed, PydanticObjectId
from pydantic import BaseModel, Field, field_validator
from pymongo import ASCENDING, IndexModel

from fao_impact_monitor.hydra.stage.stage import StageResult, get_stage_result_class
from fao_impact_monitor.utils.meta_magic import get_class_id_value


class DocumentType(StrEnum):
    WEB_PAGE = auto()
    PDF = auto()
    TELLUS = auto()


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
    """Durable product of the pipeline (collection ``documents``)."""

    url: Annotated[str, Indexed()]
    source: Annotated[str | None, Indexed()] = None
    external_id: Annotated[
        str | None,
        Indexed(
            unique=True,
            partialFilterExpression={"external_id": {"$type": "string"}},
        ),
    ] = None
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    relations: list[Relation] = Field(default_factory=list)
    # workflow_name → node_name → result history
    stage_results: dict[str, dict[str, list[StageResult]]] = Field(default_factory=dict)

    @field_validator("stage_results", mode="before")
    @classmethod
    def _hydrate_stage_results(
        cls, value: dict[str, dict[str, list[Any]]] | None
    ) -> dict[str, dict[str, list[Any]]] | None:
        """Rehydrate concrete StageResult subclasses from the registry by name."""
        if not value:
            return value
        hydrated: dict[str, dict[str, list[Any]]] = {}
        for workflow_name, nodes in value.items():
            if not isinstance(nodes, dict):
                continue
            node_map: dict[str, list[Any]] = {}
            for node_name, results in nodes.items():
                items: list[Any] = []
                for item in results:
                    if isinstance(item, StageResult) and type(item) is not StageResult:
                        items.append(item)
                        continue
                    data = item.model_dump() if isinstance(item, StageResult) else item
                    if not isinstance(data, dict):
                        items.append(item)
                        continue
                    name = data.get("name", node_name)
                    try:
                        model_cls = get_stage_result_class(str(name))
                    except KeyError:
                        model_cls = StageResult
                    items.append(model_cls.model_validate(data))
                node_map[str(node_name)] = items
            hydrated[str(workflow_name)] = node_map
        return hydrated

    @property
    def type(self) -> DocumentType:
        return DocumentType(get_class_id_value(type(self)))

    class Settings:
        name = "documents"
        is_root = True
        class_id = "type"
        indexes: ClassVar[list[IndexModel]] = [
            IndexModel(
                [("url", ASCENDING), ("source", ASCENDING)],
                unique=True,
            ),
        ]

    async def push_stage_result(
        self,
        workflow_name: str,
        workflow_node_name: str,
        result: StageResult,
    ) -> None:
        """Atomically append ``result`` under ``stage_results[wf][node]``."""
        path = f"stage_results.{workflow_name}.{workflow_node_name}"
        await self.update({"$push": {path: result.model_dump(mode="python")}})
        bucket = self.stage_results.setdefault(workflow_name, {}).setdefault(
            workflow_node_name, []
        )
        if result not in bucket:
            bucket.append(result)

    async def atomic_set(self, **fields: Any) -> None:
        """Atomically ``$set`` selected top-level or dotted paths.

        Keys may be dotted Mongo paths (e.g. ``metadata.title``) or model
        field names.
        """
        await self.update({"$set": fields})
        for key, value in fields.items():
            if "." in key:
                continue
            setattr(self, key, value)

    def latest_stage_result(
        self,
        workflow_name: str,
        workflow_node_name: str,
    ) -> StageResult | None:
        """Return the most recent StageResult for ``[wf][node]``, if any."""
        results = (self.stage_results.get(workflow_name) or {}).get(
            workflow_node_name
        ) or []
        return results[-1] if results else None
