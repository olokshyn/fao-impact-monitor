"""Rebuildable semantic, lexical, and entity retrieval index."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Protocol

from pydantic import BaseModel, Field

from fao_impact_monitor.athena.model.filters import (
    FilterExpr,
    FilterSpec,
    compile_mongo,
    mongo_matches,
)
from fao_impact_monitor.athena.model.status import UnknownPolicy
from fao_impact_monitor.athena.ns import TAG_FIELDS
from fao_impact_monitor.config import VectorStoreConfig


class RetrievalRecord(BaseModel):
    jena_id: str
    record_type: str
    text: str
    embedding: list[float]
    entities: list[str] = Field(default_factory=list)
    tags: dict[str, list[str]] = Field(default_factory=dict)
    unknown_tags: list[str] = Field(default_factory=list)
    version: str
    verification_status: str
    scientific_status: str
    graph: str


class RetrievalHit(BaseModel):
    record: RetrievalRecord
    score: float
    channel: str


class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class RetrievalIndex(Protocol):
    async def upsert(self, record: RetrievalRecord) -> None: ...

    async def delete(self, jena_id: str) -> None: ...

    async def get(self, jena_id: str) -> RetrievalRecord | None: ...

    async def search_semantic(
        self,
        vector: list[float],
        spec: FilterSpec,
        limit: int,
    ) -> list[RetrievalHit]: ...

    async def search_lexical(
        self,
        query: str,
        spec: FilterSpec,
        limit: int,
    ) -> list[RetrievalHit]: ...

    async def search_entities(
        self,
        entities: list[str],
        spec: FilterSpec,
        limit: int,
    ) -> list[RetrievalHit]: ...


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _passes(record: RetrievalRecord, spec: FilterSpec) -> bool:
    document: dict[str, object] = {
        field: values for field, values in record.tags.items()
    }
    document["unknown_tags"] = record.unknown_tags
    document["verification_status"] = record.verification_status
    document["record_type"] = record.record_type
    return mongo_matches(document, compile_mongo(spec))


class MemoryIndex:
    """In-memory index that filters before it ranks."""

    def __init__(self) -> None:
        self.records: dict[str, RetrievalRecord] = {}

    async def upsert(self, record: RetrievalRecord) -> None:
        self.records[record.jena_id] = record

    async def delete(self, jena_id: str) -> None:
        self.records.pop(jena_id, None)

    async def get(self, jena_id: str) -> RetrievalRecord | None:
        return self.records.get(jena_id)

    async def search_semantic(
        self,
        vector: list[float],
        spec: FilterSpec,
        limit: int,
    ) -> list[RetrievalHit]:
        ranked = [
            RetrievalHit(
                record=record,
                score=_cosine(vector, record.embedding),
                channel="semantic",
            )
            for record in self.records.values()
            if _passes(record, spec)
        ]
        ranked.sort(key=lambda hit: hit.score, reverse=True)
        return [hit for hit in ranked if hit.score > 0][:limit]

    async def search_lexical(
        self,
        query: str,
        spec: FilterSpec,
        limit: int,
    ) -> list[RetrievalHit]:
        wanted = _tokens(query)
        ranked: list[RetrievalHit] = []
        for record in self.records.values():
            if not _passes(record, spec):
                continue
            overlap = wanted & _tokens(record.text)
            if not overlap:
                continue
            score = len(overlap) / max(1, len(wanted))
            ranked.append(RetrievalHit(record=record, score=score, channel="lexical"))
        ranked.sort(key=lambda hit: hit.score, reverse=True)
        return ranked[:limit]

    async def search_entities(
        self,
        entities: list[str],
        spec: FilterSpec,
        limit: int,
    ) -> list[RetrievalHit]:
        wanted = {entity.lower() for entity in entities}
        ranked: list[RetrievalHit] = []
        for record in self.records.values():
            if not _passes(record, spec):
                continue
            have = {entity.lower() for entity in record.entities}
            overlap = wanted & have
            if not overlap:
                continue
            ranked.append(
                RetrievalHit(
                    record=record,
                    score=len(overlap) / max(1, len(wanted)),
                    channel="entity",
                )
            )
        ranked.sort(key=lambda hit: hit.score, reverse=True)
        return ranked[:limit]


class BedrockEmbedder:
    """Titan embeddings through the same Bedrock client as the data lake."""

    def __init__(self) -> None:
        self._inner: object | None = None

    async def embed(self, texts: list[str]) -> list[list[float]]:
        import asyncio

        from fao_impact_monitor.data_lake.vectorstore import build_embeddings

        if self._inner is None:
            self._inner = build_embeddings()
        embedded: list[list[float]] = await asyncio.to_thread(
            self._inner.embed_documents,  # type: ignore[attr-defined]
            list(texts),
        )
        return embedded


class HashEmbedder:
    """Deterministic embedder for tests and offline runs."""

    def __init__(self, dimensions: int = 8) -> None:
        self.dimensions = dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(text) for text in texts]

    def _one(self, text: str) -> list[float]:
        raw = hashlib.sha256(text.encode("utf-8")).digest()
        values: list[float] = []
        while len(values) < self.dimensions:
            raw = hashlib.sha256(raw).digest()
            values.extend((byte / 127.5) - 1 for byte in raw)
        vector = values[: self.dimensions]
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


def vector_index_definition(
    config: VectorStoreConfig,
    *,
    name: str,
) -> dict[str, Any]:
    fields: list[dict[str, Any]] = [
        {
            "type": "vector",
            "path": "embedding",
            "numDimensions": config.embedding_dimensions or 1024,
            "similarity": "cosine",
        }
    ]
    for field in (*TAG_FIELDS, "unknown_tags", "verification_status", "record_type"):
        fields.append({"type": "filter", "path": field})
    return {"name": name, "type": "vectorSearch", "definition": {"fields": fields}}


def text_index_definition(*, name: str) -> dict[str, Any]:
    token_fields = {
        field: {"type": "token"}
        for field in (*TAG_FIELDS, "unknown_tags", "verification_status", "record_type")
    }
    return {
        "name": name,
        "type": "search",
        "definition": {
            "mappings": {
                "dynamic": False,
                "fields": {
                    "text": {"type": "string", "analyzer": "lucene.standard"},
                    "entities": {"type": "token"},
                    **token_fields,
                },
            }
        },
    }


def mongo_document(record: RetrievalRecord) -> dict[str, Any]:
    document = record.model_dump()
    for field, values in record.tags.items():
        document[field] = values
    return document


def build_vector_pipeline(
    vector: list[float],
    spec: FilterSpec,
    *,
    index_name: str,
    limit: int,
    num_candidates: int,
) -> list[dict[str, Any]]:
    search: dict[str, Any] = {
        "index": index_name,
        "path": "embedding",
        "queryVector": vector,
        "numCandidates": num_candidates,
        "limit": limit,
    }
    prefilter = compile_mongo(spec)
    if prefilter:
        search["filter"] = prefilter
    return [{"$vectorSearch": search}]


def compile_atlas(spec: FilterSpec) -> dict[str, Any] | None:
    """Atlas Search compound filter applied before lexical scoring."""
    if spec.root is None:
        return None
    return _atlas_expr(spec.root)


def _atlas_expr(expr: FilterExpr) -> dict[str, Any]:
    if expr.op == "and":
        return {"compound": {"must": [_atlas_expr(child) for child in expr.children]}}
    if expr.op == "or":
        return {
            "compound": {
                "should": [_atlas_expr(child) for child in expr.children],
                "minimumShouldMatch": 1,
            }
        }
    if expr.op == "not":
        return {"compound": {"mustNot": [_atlas_expr(expr.children[0])]}}
    return _atlas_tag(expr)


def _atlas_tag(expr: FilterExpr) -> dict[str, Any]:
    field = expr.field or ""
    unknown = {"equals": {"path": "unknown_tags", "value": field}}
    if expr.comparison == "include" and expr.values:
        known: dict[str, Any] = {"in": {"path": field, "value": list(expr.values)}}
    elif expr.comparison == "include":
        known = {"compound": {"mustNot": [{"exists": {"path": field}}]}}
    else:
        known = {
            "compound": {
                "must": [{"exists": {"path": field}}],
                "mustNot": [{"in": {"path": field, "value": list(expr.values)}}],
            }
        }
    if expr.unknown == UnknownPolicy.ONLY:
        return unknown
    if expr.unknown == UnknownPolicy.INCLUDE:
        return {"compound": {"should": [known, unknown], "minimumShouldMatch": 1}}
    return {"compound": {"must": [known], "mustNot": [unknown]}}


def build_text_pipeline(
    query: str,
    spec: FilterSpec,
    *,
    index_name: str,
    limit: int,
) -> list[dict[str, Any]]:
    compound: dict[str, Any] = {"must": [{"text": {"query": query, "path": "text"}}]}
    prefilter = compile_atlas(spec)
    if prefilter is not None:
        compound["filter"] = [prefilter]
    return [{"$search": {"index": index_name, "compound": compound}}, {"$limit": limit}]


class MongoRetrievalIndex:
    """Atlas Vector Search / Search index. Records are rebuildable from Jena."""

    def __init__(
        self,
        collection: Any,
        *,
        vector_index: str,
        text_index: str,
        num_candidates: int = 100,
    ) -> None:
        self._collection = collection
        self._vector_index = vector_index
        self._text_index = text_index
        self._num_candidates = num_candidates

    async def upsert(self, record: RetrievalRecord) -> None:
        await self._collection.update_one(
            {"jena_id": record.jena_id},
            {"$set": mongo_document(record)},
            upsert=True,
        )

    async def delete(self, jena_id: str) -> None:
        await self._collection.delete_one({"jena_id": jena_id})

    async def get(self, jena_id: str) -> RetrievalRecord | None:
        document = await self._collection.find_one({"jena_id": jena_id})
        if document is None:
            return None
        return RetrievalRecord.model_validate(document)

    async def search_semantic(
        self,
        vector: list[float],
        spec: FilterSpec,
        limit: int,
    ) -> list[RetrievalHit]:
        pipeline = build_vector_pipeline(
            vector,
            spec,
            index_name=self._vector_index,
            limit=limit,
            num_candidates=max(self._num_candidates, limit),
        )
        return await self._aggregate(pipeline, spec, "semantic")

    async def search_lexical(
        self,
        query: str,
        spec: FilterSpec,
        limit: int,
    ) -> list[RetrievalHit]:
        pipeline = build_text_pipeline(
            query,
            spec,
            index_name=self._text_index,
            limit=limit,
        )
        return await self._aggregate(pipeline, spec, "lexical")

    async def search_entities(
        self,
        entities: list[str],
        spec: FilterSpec,
        limit: int,
    ) -> list[RetrievalHit]:
        query: dict[str, Any] = {"entities": {"$in": entities}}
        prefilter = compile_mongo(spec)
        if prefilter:
            query = {"$and": [query, prefilter]}
        cursor = self._collection.find(query).limit(limit)
        hits: list[RetrievalHit] = []
        async for document in cursor:
            record = RetrievalRecord.model_validate(document)
            hits.append(RetrievalHit(record=record, score=1.0, channel="entity"))
        return hits

    async def _aggregate(
        self,
        pipeline: list[dict[str, Any]],
        spec: FilterSpec,
        channel: str,
    ) -> list[RetrievalHit]:
        hits: list[RetrievalHit] = []
        cursor = await self._collection.aggregate(pipeline)
        async for document in cursor:
            record = RetrievalRecord.model_validate(document)
            flat = {field: values for field, values in record.tags.items()}
            flat["unknown_tags"] = record.unknown_tags
            if not mongo_matches(flat, compile_mongo(spec)):
                continue
            score = float(document.get("score") or 0.0)
            hits.append(RetrievalHit(record=record, score=score, channel=channel))
        return hits
