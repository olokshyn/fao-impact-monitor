"""Hybrid retrieval over pipeline-owned representations with evidence expansion."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pymongo.errors import OperationFailure
from pymongo.operations import SearchIndexModel

from fao_impact_monitor.config import PdfPipelineConfig, VectorStoreConfig, get_config
from fao_impact_monitor.data_lake.document import DocumentType
from fao_impact_monitor.data_lake.vectorstore import ChunkHit, build_embeddings
from fao_impact_monitor.pdf_pipeline.models import (
    EvidenceUnit,
    PdfDocumentRecord,
    PdfEmbeddingRecord,
)
from fao_impact_monitor.utils.document_uri import file_document_uri

AggregateFn = Callable[[list[dict[str, Any]]], Awaitable[list[dict[str, Any]]]]
EmbedQueryFn = Callable[[str], Awaitable[list[float]]]

_INDEX_NOT_FOUND = 27
_INDEX_ALREADY_EXISTS = 68
_BUILDING_STATUSES = frozenset({"PENDING", "BUILDING", "IN_PROGRESS", "DOES_NOT_EXIST"})


def vector_index_definition(config: PdfPipelineConfig | None = None) -> dict[str, Any]:
    cfg = config or get_config().pdf_pipeline
    dimensions = get_config().vector_store.embedding_dimensions or 1024
    return {
        "name": cfg.vector_index_name,
        "type": "vectorSearch",
        "definition": {
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": dimensions,
                    "similarity": "cosine",
                },
                {"type": "filter", "path": "pipeline"},
                {"type": "filter", "path": "searchable"},
                {"type": "filter", "path": "countries_iso3"},
                {"type": "filter", "path": "section_id"},
            ]
        },
    }


def text_index_definition(config: PdfPipelineConfig | None = None) -> dict[str, Any]:
    cfg = config or get_config().pdf_pipeline
    return {
        "name": cfg.text_index_name,
        "type": "search",
        "definition": {
            "mappings": {
                "dynamic": False,
                "fields": {
                    "embedding_text": {"type": "string", "analyzer": "lucene.standard"},
                    "countries_iso3": {"type": "token"},
                    "pipeline": {"type": "token"},
                    "searchable": {"type": "boolean"},
                    "section_id": {"type": "token"},
                },
            }
        },
    }


def _filter(countries: Sequence[str] | None) -> dict[str, Any]:
    result: dict[str, Any] = {"pipeline": "pdf_pipeline", "searchable": True}
    if countries:
        result["countries_iso3"] = {"$in": list(countries)}
    return result


def _search_definition(existing: Mapping[str, Any] | None) -> Any:
    if existing is None:
        return None
    return existing.get("latestDefinition", existing.get("definition"))


def _definitions_compatible(actual: Any, desired: Any) -> bool:
    """True when ``desired`` is structurally contained in ``actual``.

    Atlas Search may add default fields to a stored definition, so exact
    equality would recreate indexes on every call.
    """
    if actual == desired:
        return True
    if isinstance(desired, dict) and isinstance(actual, dict):
        return all(
            key in actual and _definitions_compatible(actual[key], value)
            for key, value in desired.items()
        )
    if isinstance(desired, list) and isinstance(actual, list):
        if len(desired) != len(actual):
            return False
        return all(
            _definitions_compatible(item, want)
            for item, want in zip(actual, desired, strict=True)
        )
    return False


def _is_missing_search_index(exc: OperationFailure) -> bool:
    if exc.code == _INDEX_NOT_FOUND:
        return True
    return "not found" in str(exc).lower()


def _is_search_index_already_exists(exc: OperationFailure) -> bool:
    if exc.code == _INDEX_ALREADY_EXISTS:
        return True
    message = str(exc).lower()
    return "already exists" in message or "duplicate index" in message


def _is_search_index_management_unavailable(exc: OperationFailure) -> bool:
    return "search index management" in str(exc).lower()


async def _list_search_indexes_by_name(
    collection: Any,
    *,
    attempts: int = 30,
    delay_seconds: float = 2.0,
) -> dict[Any, Any]:
    last_error: OperationFailure | None = None
    for _ in range(attempts):
        try:
            return {
                row.get("name"): row
                async for row in await collection.list_search_indexes()
            }
        except OperationFailure as exc:
            if not _is_search_index_management_unavailable(exc):
                raise
            last_error = exc
            await asyncio.sleep(delay_seconds)
    assert last_error is not None
    raise last_error


async def ensure_pdf_pipeline_indexes(
    config: PdfPipelineConfig | None = None,
    *,
    recreate: bool = False,
) -> None:
    """Create isolated Atlas indexes without altering data-lake index definitions.

    Safe under concurrent research workers: a missing index on drop and an
    already-existing index on create are ignored, and in-progress indexes are
    left for the process that started them. ``recreate=True`` drops and
    rebuilds even when a compatible index already exists.
    """
    collection = PdfEmbeddingRecord.get_pymongo_collection()
    await collection.create_index([("pipeline", 1), ("owner_id", 1)])
    await collection.create_index([("pipeline", 1), ("section_id", 1)])
    existing = await _list_search_indexes_by_name(collection)
    for spec in (vector_index_definition(config), text_index_definition(config)):
        name = spec["name"]
        prior = existing.get(name)
        desired = spec["definition"]
        actual = _search_definition(prior)
        status = str((prior or {}).get("status") or "").upper()
        if (
            not recreate
            and prior is not None
            and _definitions_compatible(actual, desired)
        ):
            continue
        if not recreate and prior is not None and status in _BUILDING_STATUSES:
            continue
        if prior is not None or recreate:
            try:
                await collection.drop_search_index(name)
            except OperationFailure as exc:
                if not _is_missing_search_index(exc):
                    raise
        try:
            await collection.create_search_index(
                SearchIndexModel(name=name, type=spec["type"], definition=desired)
            )
        except OperationFailure as exc:
            if not _is_search_index_already_exists(exc):
                raise


def hybrid_pipeline(
    query: str,
    vector: Sequence[float],
    *,
    countries_iso3: Sequence[str] | None,
    limit: int,
    config: PdfPipelineConfig | None = None,
    fused_limit: int | None = None,
) -> list[dict[str, Any]]:
    cfg = config or get_config().pdf_pipeline
    text_filters: list[dict[str, Any]] = [
        {"text": {"query": query, "path": "embedding_text"}},
        {"equals": {"path": "pipeline", "value": "pdf_pipeline"}},
        {"equals": {"path": "searchable", "value": True}},
    ]
    if countries_iso3:
        text_filters.append(
            {"in": {"path": "countries_iso3", "value": list(countries_iso3)}}
        )
    return [
        {
            "$rankFusion": {
                "input": {
                    "pipelines": {
                        "vector": [
                            {
                                "$vectorSearch": {
                                    "index": cfg.vector_index_name,
                                    "path": "embedding",
                                    "queryVector": list(vector),
                                    "numCandidates": max(100, limit * 10),
                                    "limit": limit * 3,
                                    "filter": _filter(countries_iso3),
                                }
                            }
                        ],
                        "text": [
                            {
                                "$search": {
                                    "index": cfg.text_index_name,
                                    "compound": {"must": text_filters},
                                }
                            },
                            {"$limit": limit * 3},
                        ],
                    }
                }
            }
        },
        {"$limit": fused_limit or limit * 3},
        {"$addFields": {"score": {"$meta": "score"}}},
    ]


class PdfEvidenceVectorStore:
    """A VectorStore-compatible view that expands provenance bundles before return."""

    def __init__(
        self,
        *,
        config: PdfPipelineConfig | None = None,
        vector_config: VectorStoreConfig | None = None,
        embed_query_fn: EmbedQueryFn | None = None,
        aggregate_fn: AggregateFn | None = None,
    ) -> None:
        self.config = config or get_config().pdf_pipeline
        self.vector_config = vector_config or get_config().vector_store
        self._embed_query_fn = embed_query_fn
        self._aggregate_fn = aggregate_fn

    async def _embed(self, query: str) -> list[float]:
        if self._embed_query_fn is not None:
            return await self._embed_query_fn(query)
        return await build_embeddings(
            vector_store_config=self.vector_config
        ).aembed_query(query)

    async def _aggregate(self, pipeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self._aggregate_fn is not None:
            return await self._aggregate_fn(pipeline)
        cursor = await PdfEmbeddingRecord.get_pymongo_collection().aggregate(pipeline)
        return [row async for row in cursor]

    async def search(
        self,
        query: str,
        *,
        countries_iso3: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> list[ChunkHit]:
        if not query.strip():
            return []
        result_limit = limit or self.vector_config.limit
        rows = await self._aggregate(
            hybrid_pipeline(
                query,
                await self._embed(query),
                countries_iso3=countries_iso3,
                limit=result_limit,
                config=self.config,
            )
        )
        seen: set[str] = set()
        hits: list[ChunkHit] = []
        for row in rows:
            owner_id = row.get("owner_id")
            if not isinstance(owner_id, str) or owner_id in seen:
                continue
            if row.get("owner_kind") == "section":
                for section_unit in await self._section_evidence(
                    owner_id, countries_iso3, result_limit
                ):
                    if section_unit.evidence_id not in seen:
                        hits.append(await self._hit(section_unit, row.get("score")))
                        seen.add(section_unit.evidence_id)
                continue
            evidence_id = row.get("evidence_id")
            if not isinstance(evidence_id, str):
                continue
            unit = await EvidenceUnit.find_one(EvidenceUnit.evidence_id == evidence_id)
            if unit is not None and unit.searchable:
                hits.append(await self._hit(unit, row.get("score")))
                seen.add(evidence_id)
            if len(hits) >= result_limit:
                break
        return hits[:result_limit]

    async def search_embeddings(
        self,
        query: str,
        *,
        countries_iso3: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return raw ranked embedding representations without owner deduplication."""
        if not query.strip():
            return []
        result_limit = limit or self.vector_config.limit
        return await self._aggregate(
            hybrid_pipeline(
                query,
                await self._embed(query),
                countries_iso3=countries_iso3,
                limit=result_limit,
                config=self.config,
                fused_limit=result_limit,
            )
        )

    async def _section_evidence(
        self, section_id: str, countries: Sequence[str] | None, limit: int
    ) -> list[EvidenceUnit]:
        query: dict[str, Any] = {"section_id": section_id, "searchable": True}
        if countries:
            query["countries.iso3"] = {"$in": list(countries)}
        cursor = (
            EvidenceUnit.get_pymongo_collection()
            .find(query)
            .sort("physical_pages", 1)
            .limit(limit)
        )
        return [EvidenceUnit.model_validate(row) async for row in cursor]

    async def _hit(self, unit: EvidenceUnit, score: Any) -> ChunkHit:
        if unit.id is None:
            raise RuntimeError("Retrieved evidence unit has no Mongo id")
        document = await PdfDocumentRecord.get(unit.document_id)
        return ChunkHit(
            document_id=unit.id,
            document_url=self._document_url(unit, document),
            document_title=document.title if document is not None else None,
            document_meta={
                "pipeline": "pdf_pipeline",
                "evidence_id": unit.evidence_id,
                "section_id": unit.section_id,
                "source_text": unit.source_text,
                "physical_pages": list(unit.physical_pages),
                "printed_pages": list(unit.printed_pages),
                "events": [event.model_dump(mode="json") for event in unit.events],
                "verified_visual_facts": [
                    fact.model_dump(mode="json") for fact in unit.verified_visual_facts
                ],
                "source_regions": [
                    region.model_dump(mode="json") for region in unit.source_regions
                ],
                "visual_artifacts": self._visual_artifacts(unit),
            },
            document_type=DocumentType.PDF,
            document_source="PdfEvidencePipeline",
            chunk_index=unit.physical_pages[0] - 1,
            chunk_text=await self._bundle(unit),
            countries_iso3=[country.iso3 for country in unit.countries],
            score=float(score) if isinstance(score, (int, float)) else None,
        )

    def _visual_artifacts(self, unit: EvidenceUnit) -> list[dict[str, Any]]:
        if unit.source_modality == "text" and not unit.verified_visual_facts:
            return []
        verified_region_ids = {
            region_id
            for fact in unit.verified_visual_facts
            for region_id in fact.supporting_region_ids
        }
        artifacts: list[dict[str, Any]] = []
        seen_sha256: set[str] = set()
        for region in unit.source_regions:
            artifact = region.crop_artifact or region.page_artifact
            if verified_region_ids and region.region_id not in verified_region_ids:
                continue
            if artifact.sha256 in seen_sha256:
                continue
            seen_sha256.add(artifact.sha256)
            artifacts.append(
                {
                    "artifact_id": region.region_id,
                    "path": str(
                        (self.config.artifact_dir / artifact.relative_path).resolve()
                    ),
                    "sha256": artifact.sha256,
                    "media_type": artifact.media_type,
                    "physical_page": region.physical_page,
                }
            )
        return artifacts

    def _document_url(
        self, unit: EvidenceUnit, document: PdfDocumentRecord | None
    ) -> str:
        if document is not None and document.discovered_paths:
            return file_document_uri(Path(document.discovered_paths[0]))
        artifact_path = (
            unit.source_regions[0].page_artifact.relative_path.rsplit("/pages/", 1)[0]
            + "/source.pdf"
        )
        return file_document_uri(Path(artifact_path))

    async def _bundle(self, unit: EvidenceUnit) -> str:
        parts = [unit.canonical_evidence_text]
        ids = [*unit.scope_evidence_ids, *unit.continuation_evidence_ids]
        if ids:
            cursor = EvidenceUnit.get_pymongo_collection().find(
                {"evidence_id": {"$in": ids}}
            )
            related: dict[str, EvidenceUnit] = {}
            async for row in cursor:
                item = EvidenceUnit.model_validate(row)
                related[item.evidence_id] = item
            for evidence_id in ids:
                related_unit = related.get(evidence_id)
                if related_unit is None:
                    continue
                label = (
                    "SCOPE SOURCE EVIDENCE"
                    if evidence_id in unit.scope_evidence_ids
                    else "CONTINUATION SOURCE EVIDENCE"
                )
                page = related_unit.physical_pages[0]
                printed = (
                    related_unit.printed_pages[0]
                    if related_unit.printed_pages
                    else None
                )
                page_label = f"physical page {page}"
                if printed is not None:
                    page_label += f" | printed page {printed}"
                if related_unit.source_text:
                    parts.append(
                        f"[{label} | {page_label}]\n{related_unit.source_text}"
                    )
                for fact in related_unit.verified_visual_facts:
                    parts.append(
                        f"[VERIFIED VISUAL FACT | {page_label} | region {','.join(fact.supporting_region_ids)}]\n{fact.text}"
                    )
        return "\n\n".join(part for part in parts if part)
