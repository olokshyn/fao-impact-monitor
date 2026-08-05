"""Unit tests for vectorstore pipeline builders and VectorStore search API."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

from beanie import PydanticObjectId

from fao_impact_monitor.config import VectorStoreConfig
from fao_impact_monitor.data_lake.document import DocumentType
from fao_impact_monitor.data_lake.vectorstore import (
    DEFAULT_EMBEDDING_DIMENSIONS,
    VectorStore,
    build_bm25_pipeline,
    build_hybrid_pipeline,
    build_vector_pipeline,
    ensure_indexes,
    text_search_index_definition,
    vector_search_index_definition,
)

T = TypeVar("T")
RunAsync = Callable[[Coroutine[Any, Any, T]], T]


def _cfg() -> VectorStoreConfig:
    return VectorStoreConfig(
        embedding_model="amazon.titan-embed-text-v2:0",
        embedding_dimensions=8,
        vector_index_name="vec_idx",
        text_index_name="text_idx",
        vector_num_candidates=50,
        limit=5,
        vector_weight=0.7,
        text_weight=0.3,
    )


def test_vector_index_includes_countries_filter() -> None:
    definition = vector_search_index_definition(_cfg())
    assert definition["name"] == "vec_idx"
    assert definition["type"] == "vectorSearch"
    fields = definition["definition"]["fields"]
    assert fields[0]["path"] == "embedding"
    assert fields[0]["numDimensions"] == 8
    assert fields[1] == {"type": "filter", "path": "countries_iso3"}


def test_text_index_includes_chunk_text_and_countries() -> None:
    definition = text_search_index_definition(_cfg())
    assert definition["name"] == "text_idx"
    assert definition["type"] == "search"
    fields = definition["definition"]["mappings"]["fields"]
    assert "chunk_text" in fields
    assert fields["countries_iso3"]["type"] == "token"


def test_default_embedding_dimensions() -> None:
    cfg = VectorStoreConfig(embedding_dimensions=None)
    definition = vector_search_index_definition(cfg)
    assert (
        definition["definition"]["fields"][0]["numDimensions"]
        == DEFAULT_EMBEDDING_DIMENSIONS
    )


def test_ensure_indexes_recreates_mismatched_vector_index(
    run_async: RunAsync[Any],
) -> None:
    cfg = _cfg()
    desired = vector_search_index_definition(cfg)
    stale = {
        "name": cfg.vector_index_name,
        "type": "vectorSearch",
        "latestDefinition": {
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": 3072,
                    "similarity": "cosine",
                }
            ]
        },
    }
    text_desired = text_search_index_definition(cfg)
    existing = [
        stale,
        {"name": cfg.text_index_name, "latestDefinition": text_desired["definition"]},
    ]

    class _Cursor:
        def __init__(self, docs: list[dict[str, Any]]) -> None:
            self._docs = docs

        def __aiter__(self) -> _Cursor:
            self._iter = iter(self._docs)
            return self

        async def __anext__(self) -> dict[str, Any]:
            try:
                return next(self._iter)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

    class _Collection:
        def __init__(self) -> None:
            self.dropped: list[str] = []
            self.created: list[Any] = []

        async def create_index(self, _key: Any, **_kwargs: Any) -> str:
            return "ok"

        async def list_search_indexes(self) -> _Cursor:
            return _Cursor(existing)

        async def drop_search_index(self, name: str) -> None:
            self.dropped.append(name)

        async def create_search_index(self, model: Any) -> str:
            self.created.append(model)
            name = model.document.get("name", "idx")
            return str(name)

    collection = _Collection()
    run_async(ensure_indexes(collection, config=cfg))  # type: ignore[arg-type]
    assert collection.dropped == [cfg.vector_index_name]
    assert len(collection.created) == 1
    assert collection.created[0].document["name"] == cfg.vector_index_name
    assert (
        collection.created[0].document["definition"]["fields"][0]["numDimensions"]
        == desired["definition"]["fields"][0]["numDimensions"]
    )


def test_build_vector_pipeline_with_country_filter() -> None:
    pipeline = build_vector_pipeline(
        [0.1, 0.2],
        config=_cfg(),
        countries_iso3=["KEN"],
        limit=3,
    )
    assert pipeline[0]["$vectorSearch"]["index"] == "vec_idx"
    assert pipeline[0]["$vectorSearch"]["filter"] == {
        "countries_iso3": {"$in": ["KEN"]}
    }
    assert pipeline[0]["$vectorSearch"]["limit"] == 3
    assert pipeline[1]["$addFields"]["score"] == {"$meta": "vectorSearchScore"}


def test_build_bm25_pipeline_with_country_filter() -> None:
    pipeline = build_bm25_pipeline(
        "drought impact",
        config=_cfg(),
        countries_iso3=["KEN", "UGA"],
    )
    search = pipeline[0]["$search"]
    assert search["index"] == "text_idx"
    must = search["compound"]["must"]
    assert must[0]["text"]["path"] == "chunk_text"
    assert must[1]["in"] == {
        "path": "countries_iso3",
        "value": ["KEN", "UGA"],
    }


def test_build_hybrid_pipeline_uses_rank_fusion() -> None:
    pipeline = build_hybrid_pipeline(
        "food security",
        [0.1, 0.2],
        config=_cfg(),
        countries_iso3=["KEN"],
    )
    fusion = pipeline[0]["$rankFusion"]
    assert "vectorPipeline" in fusion["input"]["pipelines"]
    assert "textPipeline" in fusion["input"]["pipelines"]
    assert fusion["combination"]["weights"] == {
        "vectorPipeline": 0.7,
        "textPipeline": 0.3,
    }
    vector_stages = fusion["input"]["pipelines"]["vectorPipeline"]
    assert "$vectorSearch" in vector_stages[0]
    assert vector_stages[0]["$vectorSearch"]["filter"]["countries_iso3"]["$in"] == [
        "KEN"
    ]
    text_stages = fusion["input"]["pipelines"]["textPipeline"]
    assert "$search" in text_stages[0]
    assert pipeline[1] == {"$limit": 5}


def test_vectorstore_search_calls_hybrid_aggregate(
    run_async: RunAsync[Any],
) -> None:
    captured: list[list[dict[str, Any]]] = []
    doc_id = PydanticObjectId()

    async def embed_query(query: str) -> list[float]:
        assert query == "maize yields"
        return [0.1, 0.2]

    async def aggregate(pipeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
        captured.append(pipeline)
        return [
            {
                "document_id": doc_id,
                "document_url": "https://example.com/a.pdf",
                "document_external_id": None,
                "document_title": "A",
                "document_meta": {},
                "document_type": DocumentType.PDF,
                "document_source": "FaoRepository",
                "chunk_index": 0,
                "chunk_text": "Maize yields fell in Kenya.",
                "countries_iso3": ["KEN"],
                "score": 0.9,
            }
        ]

    store = VectorStore(
        config=_cfg(),
        embed_query_fn=embed_query,
        aggregate_fn=aggregate,
    )
    hits = run_async(store.search("maize yields", countries_iso3=["KEN"]))
    assert len(hits) == 1
    assert hits[0].chunk_text.startswith("Maize")
    assert hits[0].countries_iso3 == ["KEN"]
    assert hits[0].score == 0.9
    assert "$rankFusion" in captured[0][0]


def test_vectorstore_search_vector_and_bm25(
    run_async: RunAsync[Any],
) -> None:
    seen: list[str] = []
    doc_id = PydanticObjectId()

    async def embed_query(query: str) -> list[float]:
        return [1.0]

    async def aggregate(pipeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if "$vectorSearch" in pipeline[0]:
            seen.append("vector")
        elif "$search" in pipeline[0]:
            seen.append("bm25")
        return [
            {
                "document_id": doc_id,
                "document_url": "https://example.com/a.pdf",
                "document_type": DocumentType.PDF,
                "chunk_index": 0,
                "chunk_text": "text",
                "countries_iso3": [],
                "score": 1.0,
            }
        ]

    store = VectorStore(
        config=_cfg(),
        embed_query_fn=embed_query,
        aggregate_fn=aggregate,
    )
    run_async(store.search_vector("q"))
    run_async(store.search_bm25("q"))
    assert seen == ["vector", "bm25"]


def test_vectorstore_search_skips_empty_query_without_embedding(
    run_async: RunAsync[Any],
) -> None:
    called = False

    async def embed_query(query: str) -> list[float]:
        nonlocal called
        called = True
        return [0.1]

    async def aggregate(pipeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
        raise AssertionError(f"aggregate should not run for empty query: {pipeline}")

    store = VectorStore(
        config=_cfg(),
        embed_query_fn=embed_query,
        aggregate_fn=aggregate,
    )
    assert run_async(store.search("   ")) == []
    assert run_async(store.search_vector("")) == []
    assert run_async(store.search_bm25("\n")) == []
    assert called is False
