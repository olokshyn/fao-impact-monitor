"""MongoDB Atlas vector + BM25 hybrid search over chunk embeddings."""

from __future__ import annotations

import logging
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Annotated, Any, ClassVar

from beanie import Document as BeanieDocument
from beanie import Indexed, PydanticObjectId
from langchain_aws import BedrockEmbeddings
from pydantic import BaseModel, Field
from pymongo import ASCENDING, IndexModel
from pymongo.asynchronous.collection import AsyncCollection
from pymongo.operations import SearchIndexModel

from fao_impact_monitor.config import (
    AwsBedrockConfig,
    VectorStoreConfig,
    get_config,
)
from fao_impact_monitor.data_lake.document import DocumentType

# Titan Text Embeddings V2 default output size when dimensions is unset.
DEFAULT_EMBEDDING_DIMENSIONS = 1024

logger = logging.getLogger(__name__)

EmbedQueryFn = Callable[[str], Awaitable[list[float]]]
AggregateFn = Callable[[list[dict[str, Any]]], Awaitable[list[dict[str, Any]]]]


class ChunkFields(BaseModel):
    """Fields shared by stored embeddings and search hits."""

    document_id: Annotated[PydanticObjectId, Indexed()]
    document_url: Annotated[str, Indexed()]
    document_external_id: str | None = None
    document_title: str | None = None
    document_meta: dict[str, Any] = Field(default_factory=dict)
    document_type: DocumentType
    document_source: str | None = None
    chunk_index: int
    chunk_text: str
    countries_iso3: list[str] = Field(default_factory=list)


class ChunkEmbedding(BeanieDocument, ChunkFields):
    """One embedded chunk stored for vector / BM25 retrieval."""

    embedding: list[float]

    class Settings:
        name = "embeddings"
        indexes: ClassVar[list[IndexModel]] = [
            IndexModel(
                [("document_url", ASCENDING), ("document_source", ASCENDING)],
            ),
        ]


class ChunkHit(ChunkFields):
    """A retrieved chunk with fusion / search score."""

    score: float | None = None


def embedding_dimensions(config: VectorStoreConfig | None = None) -> int:
    cfg = config or get_config().vector_store
    if cfg.embedding_dimensions is not None:
        return cfg.embedding_dimensions
    return DEFAULT_EMBEDDING_DIMENSIONS


def build_embeddings(
    *,
    vector_store_config: VectorStoreConfig | None = None,
    aws_bedrock_config: AwsBedrockConfig | None = None,
) -> BedrockEmbeddings:
    """Build LangChain BedrockEmbeddings for Titan (bedrock-runtime)."""
    config = get_config()
    vs = vector_store_config or config.vector_store
    aws = aws_bedrock_config or config.aws_bedrock
    api_key = aws.api_key.get_secret_value()
    if api_key:
        # boto3 picks this up for InvokeModel bearer-token auth.
        os.environ["AWS_BEARER_TOKEN_BEDROCK"] = api_key
    kwargs: dict[str, Any] = {
        "model_id": vs.embedding_model,
        "region_name": aws.region,
        "model_kwargs": {"normalize": True},
    }
    if vs.embedding_dimensions is not None:
        kwargs["dimensions"] = vs.embedding_dimensions
    return BedrockEmbeddings(**kwargs)


def vector_search_index_definition(
    config: VectorStoreConfig | None = None,
) -> dict[str, Any]:
    """Atlas Vector Search index definition.

    ``countries_iso3`` is a required filter field so queries can filter by
    country efficiently inside ``$vectorSearch``.
    """
    cfg = config or get_config().vector_store
    return {
        "name": cfg.vector_index_name,
        "type": "vectorSearch",
        "definition": {
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": embedding_dimensions(cfg),
                    "similarity": "cosine",
                },
                {
                    "type": "filter",
                    "path": "countries_iso3",
                },
            ]
        },
    }


def text_search_index_definition(
    config: VectorStoreConfig | None = None,
) -> dict[str, Any]:
    """Atlas Search (Lucene/BM25) index definition on ``chunk_text``.

    ``countries_iso3`` is indexed as a token field for efficient filtering.
    """
    cfg = config or get_config().vector_store
    return {
        "name": cfg.text_index_name,
        "type": "search",
        "definition": {
            "mappings": {
                "dynamic": False,
                "fields": {
                    "chunk_text": {
                        "type": "string",
                        "analyzer": "lucene.standard",
                    },
                    "countries_iso3": {
                        "type": "token",
                    },
                },
            }
        },
    }


def _existing_search_definition(existing: Mapping[str, Any]) -> Any:
    return existing.get("latestDefinition", existing.get("definition"))


async def ensure_indexes(
    collection: AsyncCollection[Any],
    *,
    config: VectorStoreConfig | None = None,
) -> None:
    """Create vector + text search indexes, recreating on definition mismatch.

    Atlas rejects ``create_search_index`` when a same-named index already exists
    with a different definition (e.g. after changing embedding dimensions).
    """
    cfg = config or get_config().vector_store
    await collection.create_index("document_id")
    await collection.create_index("document_url")
    await collection.create_index(
        [("document_url", ASCENDING), ("document_source", ASCENDING)],
    )

    existing_by_name: dict[str, Mapping[str, Any]] = {}
    async for doc in await collection.list_search_indexes():
        name = doc.get("name")
        if isinstance(name, str):
            existing_by_name[name] = doc

    for index_spec in (
        vector_search_index_definition(cfg),
        text_search_index_definition(cfg),
    ):
        name = index_spec["name"]
        desired = index_spec["definition"]
        existing = existing_by_name.get(name)
        if existing is not None:
            if _existing_search_definition(existing) == desired:
                logger.info("Search index %s already up to date", name)
                continue
            logger.info(
                "Search index %s definition changed; dropping and recreating",
                name,
            )
            await collection.drop_search_index(name)

        await collection.create_search_index(
            SearchIndexModel(
                definition=desired,
                name=name,
                type=index_spec["type"],
            )
        )
        logger.info("Created search index %s (%s)", name, index_spec["type"])


def _countries_filter(
    countries_iso3: Sequence[str] | None,
) -> dict[str, Any] | None:
    if not countries_iso3:
        return None
    return {"countries_iso3": {"$in": list(countries_iso3)}}


def build_vector_pipeline(
    query_vector: Sequence[float],
    *,
    config: VectorStoreConfig | None = None,
    countries_iso3: Sequence[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    cfg = config or get_config().vector_store
    vs: dict[str, Any] = {
        "index": cfg.vector_index_name,
        "path": "embedding",
        "queryVector": list(query_vector),
        "numCandidates": cfg.vector_num_candidates,
        "limit": limit if limit is not None else cfg.limit,
    }
    filt = _countries_filter(countries_iso3)
    if filt is not None:
        vs["filter"] = filt
    return [
        {"$vectorSearch": vs},
        {
            "$addFields": {
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]


def build_bm25_pipeline(
    query: str,
    *,
    config: VectorStoreConfig | None = None,
    countries_iso3: Sequence[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    cfg = config or get_config().vector_store
    must: list[dict[str, Any]] = [
        {
            "text": {
                "query": query,
                "path": "chunk_text",
            }
        }
    ]
    if countries_iso3:
        must.append(
            {
                "in": {
                    "path": "countries_iso3",
                    "value": list(countries_iso3),
                }
            }
        )
    return [
        {
            "$search": {
                "index": cfg.text_index_name,
                "compound": {"must": must},
            }
        },
        {"$limit": limit if limit is not None else cfg.limit},
        {
            "$addFields": {
                "score": {"$meta": "searchScore"},
            }
        },
    ]


def build_hybrid_pipeline(
    query: str,
    query_vector: Sequence[float],
    *,
    config: VectorStoreConfig | None = None,
    countries_iso3: Sequence[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Build ``$rankFusion`` pipeline combining vector + BM25 search."""
    cfg = config or get_config().vector_store
    result_limit = limit if limit is not None else cfg.limit
    vector_stages = [
        stage
        for stage in build_vector_pipeline(
            query_vector,
            config=cfg,
            countries_iso3=countries_iso3,
            limit=result_limit,
        )
        if "$addFields" not in stage
    ]
    text_stages = [
        stage
        for stage in build_bm25_pipeline(
            query,
            config=cfg,
            countries_iso3=countries_iso3,
            limit=result_limit,
        )
        if "$addFields" not in stage
    ]
    return [
        {
            "$rankFusion": {
                "input": {
                    "pipelines": {
                        "vectorPipeline": vector_stages,
                        "textPipeline": text_stages,
                    }
                },
                "combination": {
                    "weights": {
                        "vectorPipeline": cfg.vector_weight,
                        "textPipeline": cfg.text_weight,
                    }
                },
            }
        },
        {"$limit": result_limit},
        {
            "$addFields": {
                "score": {"$meta": "score"},
            }
        },
    ]


def _hit_from_doc(doc: Mapping[str, Any]) -> ChunkHit:
    return ChunkHit.model_validate(doc)


async def _default_aggregate(
    pipeline: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    collection = ChunkEmbedding.get_pymongo_collection()
    cursor = await collection.aggregate(pipeline)
    return [doc async for doc in cursor]


async def _default_embed_query(query: str) -> list[float]:
    embeddings = build_embeddings()
    return await embeddings.aembed_query(query)


class VectorStore:
    """Lookup API over the embeddings collection."""

    def __init__(
        self,
        *,
        config: VectorStoreConfig | None = None,
        embed_query_fn: EmbedQueryFn | None = None,
        aggregate_fn: AggregateFn | None = None,
    ) -> None:
        self._config = config
        self._embed_query_fn = embed_query_fn
        self._aggregate_fn = aggregate_fn

    @property
    def config(self) -> VectorStoreConfig:
        return self._config or get_config().vector_store

    async def _embed_query(self, query: str) -> list[float]:
        if not query.strip():
            raise ValueError("Cannot embed an empty query")
        if self._embed_query_fn is not None:
            return await self._embed_query_fn(query)
        return await _default_embed_query(query)

    async def _aggregate(self, pipeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self._aggregate_fn is not None:
            return await self._aggregate_fn(pipeline)
        return await _default_aggregate(pipeline)

    async def search_vector(
        self,
        query: str,
        *,
        countries_iso3: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> list[ChunkHit]:
        if not query.strip():
            logger.warning("search_vector skipped: empty query")
            return []
        query_vector = await self._embed_query(query)
        pipeline = build_vector_pipeline(
            query_vector,
            config=self.config,
            countries_iso3=countries_iso3,
            limit=limit,
        )
        docs = await self._aggregate(pipeline)
        return [_hit_from_doc(doc) for doc in docs]

    async def search_bm25(
        self,
        query: str,
        *,
        countries_iso3: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> list[ChunkHit]:
        if not query.strip():
            logger.warning("search_bm25 skipped: empty query")
            return []
        pipeline = build_bm25_pipeline(
            query,
            config=self.config,
            countries_iso3=countries_iso3,
            limit=limit,
        )
        docs = await self._aggregate(pipeline)
        return [_hit_from_doc(doc) for doc in docs]

    async def search(
        self,
        query: str,
        *,
        countries_iso3: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> list[ChunkHit]:
        """Hybrid BM25 + vector search fused with ``$rankFusion``."""
        if not query.strip():
            logger.warning("search skipped: empty query")
            return []
        query_vector = await self._embed_query(query)
        pipeline = build_hybrid_pipeline(
            query,
            query_vector,
            config=self.config,
            countries_iso3=countries_iso3,
            limit=limit,
        )
        docs = await self._aggregate(pipeline)
        return [_hit_from_doc(doc) for doc in docs]
