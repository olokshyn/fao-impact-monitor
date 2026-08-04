"""Shared MongoDB client and Beanie initialization for the data lake."""

from __future__ import annotations

from typing import Any

from beanie import init_beanie
from pymongo import AsyncMongoClient, MongoClient
from pymongo.asynchronous.database import AsyncDatabase

from fao_impact_monitor.config import MongoConfig, get_config
from fao_impact_monitor.data_lake.document import Document
from fao_impact_monitor.data_lake.documents.pdf_document import PdfDocument
from fao_impact_monitor.data_lake.documents.tellus_document import TellusDocument
from fao_impact_monitor.data_lake.documents.web_page_document import WebPageDocument
from fao_impact_monitor.data_lake.pipeline import (
    PdfCrawlPipeline,
    PdfProcessPipeline,
    Pipeline,
    TellusProcessPipeline,
)
from fao_impact_monitor.data_lake.stage import StageVersion
from fao_impact_monitor.data_lake.stages.country_detect_stage import (
    CountryDetectStageVersion,
)
from fao_impact_monitor.data_lake.stages.embed_chunks_stage import (
    EmbedChunksStageVersion,
)
from fao_impact_monitor.data_lake.stages.pdf_crawl_stage import PdfCrawlStageVersion
from fao_impact_monitor.data_lake.stages.pdf_extract_stage import PdfExtractStageVersion
from fao_impact_monitor.data_lake.stages.tellus_document_fetch_stage import (
    TellusDocumentFetchStageVersion,
)
from fao_impact_monitor.data_lake.vectorstore import ChunkEmbedding

# Beanie document models registered for the data lake (and vectorstore).
DATA_LAKE_DOCUMENT_MODELS: list[type[Any]] = [
    Document,
    WebPageDocument,
    PdfDocument,
    TellusDocument,
    ChunkEmbedding,
    Pipeline,
    PdfCrawlPipeline,
    PdfProcessPipeline,
    TellusProcessPipeline,
    StageVersion,
    PdfCrawlStageVersion,
    PdfExtractStageVersion,
    TellusDocumentFetchStageVersion,
    CountryDetectStageVersion,
    EmbedChunksStageVersion,
]


def get_mongo_config(**overrides: Any) -> MongoConfig:
    """Return Mongo settings from config, optionally overridden."""
    config = get_config().mongo
    if not overrides:
        return config
    return config.model_copy(update=overrides)


def create_mongo_client(config: MongoConfig | None = None) -> MongoClient[Any]:
    """Create a sync ``MongoClient`` from shared config.

    Username/password are embedded in ``MongoConfig.uri``
    (``MONGO_USERNAME`` / ``MONGO_PASSWORD``).
    """
    cfg = config or get_mongo_config()
    return MongoClient(cfg.uri)


def create_async_mongo_client(
    config: MongoConfig | None = None,
) -> AsyncMongoClient[Any]:
    """Create an ``AsyncMongoClient`` from shared config.

    Username/password are embedded in ``MongoConfig.uri``
    (``MONGO_USERNAME`` / ``MONGO_PASSWORD``).
    """
    cfg = config or get_mongo_config()
    return AsyncMongoClient(cfg.uri)


async def init_data_lake_beanie(
    database: AsyncDatabase[Any] | Any,
    *,
    skip_indexes: bool = False,
) -> None:
    """Initialize Beanie with all data-lake document models on ``database``."""
    await init_beanie(
        database=database,
        document_models=DATA_LAKE_DOCUMENT_MODELS,
        skip_indexes=skip_indexes,
    )


async def connect_data_lake(
    config: MongoConfig | None = None,
    *,
    skip_indexes: bool = False,
) -> AsyncMongoClient[Any]:
    """Create an async client and initialize Beanie for the configured database."""
    cfg = config or get_mongo_config()
    client = create_async_mongo_client(cfg)
    await init_data_lake_beanie(client[cfg.db_name], skip_indexes=skip_indexes)
    return client
