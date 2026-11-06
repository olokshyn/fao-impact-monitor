"""Compose Fuseki, object storage, MongoDB, and the worker broker."""

from __future__ import annotations

from typing import Any

from pymongo import AsyncMongoClient, MongoClient

from fao_impact_monitor.athena.agents.discovery import SdkRelationProposer
from fao_impact_monitor.athena.agents.external import DisabledExternalRetriever
from fao_impact_monitor.athena.agents.ingestion import SdkClaimExtractor
from fao_impact_monitor.athena.agents.ontology import SdkOntologyProposer
from fao_impact_monitor.athena.agents.research import SdkResearchReasoner
from fao_impact_monitor.athena.agents.verification import SdkEvidenceVerifier
from fao_impact_monitor.athena.config import (
    AthenaConfig,
    get_athena_config,
    policies_from_config,
)
from fao_impact_monitor.athena.context import AthenaContext
from fao_impact_monitor.athena.jobs.batch import BatchGate
from fao_impact_monitor.athena.jobs.broker import CeleryBroker
from fao_impact_monitor.athena.observe import MemoryDecisionLog, MongoDecisionLog
from fao_impact_monitor.athena.storage.graph import FusekiGraph, ensure_ontology
from fao_impact_monitor.athena.storage.index import BedrockEmbedder, MongoRetrievalIndex
from fao_impact_monitor.athena.storage.objects import GcsObjectStore, LocalObjectStore
from fao_impact_monitor.config import get_config


def build_context(config: AthenaConfig | None = None) -> AthenaContext:
    settings = config or get_athena_config()
    graph = FusekiGraph(settings.fuseki_query_url, settings.fuseki_update_url)
    ensure_ontology(graph)
    if settings.object_store == "gcs":
        objects: LocalObjectStore | GcsObjectStore = GcsObjectStore(settings.gcs_bucket)
    else:
        objects = LocalObjectStore(settings.object_store_root)
    mongo = get_config().mongo
    sync: MongoClient[dict[str, Any]] = MongoClient(mongo.uri)
    async_client: AsyncMongoClient[dict[str, Any]] = AsyncMongoClient(mongo.uri)
    database = settings.mongo_db_name
    decisions = MemoryDecisionLog(MongoDecisionLog(sync[database]["decision_log"]))
    index = MongoRetrievalIndex(
        async_client[database]["retrieval"],
        vector_index=settings.vector_index_name,
        text_index=settings.text_index_name,
    )
    embedder = BedrockEmbedder()
    broker = CeleryBroker()
    gate = BatchGate(graph, broker, policies_from_config(settings))
    return AthenaContext(
        config=settings,
        graph=graph,
        objects=objects,
        index=index,
        decisions=decisions,
        embedder=embedder,
        extractor=SdkClaimExtractor(settings.ingestion_model, decisions),
        verifier=SdkEvidenceVerifier(settings.verifier_model, decisions),
        discoverer=SdkRelationProposer(settings.discovery_model, decisions),
        ontologist=SdkOntologyProposer(settings.ontology_model, decisions),
        researcher=SdkResearchReasoner(settings.research_model, decisions),
        external=DisabledExternalRetriever(),
        broker=broker,
        gate=gate,
    )
