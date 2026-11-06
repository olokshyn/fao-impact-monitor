"""Runtime dependencies shared by the API and workers."""

from __future__ import annotations

from dataclasses import dataclass

from fao_impact_monitor.athena.agents.protocols import (
    ClaimExtractor,
    EvidenceVerifier,
    ExternalRetriever,
    OntologyProposer,
    RelationProposer,
    ResearchReasoner,
)
from fao_impact_monitor.athena.config import AthenaConfig
from fao_impact_monitor.athena.jobs.batch import BatchGate, Broker
from fao_impact_monitor.athena.observe import DecisionLog
from fao_impact_monitor.athena.storage.graph import GraphStore
from fao_impact_monitor.athena.storage.index import Embedder, RetrievalIndex
from fao_impact_monitor.athena.storage.objects import ObjectStore


@dataclass
class AthenaContext:
    config: AthenaConfig
    graph: GraphStore
    objects: ObjectStore
    index: RetrievalIndex
    decisions: DecisionLog
    embedder: Embedder
    extractor: ClaimExtractor
    verifier: EvidenceVerifier
    discoverer: RelationProposer
    ontologist: OntologyProposer
    researcher: ResearchReasoner
    external: ExternalRetriever
    broker: Broker
    gate: BatchGate
