"""Shared fakes for Athena unit tests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from fao_impact_monitor.athena.agents.schemas import (
    ClaimDraft,
    DiscoveryRequest,
    DiscoveryResult,
    EvidenceDraft,
    ExtractionRequest,
    ExtractionResult,
    OntologyDecision,
    OntologyRequest,
    ProposedRelation,
    ResearchPlan,
    SubQuestion,
    Synthesis,
    VerificationJudgment,
    VerificationPacket,
)
from fao_impact_monitor.athena.config import AthenaConfig, policies_from_config
from fao_impact_monitor.athena.context import AthenaContext
from fao_impact_monitor.athena.jobs.batch import BatchGate, InProcessBroker
from fao_impact_monitor.athena.model.status import ScientificStatus
from fao_impact_monitor.athena.observe import DecisionRecord, MemoryDecisionLog
from fao_impact_monitor.athena.storage.graph import MemoryGraph, ensure_ontology
from fao_impact_monitor.athena.storage.index import HashEmbedder, MemoryIndex
from fao_impact_monitor.athena.storage.objects import LocalObjectStore


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 1, 1, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


class RecordingExtractor:
    def __init__(self, result: ExtractionResult | None = None) -> None:
        self.result = result
        self.calls = 0

    async def extract(self, request: ExtractionRequest) -> ExtractionResult:
        self.calls += 1
        if self.result is not None:
            return self.result
        excerpt = request.text.strip().split("\n", 1)[0]
        return ExtractionResult(
            evidence=[EvidenceDraft(excerpt=excerpt)],
            claims=[
                ClaimDraft(
                    original_wording=excerpt,
                    normalized=excerpt,
                    entities=["soil biota"],
                    applicable_tags=request.tags,
                    evidence_index=0,
                    concept_label="soil biota",
                )
            ],
        )


class RecordingVerifier:
    def __init__(
        self, *, causality: bool | None = True, faithful: bool | None = True
    ) -> None:
        self.causality = causality
        self.faithful = faithful
        self.calls = 0

    async def judge(self, packet: VerificationPacket) -> VerificationJudgment:
        self.calls += 1
        return VerificationJudgment(
            faithful=self.faithful,
            relation_supported=True,
            context_compatible=True,
            derivation_valid=True,
            expresses_causality=self.causality,
            justification="The excerpt supports the recorded statement.",
        )


class RecordingDiscoverer:
    def __init__(self, relations: list[ProposedRelation] | None = None) -> None:
        self.relations = relations or []
        self.calls = 0

    async def propose(self, request: DiscoveryRequest) -> DiscoveryResult:
        self.calls += 1
        return DiscoveryResult(relations=list(self.relations))


class RecordingOntologist:
    def __init__(self, decision: OntologyDecision | None = None) -> None:
        self.decision = decision or OntologyDecision(reuse_name="causes")

    async def propose(self, request: OntologyRequest) -> OntologyDecision:
        del request
        return self.decision


class RecordingResearcher:
    def __init__(self) -> None:
        self.packs: list[str] = []

    async def plan(self, question: str) -> ResearchPlan:
        return ResearchPlan(
            subquestions=[SubQuestion(text=question, variables=["soil biota"])],
            assumptions=["Only admissible evidence is used."],
        )

    async def synthesize(self, question: str, pack: str) -> Synthesis:
        del question
        self.packs.append(pack)
        return Synthesis(
            answer="Qualitative assessment.",
            assumptions=["No quantitative model is available."],
            gaps=["A field model is missing."],
            qualitative=True,
        )


def make_context(
    tmp_path: Path,
    *,
    extractor: RecordingExtractor | None = None,
    verifier: RecordingVerifier | None = None,
    discoverer: RecordingDiscoverer | None = None,
    ontologist: RecordingOntologist | None = None,
    researcher: RecordingResearcher | None = None,
    batch_size: int = 1,
    wait_seconds: float = 0,
    clock: Clock | None = None,
    broker: InProcessBroker | None = None,
) -> AthenaContext:
    config = AthenaConfig(
        object_store_root=tmp_path / "store",
        ingest_batch_size=batch_size,
        ingest_batch_wait_seconds=wait_seconds,
        verify_batch_size=batch_size,
        verify_batch_wait_seconds=wait_seconds,
        discover_batch_size=batch_size,
        discover_batch_wait_seconds=wait_seconds,
        backfill_batch_size=batch_size,
        backfill_batch_wait_seconds=wait_seconds,
        index_batch_size=batch_size,
        index_batch_wait_seconds=wait_seconds,
        research_batch_size=1,
        research_batch_wait_seconds=0,
        maintain_batch_size=batch_size,
        maintain_batch_wait_seconds=wait_seconds,
        candidate_limit=5,
        research_max_rounds=1,
    )
    graph = MemoryGraph()
    ensure_ontology(graph)
    decisions = MemoryDecisionLog()
    chosen_broker = broker or InProcessBroker()
    return AthenaContext(
        config=config,
        graph=graph,
        objects=LocalObjectStore(tmp_path / "objects"),
        index=MemoryIndex(),
        decisions=decisions,
        embedder=HashEmbedder(dimensions=8),
        extractor=extractor or RecordingExtractor(),
        verifier=verifier or RecordingVerifier(),
        discoverer=discoverer or RecordingDiscoverer(),
        ontologist=ontologist or RecordingOntologist(),
        researcher=researcher or RecordingResearcher(),
        external=__import__(
            "fao_impact_monitor.athena.agents.external",
            fromlist=["DisabledExternalRetriever"],
        ).DisabledExternalRetriever(),
        broker=chosen_broker,
        gate=BatchGate(
            graph,
            chosen_broker,
            policies_from_config(config),
            clock=clock,
        ),
    )


def published_batches(ctx: AthenaContext) -> list[tuple[str, str]]:
    assert isinstance(ctx.broker, InProcessBroker)
    return ctx.broker.published


def decision_records(ctx: AthenaContext) -> list[DecisionRecord]:
    assert isinstance(ctx.decisions, MemoryDecisionLog)
    return ctx.decisions.records


def causes_proposal(subject: str, obj: str, evidence: str) -> ProposedRelation:
    return ProposedRelation(
        subject_iri=subject,
        object_iri=obj,
        relation="causes",
        evidence_iris=[evidence],
        scientific_status=ScientificStatus.SOURCE_REPORTED,
    )
