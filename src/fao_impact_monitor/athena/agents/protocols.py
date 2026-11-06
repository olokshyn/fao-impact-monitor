"""Agent ports. Tests substitute fakes; production uses the OpenAI Agents SDK."""

from __future__ import annotations

from typing import Protocol

from fao_impact_monitor.athena.agents.schemas import (
    DiscoveryRequest,
    DiscoveryResult,
    ExternalHit,
    ExtractionRequest,
    ExtractionResult,
    OntologyDecision,
    OntologyRequest,
    ResearchPlan,
    Synthesis,
    VerificationJudgment,
    VerificationPacket,
)


class ClaimExtractor(Protocol):
    async def extract(self, request: ExtractionRequest) -> ExtractionResult: ...


class EvidenceVerifier(Protocol):
    async def judge(self, packet: VerificationPacket) -> VerificationJudgment: ...


class RelationProposer(Protocol):
    async def propose(self, request: DiscoveryRequest) -> DiscoveryResult: ...


class OntologyProposer(Protocol):
    async def propose(self, request: OntologyRequest) -> OntologyDecision: ...


class ResearchReasoner(Protocol):
    async def plan(self, question: str) -> ResearchPlan: ...

    async def synthesize(self, question: str, pack: str) -> Synthesis: ...


class ExternalRetriever(Protocol):
    enabled: bool

    async def search(self, query: str) -> list[ExternalHit]: ...
