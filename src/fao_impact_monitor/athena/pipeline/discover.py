"""Discover relations for one claim against a bounded candidate set."""

from __future__ import annotations

from fao_impact_monitor.athena.agents.schemas import DiscoveryRequest
from fao_impact_monitor.athena.context import AthenaContext
from fao_impact_monitor.athena.model.filters import FilterSpec
from fao_impact_monitor.athena.model.records import ClaimRecord, EvidenceRecord
from fao_impact_monitor.athena.model.status import VerificationStatus
from fao_impact_monitor.athena.pipeline.assertions import persist_proposed
from fao_impact_monitor.athena.pipeline.load import load_claims, load_evidence
from fao_impact_monitor.athena.retrieval.candidates import retrieve_candidates
from fao_impact_monitor.athena.storage.ontology import (
    LocationHierarchy,
    relation_definitions,
)
from fao_impact_monitor.athena.storage.rdf import read_model


async def discover_for_claim(
    ctx: AthenaContext,
    claim_iri: str,
    *,
    spec: FilterSpec | None = None,
    extra: list[ClaimRecord] | None = None,
) -> list[str]:
    claim = read_model(ctx.graph, claim_iri, ClaimRecord)
    if claim is None or claim.verification_status != VerificationStatus.VERIFIED:
        return []
    evidence = load_evidence(ctx.graph)
    hierarchy = LocationHierarchy.from_graph(ctx.graph)
    active = spec or FilterSpec()
    candidates = await retrieve_candidates(
        ctx.graph,
        ctx.index,
        ctx.embedder,
        query=claim.normalized,
        entities=claim.entities,
        spec=active,
        evidence=evidence,
        ancestors_of=hierarchy.ancestors,
        limit=ctx.config.candidate_limit,
        seed_claim=claim.iri,
    )
    others = [item for item in candidates if item.iri != claim.iri]
    seen = {item.iri for item in others}
    for item in extra or []:
        if item.iri != claim.iri and item.iri not in seen:
            others.append(item)
            seen.add(item.iri)
    if not others:
        return []
    proposal = await ctx.discoverer.propose(
        DiscoveryRequest(
            claim_text=f"{claim.iri}\n{claim.normalized}",
            candidates=[f"{item.iri}\n{item.normalized}" for item in others],
            existing_relations=[
                item.label for item in relation_definitions(ctx.graph).values()
            ],
        )
    )
    known = load_claims(ctx.graph)
    created: list[str] = []
    for relation in proposal.relations:
        subject = known.get(relation.subject_iri, claim)
        obj = known.get(relation.object_iri)
        if obj is None or subject.iri == obj.iri:
            continue
        source_text = _excerpts(
            evidence, relation.evidence_iris or subject.evidence_iris
        )
        stored = await persist_proposed(
            ctx,
            subject=subject,
            obj=obj,
            relation_name=relation.relation,
            evidence_iris=relation.evidence_iris or subject.evidence_iris,
            conditions=relation.conditions,
            scientific=relation.scientific_status,
            tentative=relation.tentative,
            study_design=relation.study_design,
            source_text=source_text,
            new_type=relation.new_type,
        )
        if stored is not None:
            created.append(stored.iri)
    return created


def _excerpts(evidence: dict[str, EvidenceRecord], iris: list[str]) -> str:
    parts = [evidence[iri].excerpt for iri in iris if iri in evidence]
    return "\n".join(parts)
