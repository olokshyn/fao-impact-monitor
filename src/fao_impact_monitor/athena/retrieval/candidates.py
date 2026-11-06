"""Bounded candidate retrieval. Filters run before ranking and again in Jena."""

from __future__ import annotations

from collections.abc import Callable

from fao_impact_monitor.athena.model.filters import FilterSpec, without_fields
from fao_impact_monitor.athena.model.records import ClaimRecord, EvidenceRecord
from fao_impact_monitor.athena.ns import GRAPH_ACCEPTED, PROVENANCE_FIELDS, term
from fao_impact_monitor.athena.retrieval.admissibility import claim_admission
from fao_impact_monitor.athena.storage.graph import GraphStore
from fao_impact_monitor.athena.storage.index import (
    Embedder,
    RetrievalHit,
    RetrievalIndex,
    RetrievalRecord,
)
from fao_impact_monitor.athena.storage.rdf import n3_iri, read_model


async def retrieve_candidates(
    graph: GraphStore,
    index: RetrievalIndex,
    embedder: Embedder,
    *,
    query: str,
    entities: list[str],
    spec: FilterSpec,
    evidence: dict[str, EvidenceRecord],
    ancestors_of: Callable[[str], set[str]],
    limit: int,
    seed_claim: str | None = None,
) -> list[ClaimRecord]:
    """Union semantic, lexical, entity, and graph hits, then recheck Jena."""
    prefilter = without_fields(spec, set(PROVENANCE_FIELDS))
    vector = (await embedder.embed([query]))[0]
    hits = [
        *(await index.search_semantic(vector, prefilter, limit)),
        *(await index.search_lexical(query, prefilter, limit)),
        *(await index.search_entities(entities, prefilter, limit)),
        *_neighbor_hits(graph, seed_claim),
    ]
    fused: dict[str, RetrievalHit] = {}
    for hit in hits:
        if hit.record.record_type != "claim":
            continue
        current = fused.get(hit.record.jena_id)
        if current is None or hit.score > current.score:
            fused[hit.record.jena_id] = hit
    ranked = sorted(fused.values(), key=lambda hit: hit.score, reverse=True)
    accepted: list[ClaimRecord] = []
    for hit in ranked:
        claim = read_model(graph, hit.record.jena_id, ClaimRecord)
        if claim is None:
            continue
        if claim.record_version != hit.record.version and hit.channel != "graph":
            # Stale index tags are ignored. The Jena record is checked below.
            pass
        decision = claim_admission(claim, evidence, spec, ancestors_of)
        if not decision.allowed:
            continue
        accepted.append(claim)
        if len(accepted) >= limit:
            break
    return accepted


def _neighbor_hits(graph: GraphStore, seed_claim: str | None) -> list[RetrievalHit]:
    if seed_claim is None:
        return []
    rows = graph.query(
        f"""
        SELECT ?other WHERE {{
          GRAPH {n3_iri(GRAPH_ACCEPTED)} {{
            {n3_iri(seed_claim)} {n3_iri(term("concept"))} ?concept .
            ?other a {n3_iri(term("Claim"))} ;
                   {n3_iri(term("concept"))} ?concept .
            FILTER(?other != {n3_iri(seed_claim)})
          }}
        }}
        """
    )
    hits: list[RetrievalHit] = []
    for row in rows:
        claim = read_model(graph, row["other"], ClaimRecord)
        if claim is None:
            continue
        hits.append(
            RetrievalHit(
                record=RetrievalRecord(
                    jena_id=claim.iri,
                    record_type="claim",
                    text=claim.normalized,
                    embedding=[],
                    entities=claim.entities,
                    tags=claim.tags,
                    unknown_tags=claim.unknown_tags,
                    version=claim.record_version,
                    verification_status=claim.verification_status.value,
                    scientific_status=claim.scientific_status.value,
                    graph=GRAPH_ACCEPTED,
                ),
                score=0.5,
                channel="graph",
            )
        )
    return hits
