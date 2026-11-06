"""Publish committed Jena changes into the retrieval index."""

from __future__ import annotations

from fao_impact_monitor.athena.context import AthenaContext
from fao_impact_monitor.athena.model.records import (
    ClaimRecord,
    ConceptRecord,
    TypeProfile,
)
from fao_impact_monitor.athena.model.status import VerificationStatus
from fao_impact_monitor.athena.ns import GRAPH_ACCEPTED, GRAPH_OPERATIONS, term
from fao_impact_monitor.athena.storage.graph import GraphStore
from fao_impact_monitor.athena.storage.index import RetrievalRecord
from fao_impact_monitor.athena.storage.rdf import (
    n3_iri,
    n3_str,
    read_graph_of,
    read_model,
)


async def publish_outbox(ctx: AthenaContext, event_iri: str) -> None:
    rows = ctx.graph.query(
        f"""
        SELECT ?target ?version ?status WHERE {{
          GRAPH {n3_iri(GRAPH_OPERATIONS)} {{
            {n3_iri(event_iri)} {n3_iri(term("target"))} ?target ;
                {n3_iri(term("recordVersion"))} ?version ;
                {n3_iri(term("status"))} ?status .
          }}
        }}
        """
    )
    if not rows:
        return
    row = rows[0]
    if row["status"] == "sent":
        return
    target = row["target"]
    await _sync_target(ctx, target)
    ctx.graph.update(
        f"""
        DELETE {{
          GRAPH {n3_iri(GRAPH_OPERATIONS)} {{
            {n3_iri(event_iri)} {n3_iri(term("status"))} ?status
          }}
        }}
        WHERE {{
          GRAPH {n3_iri(GRAPH_OPERATIONS)} {{
            {n3_iri(event_iri)} {n3_iri(term("status"))} ?status
          }}
        }};
        INSERT DATA {{
          GRAPH {n3_iri(GRAPH_OPERATIONS)} {{
            {n3_iri(event_iri)} {n3_iri(term("status"))} {n3_str("sent")}
          }}
        }}
        """
    )


async def _sync_target(ctx: AthenaContext, target: str) -> None:
    rdf_type = _rdf_type(ctx.graph, target)
    if rdf_type == term("Claim"):
        claim = read_model(ctx.graph, target, ClaimRecord)
        if claim is None:
            await ctx.index.delete(target)
            return
        graph = read_graph_of(ctx.graph, target)
        if (
            graph == GRAPH_ACCEPTED
            and claim.verification_status == VerificationStatus.VERIFIED
        ):
            await ctx.index.upsert(await _claim_record(ctx, claim, graph))
        else:
            await ctx.index.delete(target)
        return
    if rdf_type == term("Concept"):
        concept = read_model(ctx.graph, target, ConceptRecord)
        if concept is not None:
            await ctx.index.upsert(await _concept_record(ctx, concept))
        return
    if rdf_type == term("TypeProfile"):
        profile = read_model(ctx.graph, target, TypeProfile)
        if profile is not None:
            await ctx.index.upsert(await _profile_record(ctx, profile))
        return
    await ctx.index.delete(target)


def _rdf_type(store: GraphStore, iri: str) -> str | None:
    rows = store.query(
        "SELECT ?type WHERE { GRAPH ?g { " + n3_iri(iri) + " a ?type } }"
    )
    for row in rows:
        if row["type"].rsplit("/", 1)[-1] in {"Claim", "Concept", "TypeProfile"}:
            return row["type"]
    return None


async def _claim_record(
    ctx: AthenaContext, claim: ClaimRecord, graph: str
) -> RetrievalRecord:
    text = "\n".join(
        [
            claim.normalized,
            claim.original_wording,
            " ".join(claim.entities),
            " ".join(claim.processes),
        ]
    )
    embedding = (await ctx.embedder.embed([text]))[0]
    return RetrievalRecord(
        jena_id=claim.iri,
        record_type="claim",
        text=text,
        embedding=embedding,
        entities=claim.entities,
        tags=claim.tags,
        unknown_tags=claim.unknown_tags,
        version=claim.record_version,
        verification_status=claim.verification_status.value,
        scientific_status=claim.scientific_status.value,
        graph=graph,
    )


async def _concept_record(
    ctx: AthenaContext, concept: ConceptRecord
) -> RetrievalRecord:
    text = "\n".join(
        [
            concept.label,
            *concept.aliases,
            *concept.mechanisms,
            *concept.representative_claims,
        ]
    )
    embedding = (await ctx.embedder.embed([text]))[0]
    return RetrievalRecord(
        jena_id=concept.iri,
        record_type="concept",
        text=text,
        embedding=embedding,
        entities=[concept.label, *concept.aliases],
        version=concept.record_version,
        verification_status="verified",
        scientific_status="schema",
        graph=GRAPH_ACCEPTED,
    )


async def _profile_record(ctx: AthenaContext, profile: TypeProfile) -> RetrievalRecord:
    embedding = (await ctx.embedder.embed([profile.text]))[0]
    return RetrievalRecord(
        jena_id=profile.iri,
        record_type="type_profile",
        text=profile.text,
        embedding=embedding,
        entities=profile.aliases,
        version=profile.record_version,
        verification_status="verified",
        scientific_status="schema",
        graph=GRAPH_ACCEPTED,
    )
