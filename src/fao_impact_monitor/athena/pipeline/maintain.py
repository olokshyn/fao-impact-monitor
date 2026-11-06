"""Deduplicate assertions, invalidate derivations, and keep disagreements."""

from __future__ import annotations

from collections import defaultdict

from fao_impact_monitor.athena.context import AthenaContext
from fao_impact_monitor.athena.model.records import RelationAssertionRecord
from fao_impact_monitor.athena.model.status import ScientificStatus, VerificationStatus
from fao_impact_monitor.athena.ns import GRAPH_CANDIDATES, GRAPH_OPERATIONS, term
from fao_impact_monitor.athena.pipeline.load import (
    load_assertions,
    load_claims,
    load_evidence,
)
from fao_impact_monitor.athena.pipeline.verify import record_verification
from fao_impact_monitor.athena.storage.rdf import delete_record, n3_iri, write_record


def maintain(
    ctx: AthenaContext, changed_iris: list[str] | None = None
) -> dict[str, list[str]]:
    deduped = _dedup(ctx)
    invalidated = [
        iri for changed in changed_iris or [] for iri in _invalidate(ctx, changed)
    ]
    dropped = _drop_caches(ctx, changed_iris or [])
    unresolved = [
        claim.iri
        for claim in load_claims(ctx.graph).values()
        if claim.verification_status
        in {VerificationStatus.PENDING, VerificationStatus.REVIEW}
    ]
    return {
        "deduped": deduped,
        "invalidated": invalidated,
        "dropped_caches": dropped,
        "unresolved": unresolved,
    }


def _dedup(ctx: AthenaContext) -> list[str]:
    evidence = load_evidence(ctx.graph)
    groups: dict[
        tuple[str, str, str, tuple[str, ...]], list[RelationAssertionRecord]
    ] = defaultdict(list)
    for assertion in load_assertions(ctx.graph).values():
        excerpts = tuple(
            sorted(
                evidence[iri].excerpt
                for iri in assertion.evidence_iris
                if iri in evidence
            )
        )
        key = (
            assertion.subject_iri,
            assertion.relation_iri,
            assertion.object_iri,
            excerpts,
        )
        groups[key].append(assertion)
    removed: list[str] = []
    for group in groups.values():
        if len(group) < 2:
            continue
        group.sort(
            key=lambda item: (
                item.verification_status != VerificationStatus.VERIFIED,
                item.iri,
            )
        )
        for extra in group[1:]:
            delete_record(ctx.graph, extra.iri)
            removed.append(extra.iri)
    return removed


def _invalidate(ctx: AthenaContext, changed: str) -> list[str]:
    removed: list[str] = []
    for assertion in load_assertions(ctx.graph).values():
        derivation = assertion.derivation
        if derivation is None or changed not in derivation.premises:
            continue
        if assertion.scientific_status != ScientificStatus.DERIVED:
            continue
        updated = assertion.model_copy(
            update={"verification_status": VerificationStatus.REJECTED}
        )
        write_record(ctx.graph, updated, GRAPH_CANDIDATES, index=False)
        record_verification(
            ctx.graph,
            ctx.decisions,
            target_iri=assertion.iri,
            status=VerificationStatus.REJECTED,
            checks=["derivation"],
            justification=f"Premise {changed} changed or was retracted.",
            verifier="maintenance",
            model_version="deterministic",
        )
        removed.append(assertion.iri)
    return removed


def _drop_caches(ctx: AthenaContext, changed: list[str]) -> list[str]:
    if not changed:
        return []
    rows = ctx.graph.query(
        f"""
        SELECT ?cache ?dep WHERE {{
          GRAPH {n3_iri(GRAPH_OPERATIONS)} {{
            ?cache a {n3_iri(term("CacheEntry"))} ;
                   {n3_iri(term("dependsOn"))} ?dep .
          }}
        }}
        """
    )
    drop = {row["cache"] for row in rows if row["dep"] in changed}
    for iri in drop:
        delete_record(ctx.graph, iri)
    return sorted(drop)
