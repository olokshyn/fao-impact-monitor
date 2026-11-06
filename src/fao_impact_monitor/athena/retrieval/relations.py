"""Relation queries. Parent types include descendants. Symmetry is explicit."""

from __future__ import annotations

from collections.abc import Callable

from fao_impact_monitor.athena.model.filters import FilterSpec
from fao_impact_monitor.athena.model.records import (
    ClaimRecord,
    EvidenceRecord,
    RelationAssertionRecord,
)
from fao_impact_monitor.athena.model.status import VerificationStatus
from fao_impact_monitor.athena.ns import (
    GRAPH_ACCEPTED,
    GRAPH_ONTOLOGY,
    RDFS_SUB_PROPERTY,
    relation,
    term,
)
from fao_impact_monitor.athena.retrieval.admissibility import assertion_admission
from fao_impact_monitor.athena.storage.graph import GraphStore
from fao_impact_monitor.athena.storage.ontology import find_relation
from fao_impact_monitor.athena.storage.rdf import n3_iri, n3_str, read_model


class RelationView:
    def __init__(
        self,
        assertion: RelationAssertionRecord,
        *,
        extrapolation: bool,
        direction: str,
    ) -> None:
        self.assertion = assertion
        self.extrapolation = extrapolation
        self.direction = direction


def query_relations(
    store: GraphStore,
    claim_iri: str,
    parent: str,
    direction: str,
    spec: FilterSpec,
    ancestors_of: Callable[[str], set[str]],
    claims: dict[str, ClaimRecord],
    evidence: dict[str, EvidenceRecord],
) -> list[RelationView]:
    definition = find_relation(store, parent)
    if definition is None:
        return []
    subject = n3_iri(claim_iri)
    if definition.directionality == "symmetric" or direction == "both":
        constraint = f"FILTER(?subject = {subject} || ?object = {subject})"
    elif direction == "outgoing":
        constraint = f"FILTER(?subject = {subject})"
    elif direction == "incoming":
        constraint = f"FILTER(?object = {subject})"
    else:
        raise ValueError(f"Unknown direction {direction}")
    rows = store.query(
        f"""
        SELECT ?assertion ?subject ?object WHERE {{
          GRAPH {n3_iri(GRAPH_ACCEPTED)} {{
            ?assertion a {n3_iri(term("RelationAssertion"))} ;
                       {n3_iri(term("subject"))} ?subject ;
                       {n3_iri(term("object"))} ?object ;
                       {n3_iri(term("relation"))} ?relation ;
                       {n3_iri(term("verificationStatus"))} {n3_str(VerificationStatus.VERIFIED.value)} .
            {constraint}
          }}
          GRAPH {n3_iri(GRAPH_ONTOLOGY)} {{
            ?relation {n3_iri(RDFS_SUB_PROPERTY)}* {n3_iri(definition.iri)} .
          }}
        }}
        """
    )
    assertions = {
        iri: record
        for iri, record in (
            (
                row["assertion"],
                read_model(store, row["assertion"], RelationAssertionRecord),
            )
            for row in rows
        )
        if record is not None
    }
    views: list[RelationView] = []
    for row in rows:
        assertion = assertions.get(row["assertion"])
        if assertion is None:
            continue
        decision = assertion_admission(
            assertion,
            claims,
            evidence,
            assertions,
            spec,
            ancestors_of,
        )
        if not decision.allowed:
            continue
        if row["subject"] == claim_iri:
            hop = "outgoing"
        else:
            hop = "incoming"
        views.append(
            RelationView(assertion, extrapolation=decision.extrapolation, direction=hop)
        )
    return views


def causes_iri() -> str:
    return relation("causes")
