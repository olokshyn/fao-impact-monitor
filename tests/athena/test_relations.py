from __future__ import annotations

from pathlib import Path

from fao_impact_monitor.athena.model.filters import FilterExpr, FilterSpec
from fao_impact_monitor.athena.model.records import (
    ClaimRecord,
    EvidenceRecord,
    RelationAssertionRecord,
)
from fao_impact_monitor.athena.model.status import ScientificStatus, VerificationStatus
from fao_impact_monitor.athena.ns import GRAPH_ACCEPTED, relation
from fao_impact_monitor.athena.pipeline.load import load_claims, load_evidence
from fao_impact_monitor.athena.retrieval.relations import query_relations
from fao_impact_monitor.athena.storage.ontology import LocationHierarchy
from fao_impact_monitor.athena.storage.rdf import write_record
from tests.athena.support import make_context


def _claim(iri: str, evidence: str, country: str, source_type: str) -> ClaimRecord:
    return ClaimRecord(
        iri=iri,
        original_wording=iri,
        normalized=iri,
        evidence_iris=[evidence],
        concept_iri="https://fao.org/athena/concept/yield",
        concept_label="yield",
        concept_type="Phenomenon",
        scientific_status=ScientificStatus.SOURCE_REPORTED,
        verification_status=VerificationStatus.VERIFIED,
        tags={"country": [country], "source_type": [source_type]},
    )


def _evidence(iri: str, source_type: str, country: str) -> EvidenceRecord:
    return EvidenceRecord(
        iri=iri,
        excerpt=iri,
        source_iri="https://fao.org/athena/source/s",
        source_version="1",
        tags={"country": [country], "source_type": [source_type]},
    )


def test_descendant_directional_and_symmetric_queries(tmp_path: Path) -> None:
    ctx = make_context(tmp_path)
    journal = _evidence(
        "https://fao.org/athena/evidence/journal", "peer-reviewed", "KEN"
    )
    web = _evidence("https://fao.org/athena/evidence/web", "web-page", "KEN")
    loss = _claim(
        "https://fao.org/athena/claim/loss", journal.iri, "KEN", "peer-reviewed"
    )
    drought = _claim(
        "https://fao.org/athena/claim/drought", journal.iri, "KEN", "peer-reviewed"
    )
    other = _claim("https://fao.org/athena/claim/other", web.iri, "KEN", "web-page")
    for record in (journal, web, loss, drought, other):
        write_record(ctx.graph, record, GRAPH_ACCEPTED, index=False)
    causes = RelationAssertionRecord(
        iri="https://fao.org/athena/assertion/yield",
        subject_iri=drought.iri,
        object_iri=loss.iri,
        relation_iri=relation("yieldLoss"),
        evidence_iris=[journal.iri, web.iri],
        scientific_status=ScientificStatus.SOURCE_REPORTED,
        verification_status=VerificationStatus.VERIFIED,
        tags={"country": ["KEN"]},
    )
    contradiction = RelationAssertionRecord(
        iri="https://fao.org/athena/assertion/contra",
        subject_iri=drought.iri,
        object_iri=other.iri,
        relation_iri=relation("contradicts"),
        evidence_iris=[journal.iri],
        scientific_status=ScientificStatus.SOURCE_REPORTED,
        verification_status=VerificationStatus.VERIFIED,
        tags={"country": ["KEN"]},
    )
    write_record(ctx.graph, causes, GRAPH_ACCEPTED, index=False)
    write_record(ctx.graph, contradiction, GRAPH_ACCEPTED, index=False)
    ancestors = LocationHierarchy.from_graph(ctx.graph).ancestors
    claims = load_claims(ctx.graph)
    evidence = load_evidence(ctx.graph)
    outgoing = query_relations(
        ctx.graph,
        drought.iri,
        "causes",
        "outgoing",
        FilterSpec(),
        ancestors,
        claims,
        evidence,
    )
    assert [view.assertion.iri for view in outgoing] == [causes.iri]
    incoming = query_relations(
        ctx.graph,
        loss.iri,
        "causes",
        "incoming",
        FilterSpec(),
        ancestors,
        claims,
        evidence,
    )
    assert [view.assertion.iri for view in incoming] == [causes.iri]
    mirrored = query_relations(
        ctx.graph,
        other.iri,
        "contradicts",
        "both",
        FilterSpec(),
        ancestors,
        claims,
        evidence,
    )
    assert [view.assertion.iri for view in mirrored] == [contradiction.iri]
    not_outgoing = query_relations(
        ctx.graph,
        loss.iri,
        "causes",
        "outgoing",
        FilterSpec(),
        ancestors,
        claims,
        evidence,
    )
    assert not_outgoing == []


def test_web_evidence_can_be_removed_while_journal_support_remains(
    tmp_path: Path,
) -> None:
    ctx = make_context(tmp_path)
    journal = _evidence(
        "https://fao.org/athena/evidence/journal", "peer-reviewed", "KEN"
    )
    web = _evidence("https://fao.org/athena/evidence/web", "web-page", "KEN")
    left = _claim("https://fao.org/athena/claim/a", journal.iri, "KEN", "peer-reviewed")
    right = _claim(
        "https://fao.org/athena/claim/b", journal.iri, "KEN", "peer-reviewed"
    )
    for record in (journal, web, left, right):
        write_record(ctx.graph, record, GRAPH_ACCEPTED, index=False)
    assertion = RelationAssertionRecord(
        iri="https://fao.org/athena/assertion/mixed",
        subject_iri=left.iri,
        object_iri=right.iri,
        relation_iri=relation("supports"),
        evidence_iris=[journal.iri, web.iri],
        scientific_status=ScientificStatus.SOURCE_REPORTED,
        verification_status=VerificationStatus.VERIFIED,
    )
    write_record(ctx.graph, assertion, GRAPH_ACCEPTED, index=False)
    spec = FilterSpec(
        root=FilterExpr(
            op="tag",
            field="source_type",
            values=["web-page"],
            comparison="exclude",
        )
    )
    views = query_relations(
        ctx.graph,
        left.iri,
        "supports",
        "outgoing",
        spec,
        LocationHierarchy.from_graph(ctx.graph).ancestors,
        load_claims(ctx.graph),
        load_evidence(ctx.graph),
    )
    assert len(views) == 1
    from fao_impact_monitor.athena.retrieval.admissibility import admissible_evidence

    kept = admissible_evidence(
        views[0].assertion.evidence_iris,
        load_evidence(ctx.graph),
        spec,
        lambda _code: set(),
    )
    assert [item.iri for item in kept] == [journal.iri]
