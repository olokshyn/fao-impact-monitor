from __future__ import annotations

import asyncio
from pathlib import Path

from fao_impact_monitor.athena.agents.schemas import (
    ClaimDraft,
    EvidenceDraft,
    ExtractionResult,
    NewTypeDraft,
    OntologyDecision,
    RelationDraft,
)
from fao_impact_monitor.athena.model.filters import FilterExpr, FilterSpec
from fao_impact_monitor.athena.model.records import (
    Derivation,
    RelationAssertionRecord,
)
from fao_impact_monitor.athena.model.status import ScientificStatus, VerificationStatus
from fao_impact_monitor.athena.ns import GRAPH_ACCEPTED, relation
from fao_impact_monitor.athena.pipeline.handlers import process_batch
from fao_impact_monitor.athena.pipeline.ingest import StoredDocument, ingest_document
from fao_impact_monitor.athena.pipeline.load import load_assertions, load_claims
from fao_impact_monitor.athena.pipeline.maintain import maintain
from fao_impact_monitor.athena.pipeline.ontology_ops import activate_type
from fao_impact_monitor.athena.pipeline.research import run_research
from fao_impact_monitor.athena.storage.rdf import read_model, write_record
from tests.athena.support import (
    RecordingDiscoverer,
    RecordingExtractor,
    RecordingOntologist,
    RecordingResearcher,
    RecordingVerifier,
    causes_proposal,
    decision_records,
    make_context,
    published_batches,
)


def test_ingestion_verifies_citation_and_logs_the_claim(tmp_path: Path) -> None:
    text = "Drought in Kenya decreased maize yield."
    ctx = make_context(
        tmp_path,
        extractor=RecordingExtractor(
            ExtractionResult(
                evidence=[EvidenceDraft(excerpt=text)],
                claims=[
                    ClaimDraft(
                        original_wording=text,
                        normalized="Drought decreased maize yield in Kenya.",
                        entities=["drought", "maize"],
                        applicable_tags={"country": ["KEN"]},
                        mentioned_places=["Ethiopia"],
                        evidence_index=0,
                        concept_label="maize yield",
                    )
                ],
            )
        ),
    )
    ref = ctx.objects.put("upload/doc", text.encode(), media_type="text/plain")
    ids = asyncio.run(
        ingest_document(
            ctx,
            StoredDocument(
                logical_id=ref.logical_id,
                version=ref.version,
                sha256=ref.sha256,
                media_type=ref.media_type,
                tags={
                    "source_type": ["peer-reviewed"],
                    "acquisition_channel": ["web-search"],
                },
            ),
        )
    )
    assert ids
    claim = read_model(ctx.graph, ids[0], load_claims(ctx.graph)[ids[0]].__class__)
    assert claim is not None
    assert claim.verification_status == VerificationStatus.VERIFIED
    assert claim.tags["country"] == ["KEN"]
    assert "eastern-africa" in claim.tags["broader_region"]
    assert claim.tag_derivations
    assert "Ethiopia" not in claim.tags.get("country", [])
    assert claim.tags["source_type"] == ["peer-reviewed"]
    assert claim.tags["acquisition_channel"] == ["web-search"]
    assert any(record.claim_ids == [claim.iri] for record in decision_records(ctx))
    again = asyncio.run(
        ingest_document(
            ctx,
            StoredDocument(
                logical_id=ref.logical_id,
                version=ref.version,
                sha256=ref.sha256,
                media_type=ref.media_type,
            ),
        )
    )
    assert again == ids


def test_correlation_cannot_pass_as_source_reported_cause(tmp_path: Path) -> None:
    text = "Rainfall correlated with yield."
    ctx = make_context(tmp_path, verifier=RecordingVerifier(causality=False))
    ctx.extractor = RecordingExtractor(
        ExtractionResult(
            evidence=[EvidenceDraft(excerpt=text)],
            claims=[
                ClaimDraft(
                    original_wording=text,
                    normalized=text,
                    evidence_index=0,
                    concept_label="yield",
                    applicable_tags={"country": ["KEN"]},
                )
            ],
            relations=[
                RelationDraft(
                    subject_index=0,
                    object_index=0,
                    relation="causes",
                    evidence_indexes=[0],
                )
            ],
        )
    )
    ref = ctx.objects.put("upload/corr", text.encode(), media_type="text/plain")
    asyncio.run(
        ingest_document(
            ctx,
            StoredDocument(
                logical_id=ref.logical_id,
                version=ref.version,
                sha256=ref.sha256,
                media_type=ref.media_type,
            ),
        )
    )
    assertions = list(load_assertions(ctx.graph).values())
    assert assertions
    assert assertions[0].verification_status == VerificationStatus.REJECTED
    assert assertions[0].relation_iri == relation("causes")


def test_new_type_does_not_create_an_assertion(tmp_path: Path) -> None:
    ctx = make_context(
        tmp_path,
        ontologist=RecordingOntologist(
            OntologyDecision(
                definition="A decrease in maize yield per hectare.",
                parent="decreases",
                subject_type="Exposure",
                object_type="AgriculturalVariable",
                directionality="directional",
                examples=["Drought caused maize yield loss."],
            )
        ),
    )
    activated = asyncio.run(
        activate_type(
            ctx,
            NewTypeDraft(
                name="maize yield loss",
                definition="A decrease in maize yield per hectare.",
                parent="decreases",
                subject_type="Exposure",
                object_type="AgriculturalVariable",
                directionality="directional",
                examples=["Drought caused maize yield loss."],
            ),
        )
    )
    assert activated is not None
    assert load_assertions(ctx.graph) == {}
    assert any(
        task_type == "backfill" for task_type, _batch_id in published_batches(ctx)
    )


def test_maintenance_drops_duplicate_and_derived_links_only(tmp_path: Path) -> None:
    ctx = make_context(tmp_path)
    evidence = "https://fao.org/athena/evidence/e"
    from fao_impact_monitor.athena.model.records import EvidenceRecord

    write_record(
        ctx.graph,
        EvidenceRecord(
            iri=evidence,
            excerpt="same",
            source_iri="https://fao.org/athena/source/s",
            source_version="1",
        ),
        GRAPH_ACCEPTED,
        index=False,
    )
    first = RelationAssertionRecord(
        iri="https://fao.org/athena/assertion/1",
        subject_iri="https://fao.org/athena/claim/a",
        object_iri="https://fao.org/athena/claim/b",
        relation_iri=relation("causes"),
        evidence_iris=[evidence],
        scientific_status=ScientificStatus.SOURCE_REPORTED,
        verification_status=VerificationStatus.VERIFIED,
    )
    duplicate = first.model_copy(update={"iri": "https://fao.org/athena/assertion/2"})
    kept = RelationAssertionRecord(
        iri="https://fao.org/athena/assertion/contra",
        subject_iri="https://fao.org/athena/claim/a",
        object_iri="https://fao.org/athena/claim/c",
        relation_iri=relation("contradicts"),
        evidence_iris=[evidence],
        scientific_status=ScientificStatus.SOURCE_REPORTED,
        verification_status=VerificationStatus.VERIFIED,
    )
    derived = RelationAssertionRecord(
        iri="https://fao.org/athena/assertion/derived",
        subject_iri="https://fao.org/athena/claim/a",
        object_iri="https://fao.org/athena/claim/d",
        relation_iri=relation("causes"),
        evidence_iris=[evidence],
        scientific_status=ScientificStatus.DERIVED,
        verification_status=VerificationStatus.VERIFIED,
        derivation=Derivation(
            premises=["https://fao.org/athena/claim/a"], rule="chain"
        ),
    )
    for record in (first, duplicate, kept, derived):
        write_record(ctx.graph, record, GRAPH_ACCEPTED, index=False)
    report = maintain(ctx, ["https://fao.org/athena/claim/a"])
    remaining = set(load_assertions(ctx.graph))
    assert duplicate.iri in report["deduped"] or first.iri in report["deduped"]
    assert kept.iri in remaining
    assert derived.iri in report["invalidated"]
    invalidated = read_model(ctx.graph, derived.iri, RelationAssertionRecord)
    assert invalidated is not None
    assert invalidated.verification_status == VerificationStatus.REJECTED


def test_research_hides_excluded_evidence_and_stays_qualitative(tmp_path: Path) -> None:
    text = "Kenyan field soils lost soil biota after microplastic exposure."
    researcher = RecordingResearcher()
    ctx = make_context(tmp_path, researcher=researcher)
    ref = ctx.objects.put("upload/kenya", text.encode(), media_type="text/plain")
    asyncio.run(
        ingest_document(
            ctx,
            StoredDocument(
                logical_id=ref.logical_id,
                version=ref.version,
                sha256=ref.sha256,
                media_type=ref.media_type,
                tags={"country": ["KEN"], "source_type": ["peer-reviewed"]},
            ),
        )
    )
    for task_type, batch_id in list(published_batches(ctx)):
        if task_type == "index_sync":
            asyncio.run(process_batch(ctx, batch_id))
    web = "A web page said soil biota increased everywhere."
    web_ref = ctx.objects.put("upload/web", web.encode(), media_type="text/plain")
    ctx.extractor = RecordingExtractor(
        ExtractionResult(
            evidence=[EvidenceDraft(excerpt=web)],
            claims=[
                ClaimDraft(
                    original_wording=web,
                    normalized=web,
                    evidence_index=0,
                    concept_label="soil biota",
                    applicable_tags={"country": ["KEN"]},
                )
            ],
        )
    )
    asyncio.run(
        ingest_document(
            ctx,
            StoredDocument(
                logical_id=web_ref.logical_id,
                version=web_ref.version,
                sha256=web_ref.sha256,
                media_type=web_ref.media_type,
                tags={"country": ["KEN"], "source_type": ["web-page"]},
            ),
        )
    )
    for task_type, batch_id in list(published_batches(ctx)):
        if task_type == "index_sync":
            asyncio.run(process_batch(ctx, batch_id))
    result = asyncio.run(
        run_research(
            ctx,
            "How do microplastics affect soil biota in Kenya?",
            FilterSpec(
                root=FilterExpr(
                    op="tag",
                    field="source_type",
                    values=["web-page"],
                    comparison="exclude",
                )
            ),
            rounds=1,
            external=True,
        )
    )
    assert result.strict.qualitative is True
    assert result.strict.quantitative_model is None
    assert any("disabled" in gap for gap in result.strict.gaps)
    assert researcher.packs
    assert web not in researcher.packs[0]
    assert "soil biota" in researcher.packs[0]


def test_discover_uses_the_same_verifier(tmp_path: Path) -> None:
    text = "Drought decreased maize yield."
    ctx = make_context(tmp_path, discoverer=RecordingDiscoverer())
    ref = ctx.objects.put("upload/a", text.encode(), media_type="text/plain")
    first = asyncio.run(
        ingest_document(
            ctx,
            StoredDocument(
                logical_id=ref.logical_id,
                version=ref.version,
                sha256=ref.sha256,
                media_type=ref.media_type,
                tags={"country": ["KEN"], "source_type": ["peer-reviewed"]},
            ),
        )
    )
    other_text = "Maize yield loss reduced food availability."
    ctx.extractor = RecordingExtractor(
        ExtractionResult(
            evidence=[EvidenceDraft(excerpt=other_text)],
            claims=[
                ClaimDraft(
                    original_wording=other_text,
                    normalized=other_text,
                    entities=["maize"],
                    evidence_index=0,
                    concept_label="food availability",
                    applicable_tags={"country": ["KEN"]},
                )
            ],
        )
    )
    other_ref = ctx.objects.put(
        "upload/b", other_text.encode(), media_type="text/plain"
    )
    second = asyncio.run(
        ingest_document(
            ctx,
            StoredDocument(
                logical_id=other_ref.logical_id,
                version=other_ref.version,
                sha256=other_ref.sha256,
                media_type=other_ref.media_type,
                tags={"country": ["KEN"], "source_type": ["peer-reviewed"]},
            ),
        )
    )
    evidence = load_claims(ctx.graph)[first[0]].evidence_iris[0]
    ctx.discoverer = RecordingDiscoverer(
        [causes_proposal(first[0], second[0], evidence)]
    )
    published = list(published_batches(ctx))
    for task_type, batch_id in published:
        if task_type == "index_sync":
            asyncio.run(process_batch(ctx, batch_id))
    for task_type, batch_id in published:
        if task_type == "discover":
            asyncio.run(process_batch(ctx, batch_id))
    assert any(
        item.relation_iri == relation("causes")
        and item.verification_status == VerificationStatus.VERIFIED
        for item in load_assertions(ctx.graph).values()
    )
