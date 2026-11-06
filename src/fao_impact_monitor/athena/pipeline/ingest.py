"""Turn documents and cited raw claims into verified records."""

from __future__ import annotations

from pydantic import BaseModel, Field

from fao_impact_monitor.athena.agents.schemas import (
    ClaimDraft,
    EvidenceDraft,
    ExtractionRequest,
    RelationDraft,
)
from fao_impact_monitor.athena.context import AthenaContext
from fao_impact_monitor.athena.model.records import (
    ClaimRecord,
    ConceptRecord,
    EvidenceRecord,
    ObjectPointer,
    PublicationMetadata,
    SourceRecord,
    TagField,
)
from fao_impact_monitor.athena.model.status import ScientificStatus, VerificationStatus
from fao_impact_monitor.athena.ns import GRAPH_ACCEPTED
from fao_impact_monitor.athena.observe import log_claim
from fao_impact_monitor.athena.pipeline.tags import derive_broader_regions, merge_tags
from fao_impact_monitor.athena.pipeline.verify import (
    citation_errors,
    claim_schema_errors,
    combine_judgment,
    packet_for_claim,
    record_verification,
)
from fao_impact_monitor.athena.storage.graph import ensure_ontology
from fao_impact_monitor.athena.storage.objects import sha256_bytes
from fao_impact_monitor.athena.storage.ontology import LocationHierarchy, tag_registry
from fao_impact_monitor.athena.storage.rdf import (
    claim_iri,
    concept_iri,
    evidence_iri,
    knowledge_graph,
    source_iri,
    utc_stamp,
    write_record,
)


class StoredDocument(BaseModel):
    logical_id: str
    version: str
    sha256: str
    media_type: str
    external_id: str | None = None
    url: str | None = None
    doi: str | None = None
    title: str | None = None
    tags: dict[str, list[str]] = Field(default_factory=dict)
    metadata: dict[str, str] = Field(default_factory=dict)


class RawClaim(BaseModel):
    wording: str
    excerpt: str
    tags: dict[str, list[str]] = Field(default_factory=dict)
    source: StoredDocument | None = None


def text_from_bytes(data: bytes, media_type: str) -> str:
    if media_type == "application/pdf" or data.startswith(b"%PDF"):
        import fitz

        document = fitz.open(stream=data, filetype="pdf")
        try:
            return "\n".join(str(page.get_text()) for page in document)
        finally:
            document.close()
    return data.decode("utf-8", errors="replace")


async def ingest_document(ctx: AthenaContext, document: StoredDocument) -> list[str]:
    ensure_ontology(ctx.graph)
    data = ctx.objects.get(document.logical_id, version=document.version)
    text = text_from_bytes(data, document.media_type)
    source = _source_from_document(document, text)
    write_record(ctx.graph, source, GRAPH_ACCEPTED, index=False)
    extracted = await ctx.extractor.extract(
        ExtractionRequest(
            text=text,
            tags=document.tags,
            metadata=document.metadata,
        )
    )
    return await _materialize(
        ctx, source, text, extracted.evidence, extracted.claims, extracted.relations
    )


async def ingest_raw_claim(ctx: AthenaContext, raw: RawClaim) -> list[str]:
    ensure_ontology(ctx.graph)
    if raw.source is None:
        encoded = raw.excerpt.encode("utf-8")
        ref = ctx.objects.put(
            f"excerpt/{sha256_bytes(encoded)}",
            encoded,
            media_type="text/plain",
        )
        document = StoredDocument(
            logical_id=ref.logical_id,
            version=ref.version,
            sha256=ref.sha256,
            media_type=ref.media_type,
            tags=raw.tags,
        )
        text = raw.excerpt
    else:
        document = raw.source
        text = text_from_bytes(
            ctx.objects.get(document.logical_id, version=document.version),
            document.media_type,
        )
    source = _source_from_document(document, text)
    write_record(ctx.graph, source, GRAPH_ACCEPTED, index=False)
    from fao_impact_monitor.athena.agents.schemas import ClaimDraft, EvidenceDraft

    extracted_evidence = [EvidenceDraft(excerpt=raw.excerpt)]
    extracted_claims = [
        ClaimDraft(
            original_wording=raw.wording,
            normalized=raw.wording,
            applicable_tags=raw.tags,
            evidence_index=0,
            concept_label=raw.wording[:80],
        )
    ]
    return await _materialize(
        ctx, source, text, extracted_evidence, extracted_claims, []
    )


def _source_from_document(document: StoredDocument, text: str) -> SourceRecord:
    content_hash = sha256_bytes(text.encode("utf-8"))
    return SourceRecord(
        iri=source_iri(document.sha256),
        external_id=document.external_id,
        url=document.url,
        doi=document.doi,
        metadata=PublicationMetadata(title=document.title),
        tags=document.tags,
        snapshot=ObjectPointer(
            logical_id=document.logical_id,
            sha256=document.sha256,
            version=document.version,
            media_type=document.media_type,
        ),
        content_hash=content_hash,
        version=document.version,
    )


async def _materialize(
    ctx: AthenaContext,
    source: SourceRecord,
    text: str,
    drafts_evidence: list[EvidenceDraft],
    drafts_claims: list[ClaimDraft],
    drafts_relations: list[RelationDraft],
) -> list[str]:
    registry = tag_registry(ctx.graph)
    hierarchy = LocationHierarchy.from_graph(ctx.graph)
    evidence_records: list[EvidenceRecord] = [
        _evidence_from_draft(draft, source, text) for draft in drafts_evidence
    ]
    claim_ids: list[str] = []
    claims: list[ClaimRecord] = []
    for claim_draft in drafts_claims:
        if claim_draft.evidence_index < 0 or claim_draft.evidence_index >= len(
            evidence_records
        ):
            continue
        evidence = evidence_records[claim_draft.evidence_index]
        claim = _claim_from_draft(claim_draft, evidence, source, registry, hierarchy)
        write_record(ctx.graph, evidence, GRAPH_ACCEPTED, index=False)
        claim = await _verify_claim(
            ctx, claim, evidence, text, source.version, registry
        )
        write_record(
            ctx.graph,
            claim,
            knowledge_graph(claim.scientific_status, claim.verification_status),
            index=True,
        )
        _write_concept(ctx, claim)
        log_claim(
            ctx.decisions,
            claim_id=claim.iri,
            stage="ingest",
            status=claim.verification_status.value,
            detail=claim.normalized,
            timestamp=utc_stamp(),
        )
        claims.append(claim)
        claim_ids.append(claim.iri)
        if claim.verification_status == VerificationStatus.VERIFIED:
            ctx.gate.submit("discover", {"claim_iri": claim.iri})
    for relation_draft in drafts_relations:
        await _persist_explicit(ctx, relation_draft, claims, evidence_records, text)
    return claim_ids


def _evidence_from_draft(
    draft: object, source: SourceRecord, text: str
) -> EvidenceRecord:
    from fao_impact_monitor.athena.agents.schemas import EvidenceDraft

    assert isinstance(draft, EvidenceDraft)
    start = draft.start_offset
    end = draft.end_offset
    if start is None or end is None:
        found = text.find(draft.excerpt)
        if found >= 0:
            start = found
            end = found + len(draft.excerpt)
    locator = "|".join(
        [draft.page or "", draft.section or "", draft.table_cell or "", str(start)]
    )
    return EvidenceRecord(
        iri=evidence_iri(source.content_hash, locator, draft.excerpt),
        excerpt=draft.excerpt,
        source_iri=source.iri,
        source_version=source.version,
        page=draft.page,
        section=draft.section,
        table_cell=draft.table_cell,
        start_offset=start,
        end_offset=end,
        tags=dict(source.tags),
        unknown_tags=list(source.unknown_tags),
    )


def _claim_from_draft(
    draft: object,
    evidence: EvidenceRecord,
    source: SourceRecord,
    registry: dict[str, TagField],
    hierarchy: LocationHierarchy,
) -> ClaimRecord:
    from fao_impact_monitor.athena.agents.schemas import ClaimDraft

    assert isinstance(draft, ClaimDraft)
    tags, unknown = merge_tags(
        registry,
        source.tags,
        draft.applicable_tags,
        draft.unknown_tags,
    )
    tags, unknown, derivations = derive_broader_regions(tags, unknown, hierarchy)
    evidence.tags = tags
    evidence.unknown_tags = unknown
    return ClaimRecord(
        iri=claim_iri(
            draft.normalized,
            [evidence.iri],
            draft.conditions,
            draft.location,
            draft.time_scope,
            draft.population,
        ),
        original_wording=draft.original_wording,
        normalized=draft.normalized,
        entities=draft.entities,
        processes=draft.processes,
        quantities=draft.quantities,
        location=draft.location,
        time_scope=draft.time_scope,
        population=draft.population,
        conditions=draft.conditions,
        tags=tags,
        unknown_tags=unknown,
        tag_derivations=derivations,
        evidence_iris=[evidence.iri],
        concept_iri=concept_iri(draft.concept_label),
        concept_label=draft.concept_label,
        concept_type=draft.concept_type,
        scientific_status=ScientificStatus.SOURCE_REPORTED,
        verification_status=VerificationStatus.PENDING,
    )


async def _verify_claim(
    ctx: AthenaContext,
    claim: ClaimRecord,
    evidence: EvidenceRecord,
    text: str,
    source_version: str,
    registry: dict[str, TagField],
) -> ClaimRecord:
    errors = citation_errors(evidence, text, source_version)
    errors.extend(
        claim_schema_errors(claim, registry, evidence_present=evidence.excerpt in text)
    )
    if errors:
        status = VerificationStatus.REJECTED
        justification = "; ".join(errors)
        model = "deterministic"
    else:
        judgment = await ctx.verifier.judge(packet_for_claim(claim, evidence.excerpt))
        status = combine_judgment(
            [],
            judgment,
            check_relation=False,
            derived=False,
            needs_causality=False,
        )
        justification = judgment.justification
        model = ctx.config.verifier_model
        claim = claim.model_copy(
            update={
                "tentative": judgment.tentative,
                "study_design": judgment.study_design,
            }
        )
    record_verification(
        ctx.graph,
        ctx.decisions,
        target_iri=claim.iri,
        status=status,
        checks=["citation", "schema", "faithful-extraction"],
        justification=justification,
        verifier="verification",
        model_version=model,
    )
    return claim.model_copy(update={"verification_status": status})


def _write_concept(ctx: AthenaContext, claim: ClaimRecord) -> None:
    existing = read_concept(ctx, claim.concept_iri)
    aliases = list(existing.aliases if existing is not None else [])
    representatives = list(
        existing.representative_claims if existing is not None else []
    )
    if claim.iri not in representatives:
        representatives.append(claim.iri)
    record = ConceptRecord(
        iri=claim.concept_iri,
        label=claim.concept_label,
        concept_type=claim.concept_type,
        aliases=aliases,
        representative_claims=representatives,
        mechanisms=list(existing.mechanisms if existing is not None else []),
    )
    write_record(ctx.graph, record, GRAPH_ACCEPTED, index=True)


def read_concept(ctx: AthenaContext, iri: str) -> ConceptRecord | None:
    from fao_impact_monitor.athena.storage.rdf import read_model

    return read_model(ctx.graph, iri, ConceptRecord)


async def _persist_explicit(
    ctx: AthenaContext,
    draft: object,
    claims: list[ClaimRecord],
    evidence_records: list[EvidenceRecord],
    text: str,
) -> None:
    from fao_impact_monitor.athena.agents.schemas import RelationDraft
    from fao_impact_monitor.athena.pipeline.assertions import persist_proposed

    if not isinstance(draft, RelationDraft):
        return
    if draft.subject_index >= len(claims) or draft.object_index >= len(claims):
        return
    evidence_iris = [
        evidence_records[index].iri
        for index in draft.evidence_indexes
        if 0 <= index < len(evidence_records)
    ]
    await persist_proposed(
        ctx,
        subject=claims[draft.subject_index],
        obj=claims[draft.object_index],
        relation_name=draft.relation,
        evidence_iris=evidence_iris or [evidence.iri for evidence in evidence_records],
        conditions=draft.conditions,
        scientific=draft.scientific_status,
        tentative=draft.tentative,
        study_design=draft.study_design,
        source_text=text,
        new_type=None,
    )
