"""Persist a proposed relation after ontology reuse and verification."""

from __future__ import annotations

from fao_impact_monitor.athena.agents.schemas import NewTypeDraft
from fao_impact_monitor.athena.context import AthenaContext
from fao_impact_monitor.athena.model.records import (
    ClaimRecord,
    Derivation,
    EvidenceRecord,
    RelationAssertionRecord,
)
from fao_impact_monitor.athena.model.status import ScientificStatus, VerificationStatus
from fao_impact_monitor.athena.observe import log_claim
from fao_impact_monitor.athena.pipeline.ontology_ops import activate_type
from fao_impact_monitor.athena.pipeline.verify import (
    assertion_schema_errors,
    citation_errors,
    combine_judgment,
    needs_causality,
    packet_for_assertion,
    record_verification,
)
from fao_impact_monitor.athena.storage.ontology import (
    find_relation,
    relation_definitions,
)
from fao_impact_monitor.athena.storage.rdf import (
    assertion_iri,
    knowledge_graph,
    read_model,
    utc_stamp,
    write_record,
)


async def persist_proposed(
    ctx: AthenaContext,
    *,
    subject: ClaimRecord,
    obj: ClaimRecord,
    relation_name: str,
    evidence_iris: list[str],
    conditions: list[str],
    scientific: ScientificStatus,
    tentative: bool,
    study_design: str | None,
    source_text: str,
    new_type: NewTypeDraft | None,
    derivation: Derivation | None = None,
) -> RelationAssertionRecord | None:
    relation = find_relation(ctx.graph, relation_name)
    if relation is None and new_type is not None:
        activated = await activate_type(ctx, new_type)
        relation = None if activated is None else find_relation(ctx.graph, activated)
    if relation is None:
        relation = find_relation(ctx.graph, relation_name)
    if relation is None:
        return None
    if derivation is not None:
        scientific = ScientificStatus.DERIVED
    assertion = RelationAssertionRecord(
        iri=assertion_iri(
            subject.iri, relation.iri, obj.iri, evidence_iris, conditions
        ),
        subject_iri=subject.iri,
        object_iri=obj.iri,
        relation_iri=relation.iri,
        conditions=conditions,
        evidence_iris=evidence_iris,
        derivation=derivation,
        scientific_status=scientific,
        verification_status=VerificationStatus.PENDING,
        tags=_shared_tags(subject.tags, obj.tags),
        unknown_tags=sorted(set(subject.unknown_tags) | set(obj.unknown_tags)),
        tentative=tentative,
        study_design=study_design,
    )
    errors: list[str] = []
    excerpts: list[str] = []
    for evidence_id in evidence_iris:
        evidence = read_model(ctx.graph, evidence_id, EvidenceRecord)
        if evidence is None:
            errors.append("missing evidence")
            continue
        errors.extend(citation_errors(evidence, source_text, evidence.source_version))
        excerpts.append(evidence.excerpt)
    definitions = relation_definitions(ctx.graph)
    errors.extend(
        assertion_schema_errors(
            assertion,
            definitions.get(relation.iri),
            subject_type=subject.concept_type,
            object_type=obj.concept_type,
        )
    )
    if errors:
        status = VerificationStatus.REJECTED
        justification = "; ".join(errors)
        model = "deterministic"
    else:
        judgment = await ctx.verifier.judge(
            packet_for_assertion(
                assertion, "\n".join(excerpts), subject.original_wording
            )
        )
        status = combine_judgment(
            [],
            judgment,
            check_relation=True,
            derived=assertion.scientific_status == ScientificStatus.DERIVED,
            needs_causality=needs_causality(
                ctx.graph,
                assertion.relation_iri,
                assertion.scientific_status,
            ),
        )
        justification = judgment.justification
        model = ctx.config.verifier_model
        assertion = assertion.model_copy(
            update={
                "tentative": judgment.tentative,
                "study_design": judgment.study_design,
            }
        )
    assertion = assertion.model_copy(update={"verification_status": status})
    write_record(
        ctx.graph,
        assertion,
        knowledge_graph(assertion.scientific_status, assertion.verification_status),
        index=False,
    )
    record_verification(
        ctx.graph,
        ctx.decisions,
        target_iri=assertion.iri,
        status=status,
        checks=["citation", "schema", "relation-support", "context", "derivation"],
        justification=justification,
        verifier="verification",
        model_version=model,
    )
    log_claim(
        ctx.decisions,
        claim_id=subject.iri,
        stage="relation",
        status=status.value,
        detail=assertion.iri,
        timestamp=utc_stamp(),
    )
    return assertion


def _shared_tags(
    left: dict[str, list[str]],
    right: dict[str, list[str]],
) -> dict[str, list[str]]:
    shared: dict[str, list[str]] = {}
    for field in sorted(set(left) | set(right)):
        values = sorted(set(left.get(field, [])) & set(right.get(field, [])))
        if not values:
            values = sorted(set(left.get(field, [])) | set(right.get(field, [])))
        if values:
            shared[field] = values
    return shared
