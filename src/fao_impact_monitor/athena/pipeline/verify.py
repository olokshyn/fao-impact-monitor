"""Deterministic citation and schema checks, plus the evidence-reading verdict."""

from __future__ import annotations

from uuid import uuid4

from fao_impact_monitor.athena.agents.schemas import (
    VerificationJudgment,
    VerificationPacket,
)
from fao_impact_monitor.athena.model.records import (
    ClaimRecord,
    EvidenceRecord,
    RelationAssertionRecord,
    RelationDefinition,
    TagField,
    VerificationRecord,
)
from fao_impact_monitor.athena.model.status import (
    ScientificStatus,
    VerificationStatus,
)
from fao_impact_monitor.athena.ns import relation, term
from fao_impact_monitor.athena.observe import DecisionLog, DecisionRecord, input_hash
from fao_impact_monitor.athena.storage.graph import GraphStore
from fao_impact_monitor.athena.storage.ontology import is_descendant
from fao_impact_monitor.athena.storage.rdf import utc_stamp, write_verification


def citation_errors(
    evidence: EvidenceRecord, source_text: str, source_version: str
) -> list[str]:
    errors: list[str] = []
    if evidence.source_version != source_version:
        errors.append("source version mismatch")
    if evidence.excerpt not in source_text:
        errors.append("excerpt not in stored source")
    if evidence.start_offset is not None and evidence.end_offset is not None:
        snippet = source_text[evidence.start_offset : evidence.end_offset]
        if evidence.excerpt != snippet and evidence.excerpt not in snippet:
            errors.append("offsets do not contain the excerpt")
    return errors


def tag_errors(tags: dict[str, list[str]], registry: dict[str, TagField]) -> list[str]:
    errors: list[str] = []
    for field, values in tags.items():
        spec = registry.get(field)
        if spec is None:
            errors.append(f"unknown tag field {field}")
            continue
        if spec.multiplicity == "one" and len(values) > 1:
            errors.append(f"{field} allows one value")
        if spec.vocabulary:
            unknown_values = [value for value in values if value not in spec.vocabulary]
            if unknown_values:
                errors.append(
                    f"{field} value not in vocabulary: {', '.join(unknown_values)}"
                )
    return errors


def claim_schema_errors(
    claim: ClaimRecord,
    registry: dict[str, TagField],
    *,
    evidence_present: bool,
) -> list[str]:
    errors = tag_errors(claim.tags, registry)
    if not claim.evidence_iris or not evidence_present:
        errors.append("claim has no stored evidence")
    if not claim.normalized.strip():
        errors.append("claim has no normalized proposition")
    return errors


def assertion_schema_errors(
    assertion: RelationAssertionRecord,
    definition: RelationDefinition | None,
    *,
    subject_type: str | None,
    object_type: str | None,
) -> list[str]:
    errors: list[str] = []
    if definition is None:
        errors.append("relation type is not in the ontology")
        return errors
    if not assertion.evidence_iris:
        errors.append("relation has no evidence")
    if assertion.scientific_status == ScientificStatus.DERIVED:
        derivation = assertion.derivation
        if derivation is None or not derivation.premises or not derivation.rule:
            errors.append("derived relation is missing premises or a rule")
        elif derivation.assumptions is None:
            errors.append("derived relation is missing assumptions")
    if not _endpoint_ok(subject_type, definition.subject_type):
        errors.append("subject endpoint type is not permitted")
    if not _endpoint_ok(object_type, definition.object_type):
        errors.append("object endpoint type is not permitted")
    return errors


def _endpoint_ok(actual: str | None, required: str) -> bool:
    if required.endswith("/Phenomenon"):
        return True
    if actual is None:
        return False
    return actual == required or required.endswith("/" + actual)


def combine_judgment(
    deterministic_errors: list[str],
    judgment: VerificationJudgment,
    *,
    check_relation: bool,
    derived: bool,
    needs_causality: bool,
) -> VerificationStatus:
    if deterministic_errors:
        return VerificationStatus.REJECTED
    flags: list[bool | None] = [judgment.faithful]
    if check_relation:
        flags.extend([judgment.relation_supported, judgment.context_compatible])
    if derived:
        flags.append(judgment.derivation_valid)
    if needs_causality:
        flags.append(judgment.expresses_causality)
    if any(flag is False for flag in flags):
        return VerificationStatus.REJECTED
    if any(flag is None for flag in flags):
        return VerificationStatus.REVIEW
    return VerificationStatus.VERIFIED


def needs_causality(
    store: GraphStore, relation_iri: str, scientific: ScientificStatus
) -> bool:
    if scientific != ScientificStatus.SOURCE_REPORTED:
        return False
    return is_descendant(store, relation_iri, relation("causes"))


def record_verification(
    store: GraphStore,
    decisions: DecisionLog,
    *,
    target_iri: str,
    status: VerificationStatus,
    checks: list[str],
    justification: str,
    verifier: str,
    model_version: str,
) -> VerificationRecord:
    record = VerificationRecord(
        iri=term(f"verification/{uuid4().hex}"),
        target_iri=target_iri,
        checks=checks,
        result=status,
        justification=justification,
        verifier=verifier,
        model_version=model_version,
        timestamp=utc_stamp(),
    )
    write_verification(store, record)
    decisions.write(
        DecisionRecord(
            agent=verifier,
            model=model_version,
            stage="verify",
            input_hash=input_hash(target_iri + justification),
            output={"result": status.value, "checks": checks},
            claim_ids=[target_iri],
            timestamp=record.timestamp,
        )
    )
    return record


def packet_for_claim(claim: ClaimRecord, excerpt: str) -> VerificationPacket:
    return VerificationPacket(
        excerpt=excerpt,
        claim_wording=claim.original_wording,
        normalized=claim.normalized,
        quantities=claim.quantities,
        conditions=claim.conditions,
    )


def packet_for_assertion(
    assertion: RelationAssertionRecord,
    excerpt: str,
    wording: str,
) -> VerificationPacket:
    derivation = assertion.derivation
    return VerificationPacket(
        excerpt=excerpt,
        claim_wording=wording,
        normalized=wording,
        conditions=assertion.conditions,
        relation=assertion.relation_iri,
        scientific_status=assertion.scientific_status.value,
        derivation_rule=None if derivation is None else derivation.rule,
        premises=[] if derivation is None else derivation.premises,
        assumptions=[] if derivation is None else derivation.assumptions,
    )
