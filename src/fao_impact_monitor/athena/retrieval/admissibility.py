"""Decide whether a claim, evidence item, or relation may participate."""

from __future__ import annotations

from collections.abc import Callable

from fao_impact_monitor.athena.model.filters import (
    Admission,
    FilterSpec,
    admit,
    without_fields,
)
from fao_impact_monitor.athena.model.records import (
    ClaimRecord,
    EvidenceRecord,
    RelationAssertionRecord,
)
from fao_impact_monitor.athena.model.status import (
    ScientificStatus,
    VerificationStatus,
)
from fao_impact_monitor.athena.ns import PROVENANCE_FIELDS


def claim_admission(
    claim: ClaimRecord,
    evidence: dict[str, EvidenceRecord],
    spec: FilterSpec,
    ancestors_of: Callable[[str], set[str]],
) -> Admission:
    """Geography applies to the claim. Provenance applies to each evidence item."""
    if claim.verification_status != VerificationStatus.VERIFIED:
        return Admission(False, False)
    if claim.scientific_status == ScientificStatus.HYPOTHETICAL:
        return Admission(False, False)
    applicability = admit(
        without_fields(spec, set(PROVENANCE_FIELDS)),
        claim.tags,
        claim.unknown_tags,
        ancestors_of,
    )
    if not applicability.allowed:
        return Admission(False, False)
    supporting = [
        admit(spec, item.tags, item.unknown_tags, ancestors_of)
        for iri in claim.evidence_iris
        if (item := evidence.get(iri)) is not None
    ]
    kept = [item for item in supporting if item.allowed]
    if not kept:
        return Admission(False, False)
    return Admission(
        True,
        applicability.extrapolation or any(item.extrapolation for item in kept),
    )


def admissible_evidence(
    evidence_iris: list[str],
    evidence: dict[str, EvidenceRecord],
    spec: FilterSpec,
    ancestors_of: Callable[[str], set[str]],
) -> list[EvidenceRecord]:
    kept: list[EvidenceRecord] = []
    for iri in evidence_iris:
        item = evidence.get(iri)
        if item is None:
            continue
        if admit(spec, item.tags, item.unknown_tags, ancestors_of).allowed:
            kept.append(item)
    return kept


def assertion_admission(
    assertion: RelationAssertionRecord,
    claims: dict[str, ClaimRecord],
    evidence: dict[str, EvidenceRecord],
    assertions: dict[str, RelationAssertionRecord],
    spec: FilterSpec,
    ancestors_of: Callable[[str], set[str]],
    seen: set[str] | None = None,
) -> Admission:
    """A relation is usable only with an admissible evidence set and premises."""
    stack = seen or set()
    if assertion.iri in stack:
        return Admission(False, False)
    if assertion.verification_status != VerificationStatus.VERIFIED:
        return Admission(False, False)
    if assertion.scientific_status == ScientificStatus.HYPOTHETICAL:
        return Admission(False, False)
    endpoint_admissions: list[Admission] = []
    for endpoint in (assertion.subject_iri, assertion.object_iri):
        claim = claims.get(endpoint)
        if claim is None:
            return Admission(False, False)
        decision = claim_admission(claim, evidence, spec, ancestors_of)
        if spec.strict and not decision.allowed:
            return Admission(False, False)
        endpoint_admissions.append(decision)
    if not spec.strict and not any(item.allowed for item in endpoint_admissions):
        return Admission(False, False)
    supporting = admissible_evidence(
        assertion.evidence_iris,
        evidence,
        spec,
        ancestors_of,
    )
    if not supporting:
        return Admission(False, False)
    premise_extrapolation = False
    if assertion.derivation is not None:
        for premise in assertion.derivation.premises:
            premise_assertion = assertions.get(premise)
            if premise_assertion is None:
                return Admission(False, False)
            premise_decision = assertion_admission(
                premise_assertion,
                claims,
                evidence,
                assertions,
                spec,
                ancestors_of,
                stack | {assertion.iri},
            )
            if not premise_decision.allowed:
                return Admission(False, False)
            premise_extrapolation = (
                premise_extrapolation or premise_decision.extrapolation
            )
    extrapolation = (
        any(item.extrapolation for item in endpoint_admissions) or premise_extrapolation
    )
    return Admission(True, extrapolation)
