"""Map knowledge records to SPARQL updates and read them back."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel
from rdflib import Literal, URIRef

from fao_impact_monitor.athena.model.records import (
    ClaimRecord,
    ConceptRecord,
    EvidenceRecord,
    RelationAssertionRecord,
    RelationDefinition,
    SourceRecord,
    TypeProfile,
    VerificationRecord,
)
from fao_impact_monitor.athena.model.status import (
    ScientificStatus,
    VerificationStatus,
)
from fao_impact_monitor.athena.ns import (
    GRAPH_ACCEPTED,
    GRAPH_CANDIDATES,
    GRAPH_ONTOLOGY,
    GRAPH_OPERATIONS,
    OWL_OBJECT_PROPERTY,
    RDFS_LABEL,
    RDFS_SUB_PROPERTY,
    tag_predicate,
    term,
)
from fao_impact_monitor.athena.storage.graph import GraphStore

KnowledgeRecord = (
    ClaimRecord
    | RelationAssertionRecord
    | ConceptRecord
    | TypeProfile
    | SourceRecord
    | EvidenceRecord
)


def digest(*parts: str) -> str:
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def source_iri(content_hash: str) -> str:
    return term(f"source/{content_hash}")


def evidence_iri(source_hash: str, locator: str, excerpt: str) -> str:
    return term(f"evidence/{digest(source_hash, locator, excerpt)}")


def claim_iri(
    normalized: str,
    evidence_ids: list[str],
    conditions: list[str],
    location: str | None,
    time_scope: str | None,
    population: str | None,
) -> str:
    return term(
        "claim/"
        + digest(
            normalized,
            "|".join(sorted(evidence_ids)),
            "|".join(conditions),
            location or "",
            time_scope or "",
            population or "",
        )
    )


def assertion_iri(
    subject: str,
    relation: str,
    obj: str,
    evidence_ids: list[str],
    conditions: list[str],
) -> str:
    return term(
        "assertion/"
        + digest(
            subject,
            relation,
            obj,
            "|".join(sorted(evidence_ids)),
            "|".join(conditions),
        )
    )


def concept_iri(label: str) -> str:
    slug = "-".join(
        part
        for part in "".join(
            char.lower() if char.isalnum() else " " for char in label
        ).split()
    )
    return term(f"concept/{slug or 'concept'}")


def assign_version(model: BaseModel) -> str:
    data = model.model_dump(mode="json")
    data.pop("record_version", None)
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return digest(encoded)


def n3_iri(value: str) -> str:
    return URIRef(value).n3()


def n3_str(value: str) -> str:
    return Literal(value).n3()


def knowledge_graph(
    scientific: ScientificStatus,
    verification: VerificationStatus,
) -> str:
    if scientific == ScientificStatus.HYPOTHETICAL:
        return GRAPH_CANDIDATES
    if verification == VerificationStatus.VERIFIED:
        return GRAPH_ACCEPTED
    return GRAPH_CANDIDATES


def utc_stamp() -> str:
    return datetime.now(UTC).isoformat()


def read_model[T: BaseModel](store: GraphStore, iri: str, model: type[T]) -> T | None:
    query = (
        "SELECT ?json WHERE { GRAPH ?g { "
        + n3_iri(iri)
        + " "
        + n3_iri(term("recordJson"))
        + " ?json } }"
    )
    rows = store.query(query)
    if not rows:
        return None
    return model.model_validate_json(rows[0]["json"])


def read_graph_of(store: GraphStore, iri: str) -> str | None:
    rows = store.query(
        "SELECT ?g WHERE { GRAPH ?g { "
        + n3_iri(iri)
        + " "
        + n3_iri(term("recordJson"))
        + " ?json } }"
    )
    if not rows:
        return None
    return rows[0]["g"]


def tag_lines(
    subject: str, tags: dict[str, list[str]], unknown: list[str]
) -> list[str]:
    lines: list[str] = []
    node = n3_iri(subject)
    for field, values in sorted(tags.items()):
        predicate = n3_iri(tag_predicate(field))
        for value in values:
            lines.append(f"{node} {predicate} {n3_str(value)} .")
    unknown_predicate = n3_iri(term("unknownTag"))
    for field in unknown:
        lines.append(f"{node} {unknown_predicate} {n3_str(field)} .")
    return lines


def _typed_json(iri: str, rdf_type: str, record: BaseModel) -> tuple[str, BaseModel]:
    version = assign_version(record)
    stamped = record.model_copy(update={"record_version": version})
    lines = [
        f"{n3_iri(iri)} a {n3_iri(rdf_type)} .",
        f"{n3_iri(iri)} {n3_iri(term('recordJson'))} {n3_str(stamped.model_dump_json())} .",
        f"{n3_iri(iri)} {n3_iri(term('recordVersion'))} {n3_str(version)} .",
    ]
    return "\n".join(lines), stamped


def write_record(
    store: GraphStore, record: KnowledgeRecord, graph_iri: str, *, index: bool
) -> str:
    """Upsert a record. Returns the outbox event IRI when indexing is requested."""
    body = _body(record)
    version = assign_version(record)
    operations = ""
    if index:
        event = term(f"outbox/{uuid4().hex}")
        work = term(f"work/{uuid4().hex}")
        now = utc_stamp()
        operations = f"""
        GRAPH {n3_iri(GRAPH_OPERATIONS)} {{
        {n3_iri(event)} a {n3_iri(term("OutboxEvent"))} ;
            {n3_iri(term("target"))} {n3_iri(record.iri)} ;
            {n3_iri(term("op"))} {n3_str("upsert")} ;
            {n3_iri(term("recordVersion"))} {n3_str(version)} ;
            {n3_iri(term("status"))} {n3_str("pending")} ;
            {n3_iri(term("createdAt"))} {n3_str(now)} .
        {n3_iri(work)} a {n3_iri(term("PendingWork"))} ;
            {n3_iri(term("taskType"))} {n3_str("index_sync")} ;
            {n3_iri(term("outbox"))} {n3_iri(event)} ;
            {n3_iri(term("enqueuedAt"))} {n3_str(now)} .
        }}
        """
    else:
        event = ""
    store.update(
        f"""
        DELETE {{ GRAPH ?g {{ {n3_iri(record.iri)} ?p ?o }} }}
        WHERE {{
          VALUES ?g {{
            {n3_iri(GRAPH_ACCEPTED)} {n3_iri(GRAPH_CANDIDATES)} {n3_iri(GRAPH_ONTOLOGY)}
          }}
          GRAPH ?g {{ {n3_iri(record.iri)} ?p ?o }}
        }};
        INSERT DATA {{
          GRAPH {n3_iri(graph_iri)} {{
            {body}
          }}
          {operations}
        }}
        """
    )
    return event if index else ""


def _body(record: KnowledgeRecord) -> str:
    if isinstance(record, ClaimRecord):
        return _claim_body(record)
    if isinstance(record, RelationAssertionRecord):
        return _assertion_body(record)
    if isinstance(record, ConceptRecord):
        return _concept_body(record)
    if isinstance(record, TypeProfile):
        return _profile_body(record)
    if isinstance(record, SourceRecord):
        return _source_body(record)
    return _evidence_body(record)


def _claim_body(record: ClaimRecord) -> str:
    body, _stamped = _typed_json(record.iri, term("Claim"), record)
    lines = [
        body,
        f"{n3_iri(record.iri)} {n3_iri(term('verificationStatus'))} {n3_str(record.verification_status.value)} .",
        f"{n3_iri(record.iri)} {n3_iri(term('scientificStatus'))} {n3_str(record.scientific_status.value)} .",
        f"{n3_iri(record.iri)} {n3_iri(term('concept'))} {n3_iri(record.concept_iri)} .",
        f"{n3_iri(record.iri)} {n3_iri(term('normalized'))} {n3_str(record.normalized)} .",
        *_tag_block(record.iri, record.tags, record.unknown_tags),
    ]
    for evidence in record.evidence_iris:
        lines.append(
            f"{n3_iri(record.iri)} {n3_iri(term('evidence'))} {n3_iri(evidence)} ."
        )
    for entity in record.entities:
        lines.append(
            f"{n3_iri(record.iri)} {n3_iri(term('entity'))} {n3_str(entity)} ."
        )
    return "\n".join(lines)


def _assertion_body(record: RelationAssertionRecord) -> str:
    body, _stamped = _typed_json(record.iri, term("RelationAssertion"), record)
    lines = [
        body,
        f"{n3_iri(record.iri)} {n3_iri(term('subject'))} {n3_iri(record.subject_iri)} .",
        f"{n3_iri(record.iri)} {n3_iri(term('object'))} {n3_iri(record.object_iri)} .",
        f"{n3_iri(record.iri)} {n3_iri(term('relation'))} {n3_iri(record.relation_iri)} .",
        f"{n3_iri(record.iri)} {n3_iri(term('verificationStatus'))} {n3_str(record.verification_status.value)} .",
        f"{n3_iri(record.iri)} {n3_iri(term('scientificStatus'))} {n3_str(record.scientific_status.value)} .",
        *_tag_block(record.iri, record.tags, record.unknown_tags),
    ]
    for evidence in record.evidence_iris:
        lines.append(
            f"{n3_iri(record.iri)} {n3_iri(term('evidence'))} {n3_iri(evidence)} ."
        )
    if record.derivation is not None:
        for premise in record.derivation.premises:
            lines.append(
                f"{n3_iri(record.iri)} {n3_iri(term('premise'))} {n3_iri(premise)} ."
            )
        lines.append(
            f"{n3_iri(record.iri)} {n3_iri(term('derivationRule'))} {n3_str(record.derivation.rule)} ."
        )
    return "\n".join(lines)


def _concept_body(record: ConceptRecord) -> str:
    body, _stamped = _typed_json(record.iri, term("Concept"), record)
    lines = [
        body,
        f"{n3_iri(record.iri)} {n3_iri(RDFS_LABEL)} {n3_str(record.label)} .",
        f"{n3_iri(record.iri)} {n3_iri(term('entityType'))} {n3_str(record.concept_type)} .",
    ]
    for alias in record.aliases:
        lines.append(f"{n3_iri(record.iri)} {n3_iri(term('alias'))} {n3_str(alias)} .")
    return "\n".join(lines)


def _profile_body(record: TypeProfile) -> str:
    body, _stamped = _typed_json(record.iri, term("TypeProfile"), record)
    return "\n".join(
        [
            body,
            f"{n3_iri(record.iri)} {n3_iri(term('relation'))} {n3_iri(record.relation_iri)} .",
        ]
    )


def _source_body(record: SourceRecord) -> str:
    body, _stamped = _typed_json(record.iri, term("Source"), record)
    lines = [
        body,
        f"{n3_iri(record.iri)} {n3_iri(term('contentHash'))} {n3_str(record.content_hash)} .",
        f"{n3_iri(record.iri)} {n3_iri(term('version'))} {n3_str(record.version)} .",
        *_tag_block(record.iri, record.tags, record.unknown_tags),
    ]
    return "\n".join(lines)


def _evidence_body(record: EvidenceRecord) -> str:
    body, _stamped = _typed_json(record.iri, term("Evidence"), record)
    lines = [
        body,
        f"{n3_iri(record.iri)} {n3_iri(term('source'))} {n3_iri(record.source_iri)} .",
        f"{n3_iri(record.iri)} {n3_iri(term('sourceVersion'))} {n3_str(record.source_version)} .",
        f"{n3_iri(record.iri)} {n3_iri(term('excerpt'))} {n3_str(record.excerpt)} .",
        *_tag_block(record.iri, record.tags, record.unknown_tags),
    ]
    optional = (
        ("page", record.page),
        ("section", record.section),
        ("tableCell", record.table_cell),
    )
    for name, value in optional:
        if value is not None:
            lines.append(f"{n3_iri(record.iri)} {n3_iri(term(name))} {n3_str(value)} .")
    if record.start_offset is not None:
        lines.append(
            f"{n3_iri(record.iri)} {n3_iri(term('startOffset'))} {n3_str(str(record.start_offset))} ."
        )
    if record.end_offset is not None:
        lines.append(
            f"{n3_iri(record.iri)} {n3_iri(term('endOffset'))} {n3_str(str(record.end_offset))} ."
        )
    return "\n".join(lines)


def _tag_block(
    subject: str, tags: dict[str, list[str]], unknown: list[str]
) -> list[str]:
    return tag_lines(subject, tags, unknown)


def write_verification(store: GraphStore, record: VerificationRecord) -> None:
    lines = [
        f"{n3_iri(record.iri)} a {n3_iri(term('VerificationRecord'))} .",
        f"{n3_iri(record.iri)} {n3_iri(term('target'))} {n3_iri(record.target_iri)} .",
        f"{n3_iri(record.iri)} {n3_iri(term('result'))} {n3_str(record.result.value)} .",
        f"{n3_iri(record.iri)} {n3_iri(term('justification'))} {n3_str(record.justification)} .",
        f"{n3_iri(record.iri)} {n3_iri(term('verifier'))} {n3_str(record.verifier)} .",
        f"{n3_iri(record.iri)} {n3_iri(term('modelVersion'))} {n3_str(record.model_version)} .",
        f"{n3_iri(record.iri)} {n3_iri(term('timestamp'))} {n3_str(record.timestamp)} .",
        f"{n3_iri(record.iri)} {n3_iri(term('recordJson'))} {n3_str(record.model_dump_json())} .",
    ]
    for check in record.checks:
        lines.append(f"{n3_iri(record.iri)} {n3_iri(term('check'))} {n3_str(check)} .")
    store.update(
        "INSERT DATA { GRAPH "
        + n3_iri(GRAPH_OPERATIONS)
        + " {\n"
        + "\n".join(lines)
        + "\n} }"
    )


def write_definition(store: GraphStore, definition: RelationDefinition) -> None:
    lines = [
        f"{n3_iri(definition.iri)} a {n3_iri(OWL_OBJECT_PROPERTY)} .",
        f"{n3_iri(definition.iri)} {n3_iri(RDFS_LABEL)} {n3_str(definition.label)} .",
        f"{n3_iri(definition.iri)} {n3_iri(term('definition'))} {n3_str(definition.definition)} .",
        f"{n3_iri(definition.iri)} {n3_iri(term('directionality'))} {n3_str(definition.directionality)} .",
        f"{n3_iri(definition.iri)} {n3_iri(term('subjectType'))} {n3_iri(definition.subject_type)} .",
        f"{n3_iri(definition.iri)} {n3_iri(term('objectType'))} {n3_iri(definition.object_type)} .",
        f"{n3_iri(definition.iri)} {n3_iri(term('version'))} {n3_str(definition.version)} .",
        f"{n3_iri(definition.iri)} {n3_iri(term('recordJson'))} {n3_str(definition.model_dump_json())} .",
    ]
    for parent in definition.parents:
        lines.append(
            f"{n3_iri(definition.iri)} {n3_iri(RDFS_SUB_PROPERTY)} {n3_iri(parent)} ."
        )
    for example in definition.examples:
        lines.append(
            f"{n3_iri(definition.iri)} {n3_iri(term('example'))} {n3_str(example)} ."
        )
    for alias in definition.aliases:
        lines.append(
            f"{n3_iri(definition.iri)} {n3_iri(term('alias'))} {n3_str(alias)} ."
        )
    store.update(
        "INSERT DATA { GRAPH "
        + n3_iri(GRAPH_ONTOLOGY)
        + " {\n"
        + "\n".join(lines)
        + "\n} }"
    )


def delete_record(store: GraphStore, iri: str) -> None:
    store.update(
        f"""
        DELETE {{ GRAPH ?g {{ {n3_iri(iri)} ?p ?o }} }}
        WHERE {{
          VALUES ?g {{
            {n3_iri(GRAPH_ACCEPTED)} {n3_iri(GRAPH_CANDIDATES)} {n3_iri(GRAPH_OPERATIONS)}
          }}
          GRAPH ?g {{ {n3_iri(iri)} ?p ?o }}
        }}
        """
    )
