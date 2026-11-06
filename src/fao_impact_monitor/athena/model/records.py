"""In-process knowledge records stored authoritatively in Jena."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from fao_impact_monitor.athena.model.status import (
    ScientificStatus,
    VerificationStatus,
)


class Quantity(BaseModel):
    value: str
    unit: str | None = None
    comparator: str | None = None


class TagDerivation(BaseModel):
    field: str
    value: str
    rule: str
    premises: list[str] = Field(default_factory=list)


class ObjectPointer(BaseModel):
    logical_id: str
    sha256: str
    version: str
    media_type: str


class PublicationMetadata(BaseModel):
    title: str | None = None
    publication_date: str | None = None
    authors: list[str] = Field(default_factory=list)
    publishing_group: str | None = None
    publisher: str | None = None
    journal: str | None = None


class SourceRecord(BaseModel):
    iri: str
    external_id: str | None = None
    url: str | None = None
    doi: str | None = None
    metadata: PublicationMetadata = Field(default_factory=PublicationMetadata)
    tags: dict[str, list[str]] = Field(default_factory=dict)
    unknown_tags: list[str] = Field(default_factory=list)
    mentioned_places: list[str] = Field(default_factory=list)
    snapshot: ObjectPointer
    content_hash: str
    version: str
    record_version: str = ""


class EvidenceRecord(BaseModel):
    iri: str
    excerpt: str
    source_iri: str
    source_version: str
    page: str | None = None
    section: str | None = None
    table_cell: str | None = None
    start_offset: int | None = None
    end_offset: int | None = None
    tags: dict[str, list[str]] = Field(default_factory=dict)
    unknown_tags: list[str] = Field(default_factory=list)
    record_version: str = ""


class ClaimRecord(BaseModel):
    iri: str
    original_wording: str
    normalized: str
    entities: list[str] = Field(default_factory=list)
    processes: list[str] = Field(default_factory=list)
    quantities: list[Quantity] = Field(default_factory=list)
    location: str | None = None
    time_scope: str | None = None
    population: str | None = None
    conditions: list[str] = Field(default_factory=list)
    tags: dict[str, list[str]] = Field(default_factory=dict)
    unknown_tags: list[str] = Field(default_factory=list)
    tag_derivations: list[TagDerivation] = Field(default_factory=list)
    evidence_iris: list[str]
    concept_iri: str
    concept_label: str
    concept_type: str = "Phenomenon"
    scientific_status: ScientificStatus
    verification_status: VerificationStatus
    tentative: bool = False
    study_design: str | None = None
    record_version: str = ""


class Derivation(BaseModel):
    premises: list[str]
    rule: str
    model: str | None = None
    assumptions: list[str] = Field(default_factory=list)


class RelationAssertionRecord(BaseModel):
    iri: str
    subject_iri: str
    object_iri: str
    relation_iri: str
    direction: str | None = None
    conditions: list[str] = Field(default_factory=list)
    evidence_iris: list[str]
    derivation: Derivation | None = None
    scientific_status: ScientificStatus
    verification_status: VerificationStatus
    tags: dict[str, list[str]] = Field(default_factory=dict)
    unknown_tags: list[str] = Field(default_factory=list)
    tentative: bool = False
    study_design: str | None = None
    record_version: str = ""


class RelationDefinition(BaseModel):
    iri: str
    label: str
    definition: str
    parents: list[str] = Field(default_factory=list)
    subject_type: str
    object_type: str
    directionality: str
    examples: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    version: str


class VerificationRecord(BaseModel):
    iri: str
    target_iri: str
    checks: list[str]
    result: VerificationStatus
    justification: str
    verifier: str
    model_version: str
    timestamp: str


class ConceptRecord(BaseModel):
    iri: str
    label: str
    concept_type: str = "Phenomenon"
    aliases: list[str] = Field(default_factory=list)
    representative_claims: list[str] = Field(default_factory=list)
    mechanisms: list[str] = Field(default_factory=list)
    record_version: str = ""


class TypeProfile(BaseModel):
    iri: str
    relation_iri: str
    text: str
    aliases: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    record_version: str = ""


class TagField(BaseModel):
    name: str
    value_type: str
    multiplicity: Literal["one", "many"]
    vocabulary: list[str] = Field(default_factory=list)
    extraction_rule: str
    inheritance_rule: str
    inherits: bool = False


class RetrievalRecipe(BaseModel):
    relation_iri: str
    subject_type: str
    object_type: str
    aliases: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    terms: list[str] = Field(default_factory=list)


class CitedStatement(BaseModel):
    text: str
    claim_iri: str
    evidence_iris: list[str]
    scientific_status: str
    extrapolation: bool = False
    relation_iri: str | None = None


class ContradictionView(BaseModel):
    assertion_iri: str
    summary: str
    claim_iris: list[str]
    evidence_iris: list[str]


class EvidenceView(BaseModel):
    mode: Literal["strict", "extrapolation"]
    statements: list[CitedStatement] = Field(default_factory=list)
    contradictions: list[ContradictionView] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    qualitative: bool = True
    quantitative_model: str | None = None
    uncertainty: str | None = None


class ResearchResult(BaseModel):
    question: str
    strict: EvidenceView
    extrapolation: EvidenceView | None = None
    discovered_claim_ids: list[str] = Field(default_factory=list)
    discovered_assertion_ids: list[str] = Field(default_factory=list)
