"""Structured inputs and outputs for Athena agents."""

from __future__ import annotations

from pydantic import BaseModel, Field

from fao_impact_monitor.athena.model.records import Quantity
from fao_impact_monitor.athena.model.status import ScientificStatus


class EvidenceDraft(BaseModel):
    excerpt: str
    page: str | None = None
    section: str | None = None
    table_cell: str | None = None
    start_offset: int | None = None
    end_offset: int | None = None


class ClaimDraft(BaseModel):
    original_wording: str
    normalized: str
    entities: list[str] = Field(default_factory=list)
    processes: list[str] = Field(default_factory=list)
    quantities: list[Quantity] = Field(default_factory=list)
    location: str | None = None
    time_scope: str | None = None
    population: str | None = None
    conditions: list[str] = Field(default_factory=list)
    applicable_tags: dict[str, list[str]] = Field(default_factory=dict)
    unknown_tags: list[str] = Field(default_factory=list)
    mentioned_places: list[str] = Field(default_factory=list)
    evidence_index: int
    concept_label: str
    concept_type: str = "Phenomenon"
    concept_aliases: list[str] = Field(default_factory=list)


class RelationDraft(BaseModel):
    subject_index: int
    object_index: int
    relation: str
    conditions: list[str] = Field(default_factory=list)
    evidence_indexes: list[int] = Field(default_factory=list)
    tentative: bool = False
    study_design: str | None = None
    scientific_status: ScientificStatus = ScientificStatus.SOURCE_REPORTED


class ExtractionRequest(BaseModel):
    text: str
    tags: dict[str, list[str]] = Field(default_factory=dict)
    unknown_tags: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)
    raw_claim: bool = False


class ExtractionResult(BaseModel):
    evidence: list[EvidenceDraft] = Field(default_factory=list)
    claims: list[ClaimDraft] = Field(default_factory=list)
    relations: list[RelationDraft] = Field(default_factory=list)


class VerificationPacket(BaseModel):
    excerpt: str
    claim_wording: str
    normalized: str
    quantities: list[Quantity] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    relation: str | None = None
    scientific_status: str | None = None
    derivation_rule: str | None = None
    premises: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class VerificationJudgment(BaseModel):
    faithful: bool | None = None
    relation_supported: bool | None = None
    context_compatible: bool | None = None
    derivation_valid: bool | None = None
    expresses_causality: bool | None = None
    tentative: bool = False
    study_design: str | None = None
    justification: str


class NewTypeDraft(BaseModel):
    name: str
    definition: str
    parent: str
    subject_type: str
    object_type: str
    directionality: str
    examples: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)


class ProposedRelation(BaseModel):
    subject_iri: str
    object_iri: str
    relation: str
    conditions: list[str] = Field(default_factory=list)
    evidence_iris: list[str] = Field(default_factory=list)
    scientific_status: ScientificStatus = ScientificStatus.SOURCE_REPORTED
    tentative: bool = False
    study_design: str | None = None
    direction: str | None = None
    new_type: NewTypeDraft | None = None


class DiscoveryRequest(BaseModel):
    claim_text: str
    candidates: list[str]
    existing_relations: list[str] = Field(default_factory=list)


class DiscoveryResult(BaseModel):
    relations: list[ProposedRelation] = Field(default_factory=list)


class OntologyRequest(BaseModel):
    proposal: NewTypeDraft
    existing: list[str]


class OntologyDecision(BaseModel):
    reuse_name: str | None = None
    definition: str = ""
    parent: str = ""
    subject_type: str = ""
    object_type: str = ""
    directionality: str = "directional"
    examples: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)


class SubQuestion(BaseModel):
    text: str
    variables: list[str] = Field(default_factory=list)
    contexts: list[str] = Field(default_factory=list)


class ResearchPlan(BaseModel):
    subquestions: list[SubQuestion] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class Synthesis(BaseModel):
    answer: str
    assumptions: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    qualitative: bool = True
    quantitative_model: str | None = None
    uncertainty: str | None = None
    direct_causal_claim_ids: list[str] = Field(default_factory=list)


class ExternalHit(BaseModel):
    title: str | None = None
    url: str | None = None
    text: str
    tags: dict[str, list[str]] = Field(default_factory=dict)
    acquisition_channel: str = "web-search"
