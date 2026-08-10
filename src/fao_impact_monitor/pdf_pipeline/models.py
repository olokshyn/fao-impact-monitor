"""Mongo schemas for source-grounded PDF evidence.

Generated retrieval text is intentionally kept separate from source evidence.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, ClassVar, Literal, cast

from beanie import Document as BeanieDocument
from beanie import PydanticObjectId
from pydantic import BaseModel, Field, model_validator
from pymongo import ASCENDING, IndexModel

ElNinoEventId = Literal[
    "el_nino_1997_98",
    "el_nino_2015_16",
    "el_nino_2018_19",
    "el_nino_2023_24",
    "el_nino_2026_27",
]
EnsoRelationship = Literal[
    "attributed", "associated", "explicitly_not_attributed", "uncertain", "unrelated"
]
EvidenceKind = Literal[
    "scope_context",
    "hazard",
    "agrifood_impact",
    "livelihood_impact",
    "response_outcome",
]
AssertionMode = Literal[
    "contextual",
    "observed",
    "reported",
    "estimated",
    "forecast",
    "scenario",
    "measured_response",
]
SourceModality = Literal["text", "table", "chart", "map", "diagram", "mixed"]
RepresentationKind = Literal[
    "evidence_source_text",
    "evidence_visual_facts",
    "evidence_contextualized",
    "section_context",
]

ELIGIBLE_EVENTS: frozenset[str] = frozenset(
    {
        "el_nino_1997_98",
        "el_nino_2015_16",
        "el_nino_2018_19",
        "el_nino_2023_24",
        "el_nino_2026_27",
    }
)


class ArtifactRef(BaseModel):
    relative_path: str
    sha256: str
    media_type: Literal["application/pdf", "image/png"]
    width_px: int | None = None
    height_px: int | None = None


class BoundingBox(BaseModel):
    """PDF points after rotation normalization; top-left origin."""

    x0: float
    y0: float
    x1: float
    y1: float

    @model_validator(mode="after")
    def has_positive_area(self) -> BoundingBox:
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("bounding_box must have positive area")
        return self


class SourceRegion(BaseModel):
    region_id: str
    physical_page: int
    printed_page: str | None = None
    page_width_points: float
    page_height_points: float
    rotation_degrees: Literal[0, 90, 180, 270] = 0
    bounding_box: BoundingBox | None = None
    extracted_text: str | None = None
    extracted_text_sha256: str | None = None
    page_artifact: ArtifactRef
    crop_artifact: ArtifactRef | None = None

    @model_validator(mode="after")
    def extracted_text_hash_matches_presence(self) -> SourceRegion:
        if bool(self.extracted_text) != bool(self.extracted_text_sha256):
            raise ValueError(
                "extracted_text and extracted_text_sha256 must appear together"
            )
        return self


class EventContext(BaseModel):
    event_id: ElNinoEventId
    relationship: EnsoRelationship
    origin: Literal["explicit", "inherited"]
    supporting_evidence_ids: list[str] = Field(default_factory=list)


class CountryContext(BaseModel):
    iso3: str = Field(min_length=3, max_length=3)
    role: Literal["subject", "mentioned"]
    origin: Literal["explicit", "inherited"]
    supporting_evidence_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def normalize_iso3(self) -> CountryContext:
        self.iso3 = self.iso3.upper()
        return self


class ValidationResult(BaseModel):
    verdict: Literal["passed", "failed", "not_applicable"]
    checked_at: datetime
    validator: str
    details: str | None = None
    supporting_region_ids: list[str] = Field(default_factory=list)


class ModelVersions(BaseModel):
    gemini: str
    luna: str
    titan: str


class PromptVersions(BaseModel):
    document_structure: str = "v2"
    boundary_validation: str = "v2"
    evidence_extraction: str = "v4"
    visual_verification: str = "v2"
    section_summary: str = "v1"


class VerifiedVisualFact(BaseModel):
    fact_id: str
    text: str
    supporting_region_ids: list[str] = Field(min_length=1)
    verifier_verdict: Literal["entailed"]
    verifier_model: str
    verifier_prompt_version: str


class SectionContextCard(BeanieDocument):
    section_id: str
    document_id: PydanticObjectId
    document_sha256: str
    parent_section_id: str | None = None
    ordinal: int
    level: int
    title: str
    hierarchy_path: list[str]
    physical_page_start: int
    physical_page_end: int
    printed_page_start: str | None = None
    printed_page_end: str | None = None
    countries: list[CountryContext] = Field(default_factory=list)
    regions: list[str] = Field(default_factory=list)
    events: list[EventContext] = Field(default_factory=list)
    reporting_modes: set[AssertionMode] = Field(default_factory=set)
    scope_evidence_ids: list[str] = Field(default_factory=list)
    summary: str
    retrieval_text: str
    boundary_validation: ValidationResult
    context_validation: ValidationResult
    model_versions: ModelVersions
    prompt_versions: PromptVersions
    pipeline: Literal["pdf_pipeline"] = "pdf_pipeline"
    created_at: datetime

    class Settings:
        name = "pdf_sections"
        indexes: ClassVar[list[IndexModel]] = [
            IndexModel([("section_id", ASCENDING)], unique=True),
            IndexModel([("document_id", ASCENDING), ("ordinal", ASCENDING)]),
        ]


class EvidenceUnit(BeanieDocument):
    evidence_id: str
    document_id: PydanticObjectId
    document_sha256: str
    section_id: str
    evidence_kind: EvidenceKind
    assertion_mode: AssertionMode
    source_modality: SourceModality
    source_regions: list[SourceRegion] = Field(min_length=1)
    physical_pages: list[int] = Field(min_length=1)
    printed_pages: list[str] = Field(default_factory=list)
    source_text: str | None = None
    verified_visual_facts: list[VerifiedVisualFact] = Field(default_factory=list)
    unit_description: str
    retrieval_text: str
    countries: list[CountryContext] = Field(default_factory=list)
    events: list[EventContext] = Field(default_factory=list)
    scope_evidence_ids: list[str] = Field(default_factory=list)
    continuation_evidence_ids: list[str] = Field(default_factory=list)
    canonical_evidence_text: str
    searchable: bool
    exclusion_reason: str | None = None
    extraction_validation: ValidationResult
    visual_validation: ValidationResult | None = None
    eligibility_validation: ValidationResult
    model_versions: ModelVersions
    prompt_versions: PromptVersions
    pipeline: Literal["pdf_pipeline"] = "pdf_pipeline"
    created_at: datetime

    @model_validator(mode="after")
    def source_text_is_exact_region_concat(self) -> EvidenceUnit:
        exact = "\n".join(
            region.extracted_text
            for region in self.source_regions
            if region.extracted_text
        )
        if self.source_text is not None and self.source_text != exact:
            raise ValueError(
                "source_text must be assembled exactly from source_regions"
            )
        if self.searchable and not self.canonical_evidence_text:
            raise ValueError("searchable evidence needs canonical_evidence_text")
        return self

    class Settings:
        name = "pdf_evidence"
        indexes: ClassVar[list[IndexModel]] = [
            IndexModel([("evidence_id", ASCENDING)], unique=True),
            IndexModel([("section_id", ASCENDING), ("searchable", ASCENDING)]),
            IndexModel([("document_id", ASCENDING), ("physical_pages", ASCENDING)]),
        ]


class PdfDocumentRecord(BeanieDocument):
    content_sha256: str
    original_filename: str
    discovered_paths: list[str]
    source_artifact: ArtifactRef
    title: str | None = None
    publication_date: date | None = None
    reporting_period: str | None = None
    purpose: str | None = None
    physical_page_count: int
    covered_events: list[EventContext] = Field(default_factory=list)
    root_section_ids: list[str] = Field(default_factory=list)
    structure_analysis: dict[str, Any] | None = None
    completed_section_ids: list[str] = Field(default_factory=list)
    status: Literal["processing", "completed", "failed"]
    failure: str | None = None
    completed_at: datetime | None = None
    model_versions: ModelVersions
    prompt_versions: PromptVersions
    pipeline: Literal["pdf_pipeline"] = "pdf_pipeline"
    created_at: datetime
    updated_at: datetime

    class Settings:
        name = "pdf_documents"
        indexes: ClassVar[list[IndexModel]] = [
            IndexModel([("content_sha256", ASCENDING)], unique=True),
            IndexModel([("status", ASCENDING)]),
        ]


class PdfEmbeddingRecord(BeanieDocument):
    """One Titan representation.  ``chunk_text`` is never retrieval text."""

    document_id: PydanticObjectId
    document_url: str
    document_external_id: str | None = None
    document_title: str | None = None
    document_meta: dict[str, Any] = Field(default_factory=dict)
    document_type: Literal["pdf"] = "pdf"
    document_source: Literal["PdfEvidencePipeline"] = "PdfEvidencePipeline"
    chunk_index: int
    chunk_text: str
    countries_iso3: list[str] = Field(default_factory=list)
    embedding: list[float]
    pipeline: Literal["pdf_pipeline"] = "pdf_pipeline"
    owner_kind: Literal["evidence", "section"]
    owner_id: str
    evidence_id: str | None = None
    section_id: str
    representation_kind: RepresentationKind
    embedding_text: str
    event_ids: list[ElNinoEventId] = Field(default_factory=list)
    enso_relationships: list[EnsoRelationship] = Field(default_factory=list)
    assertion_mode: AssertionMode | None = None
    searchable: bool = True

    class Settings:
        name = "embeddings"
        indexes: ClassVar[list[IndexModel]] = [
            IndexModel([("pipeline", ASCENDING), ("owner_id", ASCENDING)]),
            IndexModel([("pipeline", ASCENDING), ("section_id", ASCENDING)]),
        ]


def effective_events(
    direct: list[EventContext], inherited: list[EventContext]
) -> list[EventContext]:
    """Apply direct-over-inherited and conflicting-inheritance precedence."""
    by_event: dict[str, list[EventContext]] = {}
    for item in direct:
        by_event.setdefault(item.event_id, []).append(item)
    answer: list[EventContext] = []
    for event_id, values in by_event.items():
        # Explicit negation is never softened by a weaker direct statement.
        if any(value.relationship == "explicitly_not_attributed" for value in values):
            answer.append(
                next(
                    value
                    for value in values
                    if value.relationship == "explicitly_not_attributed"
                )
            )
        elif len({value.relationship for value in values}) == 1:
            answer.append(values[0])
        else:
            answer.append(
                EventContext(
                    event_id=cast(ElNinoEventId, event_id),
                    relationship="uncertain",
                    origin="explicit",
                )
            )
    inherited_by_event: dict[str, list[EventContext]] = {}
    for item in inherited:
        if item.event_id not in by_event:
            inherited_by_event.setdefault(item.event_id, []).append(item)
    for event_id, values in inherited_by_event.items():
        relationships = {value.relationship for value in values}
        if len(relationships) == 1:
            answer.append(values[0].model_copy(update={"origin": "inherited"}))
        else:
            answer.append(
                EventContext(
                    event_id=cast(ElNinoEventId, event_id),
                    relationship="uncertain",
                    origin="inherited",
                )
            )
    return answer


def is_searchable(events: list[EventContext]) -> tuple[bool, str | None]:
    if any(event.relationship == "explicitly_not_attributed" for event in events):
        return False, "explicitly_not_attributed"
    if any(
        event.event_id in ELIGIBLE_EVENTS
        and event.relationship in {"attributed", "associated"}
        for event in events
    ):
        return True, None
    return False, "no_eligible_attributed_or_associated_event"
