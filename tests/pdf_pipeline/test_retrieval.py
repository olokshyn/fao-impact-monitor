from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from beanie import PydanticObjectId

from fao_impact_monitor.config import PdfPipelineConfig
from fao_impact_monitor.pdf_pipeline.ingest import PdfEvidenceIngestor
from fao_impact_monitor.pdf_pipeline.models import (
    ArtifactRef,
    EvidenceUnit,
    ModelVersions,
    PdfDocumentRecord,
    PromptVersions,
    SourceRegion,
    ValidationResult,
)
from fao_impact_monitor.pdf_pipeline.retrieval import (
    PdfEvidenceVectorStore,
    hybrid_pipeline,
    text_index_definition,
    vector_index_definition,
)


def test_hybrid_search_is_scoped_to_pipeline_owned_eligible_records() -> None:
    pipeline = hybrid_pipeline(
        "maize drought", [0.1, 0.2], countries_iso3=["ZMB"], limit=4
    )
    vector = pipeline[0]["$rankFusion"]["input"]["pipelines"]["vector"][0][
        "$vectorSearch"
    ]
    text = pipeline[0]["$rankFusion"]["input"]["pipelines"]["text"][0]["$search"]
    assert vector["filter"] == {
        "pipeline": "pdf_pipeline",
        "searchable": True,
        "countries_iso3": {"$in": ["ZMB"]},
    }
    assert text["compound"]["must"][0]["text"]["path"] == "embedding_text"
    assert vector_index_definition()["definition"]["fields"][0]["path"] == "embedding"
    assert (
        "embedding_text" in text_index_definition()["definition"]["mappings"]["fields"]
    )


def test_visual_search_hit_uses_crop_artifact_path(tmp_path: Path) -> None:
    text = "Exact map caption."
    page = ArtifactRef(
        relative_path="sha/pages/page-0002.png",
        sha256="a" * 64,
        media_type="image/png",
    )
    crop = ArtifactRef(
        relative_path="sha/crops/map.png",
        sha256="b" * 64,
        media_type="image/png",
    )
    validation = ValidationResult(
        verdict="passed",
        checked_at=datetime.now(UTC),
        validator="test",
    )
    unit = EvidenceUnit.model_construct(
        evidence_id="e1",
        document_id=PydanticObjectId("507f1f77bcf86cd799439011"),
        document_sha256="c" * 64,
        section_id="s1",
        evidence_kind="hazard",
        assertion_mode="forecast",
        source_modality="map",
        source_regions=[
            SourceRegion(
                region_id="r1",
                physical_page=2,
                page_width_points=600,
                page_height_points=800,
                extracted_text=text,
                extracted_text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                page_artifact=page,
                crop_artifact=crop,
            )
        ],
        physical_pages=[2],
        source_text=text,
        unit_description="Map description",
        retrieval_text="Somalia forecast map",
        canonical_evidence_text=text,
        searchable=True,
        extraction_validation=validation,
        eligibility_validation=validation,
        model_versions=ModelVersions(gemini="g", luna="l", titan="t"),
        prompt_versions=PromptVersions(),
        created_at=datetime.now(UTC),
    )
    store = PdfEvidenceVectorStore(config=PdfPipelineConfig(artifact_dir=tmp_path))

    artifacts = store._visual_artifacts(unit)

    assert artifacts == [
        {
            "artifact_id": "r1",
            "path": str((tmp_path / "sha/crops/map.png").resolve()),
            "sha256": "b" * 64,
            "media_type": "image/png",
            "physical_page": 2,
        }
    ]

    embedding = PdfEvidenceIngestor(
        config=PdfPipelineConfig(artifact_dir=tmp_path)
    )._embedding_base(
        unit,
        "evidence_contextualized",
        unit.retrieval_text,
        Path("report.pdf"),
        document_title="Anticipating El Niño",
    )
    assert embedding["document_title"] == "Anticipating El Niño"
    assert embedding["document_url"] == "file://report.pdf"

    document = PdfDocumentRecord.model_construct(
        discovered_paths=["fao_data/El Niño Plan.pdf"]
    )
    assert store._document_url(unit, document) == (
        "file://fao_data/El%20Ni%C3%B1o%20Plan.pdf"
    )


def test_search_embeddings_returns_raw_representations_without_deduplication() -> None:
    rows: list[dict[str, Any]] = [
        {"owner_id": "e1", "representation_kind": "evidence_source_text"},
        {"owner_id": "e1", "representation_kind": "evidence_contextualized"},
    ]
    captured: dict[str, object] = {}

    async def embed(_query: str) -> list[float]:
        return [0.1, 0.2]

    async def aggregate(pipeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
        captured["pipeline"] = pipeline
        return rows

    store = PdfEvidenceVectorStore(
        embed_query_fn=embed,
        aggregate_fn=aggregate,
    )

    result = asyncio.run(
        store.search_embeddings("maize loss", countries_iso3=["KEN"], limit=2)
    )

    assert result == rows
    pipeline = captured["pipeline"]
    assert isinstance(pipeline, list)
    assert pipeline[-2] == {"$limit": 2}
