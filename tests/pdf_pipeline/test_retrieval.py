from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from beanie import PydanticObjectId
from pymongo.errors import OperationFailure

from fao_impact_monitor.config import PdfPipelineConfig
from fao_impact_monitor.pdf_pipeline.ingest import PdfEvidenceIngestor
from fao_impact_monitor.pdf_pipeline.models import (
    ArtifactRef,
    EventContext,
    EvidenceUnit,
    ModelVersions,
    PdfDocumentRecord,
    PdfEmbeddingRecord,
    PromptVersions,
    SourceRegion,
    ValidationResult,
    VerifiedVisualFact,
)
from fao_impact_monitor.pdf_pipeline.retrieval import (
    PdfEvidenceVectorStore,
    ensure_pdf_pipeline_indexes,
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


def test_visual_search_hit_uses_crop_artifact_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        id=PydanticObjectId("507f1f77bcf86cd799439012"),
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
        printed_pages=["1"],
        source_text=text,
        verified_visual_facts=[
            VerifiedVisualFact(
                fact_id="fact-1",
                text="The map shows 35% affected.",
                supporting_region_ids=["r1"],
                verifier_verdict="entailed",
                verifier_model="test",
                verifier_prompt_version="v1",
            )
        ],
        unit_description="Map description",
        retrieval_text="Somalia forecast map",
        canonical_evidence_text=text,
        searchable=True,
        events=[
            EventContext(
                event_id="el_nino_2015_16",
                relationship="associated",
                origin="explicit",
            )
        ],
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

    monkeypatch.setattr(
        PdfDocumentRecord,
        "get",
        AsyncMock(return_value=document),
    )
    hit = asyncio.run(store._hit(unit, 0.9))
    assert hit.document_meta["evidence_id"] == "e1"
    assert hit.document_meta["source_text"] == text
    assert hit.document_meta["physical_pages"] == [2]
    assert hit.document_meta["printed_pages"] == ["1"]
    assert hit.document_meta["events"][0]["event_id"] == "el_nino_2015_16"
    assert hit.document_meta["verified_visual_facts"][0]["text"] == (
        "The map shows 35% affected."
    )


def test_bundle_appends_visual_facts_missing_from_canonical() -> None:
    unit = EvidenceUnit.model_construct(
        evidence_id="e-visual",
        canonical_evidence_text="[TARGET SOURCE EVIDENCE | physical page 25]\nNone.",
        physical_pages=[25],
        printed_pages=["22"],
        scope_evidence_ids=[],
        continuation_evidence_ids=[],
        verified_visual_facts=[
            VerifiedVisualFact(
                fact_id="fact-india",
                text=(
                    "India: Wheat 5-yr avg=111.8, 2026=120.2, Change 2026/2025=-1.2%"
                ),
                supporting_region_ids=["r1"],
                verifier_verdict="entailed",
                verifier_model="test",
                verifier_prompt_version="v1",
            )
        ],
    )
    store = PdfEvidenceVectorStore()
    bundled = asyncio.run(store._bundle(unit))
    assert "India: Wheat 5-yr avg=111.8" in bundled
    assert "[VERIFIED VISUAL FACT | physical page 25 | printed page 22" in bundled


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


class _SearchIndexCursor:
    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self._docs = docs

    def __aiter__(self) -> _SearchIndexCursor:
        self._iter = iter(self._docs)
        return self

    async def __anext__(self) -> dict[str, Any]:
        try:
            return next(self._iter)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


class _SearchIndexCollection:
    def __init__(
        self,
        existing: list[dict[str, Any]],
        *,
        drop_error: OperationFailure | None = None,
        create_error: OperationFailure | None = None,
    ) -> None:
        self.existing = existing
        self.drop_error = drop_error
        self.create_error = create_error
        self.dropped: list[str] = []
        self.created: list[Any] = []

    async def create_index(self, _key: Any, **_kwargs: Any) -> str:
        return "ok"

    async def list_search_indexes(self) -> _SearchIndexCursor:
        return _SearchIndexCursor(self.existing)

    async def drop_search_index(self, name: str) -> None:
        self.dropped.append(name)
        if self.drop_error is not None:
            raise self.drop_error

    async def create_search_index(self, model: Any) -> str:
        self.created.append(model)
        if self.create_error is not None:
            raise self.create_error
        return str(model.document.get("name", "idx"))


def _install_search_collection(
    monkeypatch: pytest.MonkeyPatch, collection: _SearchIndexCollection
) -> None:
    monkeypatch.setattr(
        PdfEmbeddingRecord,
        "get_pymongo_collection",
        staticmethod(lambda: collection),
    )


def test_ensure_indexes_ignores_concurrent_drop_of_missing_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vector = vector_index_definition()
    text = text_index_definition()
    collection = _SearchIndexCollection(
        [
            {
                "name": vector["name"],
                "status": "READY",
                "latestDefinition": vector["definition"],
            },
            {
                "name": text["name"],
                "status": "READY",
                "latestDefinition": {"mappings": {"dynamic": True}},
            },
        ],
        drop_error=OperationFailure(
            "Index pdf_pipeline_text_index not found in fao_impact_monitor.embeddings",
            27,
        ),
    )
    _install_search_collection(monkeypatch, collection)

    asyncio.run(ensure_pdf_pipeline_indexes())

    assert collection.dropped == [text["name"]]
    assert len(collection.created) == 1
    assert collection.created[0].document["name"] == text["name"]


def test_ensure_indexes_ignores_create_when_peer_already_created(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection = _SearchIndexCollection(
        [],
        create_error=OperationFailure("Index already exists with that name", 68),
    )
    _install_search_collection(monkeypatch, collection)

    asyncio.run(ensure_pdf_pipeline_indexes())

    assert collection.dropped == []
    assert [model.document["name"] for model in collection.created] == [
        vector_index_definition()["name"],
        text_index_definition()["name"],
    ]


def test_ensure_indexes_leaves_in_progress_index_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vector = vector_index_definition()
    text = text_index_definition()
    collection = _SearchIndexCollection(
        [
            {
                "name": vector["name"],
                "status": "PENDING",
                "latestDefinition": {},
            },
            {
                "name": text["name"],
                "status": "READY",
                "latestDefinition": text["definition"],
            },
        ]
    )
    _install_search_collection(monkeypatch, collection)

    asyncio.run(ensure_pdf_pipeline_indexes())

    assert collection.dropped == []
    assert collection.created == []


def test_ensure_indexes_skips_ready_index_when_atlas_adds_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vector = vector_index_definition()
    text = text_index_definition()
    vector_actual = {
        "fields": [
            {**vector["definition"]["fields"][0], "hnswOptions": {"maxEdges": 16}},
            *vector["definition"]["fields"][1:],
        ]
    }
    collection = _SearchIndexCollection(
        [
            {
                "name": vector["name"],
                "status": "READY",
                "latestDefinition": vector_actual,
            },
            {
                "name": text["name"],
                "status": "READY",
                "latestDefinition": text["definition"],
            },
        ]
    )
    _install_search_collection(monkeypatch, collection)

    asyncio.run(ensure_pdf_pipeline_indexes())

    assert collection.dropped == []
    assert collection.created == []


def test_ensure_indexes_recreate_drops_compatible_indexes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vector = vector_index_definition()
    text = text_index_definition()
    collection = _SearchIndexCollection(
        [
            {
                "name": vector["name"],
                "status": "READY",
                "latestDefinition": vector["definition"],
            },
            {
                "name": text["name"],
                "status": "READY",
                "latestDefinition": text["definition"],
            },
        ]
    )
    _install_search_collection(monkeypatch, collection)

    asyncio.run(ensure_pdf_pipeline_indexes(recreate=True))

    assert collection.dropped == [vector["name"], text["name"]]
    assert [model.document["name"] for model in collection.created] == [
        vector["name"],
        text["name"],
    ]
