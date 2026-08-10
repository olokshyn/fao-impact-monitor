from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Any

from fao_impact_monitor.config import PdfPipelineConfig
from fao_impact_monitor.pdf_pipeline.artifacts import PdfArtifacts
from fao_impact_monitor.pdf_pipeline.ingest import PdfEvidenceIngestor
from fao_impact_monitor.pdf_pipeline.models import ArtifactRef, SourceRegion


class VisualClient:
    def __init__(self, *, initial_entailed: bool = True) -> None:
        self.initial_entailed = initial_entailed
        self.descriptions = 0
        self.verifications: list[str] = []
        self.verification_batches: list[list[str]] = []

    async def describe_visual(
        self, image_path: Path, *, context: str
    ) -> dict[str, Any]:
        assert image_path.is_file()
        assert context
        self.descriptions += 1
        return {"facts": [{"text": "The map shows forecast rainfall zones."}]}

    async def verify_visual(self, image_path: Path, facts: list[str]) -> dict[str, Any]:
        assert image_path.is_file()
        self.verifications.extend(facts)
        self.verification_batches.append(facts)
        return {
            "facts": [
                {
                    "text": text,
                    "entailed": (
                        self.initial_entailed
                        if text == "Unverified initial description."
                        else True
                    ),
                }
                for text in facts
            ]
        }


def _region(tmp_path: Path) -> tuple[PdfArtifacts, SourceRegion]:
    content_sha = "a" * 64
    artifacts = PdfArtifacts(tmp_path, content_sha, 180)
    image_path = artifacts.directory / "crops" / "map.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"png")
    artifact = ArtifactRef(
        relative_path=str(image_path.relative_to(tmp_path)),
        sha256=hashlib.sha256(b"png").hexdigest(),
        media_type="image/png",
    )
    return artifacts, SourceRegion(
        region_id="e1:r1",
        physical_page=2,
        page_width_points=600,
        page_height_points=800,
        page_artifact=artifact,
        crop_artifact=artifact,
    )


def test_visual_unit_without_candidates_gets_described_and_verified(
    tmp_path: Path,
) -> None:
    artifacts, region = _region(tmp_path)
    client = VisualClient()
    ingestor = PdfEvidenceIngestor(config=PdfPipelineConfig(artifact_dir=tmp_path))

    facts = asyncio.run(
        ingestor._verify_visual_facts(
            {"unit_description": "Rainfall forecast map", "visual_facts": []},
            [region],
            artifacts,
            client,  # type: ignore[arg-type]
            "e1",
            modality="map",
        )
    )

    assert client.descriptions == 1
    assert [fact.text for fact in facts] == ["The map shows forecast rainfall zones."]
    assert facts[0].supporting_region_ids == ["e1:r1"]


def test_rejected_initial_fact_falls_back_to_dedicated_visual_description(
    tmp_path: Path,
) -> None:
    artifacts, region = _region(tmp_path)
    client = VisualClient(initial_entailed=False)
    ingestor = PdfEvidenceIngestor(config=PdfPipelineConfig(artifact_dir=tmp_path))

    facts = asyncio.run(
        ingestor._verify_visual_facts(
            {
                "unit_description": "Rainfall forecast map",
                "visual_facts": [
                    {"text": "Unverified initial description.", "region_indexes": [0]}
                ],
            },
            [region],
            artifacts,
            client,  # type: ignore[arg-type]
            "e1",
            modality="map",
        )
    )

    assert client.descriptions == 1
    assert client.verifications == [
        "Unverified initial description.",
        "The map shows forecast rainfall zones.",
    ]
    assert len(facts) == 1


def test_visual_unit_retains_no_unverified_summary_when_verification_fails(
    tmp_path: Path,
) -> None:
    artifacts, region = _region(tmp_path)

    class RejectingClient(VisualClient):
        async def verify_visual(
            self, image_path: Path, facts: list[str]
        ) -> dict[str, Any]:
            del image_path
            return {"facts": [{"text": facts[0], "entailed": False}]}

    ingestor = PdfEvidenceIngestor(config=PdfPipelineConfig(artifact_dir=tmp_path))

    facts = asyncio.run(
        ingestor._verify_visual_facts(
            {"unit_description": "Rainfall forecast map", "visual_facts": []},
            [region],
            artifacts,
            RejectingClient(),  # type: ignore[arg-type]
            "e1",
            modality="map",
        )
    )

    assert facts == []


def test_visual_facts_for_one_crop_are_verified_in_one_batch(tmp_path: Path) -> None:
    artifacts, region = _region(tmp_path)
    client = VisualClient()
    ingestor = PdfEvidenceIngestor(config=PdfPipelineConfig(artifact_dir=tmp_path))

    facts = asyncio.run(
        ingestor._verify_visual_facts(
            {
                "unit_description": "Rainfall forecast map",
                "visual_facts": [
                    {"text": "The north is wetter.", "region_indexes": [0]},
                    {"text": "The south is drier.", "region_indexes": [0]},
                ],
            },
            [region],
            artifacts,
            client,  # type: ignore[arg-type]
            "e1",
            modality="map",
        )
    )

    assert client.verification_batches == [
        ["The north is wetter.", "The south is drier."]
    ]
    assert [fact.text for fact in facts] == [
        "The north is wetter.",
        "The south is drier.",
    ]


def test_visual_facts_are_split_into_bounded_batches(tmp_path: Path) -> None:
    artifacts, region = _region(tmp_path)
    client = VisualClient()
    ingestor = PdfEvidenceIngestor(config=PdfPipelineConfig(artifact_dir=tmp_path))
    descriptions = [f"Visual fact {index}." for index in range(6)]

    facts = asyncio.run(
        ingestor._verify_visual_facts(
            {
                "unit_description": "Rainfall forecast map",
                "visual_facts": [
                    {"text": text, "region_indexes": [0]} for text in descriptions
                ],
            },
            [region],
            artifacts,
            client,  # type: ignore[arg-type]
            "e1",
            modality="map",
        )
    )

    assert [len(batch) for batch in client.verification_batches] == [5, 1]
    assert [fact.text for fact in facts] == descriptions
