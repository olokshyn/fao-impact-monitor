from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from fao_impact_monitor.pdf_pipeline.ingest import (
    _event_contexts,
    _events_from_source_text,
    _reporting_modes,
)
from fao_impact_monitor.pdf_pipeline.models import (
    ArtifactRef,
    CountryContext,
    EventContext,
    EvidenceUnit,
    ModelVersions,
    PromptVersions,
    SourceRegion,
    ValidationResult,
    effective_events,
    is_searchable,
)


def _region(text: str = "exact source text") -> SourceRegion:
    import hashlib

    return SourceRegion(
        region_id="r1",
        physical_page=2,
        page_width_points=600,
        page_height_points=800,
        extracted_text=text,
        extracted_text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        page_artifact=ArtifactRef(
            relative_path="abc/pages/page-0002.png",
            sha256="a" * 64,
            media_type="image/png",
        ),
    )


def _validation() -> ValidationResult:
    return ValidationResult(
        verdict="passed", checked_at=datetime.now(UTC), validator="test"
    )


def test_effective_event_direct_negation_wins_and_is_not_searchable() -> None:
    events = effective_events(
        [
            EventContext(
                event_id="el_nino_2023_24",
                relationship="explicitly_not_attributed",
                origin="explicit",
            )
        ],
        [
            EventContext(
                event_id="el_nino_2023_24",
                relationship="associated",
                origin="inherited",
            )
        ],
    )
    assert events[0].relationship == "explicitly_not_attributed"
    assert is_searchable(events) == (False, "explicitly_not_attributed")


def test_conflicting_inherited_context_becomes_uncertain() -> None:
    events = effective_events(
        [],
        [
            EventContext(
                event_id="el_nino_2026_27",
                relationship="associated",
                origin="inherited",
            ),
            EventContext(
                event_id="el_nino_2026_27", relationship="unrelated", origin="inherited"
            ),
        ],
    )
    assert events[0].relationship == "uncertain"
    assert is_searchable(events)[0] is False


def test_single_inherited_event_is_marked_inherited() -> None:
    events = effective_events(
        [],
        [
            EventContext(
                event_id="el_nino_2023_24",
                relationship="associated",
                origin="explicit",
                supporting_evidence_ids=["scope-1"],
            )
        ],
    )
    assert events[0].origin == "inherited"
    assert events[0].supporting_evidence_ids == ["scope-1"]


def test_event_context_normalizes_gemini_year_alias() -> None:
    events = _event_contexts(
        [{"event_id": "El Niño 2023–24", "relationship": "associated"}]
    )
    assert [event.event_id for event in events] == ["el_nino_2023_24"]


def test_explicit_el_nino_year_recovers_omitted_event() -> None:
    events = _events_from_source_text(
        "The conjunction of an El Niño and a positive Indian Ocean Dipole "
        "in the third quarter of 2023 will result in above-average rainfall.",
        document_events=[],
        supporting_evidence_id="scope-1",
    )
    assert [(event.event_id, event.relationship) for event in events] == [
        ("el_nino_2023_24", "attributed")
    ]


def test_generic_el_nino_scope_uses_single_document_episode() -> None:
    events = _events_from_source_text(
        "An approaching El Niño climate event has the potential to negatively "
        "affect 1.2 million people in Somalia this year.",
        document_events=[
            EventContext(
                event_id="el_nino_2023_24",
                relationship="associated",
                origin="explicit",
            )
        ],
        supporting_evidence_id="scope-1",
    )
    assert events[0].event_id == "el_nino_2023_24"
    assert events[0].supporting_evidence_ids == ["scope-1"]
    assert is_searchable(events)[0] is True


def test_evidence_rejects_paraphrase_as_source_text() -> None:
    with pytest.raises(ValidationError, match="assembled exactly"):
        EvidenceUnit(
            evidence_id="e1",
            document_id="64b64c1f0efbde5b4f7e7777",
            document_sha256="a" * 64,
            section_id="s1",
            evidence_kind="agrifood_impact",
            assertion_mode="observed",
            source_modality="text",
            source_regions=[_region()],
            physical_pages=[2],
            source_text="a model paraphrase",
            unit_description="generated description",
            retrieval_text="generated retrieval text",
            countries=[CountryContext(iso3="ZMB", role="subject", origin="inherited")],
            events=[
                EventContext(
                    event_id="el_nino_2023_24",
                    relationship="associated",
                    origin="inherited",
                )
            ],
            canonical_evidence_text="[TARGET SOURCE EVIDENCE]\nexact source text",
            searchable=True,
            extraction_validation=_validation(),
            eligibility_validation=_validation(),
            model_versions=ModelVersions(gemini="g", luna="l", titan="t"),
            prompt_versions=PromptVersions(),
            created_at=datetime.now(UTC),
        )


def test_reporting_modes_drop_non_assertion_labels() -> None:
    assert _reporting_modes(["forecast", "anticipatory_action", "early_warning"]) == {
        "forecast"
    }
