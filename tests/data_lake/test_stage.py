from datetime import UTC, datetime

from fao_impact_monitor.data_lake.stage import StageVersion


class IngestStageVersion(StageVersion):
    class Settings:
        class_id_value = "ingest"


def test_stage_version_name_and_version_id() -> None:
    created_at = datetime(2026, 1, 15, tzinfo=UTC)
    doc = IngestStageVersion.model_construct(
        version_id="hash-abc",
        created_at=created_at,
    )

    assert doc.name == "ingest"
    assert doc.version_id == "hash-abc"
    assert doc.model_dump() == {
        "id": None,
        "version_id": "hash-abc",
        "created_at": created_at,
    }


def test_stage_version_has_unique_version_id_index() -> None:
    field = IngestStageVersion.model_fields["version_id"]
    assert any(
        getattr(meta, "_indexed", None) == (1, {"unique": True})
        for meta in field.metadata
    )


def test_stage_version_created_at_defaults_to_aware_utc() -> None:
    field = IngestStageVersion.model_fields["created_at"]
    assert field.default_factory is not None
    value = field.default_factory()
    assert isinstance(value, datetime)
    assert value.tzinfo is not None
