from __future__ import annotations

from datetime import timedelta

from fao_impact_monitor.athena.jobs.batch import InProcessBroker
from fao_impact_monitor.athena.model.records import (
    ClaimRecord,
    EvidenceRecord,
    SourceRecord,
)
from fao_impact_monitor.athena.model.status import ScientificStatus, VerificationStatus
from fao_impact_monitor.athena.ns import GRAPH_ACCEPTED, relation
from fao_impact_monitor.athena.storage.objects import LocalObjectStore, copy_objects
from fao_impact_monitor.athena.storage.ontology import (
    LocationHierarchy,
    is_descendant,
    tag_registry,
)
from fao_impact_monitor.athena.storage.rdf import (
    claim_iri,
    evidence_iri,
    source_iri,
    write_record,
)
from tests.athena.support import Clock, make_context


def test_ontology_seed_has_hierarchy_tags_and_locations(tmp_path: object) -> None:
    from pathlib import Path

    ctx = make_context(Path(str(tmp_path)))
    registry = tag_registry(ctx.graph)
    assert registry["country"].inherits is False
    assert registry["acquisition_channel"].inherits is True
    assert "web-page" in registry["source_type"].vocabulary
    assert "web-search" in registry["acquisition_channel"].vocabulary
    assert is_descendant(ctx.graph, relation("yieldLoss"), relation("causes"))
    assert is_descendant(ctx.graph, relation("increases"), relation("causes"))
    assert not is_descendant(ctx.graph, relation("correlatesWith"), relation("causes"))
    ancestors = LocationHierarchy.from_graph(ctx.graph).ancestors("KEN")
    assert ancestors == {"eastern-africa"}


def test_claim_write_is_idempotent(tmp_path: object) -> None:
    from pathlib import Path

    ctx = make_context(Path(str(tmp_path)))
    evidence = EvidenceRecord(
        iri=evidence_iri("hash", "p1", "Maize yield fell."),
        excerpt="Maize yield fell.",
        source_iri=source_iri("hash"),
        source_version="1",
    )
    claim = ClaimRecord(
        iri=claim_iri("yield fell", [evidence.iri], [], "KEN", "2024", None),
        original_wording="Maize yield fell.",
        normalized="yield fell",
        evidence_iris=[evidence.iri],
        concept_iri="https://fao.org/athena/concept/yield",
        concept_label="yield",
        scientific_status=ScientificStatus.SOURCE_REPORTED,
        verification_status=VerificationStatus.VERIFIED,
        tags={"country": ["KEN"]},
    )
    write_record(ctx.graph, claim, GRAPH_ACCEPTED, index=False)
    write_record(ctx.graph, claim, GRAPH_ACCEPTED, index=False)
    rows = ctx.graph.query(
        "SELECT ?json WHERE { GRAPH ?g { ?s <https://fao.org/athena/recordJson> ?json } }"
    )
    assert len(rows) == 1


def test_batch_waits_for_count_or_age(tmp_path: object) -> None:
    from pathlib import Path

    clock = Clock()
    broker = InProcessBroker()
    ctx = make_context(
        Path(str(tmp_path)), batch_size=3, wait_seconds=30, clock=clock, broker=broker
    )
    first = ctx.gate.submit("ingest", {"documents": []})
    second = ctx.gate.submit("verify", {"target_iri": "x"})
    assert broker.published == []
    assert ctx.gate.read_job(first) is not None
    assert ctx.gate.read_job(second) is not None
    clock.now += timedelta(seconds=31)
    flushed = ctx.gate.flush_ready()
    assert len(flushed) == 2
    assert {task for task, _batch in broker.published} == {"ingest", "verify"}


def test_failed_publish_can_be_recovered(tmp_path: object) -> None:
    from pathlib import Path

    broker = InProcessBroker()
    broker.fail_times = 1
    ctx = make_context(Path(str(tmp_path)), broker=broker)
    ctx.gate.submit("research", {"question": "q", "filters": {}})
    assert broker.published == []
    recovered = ctx.gate.recover()
    assert recovered
    assert broker.published


def test_object_copy_keeps_hashes_and_old_generations(tmp_path: object) -> None:
    from pathlib import Path

    root = Path(str(tmp_path))
    source = LocalObjectStore(root / "a")
    dest = LocalObjectStore(root / "b")
    source.put("upload/doc", b"one", media_type="text/plain")
    updated = source.put("upload/doc", b"two", media_type="text/plain")
    copied = copy_objects(source, dest)
    assert copied == 2
    assert dest.get(updated.logical_id, version=updated.version) == b"two"
    assert dest.stat(updated.logical_id).sha256 == updated.sha256


def test_source_record_round_trip(tmp_path: object) -> None:
    from pathlib import Path

    ctx = make_context(Path(str(tmp_path)))
    source = SourceRecord(
        iri=source_iri("abc"),
        snapshot=__import__(
            "fao_impact_monitor.athena.model.records",
            fromlist=["ObjectPointer"],
        ).ObjectPointer(
            logical_id="upload/abc",
            sha256="abc",
            version="1",
            media_type="text/plain",
        ),
        content_hash="abc",
        version="1",
        tags={"source_type": ["peer-reviewed"], "acquisition_channel": ["web-search"]},
    )
    write_record(ctx.graph, source, GRAPH_ACCEPTED, index=False)
    loaded = __import__(
        "fao_impact_monitor.athena.storage.rdf",
        fromlist=["read_model"],
    ).read_model(ctx.graph, source.iri, SourceRecord)
    assert loaded is not None
    assert loaded.tags["source_type"] == ["peer-reviewed"]
    assert loaded.tags["acquisition_channel"] == ["web-search"]
