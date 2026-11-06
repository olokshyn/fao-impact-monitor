from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi.testclient import TestClient

from fao_impact_monitor.athena.api.app import create_app
from fao_impact_monitor.athena.eval.metrics import candidate_recall, relation_precision
from fao_impact_monitor.athena.model.filters import FilterSpec
from fao_impact_monitor.athena.model.records import ClaimRecord, EvidenceRecord
from fao_impact_monitor.athena.model.status import ScientificStatus, VerificationStatus
from fao_impact_monitor.athena.ns import GRAPH_ACCEPTED
from fao_impact_monitor.athena.pipeline.handlers import process_batch
from fao_impact_monitor.athena.retrieval.candidates import retrieve_candidates
from fao_impact_monitor.athena.storage.index import RetrievalRecord
from fao_impact_monitor.athena.storage.outbox import publish_outbox
from fao_impact_monitor.athena.storage.rdf import write_record
from tests.athena.support import make_context, published_batches


def test_api_accepts_a_batch_and_returns_a_job(tmp_path: Path) -> None:
    ctx = make_context(tmp_path)
    client = TestClient(create_app(ctx))
    created = client.post(
        "/ingestions",
        json={
            "documents": [
                {
                    "text": "Drought decreased maize yield in Kenya.",
                    "tags": {"country": ["KEN"], "source_type": ["peer-reviewed"]},
                }
            ],
            "claims": [
                {
                    "wording": "Irrigation increased vegetable supply.",
                    "excerpt": "Irrigation increased vegetable supply.",
                    "tags": {"country": ["KEN"]},
                }
            ],
        },
    )
    assert created.status_code == 202
    job_id = created.json()["job_id"]
    assert client.get(f"/jobs/{job_id}").json()["status"] == "queued"
    assert client.post("/ingestions", json={}).status_code == 400
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["checks"]["fuseki"] == "ok"
    task_type, batch_id = published_batches(ctx)[0]
    assert task_type == "ingest"
    asyncio.run(process_batch(ctx, batch_id))
    finished = client.get(f"/jobs/{job_id}").json()
    assert finished["status"] == "succeeded"
    claim_ids = finished["result"]["claim_ids"]
    assert len(claim_ids) == 2
    evidence_id = None
    from fao_impact_monitor.athena.pipeline.load import load_claims

    claim = load_claims(ctx.graph)[claim_ids[0]]
    evidence_id = claim.evidence_iris[0]
    evidence = client.get(f"/evidence/{evidence_id}")
    assert evidence.status_code == 200
    assert "Drought decreased maize yield" in evidence.json()["excerpt"]


def test_retrieval_recall_and_relation_precision(tmp_path: Path) -> None:
    ctx = make_context(tmp_path)
    evidence = EvidenceRecord(
        iri="https://fao.org/athena/evidence/e",
        excerpt="soil biota",
        source_iri="https://fao.org/athena/source/s",
        source_version="1",
        tags={"country": ["KEN"]},
    )
    write_record(ctx.graph, evidence, GRAPH_ACCEPTED, index=False)

    def claim(local: str, entities: list[str]) -> ClaimRecord:
        return ClaimRecord(
            iri=f"https://fao.org/athena/claim/{local}",
            original_wording=local,
            normalized=local,
            entities=entities,
            evidence_iris=[evidence.iri],
            concept_iri="https://fao.org/athena/concept/soil",
            concept_label="soil",
            scientific_status=ScientificStatus.SOURCE_REPORTED,
            verification_status=VerificationStatus.VERIFIED,
            tags={"country": ["KEN"]},
        )

    soil = claim("microplastic-soil-biota", ["microplastic", "soil biota"])
    food = claim("soil-biota-food", ["soil biota", "food availability"])
    unrelated = claim("gdp", ["gdp"])
    for record in (soil, food, unrelated):
        write_record(ctx.graph, record, GRAPH_ACCEPTED, index=True)
    rows = ctx.graph.query(
        """
        SELECT ?event WHERE {
          GRAPH <https://fao.org/athena/graph/operations> {
            ?event a <https://fao.org/athena/OutboxEvent> ;
                   <https://fao.org/athena/status> "pending" .
          }
        }
        """
    )
    for row in rows:
        asyncio.run(publish_outbox(ctx, row["event"]))
    retrieved = asyncio.run(
        retrieve_candidates(
            ctx.graph,
            ctx.index,
            ctx.embedder,
            query="microplastic soil biota",
            entities=["microplastic", "soil biota"],
            spec=FilterSpec(root=None),
            evidence={evidence.iri: evidence},
            ancestors_of=lambda _code: set(),
            limit=5,
            seed_claim=soil.iri,
        )
    )
    recall = candidate_recall(
        {item.iri for item in retrieved},
        {soil.iri, food.iri},
    )
    precision = relation_precision(
        {
            ("microplastic", "causes", "soil biota"),
            ("microplastic", "correlatesWith", "gdp"),
        },
        {("microplastic", "causes", "soil biota")},
    )
    assert recall == 1.0
    assert precision == 0.5


def test_stale_index_tags_cannot_bypass_the_filter(tmp_path: Path) -> None:
    ctx = make_context(tmp_path)
    evidence = EvidenceRecord(
        iri="https://fao.org/athena/evidence/e",
        excerpt="Ethiopia only",
        source_iri="https://fao.org/athena/source/s",
        source_version="1",
        tags={"country": ["ETH"]},
    )
    claim = ClaimRecord(
        iri="https://fao.org/athena/claim/eth",
        original_wording="Ethiopia only",
        normalized="Ethiopia only",
        entities=["drought"],
        evidence_iris=[evidence.iri],
        concept_iri="https://fao.org/athena/concept/drought",
        concept_label="drought",
        scientific_status=ScientificStatus.SOURCE_REPORTED,
        verification_status=VerificationStatus.VERIFIED,
        tags={"country": ["ETH"]},
    )
    write_record(ctx.graph, evidence, GRAPH_ACCEPTED, index=False)
    write_record(ctx.graph, claim, GRAPH_ACCEPTED, index=False)
    stored = read_model_version(ctx, claim.iri)
    asyncio.run(
        ctx.index.upsert(
            RetrievalRecord(
                jena_id=claim.iri,
                record_type="claim",
                text=claim.normalized,
                embedding=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                entities=claim.entities,
                tags={"country": ["KEN"]},
                version=stored + "-stale",
                verification_status="verified",
                scientific_status="source-reported",
                graph=GRAPH_ACCEPTED,
            )
        )
    )
    from fao_impact_monitor.athena.model.filters import FilterExpr

    found = asyncio.run(
        retrieve_candidates(
            ctx.graph,
            ctx.index,
            ctx.embedder,
            query="drought",
            entities=["drought"],
            spec=FilterSpec(root=FilterExpr(op="tag", field="country", values=["KEN"])),
            evidence={evidence.iri: evidence},
            ancestors_of=lambda _code: set(),
            limit=5,
        )
    )
    assert found == []


def read_model_version(ctx: object, iri: str) -> str:
    from fao_impact_monitor.athena.context import AthenaContext
    from fao_impact_monitor.athena.storage.rdf import read_model

    assert isinstance(ctx, AthenaContext)
    loaded = read_model(ctx.graph, iri, ClaimRecord)
    assert loaded is not None
    return loaded.record_version
