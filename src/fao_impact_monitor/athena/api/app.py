"""HTTP API for ingestion, research, jobs, relations, and evidence."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from fao_impact_monitor.athena.context import AthenaContext
from fao_impact_monitor.athena.model.filters import FilterExpr, FilterSpec
from fao_impact_monitor.athena.model.records import EvidenceRecord
from fao_impact_monitor.athena.pipeline.ingest import RawClaim, StoredDocument
from fao_impact_monitor.athena.pipeline.load import load_claims, load_evidence
from fao_impact_monitor.athena.retrieval.relations import query_relations
from fao_impact_monitor.athena.storage.objects import sha256_bytes
from fao_impact_monitor.athena.storage.ontology import LocationHierarchy
from fao_impact_monitor.athena.storage.rdf import read_model


class DocumentInput(BaseModel):
    filename: str | None = None
    media_type: str = "text/plain"
    text: str | None = None
    content_base64: str | None = None
    external_id: str | None = None
    url: str | None = None
    doi: str | None = None
    title: str | None = None
    tags: dict[str, list[str]] = Field(default_factory=dict)
    metadata: dict[str, str] = Field(default_factory=dict)


class IngestionBatch(BaseModel):
    documents: list[DocumentInput] = Field(default_factory=list)
    claims: list[RawClaim] = Field(default_factory=list)
    idempotency_key: str | None = None


class ResearchBody(BaseModel):
    question: str
    filters: FilterSpec = Field(default_factory=FilterSpec)
    rounds: int | None = None
    transferability: bool = False
    external: bool = False
    idempotency_key: str | None = None


def create_app(ctx: AthenaContext | None = None) -> FastAPI:
    app = FastAPI(title="Athena")
    if ctx is not None:
        app.state.ctx = ctx

    @app.post("/ingestions", status_code=202)
    def create_ingestion(body: IngestionBatch, request: Request) -> dict[str, str]:
        services = _ctx(request)
        if not body.documents and not body.claims:
            raise HTTPException(status_code=400, detail="Batch is empty")
        documents = [_store_document(services, document) for document in body.documents]
        job_id = services.gate.submit(
            "ingest",
            {
                "documents": [
                    document.model_dump(mode="json") for document in documents
                ],
                "claims": [claim.model_dump(mode="json") for claim in body.claims],
            },
            idempotency_key=body.idempotency_key,
        )
        return {"job_id": job_id}

    @app.post("/research", status_code=202)
    def create_research(body: ResearchBody, request: Request) -> dict[str, str]:
        services = _ctx(request)
        spec = body.filters.model_copy(update={"transferability": body.transferability})
        job_id = services.gate.submit(
            "research",
            {
                "question": body.question,
                "filters": spec.model_dump(mode="json"),
                "rounds": body.rounds or services.config.research_max_rounds,
                "transferability": body.transferability,
                "external": body.external,
            },
            idempotency_key=body.idempotency_key,
        )
        return {"job_id": job_id}

    @app.get("/jobs/{job_id:path}")
    def read_job(job_id: str, request: Request) -> dict[str, object]:
        job = _ctx(request).gate.read_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return job.model_dump(mode="json")

    @app.get("/claims/{claim_id:path}/relations")
    def read_relations(
        claim_id: str,
        request: Request,
        parent: str,
        direction: str = "both",
        strict: bool = True,
        transferability: bool = False,
        filters: str | None = None,
    ) -> dict[str, object]:
        services = _ctx(request)
        spec = FilterSpec(strict=strict, transferability=transferability)
        if filters:
            spec = spec.model_copy(
                update={"root": FilterExpr.model_validate_json(filters)}
            )
        hierarchy = LocationHierarchy.from_graph(services.graph)
        views = query_relations(
            services.graph,
            claim_id,
            parent,
            direction,
            spec,
            hierarchy.ancestors,
            load_claims(services.graph),
            load_evidence(services.graph),
        )
        return {
            "relations": [
                {
                    "iri": view.assertion.iri,
                    "subject": view.assertion.subject_iri,
                    "object": view.assertion.object_iri,
                    "relation": view.assertion.relation_iri,
                    "direction": view.direction,
                    "extrapolation": view.extrapolation,
                    "scientific_status": view.assertion.scientific_status.value,
                    "evidence": view.assertion.evidence_iris,
                }
                for view in views
            ]
        }

    @app.get("/evidence/{evidence_id:path}")
    def read_evidence(evidence_id: str, request: Request) -> dict[str, object]:
        evidence = read_model(_ctx(request).graph, evidence_id, EvidenceRecord)
        if evidence is None:
            raise HTTPException(status_code=404, detail="Evidence not found")
        return evidence.model_dump(mode="json")

    @app.get("/health")
    def health(request: Request) -> dict[str, object]:
        services = _ctx(request)
        checks: dict[str, str] = {}
        probes: list[tuple[str, Callable[[], object]]] = [
            ("fuseki", lambda: services.graph.ask("ASK { ?s ?p ?o }")),
            ("objects", services.objects.probe),
            ("broker", services.broker.probe),
        ]
        for name, probe in probes:
            try:
                probe()
                checks[name] = "ok"
            except Exception as exc:  # noqa: BLE001 - health reports any dependency failure
                checks[name] = str(exc)
        status = "ok" if all(value == "ok" for value in checks.values()) else "degraded"
        return {"status": status, "checks": checks}

    return app


def _ctx(request: Request) -> AthenaContext:
    return cast(AthenaContext, request.app.state.ctx)


def _store_document(ctx: AthenaContext, document: DocumentInput) -> StoredDocument:
    if document.content_base64:
        import base64

        data = base64.b64decode(document.content_base64)
    elif document.text is not None:
        data = document.text.encode("utf-8")
    else:
        raise HTTPException(
            status_code=400, detail="Document needs text or content_base64"
        )
    digest = sha256_bytes(data)
    ref = ctx.objects.put(f"upload/{digest}", data, media_type=document.media_type)
    return StoredDocument(
        logical_id=ref.logical_id,
        version=ref.version,
        sha256=ref.sha256,
        media_type=ref.media_type,
        external_id=document.external_id,
        url=document.url,
        doi=document.doi,
        title=document.title or document.filename,
        tags=document.tags,
        metadata=document.metadata,
    )
