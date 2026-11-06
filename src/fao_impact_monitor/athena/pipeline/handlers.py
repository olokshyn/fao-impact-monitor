"""Run a committed batch outside the API process."""

from __future__ import annotations

import logging

from fao_impact_monitor.athena.context import AthenaContext
from fao_impact_monitor.athena.jobs.batch import JobRecord
from fao_impact_monitor.athena.model.filters import FilterSpec
from fao_impact_monitor.athena.model.records import ClaimRecord, ResearchResult
from fao_impact_monitor.athena.pipeline.backfill import run_backfill
from fao_impact_monitor.athena.pipeline.discover import discover_for_claim
from fao_impact_monitor.athena.pipeline.ingest import (
    RawClaim,
    StoredDocument,
    _verify_claim,
    ingest_document,
    ingest_raw_claim,
    text_from_bytes,
)
from fao_impact_monitor.athena.pipeline.load import load_evidence
from fao_impact_monitor.athena.pipeline.maintain import maintain
from fao_impact_monitor.athena.pipeline.research import run_research
from fao_impact_monitor.athena.storage.ontology import tag_registry
from fao_impact_monitor.athena.storage.outbox import publish_outbox
from fao_impact_monitor.athena.storage.rdf import (
    knowledge_graph,
    read_model,
    write_record,
)

logger = logging.getLogger(__name__)


async def process_batch(ctx: AthenaContext, batch_id: str) -> None:
    batch = ctx.gate.read_batch(batch_id)
    if batch is None or batch.status == "done":
        return
    ctx.gate.mark_batch_running(batch_id)
    errors: list[str] = []
    for member in batch.members:
        try:
            if batch.task_type == "index_sync":
                if member.outbox_iri:
                    await publish_outbox(ctx, member.outbox_iri)
                continue
            if member.job_iri is None:
                continue
            job = ctx.gate.read_job(member.job_iri)
            if job is None or job.status == "succeeded":
                continue
            ctx.gate.mark_job(job.iri, "running")
            result = await _dispatch(ctx, batch.task_type, job)
            ctx.gate.mark_job(job.iri, "succeeded", result=result)
        except Exception as exc:
            logger.exception("Athena batch member failed")
            errors.append(str(exc))
            if member.job_iri is not None:
                ctx.gate.mark_job(member.job_iri, "failed", error=str(exc))
    if errors:
        raise RuntimeError("; ".join(errors))
    ctx.gate.mark_batch_done(batch_id)
    ctx.gate.flush_ready()


async def _dispatch(
    ctx: AthenaContext, task_type: str, job: JobRecord
) -> dict[str, object]:
    if task_type == "ingest":
        return await _ingest(ctx, job)
    if task_type == "discover":
        created = await discover_for_claim(ctx, str(job.payload["claim_iri"]))
        return {"assertions": created}
    if task_type == "verify":
        return await _reverify(ctx, str(job.payload["target_iri"]))
    if task_type == "backfill":
        return await run_backfill(ctx, str(job.payload["relation_iri"]))
    if task_type == "research":
        spec = FilterSpec.model_validate(job.payload.get("filters") or {})
        if job.payload.get("transferability"):
            spec = spec.model_copy(update={"transferability": True})
        rounds = int(str(job.payload.get("rounds") or ctx.config.research_max_rounds))
        result = await run_research(
            ctx,
            str(job.payload["question"]),
            spec,
            rounds=rounds,
            external=bool(job.payload.get("external")),
        )
        return result.model_dump(mode="json")
    if task_type == "maintain":
        changed = job.payload.get("changed_iris")
        iris = [str(item) for item in changed] if isinstance(changed, list) else []
        report = maintain(ctx, iris)
        return {key: value for key, value in report.items()}
    raise KeyError(task_type)


async def _ingest(ctx: AthenaContext, job: JobRecord) -> dict[str, object]:
    documents = job.payload.get("documents")
    claims = job.payload.get("claims")
    ids: list[str] = []
    if isinstance(documents, list):
        for item in documents:
            ids.extend(await ingest_document(ctx, StoredDocument.model_validate(item)))
            ctx.gate.mark_job(job.iri, "running", checkpoint={"claim_ids": ids})
    if isinstance(claims, list):
        for item in claims:
            ids.extend(await ingest_raw_claim(ctx, RawClaim.model_validate(item)))
    return {"claim_ids": ids}


async def _reverify(ctx: AthenaContext, target_iri: str) -> dict[str, object]:
    claim = read_model(ctx.graph, target_iri, ClaimRecord)
    if claim is None or not claim.evidence_iris:
        return {"status": "missing"}
    evidence = load_evidence(ctx.graph).get(claim.evidence_iris[0])
    if evidence is None:
        return {"status": "missing"}
    from fao_impact_monitor.athena.model.records import SourceRecord

    source = read_model(ctx.graph, evidence.source_iri, SourceRecord)
    if source is None:
        return {"status": "missing"}
    data = ctx.objects.get(source.snapshot.logical_id, version=source.snapshot.version)
    text = text_from_bytes(data, source.snapshot.media_type)
    updated = await _verify_claim(
        ctx,
        claim,
        evidence,
        text,
        source.version,
        tag_registry(ctx.graph),
    )
    write_record(
        ctx.graph,
        updated,
        knowledge_graph(updated.scientific_status, updated.verification_status),
        index=True,
    )
    return {"status": updated.verification_status.value}


def research_result(payload: dict[str, object]) -> ResearchResult:
    return ResearchResult.model_validate(payload)
