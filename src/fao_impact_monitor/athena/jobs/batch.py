"""Restart-safe jobs and the count-or-time batch gate."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol
from uuid import uuid4

from pydantic import BaseModel, Field

from fao_impact_monitor.athena.config import BatchPolicy
from fao_impact_monitor.athena.ns import GRAPH_OPERATIONS, term
from fao_impact_monitor.athena.storage.graph import GraphStore
from fao_impact_monitor.athena.storage.rdf import digest, n3_iri, n3_str, utc_stamp

logger = logging.getLogger(__name__)


class WorkItem(BaseModel):
    iri: str
    task_type: str
    enqueued_at: str
    job_iri: str | None = None
    outbox_iri: str | None = None


class JobRecord(BaseModel):
    iri: str
    task_type: str
    status: str
    payload: dict[str, object] = Field(default_factory=dict)
    result: dict[str, object] | None = None
    error: str | None = None
    checkpoint: dict[str, object] | None = None
    created_at: str
    updated_at: str


class BatchRecord(BaseModel):
    iri: str
    task_type: str
    status: str
    members: list[WorkItem]
    created_at: str


class Broker(Protocol):
    def publish(self, task_type: str, batch_id: str) -> None: ...

    def probe(self) -> None: ...


class InProcessBroker:
    """Records publications so tests can run the batch without RabbitMQ."""

    def __init__(self) -> None:
        self.published: list[tuple[str, str]] = []
        self.fail_times = 0

    def publish(self, task_type: str, batch_id: str) -> None:
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("broker unavailable")
        self.published.append((task_type, batch_id))

    def probe(self) -> None:
        return None


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value)


class BatchGate:
    """Commit work before publishing it. Flush by count or by age."""

    def __init__(
        self,
        graph: GraphStore,
        broker: Broker,
        policies: dict[str, BatchPolicy],
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._graph = graph
        self._broker = broker
        self._policies = policies
        self._clock = clock or (lambda: datetime.now(UTC))

    def submit(
        self,
        task_type: str,
        payload: dict[str, object],
        *,
        idempotency_key: str | None = None,
    ) -> str:
        if task_type not in self._policies:
            raise KeyError(f"No batch policy for {task_type}")
        iri = (
            term(f"job/{digest(idempotency_key)}")
            if idempotency_key
            else term(f"job/{uuid4().hex}")
        )
        existing = self.read_job(iri)
        if existing is not None:
            return iri
        now = self._clock().isoformat()
        job = JobRecord(
            iri=iri,
            task_type=task_type,
            status="committed",
            payload=payload,
            created_at=now,
            updated_at=now,
        )
        work = WorkItem(
            iri=term(f"work/{uuid4().hex}"),
            task_type=task_type,
            enqueued_at=now,
            job_iri=iri,
        )
        self._insert_job(job, work)
        self.flush_ready()
        return iri

    def flush_ready(self) -> list[str]:
        published: list[str] = []
        for task_type in self._policies:
            while True:
                batch_id = self._flush_type(task_type)
                if batch_id is None:
                    break
                published.append(batch_id)
        return published

    def recover(self) -> list[str]:
        """Republish batches whose broker publication never completed."""
        republished: list[str] = []
        for batch in self._batches("publish_pending"):
            self._broker.publish(batch.task_type, batch.iri)
            self._set_batch_status(batch.iri, "queued")
            republished.append(batch.iri)
        republished.extend(self.flush_ready())
        return republished

    def read_job(self, iri: str) -> JobRecord | None:
        rows = self._graph.query(
            f"""
            SELECT ?json WHERE {{
              GRAPH {n3_iri(GRAPH_OPERATIONS)} {{
                {n3_iri(iri)} {n3_iri(term("recordJson"))} ?json .
              }}
            }}
            """
        )
        if not rows:
            return None
        return JobRecord.model_validate_json(rows[0]["json"])

    def read_batch(self, iri: str) -> BatchRecord | None:
        rows = self._graph.query(
            f"""
            SELECT ?json WHERE {{
              GRAPH {n3_iri(GRAPH_OPERATIONS)} {{
                {n3_iri(iri)} {n3_iri(term("recordJson"))} ?json .
              }}
            }}
            """
        )
        if not rows:
            return None
        return BatchRecord.model_validate_json(rows[0]["json"])

    def mark_job(
        self,
        iri: str,
        status: str,
        *,
        result: dict[str, object] | None = None,
        error: str | None = None,
        checkpoint: dict[str, object] | None = None,
    ) -> None:
        current = self.read_job(iri)
        if current is None:
            raise KeyError(iri)
        updated = current.model_copy(
            update={
                "status": status,
                "updated_at": utc_stamp(),
                "result": current.result if result is None else result,
                "error": error if error is not None else current.error,
                "checkpoint": current.checkpoint if checkpoint is None else checkpoint,
            }
        )
        self._replace_json(iri, updated.model_dump_json(), "Job")

    def mark_batch_done(self, iri: str) -> None:
        self._set_batch_status(iri, "done")

    def mark_batch_running(self, iri: str) -> None:
        self._set_batch_status(iri, "running")

    def _flush_type(self, task_type: str) -> str | None:
        pending = self._pending(task_type)
        if not pending:
            return None
        policy = self._policies[task_type]
        oldest = _parse_time(pending[0].enqueued_at)
        age = (self._clock() - oldest).total_seconds()
        if len(pending) >= policy.min_batch_size:
            chosen = pending[: policy.min_batch_size]
        elif age >= policy.max_wait_seconds:
            chosen = pending
        else:
            return None
        batch = BatchRecord(
            iri=term(f"batch/{uuid4().hex}"),
            task_type=task_type,
            status="publish_pending",
            members=chosen,
            created_at=self._clock().isoformat(),
        )
        self._commit_batch(batch)
        try:
            self._broker.publish(task_type, batch.iri)
        except Exception:
            logger.exception("Broker publication failed for %s", batch.iri)
            return batch.iri
        self._set_batch_status(batch.iri, "queued")
        for member in chosen:
            if member.job_iri is not None:
                self.mark_job(member.job_iri, "queued")
        return batch.iri

    def _pending(self, task_type: str) -> list[WorkItem]:
        rows = self._graph.query(
            f"""
            SELECT ?item ?job ?outbox ?at WHERE {{
              GRAPH {n3_iri(GRAPH_OPERATIONS)} {{
                ?item a {n3_iri(term("PendingWork"))} ;
                      {n3_iri(term("taskType"))} {n3_str(task_type)} ;
                      {n3_iri(term("enqueuedAt"))} ?at .
                OPTIONAL {{ ?item {n3_iri(term("job"))} ?job }}
                OPTIONAL {{ ?item {n3_iri(term("outbox"))} ?outbox }}
              }}
            }}
            ORDER BY ?at
            """
        )
        return [
            WorkItem(
                iri=row["item"],
                task_type=task_type,
                enqueued_at=row["at"],
                job_iri=row["job"] or None,
                outbox_iri=row["outbox"] or None,
            )
            for row in rows
        ]

    def _batches(self, status: str) -> list[BatchRecord]:
        rows = self._graph.query(
            f"""
            SELECT ?json WHERE {{
              GRAPH {n3_iri(GRAPH_OPERATIONS)} {{
                ?batch a {n3_iri(term("Batch"))} ;
                       {n3_iri(term("batchStatus"))} {n3_str(status)} ;
                       {n3_iri(term("recordJson"))} ?json .
              }}
            }}
            """
        )
        return [BatchRecord.model_validate_json(row["json"]) for row in rows]

    def _insert_job(self, job: JobRecord, work: WorkItem) -> None:
        job_json = n3_str(job.model_dump_json())
        self._graph.update(
            f"""
            INSERT DATA {{
              GRAPH {n3_iri(GRAPH_OPERATIONS)} {{
                {n3_iri(job.iri)} a {n3_iri(term("Job"))} ;
                    {n3_iri(term("taskType"))} {n3_str(job.task_type)} ;
                    {n3_iri(term("jobStatus"))} {n3_str(job.status)} ;
                    {n3_iri(term("recordJson"))} {job_json} .
                {n3_iri(work.iri)} a {n3_iri(term("PendingWork"))} ;
                    {n3_iri(term("taskType"))} {n3_str(work.task_type)} ;
                    {n3_iri(term("job"))} {n3_iri(work.job_iri or job.iri)} ;
                    {n3_iri(term("enqueuedAt"))} {n3_str(work.enqueued_at)} .
              }}
            }}
            """
        )

    def _commit_batch(self, batch: BatchRecord) -> None:
        deletes = "\n".join(
            f"""
            DELETE {{ GRAPH {n3_iri(GRAPH_OPERATIONS)} {{ {n3_iri(member.iri)} ?p ?o }} }}
            WHERE {{ GRAPH {n3_iri(GRAPH_OPERATIONS)} {{ {n3_iri(member.iri)} ?p ?o }} }} ;
            """
            for member in batch.members
        )
        members = "\n".join(
            f"{n3_iri(batch.iri)} {n3_iri(term('member'))} {n3_iri(member.iri)} ."
            for member in batch.members
        )
        self._graph.update(
            deletes
            + f"""
            INSERT DATA {{
              GRAPH {n3_iri(GRAPH_OPERATIONS)} {{
                {n3_iri(batch.iri)} a {n3_iri(term("Batch"))} ;
                    {n3_iri(term("taskType"))} {n3_str(batch.task_type)} ;
                    {n3_iri(term("batchStatus"))} {n3_str(batch.status)} ;
                    {n3_iri(term("recordJson"))} {n3_str(batch.model_dump_json())} .
                {members}
              }}
            }}
            """
        )

    def _set_batch_status(self, iri: str, status: str) -> None:
        current = self.read_batch(iri)
        if current is None:
            raise KeyError(iri)
        updated = current.model_copy(update={"status": status})
        self._replace_json(iri, updated.model_dump_json(), "Batch")
        self._graph.update(
            f"""
            DELETE {{
              GRAPH {n3_iri(GRAPH_OPERATIONS)} {{
                {n3_iri(iri)} {n3_iri(term("batchStatus"))} ?status
              }}
            }}
            WHERE {{
              GRAPH {n3_iri(GRAPH_OPERATIONS)} {{
                {n3_iri(iri)} {n3_iri(term("batchStatus"))} ?status
              }}
            }};
            INSERT DATA {{
              GRAPH {n3_iri(GRAPH_OPERATIONS)} {{
                {n3_iri(iri)} {n3_iri(term("batchStatus"))} {n3_str(status)}
              }}
            }}
            """
        )

    def _replace_json(self, iri: str, payload: str, rdf_type: str) -> None:
        self._graph.update(
            f"""
            DELETE {{
              GRAPH {n3_iri(GRAPH_OPERATIONS)} {{
                {n3_iri(iri)} {n3_iri(term("recordJson"))} ?json
              }}
            }}
            WHERE {{
              GRAPH {n3_iri(GRAPH_OPERATIONS)} {{
                {n3_iri(iri)} {n3_iri(term("recordJson"))} ?json
              }}
            }};
            INSERT DATA {{
              GRAPH {n3_iri(GRAPH_OPERATIONS)} {{
                {n3_iri(iri)} a {n3_iri(term(rdf_type))} ;
                    {n3_iri(term("recordJson"))} {n3_str(payload)} .
              }}
            }}
            """
        )
