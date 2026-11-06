"""Worker entrypoints. Publishing happens after the job record is committed."""

from __future__ import annotations

import asyncio

from fao_impact_monitor.athena.jobs.celery_app import celery_app


@celery_app.task(name="athena.run_batch", bind=True, max_retries=3)  # type: ignore[untyped-decorator]
def run_batch(self: object, task_type: str, batch_id: str) -> None:
    del task_type
    try:
        asyncio.run(_process(batch_id))
    except Exception as exc:
        raise self.retry(exc=exc, countdown=10) from exc  # type: ignore[attr-defined]


@celery_app.task(name="athena.flush_batches")  # type: ignore[untyped-decorator]
def flush_batches() -> list[str]:
    from fao_impact_monitor.athena.runtime import build_context

    return build_context().gate.flush_ready()


@celery_app.task(name="athena.recover_jobs")  # type: ignore[untyped-decorator]
def recover_jobs() -> list[str]:
    from fao_impact_monitor.athena.runtime import build_context

    return build_context().gate.recover()


async def _process(batch_id: str) -> None:
    from fao_impact_monitor.athena.pipeline.handlers import process_batch
    from fao_impact_monitor.athena.runtime import build_context

    await process_batch(build_context(), batch_id)
