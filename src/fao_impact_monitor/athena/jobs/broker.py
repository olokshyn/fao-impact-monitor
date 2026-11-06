"""Celery publisher. The call is separate from the committed job record."""

from __future__ import annotations


class CeleryBroker:
    def publish(self, task_type: str, batch_id: str) -> None:
        from fao_impact_monitor.athena.jobs.celery_app import celery_app

        celery_app.send_task("athena.run_batch", args=[task_type, batch_id])

    def probe(self) -> None:
        from fao_impact_monitor.athena.jobs.celery_app import celery_app

        connection = celery_app.connection()
        connection.ensure_connection(max_retries=1)
        connection.release()
