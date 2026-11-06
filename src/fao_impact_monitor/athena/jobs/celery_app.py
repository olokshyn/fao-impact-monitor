"""Celery application. Tasks are registered when this module is imported."""

from __future__ import annotations

from celery import Celery

from fao_impact_monitor.athena.config import get_athena_config

celery_app = Celery("athena", broker=get_athena_config().broker_url)
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    task_acks_late=True,
    broker_connection_retry_on_startup=True,
    beat_schedule={
        "flush-batches": {"task": "athena.flush_batches", "schedule": 5.0},
        "recover-jobs": {"task": "athena.recover_jobs", "schedule": 30.0},
    },
)


def register_tasks() -> None:
    import fao_impact_monitor.athena.jobs.tasks  # noqa: F401
