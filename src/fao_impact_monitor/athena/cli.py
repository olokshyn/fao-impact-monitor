"""Serve the API or the batched worker."""

from __future__ import annotations

import typer
import uvicorn

from fao_impact_monitor.athena.config import get_athena_config

cli = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


@cli.command()
def serve() -> None:
    """Start the FastAPI service. Ingestion and research run on workers."""
    from fao_impact_monitor.athena.api.app import create_app
    from fao_impact_monitor.athena.runtime import build_context

    settings = get_athena_config()
    app = create_app(build_context(settings))
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)


@cli.command()
def worker() -> None:
    """Run Celery with its beat scheduler so batches flush on count or time."""
    from fao_impact_monitor.athena.jobs.celery_app import celery_app, register_tasks

    register_tasks()
    celery_app.worker_main(argv=["worker", "--beat", "--loglevel=INFO"])


def main() -> None:
    cli()


def worker_main() -> None:
    worker()
