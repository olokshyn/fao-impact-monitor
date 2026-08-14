"""Typer entry point for the standalone PDF evidence pipeline."""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import TypedDict

import typer

from fao_impact_monitor.pdf_pipeline.ingest import PdfEvidenceIngestor
from fao_impact_monitor.pdf_pipeline.mongo import (
    connect_pdf_pipeline,
    reset_pdf_pipeline,
)
from fao_impact_monitor.pdf_pipeline.retrieval import ensure_pdf_pipeline_indexes

app = typer.Typer(
    help="Ingest graphically rich FAO PDFs as provenance-bearing evidence."
)

logger = logging.getLogger(__name__)


class _WorkerResult(TypedDict):
    processed: int
    skipped: int
    failed: int
    path: str
    pid: int
    elapsed_seconds: float
    error: str | None


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(processName)s[%(process)d] | %(levelname)s | %(message)s"
        ),
        datefmt="%H:%M:%S",
        force=True,
    )


def _ingest_pdf_worker(
    pdf_path: str, force: bool, position: int, total: int
) -> _WorkerResult:
    """Process one PDF with resources created inside the spawned process."""
    _configure_logging()
    started_at = perf_counter()
    path = Path(pdf_path)

    async def run() -> bool:
        logger.info("[%d/%d] Worker starting %s", position, total, path)
        client = await connect_pdf_pipeline()
        try:
            return await PdfEvidenceIngestor().ingest_pdf(path, force=force)
        finally:
            await client.close()
            logger.info("[%d/%d] Worker MongoDB connection closed", position, total)

    try:
        changed = asyncio.run(run())
    except Exception as error:  # noqa: BLE001 - return an isolated worker failure
        logger.error(
            "[%d/%d] Worker failed %s after %.1fs: %s",
            position,
            total,
            path,
            perf_counter() - started_at,
            error,
        )
        return {
            "processed": 0,
            "skipped": 0,
            "failed": 1,
            "path": pdf_path,
            "pid": os.getpid(),
            "elapsed_seconds": perf_counter() - started_at,
            "error": str(error) or type(error).__name__,
        }
    return {
        "processed": int(changed),
        "skipped": int(not changed),
        "failed": 0,
        "path": pdf_path,
        "pid": os.getpid(),
        "elapsed_seconds": perf_counter() - started_at,
        "error": None,
    }


def _run_directory_workers(
    pdfs: list[Path], *, force: bool, workers: int
) -> dict[str, int]:
    counts = {"processed": 0, "skipped": 0, "failed": 0}
    if not pdfs:
        logger.info("No eligible PDFs to process")
        return counts

    worker_count = min(workers, len(pdfs))
    logger.info(
        "Starting %d spawned worker process(es) for %d PDF(s)",
        worker_count,
        len(pdfs),
    )
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=context,
    ) as executor:
        futures = [
            executor.submit(_ingest_pdf_worker, str(pdf), force, index, len(pdfs))
            for index, pdf in enumerate(pdfs, start=1)
        ]
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as error:  # noqa: BLE001 - a worker process may crash
                counts["failed"] += 1
                logger.error("Worker process crashed: %s", error)
                continue
            counts["processed"] += result["processed"]
            counts["skipped"] += result["skipped"]
            counts["failed"] += result["failed"]
            logger.info(
                "Worker PID %d finished %s in %.1fs: processed=%d skipped=%d failed=%d",
                result["pid"],
                result["path"],
                result["elapsed_seconds"],
                result["processed"],
                result["skipped"],
                result["failed"],
            )
    return counts


@app.callback()
def main() -> None:
    """Standalone PDF evidence commands."""


@app.command()
def ingest(
    pdf_dir: Path = typer.Argument(  # noqa: B008
        Path("fao_data"), exists=True, file_okay=False
    ),
    force: bool = typer.Option(
        False,
        help="Discard saved checkpoints and rebuild the selected PDF(s).",
    ),
    pdf: Path | None = typer.Option(  # noqa: B008
        None,
        "--pdf",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Resume this PDF if failed; restart it if completed or with --force.",
    ),
    limit: int | None = typer.Option(
        None,
        min=1,
        help="Maximum documents: first N with --force, next N unprocessed otherwise.",
    ),
    workers: int = typer.Option(
        1,
        "--workers",
        "-w",
        min=1,
        help="Spawn N independent worker processes for directory ingestion.",
    ),
    reset: bool = typer.Option(
        False, help="Delete only pdf_pipeline Mongo records first."
    ),
) -> None:
    """Populate pdf_documents, pdf_sections, pdf_evidence and pipeline embeddings."""

    _configure_logging()

    async def run_single() -> dict[str, int]:
        logger.info("Connecting to MongoDB")
        client = await connect_pdf_pipeline()
        try:
            if reset:
                logger.info("Resetting records owned by pdf_pipeline")
                await reset_pdf_pipeline()
            logger.info("Ensuring PDF search indexes")
            await ensure_pdf_pipeline_indexes()
            assert pdf is not None
            return await PdfEvidenceIngestor().ingest_single_pdf(pdf, force=force)
        finally:
            await client.close()
            logger.info("MongoDB connection closed")

    async def select_directory() -> list[Path]:
        logger.info("Connecting to MongoDB for directory selection")
        client = await connect_pdf_pipeline()
        try:
            if reset:
                logger.info("Resetting records owned by pdf_pipeline")
                await reset_pdf_pipeline()
            logger.info("Ensuring PDF search indexes")
            await ensure_pdf_pipeline_indexes()
            return await PdfEvidenceIngestor().select_directory_pdfs(
                pdf_dir, force=force, limit=limit
            )
        finally:
            await client.close()
            logger.info("Parent MongoDB connection closed")

    if pdf is not None:
        if workers != 1:
            logger.info("Ignoring --workers for explicit single-PDF ingestion")
        result = asyncio.run(run_single())
    else:
        selected = asyncio.run(select_directory())
        started_at = perf_counter()
        result = _run_directory_workers(selected, force=force, workers=workers)
        logger.info(
            "Folder ingestion finished in %.1fs: processed=%d skipped=%d failed=%d",
            perf_counter() - started_at,
            result["processed"],
            result["skipped"],
            result["failed"],
        )
    typer.echo(result)


@app.command("ensure-indexes")
def ensure_indexes_command(
    recreate: bool = typer.Option(
        False,
        "--recreate",
        help="Drop existing PDF search indexes and create them again.",
    ),
) -> None:
    """Create Atlas Search indexes for the PDF evidence collection."""
    _configure_logging()

    async def run() -> None:
        logger.info("Connecting to MongoDB")
        client = await connect_pdf_pipeline()
        try:
            logger.info(
                "Recreating PDF search indexes"
                if recreate
                else "Ensuring PDF search indexes"
            )
            await ensure_pdf_pipeline_indexes(recreate=recreate)
        finally:
            await client.close()
            logger.info("MongoDB connection closed")

    asyncio.run(run())
    typer.echo("PDF search indexes are ready")
