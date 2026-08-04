"""PDF → per-page markdown extract stage powered by Docling."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import queue
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import Field

from fao_impact_monitor.config import PdfExtractConfig, get_config
from fao_impact_monitor.data_lake.common import Status
from fao_impact_monitor.data_lake.document import Document, DocumentType
from fao_impact_monitor.data_lake.documents.pdf_document import PdfDocument
from fao_impact_monitor.data_lake.stage import (
    Stage,
    StageResult,
    StageVersion,
)
from fao_impact_monitor.data_lake.stages.pdf_crawl_stage import (
    PDF_CRAWL_STAGE_NAME,
    PdfCrawlStageResult,
)

logger = logging.getLogger(__name__)

PDF_EXTRACT_STAGE_NAME = "pdf_extract"
_DOCLING_PIPELINE_ID = "ocr+tables+page_range"

SubmitFn = Callable[..., Awaitable["PdfExtractStageResult"]]


class PdfExtractStageResult(StageResult):
    name: str = PDF_EXTRACT_STAGE_NAME
    title: str | None = None
    num_pages: int = 0
    page_paths: list[str] = Field(default_factory=list)


class PdfExtractStageVersion(StageVersion):
    pipeline_id: str

    class Settings:
        class_id_value = PDF_EXTRACT_STAGE_NAME


@dataclass(slots=True)
class _ExtractJob:
    pdf_path: Path
    out_dir: Path
    version_id: str
    fallback_title: str | None
    loop: asyncio.AbstractEventLoop
    future: asyncio.Future[PdfExtractStageResult]


class DoclingWorker:
    """Single-thread Docling worker: loads weights once; queue carries paths only.

    Callers are expected on a single event-loop thread (many coroutines OK).
    """

    _instance: DoclingWorker | None = None

    def __init__(self) -> None:
        self._jobs: queue.Queue[_ExtractJob | None] = queue.Queue()
        self._thread = threading.Thread(
            target=self._run,
            name="docling-worker",
            daemon=True,
        )
        self._started = False

    @classmethod
    def instance(cls) -> DoclingWorker:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _ensure_started(self) -> None:
        if self._started:
            return
        self._thread.start()
        self._started = True

    async def submit(
        self,
        *,
        pdf_path: Path,
        out_dir: Path,
        version_id: str,
        fallback_title: str | None,
    ) -> PdfExtractStageResult:
        self._ensure_started()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[PdfExtractStageResult] = loop.create_future()
        self._jobs.put(
            _ExtractJob(
                pdf_path=pdf_path,
                out_dir=out_dir,
                version_id=version_id,
                fallback_title=fallback_title,
                loop=loop,
                future=future,
            )
        )
        return await future

    def _run(self) -> None:
        # Import and weight load happen only on this worker thread.
        converter = _build_converter()
        while True:
            job = self._jobs.get()
            if job is None:
                return
            try:
                result = _process_job(converter, job)
            except Exception as exc:
                logger.exception("Docling extract failed for %s", job.pdf_path)
                _set_future_exception(job, exc)
            else:
                _set_future_result(job, result)

    def shutdown(self) -> None:
        """Stop the worker loop (tests / process teardown)."""
        if not self._started:
            return
        self._jobs.put(None)
        self._thread.join(timeout=30)
        self._started = False


def _build_converter() -> Any:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pipeline_options = PdfPipelineOptions(do_ocr=True, do_table_structure=True)
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )


def _process_job(
    converter: Any,
    job: _ExtractJob,
) -> PdfExtractStageResult:
    import pypdfium2 as pdfium

    if not job.pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {job.pdf_path}")

    # Read/open PDF on the worker thread (bytes stay off the queue).
    pdf = pdfium.PdfDocument(str(job.pdf_path))
    try:
        num_pages = len(pdf)
    finally:
        pdf.close()

    if num_pages < 1:
        raise ValueError(f"PDF has no pages: {job.pdf_path}")

    job.out_dir.mkdir(parents=True, exist_ok=True)
    page_paths: list[str] = []
    title: str | None = None

    for page_no in range(1, num_pages + 1):
        conv = converter.convert(str(job.pdf_path), page_range=(page_no, page_no))
        markdown = conv.document.export_to_markdown()
        page_path = job.out_dir / f"page_{page_no:04d}.md"
        # Markdown is written on the worker thread (content stays off the queue).
        page_path.write_text(markdown, encoding="utf-8")
        page_paths.append(str(page_path))
        if page_no == 1:
            title = _extract_title(conv.document, job.fallback_title)

    if title is None:
        title = job.fallback_title

    return PdfExtractStageResult(
        version_id=job.version_id,
        status=Status.COMPLETED,
        title=title,
        num_pages=num_pages,
        page_paths=page_paths,
    )


def _extract_title(document: Any, fallback: str | None) -> str | None:
    """Pick the document title from page-1 layout labels.

    Prefer top-level headings the way markdown ranks them: ``TITLE`` (``#``)
    over ``SECTION_HEADER`` level 1 (``##``) over deeper levels — even when a
    deeper heading is longer. Within the best heading rank, prefer the longest
    text (cover chrome is often short; the real title is usually longer).
    """
    from docling_core.types.doc.labels import DocItemLabel

    # rank 0 = TITLE (#); section headers use their Docling level (1, 2, ...).
    best_rank: int | None = None
    best_texts: list[str] = []
    for item, _level in document.iterate_items():
        label = getattr(item, "label", None)
        text = " ".join((getattr(item, "text", None) or "").split())
        if not text:
            continue
        if label == DocItemLabel.TITLE:
            rank = 0
        elif label == DocItemLabel.SECTION_HEADER:
            level = getattr(item, "level", None)
            rank = level if isinstance(level, int) and level >= 1 else 1
        else:
            continue
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best_texts = [text]
        elif rank == best_rank:
            best_texts.append(text)
    if best_texts:
        return max(best_texts, key=len)
    return fallback


def _set_future_result(job: _ExtractJob, result: PdfExtractStageResult) -> None:
    def _set() -> None:
        if not job.future.done():
            job.future.set_result(result)

    job.loop.call_soon_threadsafe(_set)


def _set_future_exception(job: _ExtractJob, exc: BaseException) -> None:
    def _set() -> None:
        if not job.future.done():
            job.future.set_exception(exc)

    job.loop.call_soon_threadsafe(_set)


class PdfExtractStage(Stage):
    name = PDF_EXTRACT_STAGE_NAME

    def __init__(
        self,
        *,
        config: PdfExtractConfig | None = None,
        submit_fn: SubmitFn | None = None,
    ) -> None:
        self._config = config
        self._submit_fn = submit_fn

    async def get_version(self) -> StageVersion:
        version_id = hashlib.sha256(_DOCLING_PIPELINE_ID.encode()).hexdigest()[:32]
        existing = await PdfExtractStageVersion.find_one(
            PdfExtractStageVersion.version_id == version_id
        )
        if existing is not None:
            return existing
        version = PdfExtractStageVersion(
            version_id=version_id,
            pipeline_id=_DOCLING_PIPELINE_ID,
        )
        await version.insert()
        return version

    async def run(
        self,
        document: Document,
        stage_params: dict[str, Any],
        prev_stages: list[StageResult],
    ) -> StageResult:
        del stage_params, prev_stages
        cfg = self._config or get_config().pdf_extract
        version = await self.get_version()
        version_id = version.version_id

        if document.type != DocumentType.PDF:
            return PdfExtractStageResult(
                version_id=version_id,
                status=Status.FAILED,
                error=f"pdf_extract requires a PDF document, got {document.type}",
            )

        crawl_path = _resolve_crawl_content_path(document)
        if crawl_path is None:
            return PdfExtractStageResult(
                version_id=version_id,
                status=Status.FAILED,
                error="Missing completed pdf_crawl content_path on document",
            )

        pdf_path = Path(crawl_path)
        if not pdf_path.is_file():
            return PdfExtractStageResult(
                version_id=version_id,
                status=Status.FAILED,
                error=f"PDF file not found: {pdf_path}",
            )

        if document.id is None:
            return PdfExtractStageResult(
                version_id=version_id,
                status=Status.FAILED,
                error="PDF document must be saved before pdf_extract",
            )

        out_dir = cfg.save_dir / str(document.id)
        try:
            result = await self._submit(
                pdf_path=pdf_path,
                out_dir=out_dir,
                version_id=version_id,
                fallback_title=document.title,
            )
        except Exception as exc:
            logger.exception("pdf_extract failed for %s", document.url)
            return PdfExtractStageResult(
                version_id=version_id,
                status=Status.FAILED,
                error=str(exc),
            )

        if result.status == Status.COMPLETED:
            # Prefer a crawl-agent title already set on the PDF; use Docling
            # only when none was discovered (or it expired before this PDF).
            if document.title:
                result = result.model_copy(update={"title": document.title})
            elif result.title:
                document.title = result.title
            if isinstance(document, PdfDocument):
                document.page_paths = list(result.page_paths)
        return result

    async def _submit(
        self,
        *,
        pdf_path: Path,
        out_dir: Path,
        version_id: str,
        fallback_title: str | None,
    ) -> PdfExtractStageResult:
        if self._submit_fn is not None:
            return await self._submit_fn(
                pdf_path=pdf_path,
                out_dir=out_dir,
                version_id=version_id,
                fallback_title=fallback_title,
            )
        return await DoclingWorker.instance().submit(
            pdf_path=pdf_path,
            out_dir=out_dir,
            version_id=version_id,
            fallback_title=fallback_title,
        )


def _resolve_crawl_content_path(document: Document) -> str | None:
    results = document.stage_results.get(PDF_CRAWL_STAGE_NAME)
    if not results:
        return None
    latest = results[-1]
    if isinstance(latest, PdfCrawlStageResult):
        crawl = latest
    else:
        crawl = PdfCrawlStageResult.model_validate(latest.model_dump())
    if crawl.status != Status.COMPLETED or not crawl.content_path:
        return None
    return crawl.content_path
