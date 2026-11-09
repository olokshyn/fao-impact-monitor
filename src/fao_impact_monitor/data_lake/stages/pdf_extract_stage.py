"""PDF → per-page markdown extract stage powered by pdf-extractor-ai."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from pdf_extractor_ai import PdfExtractor
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


class PdfExtractWorker:
    """One pdf-extractor-ai worker. Docling weights load once on that thread.

    Callers are expected on a single event-loop thread (many coroutines OK).
    """

    _instance: PdfExtractWorker | None = None

    def __init__(self) -> None:
        # num_workers=1 matches the previous single converter thread.
        # converter_factory keeps OCR, tables, cell matching, and CPU.
        self._extractor = PdfExtractor(
            num_workers=1,
            converter_factory=_build_converter,
        )

    @classmethod
    def instance(cls) -> PdfExtractWorker:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def submit(
        self,
        *,
        pdf_path: Path,
        out_dir: Path,
        version_id: str,
        fallback_title: str | None,
    ) -> PdfExtractStageResult:
        return await asyncio.to_thread(
            _process_pdf,
            self._extractor,
            pdf_path,
            out_dir,
            version_id,
            fallback_title,
        )

    def shutdown(self) -> None:
        """Stop the extractor threads (tests / process teardown)."""
        self._extractor.shutdown()
        if PdfExtractWorker._instance is self:
            PdfExtractWorker._instance = None


def _build_converter(do_ocr: bool) -> Any:
    """Build the converter PdfExtractor runs for this stage.

    MPS on Apple Silicon aborts in RT-DETR layout resize; keep inference on CPU.
    Leave table-cell matching, image scale, and OCR languages at Docling defaults:
    that is the pipeline this stage used before pdf-extractor-ai.
    """
    from docling.datamodel.accelerator_options import (
        AcceleratorDevice,
        AcceleratorOptions,
    )
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pipeline_options = PdfPipelineOptions(
        do_ocr=do_ocr,
        do_table_structure=True,
        accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU),
    )
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )


def _process_pdf(
    extractor: PdfExtractor,
    pdf_path: Path,
    out_dir: Path,
    version_id: str,
    fallback_title: str | None,
) -> PdfExtractStageResult:
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # do_ocr=True: this stage always OCRs, including born-digital pages.
    document = extractor.extract(
        pdf_path.read_bytes(),
        do_ocr=True,
        name=pdf_path.name,
    )
    page_numbers = sorted(int(number) for number in document.pages)
    if not page_numbers:
        raise ValueError(f"PDF has no pages: {pdf_path}")
    num_pages = page_numbers[-1]
    if num_pages < 1:
        raise ValueError(f"PDF has no pages: {pdf_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    page_paths: list[str] = []
    for page_no in range(1, num_pages + 1):
        # Docling's own markdown, one file per page, as this stage wrote before.
        markdown = document.export_to_markdown(page_no=page_no)
        page_path = out_dir / f"page_{page_no:04d}.md"
        page_path.write_text(markdown, encoding="utf-8")
        page_paths.append(str(page_path))

    title = _extract_title(document, fallback_title)
    return PdfExtractStageResult(
        version_id=version_id,
        status=Status.COMPLETED,
        title=title,
        num_pages=num_pages,
        page_paths=page_paths,
    )


def _is_first_page(item: Any) -> bool:
    """True for page 1, and for items that carry no page provenance."""
    prov = getattr(item, "prov", None) or []
    if not prov:
        return True
    page_no = getattr(prov[0], "page_no", None)
    return page_no in (None, 1)


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
        if not _is_first_page(item):
            continue
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
            # Prefer a crawl-agent title already set on the PDF; use extraction
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
        return await PdfExtractWorker.instance().submit(
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
