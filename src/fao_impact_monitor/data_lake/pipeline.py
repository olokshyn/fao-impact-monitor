from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

from beanie import Document as BeanieDocument
from beanie import Indexed, PydanticObjectId
from pydantic import BaseModel, Field, PrivateAttr

from fao_impact_monitor.data_lake.common import Status

from .document import Document, DocumentType, RelationSide
from .documents.pdf_document import PdfDocument
from .documents.tellus_document import TellusDocument
from .documents.web_page_document import WebPageDocument
from .stage import StageResult, get_stage
from .stages.country_detect_stage import (
    CHUNK_ITERATOR_PARAM,
    COUNTRY_DETECT_STAGE_NAME,
)
from .stages.embed_chunks_stage import EMBED_CHUNKS_STAGE_NAME
from .stages.pdf_crawl_stage import (
    PDF_CRAWL_STAGE_NAME,
    PIPELINE_FOR_PDF_PARAM,
    PIPELINE_FOR_WEB_PARAM,
)
from .stages.pdf_extract_stage import PDF_EXTRACT_STAGE_NAME, PdfExtractStageResult
from .stages.tellus_document_fetch_stage import (
    TELLUS_DOCUMENT_FETCH_STAGE_NAME,
    TellusDocumentFetchStageResult,
)

logger = logging.getLogger(__name__)

PIPELINE_PDF_CRAWL = "pdf_crawl"
PIPELINE_PDF_PROCESS = "pdf_process"
PIPELINE_TELLUS_PROCESS = "tellus_process"

_PIPELINE_REGISTRY: dict[str, type[Pipeline]] = {}


class PipelineStep(BaseModel):
    stage_name: str
    params: dict[str, Any]


class Pipeline(BeanieDocument):
    name: Annotated[str, Indexed(unique=True)]
    steps: list[PipelineStep] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    _stage_ok: dict[str, int] = PrivateAttr(default_factory=dict)
    _stage_error: dict[str, int] = PrivateAttr(default_factory=dict)

    def is_completed(self, document: Document) -> bool:
        for step in self.steps:
            if self._get_stage_result(document, step) is None:
                return False
        return True

    def _record_stage_outcome(self, stage_name: str, status: Status) -> None:
        if status == Status.COMPLETED:
            self._stage_ok[stage_name] = self._stage_ok.get(stage_name, 0) + 1
        else:
            self._stage_error[stage_name] = self._stage_error.get(stage_name, 0) + 1

    def stage_stats_summary(self) -> str:
        parts: list[str] = []
        for step in self.steps:
            name = step.stage_name
            ok = self._stage_ok.get(name, 0)
            err = self._stage_error.get(name, 0)
            if ok or err:
                parts.append(f"{name}: ok={ok} errors={err}")
        return "; ".join(parts) if parts else "no stages run"

    def log_stage_stats(self) -> None:
        logger.info(
            "pipeline %s stage stats: %s", self.name, self.stage_stats_summary()
        )

    async def run(self, document: Document) -> None:
        # TODO: re-run stage if the upstream or the current stage version has changed
        existing = await _find_document_by_url(document.url)
        if existing is not None:
            document = existing
            if document.is_pipeline_completed(self.name):
                logger.info(
                    "pipeline %s: skip %s (already completed)",
                    self.name,
                    document.url,
                )
                return

        logger.info("pipeline %s: start %s", self.name, document.url)
        if self.name not in document.pipeline_statuses:
            document.set_pipeline_status(self.name, Status.PENDING)
        document.set_pipeline_status(self.name, Status.RUNNING)
        # Persist RUNNING only for docs already in Mongo. New docs are inserted by
        # stages (avoids inserting a WEB_PAGE that a stage may replace with PDF).
        if document.id is not None:
            await document.save()

        try:
            for step in self.steps:
                if self._get_stage_result(document, step) is not None:
                    logger.info(
                        "pipeline %s: skip stage %s for %s (already completed)",
                        self.name,
                        step.stage_name,
                        document.url,
                    )
                    continue
                logger.info(
                    "pipeline %s: running stage %s for %s",
                    self.name,
                    step.stage_name,
                    document.url,
                )
                stage = get_stage(step.stage_name)
                prev_results = self._get_prev_results(document, step)
                result = await stage.run(document, step.params, prev_results)
                refreshed = await _find_document_by_url(document.url)
                if refreshed is not None:
                    document = refreshed
                if not _stage_result_already_recorded(
                    document, step.stage_name, result
                ):
                    document.stage_results.setdefault(step.stage_name, []).append(
                        result
                    )
                if self.name not in document.pipeline_statuses:
                    document.set_pipeline_status(self.name, Status.RUNNING)
                await document.save()

                self._record_stage_outcome(step.stage_name, result.status)
                logger.info(
                    "pipeline %s: stage %s finished for %s status=%s; "
                    "stage totals ok=%s errors=%s",
                    self.name,
                    step.stage_name,
                    document.url,
                    result.status,
                    self._stage_ok.get(step.stage_name, 0),
                    self._stage_error.get(step.stage_name, 0),
                )

            # Reload so cascade sees relations/status written by stages (e.g. DFS).
            refreshed = await _find_document_by_url(document.url)
            if refreshed is not None:
                document = refreshed

            await self._cascade_children(document)

            if self.is_completed(document):
                document.set_pipeline_status(self.name, Status.COMPLETED)
            else:
                document.set_pipeline_status(self.name, Status.FAILED)
            await document.save()
            logger.info(
                "pipeline %s: finished %s status=%s",
                self.name,
                document.url,
                document.pipeline_status(self.name),
            )
        except Exception:
            document.set_pipeline_status(self.name, Status.FAILED)
            if document.id is not None:
                await document.save()
            logger.exception(
                "pipeline %s: failed for %s",
                self.name,
                document.url,
            )
            raise

    async def _cascade_children(self, document: Document) -> None:
        pipelines: dict[str, Pipeline] = {}
        cascaded = 0
        for rel in document.relations:
            if rel.side != RelationSide.TO:
                continue
            child = await _find_document_by_id(rel.d_id, rel.d_type)
            if child is None:
                continue
            for pipeline_name, status in list(child.pipeline_statuses.items()):
                if status == Status.COMPLETED:
                    continue
                child_pipeline = pipelines.setdefault(
                    pipeline_name, get_pipeline(pipeline_name)
                )
                await child_pipeline.run(child)
                cascaded += 1
        if cascaded:
            logger.info(
                "pipeline %s: cascaded %s child run(s) from %s",
                self.name,
                cascaded,
                document.url,
            )
        for child_pipeline in pipelines.values():
            child_pipeline.log_stage_stats()

    def _get_stage_result(
        self, document: Document, step: PipelineStep
    ) -> StageResult | None:
        stage_results = document.stage_results.get(step.stage_name)
        if not stage_results or stage_results[-1].status != Status.COMPLETED:
            return None
        return stage_results[-1]

    def _get_prev_results(
        self, document: Document, current_step: PipelineStep
    ) -> list[StageResult]:
        prev_results: list[StageResult] = []
        for step in self.steps:
            if step.stage_name == current_step.stage_name:
                break
            result = self._get_stage_result(document, step)
            if result is None:
                break
            prev_results.append(result)
        return prev_results


def register_pipeline(name: str, pipeline_cls: type[Pipeline]) -> None:
    _PIPELINE_REGISTRY[name] = pipeline_cls


def get_pipeline(name: str) -> Pipeline:
    try:
        pipeline_cls = _PIPELINE_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown pipeline: {name!r}") from exc
    return pipeline_cls()


def _stage_result_already_recorded(
    document: Document,
    stage_name: str,
    result: StageResult,
) -> bool:
    """True if the stage already persisted ``result`` on ``document``."""
    results = document.stage_results.get(stage_name)
    if not results:
        return False
    latest = results[-1]
    return (
        latest.version_id == result.version_id
        and latest.status == result.status
        and latest.error == result.error
    )


async def _find_document_by_url(url: str) -> Document | None:
    return await Document.find_one(Document.url == url)


async def find_tellus_document_by_external_id(
    external_id: str,
) -> TellusDocument | None:
    return await TellusDocument.find_one(TellusDocument.external_id == external_id)


async def _find_document_by_id(
    document_id: PydanticObjectId,
    document_type: DocumentType,
) -> Document | None:
    if document_type == DocumentType.WEB_PAGE:
        return await WebPageDocument.get(document_id)
    if document_type == DocumentType.PDF:
        return await PdfDocument.get(document_id)
    if document_type == DocumentType.TELLUS:
        return await TellusDocument.get(document_id)
    return None


def extracted_pdf_chunk_iterator(prev_stages: list[StageResult]) -> Iterator[str]:
    """Yield markdown text for each page from a completed pdf_extract result."""
    extract = _resolve_pdf_extract(prev_stages)
    if extract is None:
        raise ValueError(
            f"country_detect requires a completed {PDF_EXTRACT_STAGE_NAME} "
            "result in previous stages"
        )
    for page_path in extract.page_paths:
        path = Path(page_path)
        if not path.is_file():
            raise ValueError(f"pdf_extract page file not found: {page_path}")
        yield path.read_text(encoding="utf-8")


def extracted_tellus_chunk_iterator(prev_stages: list[StageResult]) -> Iterator[str]:
    """Yield markdown text for each chunk from a completed tellus fetch result."""
    fetch = _resolve_tellus_fetch(prev_stages)
    if fetch is None:
        raise ValueError(
            f"country_detect requires a completed {TELLUS_DOCUMENT_FETCH_STAGE_NAME} "
            "result in previous stages"
        )
    for page_path in fetch.page_paths:
        path = Path(page_path)
        if not path.is_file():
            raise ValueError(f"tellus_document_fetch page file not found: {page_path}")
        yield path.read_text(encoding="utf-8")


def _resolve_pdf_extract(
    prev_stages: list[StageResult],
) -> PdfExtractStageResult | None:
    for result in reversed(prev_stages):
        if result.name != PDF_EXTRACT_STAGE_NAME:
            continue
        if isinstance(result, PdfExtractStageResult):
            extract = result
        else:
            extract = PdfExtractStageResult.model_validate(result.model_dump())
        if extract.status != Status.COMPLETED:
            return None
        return extract
    return None


def _resolve_tellus_fetch(
    prev_stages: list[StageResult],
) -> TellusDocumentFetchStageResult | None:
    for result in reversed(prev_stages):
        if result.name != TELLUS_DOCUMENT_FETCH_STAGE_NAME:
            continue
        if isinstance(result, TellusDocumentFetchStageResult):
            fetch = result
        else:
            fetch = TellusDocumentFetchStageResult.model_validate(result.model_dump())
        if fetch.status != Status.COMPLETED:
            return None
        return fetch
    return None


class PdfCrawlPipeline(Pipeline):
    name: Annotated[str, Indexed(unique=True)] = PIPELINE_PDF_CRAWL
    steps: list[PipelineStep] = Field(
        default_factory=lambda: [
            PipelineStep(
                stage_name=PDF_CRAWL_STAGE_NAME,
                params={
                    PIPELINE_FOR_WEB_PARAM: PIPELINE_PDF_CRAWL,
                    PIPELINE_FOR_PDF_PARAM: PIPELINE_PDF_PROCESS,
                },
            ),
        ]
    )


class PdfProcessPipeline(Pipeline):
    name: Annotated[str, Indexed(unique=True)] = PIPELINE_PDF_PROCESS
    steps: list[PipelineStep] = Field(
        default_factory=lambda: [
            PipelineStep(stage_name=PDF_EXTRACT_STAGE_NAME, params={}),
            PipelineStep(
                stage_name=COUNTRY_DETECT_STAGE_NAME,
                params={CHUNK_ITERATOR_PARAM: extracted_pdf_chunk_iterator},
            ),
            PipelineStep(
                stage_name=EMBED_CHUNKS_STAGE_NAME,
                params={CHUNK_ITERATOR_PARAM: extracted_pdf_chunk_iterator},
            ),
        ]
    )


class TellusProcessPipeline(Pipeline):
    name: Annotated[str, Indexed(unique=True)] = PIPELINE_TELLUS_PROCESS
    steps: list[PipelineStep] = Field(
        default_factory=lambda: [
            PipelineStep(stage_name=TELLUS_DOCUMENT_FETCH_STAGE_NAME, params={}),
            PipelineStep(
                stage_name=COUNTRY_DETECT_STAGE_NAME,
                params={CHUNK_ITERATOR_PARAM: extracted_tellus_chunk_iterator},
            ),
            PipelineStep(
                stage_name=EMBED_CHUNKS_STAGE_NAME,
                params={CHUNK_ITERATOR_PARAM: extracted_tellus_chunk_iterator},
            ),
        ]
    )


register_pipeline(PIPELINE_PDF_CRAWL, PdfCrawlPipeline)
register_pipeline(PIPELINE_PDF_PROCESS, PdfProcessPipeline)
register_pipeline(PIPELINE_TELLUS_PROCESS, TellusProcessPipeline)
