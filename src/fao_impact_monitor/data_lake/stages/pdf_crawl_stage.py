"""BFS PDF crawl stage driven by a LangGraph link-extraction agent."""

import hashlib
import logging
from collections import deque
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import urljoin

from beanie import PydanticObjectId
from langchain_core.language_models.chat_models import BaseChatModel

from fao_impact_monitor.config import PdfCrawlConfig, get_config
from fao_impact_monitor.data_lake.document import (
    Document,
    DocumentType,
    Relation,
    RelationSide,
    RelationType,
)
from fao_impact_monitor.data_lake.documents.pdf_document import PdfDocument
from fao_impact_monitor.data_lake.documents.web_page_document import WebPageDocument
from fao_impact_monitor.data_lake.scrapling import (
    HTML_MAGIC_BYTES,
    PDF_MAGIC_BYTES,
    reliable_fetch,
)
from fao_impact_monitor.data_lake.stage import (
    Stage,
    StageResult,
    StageStatus,
    StageVersion,
)
from fao_impact_monitor.data_lake.stages.pdf_crawl_agent import extract_page_urls

logger = logging.getLogger(__name__)

PDF_CRAWL_STAGE_NAME = "pdf_crawl"
PIPELINE_FOR_WEB_PARAM = "pipeline_for_web"
PIPELINE_FOR_PDF_PARAM = "pipeline_for_pdf"

FetchFn = Callable[..., Awaitable[bytes]]
ExtractFn = Callable[..., Awaitable[list[str]]]

ContentKind = Literal["pdf", "html", "other"]


def _require_pipeline_params(stage_params: dict[str, Any]) -> tuple[str, str]:
    """Return ``(pipeline_for_web, pipeline_for_pdf)`` from stage params."""
    try:
        pipeline_for_web = stage_params[PIPELINE_FOR_WEB_PARAM]
        pipeline_for_pdf = stage_params[PIPELINE_FOR_PDF_PARAM]
    except KeyError as exc:
        raise ValueError(
            f"pdf_crawl requires stage_params "
            f"{PIPELINE_FOR_WEB_PARAM!r} and {PIPELINE_FOR_PDF_PARAM!r}"
        ) from exc
    if not isinstance(pipeline_for_web, str) or not pipeline_for_web:
        raise ValueError(f"{PIPELINE_FOR_WEB_PARAM} must be a non-empty string")
    if not isinstance(pipeline_for_pdf, str) or not pipeline_for_pdf:
        raise ValueError(f"{PIPELINE_FOR_PDF_PARAM} must be a non-empty string")
    return pipeline_for_web, pipeline_for_pdf


class PdfCrawlStageResult(StageResult):
    name: str = PDF_CRAWL_STAGE_NAME
    content_path: str | None = None


class PdfCrawlStageVersion(StageVersion):
    llm_model: str
    max_url_depth: int
    max_urls_per_page: int
    max_pdfs: int

    class Settings:
        class_id_value = PDF_CRAWL_STAGE_NAME


class PdfCrawlStage(Stage):
    name = PDF_CRAWL_STAGE_NAME

    def __init__(
        self,
        *,
        fetch_fn: FetchFn | None = None,
        extract_fn: ExtractFn | None = None,
        chat_model: BaseChatModel | None = None,
        config: PdfCrawlConfig | None = None,
    ) -> None:
        self._fetch_fn = fetch_fn or reliable_fetch
        self._extract_fn = extract_fn
        self._chat_model = chat_model
        self._config = config

    async def get_version(self) -> StageVersion:
        cfg = self._config or get_config().pdf_crawl
        payload = (
            f"{cfg.llm_model}|{cfg.max_url_depth}|{cfg.max_urls_per_page}|"
            f"{cfg.max_pdfs}|{cfg.max_agent_retries}"
        )
        version_id = hashlib.sha256(payload.encode()).hexdigest()[:32]
        existing = await PdfCrawlStageVersion.find_one(
            PdfCrawlStageVersion.version_id == version_id
        )
        if existing is not None:
            return existing
        version = PdfCrawlStageVersion(
            version_id=version_id,
            llm_model=cfg.llm_model,
            max_url_depth=cfg.max_url_depth,
            max_urls_per_page=cfg.max_urls_per_page,
            max_pdfs=cfg.max_pdfs,
        )
        await version.insert()
        return version

    async def run(
        self,
        document: Document,
        stage_params: dict[str, Any],
        prev_stages: list[StageResult],
    ) -> StageResult:
        del prev_stages
        cfg = self._config or get_config().pdf_crawl
        version = await self._ensure_version(cfg)
        version_id = version.version_id

        try:
            pipeline_for_web, pipeline_for_pdf = _require_pipeline_params(stage_params)
        except ValueError as exc:
            return PdfCrawlStageResult(
                version_id=version_id,
                status=StageStatus.FAILED,
                error=str(exc),
            )

        if document.type != DocumentType.WEB_PAGE:
            return PdfCrawlStageResult(
                version_id=version_id,
                status=StageStatus.FAILED,
                error=f"pdf_crawl requires a WEB_PAGE seed, got {document.type}",
            )

        existing = await _find_by_url(document.url)
        if existing is not None and _has_completed_pdf_crawl(existing):
            prior = _latest_pdf_crawl_result(existing)
            return PdfCrawlStageResult(
                version_id=version_id,
                status=StageStatus.COMPLETED,
                content_path=prior.content_path if prior else None,
            )

        try:
            body = await self._fetch_fn(url=document.url)
        except Exception as exc:
            logger.exception("Failed to fetch seed URL %s", document.url)
            return PdfCrawlStageResult(
                version_id=version_id,
                status=StageStatus.FAILED,
                error=f"Failed to fetch seed URL: {exc}",
            )

        kind = classify_content(body)
        if kind == "other":
            return PdfCrawlStageResult(
                version_id=version_id,
                status=StageStatus.FAILED,
                error="Seed URL content is neither PDF nor HTML",
            )

        if kind == "pdf":
            if existing is not None:
                return PdfCrawlStageResult(
                    version_id=version_id,
                    status=StageStatus.FAILED,
                    error=(
                        "Seed URL is already saved in Mongo but fetched content "
                        "is PDF; refusing to convert document type"
                    ),
                )
            if len(body) > cfg.max_pdf_size:
                return PdfCrawlStageResult(
                    version_id=version_id,
                    status=StageStatus.FAILED,
                    error=f"PDF exceeds max_pdf_size ({cfg.max_pdf_size} bytes)",
                )
            pdf_doc = PdfDocument(
                url=document.url,
                title=document.title,
                pipeline_name=pipeline_for_pdf,
                relations=list(document.relations),
            )
            content_path = await _persist_content(pdf_doc, body, cfg.save_dir)
            result = PdfCrawlStageResult(
                version_id=version_id,
                status=StageStatus.COMPLETED,
                content_path=content_path,
            )
            _append_stage_result(pdf_doc, result)
            await pdf_doc.save()
            return result

        # HTML seed
        seed_doc = await _ensure_web_page(
            document,
            existing=existing,
            pipeline_for_web=pipeline_for_web,
        )
        content_path = await _persist_content(seed_doc, body, cfg.web_page_save_dir)
        seed_result = PdfCrawlStageResult(
            version_id=version_id,
            status=StageStatus.COMPLETED,
            content_path=content_path,
        )
        _append_stage_result(seed_doc, seed_result)
        await seed_doc.save()

        page_text = _decode_html(body)
        child_urls = await self._extract_urls(
            page_url=seed_doc.url,
            page_body=page_text,
            cfg=cfg,
        )
        await self._bfs(
            seed_doc=seed_doc,
            initial_urls=child_urls,
            cfg=cfg,
            version_id=version_id,
            pipeline_for_web=pipeline_for_web,
            pipeline_for_pdf=pipeline_for_pdf,
        )
        # Refresh seed result after BFS may have updated relations.
        await seed_doc.save()
        return seed_result

    async def _ensure_version(self, cfg: PdfCrawlConfig) -> StageVersion:
        del cfg
        return await self.get_version()

    async def _extract_urls(
        self,
        *,
        page_url: str,
        page_body: str,
        cfg: PdfCrawlConfig,
    ) -> list[str]:
        if self._extract_fn is not None:
            return await self._extract_fn(
                page_url=page_url,
                page_body=page_body,
                max_urls=cfg.max_urls_per_page,
                max_retries=cfg.max_agent_retries,
                model=self._chat_model,
            )
        return await extract_page_urls(
            page_url=page_url,
            page_body=page_body,
            max_urls=cfg.max_urls_per_page,
            max_retries=cfg.max_agent_retries,
            model=self._chat_model,
        )

    async def _bfs(
        self,
        *,
        seed_doc: WebPageDocument,
        initial_urls: list[str],
        cfg: PdfCrawlConfig,
        version_id: str,
        pipeline_for_web: str,
        pipeline_for_pdf: str,
    ) -> None:
        # Queue stores (url, depth). Each URL is scheduled at most once via
        # ``visited``. Additional parents that discover the same URL are recorded
        # in ``parents_by_url`` and linked when the URL finishes (or immediately
        # if it is already COMPLETED).
        queue: deque[tuple[str, int]] = deque()
        visited: set[str] = {seed_doc.url}
        parents_by_url: dict[str, list[Document]] = {}
        pdf_count = 0

        async def schedule(
            *,
            raw_url: str,
            parent: Document,
            depth: int,
        ) -> None:
            absolute = urljoin(parent.url, raw_url)
            if absolute in visited:
                existing = await _find_by_url(absolute)
                if existing is not None and _has_completed_pdf_crawl(existing):
                    await _link_bidirectional(parent, existing)
                else:
                    parents_by_url.setdefault(absolute, []).append(parent)
                return
            visited.add(absolute)
            parents_by_url.setdefault(absolute, []).append(parent)
            queue.append((absolute, depth))

        for raw_url in initial_urls:
            await schedule(raw_url=raw_url, parent=seed_doc, depth=1)

        while queue:
            if pdf_count >= cfg.max_pdfs:
                logger.info("Stopping BFS: reached max_pdfs=%s", cfg.max_pdfs)
                break

            url, depth = queue.popleft()
            if depth > cfg.max_url_depth:
                parents_by_url.pop(url, None)
                continue

            parents = parents_by_url.pop(url, [])
            existing = await _find_by_url(url)
            if existing is not None and _has_completed_pdf_crawl(existing):
                for parent in parents:
                    await _link_bidirectional(parent, existing)
                continue

            try:
                body = await self._fetch_fn(url=url)
            except Exception:
                logger.exception("Failed to fetch %s; skipping", url)
                continue

            kind = classify_content(body)
            if kind == "other":
                logger.info("Discarding non-PDF/non-HTML URL %s", url)
                continue

            if kind == "pdf":
                if existing is not None and existing.type == DocumentType.WEB_PAGE:
                    logger.error(
                        "URL %s is saved as WEB_PAGE but content is PDF; skipping",
                        url,
                    )
                    continue
                if len(body) > cfg.max_pdf_size:
                    logger.warning("Skipping oversized PDF %s", url)
                    continue
                if existing is not None and existing.type == DocumentType.PDF:
                    pdf_doc = existing
                else:
                    pdf_doc = PdfDocument(
                        url=url,
                        pipeline_name=pipeline_for_pdf,
                    )
                content_path = await _persist_content(pdf_doc, body, cfg.save_dir)
                result = PdfCrawlStageResult(
                    version_id=version_id,
                    status=StageStatus.COMPLETED,
                    content_path=content_path,
                )
                _append_stage_result(pdf_doc, result)
                for parent in parents:
                    await _link_bidirectional(parent, pdf_doc)
                await pdf_doc.save()
                pdf_count += 1
                continue

            # HTML
            if existing is not None and existing.type == DocumentType.PDF:
                logger.error(
                    "URL %s is saved as PDF but content is HTML; skipping",
                    url,
                )
                continue
            if existing is not None:
                page_doc = existing
            else:
                page_doc = WebPageDocument(
                    url=url,
                    pipeline_name=pipeline_for_web,
                )
            content_path = await _persist_content(page_doc, body, cfg.web_page_save_dir)
            result = PdfCrawlStageResult(
                version_id=version_id,
                status=StageStatus.COMPLETED,
                content_path=content_path,
            )
            _append_stage_result(page_doc, result)
            for parent in parents:
                await _link_bidirectional(parent, page_doc)
            await page_doc.save()

            if depth >= cfg.max_url_depth:
                continue

            page_text = _decode_html(body)
            child_urls = await self._extract_urls(
                page_url=page_doc.url,
                page_body=page_text,
                cfg=cfg,
            )
            for raw_url in child_urls:
                await schedule(raw_url=raw_url, parent=page_doc, depth=depth + 1)


def classify_content(body: bytes) -> ContentKind:
    if body.startswith(PDF_MAGIC_BYTES):
        return "pdf"
    head = body.lstrip()[:512].lower()
    if head.startswith(HTML_MAGIC_BYTES) or b"<html" in head or b"<!doctype" in head:
        return "html"
    return "other"


def _decode_html(body: bytes) -> str:
    return body.decode("utf-8", errors="replace")


async def _find_by_url(url: str) -> Document | None:
    # Query concrete subclasses: root Document.find is unreliable with some
    # Mongo mocks / inheritance setups.
    page = await WebPageDocument.find_one(WebPageDocument.url == url)
    if page is not None:
        return page
    return await PdfDocument.find_one(PdfDocument.url == url)


def _has_completed_pdf_crawl(document: Document) -> bool:
    result = _latest_pdf_crawl_result(document)
    return result is not None and result.status == StageStatus.COMPLETED


def _latest_pdf_crawl_result(document: Document) -> PdfCrawlStageResult | None:
    results = document.stage_results.get(PDF_CRAWL_STAGE_NAME)
    if not results:
        return None
    latest = results[-1]
    if isinstance(latest, PdfCrawlStageResult):
        return latest
    return PdfCrawlStageResult.model_validate(latest.model_dump())


def _append_stage_result(document: Document, result: PdfCrawlStageResult) -> None:
    document.stage_results.setdefault(PDF_CRAWL_STAGE_NAME, []).append(result)


async def _ensure_web_page(
    seed: Document,
    *,
    existing: Document | None,
    pipeline_for_web: str,
) -> WebPageDocument:
    if existing is not None:
        if existing.type != DocumentType.WEB_PAGE:
            raise TypeError(
                f"Expected WEB_PAGE for HTML seed, found {existing.type} at {seed.url}"
            )
        return cast(WebPageDocument, existing)
    if isinstance(seed, WebPageDocument) and seed.id is None:
        seed.pipeline_name = pipeline_for_web
        await seed.insert()
        return seed
    page = WebPageDocument(
        url=seed.url,
        title=seed.title,
        pipeline_name=pipeline_for_web,
        relations=list(seed.relations),
    )
    await page.insert()
    return page


async def _persist_content(
    document: Document,
    body: bytes,
    directory: Path,
) -> str:
    if document.id is None:
        await document.insert()
    assert document.id is not None
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / str(document.id)
    path.write_bytes(body)
    return str(path)


def _relation_exists(
    document: Document,
    *,
    side: RelationSide,
    other_id: PydanticObjectId,
) -> bool:
    return any(
        rel.type == RelationType.URL_LINK and rel.side == side and rel.d_id == other_id
        for rel in document.relations
    )


async def _link_bidirectional(parent: Document, child: Document) -> None:
    assert parent.id is not None
    assert child.id is not None
    if not _relation_exists(parent, side=RelationSide.TO, other_id=child.id):
        parent.relations.append(
            Relation(
                type=RelationType.URL_LINK,
                side=RelationSide.TO,
                d_id=child.id,
                d_type=child.type,
            )
        )
    if not _relation_exists(child, side=RelationSide.FROM, other_id=parent.id):
        child.relations.append(
            Relation(
                type=RelationType.URL_LINK,
                side=RelationSide.FROM,
                d_id=parent.id,
                d_type=parent.type,
            )
        )
    await parent.save()
    await child.save()
