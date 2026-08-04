"""FAO Knowledge Repository data source via PDF crawl pipeline."""

from __future__ import annotations

import logging
from collections import deque
from urllib.parse import quote

from beanie import PydanticObjectId

from fao_impact_monitor.data_lake.common import Status
from fao_impact_monitor.data_lake.document import (
    Document,
    DocumentType,
    RelationSide,
    RelationType,
)
from fao_impact_monitor.data_lake.documents.pdf_document import PdfDocument
from fao_impact_monitor.data_lake.documents.web_page_document import WebPageDocument
from fao_impact_monitor.data_lake.pipeline import PIPELINE_PDF_CRAWL, get_pipeline
from fao_impact_monitor.data_source.data_source import DataResult, DataSource
from fao_impact_monitor.data_source.data_source_config import DataSourceConfig
from fao_impact_monitor.metric.metric import Metric

logger = logging.getLogger(__name__)

SEARCH_PLACEHOLDER = "{search}"


class FaoRepositoryDataSourceConfig(DataSourceConfig):
    url: str
    searches: list[str | dict[str, str]] | None = None


class FaoRepositoryDataResult(DataResult):
    document: PdfDocument


class FaoRepository(DataSource):
    source: str = "FaoRepository"

    async def get_data(
        self,
        metric: Metric,
        data_source_config: DataSourceConfig,
        country_iso3: str,
        *,
        year_start: int | None = None,
        year_end: int | None = None,
    ) -> list[DataResult]:
        del metric, country_iso3, year_start, year_end
        config = FaoRepositoryDataSourceConfig.model_validate(
            data_source_config.model_dump()
        )
        seed_urls = _expand_repository_urls(config.url, config.searches)
        pipeline = get_pipeline(PIPELINE_PDF_CRAWL)
        logger.info(
            "FaoRepository: crawling %s seed URL(s) via %s",
            len(seed_urls),
            PIPELINE_PDF_CRAWL,
        )

        seen_ids: set[PydanticObjectId] = set()
        results: list[DataResult] = []
        for index, seed_url in enumerate(seed_urls, start=1):
            logger.info(
                "FaoRepository: [%s/%s] seed %s",
                index,
                len(seed_urls),
                seed_url,
            )
            seed = WebPageDocument(
                url=seed_url,
                source=self.source,
                pipeline_statuses={PIPELINE_PDF_CRAWL: Status.PENDING},
            )
            await pipeline.run(seed)
            refreshed = await WebPageDocument.find_one(WebPageDocument.url == seed_url)
            if refreshed is None:
                logger.warning("Seed page missing after crawl: %s", seed_url)
                continue
            for pdf in await _collect_reachable_pdfs(refreshed):
                if pdf.id is None or pdf.id in seen_ids:
                    continue
                seen_ids.add(pdf.id)
                results.append(_to_data_result(pdf))

        pipeline.log_stage_stats()
        logger.info(
            "FaoRepository: done — returned %s PDF DataResult(s)",
            len(results),
        )
        return results


def _encode_search_entry(entry: str | dict[str, str]) -> str:
    if isinstance(entry, str):
        return quote(entry, safe="")
    return "&".join(f"{key}={quote(value, safe='')}" for key, value in entry.items())


def _expand_repository_urls(
    url: str,
    searches: list[str | dict[str, str]] | None,
) -> list[str]:
    has_placeholder = SEARCH_PLACEHOLDER in url
    has_searches = bool(searches)
    if has_placeholder and not has_searches:
        raise ValueError(
            f"URL contains {SEARCH_PLACEHOLDER!r} but searches is empty or missing"
        )
    if has_searches and not has_placeholder:
        raise ValueError(
            f"searches is non-empty but URL does not contain {SEARCH_PLACEHOLDER!r}"
        )
    if not has_searches:
        return [url]
    assert searches is not None
    return [
        url.replace(SEARCH_PLACEHOLDER, _encode_search_entry(entry))
        for entry in searches
    ]


async def _collect_reachable_pdfs(seed: Document) -> list[PdfDocument]:
    """BFS over URL_LINK TO relations; collect PDF docs (including re-linked)."""
    pdfs: list[PdfDocument] = []
    seen_pdf_ids: set[PydanticObjectId] = set()
    visited_page_ids: set[PydanticObjectId] = set()
    queue: deque[Document] = deque([seed])

    while queue:
        current = queue.popleft()
        if current.id is None:
            continue
        if current.type == DocumentType.WEB_PAGE:
            if current.id in visited_page_ids:
                continue
            visited_page_ids.add(current.id)

        for rel in current.relations:
            if rel.type != RelationType.URL_LINK or rel.side != RelationSide.TO:
                continue
            if rel.d_type == DocumentType.PDF:
                if rel.d_id in seen_pdf_ids:
                    continue
                pdf = await PdfDocument.get(rel.d_id)
                if pdf is None:
                    continue
                seen_pdf_ids.add(rel.d_id)
                pdfs.append(pdf)
            elif rel.d_type == DocumentType.WEB_PAGE:
                if rel.d_id in visited_page_ids:
                    continue
                page = await WebPageDocument.get(rel.d_id)
                if page is not None:
                    queue.append(page)
    return pdfs


def _to_data_result(document: PdfDocument) -> FaoRepositoryDataResult:
    return FaoRepositoryDataResult(
        source="FaoRepository",
        title=document.title,
        url=document.url,
        citation=document.citation,
        metadata=dict(document.metadata),
        document=document,
    )
