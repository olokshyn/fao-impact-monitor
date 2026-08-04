"""Tellus data source: search FAO Tellus and process matching documents."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from fao_impact_monitor.agent.query_generator_agent import generate_queries
from fao_impact_monitor.data_lake.common import Status
from fao_impact_monitor.data_lake.documents.tellus_document import TellusDocument
from fao_impact_monitor.data_lake.pipeline import (
    PIPELINE_TELLUS_PROCESS,
    find_tellus_document_by_external_id,
    get_pipeline,
)
from fao_impact_monitor.data_provider.tellus_provider import tellus_search_chunks
from fao_impact_monitor.data_source.data_source import DataResult, DataSource
from fao_impact_monitor.data_source.data_source_config import DataSourceConfig
from fao_impact_monitor.metric.metric import Metric
from fao_impact_monitor.utils.country import iso3_to_country_name

logger = logging.getLogger(__name__)


def _group_matched_pages(
    chunks: list[dict[str, Any]],
) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for chunk in chunks:
        document_id = str(chunk.get("document_id") or "").strip()
        if not document_id:
            continue
        pages = grouped.setdefault(document_id, [])
        page_num = chunk.get("page_num")
        if isinstance(page_num, int) and page_num not in pages:
            pages.append(page_num)
    return grouped


def _merge_matched_pages(
    into: dict[str, list[int]],
    incoming: dict[str, list[int]],
) -> None:
    for document_id, pages in incoming.items():
        existing = into.setdefault(document_id, [])
        for page in pages:
            if page not in existing:
                existing.append(page)


async def _map_semaphore[T, R](
    limit: int,
    items: Sequence[T],
    mapper: Callable[[T], Awaitable[R]],
) -> list[R]:
    sem = asyncio.Semaphore(limit)

    async def run(item: T) -> R:
        async with sem:
            return await mapper(item)

    return list(await asyncio.gather(*(run(item) for item in items)))


def _queries_for_metric(metric: Metric) -> Awaitable[list[str]]:
    return generate_queries(
        research_question=metric.name,
        explanation=(
            f"{metric.description} "
            f"In your answer provide quantitative results in {metric.unit}."
        ),
        example=metric.example,
    )


class TellusDataSource(DataSource):
    source: str = "Tellus"

    async def get_data(
        self,
        metric: Metric,
        data_source_config: DataSourceConfig,
        country_iso3: str,
        *,
        year_start: int | None = None,
        year_end: int | None = None,
    ) -> list[DataResult]:
        del data_source_config, year_start, year_end
        return await self.get_data_for_metrics([metric], country_iso3)

    async def get_data_for_metrics(
        self,
        metrics: list[Metric],
        country_iso3: str,
        *,
        tellus_max_requests: int = 4,
    ) -> list[DataResult]:
        if not metrics:
            return []

        country = iso3_to_country_name(country_iso3)
        metric_names = [m.name for m in metrics]
        logger.info(
            "TellusDataSource: country=%s (%s) metrics=%s tellus_max_requests=%s",
            country_iso3,
            country,
            metric_names,
            tellus_max_requests,
        )

        query_lists = await asyncio.gather(
            *(_queries_for_metric(metric) for metric in metrics)
        )
        for metric, queries in zip(metrics, query_lists, strict=True):
            logger.info(
                "TellusDataSource: metric=%r queries=%s",
                metric.name,
                queries,
            )

        all_queries = [q for queries in query_lists for q in queries]

        async def search(query: str) -> dict[str, Any]:
            return await tellus_search_chunks(
                query,
                countries_iso3=[country_iso3],
            )

        responses = await _map_semaphore(tellus_max_requests, all_queries, search)

        matched_by_doc: dict[str, list[int]] = {}
        total_chunks = 0
        for response in responses:
            chunks = response.get("chunks") or []
            if not isinstance(chunks, list):
                raise TypeError(
                    f"Tellus search chunks must be a list, got {type(chunks)}"
                )
            chunk_dicts = [c for c in chunks if isinstance(c, dict)]
            total_chunks += len(chunk_dicts)
            _merge_matched_pages(matched_by_doc, _group_matched_pages(chunk_dicts))

        logger.info(
            "TellusDataSource: %s search(es), %s chunk(s) → %s unique document(s)",
            len(all_queries),
            total_chunks,
            len(matched_by_doc),
        )

        pipeline = get_pipeline(PIPELINE_TELLUS_PROCESS)
        total = len(matched_by_doc)
        items = list(enumerate(matched_by_doc.items(), start=1))

        async def process_document(
            item: tuple[int, tuple[str, list[int]]],
        ) -> DataResult | None:
            index, (document_id, new_pages) = item
            logger.info(
                "TellusDataSource: [%s/%s] pipeline %s for document_id=%s "
                "matched_pages=%s",
                index,
                total,
                PIPELINE_TELLUS_PROCESS,
                document_id,
                new_pages,
            )
            existing = await find_tellus_document_by_external_id(document_id)
            if existing is not None:
                existing.matched_pages = sorted(
                    set(existing.matched_pages) | set(new_pages)
                )
                if existing.source is None:
                    existing.source = self.source
                await existing.save()
                doc = existing
            else:
                doc = TellusDocument(
                    url=f"tellus://{document_id}",
                    external_id=document_id,
                    source=self.source,
                    matched_pages=sorted(new_pages),
                    pipeline_statuses={PIPELINE_TELLUS_PROCESS: Status.PENDING},
                )

            await pipeline.run(doc)
            refreshed = await find_tellus_document_by_external_id(document_id)
            if refreshed is None:
                logger.warning(
                    "Tellus document %s missing after pipeline run", document_id
                )
                return None
            status = refreshed.pipeline_status(PIPELINE_TELLUS_PROCESS)
            logger.info(
                "TellusDataSource: [%s/%s] document_id=%s pipeline_status=%s",
                index,
                total,
                document_id,
                status,
            )
            return _to_data_result(refreshed)

        processed = await _map_semaphore(
            tellus_max_requests,
            items,
            process_document,
        )
        results = [r for r in processed if r is not None]

        pipeline.log_stage_stats()
        logger.info(
            "TellusDataSource: done — returned %s DataResult(s) from %s document(s)",
            len(results),
            total,
        )
        return results


def _to_data_result(document: TellusDocument) -> DataResult:
    meta_url = document.metadata.get("url")
    url = meta_url if isinstance(meta_url, str) and meta_url else document.url
    return DataResult(
        source="Tellus",
        title=document.title,
        url=url,
        citation=document.citation,
        metadata=dict(document.metadata),
    )
