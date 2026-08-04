"""Tellus data source: search FAO Tellus and process matching documents."""

from __future__ import annotations

import logging
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
        country = iso3_to_country_name(country_iso3)
        queries = await generate_queries(
            research_question=metric.name,
            explanation=(
                f"{metric.description} "
                f"In your answer provide quantitative results in {metric.unit}."
            ),
            example=metric.example,
        )
        logger.info(
            "TellusDataSource: country=%s (%s) metric=%r queries=%s",
            country_iso3,
            country,
            metric.name,
            queries,
        )

        matched_by_doc: dict[str, list[int]] = {}
        total_chunks = 0
        for query in queries:
            response = await tellus_search_chunks(
                query,
                countries_iso3=[country_iso3],
            )
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
            len(queries),
            total_chunks,
            len(matched_by_doc),
        )

        results: list[DataResult] = []
        pipeline = get_pipeline(PIPELINE_TELLUS_PROCESS)
        total = len(matched_by_doc)
        for index, (document_id, new_pages) in enumerate(
            matched_by_doc.items(), start=1
        ):
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
                continue
            status = refreshed.pipeline_status(PIPELINE_TELLUS_PROCESS)
            logger.info(
                "TellusDataSource: [%s/%s] document_id=%s pipeline_status=%s",
                index,
                total,
                document_id,
                status,
            )
            results.append(_to_data_result(refreshed))

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
