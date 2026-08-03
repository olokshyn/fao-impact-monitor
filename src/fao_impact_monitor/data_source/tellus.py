"""Tellus data source: search FAO Tellus and process matching documents."""

from __future__ import annotations

import logging
from typing import Any

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
        query = f"{country}: {metric.description}"
        response = await tellus_search_chunks(query)
        chunks = response.get("chunks") or []
        if not isinstance(chunks, list):
            raise TypeError(f"Tellus search chunks must be a list, got {type(chunks)}")
        chunk_dicts = [c for c in chunks if isinstance(c, dict)]
        matched_by_doc = _group_matched_pages(chunk_dicts)

        results: list[DataResult] = []
        pipeline = get_pipeline(PIPELINE_TELLUS_PROCESS)
        for document_id, new_pages in matched_by_doc.items():
            existing = await find_tellus_document_by_external_id(document_id)
            if existing is not None:
                existing.matched_pages = sorted(
                    set(existing.matched_pages) | set(new_pages)
                )
                await existing.save()
                doc = existing
            else:
                doc = TellusDocument(
                    url=f"tellus://{document_id}",
                    external_id=document_id,
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
            results.append(_to_data_result(refreshed))
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
