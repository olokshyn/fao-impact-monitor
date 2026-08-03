"""Unit tests for TellusDataSource."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any, TypeVar
from unittest.mock import AsyncMock, MagicMock

import pytest

from fao_impact_monitor.data_lake.common import Status
from fao_impact_monitor.data_lake.document import Document
from fao_impact_monitor.data_lake.documents.tellus_document import TellusDocument
from fao_impact_monitor.data_lake.pipeline import PIPELINE_TELLUS_PROCESS
from fao_impact_monitor.data_source import TellusDataSource, get_data_source
from fao_impact_monitor.data_source.data_source import _DATA_SOURCE_CLS_REGISTRY
from fao_impact_monitor.data_source.data_source_config import DataSourceConfig
from fao_impact_monitor.metric import Metric
from fao_impact_monitor.utils.country import iso3_to_country_name

T = TypeVar("T")
RunAsync = Callable[[Coroutine[Any, Any, T]], T]


def _metric() -> Metric:
    return Metric(
        name="Water stress",
        description="Water resources availability",
        example="Water stress increased.",
        unit="%",
        data_sources=[DataSourceConfig(source="Tellus")],
    )


def test_tellus_is_registered() -> None:
    assert _DATA_SOURCE_CLS_REGISTRY["Tellus"] is TellusDataSource
    source = get_data_source("Tellus")
    assert isinstance(source, TellusDataSource)
    assert source.source == "Tellus"


def test_official_country_name_kenya() -> None:
    assert iso3_to_country_name("KEN") == "Republic of Kenya"


def test_get_data_searches_and_unions_matched_pages(
    document_store: dict[str, Document],
    run_async: RunAsync[Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del document_store
    existing = TellusDocument(
        url="tellus://doc-1",
        external_id="doc-1",
        matched_pages=[1, 2],
        title="Existing",
        metadata={
            "url": "https://example.org/doc-1",
            "citation": "FAO, Existing. - 2020 https://example.org/doc-1",
        },
        pipeline_statuses={PIPELINE_TELLUS_PROCESS: Status.COMPLETED},
    )
    run_async(existing.insert())

    async def fake_search(query: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        country = iso3_to_country_name("KEN")
        assert query == f"{country}: Water resources availability"
        return {
            "chunks": [
                {"document_id": "doc-1", "page_num": 2},
                {"document_id": "doc-1", "page_num": 5},
                {"document_id": "doc-2", "page_num": 1},
            ],
            "documents": [],
        }

    run_calls: list[str] = []

    async def fake_run(document: TellusDocument) -> None:
        assert document.external_id is not None
        run_calls.append(document.external_id)
        if document.id is None:
            document.title = f"Title {document.external_id}"
            document.metadata = {
                "url": f"https://example.org/{document.external_id}",
                "citation": f"cite-{document.external_id}",
            }
            document.set_pipeline_status(PIPELINE_TELLUS_PROCESS, Status.COMPLETED)
            await document.insert()
        else:
            await document.save()

    pipeline = MagicMock()
    pipeline.run = AsyncMock(side_effect=fake_run)

    monkeypatch.setattr(
        "fao_impact_monitor.data_source.tellus.tellus_search_chunks",
        fake_search,
    )
    monkeypatch.setattr(
        "fao_impact_monitor.data_source.tellus.get_pipeline",
        lambda _name: pipeline,
    )

    source = TellusDataSource()
    results = run_async(
        source.get_data(
            _metric(),
            DataSourceConfig(source="Tellus"),
            "KEN",
        )
    )

    refreshed = run_async(_find_by_external_id("doc-1"))
    assert refreshed is not None
    assert refreshed.matched_pages == [1, 2, 5]

    doc2 = run_async(_find_by_external_id("doc-2"))
    assert doc2 is not None
    assert doc2.matched_pages == [1]
    assert set(run_calls) == {"doc-1", "doc-2"}
    assert len(results) == 2
    assert {r.title for r in results} == {"Existing", "Title doc-2"}


def test_get_data_query_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    captured: dict[str, str] = {}

    async def fake_search(query: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        captured["query"] = query
        return {"chunks": [], "documents": []}

    monkeypatch.setattr(
        "fao_impact_monitor.data_source.tellus.tellus_search_chunks",
        fake_search,
    )
    monkeypatch.setattr(
        "fao_impact_monitor.data_source.tellus.get_pipeline",
        lambda _name: MagicMock(run=AsyncMock()),
    )

    source = TellusDataSource()
    results = asyncio.run(
        source.get_data(
            _metric(),
            DataSourceConfig(source="Tellus"),
            "KEN",
        )
    )
    assert results == []
    assert captured["query"] == "Republic of Kenya: Water resources availability"


async def _find_by_external_id(external_id: str) -> TellusDocument | None:
    return await TellusDocument.find_one(TellusDocument.external_id == external_id)
