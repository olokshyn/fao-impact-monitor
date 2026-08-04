"""Unit tests for TellusDataSource."""

from __future__ import annotations

import asyncio
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

_GENERATED_QUERIES = [
    "water stress indicators measurement methods",
    "water resources availability quantitative trends",
    "drivers of water scarcity agriculture",
]


def _metric(
    name: str = "Water stress",
    description: str = "Water resources availability",
    example: str = "Water stress increased.",
    unit: str = "%",
) -> Metric:
    return Metric(
        name=name,
        description=description,
        example=example,
        unit=unit,
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

    searched: list[tuple[str, list[str] | None]] = []

    async def fake_generate_queries(**kwargs: Any) -> list[str]:
        assert kwargs["research_question"] == "Water stress"
        assert "Water resources availability" in kwargs["explanation"]
        assert "quantitative results in %" in kwargs["explanation"]
        assert kwargs["example"] == "Water stress increased."
        return list(_GENERATED_QUERIES)

    async def fake_search(
        query: str,
        countries_iso3: list[str] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del args, kwargs
        searched.append((query, countries_iso3))
        if query == _GENERATED_QUERIES[0]:
            return {
                "chunks": [
                    {"document_id": "doc-1", "page_num": 2},
                    {"document_id": "doc-1", "page_num": 5},
                ],
                "documents": [],
            }
        if query == _GENERATED_QUERIES[1]:
            return {
                "chunks": [
                    {"document_id": "doc-2", "page_num": 1},
                ],
                "documents": [],
            }
        return {"chunks": [], "documents": []}

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
        "fao_impact_monitor.data_source.tellus.generate_queries",
        fake_generate_queries,
    )
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

    assert sorted(q for q, _ in searched) == sorted(_GENERATED_QUERIES)
    assert all(countries == ["KEN"] for _, countries in searched)

    refreshed = run_async(_find_by_external_id("doc-1"))
    assert refreshed is not None
    assert refreshed.matched_pages == [1, 2, 5]

    doc2 = run_async(_find_by_external_id("doc-2"))
    assert doc2 is not None
    assert doc2.matched_pages == [1]
    assert set(run_calls) == {"doc-1", "doc-2"}
    assert len(results) == 2
    assert {r.title for r in results} == {"Existing", "Title doc-2"}


def test_get_data_runs_generated_queries(monkeypatch: pytest.MonkeyPatch) -> None:
    searched: list[tuple[str, list[str] | None]] = []

    async def fake_generate_queries(**kwargs: Any) -> list[str]:
        del kwargs
        return list(_GENERATED_QUERIES)

    async def fake_search(
        query: str,
        countries_iso3: list[str] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del args, kwargs
        searched.append((query, countries_iso3))
        return {"chunks": [], "documents": []}

    monkeypatch.setattr(
        "fao_impact_monitor.data_source.tellus.generate_queries",
        fake_generate_queries,
    )
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
    assert sorted(q for q, _ in searched) == sorted(_GENERATED_QUERIES)
    assert all(countries == ["KEN"] for _, countries in searched)


def test_get_data_for_metrics_merges_across_metrics(
    document_store: dict[str, Document],
    run_async: RunAsync[Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del document_store
    generated_for: list[str] = []

    async def fake_generate_queries(**kwargs: Any) -> list[str]:
        name = kwargs["research_question"]
        generated_for.append(name)
        if name == "Cropland":
            return ["cropland drought query"]
        if name == "Pastureland":
            return ["pastureland flood query"]
        return []

    async def fake_search(
        query: str,
        countries_iso3: list[str] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del countries_iso3, args, kwargs
        if query == "cropland drought query":
            return {
                "chunks": [
                    {"document_id": "shared-doc", "page_num": 3},
                    {"document_id": "crop-only", "page_num": 1},
                ],
                "documents": [],
            }
        if query == "pastureland flood query":
            return {
                "chunks": [
                    {"document_id": "shared-doc", "page_num": 7},
                ],
                "documents": [],
            }
        return {"chunks": [], "documents": []}

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
        "fao_impact_monitor.data_source.tellus.generate_queries",
        fake_generate_queries,
    )
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
        source.get_data_for_metrics(
            [
                _metric(name="Cropland", description="Cropland affected"),
                _metric(name="Pastureland", description="Pastureland affected"),
            ],
            "KEN",
        )
    )

    assert sorted(generated_for) == ["Cropland", "Pastureland"]
    assert set(run_calls) == {"shared-doc", "crop-only"}
    assert len(results) == 2

    shared = run_async(_find_by_external_id("shared-doc"))
    assert shared is not None
    assert shared.matched_pages == [3, 7]


def test_search_and_pipeline_respect_max_requests(
    document_store: dict[str, Document],
    run_async: RunAsync[Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del document_store
    max_requests = 2
    search_inflight = 0
    search_peak = 0
    pipeline_inflight = 0
    pipeline_peak = 0

    async def fake_generate_queries(**kwargs: Any) -> list[str]:
        del kwargs
        return [f"query-{i}" for i in range(5)]

    async def fake_search(
        query: str,
        countries_iso3: list[str] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del countries_iso3, args, kwargs
        nonlocal search_inflight, search_peak
        search_inflight += 1
        search_peak = max(search_peak, search_inflight)
        await asyncio.sleep(0.01)
        search_inflight -= 1
        return {
            "chunks": [{"document_id": f"doc-{query}", "page_num": 1}],
            "documents": [],
        }

    async def fake_run(document: TellusDocument) -> None:
        nonlocal pipeline_inflight, pipeline_peak
        pipeline_inflight += 1
        pipeline_peak = max(pipeline_peak, pipeline_inflight)
        await asyncio.sleep(0.01)
        pipeline_inflight -= 1
        assert document.external_id is not None
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
        "fao_impact_monitor.data_source.tellus.generate_queries",
        fake_generate_queries,
    )
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
        source.get_data_for_metrics(
            [_metric()],
            "KEN",
            tellus_max_requests=max_requests,
        )
    )

    assert len(results) == 5
    assert search_peak <= max_requests
    assert pipeline_peak <= max_requests


async def _find_by_external_id(external_id: str) -> TellusDocument | None:
    return await TellusDocument.find_one(TellusDocument.external_id == external_id)
