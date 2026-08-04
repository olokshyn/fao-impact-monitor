"""Tests for FaoRepository data source."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar
from unittest.mock import AsyncMock, MagicMock
from urllib.request import urlopen

import pytest

from fao_impact_monitor.agent.pdf_crawl_agent import PdfPageExtract
from fao_impact_monitor.config import PdfCrawlConfig
from fao_impact_monitor.data_lake.common import Status
from fao_impact_monitor.data_lake.document import (
    Document,
    DocumentType,
    Relation,
    RelationSide,
    RelationType,
)
from fao_impact_monitor.data_lake.documents.pdf_document import PdfDocument
from fao_impact_monitor.data_lake.documents.web_page_document import WebPageDocument
from fao_impact_monitor.data_lake.pipeline import (
    PIPELINE_PDF_CRAWL,
    PIPELINE_PDF_PROCESS,
    PdfCrawlPipeline,
    PipelineStep,
)
from fao_impact_monitor.data_lake.stages.pdf_crawl_stage import (
    PDF_CRAWL_STAGE_NAME,
    PIPELINE_FOR_PDF_PARAM,
    PIPELINE_FOR_WEB_PARAM,
    PdfCrawlStage,
)
from fao_impact_monitor.data_source import (
    FaoRepository,
    FaoRepositoryDataResult,
    FaoRepositoryDataSourceConfig,
    get_data_source,
)
from fao_impact_monitor.data_source.data_source import _DATA_SOURCE_CLS_REGISTRY
from fao_impact_monitor.data_source.fao_repository import (
    _encode_search_entry,
    _expand_repository_urls,
)
from fao_impact_monitor.metric import Metric
from tests.data_lake.mock_http_server import MockHttpServer

T = TypeVar("T")
RunAsync = Callable[[Coroutine[Any, Any, T]], T]


def _metric() -> Metric:
    return Metric(
        name="Drought impact",
        description="Agricultural drought effects",
        example="Yield declined.",
        unit="%",
        data_sources=[
            FaoRepositoryDataSourceConfig(
                source="FaoRepository",
                url="https://example.org/repo",
            )
        ],
    )


def test_fao_repository_is_registered() -> None:
    assert _DATA_SOURCE_CLS_REGISTRY["FaoRepository"] is FaoRepository
    source = get_data_source("FaoRepository")
    assert isinstance(source, FaoRepository)
    assert source.source == "FaoRepository"


def test_encode_search_string() -> None:
    assert _encode_search_entry("el nino") == "el%20nino"
    assert _encode_search_entry("drought") == "drought"


def test_encode_search_dict() -> None:
    assert (
        _encode_search_entry({"q": "el nino", "year": "2010"})
        == "q=el%20nino&year=2010"
    )


def test_expand_urls_no_searches() -> None:
    assert _expand_repository_urls("https://fao.org/repo", None) == [
        "https://fao.org/repo"
    ]
    assert _expand_repository_urls("https://fao.org/repo", []) == [
        "https://fao.org/repo"
    ]


def test_expand_urls_string_searches() -> None:
    urls = _expand_repository_urls(
        "https://fao.org/?q={search}",
        ["el nino", "drought"],
    )
    assert urls == [
        "https://fao.org/?q=el%20nino",
        "https://fao.org/?q=drought",
    ]


def test_expand_urls_dict_searches() -> None:
    urls = _expand_repository_urls(
        "https://fao.org/?{search}",
        [{"q": "el nino", "year": "2010"}],
    )
    assert urls == ["https://fao.org/?q=el%20nino&year=2010"]


def test_expand_urls_placeholder_without_searches_raises() -> None:
    with pytest.raises(ValueError, match="searches is empty"):
        _expand_repository_urls("https://fao.org/?q={search}", None)
    with pytest.raises(ValueError, match="searches is empty"):
        _expand_repository_urls("https://fao.org/?q={search}", [])


def test_expand_urls_searches_without_placeholder_raises() -> None:
    with pytest.raises(ValueError, match="does not contain"):
        _expand_repository_urls("https://fao.org/repo", ["drought"])


def test_get_data_runs_pipeline_and_collects_pdfs(
    document_store: dict[str, Document],
    run_async: RunAsync[Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del document_store
    seed_url = "https://example.org/repo"
    child_page_url = "https://example.org/page"
    direct_pdf_url = "https://example.org/direct.pdf"
    nested_pdf_url = "https://example.org/nested.pdf"
    completed_pdf_url = "https://example.org/completed.pdf"

    run_calls: list[str] = []

    async def fake_run(document: WebPageDocument) -> None:
        run_calls.append(document.url)
        if document.id is None:
            await document.insert()

        child_page = WebPageDocument(
            url=child_page_url,
            title="Child",
            pipeline_statuses={PIPELINE_PDF_CRAWL: Status.PENDING},
        )
        await child_page.insert()
        assert child_page.id is not None

        direct_pdf = PdfDocument(url=direct_pdf_url, title="Direct")
        await direct_pdf.insert()
        nested_pdf = PdfDocument(url=nested_pdf_url, title="Nested")
        await nested_pdf.insert()
        completed_pdf = PdfDocument(
            url=completed_pdf_url,
            title="Completed",
            pipeline_statuses={PIPELINE_PDF_CRAWL: Status.COMPLETED},
        )
        await completed_pdf.insert()
        assert direct_pdf.id is not None
        assert nested_pdf.id is not None
        assert completed_pdf.id is not None

        document.relations = [
            Relation(
                type=RelationType.URL_LINK,
                side=RelationSide.TO,
                d_id=direct_pdf.id,
                d_type=DocumentType.PDF,
            ),
            Relation(
                type=RelationType.URL_LINK,
                side=RelationSide.TO,
                d_id=child_page.id,
                d_type=DocumentType.WEB_PAGE,
            ),
            Relation(
                type=RelationType.URL_LINK,
                side=RelationSide.TO,
                d_id=completed_pdf.id,
                d_type=DocumentType.PDF,
            ),
        ]
        child_page.relations = [
            Relation(
                type=RelationType.URL_LINK,
                side=RelationSide.TO,
                d_id=nested_pdf.id,
                d_type=DocumentType.PDF,
            ),
        ]
        document.set_pipeline_status(PIPELINE_PDF_CRAWL, Status.COMPLETED)
        await document.save()
        await child_page.save()

    pipeline = MagicMock()
    pipeline.run = AsyncMock(side_effect=fake_run)
    monkeypatch.setattr(
        "fao_impact_monitor.data_source.fao_repository.get_pipeline",
        lambda _name: pipeline,
    )

    source = FaoRepository()
    results = run_async(
        source.get_data(
            _metric(),
            FaoRepositoryDataSourceConfig(source="FaoRepository", url=seed_url),
            "KEN",
        )
    )

    assert run_calls == [seed_url]
    assert len(results) == 3
    assert all(isinstance(r, FaoRepositoryDataResult) for r in results)
    urls = {r.url for r in results}
    assert urls == {direct_pdf_url, nested_pdf_url, completed_pdf_url}
    titles = {r.title for r in results}
    assert titles == {"Direct", "Nested", "Completed"}
    assert all(isinstance(r.document, PdfDocument) for r in results)


def test_get_data_runs_sequentially_for_each_search(
    document_store: dict[str, Document],
    run_async: RunAsync[Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del document_store
    run_calls: list[str] = []

    async def fake_run(document: WebPageDocument) -> None:
        run_calls.append(document.url)
        if document.id is None:
            await document.insert()
        document.set_pipeline_status(PIPELINE_PDF_CRAWL, Status.COMPLETED)
        await document.save()

    pipeline = MagicMock()
    pipeline.run = AsyncMock(side_effect=fake_run)
    monkeypatch.setattr(
        "fao_impact_monitor.data_source.fao_repository.get_pipeline",
        lambda _name: pipeline,
    )

    source = FaoRepository()
    results = run_async(
        source.get_data(
            _metric(),
            FaoRepositoryDataSourceConfig(
                source="FaoRepository",
                url="https://fao.org/?q={search}",
                searches=["el nino", "drought"],
            ),
            "KEN",
        )
    )

    assert run_calls == [
        "https://fao.org/?q=el%20nino",
        "https://fao.org/?q=drought",
    ]
    assert results == []


def test_get_data_dedupes_pdfs_across_seeds(
    document_store: dict[str, Document],
    run_async: RunAsync[Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del document_store
    shared_pdf_url = "https://example.org/shared.pdf"

    async def fake_run(document: WebPageDocument) -> None:
        if document.id is None:
            await document.insert()
        existing = await PdfDocument.find_one(PdfDocument.url == shared_pdf_url)
        if existing is None:
            pdf = PdfDocument(url=shared_pdf_url, title="Shared")
            await pdf.insert()
        else:
            pdf = existing
        assert pdf.id is not None
        document.relations = [
            Relation(
                type=RelationType.URL_LINK,
                side=RelationSide.TO,
                d_id=pdf.id,
                d_type=DocumentType.PDF,
            ),
        ]
        document.set_pipeline_status(PIPELINE_PDF_CRAWL, Status.COMPLETED)
        await document.save()

    pipeline = MagicMock()
    pipeline.run = AsyncMock(side_effect=fake_run)
    monkeypatch.setattr(
        "fao_impact_monitor.data_source.fao_repository.get_pipeline",
        lambda _name: pipeline,
    )

    source = FaoRepository()
    results = run_async(
        source.get_data(
            _metric(),
            FaoRepositoryDataSourceConfig(
                source="FaoRepository",
                url="https://fao.org/?q={search}",
                searches=["a", "b"],
            ),
            "KEN",
        )
    )

    assert len(results) == 1
    assert results[0].url == shared_pdf_url


def _simple_topology(server: MockHttpServer) -> str:
    server.add_html(
        "/",
        f"""<!doctype html>
<html><body>
  <a href="{server.url("/report.pdf")}">Download report PDF</a>
  <a href="{server.url("/page")}">More docs</a>
</body></html>
""",
    )
    server.add_html(
        "/page",
        f"""<!doctype html>
<html><body>
  <a href="{server.url("/annex.pdf")}">Download annex PDF</a>
</body></html>
""",
    )
    server.add_pdf("/report.pdf")
    server.add_pdf("/annex.pdf")
    return server.url("/")


@pytest.mark.integration
def test_get_data_integration_local_crawl(
    http_server: MockHttpServer,
    document_store: dict[str, Document],
    pdf_crawl_dirs: PdfCrawlConfig,
    run_async: RunAsync[Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run real PdfCrawlPipeline against a local mock HTTP topology."""
    del document_store
    root_url = _simple_topology(http_server)

    async def fetch(*, url: str) -> bytes:
        def _get() -> bytes:
            with urlopen(url) as response:
                return bytes(response.read())

        return await asyncio.to_thread(_get)

    async def extract_links(
        *,
        page_url: str,
        page_body: str,
        max_urls: int,
        max_retries: int,
        model: Any,
    ) -> PdfPageExtract:
        del page_body, max_urls, max_retries, model
        path = page_url[len(http_server.base_url) :] or "/"
        if path == "/":
            return PdfPageExtract(
                urls=[http_server.url("/report.pdf"), http_server.url("/page")]
            )
        if path == "/page":
            return PdfPageExtract(urls=[http_server.url("/annex.pdf")])
        return PdfPageExtract()

    stage = PdfCrawlStage(
        fetch_fn=fetch,
        extract_fn=extract_links,
        config=pdf_crawl_dirs,
    )

    def _get_stage(name: str) -> PdfCrawlStage:
        assert name == PDF_CRAWL_STAGE_NAME
        return stage

    monkeypatch.setattr(
        "fao_impact_monitor.data_lake.pipeline.get_stage",
        _get_stage,
    )

    async def no_cascade(self: PdfCrawlPipeline, document: Document) -> None:
        del self, document

    monkeypatch.setattr(PdfCrawlPipeline, "_cascade_children", no_cascade)

    pipeline = PdfCrawlPipeline.model_construct(
        name=PIPELINE_PDF_CRAWL,
        steps=[
            PipelineStep(
                stage_name=PDF_CRAWL_STAGE_NAME,
                params={
                    PIPELINE_FOR_WEB_PARAM: PIPELINE_PDF_CRAWL,
                    PIPELINE_FOR_PDF_PARAM: PIPELINE_PDF_PROCESS,
                },
            )
        ],
    )
    monkeypatch.setattr(
        "fao_impact_monitor.data_source.fao_repository.get_pipeline",
        lambda _name: pipeline,
    )

    source = FaoRepository()
    results = run_async(
        source.get_data(
            _metric(),
            FaoRepositoryDataSourceConfig(source="FaoRepository", url=root_url),
            "KEN",
        )
    )

    assert len(results) == 2
    assert all(isinstance(r, FaoRepositoryDataResult) for r in results)
    urls = {r.url for r in results}
    assert urls == {http_server.url("/report.pdf"), http_server.url("/annex.pdf")}
    assert all(r.document.type == DocumentType.PDF for r in results)
