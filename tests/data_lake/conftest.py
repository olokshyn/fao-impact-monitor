import asyncio
from collections.abc import Callable, Coroutine, Iterator
from typing import Any, TypeVar

import mongomock
import pytest
from beanie import init_beanie
from mongomock_motor import AsyncMongoMockClient

from fao_impact_monitor.config import PdfCrawlConfig, PdfExtractConfig
from fao_impact_monitor.data_lake.document import Document
from fao_impact_monitor.data_lake.documents.pdf_document import PdfDocument
from fao_impact_monitor.data_lake.documents.web_page_document import WebPageDocument
from fao_impact_monitor.data_lake.stage import StageVersion
from fao_impact_monitor.data_lake.stages.pdf_crawl_stage import PdfCrawlStageVersion
from fao_impact_monitor.data_lake.stages.pdf_extract_stage import PdfExtractStageVersion
from tests.data_lake.mock_http_server import MockHttpServer, mock_http_server

T = TypeVar("T")

# Beanie passes authorizedCollections=; mongomock does not accept it.
_orig_list_collection_names = mongomock.Database.list_collection_names


def _patched_list_collection_names(
    self: Any,
    filter: dict[str, Any] | None = None,
    **kwargs: Any,
) -> list[str]:
    kwargs.pop("authorizedCollections", None)
    return _orig_list_collection_names(self, filter=filter)


mongomock.Database.list_collection_names = _patched_list_collection_names  # type: ignore[assignment]


@pytest.fixture
def http_server() -> Iterator[MockHttpServer]:
    yield from mock_http_server()


@pytest.fixture
def pdf_crawl_dirs(tmp_path: Any) -> PdfCrawlConfig:
    return PdfCrawlConfig(
        save_dir=tmp_path / "pdf",
        web_page_save_dir=tmp_path / "web_page",
        max_url_depth=10,
        max_pdfs=50,
        max_urls_per_page=10,
        max_agent_retries=3,
        llm_model="openai:openai.gpt-5.6-luna",
    )


@pytest.fixture
def pdf_extract_dirs(tmp_path: Any) -> PdfExtractConfig:
    return PdfExtractConfig(save_dir=tmp_path / "pdf_markdown")


@pytest.fixture
def event_loop() -> Iterator[asyncio.AbstractEventLoop]:
    """Dedicated loop so mongomock-motor and Beanie share one runtime per test."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()
    asyncio.set_event_loop(None)


@pytest.fixture
def run_async(
    event_loop: asyncio.AbstractEventLoop,
) -> Callable[[Coroutine[Any, Any, T]], T]:
    def _run(coro: Coroutine[Any, Any, T]) -> T:
        return event_loop.run_until_complete(coro)

    return _run


@pytest.fixture
def document_store(
    event_loop: asyncio.AbstractEventLoop,
) -> Iterator[dict[str, Document]]:
    """Initialize Beanie on mongomock and expose URL→document snapshots."""

    async def _setup() -> None:
        client: Any = AsyncMongoMockClient()
        await init_beanie(
            database=client["fao_impact_monitor_test"],
            document_models=[
                Document,
                WebPageDocument,
                PdfDocument,
                StageVersion,
                PdfCrawlStageVersion,
                PdfExtractStageVersion,
            ],
            skip_indexes=True,
        )

    event_loop.run_until_complete(_setup())

    class Store(dict[str, Document]):
        async def refresh(self) -> None:
            self.clear()
            pages = await WebPageDocument.find_all().to_list()
            pdfs = await PdfDocument.find_all().to_list()
            for doc in (*pages, *pdfs):
                self[doc.url] = doc

        def sync_refresh(self) -> None:
            event_loop.run_until_complete(self.refresh())

    yield Store()
