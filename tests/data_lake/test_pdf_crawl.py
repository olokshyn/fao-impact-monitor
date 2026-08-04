import asyncio
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any, TypeVar

from fao_impact_monitor.agent.pdf_crawl_agent import (
    PdfLinkCandidateList,
    PdfPageExtract,
    extract_page_urls,
)
from fao_impact_monitor.config import PdfCrawlConfig
from fao_impact_monitor.data_lake.common import Status
from fao_impact_monitor.data_lake.document import (
    Document,
    DocumentType,
    RelationSide,
    RelationType,
)
from fao_impact_monitor.data_lake.documents.pdf_document import PdfDocument
from fao_impact_monitor.data_lake.documents.web_page_document import WebPageDocument
from fao_impact_monitor.data_lake.stage import get_stage
from fao_impact_monitor.data_lake.stages.pdf_crawl_stage import (
    PDF_CRAWL_STAGE_NAME,
    PIPELINE_FOR_PDF_PARAM,
    PIPELINE_FOR_WEB_PARAM,
    PdfCrawlStage,
    PdfCrawlStageResult,
    _child_title_state,
    classify_content,
)

T = TypeVar("T")
RunAsync = Callable[[Coroutine[Any, Any, T]], T]

PIPELINE_FOR_WEB = "web-pipeline"
PIPELINE_FOR_PDF = "pdf-pipeline"
STAGE_PARAMS = {
    PIPELINE_FOR_WEB_PARAM: PIPELINE_FOR_WEB,
    PIPELINE_FOR_PDF_PARAM: PIPELINE_FOR_PDF,
}


def _seed(
    url: str = "https://example.com/",
    *,
    title: str | None = "Seed",
) -> WebPageDocument:
    return WebPageDocument(
        url=url,
        title=title,
        pipeline_statuses={"seed-caller-pipeline": Status.PENDING},
    )


def _refresh(store: dict[str, Document]) -> None:
    sync_refresh = getattr(store, "sync_refresh", None)
    if sync_refresh is not None:
        sync_refresh()


def test_stage_registration() -> None:
    stage = get_stage(PDF_CRAWL_STAGE_NAME)
    assert isinstance(stage, PdfCrawlStage)


def test_classify_content() -> None:
    assert classify_content(b"%PDF-1.4\n") == "pdf"
    assert classify_content(b"<!doctype html><html></html>") == "html"
    assert classify_content(b"<html><body></body></html>") == "html"
    assert classify_content(b"\n\n<!DOCTYPE HTML>") == "html"
    assert classify_content(b"plain text") == "other"


def test_seed_already_completed_returns_early(
    document_store: dict[str, Document],
    pdf_crawl_dirs: PdfCrawlConfig,
    run_async: RunAsync[Any],
) -> None:
    seed = _seed()
    prior = PdfCrawlStageResult(
        version_id="v1",
        status=Status.COMPLETED,
        content_path="/tmp/prior",
    )
    seed.stage_results = {PDF_CRAWL_STAGE_NAME: [prior]}
    run_async(seed.insert())

    async def fetch(*, url: str) -> bytes:
        raise AssertionError("should not fetch")

    stage = PdfCrawlStage(fetch_fn=fetch, config=pdf_crawl_dirs)
    result = run_async(stage.run(seed, STAGE_PARAMS, []))

    assert result.status == Status.COMPLETED
    assert isinstance(result, PdfCrawlStageResult)
    assert result.content_path == "/tmp/prior"


def test_unsaved_seed_pdf_saves_and_completes(
    document_store: dict[str, Document],
    pdf_crawl_dirs: PdfCrawlConfig,
    run_async: RunAsync[Any],
) -> None:
    seed = _seed("https://example.com/report")
    body = b"%PDF-1.4\nminimal"

    async def fetch(*, url: str) -> bytes:
        assert url == seed.url
        return body

    stage = PdfCrawlStage(fetch_fn=fetch, config=pdf_crawl_dirs)
    result = run_async(stage.run(seed, STAGE_PARAMS, []))

    assert result.status == Status.COMPLETED
    assert isinstance(result, PdfCrawlStageResult)
    assert result.content_path is not None
    assert result.content_path.endswith(".pdf")
    assert Path(result.content_path).read_bytes() == body
    _refresh(document_store)
    saved = document_store[seed.url]
    assert isinstance(saved, PdfDocument)
    assert saved.type == DocumentType.PDF
    assert saved.pipeline_status(PIPELINE_FOR_PDF) == Status.PENDING


def test_incomplete_existing_seed_is_reprocessed(
    document_store: dict[str, Document],
    pdf_crawl_dirs: PdfCrawlConfig,
    run_async: RunAsync[Any],
) -> None:
    """PENDING docs in Mongo are not treated as done; crawl runs again."""
    seed = _seed()
    seed.set_pipeline_status(PIPELINE_FOR_WEB, Status.PENDING)
    run_async(seed.insert())
    fetch_calls: list[str] = []

    async def fetch(*, url: str) -> bytes:
        fetch_calls.append(url)
        return b"<!doctype html><html><body>ok</body></html>"

    async def extract(
        *,
        page_url: str,
        page_body: str,
        max_urls: int,
        max_retries: int,
        model: Any = None,
    ) -> PdfPageExtract:
        del page_url, page_body, max_urls, max_retries, model
        return PdfPageExtract()

    stage = PdfCrawlStage(
        fetch_fn=fetch,
        extract_fn=extract,
        config=pdf_crawl_dirs,
    )
    result = run_async(stage.run(seed, STAGE_PARAMS, []))
    assert result.status == Status.COMPLETED
    assert fetch_calls == [seed.url]
    _refresh(document_store)
    saved = document_store[seed.url]
    assert _latest_or_none(saved) is not None


def _latest_or_none(document: Document) -> PdfCrawlStageResult | None:
    results = document.stage_results.get(PDF_CRAWL_STAGE_NAME)
    if not results:
        return None
    latest = results[-1]
    if isinstance(latest, PdfCrawlStageResult):
        return latest
    return PdfCrawlStageResult.model_validate(latest.model_dump())


def test_saved_seed_with_pdf_body_fails(
    document_store: dict[str, Document],
    pdf_crawl_dirs: PdfCrawlConfig,
    run_async: RunAsync[Any],
) -> None:
    seed = _seed()
    run_async(seed.insert())

    async def fetch(*, url: str) -> bytes:
        return b"%PDF-1.4\n"

    stage = PdfCrawlStage(fetch_fn=fetch, config=pdf_crawl_dirs)
    result = run_async(stage.run(seed, STAGE_PARAMS, []))

    assert result.status == Status.FAILED
    assert result.error is not None
    assert "already saved" in result.error


def test_non_html_non_pdf_seed_fails(
    document_store: dict[str, Document],
    pdf_crawl_dirs: PdfCrawlConfig,
    run_async: RunAsync[Any],
) -> None:
    seed = _seed()

    async def fetch(*, url: str) -> bytes:
        return b"not a document"

    stage = PdfCrawlStage(fetch_fn=fetch, config=pdf_crawl_dirs)
    result = run_async(stage.run(seed, STAGE_PARAMS, []))
    assert result.status == Status.FAILED
    _refresh(document_store)
    assert seed.url not in document_store


def test_html_seed_creates_children_and_bidirectional_relations(
    document_store: dict[str, Document],
    pdf_crawl_dirs: PdfCrawlConfig,
    run_async: RunAsync[Any],
) -> None:
    seed = _seed("https://example.com/")
    child_page = "https://example.com/page1"
    child_pdf = "https://example.com/file.pdf"

    pages = {
        seed.url: (
            b"<!doctype html><html><body>"
            b'<a href="/page1">Publications archive</a>'
            b'<a href="/file.pdf">Download report PDF</a>'
            b"</body></html>"
        ),
        child_page: (
            b"<!doctype html><html><body><p>No further links</p></body></html>"
        ),
        child_pdf: b"%PDF-1.4\nchild",
    }

    async def fetch(*, url: str) -> bytes:
        return pages[url]

    extract_calls: list[str] = []

    async def extract(
        *,
        page_url: str,
        page_body: str,
        max_urls: int,
        max_retries: int,
        model: Any = None,
    ) -> PdfPageExtract:
        del max_urls, max_retries, model
        extract_calls.append(page_url)
        if page_url == seed.url:
            return PdfPageExtract(urls=["/page1", "/file.pdf"])
        return PdfPageExtract()

    stage = PdfCrawlStage(
        fetch_fn=fetch,
        extract_fn=extract,
        config=pdf_crawl_dirs,
    )
    result = run_async(stage.run(seed, STAGE_PARAMS, []))

    assert result.status == Status.COMPLETED
    assert isinstance(result, PdfCrawlStageResult)
    assert result.content_path is not None
    assert result.content_path.endswith(".html")
    assert Path(result.content_path).exists()

    _refresh(document_store)
    root = document_store[seed.url]
    page = document_store[child_page]
    pdf = document_store[child_pdf]
    assert isinstance(root, WebPageDocument)
    assert isinstance(page, WebPageDocument)
    assert isinstance(pdf, PdfDocument)
    assert root.pipeline_status(PIPELINE_FOR_WEB) == Status.PENDING
    assert page.pipeline_status(PIPELINE_FOR_WEB) == Status.PENDING
    assert pdf.pipeline_status(PIPELINE_FOR_PDF) == Status.PENDING

    page_crawl = page.stage_results["pdf_crawl"][-1]
    pdf_crawl = pdf.stage_results["pdf_crawl"][-1]
    assert isinstance(page_crawl, PdfCrawlStageResult)
    assert isinstance(pdf_crawl, PdfCrawlStageResult)
    assert page_crawl.content_path is not None
    assert pdf_crawl.content_path is not None
    assert page_crawl.content_path.endswith(".html")
    assert pdf_crawl.content_path.endswith(".pdf")

    assert extract_calls == [seed.url, child_page]

    assert any(
        r.side == RelationSide.TO
        and r.d_id == page.id
        and r.type == RelationType.URL_LINK
        for r in root.relations
    )
    assert any(
        r.side == RelationSide.TO
        and r.d_id == pdf.id
        and r.type == RelationType.URL_LINK
        for r in root.relations
    )
    assert any(
        r.side == RelationSide.FROM and r.d_id == root.id for r in page.relations
    )
    assert any(r.side == RelationSide.FROM and r.d_id == root.id for r in pdf.relations)


def test_shared_child_processed_once_with_both_parent_relations(
    document_store: dict[str, Document],
    pdf_crawl_dirs: PdfCrawlConfig,
    run_async: RunAsync[Any],
) -> None:
    """Diamond graph: page1 and page2 both link to page3; fetch/agent once."""
    seed = _seed("https://example.com/")
    page1 = "https://example.com/page1"
    page2 = "https://example.com/page2"
    page3 = "https://example.com/page3"

    pages = {
        seed.url: (
            b"<!doctype html><html><body>"
            b'<a href="/page1">Publications archive</a>'
            b'<a href="/page2">Related documents</a>'
            b"</body></html>"
        ),
        page1: (
            b"<!doctype html><html><body>"
            b'<a href="/page3">Annexes library</a>'
            b"</body></html>"
        ),
        page2: (
            b"<!doctype html><html><body>"
            b'<a href="/page3">Annexes library</a>'
            b"</body></html>"
        ),
        page3: (b"<!doctype html><html><body><p>Leaf page</p></body></html>"),
    }
    fetch_counts: dict[str, int] = {}

    async def fetch(*, url: str) -> bytes:
        fetch_counts[url] = fetch_counts.get(url, 0) + 1
        return pages[url]

    extract_calls: list[str] = []

    async def extract(
        *,
        page_url: str,
        page_body: str,
        max_urls: int,
        max_retries: int,
        model: Any = None,
    ) -> PdfPageExtract:
        del page_body, max_urls, max_retries, model
        extract_calls.append(page_url)
        if page_url == seed.url:
            return PdfPageExtract(urls=["/page1", "/page2"])
        if page_url in {page1, page2}:
            return PdfPageExtract(urls=["/page3"])
        return PdfPageExtract()

    stage = PdfCrawlStage(
        fetch_fn=fetch,
        extract_fn=extract,
        config=pdf_crawl_dirs,
    )
    result = run_async(stage.run(seed, STAGE_PARAMS, []))
    assert result.status == Status.COMPLETED

    _refresh(document_store)
    assert fetch_counts[page3] == 1
    assert extract_calls.count(page3) == 1

    p1 = document_store[page1]
    p2 = document_store[page2]
    p3 = document_store[page3]
    assert len(p3.stage_results[PDF_CRAWL_STAGE_NAME]) == 1
    assert any(r.d_id == p3.id and r.side == RelationSide.TO for r in p1.relations)
    assert any(r.d_id == p3.id and r.side == RelationSide.TO for r in p2.relations)
    assert any(r.d_id == p1.id and r.side == RelationSide.FROM for r in p3.relations)
    assert any(r.d_id == p2.id and r.side == RelationSide.FROM for r in p3.relations)


def test_completed_child_gets_relations_only(
    document_store: dict[str, Document],
    pdf_crawl_dirs: PdfCrawlConfig,
    run_async: RunAsync[Any],
) -> None:
    seed = _seed("https://example.com/")
    child_url = "https://example.com/done.pdf"
    child = PdfDocument(
        url=child_url,
        pipeline_statuses={"test-pipeline": Status.PENDING},
    )
    child.stage_results = {
        PDF_CRAWL_STAGE_NAME: [
            PdfCrawlStageResult(
                version_id="v1",
                status=Status.COMPLETED,
                content_path="/tmp/done.pdf",
            )
        ]
    }
    run_async(child.insert())

    async def fetch(*, url: str) -> bytes:
        if url == seed.url:
            return (
                b"<!doctype html><html><body>"
                b'<a href="/done.pdf">Existing PDF</a>'
                b"</body></html>"
            )
        raise AssertionError(f"should not fetch {url}")

    async def extract(
        *,
        page_url: str,
        page_body: str,
        max_urls: int,
        max_retries: int,
        model: Any = None,
    ) -> PdfPageExtract:
        del page_body, max_urls, max_retries, model
        assert page_url == seed.url
        return PdfPageExtract(urls=["/done.pdf"])

    stage = PdfCrawlStage(
        fetch_fn=fetch,
        extract_fn=extract,
        config=pdf_crawl_dirs,
    )
    result = run_async(stage.run(seed, STAGE_PARAMS, []))
    assert result.status == Status.COMPLETED

    _refresh(document_store)
    root = document_store[seed.url]
    linked = document_store[child_url]
    assert any(
        r.d_id == linked.id and r.side == RelationSide.TO for r in root.relations
    )
    assert any(
        r.d_id == root.id and r.side == RelationSide.FROM for r in linked.relations
    )


def test_max_pdfs_stops_dfs(
    document_store: dict[str, Document],
    pdf_crawl_dirs: PdfCrawlConfig,
    run_async: RunAsync[Any],
) -> None:
    pdf_crawl_dirs.max_pdfs = 1
    seed = _seed("https://example.com/")

    async def fetch(*, url: str) -> bytes:
        if url == seed.url:
            return (
                b"<!doctype html><html><body>"
                b'<a href="/a.pdf">PDF A</a>'
                b'<a href="/b.pdf">PDF B</a>'
                b"</body></html>"
            )
        return b"%PDF-1.4\n" + url.encode()

    async def extract(
        *,
        page_url: str,
        page_body: str,
        max_urls: int,
        max_retries: int,
        model: Any = None,
    ) -> PdfPageExtract:
        del page_body, max_urls, max_retries, model
        if page_url == seed.url:
            return PdfPageExtract(urls=["/a.pdf", "/b.pdf"])
        return PdfPageExtract()

    stage = PdfCrawlStage(
        fetch_fn=fetch,
        extract_fn=extract,
        config=pdf_crawl_dirs,
    )
    run_async(stage.run(seed, STAGE_PARAMS, []))
    _refresh(document_store)
    pdfs = [doc for doc in document_store.values() if doc.type == DocumentType.PDF]
    assert len(pdfs) == 1


def test_max_urls_stops_dfs(
    document_store: dict[str, Document],
    pdf_crawl_dirs: PdfCrawlConfig,
    run_async: RunAsync[Any],
) -> None:
    """max_urls counts the seed; with max_urls=2 only one child is fetched."""
    pdf_crawl_dirs.max_urls = 2
    seed = _seed("https://example.com/")
    fetch_calls: list[str] = []

    async def fetch(*, url: str) -> bytes:
        fetch_calls.append(url)
        if url == seed.url:
            return (
                b"<!doctype html><html><body>"
                b'<a href="/page1">Page 1</a>'
                b'<a href="/page2">Page 2</a>'
                b"</body></html>"
            )
        return b"<!doctype html><html><body><p>leaf</p></body></html>"

    async def extract(
        *,
        page_url: str,
        page_body: str,
        max_urls: int,
        max_retries: int,
        model: Any = None,
    ) -> PdfPageExtract:
        del page_body, max_urls, max_retries, model
        if page_url == seed.url:
            return PdfPageExtract(urls=["/page1", "/page2"])
        return PdfPageExtract()

    stage = PdfCrawlStage(
        fetch_fn=fetch,
        extract_fn=extract,
        config=pdf_crawl_dirs,
    )
    run_async(stage.run(seed, STAGE_PARAMS, []))
    _refresh(document_store)
    assert fetch_calls[0] == seed.url
    assert len(fetch_calls) == 2
    assert seed.url in document_store
    children = [u for u in document_store if u != seed.url]
    assert len(children) == 1


def test_fetch_error_on_seed_fails(
    document_store: dict[str, Document],
    pdf_crawl_dirs: PdfCrawlConfig,
    run_async: RunAsync[Any],
) -> None:
    seed = _seed()

    async def fetch(*, url: str) -> bytes:
        raise RuntimeError("network down")

    stage = PdfCrawlStage(fetch_fn=fetch, config=pdf_crawl_dirs)
    result = run_async(stage.run(seed, STAGE_PARAMS, []))
    assert result.status == Status.FAILED
    assert result.error is not None
    assert "network down" in result.error


def test_missing_pipeline_stage_params_fails(
    document_store: dict[str, Document],
    pdf_crawl_dirs: PdfCrawlConfig,
    run_async: RunAsync[Any],
) -> None:
    seed = _seed()

    async def fetch(*, url: str) -> bytes:
        raise AssertionError("should not fetch")

    stage = PdfCrawlStage(fetch_fn=fetch, config=pdf_crawl_dirs)
    result = run_async(stage.run(seed, {}, []))
    assert result.status == Status.FAILED
    assert result.error is not None
    assert PIPELINE_FOR_WEB_PARAM in result.error


def test_link_agent_validation_reprompts_for_missing_urls() -> None:
    class FakeStructured:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke(self, _messages: Any) -> PdfLinkCandidateList:
            self.calls += 1
            if self.calls == 1:
                return PdfLinkCandidateList.model_validate(
                    {
                        "links": [
                            {
                                "url": "https://evil.example/not-on-page",
                                "reason": "hallucination",
                            },
                            {
                                "url": "/real.pdf",
                                "reason": "download",
                            },
                        ]
                    }
                )
            return PdfLinkCandidateList.model_validate(
                {"links": [{"url": "/real.pdf", "reason": "download"}]}
            )

    class FakeModel:
        def __init__(self) -> None:
            self.structured = FakeStructured()

        def with_structured_output(self, _schema: Any) -> FakeStructured:
            return self.structured

    body = '<!doctype html><a href="/real.pdf">Download report PDF</a>'
    model = FakeModel()
    result = asyncio.run(
        extract_page_urls(
            page_url="https://example.com/",
            page_body=body,
            max_urls=5,
            max_retries=3,
            model=model,  # type: ignore[arg-type]
        )
    )
    assert result.urls == ["/real.pdf"]
    assert result.document_title is None
    assert model.structured.calls == 2


def test_link_agent_validation_reprompts_for_missing_title() -> None:
    title = "Zimbabwe: Project Highlights - OSRO/ZIM/040/GER"

    class FakeStructured:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke(self, _messages: Any) -> PdfLinkCandidateList:
            self.calls += 1
            if self.calls == 1:
                return PdfLinkCandidateList.model_validate(
                    {
                        "document_title": "Hallucinated Title",
                        "links": [{"url": "/file.pdf", "reason": "download"}],
                    }
                )
            return PdfLinkCandidateList.model_validate(
                {
                    "document_title": title,
                    "links": [{"url": "/file.pdf", "reason": "download"}],
                }
            )

    class FakeModel:
        def __init__(self) -> None:
            self.structured = FakeStructured()

        def with_structured_output(self, _schema: Any) -> FakeStructured:
            return self.structured

    body = f'<!doctype html><h1>{title}</h1><a href="/file.pdf">Download PDF</a>'
    model = FakeModel()
    result = asyncio.run(
        extract_page_urls(
            page_url="https://example.com/handle/doc",
            page_body=body,
            max_urls=5,
            max_retries=3,
            model=model,  # type: ignore[arg-type]
        )
    )
    assert result.urls == ["/file.pdf"]
    assert result.document_title == title
    assert model.structured.calls == 2


def test_child_title_state_override_and_expiry() -> None:
    assert _child_title_state(
        detected_title="New",
        inherited_title="Old",
        inherited_ttl=2,
        validity_depth=3,
    ) == ("New", 3)
    assert _child_title_state(
        detected_title=None,
        inherited_title="Old",
        inherited_ttl=2,
        validity_depth=3,
    ) == ("Old", 1)
    assert _child_title_state(
        detected_title=None,
        inherited_title="Old",
        inherited_ttl=1,
        validity_depth=3,
    ) == (None, 0)
    assert _child_title_state(
        detected_title="Only",
        inherited_title=None,
        inherited_ttl=0,
        validity_depth=0,
    ) == (None, 0)


def test_details_page_title_applied_to_child_pdf(
    document_store: dict[str, Document],
    pdf_crawl_dirs: PdfCrawlConfig,
    run_async: RunAsync[Any],
) -> None:
    seed = _seed("https://example.com/handle/doc")
    pdf_url = "https://example.com/file.pdf"
    title = "Zimbabwe: Project Highlights - OSRO/ZIM/040/GER"

    async def fetch(*, url: str) -> bytes:
        if url == seed.url:
            return (
                b"<!doctype html><html><body>"
                b"<h1>Zimbabwe: Project Highlights - OSRO/ZIM/040/GER</h1>"
                b'<a href="/file.pdf">Download PDF</a>'
                b"</body></html>"
            )
        return b"%PDF-1.4\nchild"

    async def extract(
        *,
        page_url: str,
        page_body: str,
        max_urls: int,
        max_retries: int,
        model: Any = None,
    ) -> PdfPageExtract:
        del page_body, max_urls, max_retries, model
        if page_url == seed.url:
            return PdfPageExtract(urls=["/file.pdf"], document_title=title)
        return PdfPageExtract()

    stage = PdfCrawlStage(
        fetch_fn=fetch,
        extract_fn=extract,
        config=pdf_crawl_dirs,
    )
    run_async(stage.run(seed, STAGE_PARAMS, []))
    _refresh(document_store)
    pdf = document_store[pdf_url]
    assert isinstance(pdf, PdfDocument)
    assert pdf.title == title


def test_child_details_title_overrides_parent_title(
    document_store: dict[str, Document],
    pdf_crawl_dirs: PdfCrawlConfig,
    run_async: RunAsync[Any],
) -> None:
    seed = _seed("https://example.com/")
    details = "https://example.com/details"
    pdf_url = "https://example.com/file.pdf"

    async def fetch(*, url: str) -> bytes:
        if url == seed.url:
            return (
                b"<!doctype html><html><body>"
                b'<a href="/details">Document details</a>'
                b"</body></html>"
            )
        if url == details:
            return (
                b"<!doctype html><html><body>"
                b"<h1>Child Title</h1>"
                b'<a href="/file.pdf">Download PDF</a>'
                b"</body></html>"
            )
        return b"%PDF-1.4\nchild"

    async def extract(
        *,
        page_url: str,
        page_body: str,
        max_urls: int,
        max_retries: int,
        model: Any = None,
    ) -> PdfPageExtract:
        del page_body, max_urls, max_retries, model
        if page_url == seed.url:
            return PdfPageExtract(
                urls=["/details"],
                document_title="Parent Title",
            )
        if page_url == details:
            return PdfPageExtract(
                urls=["/file.pdf"],
                document_title="Child Title",
            )
        return PdfPageExtract()

    stage = PdfCrawlStage(
        fetch_fn=fetch,
        extract_fn=extract,
        config=pdf_crawl_dirs,
    )
    run_async(stage.run(seed, STAGE_PARAMS, []))
    _refresh(document_store)
    assert document_store[pdf_url].title == "Child Title"


def test_title_expires_after_validity_depth(
    document_store: dict[str, Document],
    pdf_crawl_dirs: PdfCrawlConfig,
    run_async: RunAsync[Any],
) -> None:
    pdf_crawl_dirs.detected_title_validity_depth = 1
    seed = _seed("https://example.com/details")
    mid = "https://example.com/mid"
    pdf_url = "https://example.com/file.pdf"

    async def fetch(*, url: str) -> bytes:
        if url == seed.url:
            return (
                b"<!doctype html><html><body>"
                b"<h1>Only One Hop</h1>"
                b'<a href="/mid">Next</a>'
                b"</body></html>"
            )
        if url == mid:
            return (
                b"<!doctype html><html><body>"
                b'<a href="/file.pdf">Download PDF</a>'
                b"</body></html>"
            )
        return b"%PDF-1.4\nchild"

    async def extract(
        *,
        page_url: str,
        page_body: str,
        max_urls: int,
        max_retries: int,
        model: Any = None,
    ) -> PdfPageExtract:
        del page_body, max_urls, max_retries, model
        if page_url == seed.url:
            return PdfPageExtract(
                urls=["/mid"],
                document_title="Only One Hop",
            )
        if page_url == mid:
            return PdfPageExtract(urls=["/file.pdf"])
        return PdfPageExtract()

    stage = PdfCrawlStage(
        fetch_fn=fetch,
        extract_fn=extract,
        config=pdf_crawl_dirs,
    )
    run_async(stage.run(seed, STAGE_PARAMS, []))
    _refresh(document_store)
    # depth=1: title valid for the mid page hop, expired before the PDF child.
    assert document_store[pdf_url].title is None
