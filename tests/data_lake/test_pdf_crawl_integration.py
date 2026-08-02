from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any, TypeVar

import pytest

from fao_impact_monitor.config import PdfCrawlConfig, get_config
from fao_impact_monitor.data_lake.document import (
    Document,
    DocumentType,
    RelationSide,
    RelationType,
)
from fao_impact_monitor.data_lake.documents.web_page_document import WebPageDocument
from fao_impact_monitor.data_lake.scrapling import fetch
from fao_impact_monitor.data_lake.stage import StageStatus
from fao_impact_monitor.data_lake.stages.pdf_crawl_stage import (
    PDF_CRAWL_STAGE_NAME,
    PIPELINE_FOR_PDF_PARAM,
    PIPELINE_FOR_WEB_PARAM,
    PdfCrawlStage,
    PdfCrawlStageResult,
)
from tests.data_lake.mock_http_server import MockHttpServer

T = TypeVar("T")
RunAsync = Callable[[Coroutine[Any, Any, T]], T]

PIPELINE_FOR_WEB = "integration-web-pipeline"
PIPELINE_FOR_PDF = "integration-pdf-pipeline"
STAGE_PARAMS = {
    PIPELINE_FOR_WEB_PARAM: PIPELINE_FOR_WEB,
    PIPELINE_FOR_PDF_PARAM: PIPELINE_FOR_PDF,
}


def _build_topology(server: MockHttpServer) -> str:
    """Register the crawl graph and return the root URL.

    Topology (page3 reachable via two parents — must be processed once):
        root -> page1 -> pdf1
                      -> page3 -> pdf3
                               -> pdf4
             -> pdf2
             -> page2 -> page3
    """
    server.add_html(
        "/",
        f"""<!doctype html>
<html><head><title>Root publications</title></head>
<body>
  <h1>FAO Impact evidence hub</h1>
  <p>Browse documentary sources related to livelihood impacts.</p>
  <ul>
    <li><a href="{server.url("/page1")}">Publications archive with technical reports</a></li>
    <li><a href="{server.url("/pdf2")}">Download overview report PDF</a></li>
    <li><a href="{server.url("/page2")}">Related documents and knowledge repository</a></li>
  </ul>
</body></html>
""",
    )
    server.add_html(
        "/page1",
        f"""<!doctype html>
<html><head><title>Publications archive</title></head>
<body>
  <h1>Publications archive</h1>
  <p>Full-text technical publications.</p>
  <ul>
    <li><a href="{server.url("/pdf1")}">Download country assessment report PDF</a></li>
    <li><a href="{server.url("/page3")}">Annexes and statistical briefs library</a></li>
  </ul>
</body></html>
""",
    )
    server.add_html(
        "/page2",
        f"""<!doctype html>
<html><head><title>Related documents</title></head>
<body>
  <h1>Related documents</h1>
  <a href="{server.url("/page3")}">Annexes and statistical briefs library</a>
</body></html>
""",
    )
    server.add_html(
        "/page3",
        f"""<!doctype html>
<html><head><title>Annex library</title></head>
<body>
  <h1>Annexes and statistical briefs</h1>
  <ul>
    <li><a href="{server.url("/pdf3")}">Download annex methodology PDF</a></li>
    <li><a href="{server.url("/pdf4")}">Download statistical brief PDF</a></li>
  </ul>
</body></html>
""",
    )
    server.add_pdf("/pdf1")
    server.add_pdf("/pdf2")
    server.add_pdf("/pdf3")
    server.add_pdf("/pdf4")
    return server.url("/")


def _has_edge(parent: Document, child: Document) -> bool:
    parent_to_child = any(
        rel.type == RelationType.URL_LINK
        and rel.side == RelationSide.TO
        and rel.d_id == child.id
        for rel in parent.relations
    )
    child_from_parent = any(
        rel.type == RelationType.URL_LINK
        and rel.side == RelationSide.FROM
        and rel.d_id == parent.id
        for rel in child.relations
    )
    return parent_to_child and child_from_parent


@pytest.mark.integration
def test_pdf_crawl_topology_with_bedrock_and_mock_server(
    http_server: MockHttpServer,
    document_store: dict[str, Document],
    pdf_crawl_dirs: PdfCrawlConfig,
    run_async: RunAsync[Any],
) -> None:
    config = get_config()
    if not config.aws_bedrock.api_key.get_secret_value():
        pytest.skip("AWS_BEDROCK_API_KEY not configured")

    root_url = _build_topology(http_server)
    pdf_crawl_dirs.max_url_depth = 5
    pdf_crawl_dirs.max_pdfs = 10
    pdf_crawl_dirs.max_urls_per_page = 10

    seed = WebPageDocument(
        url=root_url, title="Root", pipeline_name="seed-caller-pipeline"
    )
    stage = PdfCrawlStage(
        fetch_fn=fetch,
        config=pdf_crawl_dirs,
    )
    result = run_async(stage.run(seed, STAGE_PARAMS, []))

    assert result.status == StageStatus.COMPLETED
    assert result.content_path is not None
    assert result.content_path.endswith(".html")
    assert Path(result.content_path).exists()

    sync_refresh = getattr(document_store, "sync_refresh", None)
    if sync_refresh is not None:
        sync_refresh()

    docs_by_suffix: dict[str, Document] = {}
    for url, doc in document_store.items():
        suffix = url[len(http_server.base_url) :]
        docs_by_suffix[suffix or "/"] = doc

    assert "/" in docs_by_suffix
    assert "/page1" in docs_by_suffix
    assert "/page2" in docs_by_suffix
    assert "/page3" in docs_by_suffix
    for page_path in ("/", "/page1", "/page2", "/page3"):
        assert docs_by_suffix[page_path].pipeline_name == PIPELINE_FOR_WEB
    for pdf_path in ("/pdf1", "/pdf2", "/pdf3", "/pdf4"):
        assert pdf_path in docs_by_suffix, (
            f"missing {pdf_path} in {list(docs_by_suffix)}"
        )
        assert docs_by_suffix[pdf_path].type == DocumentType.PDF
        assert docs_by_suffix[pdf_path].pipeline_name == PIPELINE_FOR_PDF
        latest = docs_by_suffix[pdf_path].stage_results["pdf_crawl"][-1]
        assert isinstance(latest, PdfCrawlStageResult)
        assert latest.status == StageStatus.COMPLETED
        assert latest.content_path is not None
        assert latest.content_path.endswith(".pdf")
        assert Path(latest.content_path).exists()
        assert Path(latest.content_path).read_bytes().startswith(b"%PDF")

    root = docs_by_suffix["/"]
    page1 = docs_by_suffix["/page1"]
    page2 = docs_by_suffix["/page2"]
    page3 = docs_by_suffix["/page3"]
    pdf1 = docs_by_suffix["/pdf1"]
    pdf2 = docs_by_suffix["/pdf2"]
    pdf3 = docs_by_suffix["/pdf3"]
    pdf4 = docs_by_suffix["/pdf4"]

    # page3 is discovered from both page1 and page2 but processed only once.
    page3_results = page3.stage_results.get(PDF_CRAWL_STAGE_NAME, [])
    assert len(page3_results) == 1
    assert page3_results[0].status == StageStatus.COMPLETED

    assert _has_edge(root, page1)
    assert _has_edge(root, pdf2)
    assert _has_edge(root, page2)
    assert _has_edge(page1, pdf1)
    assert _has_edge(page1, page3)
    assert _has_edge(page2, page3)
    assert _has_edge(page3, pdf3)
    assert _has_edge(page3, pdf4)
