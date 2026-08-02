from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any, TypeVar

import pytest

from fao_impact_monitor.config import PdfExtractConfig
from fao_impact_monitor.data_lake.document import Document
from fao_impact_monitor.data_lake.documents.pdf_document import PdfDocument
from fao_impact_monitor.data_lake.stage import StageStatus
from fao_impact_monitor.data_lake.stages.pdf_crawl_stage import (
    PDF_CRAWL_STAGE_NAME,
    PdfCrawlStageResult,
)
from fao_impact_monitor.data_lake.stages.pdf_extract_stage import (
    DoclingWorker,
    PdfExtractStage,
    PdfExtractStageResult,
)

T = TypeVar("T")
RunAsync = Callable[[Coroutine[Any, Any, T]], T]

_EXPECTED_TITLE = (
    "Agricultural production risks associated with El Niño conditions and "
    "shipping disruptions linked to the Middle East conflict"
)
_EXPECTED_NUM_PAGES = 11
_TEST_PDF = Path(__file__).resolve().parents[1] / "data" / "fao_el_nino_2026.pdf"


@pytest.mark.integration
def test_pdf_extract_real_fao_el_nino_pdf(
    document_store: dict[str, Document],
    pdf_extract_dirs: PdfExtractConfig,
    run_async: RunAsync[Any],
) -> None:
    del document_store
    assert _TEST_PDF.is_file(), f"missing test PDF: {_TEST_PDF}"

    doc = PdfDocument(
        url="https://example.com/fao_el_nino_2026.pdf",
        pipeline_name="pdf_process",
    )
    run_async(doc.insert())
    assert doc.id is not None
    doc.stage_results[PDF_CRAWL_STAGE_NAME] = [
        PdfCrawlStageResult(
            version_id="crawl-v1",
            status=StageStatus.COMPLETED,
            content_path=str(_TEST_PDF),
        )
    ]

    stage = PdfExtractStage(config=pdf_extract_dirs)
    try:
        result = run_async(stage.run(doc, {}, []))
    finally:
        worker = DoclingWorker._instance
        if worker is not None:
            worker.shutdown()
            DoclingWorker._instance = None

    assert result.status == StageStatus.COMPLETED, result.error
    assert isinstance(result, PdfExtractStageResult)
    assert result.title == _EXPECTED_TITLE
    assert result.num_pages == _EXPECTED_NUM_PAGES
    assert len(result.page_paths) == _EXPECTED_NUM_PAGES
    assert all(Path(p).is_file() and p.endswith(".md") for p in result.page_paths)
    for page_path in result.page_paths:
        content = Path(page_path).read_text(encoding="utf-8")
        assert len(content) > 100, f"{page_path} too short ({len(content)} chars)"
    assert doc.title == _EXPECTED_TITLE
