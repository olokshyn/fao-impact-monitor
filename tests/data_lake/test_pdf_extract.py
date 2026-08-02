from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any, TypeVar

from fao_impact_monitor.config import PdfExtractConfig
from fao_impact_monitor.data_lake.document import Document, DocumentType
from fao_impact_monitor.data_lake.documents.pdf_document import PdfDocument
from fao_impact_monitor.data_lake.documents.web_page_document import WebPageDocument
from fao_impact_monitor.data_lake.pipeline import (
    PIPELINE_PDF_PROCESS,
    PdfProcessPipeline,
)
from fao_impact_monitor.data_lake.stage import (
    StageStatus,
    get_stage,
    get_stage_result_class,
)
from fao_impact_monitor.data_lake.stages.pdf_crawl_stage import (
    PDF_CRAWL_STAGE_NAME,
    PdfCrawlStageResult,
)
from fao_impact_monitor.data_lake.stages.pdf_extract_stage import (
    PDF_EXTRACT_STAGE_NAME,
    PdfExtractStage,
    PdfExtractStageResult,
)

T = TypeVar("T")
RunAsync = Callable[[Coroutine[Any, Any, T]], T]


def test_pdf_extract_registration() -> None:
    stage = get_stage(PDF_EXTRACT_STAGE_NAME)
    assert isinstance(stage, PdfExtractStage)
    assert get_stage_result_class(PDF_EXTRACT_STAGE_NAME) is PdfExtractStageResult


def test_pdf_process_pipeline_starts_with_extract() -> None:
    steps_field = PdfProcessPipeline.model_fields["steps"]
    assert steps_field.default_factory is not None
    steps = steps_field.default_factory()
    assert steps[0].stage_name == PDF_EXTRACT_STAGE_NAME
    assert len(steps) >= 1
    name_default = PdfProcessPipeline.model_fields["name"].default
    assert name_default == PIPELINE_PDF_PROCESS


def test_extract_happy_path_writes_md_and_sets_title(
    document_store: dict[str, Document],
    pdf_extract_dirs: PdfExtractConfig,
    run_async: RunAsync[Any],
    tmp_path: Path,
) -> None:
    del document_store
    pdf_file = tmp_path / "doc.pdf"
    pdf_file.write_bytes(b"%PDF-1.4\nfake")

    doc = PdfDocument(url="https://example.com/report.pdf", pipeline_name="pdf_process")
    run_async(doc.insert())
    assert doc.id is not None
    doc.stage_results[PDF_CRAWL_STAGE_NAME] = [
        PdfCrawlStageResult(
            version_id="crawl-v1",
            status=StageStatus.COMPLETED,
            content_path=str(pdf_file),
        )
    ]

    async def submit(
        *,
        pdf_path: Path,
        out_dir: Path,
        version_id: str,
        fallback_title: str | None,
    ) -> PdfExtractStageResult:
        assert pdf_path == pdf_file
        assert out_dir == pdf_extract_dirs.save_dir / str(doc.id)
        out_dir.mkdir(parents=True, exist_ok=True)
        page1 = out_dir / "page_0001.md"
        page2 = out_dir / "page_0002.md"
        page1.write_text("# Report Title\n\nPage one", encoding="utf-8")
        page2.write_text("Page two table", encoding="utf-8")
        return PdfExtractStageResult(
            version_id=version_id,
            status=StageStatus.COMPLETED,
            title="Report Title",
            num_pages=2,
            page_paths=[str(page1), str(page2)],
        )

    stage = PdfExtractStage(config=pdf_extract_dirs, submit_fn=submit)
    result = run_async(stage.run(doc, {}, []))

    assert result.status == StageStatus.COMPLETED
    assert isinstance(result, PdfExtractStageResult)
    assert result.title == "Report Title"
    assert result.num_pages == 2
    assert len(result.page_paths) == 2
    assert all(p.endswith(".md") for p in result.page_paths)
    assert (
        Path(result.page_paths[0])
        .read_text(encoding="utf-8")
        .startswith("# Report Title")
    )
    assert doc.title == "Report Title"


def test_extract_rejects_non_pdf(
    document_store: dict[str, Document],
    pdf_extract_dirs: PdfExtractConfig,
    run_async: RunAsync[Any],
) -> None:
    del document_store
    page = WebPageDocument(url="https://example.com/", pipeline_name="pdf_crawl")
    run_async(page.insert())

    stage = PdfExtractStage(config=pdf_extract_dirs, submit_fn=_unused_submit)
    result = run_async(stage.run(page, {}, []))

    assert result.status == StageStatus.FAILED
    assert result.error is not None
    assert "PDF" in result.error
    assert page.type == DocumentType.WEB_PAGE


def test_extract_missing_crawl_path(
    document_store: dict[str, Document],
    pdf_extract_dirs: PdfExtractConfig,
    run_async: RunAsync[Any],
) -> None:
    del document_store
    doc = PdfDocument(url="https://example.com/a.pdf", pipeline_name="pdf_process")
    run_async(doc.insert())

    stage = PdfExtractStage(config=pdf_extract_dirs, submit_fn=_unused_submit)
    result = run_async(stage.run(doc, {}, []))

    assert result.status == StageStatus.FAILED
    assert result.error is not None
    assert "content_path" in result.error


def test_extract_missing_pdf_file(
    document_store: dict[str, Document],
    pdf_extract_dirs: PdfExtractConfig,
    run_async: RunAsync[Any],
    tmp_path: Path,
) -> None:
    del document_store
    missing = tmp_path / "gone.pdf"
    doc = PdfDocument(url="https://example.com/b.pdf", pipeline_name="pdf_process")
    run_async(doc.insert())
    doc.stage_results[PDF_CRAWL_STAGE_NAME] = [
        PdfCrawlStageResult(
            version_id="crawl-v1",
            status=StageStatus.COMPLETED,
            content_path=str(missing),
        )
    ]

    stage = PdfExtractStage(config=pdf_extract_dirs, submit_fn=_unused_submit)
    result = run_async(stage.run(doc, {}, []))

    assert result.status == StageStatus.FAILED
    assert result.error is not None
    assert "not found" in result.error


async def _unused_submit(
    *,
    pdf_path: Path,
    out_dir: Path,
    version_id: str,
    fallback_title: str | None,
) -> PdfExtractStageResult:
    del pdf_path, out_dir, version_id, fallback_title
    raise AssertionError("submit_fn should not be called")
