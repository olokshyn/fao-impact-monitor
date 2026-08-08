"""Tests for FetchStage (mocked scrapling, mongomock, temp fsspec paths)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import fsspec
import pytest

from fao_impact_monitor.hydra.config import FetchConfig
from fao_impact_monitor.hydra.scrapling import ScraplingFetchResult
from fao_impact_monitor.hydra.stage.fetch_stage import (
    ContentType,
    Fetch,
    FetchRequest,
    FetchResponse,
    FetchStage,
    FetchStageResult,
)
from fao_impact_monitor.hydra.status import Status
from fao_impact_monitor.hydra.task.task import Task
from tests.hydra.conftest import CounterDocument


def _html_body(size: int = 600) -> bytes:
    return b"<!doctype html><html><body>" + (b"x" * size) + b"</body></html>"


def _pdf_body(size: int = 600) -> bytes:
    return b"%PDF-1.4 " + (b"y" * size)


def _meta(
    *,
    body: bytes,
    fetcher: str = "async",
    status_code: int = 200,
    request_url: str = "https://example.com/page",
    response_headers: dict[str, Any] | None = None,
) -> ScraplingFetchResult:
    return ScraplingFetchResult(
        fetcher=fetcher,  # type: ignore[arg-type]
        fetcher_params={"timeout": 30},
        request_url=request_url,
        request_headers={},
        request_params={},
        status_code=status_code,
        response_headers=response_headers or {},
        body=body,
    )


@pytest.fixture
def body_dir(tmp_path: Path) -> Path:
    return tmp_path / "fetched_data" / "fetch"


@pytest.fixture
def fetch_stage(body_dir: Path) -> FetchStage:
    return FetchStage(config=FetchConfig(body_save_dir=str(body_dir)))


def test_cache_hit_skips_scrapling(
    hydra_db: None,
    run_async: Any,
    fetch_stage: FetchStage,
    monkeypatch: pytest.MonkeyPatch,
    body_dir: Path,
) -> None:
    url = "https://example.com/cached"
    body_dir.mkdir(parents=True)
    body_path = str(body_dir / "cached.pdf")
    with fsspec.open(body_path, "wb") as handle:
        handle.write(_pdf_body())

    fetch = Fetch(
        url=url,
        successful=True,
        request=FetchRequest(
            fetcher="async",
            fetcher_params={},
            url=url,
        ),
        response=FetchResponse(
            status_code=200,
            headers={"content-type": "application/pdf"},
            body_header=_pdf_body()[:512],
        ),
        content_type=ContentType.PDF,
        body_path=body_path,
    )
    run_async(fetch.insert())

    mock_fetch = AsyncMock()
    monkeypatch.setattr(
        "fao_impact_monitor.hydra.stage.fetch_stage.reliable_fetch_with_meta",
        mock_fetch,
    )

    task = Task(url=url, stage_name="fetch")
    result, ctx = run_async(fetch_stage.process(task, {}, "wf", "node_fetch"))

    mock_fetch.assert_not_awaited()
    assert isinstance(result, FetchStageResult)
    assert ctx is None
    assert result.status == Status.COMPLETED
    assert result.content_type == ContentType.PDF
    assert result.body_path == body_path
    assert result.requested_url == url
    assert result.fetched_url == url
    assert result.status_code == 200


def test_successful_fetch_writes_file_and_mongo(
    hydra_db: None,
    run_async: Any,
    fetch_stage: FetchStage,
    monkeypatch: pytest.MonkeyPatch,
    body_dir: Path,
) -> None:
    url = "https://example.com/doc.pdf"
    body = _pdf_body()
    mock_fetch = AsyncMock(
        return_value=_meta(
            body=body,
            request_url=url,
            response_headers={"content-type": "application/pdf"},
        )
    )
    monkeypatch.setattr(
        "fao_impact_monitor.hydra.stage.fetch_stage.reliable_fetch_with_meta",
        mock_fetch,
    )

    doc = CounterDocument(url=url, source="test")
    run_async(doc.insert())
    task = Task(url=url, source="test", document_id=doc.id, stage_name="fetch")

    result, ctx = run_async(fetch_stage.process(task, {}, "wf", "node_fetch"))

    assert result.status == Status.COMPLETED
    assert ctx is None
    assert result.content_type == ContentType.PDF
    assert result.body_path is not None
    assert result.body_path.endswith(".pdf")
    assert result.requested_url == url
    assert result.fetched_url == url
    assert result.status_code == 200

    stored = run_async(Fetch.find_one(Fetch.url == url))
    assert stored is not None
    assert stored.successful is True
    assert stored.content_type == ContentType.PDF
    assert stored.body_path == result.body_path
    assert stored.id is not None
    assert result.body_path == str(body_dir / f"{stored.id}.pdf")

    with fsspec.open(result.body_path, "rb") as handle:
        assert handle.read() == body

    refreshed = run_async(CounterDocument.get(doc.id))
    assert refreshed is not None
    stage_results = refreshed.stage_results["wf"]["node_fetch"]
    assert len(stage_results) == 1
    assert stage_results[0].status == Status.COMPLETED


def test_body_path_extension_matches_content_type(
    hydra_db: None,
    run_async: Any,
    fetch_stage: FetchStage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url = "https://example.com/page"
    body = _html_body()
    monkeypatch.setattr(
        "fao_impact_monitor.hydra.stage.fetch_stage.reliable_fetch_with_meta",
        AsyncMock(
            return_value=_meta(
                body=body,
                request_url=url,
                response_headers={"content-type": "text/html"},
            )
        ),
    )

    result, ctx = run_async(
        fetch_stage.process(
            Task(url=url, stage_name="fetch"),
            {},
            "wf",
            "n",
        )
    )
    assert result.status == Status.COMPLETED
    assert ctx is None
    assert result.content_type == ContentType.HTML
    assert result.body_path is not None
    assert result.body_path.endswith(".html")


def test_failure_useless_body(
    hydra_db: None,
    run_async: Any,
    fetch_stage: FetchStage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url = "https://example.com/empty"
    monkeypatch.setattr(
        "fao_impact_monitor.hydra.stage.fetch_stage.reliable_fetch_with_meta",
        AsyncMock(
            return_value=_meta(
                body=b"tiny",
                request_url=url,
            )
        ),
    )

    result, ctx = run_async(
        fetch_stage.process(
            Task(url=url, stage_name="fetch"),
            {},
            "wf",
            "n",
        )
    )
    assert result.status == Status.FAILED
    assert ctx is None
    stored = run_async(Fetch.find_one(Fetch.url == url))
    assert stored is not None
    assert stored.successful is False
    assert stored.body_path is None
    assert stored.request is None
    assert stored.response is None
    assert stored.error == "Fetch unsuccessful"
    assert result.error == stored.error


def test_unique_url_upsert_failed_then_success(
    hydra_db: None,
    run_async: Any,
    fetch_stage: FetchStage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url = "https://example.com/retry-me"
    monkeypatch.setattr(
        "fao_impact_monitor.hydra.stage.fetch_stage.reliable_fetch_with_meta",
        AsyncMock(return_value=_meta(body=b"x", request_url=url)),
    )
    failed, failed_ctx = run_async(
        fetch_stage.process(
            Task(url=url, stage_name="fetch"),
            {},
            "wf",
            "n",
        )
    )
    assert failed.status == Status.FAILED
    assert failed_ctx is None
    first = run_async(Fetch.find_one(Fetch.url == url))
    assert first is not None
    first_id = first.id

    pdf = _pdf_body()
    monkeypatch.setattr(
        "fao_impact_monitor.hydra.stage.fetch_stage.reliable_fetch_with_meta",
        AsyncMock(
            return_value=_meta(
                body=pdf,
                request_url=url,
                response_headers={"content-type": "application/pdf"},
            )
        ),
    )
    ok, ok_ctx = run_async(
        fetch_stage.process(
            Task(url=url, stage_name="fetch"),
            {},
            "wf",
            "n",
        )
    )
    assert ok.status == Status.COMPLETED
    assert ok_ctx is None
    rows = run_async(Fetch.find(Fetch.url == url).to_list())
    assert len(rows) == 1
    assert rows[0].id == first_id
    assert rows[0].successful is True
    assert rows[0].content_type == ContentType.PDF
    assert rows[0].error is None


def test_missing_url_fails(
    hydra_db: None,
    run_async: Any,
    fetch_stage: FetchStage,
) -> None:
    result, ctx = run_async(
        fetch_stage.process(Task(stage_name="fetch"), {}, "wf", "n")
    )
    assert result.status == Status.FAILED
    assert ctx is None
    assert result.error == "Task.url is required"
