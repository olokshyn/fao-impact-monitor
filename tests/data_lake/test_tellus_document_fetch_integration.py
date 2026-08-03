"""Integration tests for TellusDocumentFetchStage against the live Tellus API."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any, TypeVar

import pytest

from fao_impact_monitor.config import TellusConfig, get_config
from fao_impact_monitor.data_lake.common import Status
from fao_impact_monitor.data_lake.document import Document
from fao_impact_monitor.data_lake.documents.tellus_document import TellusDocument
from fao_impact_monitor.data_lake.pipeline import PIPELINE_TELLUS_PROCESS
from fao_impact_monitor.data_lake.stages.tellus_document_fetch_stage import (
    TellusDocumentFetchStage,
    TellusDocumentFetchStageResult,
)
from fao_impact_monitor.data_provider.tellus_provider import tellus_search_chunks

T = TypeVar("T")
RunAsync = Callable[[Coroutine[Any, Any, T]], T]


def _require_tellus_token() -> TellusConfig:
    cfg = get_config().tellus
    if not cfg.bearer_token.get_secret_value().strip():
        pytest.skip("TELLUS_BEARER_TOKEN is not set")
    return cfg


@pytest.mark.integration
def test_tellus_document_fetch_kenya_water_resources(
    document_store: dict[str, Document],
    tellus_dirs: TellusConfig,
    run_async: RunAsync[Any],
) -> None:
    del document_store
    live_cfg = _require_tellus_token()
    # Keep filesystem under the test tmp dir; reuse live API auth/base.
    cfg = tellus_dirs.model_copy(
        update={
            "bearer_token": live_cfg.bearer_token,
            "api_base": live_cfg.api_base,
            "min_year": live_cfg.min_year,
            "max_results": live_cfg.max_results,
        }
    )

    search = run_async(tellus_search_chunks("Kenya water resources", config=cfg))
    chunks = search.get("chunks") or []
    assert isinstance(chunks, list) and chunks, "Tellus search returned no chunks"

    first = next(
        (
            chunk
            for chunk in chunks
            if isinstance(chunk, dict) and chunk.get("document_id")
        ),
        None,
    )
    assert first is not None
    document_id = str(first["document_id"])

    doc = TellusDocument(
        url=f"tellus://{document_id}",
        external_id=document_id,
        pipeline_statuses={PIPELINE_TELLUS_PROCESS: Status.PENDING},
    )
    stage = TellusDocumentFetchStage(config=cfg)
    result = run_async(stage.run(doc, {}, []))

    assert isinstance(result, TellusDocumentFetchStageResult)
    assert result.status == Status.COMPLETED, result.error
    assert result.num_pages >= 1
    assert result.page_paths
    assert doc.title
    assert doc.metadata.get("document_id") == document_id
    assert "citation" in doc.metadata
    assert "Pages" not in str(doc.metadata.get("citation", ""))
    assert len(doc.page_paths) == result.num_pages
    for page_path in doc.page_paths:
        path = Path(page_path)
        assert path.is_file()
        assert path.read_text(encoding="utf-8").strip()
