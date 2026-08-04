"""Integration tests for TellusDocumentFetchStage against the live Tellus API."""

from __future__ import annotations

import json
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

_EL_NINO_DOCUMENT_ID = "484a65df-3e46-46e1-b2f6-653499c4513b"
_EL_NINO_TITLE = (
    "2015-2016 El Nino early action and response for agriculture, "
    "food security and nutrition"
)
_EL_NINO_URL = "https://openknowledge.fao.org/handle/20.500.14283/i6049e"
_EL_NINO_METADATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "tellus_el_nino_2016_document_metadata.json"
)


def _require_tellus_token() -> TellusConfig:
    cfg = get_config().tellus
    if not cfg.bearer_token.get_secret_value().strip():
        pytest.skip("TELLUS_BEARER_TOKEN is not set")
    return cfg


def _live_tellus_config(tellus_dirs: TellusConfig) -> TellusConfig:
    live_cfg = _require_tellus_token()
    # Keep filesystem under the test tmp dir; reuse live API auth/base.
    return tellus_dirs.model_copy(
        update={
            "bearer_token": live_cfg.bearer_token,
            "api_base": live_cfg.api_base,
            "min_year": live_cfg.min_year,
            "max_results": live_cfg.max_results,
        }
    )


@pytest.mark.integration
def test_tellus_document_fetch_kenya_water_resources(
    document_store: dict[str, Document],
    tellus_dirs: TellusConfig,
    run_async: RunAsync[Any],
) -> None:
    del document_store
    cfg = _live_tellus_config(tellus_dirs)

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
    assert doc.url.startswith("http")
    assert len(doc.page_paths) == result.num_pages
    for page_path in doc.page_paths:
        path = Path(page_path)
        assert path.is_file()
        assert path.read_text(encoding="utf-8").strip()


@pytest.mark.integration
def test_tellus_document_fetch_el_nino_2016_metadata(
    document_store: dict[str, Document],
    tellus_dirs: TellusConfig,
    run_async: RunAsync[Any],
) -> None:
    del document_store
    cfg = _live_tellus_config(tellus_dirs)
    expected_metadata = json.loads(_EL_NINO_METADATA_PATH.read_text(encoding="utf-8"))

    doc = TellusDocument(
        url=f"tellus://{_EL_NINO_DOCUMENT_ID}",
        external_id=_EL_NINO_DOCUMENT_ID,
        pipeline_statuses={PIPELINE_TELLUS_PROCESS: Status.PENDING},
    )
    stage = TellusDocumentFetchStage(config=cfg)
    result = run_async(stage.run(doc, {}, []))

    assert isinstance(result, TellusDocumentFetchStageResult)
    assert result.status == Status.COMPLETED, result.error
    assert result.num_pages >= 1

    stored = run_async(_find_by_external_id(_EL_NINO_DOCUMENT_ID))
    assert stored is not None
    assert stored.external_id == _EL_NINO_DOCUMENT_ID
    assert stored.title == _EL_NINO_TITLE
    assert stored.url == _EL_NINO_URL
    assert stored.metadata == expected_metadata


async def _find_by_external_id(external_id: str) -> TellusDocument | None:
    return await TellusDocument.find_one(TellusDocument.external_id == external_id)
