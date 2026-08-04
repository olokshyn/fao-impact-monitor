"""Tellus document → per-chunk markdown fetch stage."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import Field

from fao_impact_monitor.config import TellusConfig, get_config
from fao_impact_monitor.data_lake.common import Status
from fao_impact_monitor.data_lake.document import Document, DocumentType
from fao_impact_monitor.data_lake.documents.tellus_document import TellusDocument
from fao_impact_monitor.data_lake.stage import (
    Stage,
    StageResult,
    StageVersion,
)
from fao_impact_monitor.data_provider.tellus_provider import (
    tellus_get_all_document_chunks,
)

logger = logging.getLogger(__name__)

TELLUS_DOCUMENT_FETCH_STAGE_NAME = "tellus_document_fetch"
_TELLUS_FETCH_PIPELINE_ID = "tellus-chunks-v1"
_METADATA_KEYS_TO_DROP = frozenset({"key_concepts", "organizations"})

FetchChunksFn = Callable[..., Awaitable[dict[str, Any]]]


def _sanitize_tellus_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Drop bulky fields and nested values that duplicate the parent metadata."""
    cleaned = {k: v for k, v in metadata.items() if k not in _METADATA_KEYS_TO_DROP}
    nested = cleaned.get("metadata")
    if isinstance(nested, dict):
        parent = {k: v for k, v in cleaned.items() if k != "metadata"}
        deduped = {
            key: value
            for key, value in nested.items()
            if key not in _METADATA_KEYS_TO_DROP
            and not (key in parent and parent[key] == value)
        }
        if deduped:
            cleaned["metadata"] = deduped
        else:
            cleaned.pop("metadata", None)
    return cleaned


class TellusDocumentFetchStageResult(StageResult):
    name: str = TELLUS_DOCUMENT_FETCH_STAGE_NAME
    title: str | None = None
    num_pages: int = 0
    page_paths: list[str] = Field(default_factory=list)


class TellusDocumentFetchStageVersion(StageVersion):
    pipeline_id: str

    class Settings:
        class_id_value = TELLUS_DOCUMENT_FETCH_STAGE_NAME


class TellusDocumentFetchStage(Stage):
    name = TELLUS_DOCUMENT_FETCH_STAGE_NAME

    def __init__(
        self,
        *,
        config: TellusConfig | None = None,
        fetch_fn: FetchChunksFn | None = None,
    ) -> None:
        self._config = config
        self._fetch_fn = fetch_fn

    async def get_version(self) -> StageVersion:
        version_id = hashlib.sha256(_TELLUS_FETCH_PIPELINE_ID.encode()).hexdigest()[:32]
        existing = await TellusDocumentFetchStageVersion.find_one(
            TellusDocumentFetchStageVersion.version_id == version_id
        )
        if existing is not None:
            return existing
        version = TellusDocumentFetchStageVersion(
            version_id=version_id,
            pipeline_id=_TELLUS_FETCH_PIPELINE_ID,
        )
        await version.insert()
        return version

    async def run(
        self,
        document: Document,
        stage_params: dict[str, Any],
        prev_stages: list[StageResult],
    ) -> StageResult:
        del stage_params, prev_stages
        cfg = self._config or get_config().tellus
        version = await self.get_version()
        version_id = version.version_id

        if document.type != DocumentType.TELLUS or not isinstance(
            document, TellusDocument
        ):
            return TellusDocumentFetchStageResult(
                version_id=version_id,
                status=Status.FAILED,
                error=(
                    "tellus_document_fetch requires a TellusDocument, "
                    f"got {document.type}"
                ),
            )

        if not document.external_id:
            return TellusDocumentFetchStageResult(
                version_id=version_id,
                status=Status.FAILED,
                error="TellusDocument.external_id is required",
            )

        try:
            payload = await self._fetch(document.external_id, cfg)
        except Exception as exc:
            logger.exception(
                "tellus_document_fetch failed for %s", document.external_id
            )
            return TellusDocumentFetchStageResult(
                version_id=version_id,
                status=Status.FAILED,
                error=str(exc),
            )
        document_id = payload.get("document_id")
        if not document_id or document_id != document.external_id:
            return TellusDocumentFetchStageResult(
                version_id=version_id,
                status=Status.FAILED,
                error=f"Document ID mismatch: {document_id} != {document.external_id}",
            )
        metadata = payload.get("document")
        if not metadata or not isinstance(metadata, dict):
            return TellusDocumentFetchStageResult(
                version_id=version_id,
                status=Status.FAILED,
                error=f"No metadata found for document {document.external_id}",
            )
        url = metadata.get("handle_url")
        if not url or not isinstance(url, str):
            return TellusDocumentFetchStageResult(
                version_id=version_id,
                status=Status.FAILED,
                error=f"No URL found for document {document.external_id}",
            )
        title: str | None = None
        for key in [
            "title",
            "title_english",
            "title_original",
            "subtitle_english",
            "subtitle_original",
        ]:
            title = metadata.get(key)
            if title and isinstance(title, str):
                break
        if not title or not isinstance(title, str):
            return TellusDocumentFetchStageResult(
                version_id=version_id,
                status=Status.FAILED,
                error=f"No title found for document {document.external_id}",
            )

        document.url = url
        document.title = title
        document.metadata = _sanitize_tellus_metadata(metadata)

        chunks = payload.get("chunks")
        if not chunks or not isinstance(chunks, list):
            return TellusDocumentFetchStageResult(
                version_id=version_id,
                status=Status.FAILED,
                error=f"No chunks found for document {document.external_id}",
            )

        if document.id is None:
            await document.insert()

        out_dir = cfg.save_dir / str(document.id)
        out_dir.mkdir(parents=True, exist_ok=True)
        page_paths: list[str] = []
        for index, chunk in enumerate(chunks, start=1):
            content = str(chunk.get("content") or "")
            page_path = out_dir / f"page_{index:04d}.md"
            page_path.write_text(content, encoding="utf-8")
            page_paths.append(str(page_path))

        document.page_paths = page_paths
        await document.save()

        return TellusDocumentFetchStageResult(
            version_id=version_id,
            status=Status.COMPLETED,
            title=title,
            num_pages=len(page_paths),
            page_paths=page_paths,
        )

    async def _fetch(self, document_id: str, cfg: TellusConfig) -> dict[str, Any]:
        if self._fetch_fn is not None:
            return await self._fetch_fn(document_id)
        return await tellus_get_all_document_chunks(document_id, config=cfg)
