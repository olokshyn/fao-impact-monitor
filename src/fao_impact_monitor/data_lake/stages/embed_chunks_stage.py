"""Embed document chunks and persist them in the vectorstore collection."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, cast

from fao_impact_monitor.config import VectorStoreConfig, get_config
from fao_impact_monitor.data_lake.common import Status
from fao_impact_monitor.data_lake.document import Document
from fao_impact_monitor.data_lake.stage import (
    Stage,
    StageResult,
    StageVersion,
)
from fao_impact_monitor.data_lake.stages.country_detect_stage import (
    CHUNK_ITERATOR_PARAM,
    COUNTRY_DETECT_STAGE_NAME,
    ChunkIterator,
    CountryDetectStageResult,
)
from fao_impact_monitor.data_lake.vectorstore import ChunkEmbedding, build_embeddings

logger = logging.getLogger(__name__)

EMBED_CHUNKS_STAGE_NAME = "embed_chunks"

EmbedFn = Callable[[Sequence[str]], Awaitable[list[list[float]]]]


class EmbedChunksStageResult(StageResult):
    name: str = EMBED_CHUNKS_STAGE_NAME
    chunk_count: int = 0


class EmbedChunksStageVersion(StageVersion):
    embedding_model: str
    embedding_dimensions: int | None = None

    class Settings:
        class_id_value = EMBED_CHUNKS_STAGE_NAME


def _require_chunk_iterator(stage_params: dict[str, Any]) -> ChunkIterator:
    chunk_iterator = stage_params.get(CHUNK_ITERATOR_PARAM)
    if chunk_iterator is None:
        raise ValueError(f"{CHUNK_ITERATOR_PARAM} must be set in stage_params")
    if not callable(chunk_iterator):
        raise TypeError(
            f"{CHUNK_ITERATOR_PARAM} must be a callable ChunkIterator, "
            f"got {type(chunk_iterator).__name__}"
        )
    return cast(ChunkIterator, chunk_iterator)


def _resolve_country_detect(
    prev_stages: list[StageResult],
) -> CountryDetectStageResult | None:
    for result in reversed(prev_stages):
        if result.name != COUNTRY_DETECT_STAGE_NAME:
            continue
        if isinstance(result, CountryDetectStageResult):
            detect = result
        else:
            detect = CountryDetectStageResult.model_validate(result.model_dump())
        if detect.status != Status.COMPLETED:
            return None
        return detect
    return None


class EmbedChunksStage(Stage):
    name = EMBED_CHUNKS_STAGE_NAME

    def __init__(
        self,
        *,
        embed_fn: EmbedFn | None = None,
        config: VectorStoreConfig | None = None,
    ) -> None:
        self._embed_fn = embed_fn
        self._config = config

    async def get_version(self) -> StageVersion:
        cfg = self._config or get_config().vector_store
        dims = "" if cfg.embedding_dimensions is None else str(cfg.embedding_dimensions)
        payload = f"{cfg.embedding_model}|{dims}"
        version_id = hashlib.sha256(payload.encode()).hexdigest()[:32]
        existing = await EmbedChunksStageVersion.find_one(
            EmbedChunksStageVersion.version_id == version_id
        )
        if existing is not None:
            return existing
        version = EmbedChunksStageVersion(
            version_id=version_id,
            embedding_model=cfg.embedding_model,
            embedding_dimensions=cfg.embedding_dimensions,
        )
        await version.insert()
        return version

    async def run(
        self,
        document: Document,
        stage_params: dict[str, Any],
        prev_stages: list[StageResult],
    ) -> StageResult:
        cfg = self._config or get_config().vector_store
        version = await self.get_version()
        version_id = version.version_id

        if document.id is None:
            return EmbedChunksStageResult(
                version_id=version_id,
                status=Status.FAILED,
                error="Document must be saved before embed_chunks",
                chunk_count=0,
            )

        chunk_iterator = _require_chunk_iterator(stage_params)
        country_detect = _resolve_country_detect(prev_stages)
        if country_detect is None:
            return EmbedChunksStageResult(
                version_id=version_id,
                status=Status.FAILED,
                error=(
                    f"embed_chunks requires a completed {COUNTRY_DETECT_STAGE_NAME} "
                    "result in previous stages"
                ),
                chunk_count=0,
            )

        logger.info("Running embed_chunks for %s", document.url)
        try:
            texts = list(chunk_iterator(prev_stages))
        except ValueError as exc:
            logger.warning("embed_chunks skipped for %s: %s", document.url, exc)
            return EmbedChunksStageResult(
                version_id=version_id,
                status=Status.FAILED,
                error=str(exc),
                chunk_count=0,
            )
        detections = country_detect.detections
        if len(detections) != len(texts):
            return EmbedChunksStageResult(
                version_id=version_id,
                status=Status.FAILED,
                error=(
                    f"country_detect detections length ({len(detections)}) does not "
                    f"match chunk count ({len(texts)})"
                ),
                chunk_count=0,
            )

        # Titan rejects empty strings; skip blank chunks but keep original indexes.
        indexed_chunks = [
            (index, text, detection)
            for index, (text, detection) in enumerate(
                zip(texts, detections, strict=True)
            )
            if text.strip()
        ]
        skipped = len(texts) - len(indexed_chunks)
        if skipped:
            logger.warning(
                "embed_chunks skipping %s empty chunk(s) for %s",
                skipped,
                document.url,
            )

        embed_texts = [text for _, text, _ in indexed_chunks]
        try:
            vectors = await self._embed_texts(embed_texts, cfg)
        except Exception as exc:
            logger.exception("embed_chunks failed for %s", document.url)
            return EmbedChunksStageResult(
                version_id=version_id,
                status=Status.FAILED,
                error=str(exc),
                chunk_count=0,
            )

        if len(vectors) != len(embed_texts):
            return EmbedChunksStageResult(
                version_id=version_id,
                status=Status.FAILED,
                error=(
                    f"embed_fn returned {len(vectors)} vectors for "
                    f"{len(embed_texts)} chunks"
                ),
                chunk_count=0,
            )

        await ChunkEmbedding.find(ChunkEmbedding.document_id == document.id).delete()

        for (index, text, detection), vector in zip(
            indexed_chunks, vectors, strict=True
        ):
            countries = (
                list(detection.countries_iso3)
                if detection.countries_iso3 is not None
                else []
            )
            chunk = ChunkEmbedding(
                document_id=document.id,
                document_url=document.url,
                document_external_id=document.external_id,
                document_title=document.title,
                document_meta=dict(document.metadata),
                document_type=document.type,
                document_source=document.source,
                chunk_index=index,
                chunk_text=text,
                countries_iso3=countries,
                embedding=vector,
            )
            await chunk.insert()

        return EmbedChunksStageResult(
            version_id=version_id,
            status=Status.COMPLETED,
            chunk_count=len(indexed_chunks),
        )

    async def _embed_texts(
        self,
        texts: Sequence[str],
        cfg: VectorStoreConfig,
    ) -> list[list[float]]:
        if self._embed_fn is not None:
            return await self._embed_fn(texts)
        if not texts:
            return []
        embeddings = build_embeddings(vector_store_config=cfg)
        batch_size = max(1, cfg.embed_batch_size)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            vectors.extend(await embeddings.aembed_documents(batch))
        return vectors
