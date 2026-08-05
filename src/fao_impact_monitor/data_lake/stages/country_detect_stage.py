"""Country detection stage over texts yielded from previous stage results."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Awaitable, Callable, Iterator
from typing import Any, cast

from pydantic import BaseModel, Field, model_validator

from fao_impact_monitor.agent.country_detect_agent import detect_countries
from fao_impact_monitor.config import CountryDetectConfig, get_config
from fao_impact_monitor.data_lake.common import Status
from fao_impact_monitor.data_lake.document import Document
from fao_impact_monitor.data_lake.stage import (
    Stage,
    StageResult,
    StageVersion,
)

logger = logging.getLogger(__name__)

COUNTRY_DETECT_STAGE_NAME = "country_detect"
CHUNK_ITERATOR_PARAM = "chunk_iterator"

ChunkIterator = Callable[[list[StageResult]], Iterator[str]]
DetectFn = Callable[..., Awaitable[tuple[list[str], list[str]]]]


class CountryDetection(BaseModel):
    countries_iso3: list[str] | None = None
    detections: list[str] | None = None
    error: str | None = None

    @model_validator(mode="after")
    def _xor_error_and_results(self) -> CountryDetection:
        has_results = self.countries_iso3 is not None and self.detections is not None
        has_error = self.error is not None
        if has_results == has_error:
            raise ValueError(
                "CountryDetection requires either (countries_iso3 and "
                "detections) or error, but not both or neither"
            )
        if (
            self.countries_iso3 is not None
            and self.detections is not None
            and len(self.countries_iso3) != len(self.detections)
        ):
            raise ValueError("countries_iso3 and detections must have the same length")
        return self


class CountryDetectStageResult(StageResult):
    name: str = COUNTRY_DETECT_STAGE_NAME
    detections: list[CountryDetection] = Field(default_factory=list)


class CountryDetectStageVersion(StageVersion):
    llm_model: str

    class Settings:
        class_id_value = COUNTRY_DETECT_STAGE_NAME


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


class CountryDetectStage(Stage):
    name = COUNTRY_DETECT_STAGE_NAME

    def __init__(
        self,
        *,
        detect_fn: DetectFn | None = None,
        config: CountryDetectConfig | None = None,
    ) -> None:
        self._detect_fn = detect_fn
        self._config = config

    async def get_version(self) -> StageVersion:
        cfg = self._config or get_config().country_detect
        version_id = hashlib.sha256(cfg.llm_model.encode()).hexdigest()[:32]
        existing = await CountryDetectStageVersion.find_one(
            CountryDetectStageVersion.version_id == version_id
        )
        if existing is not None:
            return existing
        version = CountryDetectStageVersion(
            version_id=version_id,
            llm_model=cfg.llm_model,
        )
        await version.insert()
        return version

    async def run(
        self,
        document: Document,
        stage_params: dict[str, Any],
        prev_stages: list[StageResult],
    ) -> StageResult:
        cfg = self._config or get_config().country_detect
        version = await self.get_version()
        version_id = version.version_id
        chunk_iterator = _require_chunk_iterator(stage_params)

        logger.info("Running country_detect for %s", document.url)
        # Upstream extract/fetch may have failed; the pipeline still invokes later
        # stages with truncated prev_stages. Match embed_chunks: soft-fail.
        try:
            texts = list(chunk_iterator(prev_stages))
        except ValueError as exc:
            logger.warning(
                "country_detect skipped for %s: %s",
                document.url,
                exc,
            )
            return CountryDetectStageResult(
                version_id=version_id,
                status=Status.FAILED,
                error=str(exc),
                detections=[],
            )

        detections: list[CountryDetection] = []
        for index, text in enumerate(texts):
            try:
                iso3s, spans = await self._detect(text, cfg)
                detections.append(
                    CountryDetection(
                        countries_iso3=iso3s,
                        detections=spans,
                        error=None,
                    )
                )
            except Exception as exc:
                logger.exception(
                    "country_detect failed for %s chunk %s",
                    document.url,
                    index,
                )
                detections.append(
                    CountryDetection(
                        countries_iso3=None,
                        detections=None,
                        error=str(exc),
                    )
                )

        return CountryDetectStageResult(
            version_id=version_id,
            status=Status.COMPLETED,
            detections=detections,
        )

    async def _detect(
        self,
        text: str,
        cfg: CountryDetectConfig,
    ) -> tuple[list[str], list[str]]:
        if self._detect_fn is not None:
            return await self._detect_fn(
                text,
                max_retries=cfg.max_agent_retries,
            )
        return await detect_countries(
            text,
            max_retries=cfg.max_agent_retries,
        )
