"""Section-aware, provenance-bearing PDF ingestion."""

from __future__ import annotations

import asyncio
import logging
import math
import re
import unicodedata
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, date, datetime
from itertools import pairwise
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, Protocol, cast

from beanie import PydanticObjectId

from fao_impact_monitor.config import PdfPipelineConfig, get_config
from fao_impact_monitor.data_lake.vectorstore import build_embeddings
from fao_impact_monitor.pdf_pipeline.artifacts import (
    PdfArtifacts,
    RenderedPage,
    sha256_file,
    sha256_text,
)
from fao_impact_monitor.pdf_pipeline.gemini import GeminiPdfClient
from fao_impact_monitor.pdf_pipeline.luna import LunaSectionSummarizer
from fao_impact_monitor.pdf_pipeline.models import (
    ArtifactRef,
    AssertionMode,
    BoundingBox,
    CountryContext,
    ElNinoEventId,
    EventContext,
    EvidenceUnit,
    ModelVersions,
    PdfDocumentRecord,
    PdfEmbeddingRecord,
    PromptVersions,
    SectionContextCard,
    SourceRegion,
    ValidationResult,
    VerifiedVisualFact,
    effective_events,
    is_searchable,
)
from fao_impact_monitor.pdf_pipeline.mongo import (
    claim_document,
    delete_document_embeddings,
    purge_pdf_document,
)
from fao_impact_monitor.utils.document_uri import file_document_uri

logger = logging.getLogger(__name__)


class PdfAnalysisClient(Protocol):
    async def document_structure(self, pdf_path: Path) -> dict[str, Any]: ...

    async def section_evidence(
        self,
        pdf_path: Path,
        *,
        section_title: str,
        pages: Sequence[int],
        included_pages: Sequence[int],
    ) -> dict[str, Any]: ...

    async def verify_visual(
        self, image_path: Path, facts: Sequence[str]
    ) -> dict[str, Any]: ...

    async def describe_visual(
        self, image_path: Path, *, context: str
    ) -> dict[str, Any]: ...


class SectionSummarizer(Protocol):
    async def summarize(self, *, title: str, scope_source_text: str) -> str: ...


EmbedTexts = Callable[[list[str]], Awaitable[list[list[float]]]]

_EVENT_IDS_BY_START_YEAR: dict[str, ElNinoEventId] = {
    "1997": "el_nino_1997_98",
    "2015": "el_nino_2015_16",
    "2018": "el_nino_2018_19",
    "2023": "el_nino_2023_24",
    "2026": "el_nino_2026_27",
}
_EVENT_IDS_BY_NEAR_YEAR: dict[str, ElNinoEventId] = {
    "1997": "el_nino_1997_98",
    "1998": "el_nino_1997_98",
    "2015": "el_nino_2015_16",
    "2016": "el_nino_2015_16",
    "2018": "el_nino_2018_19",
    "2019": "el_nino_2018_19",
    "2023": "el_nino_2023_24",
    "2024": "el_nino_2023_24",
    "2026": "el_nino_2026_27",
    "2027": "el_nino_2026_27",
}
_EVENT_ID_SET = frozenset(_EVENT_IDS_BY_START_YEAR.values())
_MAX_CONTAINER_SCOPE_PAGES = 3
_MAX_VISUAL_FACTS_PER_REQUEST = 5


def _now() -> datetime:
    return datetime.now(UTC)


def _validation(
    verdict: Literal["passed", "failed", "not_applicable"] = "passed",
    details: str | None = None,
) -> ValidationResult:
    return ValidationResult(
        verdict=verdict, checked_at=_now(), validator="pdf_pipeline", details=details
    )


def _versions(config: PdfPipelineConfig) -> ModelVersions:
    settings = get_config()
    return ModelVersions(
        gemini=settings.gemini.model,
        luna=config.luna_model,
        titan=settings.vector_store.embedding_model,
    )


def _event_contexts(
    values: Any, origin: Literal["explicit", "inherited"] = "explicit"
) -> list[EventContext]:
    result: list[EventContext] = []
    for value in values if isinstance(values, list) else []:
        if not isinstance(value, dict):
            continue
        event_id = _normalize_event_id(value.get("event_id"))
        relationship = value.get("relationship")
        if event_id is None or relationship not in {
            "attributed",
            "associated",
            "explicitly_not_attributed",
            "uncertain",
            "unrelated",
        }:
            continue
        result.append(
            EventContext(
                event_id=cast(Any, event_id),
                relationship=cast(Any, relationship),
                origin=origin,
            )
        )
    return result


def _normalized_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(
        character for character in decomposed if not unicodedata.combining(character)
    ).lower()


def _normalize_event_id(value: Any) -> ElNinoEventId | None:
    if not isinstance(value, str):
        return None
    normalized = _normalized_text(value).replace("–", "-").replace("—", "-")
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    if normalized in _EVENT_ID_SET:
        return normalized
    match = re.search(
        r"(?:^|_)(1997|1998|2015|2016|2018|2019|2023|2024|2026|2027)"
        r"(?:_(?:19|20)?\d{2})?(?:_|$)",
        normalized,
    )
    return _EVENT_IDS_BY_NEAR_YEAR.get(match.group(1)) if match else None


def _mentions_el_nino(value: str) -> bool:
    return re.search(r"\bel\s+nino\b", _normalized_text(value)) is not None


def _event_ids_near_el_nino(value: str) -> list[ElNinoEventId]:
    """Resolve allowed episodes only from years near an explicit El Niño mention."""
    normalized = _normalized_text(value).replace("–", "-").replace("—", "-")
    event_ids: list[ElNinoEventId] = []
    for mention in re.finditer(r"\bel\s+nino\b", normalized):
        window = normalized[max(0, mention.start() - 300) : mention.end() + 500]
        for year, event_id in _EVENT_IDS_BY_NEAR_YEAR.items():
            if re.search(rf"\b{year}\b", window) and event_id not in event_ids:
                event_ids.append(event_id)
    return event_ids


def _relationship_from_source(value: str) -> str:
    normalized = _normalized_text(value)
    negation_patterns = (
        r"(?:not|no evidence)\s+(?:caused|driven|linked|associated|attributed).{0,80}\bel\s+nino\b",
        r"\bel\s+nino\b.{0,80}(?:not|no evidence).{0,30}(?:caused|linked|associated|attributed)",
    )
    if any(re.search(pattern, normalized) for pattern in negation_patterns):
        return "explicitly_not_attributed"
    causal_patterns = (
        r"(?:caused by|due to|attributed to|driven by).{0,80}\bel\s+nino\b",
        r"\bel\s+nino\b.{0,160}(?:caused|led to|resulted in|will result in|driven|triggered|negatively affect)",
    )
    if any(re.search(pattern, normalized) for pattern in causal_patterns):
        return "attributed"
    return "associated"


def _events_from_source_text(
    source_text: str | None,
    *,
    document_events: Sequence[EventContext],
    supporting_evidence_id: str,
) -> list[EventContext]:
    """Recover event metadata from exact source text when Gemini omits it."""
    if not source_text or not _mentions_el_nino(source_text):
        return []
    event_ids = _event_ids_near_el_nino(source_text)
    if not event_ids:
        candidates = list(dict.fromkeys(event.event_id for event in document_events))
        if len(candidates) == 1:
            event_ids = candidates
    relationship = _relationship_from_source(source_text)
    return [
        EventContext(
            event_id=cast(Any, event_id),
            relationship=cast(Any, relationship),
            origin="explicit",
            supporting_evidence_ids=[supporting_evidence_id],
        )
        for event_id in event_ids
    ]


def _document_events(
    structure: Mapping[str, Any],
    sections: Sequence[Mapping[str, Any]],
    pages: Sequence[RenderedPage],
) -> list[EventContext]:
    """Collect candidate episodes without using publication year as evidence."""
    values: list[Any] = []
    raw_document_events = structure.get("events")
    if isinstance(raw_document_events, list):
        values.extend(raw_document_events)
    for section in sections:
        raw_section_events = section.get("events")
        if isinstance(raw_section_events, list):
            values.extend(raw_section_events)
    result = _event_contexts(values)
    known = {event.event_id for event in result}
    for page in pages:
        for event_id in _event_ids_near_el_nino(page.extracted_text):
            if event_id not in known:
                result.append(
                    EventContext(
                        event_id=cast(Any, event_id),
                        relationship="associated",
                        origin="explicit",
                    )
                )
                known.add(event_id)
    return result


def _country_contexts(
    values: Any, origin: Literal["explicit", "inherited"] = "explicit"
) -> list[CountryContext]:
    return [
        CountryContext(iso3=value, role="subject", origin=origin)
        for value in values
        if isinstance(values, list) and isinstance(value, str) and len(value) == 3
    ]


def _reporting_modes(values: Any) -> set[AssertionMode]:
    allowed = {
        "contextual",
        "observed",
        "reported",
        "estimated",
        "forecast",
        "scenario",
        "measured_response",
    }
    return {
        cast(AssertionMode, value)
        for value in values
        if isinstance(values, list) and value in allowed
    }


class PdfEvidenceIngestor:
    """Ingest one directory without depending on the Hydra/data-lake pipeline."""

    def __init__(
        self,
        *,
        config: PdfPipelineConfig | None = None,
        analysis_client: PdfAnalysisClient | None = None,
        summarizer: SectionSummarizer | None = None,
        embed_texts: EmbedTexts | None = None,
    ) -> None:
        self.config = config or get_config().pdf_pipeline
        self.analysis_client = analysis_client
        self.summarizer = summarizer
        self.embed_texts = embed_texts
        self.prompts = PromptVersions()

    async def ingest_directory(
        self, directory: Path, *, force: bool = False, limit: int | None = None
    ) -> dict[str, int]:
        started_at = perf_counter()
        selected = await self.select_directory_pdfs(directory, force=force, limit=limit)
        counts = {"processed": 0, "skipped": 0, "failed": 0}
        total = len(selected)
        for index, pdf in enumerate(selected, start=1):
            document_started_at = perf_counter()
            logger.info("[%d/%d] Starting %s", index, total, pdf)
            try:
                changed = await self.ingest_pdf(pdf, force=force)
                counts["processed" if changed else "skipped"] += 1
            except Exception as error:  # noqa: BLE001 - isolate per-document failures
                counts["failed"] += 1
                logger.error(
                    "[%d/%d] Failed %s after %.1fs: %s",
                    index,
                    total,
                    pdf,
                    perf_counter() - document_started_at,
                    error,
                )
                continue
            logger.info(
                "[%d/%d] %s %s in %.1fs",
                index,
                total,
                "Completed" if changed else "Skipped",
                pdf,
                perf_counter() - document_started_at,
            )
        logger.info(
            "Folder ingestion finished in %.1fs: processed=%d skipped=%d failed=%d",
            perf_counter() - started_at,
            counts["processed"],
            counts["skipped"],
            counts["failed"],
        )
        return counts

    async def select_directory_pdfs(
        self, directory: Path, *, force: bool = False, limit: int | None = None
    ) -> list[Path]:
        """Select the folder queue once before sequential or parallel processing."""
        if limit is not None and limit < 1:
            raise ValueError("limit must be at least 1")
        logger.info("Discovering PDFs under %s", directory)
        pdfs = sorted(path for path in directory.rglob("*.pdf") if path.is_file())
        logger.info("Discovered %d PDF document(s)", len(pdfs))
        if force:
            selected = pdfs[:limit]
        else:
            logger.info("Checking document hashes against MongoDB")
            hashes = {path: sha256_file(path) for path in pdfs}
            cursor = PdfDocumentRecord.get_pymongo_collection().find(
                {"content_sha256": {"$in": list(hashes.values())}},
                {"content_sha256": 1, "status": 1},
            )
            records = {
                row["content_sha256"]: row
                async for row in cursor
                if isinstance(row.get("content_sha256"), str)
            }
            selected = self._select_unprocessed(pdfs, hashes, records, limit)
        logger.info(
            "Queued %d of %d discovered PDF document(s)%s",
            len(selected),
            len(pdfs),
            " with force enabled" if force else "",
        )
        return selected

    async def ingest_single_pdf(
        self, pdf: Path, *, force: bool = False
    ) -> dict[str, int]:
        """Resume failed work; restart completed work or when force is explicit."""
        if not pdf.is_file() or pdf.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF file: {pdf}")
        content_sha256 = sha256_file(pdf)
        existing = await PdfDocumentRecord.get_pymongo_collection().find_one(
            {"content_sha256": content_sha256},
            {"status": 1, "structure_analysis": 1},
        )
        checkpoint_count = (
            await SectionContextCard.get_pymongo_collection().count_documents(
                {"document_sha256": content_sha256}
            )
        )
        has_checkpoint = checkpoint_count > 0 or (
            existing is not None
            and isinstance(existing.get("structure_analysis"), dict)
        )
        restart = force or bool(
            existing is not None
            and (existing.get("status") == "completed" or not has_checkpoint)
        )
        logger.info(
            "Starting explicit %s of %s",
            "restart" if restart else "resume",
            pdf,
        )
        changed = await self.ingest_pdf(pdf, force=restart)
        logger.info("Explicit ingestion completed for %s", pdf)
        return {"processed": int(changed), "skipped": int(not changed), "failed": 0}

    @staticmethod
    def _select_unprocessed(
        pdfs: Sequence[Path],
        hashes: Mapping[Path, str],
        records: Mapping[str, Mapping[str, Any]],
        limit: int | None,
    ) -> list[Path]:
        """Return the next unseen/failed hashes; never spend a slot on a duplicate."""
        selected: list[Path] = []
        seen_hashes: set[str] = set()
        for pdf in pdfs:
            content_sha256 = hashes[pdf]
            if content_sha256 in seen_hashes:
                continue
            seen_hashes.add(content_sha256)
            record = records.get(content_sha256)
            if record is not None and record.get("status") == "completed":
                continue
            selected.append(pdf)
            if limit is not None and len(selected) == limit:
                break
        return selected

    async def ingest_pdf(self, pdf: Path, *, force: bool = False) -> bool:
        logger.info("%s: hashing source document", pdf.name)
        content_sha256 = sha256_file(pdf)
        if force:
            logger.info("%s: deleting all prior Mongo records", pdf.name)
            await purge_pdf_document(content_sha256)
            logger.info("%s: verified prior Mongo records are gone", pdf.name)
        logger.info("%s: rendering pages and preparing artifacts", pdf.name)
        artifacts = PdfArtifacts(
            self.config.artifact_dir, content_sha256, self.config.render_dpi
        )
        source_artifact, pages = await asyncio.to_thread(artifacts.prepare, pdf)
        logger.info("%s: prepared %d page(s)", pdf.name, len(pages))
        now = _now()
        seed = {
            "content_sha256": content_sha256,
            "original_filename": pdf.name,
            "discovered_paths": [str(pdf)],
            "source_artifact": source_artifact.model_dump(),
            "physical_page_count": len(pages),
            "status": "processing",
            "model_versions": _versions(self.config).model_dump(),
            "prompt_versions": self.prompts.model_dump(),
            "pipeline": "pdf_pipeline",
            "created_at": now,
            "updated_at": now,
        }
        claimed, should_process = await claim_document(
            content_sha256=content_sha256,
            path=str(pdf),
            seed=seed,
        )
        if not should_process or claimed is None:
            logger.info("%s: already completed", pdf.name)
            return False
        document_id = PydanticObjectId(claimed["_id"])
        try:
            await self._process(pdf, artifacts, pages, document_id, content_sha256)
        except BaseException as error:
            logger.exception("%s: processing failed", pdf.name)
            failure = str(error) or type(error).__name__
            await PdfDocumentRecord.get_pymongo_collection().update_one(
                {"_id": document_id},
                {
                    "$set": {
                        "status": "failed",
                        "failure": failure,
                        "updated_at": _now(),
                    },
                    "$unset": {"completed_at": ""},
                },
            )
            try:
                await delete_document_embeddings(content_sha256)
            except Exception:
                logger.exception("%s: failed to remove partial embeddings", pdf.name)
            logger.info(
                "%s: preserved completed section checkpoints for resume",
                pdf.name,
            )
            raise
        await PdfDocumentRecord.get_pymongo_collection().update_one(
            {"_id": document_id},
            {
                "$set": {
                    "status": "completed",
                    "completed_at": _now(),
                    "updated_at": _now(),
                }
            },
        )
        logger.info("%s: marked completed in MongoDB", pdf.name)
        return True

    async def _process(
        self,
        pdf: Path,
        artifacts: PdfArtifacts,
        pages: list[RenderedPage],
        document_id: PydanticObjectId,
        content_sha256: str,
    ) -> None:
        client = self.analysis_client or GeminiPdfClient()
        document = await PdfDocumentRecord.get(document_id)
        cached_structure = document.structure_analysis if document is not None else None
        if isinstance(cached_structure, dict):
            structure = cached_structure
            logger.info("%s: resuming cached document structure", pdf.name)
        else:
            logger.info("%s: analyzing document structure", pdf.name)
            structure = await client.document_structure(
                artifacts.directory / "source.pdf"
            )
            await PdfDocumentRecord.get_pymongo_collection().update_one(
                {"_id": document_id},
                {
                    "$set": {
                        "structure_analysis": structure,
                        "title": structure.get("title"),
                        "publication_date": self._mongo_date_or_none(
                            structure.get("publication_date")
                        ),
                        "reporting_period": structure.get("reporting_period"),
                        "purpose": structure.get("purpose"),
                        "updated_at": _now(),
                    }
                },
            )
            logger.info("%s: document structure checkpoint saved", pdf.name)
        raw_sections = structure.get("sections")
        section_drafts = raw_sections if isinstance(raw_sections, list) else []
        sections: list[dict[str, Any]] = [
            draft for draft in section_drafts if isinstance(draft, dict)
        ]
        if not sections:
            sections = [
                {
                    "ordinal": 1,
                    "parent_ordinal": None,
                    "level": 1,
                    "title": structure.get("title") or pdf.stem,
                    "page_start": 1,
                    "page_end": len(pages),
                    "countries": [],
                    "regions": [],
                    "events": [],
                    "reporting_modes": [],
                    "scope_pages": [1],
                }
            ]
        logger.info("%s: identified %d section(s)", pdf.name, len(sections))
        document_events = _document_events(structure, sections, pages)
        container_ordinals = {
            parent
            for section in sections
            if isinstance((parent := section.get("parent_ordinal")), int)
        }
        checkpointed_section_ids = set(
            await SectionContextCard.get_pymongo_collection().distinct(
                "section_id", {"document_sha256": content_sha256}
            )
        )
        if checkpointed_section_ids:
            logger.info(
                "%s: found %d completed section checkpoint(s)",
                pdf.name,
                len(checkpointed_section_ids),
            )
        for position, draft in enumerate(sections, start=1):
            page_start = max(1, int(draft.get("page_start", 1)))
            page_end = min(len(pages), int(draft.get("page_end", len(pages))))
            if page_end < page_start:
                raise ValueError(f"invalid section page range {page_start}-{page_end}")
            section_id = f"{content_sha256}:s{position:03d}"
            if section_id in checkpointed_section_ids:
                logger.info(
                    "%s: section %d/%d already checkpointed; skipping model calls",
                    pdf.name,
                    position,
                    len(sections),
                )
                continue
            await EvidenceUnit.get_pymongo_collection().delete_many(
                {"section_id": section_id}
            )
            title = str(draft.get("title") or f"Section {position}")
            raw_ordinal = draft.get("ordinal")
            ordinal = raw_ordinal if isinstance(raw_ordinal, int) else position
            is_container = ordinal in container_ordinals
            raw_scope_pages = draft.get("scope_pages")
            scope_page_values = (
                raw_scope_pages if isinstance(raw_scope_pages, list) else []
            )
            scope_pages = [
                page
                for page in scope_page_values
                if isinstance(page, int) and page_start <= page <= page_end
            ]
            analysis_pages = (
                list(dict.fromkeys(scope_pages))[:_MAX_CONTAINER_SCOPE_PAGES]
                or [page_start]
                if is_container
                else list(range(page_start, page_end + 1))
            )
            logger.info(
                "%s: section %d/%d %r (pages %d-%d%s)",
                pdf.name,
                position,
                len(sections),
                title,
                page_start,
                page_end,
                f"; container scope pages {analysis_pages}" if is_container else "",
            )
            request_pages = self._with_adjacent_pages(analysis_pages, len(pages))
            section_pdf = await asyncio.to_thread(artifacts.section_pdf, request_pages)
            logger.info(
                "%s: section %d/%d sending %d-page excerpt %s; target pages %s",
                pdf.name,
                position,
                len(sections),
                len(request_pages),
                request_pages,
                analysis_pages,
            )
            analysis = await client.section_evidence(
                section_pdf,
                section_title=title,
                pages=analysis_pages,
                included_pages=request_pages,
            )
            boundary_details = "Adjacent excerpt pages confirmed candidate boundaries"
            if not is_container:
                corrected_start, corrected_end, boundary_details = (
                    self._validated_section_range(
                        analysis,
                        page_start=page_start,
                        page_end=page_end,
                        included_pages=request_pages,
                    )
                )
                if (corrected_start, corrected_end) != (page_start, page_end):
                    logger.info(
                        "%s: section %d/%d corrected boundaries from %d-%d to %d-%d",
                        pdf.name,
                        position,
                        len(sections),
                        page_start,
                        page_end,
                        corrected_start,
                        corrected_end,
                    )
                    page_start, page_end = corrected_start, corrected_end
                    analysis_pages = list(range(page_start, page_end + 1))
            raw_scope_units = analysis.get("scope_units")
            unfiltered_scope_units: list[Any] = (
                raw_scope_units if isinstance(raw_scope_units, list) else []
            )
            raw_units = analysis.get("evidence_units")
            unfiltered_units: list[Any] = (
                raw_units if isinstance(raw_units, list) else []
            )
            allowed_unit_pages = set(analysis_pages)
            scope_units = [
                unit
                for unit in unfiltered_scope_units
                if self._draft_within_pages(unit, allowed_unit_pages)
            ]
            units = [
                unit
                for unit in unfiltered_units
                if self._draft_within_pages(unit, allowed_unit_pages)
            ]
            rejected_units = (len(unfiltered_scope_units) - len(scope_units)) + (
                len(unfiltered_units) - len(units)
            )
            if rejected_units:
                logger.warning(
                    "%s: section %d/%d rejected %d unit(s) outside corrected pages %s",
                    pdf.name,
                    position,
                    len(sections),
                    rejected_units,
                    analysis_pages,
                )
            if is_container:
                units = []
            if not scope_units and not units:
                scope_units = self._fallback_scope_drafts(
                    pages=[pages[page - 1] for page in analysis_pages],
                    countries=draft.get("countries"),
                    document_events=document_events,
                    section_events=_event_contexts(draft.get("events")),
                )
                if scope_units:
                    logger.warning(
                        "%s: section %d/%d returned no evidence; recovered %d "
                        "exact-text El Niño passage(s) locally",
                        pdf.name,
                        position,
                        len(sections),
                        len(scope_units),
                    )
            logger.info(
                "%s: section %d/%d returned %d scope and %d evidence unit(s)",
                pdf.name,
                position,
                len(sections),
                len(scope_units),
                len(units),
            )
            section_scope: list[EvidenceUnit] = []
            for item_position, unit in enumerate(scope_units, start=1):
                logger.info(
                    "%s: section %d/%d validating scope unit %d/%d",
                    pdf.name,
                    position,
                    len(sections),
                    item_position,
                    len(scope_units),
                )
                try:
                    item = await self._unit_from_draft(
                        draft=unit,
                        prefix=f"{section_id}:scope{item_position:03d}",
                        document_id=document_id,
                        document_sha256=content_sha256,
                        section_id=section_id,
                        pages=pages,
                        artifacts=artifacts,
                        document_events=document_events,
                        inherited_events=[],
                        inherited_countries=[],
                        scope_ids=[],
                        client=client,
                        required_kind="scope_context",
                    )
                except (TypeError, ValueError) as error:
                    logger.warning(
                        "%s: skipping malformed scope unit %d/%d in section %d: %s",
                        pdf.name,
                        item_position,
                        len(scope_units),
                        position,
                        error,
                    )
                    continue
                section_scope.append(item)
            scope_ids = [item.evidence_id for item in section_scope]
            for scope_unit in section_scope:
                scope_unit.events = [
                    event.model_copy(
                        update={"supporting_evidence_ids": [scope_unit.evidence_id]}
                    )
                    for event in scope_unit.events
                ]
                scope_unit.countries = [
                    country.model_copy(
                        update={"supporting_evidence_ids": [scope_unit.evidence_id]}
                    )
                    for country in scope_unit.countries
                ]
            base_events = [
                event.model_copy(update={"supporting_evidence_ids": scope_ids})
                for event in _event_contexts(draft.get("events"))
            ]
            inherited_events = [item for unit in section_scope for item in unit.events]
            effective_section_events = effective_events(base_events, inherited_events)
            scope_text = "\n".join(item.source_text or "" for item in section_scope)
            summary = await self._summarize(title, scope_text)
            card = SectionContextCard(
                section_id=section_id,
                document_id=document_id,
                document_sha256=content_sha256,
                parent_section_id=self._parent_id(content_sha256, draft, sections),
                ordinal=position,
                level=int(draft.get("level", 1)),
                title=title,
                hierarchy_path=[title],
                physical_page_start=page_start,
                physical_page_end=page_end,
                printed_page_start=self._string_or_none(
                    draft.get("printed_page_start")
                ),
                printed_page_end=self._string_or_none(draft.get("printed_page_end")),
                countries=[
                    country.model_copy(update={"supporting_evidence_ids": scope_ids})
                    for country in _country_contexts(draft.get("countries"))
                ],
                regions=[str(x) for x in draft.get("regions", [])],
                events=effective_section_events,
                reporting_modes=_reporting_modes(draft.get("reporting_modes")),
                scope_evidence_ids=scope_ids,
                summary=summary,
                retrieval_text=summary,
                boundary_validation=_validation(details=boundary_details),
                context_validation=_validation(details="Backed by scope_context units"),
                model_versions=_versions(self.config),
                prompt_versions=self.prompts,
                created_at=_now(),
            )
            section_evidence = list(section_scope)
            inherited_countries = card.countries
            for item_position, unit in enumerate(units, start=1):
                logger.info(
                    "%s: section %d/%d validating evidence unit %d/%d",
                    pdf.name,
                    position,
                    len(sections),
                    item_position,
                    len(units),
                )
                try:
                    item = await self._unit_from_draft(
                        draft=unit,
                        prefix=f"{section_id}:e{item_position:03d}",
                        document_id=document_id,
                        document_sha256=content_sha256,
                        section_id=section_id,
                        pages=pages,
                        artifacts=artifacts,
                        document_events=document_events,
                        inherited_events=effective_section_events,
                        inherited_countries=inherited_countries,
                        scope_ids=scope_ids,
                        client=client,
                    )
                except (TypeError, ValueError) as error:
                    logger.warning(
                        "%s: skipping malformed evidence unit %d/%d in section %d: %s",
                        pdf.name,
                        item_position,
                        len(units),
                        position,
                        error,
                    )
                    continue
                section_evidence.append(item)

            await self._save_section_checkpoint(
                document_id=document_id,
                card=card,
                evidence=section_evidence,
            )
            logger.info(
                "%s: section %d/%d checkpoint saved (%d evidence cards)",
                pdf.name,
                position,
                len(sections),
                len(section_evidence),
            )

        saved_cards = [
            SectionContextCard.model_validate(row)
            async for row in SectionContextCard.get_pymongo_collection().find(
                {"document_id": document_id}
            )
        ]
        saved_evidence = [
            EvidenceUnit.model_validate(row)
            async for row in EvidenceUnit.get_pymongo_collection().find(
                {"document_id": document_id}
            )
        ]
        await PdfDocumentRecord.get_pymongo_collection().update_one(
            {"_id": document_id},
            {
                "$set": {
                    "title": structure.get("title"),
                    "publication_date": self._mongo_date_or_none(
                        structure.get("publication_date")
                    ),
                    "reporting_period": structure.get("reporting_period"),
                    "purpose": structure.get("purpose"),
                    "root_section_ids": [
                        card.section_id
                        for card in saved_cards
                        if card.parent_section_id is None
                    ],
                    "covered_events": [
                        event.model_dump()
                        for card in saved_cards
                        for event in card.events
                    ],
                }
            },
        )
        await delete_document_embeddings(content_sha256)
        document = await PdfDocumentRecord.get(document_id)
        logger.info("%s: building searchable embeddings", pdf.name)
        await self._embed(
            saved_cards,
            saved_evidence,
            pdf,
            document_title=document.title if document is not None else None,
        )
        logger.info("%s: cards and embeddings stored successfully", pdf.name)

    @staticmethod
    async def _save_section_checkpoint(
        *,
        document_id: PydanticObjectId,
        card: SectionContextCard,
        evidence: list[EvidenceUnit],
    ) -> None:
        """Persist evidence first and the section card last as completion marker."""
        if evidence:
            await EvidenceUnit.insert_many(evidence)
        await card.insert()
        await PdfDocumentRecord.get_pymongo_collection().update_one(
            {"_id": document_id},
            {
                "$addToSet": {"completed_section_ids": card.section_id},
                "$set": {"updated_at": _now()},
            },
        )

    @staticmethod
    def _with_adjacent_pages(
        target_pages: Sequence[int], physical_page_count: int
    ) -> list[int]:
        """Add one available source page on either side of an extraction window."""
        if not target_pages:
            raise ValueError("section evidence requires at least one target page")
        first = min(target_pages)
        last = max(target_pages)
        included = set(target_pages)
        if first > 1:
            included.add(first - 1)
        if last < physical_page_count:
            included.add(last + 1)
        return sorted(included)

    @staticmethod
    def _validated_section_range(
        analysis: Mapping[str, Any],
        *,
        page_start: int,
        page_end: int,
        included_pages: Sequence[int],
    ) -> tuple[int, int, str]:
        """Accept only a one-page boundary correction supported by the excerpt."""
        corrected_start = analysis.get("corrected_page_start")
        corrected_end = analysis.get("corrected_page_end")
        rationale = str(analysis.get("boundary_rationale") or "").strip()
        valid = (
            isinstance(corrected_start, int)
            and isinstance(corrected_end, int)
            and corrected_start <= corrected_end
            and corrected_start in included_pages
            and corrected_end in included_pages
            and abs(corrected_start - page_start) <= 1
            and abs(corrected_end - page_end) <= 1
        )
        if not valid:
            return (
                page_start,
                page_end,
                "Ignored invalid Gemini boundary correction; retained hierarchy pages",
            )
        assert isinstance(corrected_start, int)
        assert isinstance(corrected_end, int)
        details = (
            rationale or "Gemini confirmed candidate boundaries using adjacent pages"
        )
        return corrected_start, corrected_end, details

    @staticmethod
    def _draft_within_pages(draft: Any, allowed_pages: set[int]) -> bool:
        """Reject evidence emitted from context pages outside the accepted section."""
        if not isinstance(draft, dict):
            return True
        identified_pages = {
            page
            for page in draft.get("pages", [])
            if isinstance(page, int) and not isinstance(page, bool)
        }
        regions = draft.get("regions")
        if isinstance(regions, list):
            identified_pages.update(
                page
                for region in regions
                if isinstance(region, dict)
                and isinstance((page := region.get("page")), int)
                and not isinstance(page, bool)
            )
        return not identified_pages or identified_pages <= allowed_pages

    def _fallback_scope_drafts(
        self,
        *,
        pages: Sequence[RenderedPage],
        countries: Any,
        document_events: Sequence[EventContext],
        section_events: Sequence[EventContext],
    ) -> list[dict[str, Any]]:
        """Recover exact El Niño passages when section extraction is empty."""
        drafts: list[dict[str, Any]] = []
        country_values = countries if isinstance(countries, list) else []
        valid_countries = [
            value
            for value in country_values
            if isinstance(value, str) and len(value) == 3
        ]
        for page in pages:
            page_event_ids = _event_ids_near_el_nino(page.extracted_text)
            for passage in self._el_nino_passages(page.extracted_text):
                passage_event_ids = _event_ids_near_el_nino(passage)
                if not passage_event_ids and len(section_events) == 1:
                    passage_event_ids = [section_events[0].event_id]
                if not passage_event_ids and len(page_event_ids) == 1:
                    passage_event_ids = page_event_ids
                if not passage_event_ids and len(document_events) == 1:
                    passage_event_ids = [document_events[0].event_id]
                relationship = _relationship_from_source(passage)
                mode = (
                    "forecast"
                    if re.search(
                        r"\b(?:forecast|outlook|risk|likely|expected|potential|may)\b",
                        _normalized_text(passage),
                    )
                    else "reported"
                )
                drafts.append(
                    {
                        "kind": "scope_context",
                        "assertion_mode": mode,
                        "modality": "text",
                        "pages": [page.physical_page],
                        "regions": [
                            {
                                "page": page.physical_page,
                                "printed_page": None,
                                "bbox": None,
                            }
                        ],
                        "source_text": passage,
                        "unit_description": (
                            f"Exact El Niño scope passage on physical page "
                            f"{page.physical_page}"
                        ),
                        "retrieval_text": passage,
                        "countries": valid_countries,
                        "events": [
                            {
                                "event_id": event_id,
                                "relationship": relationship,
                            }
                            for event_id in passage_event_ids
                        ],
                        "visual_facts": [],
                    }
                )
        return drafts

    @staticmethod
    def _el_nino_passages(text: str) -> list[str]:
        """Return exact, minimally contextual sentence groups mentioning El Niño."""
        if not _mentions_el_nino(text):
            return []
        boundaries = [0]
        boundaries.extend(
            match.end() for match in re.finditer(r"[.!?][\"'’”)]*\s+", text)
        )
        boundaries.append(len(text))
        spans = [
            (start, end)
            for start, end in pairwise(boundaries)
            if text[start:end].strip()
        ]
        passages: list[str] = []
        for index, (start, end) in enumerate(spans):
            sentence = text[start:end]
            if not _mentions_el_nino(sentence):
                continue
            contextual_end = end
            if index + 1 < len(spans):
                next_start, next_end = spans[index + 1]
                if not _mentions_el_nino(text[next_start:next_end]):
                    contextual_end = next_end
            passage = text[start:contextual_end].strip()
            if passage and passage not in passages:
                passages.append(passage)
        return passages

    async def _unit_from_draft(
        self,
        *,
        draft: Any,
        prefix: str,
        document_id: PydanticObjectId,
        document_sha256: str,
        section_id: str,
        pages: list[RenderedPage],
        artifacts: PdfArtifacts,
        document_events: list[EventContext],
        inherited_events: list[EventContext],
        inherited_countries: list[CountryContext],
        scope_ids: list[str],
        client: PdfAnalysisClient,
        required_kind: str | None = None,
    ) -> EvidenceUnit:
        if not isinstance(draft, dict):
            raise TypeError("Gemini evidence draft must be an object")
        source_text = draft.get("source_text")
        source_text = (
            source_text.strip()
            if isinstance(source_text, str) and source_text.strip()
            else None
        )
        numbers = [
            int(page)
            for page in draft.get("pages", [])
            if isinstance(page, int) and 1 <= page <= len(pages)
        ]
        if not numbers:
            raise ValueError("evidence unit has no valid source page")
        source_pages = [pages[number - 1] for number in dict.fromkeys(numbers)]
        page_texts = self._exact_source_texts_by_page(source_text, source_pages)
        source_text = "\n".join(page_texts.values()) or None
        regions: list[SourceRegion] = []
        raw_regions = draft.get("regions")
        region_drafts: list[Any] = raw_regions if isinstance(raw_regions, list) else []
        for index, page in enumerate(source_pages):
            region_draft = (
                region_drafts[index]
                if index < len(region_drafts) and isinstance(region_drafts[index], dict)
                else {}
            )
            bbox_value = region_draft.get("bbox")
            bbox = self._normalized_bounding_box(bbox_value, prefix=prefix)
            crop = (
                artifacts.crop(page.physical_page, bbox) if bbox is not None else None
            )
            text = page_texts.get(page.physical_page)
            regions.append(
                SourceRegion(
                    region_id=f"{prefix}:r{index + 1}",
                    physical_page=page.physical_page,
                    printed_page=self._string_or_none(region_draft.get("printed_page")),
                    page_width_points=page.width_points,
                    page_height_points=page.height_points,
                    rotation_degrees=cast(
                        Literal[0, 90, 180, 270], page.rotation_degrees
                    ),
                    bounding_box=bbox,
                    extracted_text=text,
                    extracted_text_sha256=sha256_text(text) if text else None,
                    page_artifact=page.artifact,
                    crop_artifact=crop,
                )
            )
        direct_events = [
            event.model_copy(update={"supporting_evidence_ids": [prefix]})
            for event in _event_contexts(draft.get("events"))
        ]
        inferred_events = _events_from_source_text(
            source_text,
            document_events=document_events,
            supporting_evidence_id=prefix,
        )
        known_event_ids = {event.event_id for event in direct_events}
        direct_events.extend(
            event for event in inferred_events if event.event_id not in known_event_ids
        )
        events = effective_events(direct_events, inherited_events)
        direct_countries = [
            country.model_copy(update={"supporting_evidence_ids": [prefix]})
            for country in _country_contexts(draft.get("countries"))
        ]
        countries = direct_countries or [
            country.model_copy(
                update={"origin": "inherited", "supporting_evidence_ids": scope_ids}
            )
            for country in inherited_countries
        ]
        modality = self._source_modality(draft.get("modality"))
        visual_facts = await self._verify_visual_facts(
            draft,
            regions,
            artifacts,
            client,
            prefix,
            modality=modality,
        )
        searchable, exclusion = is_searchable(events)
        exact = "\n".join(
            region.extracted_text for region in regions if region.extracted_text
        )
        source_text = exact or None
        if searchable and source_text is None and not visual_facts:
            searchable = False
            exclusion = "no_validated_source_evidence"
        canonical = self._canonical(source_text, visual_facts, regions)
        kind = required_kind or draft.get("kind", "agrifood_impact")
        if kind not in {
            "scope_context",
            "hazard",
            "agrifood_impact",
            "livelihood_impact",
            "response_outcome",
        }:
            kind = "agrifood_impact"
        mode = draft.get("assertion_mode", "reported")
        if mode not in {
            "contextual",
            "observed",
            "reported",
            "estimated",
            "forecast",
            "scenario",
            "measured_response",
        }:
            mode = "reported"
        return EvidenceUnit(
            evidence_id=prefix,
            document_id=document_id,
            document_sha256=document_sha256,
            section_id=section_id,
            evidence_kind=kind,
            assertion_mode=mode,
            source_modality=modality,
            source_regions=regions,
            physical_pages=[page.physical_page for page in source_pages],
            printed_pages=[
                region.printed_page
                for region in regions
                if region.printed_page is not None
            ],
            source_text=source_text,
            verified_visual_facts=visual_facts,
            unit_description=str(draft.get("unit_description") or ""),
            retrieval_text=str(draft.get("retrieval_text") or ""),
            countries=countries,
            events=events,
            scope_evidence_ids=scope_ids,
            continuation_evidence_ids=[],
            canonical_evidence_text=canonical,
            searchable=searchable,
            exclusion_reason=exclusion,
            extraction_validation=_validation(
                details="Exact text found on cited source page"
            ),
            visual_validation=(
                _validation(
                    verdict="passed" if visual_facts else "failed",
                    details=(
                        None
                        if visual_facts
                        else "No visual fact could be independently verified"
                    ),
                )
                if modality != "text"
                else None
            ),
            eligibility_validation=_validation(
                details=exclusion or "eligible attributed/associated event"
            ),
            model_versions=_versions(self.config),
            prompt_versions=self.prompts,
            created_at=_now(),
        )

    @staticmethod
    def _normalized_bounding_box(value: Any, *, prefix: str) -> BoundingBox | None:
        """Normalize unordered Gemini coordinates and reject unusable rectangles."""
        if not isinstance(value, list) or len(value) != 4:
            return None
        try:
            raw_x0, raw_y0, raw_x1, raw_y1 = (float(item) for item in value)
        except (TypeError, ValueError):
            logger.warning(
                "Ignoring non-numeric bounding box for %s: %r", prefix, value
            )
            return None
        if not all(math.isfinite(item) for item in (raw_x0, raw_y0, raw_x1, raw_y1)):
            logger.warning("Ignoring non-finite bounding box for %s: %r", prefix, value)
            return None
        x0, x1 = sorted((raw_x0, raw_x1))
        y0, y1 = sorted((raw_y0, raw_y1))
        if x0 == x1 or y0 == y1:
            logger.warning("Ignoring zero-area bounding box for %s: %r", prefix, value)
            return None
        if (x0, y0, x1, y1) != (raw_x0, raw_y0, raw_x1, raw_y1):
            logger.warning("Normalized reversed bounding box for %s: %r", prefix, value)
        return BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)

    async def _verify_visual_facts(
        self,
        draft: dict[str, Any],
        regions: list[SourceRegion],
        artifacts: PdfArtifacts,
        client: PdfAnalysisClient,
        prefix: str,
        *,
        modality: str,
    ) -> list[VerifiedVisualFact]:
        raw_candidates = draft.get("visual_facts")
        candidates: list[Any] = (
            raw_candidates if isinstance(raw_candidates, list) else []
        )
        described = False
        if modality != "text" and not candidates:
            candidates = await self._describe_visual_candidates(
                draft, regions, artifacts, client
            )
            described = True
        result = await self._validate_visual_candidates(
            candidates, regions, artifacts, client, prefix
        )
        if modality != "text" and not result and not described:
            fallback = await self._describe_visual_candidates(
                draft, regions, artifacts, client
            )
            result = await self._validate_visual_candidates(
                fallback, regions, artifacts, client, prefix
            )
        if modality != "text" and not result:
            page_fallback = await self._describe_visual_candidates(
                draft,
                regions,
                artifacts,
                client,
                use_page_artifact=True,
            )
            result = await self._validate_visual_candidates(
                page_fallback, regions, artifacts, client, prefix
            )
        if modality != "text" and not result:
            logger.warning(
                "Visual evidence unit %s has no verified visual summary; "
                "retaining only independently validated source text",
                prefix,
            )
        return result

    async def _describe_visual_candidates(
        self,
        draft: dict[str, Any],
        regions: list[SourceRegion],
        artifacts: PdfArtifacts,
        client: PdfAnalysisClient,
        *,
        use_page_artifact: bool = False,
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        context = str(draft.get("unit_description") or "source visual")
        for index, region in enumerate(regions):
            artifact = (
                region.page_artifact
                if use_page_artifact
                else region.crop_artifact or region.page_artifact
            )
            try:
                response = await client.describe_visual(
                    artifacts.root / artifact.relative_path,
                    context=context,
                )
            except Exception as error:  # noqa: BLE001 - optional model boundary
                logger.warning(
                    "Visual description failed for page %d: %s",
                    region.physical_page,
                    error,
                )
                continue
            raw_facts = response.get("facts")
            facts: list[Any] = raw_facts if isinstance(raw_facts, list) else []
            for fact in facts:
                if isinstance(fact, dict) and isinstance(fact.get("text"), str):
                    candidates.append(
                        {
                            "text": fact["text"],
                            "region_indexes": [index],
                            "artifact_kind": ("page" if use_page_artifact else "crop"),
                        }
                    )
        return candidates

    async def _validate_visual_candidates(
        self,
        candidates: list[Any],
        regions: list[SourceRegion],
        artifacts: PdfArtifacts,
        client: PdfAnalysisClient,
        prefix: str,
    ) -> list[VerifiedVisualFact]:
        result: list[VerifiedVisualFact] = []
        grouped: dict[
            str,
            tuple[ArtifactRef, list[tuple[int, str, list[str]]]],
        ] = {}
        for index, candidate in enumerate(candidates, start=1):
            if not isinstance(candidate, dict) or not isinstance(
                candidate.get("text"), str
            ):
                continue
            indexes = candidate.get("region_indexes", [0])
            region_ids = [
                regions[item].region_id
                for item in indexes
                if isinstance(item, int) and 0 <= item < len(regions)
            ]
            use_page_artifact = candidate.get("artifact_kind") == "page"
            visual_artifact = next(
                (
                    region.page_artifact
                    if use_page_artifact
                    else region.crop_artifact or region.page_artifact
                    for region in regions
                    if region.region_id in region_ids
                ),
                None,
            )
            if visual_artifact is None:
                continue
            group = grouped.setdefault(
                visual_artifact.relative_path,
                (visual_artifact, []),
            )
            group[1].append((index, candidate["text"], region_ids))
        for visual_artifact, items in grouped.values():
            for offset in range(0, len(items), _MAX_VISUAL_FACTS_PER_REQUEST):
                batch = items[offset : offset + _MAX_VISUAL_FACTS_PER_REQUEST]
                try:
                    response = await client.verify_visual(
                        artifacts.root / visual_artifact.relative_path,
                        [text for _, text, _ in batch],
                    )
                except Exception as error:  # noqa: BLE001 - optional model boundary
                    logger.warning(
                        "Visual fact verification failed for %s (%d candidates): %s",
                        prefix,
                        len(batch),
                        error,
                    )
                    continue
                raw_facts = response.get("facts")
                facts: list[Any] = raw_facts if isinstance(raw_facts, list) else []
                entailed_texts = {
                    fact["text"]
                    for fact in facts
                    if isinstance(fact, dict)
                    and isinstance(fact.get("text"), str)
                    and fact.get("entailed") is True
                }
                for index, text, region_ids in batch:
                    if text not in entailed_texts:
                        continue
                    result.append(
                        VerifiedVisualFact(
                            fact_id=f"{prefix}:vf{index:03d}",
                            text=text,
                            supporting_region_ids=region_ids,
                            verifier_verdict="entailed",
                            verifier_model=get_config().gemini.model,
                            verifier_prompt_version=self.prompts.visual_verification,
                        )
                    )
        return result

    def _source_modality(self, value: Any) -> str:
        aliases = {
            "multimodal": "mixed",
            "figure": "chart",
            "infographic": "diagram",
        }
        normalized = aliases.get(value, value)
        if normalized in {"text", "table", "chart", "map", "diagram", "mixed"}:
            return cast(str, normalized)
        return "text"

    async def _summarize(self, title: str, source_text: str) -> str:
        if not source_text:
            return title
        summarizer = self.summarizer or LunaSectionSummarizer(self.config)
        return await summarizer.summarize(title=title, scope_source_text=source_text)

    def _canonical(
        self,
        source_text: str | None,
        facts: list[VerifiedVisualFact],
        regions: list[SourceRegion],
    ) -> str:
        first = regions[0]
        labels = f"physical page {first.physical_page}"
        if first.printed_page is not None:
            labels += f" | printed page {first.printed_page}"
        parts = [f"[TARGET SOURCE EVIDENCE | {labels}]\n{source_text or ''}".strip()]
        for fact in facts:
            parts.append(
                f"[VERIFIED VISUAL FACT | {labels} | region {','.join(fact.supporting_region_ids)}]\n{fact.text}"
            )
        return "\n\n".join(parts)

    def _parent_id(
        self, sha: str, draft: dict[str, Any], sections: list[Any]
    ) -> str | None:
        parent = draft.get("parent_ordinal")
        if not isinstance(parent, int) or parent < 1 or parent > len(sections):
            return None
        return f"{sha}:s{parent:03d}"

    def _string_or_none(self, value: Any) -> str | None:
        return value.strip() if isinstance(value, str) and value.strip() else None

    def _exact_source_text(
        self, proposed_text: str | None, pages: Sequence[RenderedPage]
    ) -> str | None:
        """Return only source bytes/text extracted locally from the cited page(s).

        Gemini often reflows line breaks, ligatures, or Unicode hyphens even when it
        has read the right passage.  A whitespace-flexible match recovers the exact
        local substring.  When it cannot, the whole cited page is safer evidence
        than storing a model-generated transcription as source text.
        """
        page_texts = self._exact_source_texts_by_page(proposed_text, pages)
        if page_texts:
            return "\n".join(page_texts.values())
        return next(
            (page.extracted_text for page in pages if page.extracted_text), None
        )

    def _exact_source_texts_by_page(
        self, proposed_text: str | None, pages: Sequence[RenderedPage]
    ) -> dict[int, str]:
        """Map each cited page to the exact local substring matched on that page.

        Column-wrapped passages often span the rightmost column of page N and the
        leftmost column of page N+1. Match the proposed text greedily across those
        pages in reading order so one evidence unit keeps both fragments.
        """
        if not proposed_text:
            return {}
        for page in pages:
            if proposed_text in page.extracted_text:
                return {page.physical_page: proposed_text}
        terms = [term for term in re.split(r"\s+", proposed_text) if term]
        if not terms:
            return {}
        if len(pages) == 1:
            matched = self._flexible_text_match(terms, pages[0].extracted_text)
            return {pages[0].physical_page: matched} if matched else {}
        remaining = terms
        page_texts: dict[int, str] = {}
        for page in pages:
            if not remaining:
                break
            matched_terms: list[str] | None = None
            matched_text: str | None = None
            for end in range(len(remaining), 0, -1):
                candidate = remaining[:end]
                matched = self._flexible_text_match(candidate, page.extracted_text)
                if matched is not None:
                    matched_terms = candidate
                    matched_text = matched
                    break
            if matched_terms is None or matched_text is None:
                if page_texts:
                    break
                continue
            page_texts[page.physical_page] = matched_text
            remaining = remaining[len(matched_terms) :]
        if page_texts and not remaining:
            return page_texts
        # Fall back to a full flexible match on any single cited page.
        for page in pages:
            matched = self._flexible_text_match(terms, page.extracted_text)
            if matched is not None:
                return {page.physical_page: matched}
        return {}

    @staticmethod
    def _flexible_text_match(terms: Sequence[str], haystack: str) -> str | None:
        if not terms:
            return None
        pattern = r"\s+".join(re.escape(term) for term in terms)
        match = re.search(pattern, haystack)
        return match.group(0) if match is not None else None

    def _mongo_date_or_none(self, value: Any) -> datetime | None:
        if not isinstance(value, str):
            return None
        try:
            parsed = date.fromisoformat(value)
        except ValueError:
            return None
        return datetime.combine(parsed, datetime.min.time(), tzinfo=UTC)

    async def _embed(
        self,
        cards: list[SectionContextCard],
        evidence: list[EvidenceUnit],
        pdf: Path,
        *,
        document_title: str | None,
    ) -> None:
        rows: list[tuple[dict[str, Any], str]] = []
        for unit in evidence:
            if not unit.searchable:
                continue
            representations = [
                ("evidence_source_text", unit.source_text),
                (
                    "evidence_visual_facts",
                    "\n".join(fact.text for fact in unit.verified_visual_facts),
                ),
                ("evidence_contextualized", unit.retrieval_text),
            ]
            for kind, text in representations:
                if text:
                    rows.append(
                        (
                            self._embedding_base(
                                unit,
                                kind,
                                text,
                                pdf,
                                document_title=document_title,
                            ),
                            text,
                        )
                    )
        for card in cards:
            if card.retrieval_text:
                rows.append(
                    (
                        {
                            "document_id": card.document_id,
                            "document_url": file_document_uri(pdf),
                            "document_title": document_title,
                            "chunk_index": card.physical_page_start - 1,
                            "chunk_text": "",
                            "countries_iso3": [
                                country.iso3 for country in card.countries
                            ],
                            "owner_kind": "section",
                            "owner_id": card.section_id,
                            "evidence_id": None,
                            "section_id": card.section_id,
                            "representation_kind": "section_context",
                            "embedding_text": card.retrieval_text,
                            "event_ids": [item.event_id for item in card.events],
                            "enso_relationships": [
                                item.relationship for item in card.events
                            ],
                            "assertion_mode": None,
                            "searchable": True,
                        },
                        card.retrieval_text,
                    )
                )
        logger.info("%s: embedding %d representation(s)", pdf.name, len(rows))
        vectors = await self._embed_texts([text for _, text in rows]) if rows else []
        if rows:
            await PdfEmbeddingRecord.insert_many(
                [
                    PdfEmbeddingRecord(**data, embedding=vector)
                    for (data, _), vector in zip(rows, vectors, strict=True)
                ]
            )
        logger.info("%s: stored %d embedding record(s)", pdf.name, len(vectors))

    def _embedding_base(
        self,
        unit: EvidenceUnit,
        kind: str,
        text: str,
        pdf: Path,
        *,
        document_title: str | None,
    ) -> dict[str, Any]:
        return {
            "document_id": unit.id,
            "document_url": file_document_uri(pdf),
            "document_title": document_title,
            "chunk_index": unit.physical_pages[0] - 1,
            "chunk_text": unit.canonical_evidence_text,
            "countries_iso3": [country.iso3 for country in unit.countries],
            "owner_kind": "evidence",
            "owner_id": unit.evidence_id,
            "evidence_id": unit.evidence_id,
            "section_id": unit.section_id,
            "representation_kind": kind,
            "embedding_text": text,
            "event_ids": [event.event_id for event in unit.events],
            "enso_relationships": [event.relationship for event in unit.events],
            "assertion_mode": unit.assertion_mode,
            "searchable": unit.searchable,
        }

    async def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self.embed_texts is not None:
            return await self.embed_texts(texts)
        embeddings = build_embeddings(vector_store_config=get_config().vector_store)
        return await embeddings.aembed_documents(texts)
