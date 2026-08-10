"""Gemini 3.6 Flash calls used for visual PDF interpretation and verification."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from google import genai
from google.genai import errors, types

from fao_impact_monitor.config import GeminiConfig, get_config

logger = logging.getLogger(__name__)

_EVENT_ID_SCHEMA: dict[str, Any] = {
    "type": "string",
    "enum": [
        "el_nino_1997_98",
        "el_nino_2015_16",
        "el_nino_2018_19",
        "el_nino_2023_24",
        "el_nino_2026_27",
    ],
}
_RELATIONSHIP_SCHEMA: dict[str, Any] = {
    "type": "string",
    "enum": [
        "attributed",
        "associated",
        "explicitly_not_attributed",
        "uncertain",
        "unrelated",
    ],
}
_ASSERTION_MODE_SCHEMA: dict[str, Any] = {
    "type": "string",
    "enum": [
        "contextual",
        "observed",
        "reported",
        "estimated",
        "forecast",
        "scenario",
        "measured_response",
    ],
}
_DOCUMENT_EVENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "event_id": _EVENT_ID_SCHEMA,
        "relationship": _RELATIONSHIP_SCHEMA,
        "support_pages": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["event_id", "relationship", "support_pages"],
}
_SECTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "ordinal": {"type": "integer"},
        "parent_ordinal": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        "level": {"type": "integer"},
        "title": {"type": "string"},
        "page_start": {"type": "integer"},
        "page_end": {"type": "integer"},
        "printed_page_start": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "printed_page_end": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "countries": {"type": "array", "items": {"type": "string"}},
        "regions": {"type": "array", "items": {"type": "string"}},
        "events": {"type": "array", "items": _DOCUMENT_EVENT_SCHEMA},
        "reporting_modes": {
            "type": "array",
            "items": _ASSERTION_MODE_SCHEMA,
        },
        "scope_pages": {"type": "array", "items": {"type": "integer"}},
    },
    "required": [
        "ordinal",
        "parent_ordinal",
        "level",
        "title",
        "page_start",
        "page_end",
        "countries",
        "regions",
        "events",
        "reporting_modes",
        "scope_pages",
    ],
}
_DOCUMENT_STRUCTURE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "publication_date": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "reporting_period": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "purpose": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "events": {"type": "array", "items": _DOCUMENT_EVENT_SCHEMA},
        "sections": {
            "type": "array",
            "items": _SECTION_SCHEMA,
            "minItems": 1,
        },
    },
    "required": ["title", "sections"],
}
_UNIT_EVENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "event_id": _EVENT_ID_SCHEMA,
        "relationship": _RELATIONSHIP_SCHEMA,
    },
    "required": ["event_id", "relationship"],
}
_REGION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "page": {"type": "integer"},
        "printed_page": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "bbox": {
            "anyOf": [
                {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4,
                },
                {"type": "null"},
            ]
        },
    },
    "required": ["page", "printed_page", "bbox"],
}
_VISUAL_FACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "region_indexes": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["text", "region_indexes"],
}
_EVIDENCE_UNIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "kind": {
            "type": "string",
            "enum": [
                "scope_context",
                "hazard",
                "agrifood_impact",
                "livelihood_impact",
                "response_outcome",
            ],
        },
        "assertion_mode": _ASSERTION_MODE_SCHEMA,
        "modality": {
            "type": "string",
            "enum": ["text", "table", "chart", "map", "diagram", "mixed"],
        },
        "pages": {"type": "array", "items": {"type": "integer"}},
        "regions": {"type": "array", "items": _REGION_SCHEMA},
        "source_text": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "unit_description": {"type": "string"},
        "retrieval_text": {"type": "string"},
        "countries": {"type": "array", "items": {"type": "string"}},
        "events": {"type": "array", "items": _UNIT_EVENT_SCHEMA},
        "visual_facts": {"type": "array", "items": _VISUAL_FACT_SCHEMA},
        "continuation_pages": {
            "type": "array",
            "items": {"type": "integer"},
        },
    },
    "required": [
        "kind",
        "assertion_mode",
        "modality",
        "pages",
        "regions",
        "source_text",
        "unit_description",
        "retrieval_text",
        "countries",
        "events",
        "visual_facts",
        "continuation_pages",
    ],
}
_SECTION_EVIDENCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "corrected_page_start": {"type": "integer"},
        "corrected_page_end": {"type": "integer"},
        "boundary_rationale": {"type": "string"},
        "scope_units": {"type": "array", "items": _EVIDENCE_UNIT_SCHEMA},
        "evidence_units": {"type": "array", "items": _EVIDENCE_UNIT_SCHEMA},
    },
    "required": [
        "corrected_page_start",
        "corrected_page_end",
        "boundary_rationale",
        "scope_units",
        "evidence_units",
    ],
}
_VISUAL_DESCRIPTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        }
    },
    "required": ["facts"],
}
_VISUAL_VERIFICATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "entailed": {"type": "boolean"},
                },
                "required": ["text", "entailed"],
            },
        }
    },
    "required": ["facts"],
}
_MAX_JSON_ATTEMPTS = 3

DOCUMENT_STRUCTURE_PROMPT = """Analyze this FAO PDF as a source-grounded evidence analyst.
Return JSON only. Identify document metadata and a complete section hierarchy. Every
section must give 1-based physical page_start/page_end, printed_page_start/printed_page_end if
visible, title, parent ordinal (or null), level, countries (ISO3 if known), regions,
events, relationship for each event, reporting modes, and page numbers which establish
scope.

EVENT EXTRACTION IS REQUIRED, NOT OPTIONAL:
- If a section explicitly says El Nino/El Niño, its events array must not be empty when
  the episode can be resolved from the cited pages or document context.
- Map 1997 or 1997-98 to el_nino_1997_98; 2015 or 2015-16 to
  el_nino_2015_16; 2018 or 2018-19 to el_nino_2018_19; 2023 or 2023-24 to
  el_nino_2023_24; and 2026 or 2026-27 to el_nino_2026_27.
- "Attributed" means the source causally links the hazard or impact to that event.
  "Associated" means the source discusses it in the event's context, including a
  multi-driver statement such as El Niño plus the Indian Ocean Dipole.
- Use explicitly_not_attributed for an explicit denial, uncertain for genuinely
  conflicting or unresolved attribution, and unrelated only when the source establishes
  that the content is unrelated. Do not infer causality from publication year alone.
- Every event needs support_pages containing the physical pages that establish both the
  episode and relationship. Recheck every section with an empty events array before
  returning it.

Eligible event IDs are el_nino_1997_98, el_nino_2015_16, el_nino_2018_19,
el_nino_2023_24, el_nino_2026_27. Use only these exact event IDs.
Schema: {title, publication_date, reporting_period, purpose, sections:[{ordinal,parent_ordinal,
level,title,page_start,page_end,countries:[ISO3],regions:[string],events:[{event_id,relationship,
support_pages:[int]}],reporting_modes:[string],scope_pages:[int]}]}"""

SECTION_EVIDENCE_PROMPT = """Analyze the target pages of this FAO PDF excerpt as one named
section. The request supplies a mapping from excerpt page numbers to original 1-based
physical PDF pages. ALWAYS use original physical page numbers in corrected_page_start,
corrected_page_end, unit pages, regions, and continuation_pages.

The excerpt may contain one preceding and one following context page. Inspect those pages
to validate the section boundaries. If a heading, continuation, or next-section heading
shows that the candidate boundary is wrong, correct it by at most the supplied adjacent
page. Otherwise return the candidate boundaries unchanged. Explain the decision briefly
in boundary_rationale. Do not extract unrelated evidence from an adjacent context page.

Return JSON only. First return scope_units that contain direct text establishing country,
El Nino event/relationship, explicit negation, and reporting mode. Then return semantic
evidence_units. An evidence unit is a self-contained connected passage, or a figure/table/
map/diagram plus caption, legend and nearby interpretation. Preserve page ranges and source
regions. For each region give a bounding box in top-left PDF points when visual, otherwise
null, and printed_page if visible. Never put a paraphrase into source_text: source_text must be exact PDF text only.
For every table, chart, map, diagram, or mixed visual unit, modality MUST be one of table,
chart, map, diagram, or mixed and visual_facts MUST contain at least one atomic textual
summary of what is visibly shown. Include titles, axes, units, dates, geography, legends,
categories, trends, and values when legible. Do not use "multimodal" as a modality. For
visual claims give visual_facts separately. Label assertion_mode accurately (forecast vs
observed etc), event context and countries as direct facts only. Do not invent values.

EVENT EXTRACTION IS REQUIRED, NOT OPTIONAL:
- A scope or evidence unit whose source_text explicitly mentions El Nino/El Niño must
  have a non-empty events array whenever the episode is stated on the section pages or
  established by the document context.
- Map 1997 or 1997-98 to el_nino_1997_98; 2015 or 2015-16 to
  el_nino_2015_16; 2018 or 2018-19 to el_nino_2018_19; 2023 or 2023-24 to
  el_nino_2023_24; and 2026 or 2026-27 to el_nino_2026_27. Use only these IDs.
- Use attributed for an explicit causal link. Use associated when the unit is framed in
  the event context or names multiple contributing climate drivers. Preserve explicit
  non-attribution as explicitly_not_attributed; do not soften it.
- Put the event on the scope unit that establishes it. Do not repeat inherited section
  events as direct facts on child units that do not themselves mention the event; the
  pipeline will inherit those events from the cited scope unit.
- Before returning JSON, recheck every scope_unit containing the phrase El Nino/El Niño
  and correct any empty events array.

Schema: {corrected_page_start:int,corrected_page_end:int,boundary_rationale:string,
scope_units:[unit], evidence_units:[unit]}; unit={kind,assertion_mode,modality,
pages:[int],regions:[{page,printed_page,bbox|null}],source_text|null,unit_description,retrieval_text,
countries:[ISO3],events:[{event_id,relationship}],visual_facts:[{text,region_indexes:[int]}],
continuation_pages:[int]}."""

VISUAL_VERIFY_PROMPT = """Verify proposed factual statements against ONLY this visual crop/page.
Return JSON: {facts:[{text,entailed:boolean}]}. A fact is entailed only when the chart,
table, map, diagram, legend or visible caption supports it exactly. Reject guessed values.
Do not use outside knowledge."""

VISUAL_DESCRIBE_PROMPT = """Describe ONLY the attached source visual as atomic factual
statements. Return JSON: {facts:[{text:string}]}. Summarize what the image visibly shows,
including its title or subject, axes, units, dates, geography, legend, categories, trends,
comparisons, and numerical values when legible. Preserve observed, estimated, and forecast
qualifiers. Do not infer causality or use outside knowledge. Return at least one fact for a
readable chart, table, map, diagram, or mixed visual. Context is supplied only to identify
the intended visual; it is not evidence."""


class GeminiPdfClient:
    def __init__(self, config: GeminiConfig | None = None) -> None:
        self.config = config or get_config().gemini
        api_key = self.config.api_key.get_secret_value()
        if not api_key:
            raise ValueError("Set GEMINI_API_KEY before running pdf-pipeline ingest")
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=self.config.request_timeout_seconds * 1_000
            ),
        )

    async def document_structure(self, pdf_path: Path) -> dict[str, Any]:
        return await self._json(
            [self._pdf_part(pdf_path), DOCUMENT_STRUCTURE_PROMPT],
            response_json_schema=_DOCUMENT_STRUCTURE_SCHEMA,
        )

    async def section_evidence(
        self,
        pdf_path: Path,
        *,
        section_title: str,
        pages: Sequence[int],
        included_pages: Sequence[int],
    ) -> dict[str, Any]:
        page_mapping = ", ".join(
            f"excerpt page {local_page} = original physical page {physical_page}"
            for local_page, physical_page in enumerate(included_pages, start=1)
        )
        context_pages = [page for page in included_pages if page not in pages]
        request = (
            f"Section title: {section_title}. Candidate original physical pages: "
            f"{list(pages)}. Adjacent context pages: {context_pages}.\n"
            f"Page mapping: {page_mapping}.\n"
            f"{SECTION_EVIDENCE_PROMPT}"
        )
        return await self._json(
            [self._pdf_part(pdf_path), request],
            response_json_schema=_SECTION_EVIDENCE_SCHEMA,
        )

    async def verify_visual(
        self, image_path: Path, facts: Sequence[str]
    ) -> dict[str, Any]:
        image = types.Part.from_bytes(
            data=image_path.read_bytes(), mime_type="image/png"
        )
        return await self._json(
            [
                image,
                (
                    f"{VISUAL_VERIFY_PROMPT}\nFacts: "
                    f"{json.dumps(list(facts), ensure_ascii=False)}"
                ),
            ],
            response_json_schema=_VISUAL_VERIFICATION_SCHEMA,
            list_key="facts",
        )

    async def describe_visual(
        self, image_path: Path, *, context: str
    ) -> dict[str, Any]:
        image = types.Part.from_bytes(
            data=image_path.read_bytes(), mime_type="image/png"
        )
        return await self._json(
            [image, f"{VISUAL_DESCRIBE_PROMPT}\nContext: {context}"],
            response_json_schema=_VISUAL_DESCRIPTION_SCHEMA,
            list_key="facts",
        )

    def _pdf_part(self, pdf_path: Path) -> types.Part:
        return types.Part.from_bytes(
            data=pdf_path.read_bytes(), mime_type="application/pdf"
        )

    async def _json(
        self,
        contents: list[Any],
        *,
        response_json_schema: dict[str, Any],
        list_key: str | None = None,
    ) -> dict[str, Any]:
        def request() -> Any:
            response = self.client.models.generate_content(
                model=self.config.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema=response_json_schema,
                ),
            )
            return response

        last_error: Exception | None = None
        for attempt in range(1, _MAX_JSON_ATTEMPTS + 1):
            try:
                response = await asyncio.to_thread(request)
                parsed = getattr(response, "parsed", None)
                value = (
                    parsed
                    if isinstance(parsed, (dict, list))
                    else self._parse_json(response.text or "")
                )
                if isinstance(value, dict):
                    return value
                if list_key is not None and isinstance(value, list):
                    logger.warning(
                        "Gemini returned a top-level list; normalizing it as %r",
                        list_key,
                    )
                    return {list_key: value}
                raise TypeError(
                    f"Gemini response was {type(value).__name__}, not a JSON object"
                )
            except (errors.APIError, json.JSONDecodeError, OSError, TypeError) as error:
                last_error = error
                if attempt == _MAX_JSON_ATTEMPTS:
                    break
                logger.warning(
                    "Gemini structured response attempt %d/%d failed: %s; retrying",
                    attempt,
                    _MAX_JSON_ATTEMPTS,
                    error,
                )
                await asyncio.sleep(2 ** (attempt - 1))
        assert last_error is not None
        raise RuntimeError(
            f"Gemini did not return the required JSON object after "
            f"{_MAX_JSON_ATTEMPTS} attempts"
        ) from last_error

    @staticmethod
    def _parse_json(raw: str) -> Any:
        text = raw.strip()
        if text.startswith("```") and text.endswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1]).strip()
        return json.loads(text)
