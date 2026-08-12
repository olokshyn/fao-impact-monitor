"""Thin async wrapper over web-scout-ai open-web research."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_SCHEMA_META_KEYS = frozenset(
    {
        "additionalProperties",
        "default",
        "description",
        "enum",
        "items",
        "properties",
        "required",
        "title",
        "type",
    }
)
_SCHEMA_ENVELOPE_PATCHED = False


class WebScoutProviderError(RuntimeError):
    """Raised when web-scout-ai research fails unexpectedly."""


class WebSource(BaseModel):
    """A successfully scraped web source usable as evidence."""

    source_id: str
    url: str
    title: str | None = None
    content: str
    query: str
    publisher: str | None = None
    publication_date: str | None = None
    access_date: str
    page_number: int | None = None
    section: str | None = None


class WebQueryStat(BaseModel):
    """One web-scout search query and how many hits the backend returned."""

    query: str
    results_returned: int = 0


class WebScoutResearchResult(BaseModel):
    """Mapped web-scout result containing only inspectable scraped sources."""

    sources: list[WebSource] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)
    query_stats: list[WebQueryStat] = Field(default_factory=list)
    snippet_only_count: int = 0
    failed_count: int = 0


WebResearchFn = Callable[..., Awaitable[Any]]


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def _source_id(url: str, content: str) -> str:
    # Keep IDs short so claim-extraction LLMs can copy them reliably.
    return f"web:{_content_hash(url + '\0' + content)}"


def _looks_like_json_schema_field(value: Any) -> bool:
    """True when *value* looks like a JSON Schema property definition."""
    if not isinstance(value, dict) or "type" not in value:
        return False
    return bool(set(value) & (_SCHEMA_META_KEYS - {"type"})) or value.get("type") in {
        "object",
        "array",
        "string",
        "boolean",
        "number",
        "integer",
        "null",
    }


def unwrap_schema_shaped_instance(data: Any) -> dict[str, Any] | None:
    """Unwrap instance values nested under a JSON Schema envelope.

    Some Gemini structured-output responses echo the schema and put the real
    field values under ``properties``. That fails Pydantic validation for
    web-scout models such as ``CoverageEvaluation``.
    """
    if not isinstance(data, dict):
        return None
    props = data.get("properties")
    if data.get("type") != "object" or not isinstance(props, dict) or not props:
        return None
    if not any(
        key in data
        for key in ("title", "required", "description", "additionalProperties")
    ):
        return None
    if any(_looks_like_json_schema_field(value) for value in props.values()):
        return None
    return props


def _patch_agents_schema_envelope_validation() -> None:
    """Make Agents SDK structured-output parsing tolerate schema envelopes.

    Applied in-process so we do not need a web-scout-ai upgrade. Idempotent.
    """
    global _SCHEMA_ENVELOPE_PATCHED
    if _SCHEMA_ENVELOPE_PATCHED:
        return

    from agents.agent_output import AgentOutputSchema
    from agents.exceptions import ModelBehaviorError

    original_validate_json = AgentOutputSchema.validate_json

    def validate_json(self: Any, json_str: str) -> Any:
        try:
            return original_validate_json(self, json_str)
        except ModelBehaviorError as exc:
            try:
                parsed = json.loads(json_str)
            except json.JSONDecodeError:
                raise exc from None
            unwrapped = unwrap_schema_shaped_instance(parsed)
            if unwrapped is None:
                raise
            logger.warning(
                "Recovered structured output nested under a JSON Schema "
                "envelope for %s",
                self.name(),
            )
            return original_validate_json(self, json.dumps(unwrapped))

    AgentOutputSchema.validate_json = validate_json  # type: ignore[method-assign]
    _SCHEMA_ENVELOPE_PATCHED = True
    logger.debug("Patched AgentOutputSchema.validate_json for schema envelopes")


def map_web_research_result(
    result: Any,
    *,
    query: str,
    access_date: str | None = None,
) -> WebScoutResearchResult:
    """Map a ``WebResearchResult`` to evidence-safe ``WebSource`` rows.

    Only ``scraped`` entries with non-empty content are accepted. Snippet-only
    and failed/blocked buckets are never treated as evidence. The WebScout
    ``synthesis`` field is ignored as an authoritative source.
    """
    access = access_date or datetime.now(tz=UTC).date().isoformat()
    scraped = getattr(result, "scraped", None) or []
    snippet_only = getattr(result, "snippet_only", None) or []
    failed_buckets = [
        getattr(result, "scrape_failed", None) or [],
        getattr(result, "blocked_by_policy", None) or [],
        getattr(result, "source_http_error", None) or [],
        getattr(result, "bot_detected", None) or [],
        getattr(result, "scraped_irrelevant", None) or [],
    ]
    sources: list[WebSource] = []
    seen: set[str] = set()
    for entry in scraped:
        url = str(getattr(entry, "url", "") or "").strip()
        content = str(getattr(entry, "content", "") or "").strip()
        if not url or not content:
            continue
        sid = _source_id(url, content)
        if sid in seen:
            continue
        seen.add(sid)
        title = getattr(entry, "title", None)
        sources.append(
            WebSource(
                source_id=sid,
                url=url,
                title=str(title).strip() if title else None,
                content=content,
                query=query,
                access_date=access,
            )
        )

    query_stats: list[WebQueryStat] = []
    seen_queries: set[str] = set()
    for item in getattr(result, "queries", None) or []:
        q = getattr(item, "query", None)
        if not isinstance(q, str) or not q.strip():
            continue
        text = q.strip()
        if text in seen_queries:
            continue
        seen_queries.add(text)
        raw_returned = getattr(item, "num_results_returned", None)
        returned = int(raw_returned) if isinstance(raw_returned, int) else 0
        query_stats.append(WebQueryStat(query=text, results_returned=returned))
    if not query_stats:
        query_stats = [WebQueryStat(query=query, results_returned=len(sources))]

    return WebScoutResearchResult(
        sources=sources,
        queries=[item.query for item in query_stats],
        query_stats=query_stats,
        snippet_only_count=len(snippet_only),
        failed_count=sum(len(bucket) for bucket in failed_buckets),
    )


async def run_web_scout_research(
    query: str,
    *,
    include_domains: list[str] | None = None,
    domain_expertise: str | None = None,
    research_depth: str | dict[str, Any] = "standard",
    web_research_fn: WebResearchFn | None = None,
) -> WebScoutResearchResult:
    """Run web-scout-ai with native models and return scraped sources only."""
    if web_research_fn is None:
        try:
            from web_scout import run_web_research as default_fn
        except ImportError as exc:  # pragma: no cover - dependency missing
            raise WebScoutProviderError("web-scout-ai is not installed") from exc
        web_research_fn = default_fn
        # Gemini sometimes returns CoverageEvaluation values nested under a
        # JSON Schema envelope; unwrap before Agents SDK validation fails.
        _patch_agents_schema_envelope_validation()

    logger.info("WebScout research: query=%r depth=%s", query, research_depth)
    try:
        result = await web_research_fn(
            query,
            include_domains=include_domains,
            domain_expertise=domain_expertise,
            research_depth=research_depth,
        )
    except Exception as exc:
        raise WebScoutProviderError(
            f"web-scout-ai failed for query={query!r}: {exc}"
        ) from exc

    mapped = map_web_research_result(result, query=query)
    logger.info(
        "WebScout research: %s scraped source(s), %s snippet_only, %s failed "
        "for query=%r",
        len(mapped.sources),
        mapped.snippet_only_count,
        mapped.failed_count,
        query,
    )
    return mapped
