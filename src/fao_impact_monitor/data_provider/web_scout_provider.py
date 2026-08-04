"""Thin async wrapper over web-scout-ai open-web research."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


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


class WebScoutResearchResult(BaseModel):
    """Mapped web-scout result containing only inspectable scraped sources."""

    sources: list[WebSource] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)
    snippet_only_count: int = 0
    failed_count: int = 0


WebResearchFn = Callable[..., Awaitable[Any]]


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def _source_id(url: str, content: str) -> str:
    return f"web:{url}:{_content_hash(content)}"


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

    query_list: list[str] = []
    for item in getattr(result, "queries", None) or []:
        q = getattr(item, "query", None)
        if isinstance(q, str) and q.strip():
            query_list.append(q.strip())
    if not query_list:
        query_list = [query]

    return WebScoutResearchResult(
        sources=sources,
        queries=query_list,
        snippet_only_count=len(snippet_only),
        failed_count=sum(len(bucket) for bucket in failed_buckets),
    )


async def run_web_scout_research(
    query: str,
    *,
    include_domains: list[str] | None = None,
    domain_expertise: str | None = None,
    research_depth: str = "standard",
    web_research_fn: WebResearchFn | None = None,
) -> WebScoutResearchResult:
    """Run web-scout-ai with native models and return scraped sources only."""
    if web_research_fn is None:
        try:
            from web_scout import run_web_research as default_fn
        except ImportError as exc:  # pragma: no cover - dependency missing
            raise WebScoutProviderError("web-scout-ai is not installed") from exc
        web_research_fn = default_fn

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
