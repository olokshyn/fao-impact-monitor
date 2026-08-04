"""LangGraph agent that generates multi-faceted search queries for a research topic."""

from __future__ import annotations

import logging
import re
from typing import Any, Literal

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from fao_impact_monitor.config import (
    AwsBedrockConfig,
    QueryGeneratorConfig,
    get_config,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a search-query generation agent.

Your job is to produce insightful search queries that, when run against a
knowledge base (vectorstore, Tellus, document search), retrieve documents
covering every important aspect of a research question — so a later step can
answer it from evidence.

Critical rules:
1. Do NOT answer the research question. Output only search queries.
2. Do NOT invent facts, figures, or conclusions.
3. The optional example answer exists ONLY to show how detailed a final answer
   should be. Never quote, paraphrase, or embed any content from the example
   into your queries.
4. Generate multi-faceted queries that study the research question from
   complementary angles, for example:
   - definitions, measurement methods, and indicators
   - quantitative results, time series, and baselines
   - drivers, causes, and contributing factors
   - geographic and temporal context
   - policy responses, interventions, and outcomes
   - closely related metrics or secondary effects
5. Each query must be a self-contained semantic search string suitable for
   retrieval systems. Prefer concrete topical phrases over yes/no questions
   or a restatement of the full research question.
6. Queries should be distinct: avoid near-duplicates that would retrieve the
   same documents.
7. Return between the requested minimum and maximum number of queries
   (inclusive). Prefer quality and coverage over padding.
"""

RESEARCH_SYSTEM_PROMPT = """\
You are a search-query generation agent for country-specific metric research.

Your job is to produce focused retrieval queries for a vector store and/or the
open web so a later researcher can answer a metric from evidence only.

Critical rules:
1. Do NOT answer the metric. Output only search queries with purpose and
   destination.
2. Do NOT invent facts, figures, or conclusions.
3. Metric.example / example answer exists ONLY to show answer depth. Never
   quote, paraphrase, or embed any of its content into queries.
4. Every query must mention or clearly imply the selected country.
5. Queries must target the metric or an explicitly listed evidence gap.
6. Prefer multi-faceted coverage over paraphrases of one query. Where useful
   cover: direct metric values, expected unit, definitions/methodology,
   country-specific findings, dates/reporting periods, and components needed
   to calculate or explain the metric.
7. Each query is a self-contained semantic search string.
8. Set destination to "vectorstore", "web", or "both" based on where the
   evidence is most likely to be found.
9. For follow-up queries, set target_gap_ids to the gap ids you are targeting.
10. Avoid repeating previously executed queries.
11. Prefer authoritative source types (FAO/UN, government, official statistics,
    institutional reports) when suggesting web-oriented queries.
"""


class SearchQuery(BaseModel):
    query: str = Field(description="Self-contained semantic search query string")
    angle: str = Field(
        description="Short label of the facet or angle this query covers",
    )


class SearchQueryList(BaseModel):
    queries: list[SearchQuery] = Field(default_factory=list)


class ResearchQuery(BaseModel):
    query: str = Field(description="Self-contained semantic search query string")
    purpose: str = Field(description="Why this query is needed")
    target_gap_ids: list[str] = Field(
        default_factory=list,
        description="Evidence gap ids this query targets (empty for initial)",
    )
    destination: Literal["vectorstore", "web", "both"] = Field(
        description="Where this query should be executed",
    )


class ResearchQueryList(BaseModel):
    queries: list[ResearchQuery] = Field(default_factory=list)


class EvidenceGapInput(BaseModel):
    """Minimal gap description passed into follow-up query generation."""

    gap_id: str
    description: str
    why_required: str
    preferred_source_type: str | None = None
    suggested_terms: list[str] = Field(default_factory=list)


class AgentState(BaseModel):
    research_question: str
    explanation: str
    example: str | None = None
    min_queries: int
    max_queries: int
    queries: list[str] = Field(default_factory=list)
    count_error: str | None = None
    retries_left: int = 0


class ResearchAgentState(BaseModel):
    research_question: str
    explanation: str
    unit: str
    country_name: str
    country_iso3: str
    example: str | None = None
    established_facts: list[str] = Field(default_factory=list)
    open_gaps: list[EvidenceGapInput] = Field(default_factory=list)
    executed_queries: list[str] = Field(default_factory=list)
    weak_terms: list[str] = Field(default_factory=list)
    preferred_destinations: list[Literal["vectorstore", "web", "both"]] = Field(
        default_factory=list
    )
    min_queries: int
    max_queries: int
    research_queries: list[ResearchQuery] = Field(default_factory=list)
    count_error: str | None = None
    retries_left: int = 0


def build_chat_model(
    query_generator_config: QueryGeneratorConfig | None = None,
    aws_bedrock_config: AwsBedrockConfig | None = None,
) -> BaseChatModel:
    config = get_config()
    query_generator_config = query_generator_config or config.query_generator
    aws_bedrock_config = aws_bedrock_config or config.aws_bedrock
    return init_chat_model(
        query_generator_config.llm_model,
        api_key=aws_bedrock_config.api_key.get_secret_value(),
        base_url=aws_bedrock_config.base_url,
        use_responses_api=True,
    )


def normalize_query(query: str) -> str:
    """Normalize a query for duplicate detection."""
    collapsed = re.sub(r"\s+", " ", query.strip().casefold())
    return collapsed


def _country_implied(query: str, country_name: str, country_iso3: str) -> bool:
    q = query.casefold()
    if country_iso3.casefold() in q:
        return True
    name = country_name.casefold()
    if name in q:
        return True
    # Allow short common forms: drop parenthetical / comma suffixes.
    short = re.split(r"[,(\[]", name, maxsplit=1)[0].strip()
    return bool(short and short in q)


def _query_too_broad(query: str) -> bool:
    tokens = [t for t in re.split(r"\s+", query.strip()) if t]
    return len(tokens) < 3


def filter_research_queries(
    queries: list[ResearchQuery],
    *,
    country_name: str,
    country_iso3: str,
    executed_queries: list[str],
    open_gap_ids: set[str],
    require_gaps: bool,
) -> list[ResearchQuery]:
    """Deterministically reject weak / duplicate research queries."""
    executed = {normalize_query(q) for q in executed_queries}
    seen: set[str] = set()
    kept: list[ResearchQuery] = []
    for item in queries:
        text = item.query.strip()
        if not text:
            continue
        key = normalize_query(text)
        if key in seen or key in executed:
            logger.info("Rejecting duplicate research query: %s", text)
            continue
        if not _country_implied(text, country_name, country_iso3):
            logger.info("Rejecting query without country: %s", text)
            continue
        if _query_too_broad(text):
            logger.info("Rejecting overly broad query: %s", text)
            continue
        if require_gaps:
            if not item.target_gap_ids:
                logger.info("Rejecting follow-up query without gap ids: %s", text)
                continue
            if open_gap_ids and not set(item.target_gap_ids) & open_gap_ids:
                logger.info("Rejecting query that does not target open gaps: %s", text)
                continue
        seen.add(key)
        kept.append(
            ResearchQuery(
                query=text,
                purpose=item.purpose.strip() or "coverage",
                target_gap_ids=list(item.target_gap_ids),
                destination=item.destination,
            )
        )
    return kept


def _user_prompt(
    *,
    research_question: str,
    explanation: str,
    example: str | None,
    min_queries: int,
    max_queries: int,
    count_error: str | None,
) -> str:
    parts = [
        f"Research question:\n{research_question}",
        f"Explanation (aspects to cover and required depth):\n{explanation}",
        (
            f"Generate between {min_queries} and {max_queries} distinct "
            "search queries (inclusive) that together cover the important "
            "angles of this research question."
        ),
        "Do not answer the research question. Return only search queries.",
    ]
    if example:
        parts.insert(
            2,
            (
                "Example answer (detail-level guidance ONLY — do not use its "
                f"content in queries):\n{example}"
            ),
        )
    if count_error:
        parts.insert(
            0,
            (
                "Your previous query list failed validation. Produce a new "
                f"list.\n{count_error}\n"
                f"Return between {min_queries} and {max_queries} queries "
                "(inclusive)."
            ),
        )
    return "\n\n".join(parts)


def _research_user_prompt(state: ResearchAgentState) -> str:
    parts = [
        f"Metric / research question:\n{state.research_question}",
        f"Analysis required:\n{state.explanation}",
        f"Expected unit / answer form:\n{state.unit}",
        f"Country: {state.country_name} ({state.country_iso3})",
        (
            f"Generate between {state.min_queries} and {state.max_queries} "
            "distinct research queries (inclusive)."
        ),
        (
            "Each query must include purpose, destination "
            "(vectorstore|web|both), and target_gap_ids when targeting gaps."
        ),
        "Do not answer the metric. Return only search queries.",
    ]
    if state.example:
        parts.insert(
            4,
            (
                "Example answer (detail-level / style guidance ONLY — never use "
                f"its content as evidence or in queries):\n{state.example}"
            ),
        )
    if state.established_facts:
        facts = "\n".join(f"- {fact}" for fact in state.established_facts)
        parts.append(f"Already established facts:\n{facts}")
    if state.open_gaps:
        gap_lines: list[str] = []
        for gap in state.open_gaps:
            terms = ", ".join(gap.suggested_terms) if gap.suggested_terms else "n/a"
            preferred = gap.preferred_source_type or "any"
            gap_lines.append(
                f"- {gap.gap_id}: {gap.description} "
                f"(why: {gap.why_required}; preferred_source={preferred}; "
                f"suggested_terms={terms})"
            )
        parts.append("Open evidence gaps to target:\n" + "\n".join(gap_lines))
    if state.executed_queries:
        executed = "\n".join(f"- {q}" for q in state.executed_queries)
        parts.append(f"Previously executed queries (do not repeat):\n{executed}")
    if state.weak_terms:
        parts.append(
            "Terms/approaches that produced weak results:\n"
            + "\n".join(f"- {t}" for t in state.weak_terms)
        )
    if state.preferred_destinations:
        parts.append(
            "Preferred destinations for this round: "
            + ", ".join(state.preferred_destinations)
        )
    if state.count_error:
        parts.insert(
            0,
            (
                "Your previous research query list failed validation. Produce a "
                f"new list.\n{state.count_error}\n"
                f"Return between {state.min_queries} and {state.max_queries} "
                "queries (inclusive)."
            ),
        )
    return "\n\n".join(parts)


def validate_query_count(
    queries: list[str],
    *,
    min_queries: int,
    max_queries: int,
) -> str | None:
    """Return an error message if count is out of bounds, else None."""
    count = len(queries)
    if count < min_queries:
        return (
            f"Got {count} quer{'y' if count == 1 else 'ies'}; "
            f"need at least {min_queries}."
        )
    if count > max_queries:
        return f"Got {count} queries; need at most {max_queries}."
    return None


def build_query_generator_agent(model: BaseChatModel) -> Any:
    """Compile a LangGraph that generates and validates search queries."""

    structured = model.with_structured_output(SearchQueryList)

    async def generate(state: AgentState) -> dict[str, Any]:
        prompt = _user_prompt(
            research_question=state.research_question,
            explanation=state.explanation,
            example=state.example,
            min_queries=state.min_queries,
            max_queries=state.max_queries,
            count_error=state.count_error,
        )
        result = await structured.ainvoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
        )
        if not isinstance(result, SearchQueryList):
            result = SearchQueryList.model_validate(result)
        queries = [item.query.strip() for item in result.queries if item.query.strip()]
        logger.info(
            "Query generator produced %s quer%s",
            len(queries),
            "y" if len(queries) == 1 else "ies",
        )
        return {"queries": queries, "count_error": None}

    async def validate(state: AgentState) -> dict[str, Any]:
        error = validate_query_count(
            state.queries,
            min_queries=state.min_queries,
            max_queries=state.max_queries,
        )
        if error is not None:
            logger.warning("Query generator count validation failed: %s", error)
            return {
                "count_error": error,
                "retries_left": state.retries_left - 1,
            }
        return {"count_error": None}

    def route_after_validate(
        state: AgentState,
    ) -> Literal["generate", "__end__"]:
        if state.count_error is not None and state.retries_left > 0:
            return "generate"
        return "__end__"

    graph = StateGraph(AgentState)
    graph.add_node("generate", generate)
    graph.add_node("validate", validate)
    graph.set_entry_point("generate")
    graph.add_edge("generate", "validate")
    graph.add_conditional_edges(
        "validate",
        route_after_validate,
        {"generate": "generate", "__end__": END},
    )
    return graph.compile()


def build_research_query_generator_agent(model: BaseChatModel) -> Any:
    """Compile a LangGraph that generates gap-aware research queries."""

    structured = model.with_structured_output(ResearchQueryList)

    async def generate(state: ResearchAgentState) -> dict[str, Any]:
        result = await structured.ainvoke(
            [
                SystemMessage(content=RESEARCH_SYSTEM_PROMPT),
                HumanMessage(content=_research_user_prompt(state)),
            ]
        )
        if not isinstance(result, ResearchQueryList):
            result = ResearchQueryList.model_validate(result)
        open_gap_ids = {g.gap_id for g in state.open_gaps}
        filtered = filter_research_queries(
            result.queries,
            country_name=state.country_name,
            country_iso3=state.country_iso3,
            executed_queries=state.executed_queries,
            open_gap_ids=open_gap_ids,
            require_gaps=bool(state.open_gaps),
        )
        logger.info(
            "Research query generator produced %s quer%s (%s after filters)",
            len(result.queries),
            "y" if len(result.queries) == 1 else "ies",
            len(filtered),
        )
        return {"research_queries": filtered, "count_error": None}

    async def validate(state: ResearchAgentState) -> dict[str, Any]:
        error = validate_query_count(
            [q.query for q in state.research_queries],
            min_queries=state.min_queries,
            max_queries=state.max_queries,
        )
        if error is not None:
            logger.warning("Research query count validation failed: %s", error)
            return {
                "count_error": error,
                "retries_left": state.retries_left - 1,
            }
        return {"count_error": None}

    def route_after_validate(
        state: ResearchAgentState,
    ) -> Literal["generate", "__end__"]:
        if state.count_error is not None and state.retries_left > 0:
            return "generate"
        return "__end__"

    graph = StateGraph(ResearchAgentState)
    graph.add_node("generate", generate)
    graph.add_node("validate", validate)
    graph.set_entry_point("generate")
    graph.add_edge("generate", "validate")
    graph.add_conditional_edges(
        "validate",
        route_after_validate,
        {"generate": "generate", "__end__": END},
    )
    return graph.compile()


async def generate_queries(
    *,
    research_question: str,
    explanation: str,
    example: str | None = None,
    config: QueryGeneratorConfig | None = None,
    model: BaseChatModel | None = None,
) -> list[str]:
    """Generate search queries within configured min/max bounds."""
    cfg = config or get_config().query_generator
    if cfg.min_queries > cfg.max_queries:
        raise ValueError(
            f"min_queries ({cfg.min_queries}) cannot exceed "
            f"max_queries ({cfg.max_queries})"
        )

    chat_model = model or build_chat_model(query_generator_config=cfg)
    agent = build_query_generator_agent(chat_model)
    final_state = AgentState.model_validate(
        await agent.ainvoke(
            {
                "research_question": research_question,
                "explanation": explanation,
                "example": example,
                "min_queries": cfg.min_queries,
                "max_queries": cfg.max_queries,
                "queries": [],
                "count_error": None,
                "retries_left": cfg.max_agent_retries,
            }
        )
    )
    queries = list(final_state.queries)
    error = validate_query_count(
        queries,
        min_queries=cfg.min_queries,
        max_queries=cfg.max_queries,
    )
    if error is None:
        return queries
    if len(queries) > cfg.max_queries:
        logger.warning(
            "Truncating %s queries to max_queries=%s after retries exhausted",
            len(queries),
            cfg.max_queries,
        )
        return queries[: cfg.max_queries]
    raise ValueError(f"Query generator produced too few queries after retries: {error}")


async def generate_research_queries(
    *,
    research_question: str,
    explanation: str,
    unit: str,
    country_name: str,
    country_iso3: str,
    example: str | None = None,
    established_facts: list[str] | None = None,
    open_gaps: list[EvidenceGapInput] | None = None,
    executed_queries: list[str] | None = None,
    weak_terms: list[str] | None = None,
    preferred_destinations: list[Literal["vectorstore", "web", "both"]] | None = None,
    min_queries: int | None = None,
    max_queries: int | None = None,
    config: QueryGeneratorConfig | None = None,
    model: BaseChatModel | None = None,
) -> list[ResearchQuery]:
    """Generate gap-aware research queries for the ResearcherAgent loop."""
    cfg = config or get_config().query_generator
    min_q = min_queries if min_queries is not None else cfg.min_queries
    max_q = max_queries if max_queries is not None else cfg.max_queries
    if min_q > max_q:
        raise ValueError(f"min_queries ({min_q}) cannot exceed max_queries ({max_q})")

    chat_model = model or build_chat_model(query_generator_config=cfg)
    agent = build_research_query_generator_agent(chat_model)
    final_state = ResearchAgentState.model_validate(
        await agent.ainvoke(
            {
                "research_question": research_question,
                "explanation": explanation,
                "unit": unit,
                "country_name": country_name,
                "country_iso3": country_iso3,
                "example": example,
                "established_facts": established_facts or [],
                "open_gaps": open_gaps or [],
                "executed_queries": executed_queries or [],
                "weak_terms": weak_terms or [],
                "preferred_destinations": preferred_destinations or [],
                "min_queries": min_q,
                "max_queries": max_q,
                "research_queries": [],
                "count_error": None,
                "retries_left": cfg.max_agent_retries,
            }
        )
    )
    queries = list(final_state.research_queries)
    error = validate_query_count(
        [q.query for q in queries],
        min_queries=min_q,
        max_queries=max_q,
    )
    if error is None:
        return queries
    if len(queries) > max_q:
        logger.warning(
            "Truncating %s research queries to max_queries=%s after retries",
            len(queries),
            max_q,
        )
        return queries[:max_q]
    # Follow-up rounds may legitimately yield fewer after filters; return what
    # remains rather than failing the whole research loop.
    if open_gaps and queries:
        logger.warning(
            "Research query generator returned %s quer%s after filters "
            "(wanted at least %s); continuing with filtered set",
            len(queries),
            "y" if len(queries) == 1 else "ies",
            min_q,
        )
        return queries
    raise ValueError(
        f"Research query generator produced too few queries after retries: {error}"
    )
