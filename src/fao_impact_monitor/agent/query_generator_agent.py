"""LangGraph agent that generates multi-faceted search queries for a research topic."""

from __future__ import annotations

import logging
from typing import Any, Literal, TypedDict

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


class SearchQuery(BaseModel):
    query: str = Field(description="Self-contained semantic search query string")
    angle: str = Field(
        description="Short label of the facet or angle this query covers",
    )


class SearchQueryList(BaseModel):
    queries: list[SearchQuery] = Field(default_factory=list)


class AgentState(TypedDict):
    research_question: str
    explanation: str
    example: str | None
    min_queries: int
    max_queries: int
    queries: list[str]
    count_error: str | None
    retries_left: int


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
            research_question=state["research_question"],
            explanation=state["explanation"],
            example=state["example"],
            min_queries=state["min_queries"],
            max_queries=state["max_queries"],
            count_error=state["count_error"],
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
            state["queries"],
            min_queries=state["min_queries"],
            max_queries=state["max_queries"],
        )
        if error is not None:
            logger.warning("Query generator count validation failed: %s", error)
            return {
                "count_error": error,
                "retries_left": state["retries_left"] - 1,
            }
        return {"count_error": None}

    def route_after_validate(
        state: AgentState,
    ) -> Literal["generate", "__end__"]:
        if state["count_error"] is not None and state["retries_left"] > 0:
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
    final_state: AgentState = await agent.ainvoke(
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
    queries = list(final_state.get("queries", []))
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
