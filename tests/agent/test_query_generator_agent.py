"""Unit tests for the query generator agent."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from fao_impact_monitor.agent.query_generator_agent import (
    SearchQuery,
    SearchQueryList,
    generate_queries,
    validate_query_count,
)
from fao_impact_monitor.config import QueryGeneratorConfig


def _config(
    *,
    min_queries: int = 3,
    max_queries: int = 5,
    max_agent_retries: int = 3,
) -> QueryGeneratorConfig:
    return QueryGeneratorConfig(
        min_queries=min_queries,
        max_queries=max_queries,
        max_agent_retries=max_agent_retries,
    )


def _queries(*texts: str) -> SearchQueryList:
    return SearchQueryList(
        queries=[SearchQuery(query=q, angle=f"angle-{i}") for i, q in enumerate(texts)]
    )


def test_validate_query_count_in_bounds() -> None:
    assert validate_query_count(["a", "b", "c"], min_queries=3, max_queries=5) is None
    assert (
        validate_query_count(["a", "b", "c", "d", "e"], min_queries=3, max_queries=5)
        is None
    )


def test_validate_query_count_out_of_bounds() -> None:
    too_few = validate_query_count(["a", "b"], min_queries=3, max_queries=5)
    assert too_few is not None
    assert "at least 3" in too_few

    too_many = validate_query_count(
        ["a", "b", "c", "d", "e", "f"],
        min_queries=3,
        max_queries=5,
    )
    assert too_many is not None
    assert "at most 5" in too_many


def test_generate_queries_returns_valid_list() -> None:
    class FakeStructured:
        async def ainvoke(self, _messages: Any) -> SearchQueryList:
            return _queries("q1", "q2", "q3")

    class FakeModel:
        def with_structured_output(self, _schema: Any) -> FakeStructured:
            return FakeStructured()

    result = asyncio.run(
        generate_queries(
            research_question="Water stress",
            explanation="Availability of water resources.",
            example="Water stress was 40%.",
            config=_config(),
            model=FakeModel(),  # type: ignore[arg-type]
        )
    )
    assert result == ["q1", "q2", "q3"]


def test_validation_reprompts_when_too_few() -> None:
    class FakeStructured:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke(self, _messages: Any) -> SearchQueryList:
            self.calls += 1
            if self.calls == 1:
                return _queries("only-one")
            return _queries("q1", "q2", "q3")

    class FakeModel:
        def __init__(self) -> None:
            self.structured = FakeStructured()

        def with_structured_output(self, _schema: Any) -> FakeStructured:
            return self.structured

    model = FakeModel()
    result = asyncio.run(
        generate_queries(
            research_question="Food insecurity",
            explanation="Prevalence of undernourishment.",
            config=_config(),
            model=model,  # type: ignore[arg-type]
        )
    )
    assert result == ["q1", "q2", "q3"]
    assert model.structured.calls == 2


def test_validation_reprompts_when_too_many() -> None:
    class FakeStructured:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke(self, _messages: Any) -> SearchQueryList:
            self.calls += 1
            if self.calls == 1:
                return _queries("a", "b", "c", "d", "e", "f")
            return _queries("q1", "q2", "q3", "q4")

    class FakeModel:
        def __init__(self) -> None:
            self.structured = FakeStructured()

        def with_structured_output(self, _schema: Any) -> FakeStructured:
            return self.structured

    model = FakeModel()
    result = asyncio.run(
        generate_queries(
            research_question="Yield gaps",
            explanation="Crop yield relative to potential.",
            config=_config(),
            model=model,  # type: ignore[arg-type]
        )
    )
    assert result == ["q1", "q2", "q3", "q4"]
    assert model.structured.calls == 2


def test_truncates_when_too_many_after_retries_exhausted() -> None:
    class FakeStructured:
        async def ainvoke(self, _messages: Any) -> SearchQueryList:
            return _queries("a", "b", "c", "d", "e", "f", "g")

    class FakeModel:
        def with_structured_output(self, _schema: Any) -> FakeStructured:
            return FakeStructured()

    result = asyncio.run(
        generate_queries(
            research_question="Livestock",
            explanation="Herd size trends.",
            config=_config(max_agent_retries=0),
            model=FakeModel(),  # type: ignore[arg-type]
        )
    )
    assert result == ["a", "b", "c", "d", "e"]


def test_raises_when_too_few_after_retries_exhausted() -> None:
    class FakeStructured:
        async def ainvoke(self, _messages: Any) -> SearchQueryList:
            return _queries("only-one", "only-two")

    class FakeModel:
        def with_structured_output(self, _schema: Any) -> FakeStructured:
            return FakeStructured()

    with pytest.raises(ValueError, match="too few queries"):
        asyncio.run(
            generate_queries(
                research_question="Fisheries",
                explanation="Catch volume.",
                config=_config(max_agent_retries=0),
                model=FakeModel(),  # type: ignore[arg-type]
            )
        )


def test_rejects_invalid_min_max_config() -> None:
    with pytest.raises(ValueError, match="min_queries"):
        asyncio.run(
            generate_queries(
                research_question="x",
                explanation="y",
                config=_config(min_queries=5, max_queries=3),
            )
        )
