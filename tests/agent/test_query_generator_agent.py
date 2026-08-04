"""Unit tests for the query generator agent."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from fao_impact_monitor.agent.query_generator_agent import (
    EvidenceGapInput,
    ResearchQuery,
    ResearchQueryList,
    SearchQuery,
    SearchQueryList,
    filter_research_queries,
    generate_queries,
    generate_research_queries,
    normalize_query,
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


def test_normalize_query_collapses_whitespace_and_case() -> None:
    assert normalize_query("  Kenya   Maize ") == normalize_query("kenya maize")


def test_filter_research_queries_rejects_duplicates_and_missing_country() -> None:
    kept = filter_research_queries(
        [
            ResearchQuery(
                query="Kenya maize yields drought",
                purpose="value",
                target_gap_ids=["gap_1"],
                destination="vectorstore",
            ),
            ResearchQuery(
                query="kenya maize yields drought",
                purpose="dup",
                target_gap_ids=["gap_1"],
                destination="vectorstore",
            ),
            ResearchQuery(
                query="global maize yields",
                purpose="no country",
                target_gap_ids=["gap_1"],
                destination="vectorstore",
            ),
            ResearchQuery(
                query="Kenya already run",
                purpose="executed",
                target_gap_ids=["gap_1"],
                destination="web",
            ),
        ],
        country_name="Kenya",
        country_iso3="KEN",
        executed_queries=["Kenya already run"],
        open_gap_ids={"gap_1"},
        require_gaps=True,
    )
    assert [q.query for q in kept] == ["Kenya maize yields drought"]


def test_generate_research_queries_initial_and_follow_up() -> None:
    class FakeStructured:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke(self, messages: Any) -> ResearchQueryList:
            self.calls += 1
            content = messages[1].content
            assert "Kenya" in content
            assert "percent" in content.lower() or "unit" in content.lower()
            if "Open evidence gaps" in content:
                assert "gap_value" in content
                assert "Previously executed" in content
                return ResearchQueryList(
                    queries=[
                        ResearchQuery(
                            query="Kenya maize production percent change 2016",
                            purpose="close value gap",
                            target_gap_ids=["gap_value"],
                            destination="both",
                        )
                    ]
                )
            return ResearchQueryList(
                queries=[
                    ResearchQuery(
                        query="Kenya maize production drought impacts",
                        purpose="direct",
                        target_gap_ids=[],
                        destination="vectorstore",
                    ),
                    ResearchQuery(
                        query="Kenya cereal yield methodology definition",
                        purpose="definition",
                        target_gap_ids=[],
                        destination="vectorstore",
                    ),
                    ResearchQuery(
                        query="Kenya maize unit tonnes reporting period",
                        purpose="unit",
                        target_gap_ids=[],
                        destination="vectorstore",
                    ),
                ]
            )

    structured = FakeStructured()

    class FakeModel:
        def with_structured_output(self, _schema: Any) -> FakeStructured:
            return structured

    initial = asyncio.run(
        generate_research_queries(
            research_question="Maize production change",
            explanation="Quantify change after drought",
            unit="percent",
            country_name="Kenya",
            country_iso3="KEN",
            example="Never copy this fabricated 99% figure.",
            config=_config(min_queries=3, max_queries=5),
            model=FakeModel(),  # type: ignore[arg-type]
        )
    )
    assert len(initial) == 3
    assert all("Kenya" in q.query for q in initial)

    follow_up = asyncio.run(
        generate_research_queries(
            research_question="Maize production change",
            explanation="Quantify change after drought",
            unit="percent",
            country_name="Kenya",
            country_iso3="KEN",
            established_facts=["Drought occurred in 2016."],
            open_gaps=[
                EvidenceGapInput(
                    gap_id="gap_value",
                    description="Missing percent change",
                    why_required="Metric unit",
                    suggested_terms=["percent change"],
                )
            ],
            executed_queries=[q.query for q in initial],
            preferred_destinations=["both"],
            min_queries=1,
            max_queries=3,
            config=_config(min_queries=1, max_queries=3),
            model=FakeModel(),  # type: ignore[arg-type]
        )
    )
    assert len(follow_up) == 1
    assert follow_up[0].target_gap_ids == ["gap_value"]
    assert structured.calls == 2
