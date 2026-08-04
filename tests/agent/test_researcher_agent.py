"""Unit tests for the ResearcherAgent evidence loop."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

from beanie import PydanticObjectId

from fao_impact_monitor.agent.query_generator_agent import ResearchQuery
from fao_impact_monitor.agent.researcher_agent import (
    AnswerStatement,
    AnswerStatementList,
    EvidenceClaim,
    EvidenceGap,
    EvidenceGapList,
    EvidenceSufficiency,
    ExtractedClaimCandidate,
    ExtractedClaimList,
    SourceReference,
    StatementRepair,
    StatementVerification,
    build_final_summary,
    match_quoted_text,
    normalize_for_quote_match,
    research,
    resolve_statement_citations,
)
from fao_impact_monitor.config import ResearcherConfig
from fao_impact_monitor.data_lake.document import DocumentType
from fao_impact_monitor.data_lake.vectorstore import ChunkHit
from fao_impact_monitor.data_source.data_source_config import DataSourceConfig
from fao_impact_monitor.metric.metric import Metric


def _metric() -> Metric:
    return Metric(
        name="Maize production change after drought",
        description="Quantify maize production change in the selected country.",
        example=(
            "In Exampleland, maize production fell by 99% according to a "
            "fabricated baseline that must never be used as evidence."
        ),
        unit="percent change",
        data_sources=[DataSourceConfig(source="vectorstore")],
    )


def _hit(
    *,
    text: str,
    url: str = "https://fao.org/kenya-maize.pdf",
    title: str = "Kenya Maize Report",
    chunk_index: int = 0,
    document_id: str | None = None,
) -> ChunkHit:
    return ChunkHit(
        document_id=PydanticObjectId(document_id or "507f1f77bcf86cd799439011"),
        document_url=url,
        document_title=title,
        document_meta={},
        document_type=DocumentType.PDF,
        document_source="fao",
        chunk_index=chunk_index,
        chunk_text=text,
        countries_iso3=["KEN"],
        score=0.9,
    )


class ScriptedModel:
    """Fake chat model returning queued structured outputs by schema name."""

    def __init__(self, scripts: dict[str, list[Any]]) -> None:
        self.scripts = {k: list(v) for k, v in scripts.items()}
        self.calls: list[str] = []

    def with_structured_output(self, schema: Any) -> Any:
        name = getattr(schema, "__name__", str(schema))
        parent = self

        class _Structured:
            async def ainvoke(self, _messages: Any) -> Any:
                parent.calls.append(name)
                queue = parent.scripts.get(name, [])
                if not queue:
                    raise AssertionError(f"No scripted response for {name}")
                return queue.pop(0)

        return _Structured()


class FakeVectorStore:
    def __init__(self, hits_by_query: dict[str, list[ChunkHit]] | None = None) -> None:
        self.hits_by_query = hits_by_query or {}
        self.calls: list[str] = []

    async def search(
        self,
        query: str,
        *,
        countries_iso3: list[str] | None = None,
        limit: int | None = None,
    ) -> list[ChunkHit]:
        del countries_iso3, limit
        self.calls.append(query)
        return list(self.hits_by_query.get(query, []))


def _config(
    *,
    max_research_iterations: int = 2,
    max_queries_per_iteration: int = 3,
    vector_results_per_query: int = 5,
    max_web_queries_per_iteration: int = 2,
    max_claim_extraction_retries: int = 1,
    max_answer_verification_retries: int = 1,
    max_agent_retries: int = 1,
) -> ResearcherConfig:
    return ResearcherConfig(
        max_research_iterations=max_research_iterations,
        max_queries_per_iteration=max_queries_per_iteration,
        vector_results_per_query=vector_results_per_query,
        max_web_queries_per_iteration=max_web_queries_per_iteration,
        max_claim_extraction_retries=max_claim_extraction_retries,
        max_answer_verification_retries=max_answer_verification_retries,
        max_agent_retries=max_agent_retries,
    )


def test_exact_and_normalized_quote_matching() -> None:
    source = "Kenya maize  production\nfell by 12%."
    assert match_quoted_text("Kenya maize  production\nfell by 12%.", source) == "exact"
    quoted = "Kenya maize production fell by 12%."
    assert match_quoted_text(quoted, source) == "normalized"
    assert match_quoted_text("Kenya maize production rose by 12%.", source) is None
    assert "agriculture" in normalize_for_quote_match("agri-\nculture")


def test_final_summary_citations_use_document_uri_and_page() -> None:
    statement = AnswerStatement(
        statement_id="stmt_001",
        text="Kenya maize production fell by 12%.",
        supporting_claim_ids=["claim_001"],
        citations=[],
    )
    claims = [
        EvidenceClaim(
            claim_id="claim_001",
            source_type="vectorstore",
            source_id="vs:1:0",
            quoted_text="Kenya maize production fell by 12%.",
            country="Kenya",
            relevance="direct",
            url="https://fao.org/kenya-maize.pdf",
            page_number=1,
        )
    ]
    sources = {
        "vs:1:0": SourceReference(
            source_id="vs:1:0",
            source_type="vectorstore",
            document_uri="https://fao.org/kenya-maize.pdf",
            document_name="Kenya Maize Report",
            page_number=1,
        )
    }
    citations = resolve_statement_citations(statement, claims, sources)
    summary = build_final_summary(
        [statement.model_copy(update={"citations": citations})]
    )
    assert "Kenya Maize Report, p. 1" in summary
    assert "(https://fao.org/kenya-maize.pdf)" in summary


def test_research_happy_path_vector_only_skips_web() -> None:
    quote = "In Kenya, maize production fell by 12% in 2016."
    hit = _hit(text=f"Preface. {quote} Conclusion.")
    query = "Kenya maize production drought 2016"

    async def gen_queries(**kwargs: Any) -> list[ResearchQuery]:
        del kwargs
        return [
            ResearchQuery(
                query=query,
                purpose="direct value",
                target_gap_ids=[],
                destination="vectorstore",
            )
        ]

    web_fn = AsyncMock(side_effect=AssertionError("web should not run"))
    model = ScriptedModel(
        {
            "ExtractedClaimList": [
                ExtractedClaimList(
                    claims=[
                        ExtractedClaimCandidate(
                            source_id=(f"vs:{hit.document_id}:{hit.chunk_index}"),
                            quoted_text=quote,
                            country="Kenya",
                            relevance="metric value",
                            metric_aspects=["value"],
                        )
                    ]
                )
            ],
            "EvidenceGapList": [
                EvidenceGapList(
                    gaps=[
                        EvidenceGap(
                            gap_id="gap_001",
                            description="none",
                            why_required="n/a",
                            status="closed",
                        )
                    ],
                    established_facts=[quote],
                )
            ],
            "EvidenceSufficiency": [
                EvidenceSufficiency(
                    is_sufficient=True,
                    supported_metric_aspects=["value"],
                    open_gap_ids=[],
                    reasoning="Value present for Kenya.",
                    next_action="draft_answer",
                    needs_web=False,
                )
            ],
            "AnswerStatementList": [
                AnswerStatementList(
                    statements=[
                        AnswerStatement(
                            statement_id="stmt_001",
                            text=quote,
                            supporting_claim_ids=["claim_001"],
                            metric_aspects=["value"],
                        )
                    ]
                )
            ],
            "StatementVerification": [
                StatementVerification(
                    statement_id="stmt_001",
                    verdict="entailed",
                    unsupported_parts=[],
                    reasoning="Direct quote support.",
                )
            ],
        }
    )
    store = FakeVectorStore({query: [hit]})
    result = asyncio.run(
        research(
            metric=_metric(),
            country_iso3="KEN",
            vector_store=store,  # type: ignore[arg-type]
            config=_config(max_research_iterations=1),
            model=model,  # type: ignore[arg-type]
            verifier_model=model,  # type: ignore[arg-type]
            generate_research_queries_fn=gen_queries,
            web_research_fn=web_fn,
        )
    )
    assert result.status == "answered"
    assert web_fn.await_count == 0
    assert store.calls == [query]
    assert result.statements[0].supporting_claim_ids == ["claim_001"]
    assert result.claims[0].match_kind == "exact"
    assert "Kenya Maize Report, p. 1" in result.final_summary
    assert result.sources[0].document_uri == hit.document_url
    claim_ids = {c.claim_id for c in result.claims}
    source_ids = {s.source_id for s in result.sources}
    for stmt in result.statements:
        assert stmt.supporting_claim_ids
        assert set(stmt.supporting_claim_ids) <= claim_ids
        for claim in result.claims:
            if claim.claim_id in stmt.supporting_claim_ids:
                assert claim.source_id in source_ids


def test_follow_up_queries_and_claim_survival_across_iterations() -> None:
    quote1 = "Kenya experienced a severe drought in 2016."
    quote2 = "In Kenya, maize production fell by 12% in 2016."
    hit1 = _hit(text=quote1, chunk_index=0, document_id="507f1f77bcf86cd799439011")
    hit2 = _hit(text=quote2, chunk_index=1, document_id="507f1f77bcf86cd799439012")
    q1 = "Kenya drought 2016 agriculture"
    q2 = "Kenya maize production change 2016 percent"

    calls: list[dict[str, Any]] = []

    async def gen_queries(**kwargs: Any) -> list[ResearchQuery]:
        calls.append(kwargs)
        if len(calls) == 1:
            return [
                ResearchQuery(
                    query=q1,
                    purpose="context",
                    target_gap_ids=[],
                    destination="vectorstore",
                )
            ]
        assert kwargs.get("open_gaps")
        assert kwargs.get("executed_queries")
        return [
            ResearchQuery(
                query=q2,
                purpose="metric value",
                target_gap_ids=["gap_value"],
                destination="vectorstore",
            )
        ]

    model = ScriptedModel(
        {
            "ExtractedClaimList": [
                ExtractedClaimList(
                    claims=[
                        ExtractedClaimCandidate(
                            source_id=f"vs:{hit1.document_id}:0",
                            quoted_text=quote1,
                            country="Kenya",
                            relevance="context",
                        )
                    ]
                ),
                ExtractedClaimList(
                    claims=[
                        ExtractedClaimCandidate(
                            source_id=f"vs:{hit2.document_id}:1",
                            quoted_text=quote2,
                            country="Kenya",
                            relevance="value",
                        )
                    ]
                ),
            ],
            "EvidenceGapList": [
                EvidenceGapList(
                    gaps=[
                        EvidenceGap(
                            gap_id="gap_value",
                            description="Missing maize production change value",
                            why_required="Metric requires percent change",
                            status="open",
                            suggested_terms=["maize production"],
                        )
                    ],
                    established_facts=[quote1],
                ),
                EvidenceGapList(
                    gaps=[
                        EvidenceGap(
                            gap_id="gap_value",
                            description="Missing maize production change value",
                            why_required="Metric requires percent change",
                            status="closed",
                        )
                    ],
                    established_facts=[quote1, quote2],
                ),
            ],
            "EvidenceSufficiency": [
                EvidenceSufficiency(
                    is_sufficient=False,
                    supported_metric_aspects=["context"],
                    open_gap_ids=["gap_value"],
                    reasoning="Need the production change value.",
                    next_action="generate_more_queries",
                    needs_web=False,
                ),
                EvidenceSufficiency(
                    is_sufficient=True,
                    supported_metric_aspects=["value"],
                    open_gap_ids=[],
                    reasoning="Value found.",
                    next_action="draft_answer",
                    needs_web=False,
                ),
            ],
            "AnswerStatementList": [
                AnswerStatementList(
                    statements=[
                        AnswerStatement(
                            statement_id="stmt_001",
                            text=quote2,
                            supporting_claim_ids=["claim_001", "claim_002"],
                        )
                    ]
                )
            ],
            "StatementVerification": [
                StatementVerification(
                    statement_id="stmt_001",
                    verdict="entailed",
                    unsupported_parts=[],
                    reasoning="ok",
                )
            ],
        }
    )
    store = FakeVectorStore({q1: [hit1], q2: [hit2]})
    result = asyncio.run(
        research(
            metric=_metric(),
            country_iso3="KEN",
            vector_store=store,  # type: ignore[arg-type]
            config=_config(max_research_iterations=2),
            model=model,  # type: ignore[arg-type]
            verifier_model=model,  # type: ignore[arg-type]
            generate_research_queries_fn=gen_queries,
            web_research_fn=AsyncMock(),
        )
    )
    assert len(calls) == 2
    assert result.status == "answered"
    assert {c.claim_id for c in result.claims} == {"claim_001", "claim_002"}
    # claim ids remain stable / sequential
    assert result.claims[0].claim_id == "claim_001"


def test_web_runs_when_needs_web_and_snippets_rejected() -> None:
    quote = "Official Kenya statistics report a 12% maize decline."
    q = "Kenya maize official statistics percent"

    async def gen_queries(**kwargs: Any) -> list[ResearchQuery]:
        iteration = len(kwargs.get("executed_queries") or [])
        dest = "web" if iteration else "vectorstore"
        return [
            ResearchQuery(
                query=q if not iteration else q + " web",
                purpose="value",
                target_gap_ids=[] if not iteration else ["gap_001"],
                destination=dest,  # type: ignore[arg-type]
            )
        ]

    async def web_fn(query: str, **kwargs: Any) -> Any:
        del kwargs
        from types import SimpleNamespace

        return SimpleNamespace(
            scraped=[
                SimpleNamespace(
                    url="https://knbs.go.ke/maize",
                    title="KNBS Maize",
                    content=quote,
                )
            ],
            snippet_only=[
                SimpleNamespace(
                    url="https://spam.example/snippet",
                    title="spam",
                    content="Kenya maize...",
                )
            ],
            scrape_failed=[],
            blocked_by_policy=[],
            source_http_error=[],
            scraped_irrelevant=[],
            bot_detected=[],
            queries=[SimpleNamespace(query=query)],
            synthesis="Do not cite this synthesis as evidence.",
        )

    class PatchingModel(ScriptedModel):
        def with_structured_output(self, schema: Any) -> Any:
            name = getattr(schema, "__name__", str(schema))
            parent = self

            class _Structured:
                async def ainvoke(self, messages: Any) -> Any:
                    parent.calls.append(name)
                    queue = parent.scripts.get(name, [])
                    if not queue:
                        raise AssertionError(f"No scripted response for {name}")
                    item = queue.pop(0)
                    if name == "ExtractedClaimList":
                        content = messages[1].content
                        marker = "source_id="
                        assert marker in content
                        sid = content.split(marker, 1)[1].split(" ", 1)[0]
                        return ExtractedClaimList(
                            claims=[
                                ExtractedClaimCandidate(
                                    source_id=sid,
                                    quoted_text=quote,
                                    country="Kenya",
                                    relevance="official value",
                                )
                            ]
                        )
                    return item

            return _Structured()

    # Iteration 1 has no new sources so extract is skipped; only one extract
    # response is needed (iteration 2 after web scrape).
    model = PatchingModel(
        {
            "ExtractedClaimList": [ExtractedClaimList(claims=[])],
            "EvidenceGapList": [
                EvidenceGapList(
                    gaps=[
                        EvidenceGap(
                            gap_id="gap_001",
                            description="Need official value",
                            why_required="Metric value missing",
                            preferred_source_type="web",
                            status="open",
                        )
                    ]
                ),
                EvidenceGapList(
                    gaps=[
                        EvidenceGap(
                            gap_id="gap_001",
                            description="Need official value",
                            why_required="Metric value missing",
                            status="closed",
                        )
                    ],
                    established_facts=[quote],
                ),
            ],
            "EvidenceSufficiency": [
                EvidenceSufficiency(
                    is_sufficient=False,
                    supported_metric_aspects=[],
                    open_gap_ids=["gap_001"],
                    reasoning="Need web",
                    next_action="generate_more_queries",
                    needs_web=True,
                ),
                EvidenceSufficiency(
                    is_sufficient=True,
                    supported_metric_aspects=["value"],
                    open_gap_ids=[],
                    reasoning="Have official value",
                    next_action="draft_answer",
                    needs_web=False,
                ),
            ],
            "AnswerStatementList": [
                AnswerStatementList(
                    statements=[
                        AnswerStatement(
                            statement_id="stmt_001",
                            text=quote,
                            supporting_claim_ids=["claim_001"],
                        )
                    ]
                )
            ],
            "StatementVerification": [
                StatementVerification(
                    statement_id="stmt_001",
                    verdict="entailed",
                    unsupported_parts=[],
                    reasoning="ok",
                )
            ],
        }
    )

    store = FakeVectorStore({})
    result = asyncio.run(
        research(
            metric=_metric(),
            country_iso3="KEN",
            vector_store=store,  # type: ignore[arg-type]
            config=_config(max_research_iterations=2),
            model=model,  # type: ignore[arg-type]
            verifier_model=model,  # type: ignore[arg-type]
            generate_research_queries_fn=gen_queries,
            web_research_fn=web_fn,
        )
    )
    assert result.status == "answered"
    assert result.claims[0].source_type == "web"
    assert result.sources[0].document_uri == "https://knbs.go.ke/maize"
    assert "synthesis" not in result.final_summary.lower()
    assert "spam.example" not in result.final_summary


def test_rejects_fabricated_quotes_and_example_leak() -> None:
    real = "Kenya harvested 3 million tonnes of maize."
    hit = _hit(text=real)
    q = "Kenya maize harvest tonnes"

    async def gen_queries(**kwargs: Any) -> list[ResearchQuery]:
        del kwargs
        return [
            ResearchQuery(
                query=q,
                purpose="value",
                target_gap_ids=[],
                destination="vectorstore",
            )
        ]

    model = ScriptedModel(
        {
            "ExtractedClaimList": [
                ExtractedClaimList(
                    claims=[
                        ExtractedClaimCandidate(
                            source_id=f"vs:{hit.document_id}:0",
                            quoted_text="Kenya maize production fell by 99%",
                            country="Kenya",
                            relevance="fabricated",
                        ),
                        ExtractedClaimCandidate(
                            source_id=f"vs:{hit.document_id}:0",
                            quoted_text=(
                                "In Exampleland, maize production fell by 99% "
                                "according to a fabricated baseline that must "
                                "never be used as evidence."
                            ),
                            country="Kenya",
                            relevance="example leak",
                        ),
                    ]
                )
            ],
            "EvidenceGapList": [
                EvidenceGapList(
                    gaps=[
                        EvidenceGap(
                            gap_id="gap_001",
                            description="No valid claims",
                            why_required="Need evidence",
                            status="open",
                        )
                    ]
                )
            ],
            "EvidenceSufficiency": [
                EvidenceSufficiency(
                    is_sufficient=False,
                    supported_metric_aspects=[],
                    open_gap_ids=["gap_001"],
                    reasoning="No claims",
                    next_action="return_insufficient_evidence",
                    needs_web=False,
                )
            ],
        }
    )
    result = asyncio.run(
        research(
            metric=_metric(),
            country_iso3="KEN",
            vector_store=FakeVectorStore({q: [hit]}),  # type: ignore[arg-type]
            config=_config(
                max_research_iterations=1,
                max_claim_extraction_retries=0,
            ),
            model=model,  # type: ignore[arg-type]
            verifier_model=model,  # type: ignore[arg-type]
            generate_research_queries_fn=gen_queries,
            web_research_fn=AsyncMock(),
        )
    )
    assert result.status == "insufficient_evidence"
    assert result.claims == []
    assert "insufficient" in result.final_summary.lower()
    assert "Exampleland" not in result.final_summary


def test_multi_country_chunk_keeps_only_selected_country_claims() -> None:
    text = (
        "In Uganda, maize rose by 5%. In Kenya, maize production fell by 12%. "
        "In Tanzania, maize was stable."
    )
    hit = _hit(text=text)
    q = "Kenya Uganda Tanzania maize production comparison"

    async def gen_queries(**kwargs: Any) -> list[ResearchQuery]:
        del kwargs
        return [
            ResearchQuery(
                query=q,
                purpose="country value",
                target_gap_ids=[],
                destination="vectorstore",
            )
        ]

    kenya_quote = "In Kenya, maize production fell by 12%."
    uganda_quote = "In Uganda, maize rose by 5%."
    model = ScriptedModel(
        {
            "ExtractedClaimList": [
                ExtractedClaimList(
                    claims=[
                        ExtractedClaimCandidate(
                            source_id=f"vs:{hit.document_id}:0",
                            quoted_text=kenya_quote,
                            country="Kenya",
                            relevance="selected country",
                        ),
                        ExtractedClaimCandidate(
                            source_id=f"vs:{hit.document_id}:0",
                            quoted_text=uganda_quote,
                            country="Uganda",
                            relevance="other country",
                        ),
                    ]
                )
            ],
            "EvidenceGapList": [
                EvidenceGapList(
                    gaps=[],
                    established_facts=[kenya_quote],
                )
            ],
            "EvidenceSufficiency": [
                EvidenceSufficiency(
                    is_sufficient=True,
                    supported_metric_aspects=["value"],
                    open_gap_ids=[],
                    reasoning="Kenya claim enough",
                    next_action="draft_answer",
                )
            ],
            "AnswerStatementList": [
                AnswerStatementList(
                    statements=[
                        AnswerStatement(
                            statement_id="stmt_001",
                            text=kenya_quote,
                            supporting_claim_ids=["claim_001"],
                        )
                    ]
                )
            ],
            "StatementVerification": [
                StatementVerification(
                    statement_id="stmt_001",
                    verdict="entailed",
                    unsupported_parts=[],
                    reasoning="ok",
                )
            ],
        }
    )
    result = asyncio.run(
        research(
            metric=_metric(),
            country_iso3="KEN",
            vector_store=FakeVectorStore({q: [hit]}),  # type: ignore[arg-type]
            config=_config(max_research_iterations=1),
            model=model,  # type: ignore[arg-type]
            verifier_model=model,  # type: ignore[arg-type]
            generate_research_queries_fn=gen_queries,
            web_research_fn=AsyncMock(),
        )
    )
    assert len(result.claims) == 1
    assert "Kenya" in result.claims[0].quoted_text
    assert "Uganda" not in result.claims[0].quoted_text


def test_statement_repair_and_query_dedup_and_limits() -> None:
    quote = "In Kenya, maize production fell by an estimated 12% in 2016."
    hit = _hit(text=quote)
    q = "Kenya maize estimated production decline 2016"

    async def gen_queries(**kwargs: Any) -> list[ResearchQuery]:
        # Always return the same query — researcher must not re-execute forever.
        return [
            ResearchQuery(
                query=q,
                purpose="value",
                target_gap_ids=[g.gap_id for g in (kwargs.get("open_gaps") or [])],
                destination="vectorstore",
            )
        ]

    model = ScriptedModel(
        {
            "ExtractedClaimList": [
                ExtractedClaimList(
                    claims=[
                        ExtractedClaimCandidate(
                            source_id=f"vs:{hit.document_id}:0",
                            quoted_text=quote,
                            country="Kenya",
                            relevance="value",
                        )
                    ]
                )
            ],
            "EvidenceGapList": [
                EvidenceGapList(
                    gaps=[],
                    established_facts=[quote],
                )
            ],
            "EvidenceSufficiency": [
                EvidenceSufficiency(
                    is_sufficient=True,
                    supported_metric_aspects=["value"],
                    open_gap_ids=[],
                    reasoning="ok",
                    next_action="draft_answer",
                )
            ],
            "AnswerStatementList": [
                AnswerStatementList(
                    statements=[
                        AnswerStatement(
                            statement_id="stmt_001",
                            text="Kenya maize production fell by 12% in 2016.",
                            supporting_claim_ids=["claim_001"],
                        )
                    ]
                )
            ],
            "StatementVerification": [
                StatementVerification(
                    statement_id="stmt_001",
                    verdict="partially_entailed",
                    unsupported_parts=["certainty"],
                    reasoning="Must keep estimated",
                    suggested_revision=quote,
                ),
                StatementVerification(
                    statement_id="stmt_001",
                    verdict="entailed",
                    unsupported_parts=[],
                    reasoning="ok",
                ),
            ],
            "StatementRepair": [
                StatementRepair(
                    statement_id="stmt_001",
                    text=quote,
                    supporting_claim_ids=["claim_001"],
                    remove=False,
                )
            ],
        }
    )
    store = FakeVectorStore({q: [hit]})
    result = asyncio.run(
        research(
            metric=_metric(),
            country_iso3="KEN",
            vector_store=store,  # type: ignore[arg-type]
            config=_config(
                max_research_iterations=1,
                max_answer_verification_retries=1,
            ),
            model=model,  # type: ignore[arg-type]
            verifier_model=model,  # type: ignore[arg-type]
            generate_research_queries_fn=gen_queries,
            web_research_fn=AsyncMock(),
        )
    )
    assert result.status == "answered"
    assert "estimated" in result.statements[0].text
    assert store.calls.count(q) == 1
