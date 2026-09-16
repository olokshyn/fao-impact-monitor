"""Tests for bounded, quantitative, PDF-first research."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from beanie import PydanticObjectId

from fao_impact_monitor.agent.query_generator_agent import ResearchQuery
from fao_impact_monitor.agent.researcher_agent import (
    AnswerStatement,
    AnswerStatementList,
    ClaimUsefulnessList,
    ClaimUsefulnessVerdict,
    EvidenceClaim,
    EvidenceGap,
    ExtractedClaimCandidate,
    ExtractedClaimList,
    PdfVerifiedVisualFact,
    ResearcherOutput,
    ResearchState,
    RetrievedChunk,
    StatementCitation,
    StatementVerification,
    VerifiedResearchVisualFact,
    VisualArtifact,
    VisualInsightCandidate,
    VisualInsightList,
    VisualInsightVerdict,
    VisualInsightVerdictList,
    _enrich_visual_chunk,
    _extract_and_validate_batches,
    _has_explicit_scope_conflict,
    _has_only_ineligible_explicit_years,
    _judge_claim_usefulness,
    _select_claims,
    _validate_claim_candidates,
    build_final_summary,
    chunk_from_hit,
    classify_researcher_status,
    format_human_markdown,
    is_direct_evidence_claim,
    match_quoted_text,
    normalize_for_quote_match,
    research,
    statements_have_quantitative_evidence,
)
from fao_impact_monitor.config import ResearcherConfig
from fao_impact_monitor.data_lake.document import DocumentType
from fao_impact_monitor.data_lake.vectorstore import ChunkHit
from fao_impact_monitor.data_provider.web_scout_provider import WebSource
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
    index: int,
    text: str,
    *,
    countries_iso3: list[str] | None = None,
) -> ChunkHit:
    return ChunkHit(
        document_id=PydanticObjectId(f"{index + 1:024x}"),
        document_url=f"file://fao_data/report-{index}.pdf",
        document_title=f"Report {index}",
        document_meta={},
        document_type=DocumentType.PDF,
        document_source="PdfEvidencePipeline",
        chunk_index=index,
        chunk_text=text,
        countries_iso3=countries_iso3 or ["KEN"],
        score=1.0 - index / 100,
    )


class ScriptedModel:
    def __init__(self, scripts: dict[str, list[Any]]) -> None:
        self.scripts = {key: list(values) for key, values in scripts.items()}
        self.calls: list[str] = []
        self.messages: list[Any] = []

    def with_structured_output(self, schema: Any, **_kwargs: Any) -> Any:
        name = getattr(schema, "__name__", str(schema))
        parent = self

        class Structured:
            async def ainvoke(self, messages: Any) -> Any:
                parent.calls.append(name)
                parent.messages.append(messages)
                queue = parent.scripts.get(name, [])
                if not queue:
                    raise AssertionError(f"No scripted response for {name}")
                return queue.pop(0)

        return Structured()


class FakeVectorStore:
    def __init__(self, hits: list[ChunkHit]) -> None:
        self.hits = hits
        self.calls: list[dict[str, Any]] = []

    async def search(
        self,
        query: str,
        *,
        countries_iso3: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> list[ChunkHit]:
        self.calls.append(
            {
                "query": query,
                "countries_iso3": countries_iso3,
                "limit": limit,
            }
        )
        return self.hits[:limit]


async def _one_pdf_query(**_kwargs: Any) -> list[ResearchQuery]:
    return [
        ResearchQuery(
            query="Kenya El Nino maize production percentage loss table",
            purpose="quantitative production evidence",
            destination="vectorstore",
        )
    ]


def _config(**overrides: Any) -> ResearcherConfig:
    values: dict[str, Any] = {
        "use_visual_evidence": False,
        "target_pdf_claims_per_metric": 20,
        "max_web_claims_per_metric": 5,
        "max_pdf_queries_per_metric": 1,
        "pdf_results_per_query": 20,
        "max_pdf_evidence_to_analyze": 50,
        "claim_extraction_batch_size": 20,
        "max_claims_per_evidence": 5,
        "max_web_searches_per_metric": 5,
        "max_web_depth": 2,
        "max_answer_verification_retries": 0,
    }
    values.update(overrides)
    return ResearcherConfig(**values)


def _candidate(hit: ChunkHit, text: str, *, fit: str) -> ExtractedClaimCandidate:
    return ExtractedClaimCandidate(
        source_id=f"vs:{hit.document_id}:{hit.chunk_index}",
        quoted_text=text,
        country="",
        relevance="Answers the requested production-change metric.",
        answer_fit=fit,  # type: ignore[arg-type]
        metric_aspects=["production change"],
    )


def _usefulness_verdicts(
    count: int,
    *,
    start: int = 1,
    verdict: str = "direct_answer",
) -> ClaimUsefulnessList:
    return ClaimUsefulnessList(
        verdicts=[
            ClaimUsefulnessVerdict(
                claim_id=f"claim_{index:03d}",
                verdict=verdict,  # type: ignore[arg-type]
                reason="Directly answers the metric.",
            )
            for index in range(start, start + count)
        ]
    )


def test_exact_and_normalized_quote_matching() -> None:
    source = "Kenya maize  production\nfell by 12%."
    assert match_quoted_text("Kenya maize  production\nfell by 12%.", source) == "exact"
    assert (
        match_quoted_text("Kenya maize production fell by 12%.", source) == "normalized"
    )
    assert match_quoted_text("Kenya maize production rose by 12%.", source) is None
    assert "agriculture" in normalize_for_quote_match("agri-\nculture")


def test_quote_match_survives_pdf_wrap_whitespace_and_hyphen_ranges() -> None:
    """PDF wraps leave ' \\n' and digit ranges that used to break validation."""
    # Space before newline must not become a sticky double space.
    assert (
        match_quoted_text(
            "did not produce the impacts anticipated",
            "did \nnot produce the impacts anticipated",
        )
        == "normalized"
    )
    assert normalize_for_quote_match("did \nnot") == "did not"

    # Digit ranges across wraps must stay "5-10", not glue into "510".
    source_range = "contribute to 5–\n10 percent of national annual production"
    assert (
        match_quoted_text(
            "contribute to 5–10 percent of national annual production",
            source_range,
        )
        == "normalized"
    )
    assert "5-10" in normalize_for_quote_match(source_range)
    assert "510" not in normalize_for_quote_match(source_range)

    # Soft hyphen + newline between digits.
    soft = "5\u00ad\n10 percent"
    assert match_quoted_text("5-10 percent", soft) == "normalized"

    # Punctuation / dash differences still match via word tokens.
    assert (
        match_quoted_text(
            "affecting only four percent of the total agricultural area.",
            "affecting only four percent of the total agricultural  area.",
        )
        == "normalized"
    )
    assert (
        match_quoted_text(
            "harvests were well below average, with some areas experiencing "
            "between 50 and 90 percent crop loss.",
            "harvests were well below average, with some areas  experiencing "
            "between 50 and 90 percent crop loss.",
        )
        == "normalized"
    )


def test_ingestion_visual_fact_without_artifact_is_validated() -> None:
    quote = "The chart shows that 35% of cropland was affected."
    hit = _hit(0, f"[VERIFIED VISUAL FACT]\n{quote}")
    chunk = chunk_from_hit(hit, "cropland affected percentage chart")
    chunk.verified_visual_facts = [PdfVerifiedVisualFact(text=quote)]
    state = ResearchState(
        metric=_metric(),
        country_iso3="KEN",
        country_name="Kenya",
        vector_chunks=[chunk],
    )

    accepted, rejected = _validate_claim_candidates(
        state,
        [_candidate(hit, quote, fit="direct_requested_unit")],
    )

    assert rejected == []
    assert len(accepted) == 1
    assert accepted[0].evidence_modality == "verified_visual_fact"
    assert accepted[0].visual_artifact_ids == []


def test_visual_fact_quote_matches_when_absent_from_chunk_text() -> None:
    quote = (
        "India: Wheat 5-yr avg=111.8, 2025=117.9, 2026=120.2; "
        "Total cereals Change 2026/2025=-1.2%"
    )
    hit = _hit(0, "[TARGET SOURCE EVIDENCE]\nNone.")
    chunk = chunk_from_hit(hit, "cereal production outlook")
    chunk.verified_visual_facts = [PdfVerifiedVisualFact(text=quote)]
    state = ResearchState(
        metric=_metric(),
        country_iso3="IND",
        country_name="Republic of India",
        vector_chunks=[chunk],
    )

    accepted, rejected = _validate_claim_candidates(
        state,
        [_candidate(hit, quote, fit="direct_related_measure")],
    )

    assert rejected == []
    assert len(accepted) == 1
    assert accepted[0].quoted_text == quote
    assert accepted[0].evidence_modality == "verified_visual_fact"


def test_chunk_from_pdf_hit_preserves_exact_evidence_provenance() -> None:
    hit = ChunkHit(
        document_id=PydanticObjectId("507f1f77bcf86cd799439011"),
        document_url="file://fao_data/report.pdf",
        document_title="El Niño report",
        document_meta={
            "pipeline": "pdf_pipeline",
            "evidence_id": "evidence-001",
            "source_text": "Affected cropland reached 35%.",
            "physical_pages": [4, 5],
            "printed_pages": ["2", "3"],
            "events": [
                {
                    "event_id": "el_nino_2015_16",
                    "relationship": "associated",
                }
            ],
            "verified_visual_facts": [{"text": "The chart labels 35%."}],
        },
        document_type=DocumentType.PDF,
        document_source="PdfEvidencePipeline",
        chunk_index=3,
        chunk_text="Retrieval bundle with inherited context.",
        countries_iso3=["KEN"],
    )

    chunk = chunk_from_hit(hit, "cropland drought")

    assert chunk.evidence_id == "evidence-001"
    assert chunk.source_text == "Affected cropland reached 35%."
    assert chunk.physical_pages == [4, 5]
    assert chunk.printed_pages == ["2", "3"]
    assert chunk.events[0].event_id == "el_nino_2015_16"
    assert chunk.events[0].relationship == "associated"
    assert chunk.verified_visual_facts[0].text == "The chart labels 35%."


def test_final_summary_citations_use_document_uri_and_page() -> None:
    statement = AnswerStatement(
        statement_id="stmt_001",
        text="Maize production fell by 12%.",
        supporting_claim_ids=["claim_001"],
        citations=[
            StatementCitation(
                document_name="Kenya Maize Report",
                document_uri="https://fao.org/kenya-maize.pdf",
                page_number=4,
            )
        ],
    )
    summary = build_final_summary([statement])
    assert "Kenya Maize Report, p. 4" in summary
    assert "https://fao.org/kenya-maize.pdf" in summary


def test_format_human_markdown_uses_validated_statements() -> None:
    output = ResearcherOutput(
        status="answered",
        country="Kenya",
        metric_name="Maize production change after drought",
        final_summary="unused",
        statements=[
            AnswerStatement(
                statement_id="stmt_001",
                text="Maize production fell by 12%.",
                supporting_claim_ids=["claim_001"],
                citations=[
                    StatementCitation(
                        document_name="Kenya Maize Report",
                        document_uri="https://fao.org/kenya-maize.pdf",
                        page_number=4,
                    )
                ],
            )
        ],
        open_gaps=[
            EvidenceGap(
                gap_id="gap_001",
                description="No livestock losses found.",
                why_required="Needed for completeness.",
            )
        ],
        research_iterations=1,
    )
    markdown = format_human_markdown(output, metric=_metric(), section_number=16)
    assert markdown.startswith("# 16. Maize production change after drought")
    assert "**Description:**" in markdown
    assert "## Context" in markdown
    assert "## Answer" in markdown
    assert "## Gaps" in markdown
    assert "([Kenya Maize Report, p. 4](https://fao.org/kenya-maize.pdf))" in markdown
    assert "No livestock losses found." in markdown


def test_vector_claim_uses_filtered_country_scope_not_quote_text() -> None:
    quote = "Production declined by 18 percent during the season."
    hit = _hit(0, quote)
    state = ResearchState(
        metric=_metric(),
        country_iso3="KEN",
        country_name="Kenya",
        vector_chunks=[
            RetrievedChunk(
                source_id=f"vs:{hit.document_id}:{hit.chunk_index}",
                document_id=str(hit.document_id),
                chunk_index=hit.chunk_index,
                document_url=hit.document_url,
                document_title=hit.document_title,
                document_source=hit.document_source,
                chunk_text=hit.chunk_text,
                countries_iso3=hit.countries_iso3,
                retrieval_query="query",
                score=hit.score,
                page_number=1,
            )
        ],
    )
    accepted, rejected = _validate_claim_candidates(
        state,
        [_candidate(hit, quote, fit="direct_requested_unit")],
    )
    assert len(accepted) == 1
    assert not rejected


def test_web_claim_recovers_mangled_source_id_and_common_country_name() -> None:
    quote = "Cropland affected by drought reached 18 percent."
    content = (
        "Ethiopia drought assessment. "
        f"{quote} "
        "National cultivated area was used as the denominator."
    )
    state = ResearchState(
        metric=_metric(),
        country_iso3="ETH",
        country_name="Federal Democratic Republic of Ethiopia",
        web_sources=[
            WebSource(
                source_id="web:001",
                url="https://faolex.fao.org/docs/pdf/eth236072.pdf",
                title="Ethiopia report",
                content=content,
                query="Ethiopia cropland",
                access_date="2026-08-10",
            )
        ],
    )
    accepted, rejected = _validate_claim_candidates(
        state,
        [
            ExtractedClaimCandidate(
                # Mimic the live failure mode: extractor drops/mangles the id.
                source_id=(
                    "web:https://faolex.fao.org/docs/pdf/eth236072.pdf:deadbeef"
                ),
                quoted_text=quote,
                country="Ethiopia",
                relevance="Answers cropland impact.",
                answer_fit="direct_requested_unit",
                url="https://faolex.fao.org/docs/pdf/eth236072.pdf",
            )
        ],
    )
    assert not rejected
    assert len(accepted) == 1
    assert accepted[0].source_id == "web:001"
    assert accepted[0].source_type == "web"


def test_empty_batch_is_retried_with_exact_copy_instruction() -> None:
    quote = "Production declined by 18 percent during the season."
    hit = _hit(0, quote)
    chunk = RetrievedChunk(
        source_id=f"vs:{hit.document_id}:{hit.chunk_index}",
        document_id=str(hit.document_id),
        chunk_index=hit.chunk_index,
        document_url=hit.document_url,
        document_title=hit.document_title,
        document_source=hit.document_source,
        chunk_text=hit.chunk_text,
        countries_iso3=hit.countries_iso3,
        retrieval_query="query",
        score=hit.score,
        page_number=1,
    )
    state = ResearchState(
        metric=_metric(),
        country_iso3="KEN",
        country_name="Kenya",
        vector_chunks=[chunk],
    )
    model = ScriptedModel(
        {
            "ExtractedClaimList": [
                ExtractedClaimList(
                    claims=[
                        _candidate(
                            hit,
                            "Production increased by 18 percent.",
                            fit="direct_requested_unit",
                        )
                    ]
                ),
                ExtractedClaimList(
                    claims=[_candidate(hit, quote, fit="direct_requested_unit")]
                ),
            ]
        }
    )

    accepted = asyncio.run(
        _extract_and_validate_batches(
            state,
            model=model,  # type: ignore[arg-type]
            chunks=[chunk],
            web_sources=[],
            batch_size=5,
            max_claims_per_source=3,
        )
    )

    assert [claim.quoted_text for claim in accepted] == [quote]
    assert len(model.calls) == 2
    assert "character-for-character" in model.messages[1][1].content


def test_quantitative_direct_claims_rank_above_context() -> None:
    state = ResearchState(
        metric=_metric(),
        country_iso3="KEN",
        country_name="Kenya",
    )
    claims = [
        EvidenceClaim(
            claim_id="context",
            source_type="vectorstore",
            source_id="a",
            quoted_text="The drought affected rural livelihoods.",
            country="Kenya",
            relevance="context",
            answer_fit="supporting_context",
            url="a.pdf",
        ),
        EvidenceClaim(
            claim_id="direct",
            source_type="vectorstore",
            source_id="b",
            quoted_text="Maize production declined by 18 percent.",
            country="Kenya",
            relevance="direct result",
            answer_fit="direct_requested_unit",
            url="b.pdf",
        ),
    ]
    assert _select_claims(state, claims, limit=1)[0].claim_id == "direct"


def test_relevance_judge_separates_answer_context_and_rejection() -> None:
    state = ResearchState(
        metric=_metric(),
        country_iso3="KEN",
        country_name="Kenya",
    )
    claims = [
        EvidenceClaim(
            claim_id=f"claim_{index:03d}",
            source_type="vectorstore",
            source_id=str(index),
            quoted_text=text,
            country="Kenya",
            relevance="candidate",
            answer_fit="supporting_context",
            url="kenya.pdf",
        )
        for index, text in enumerate(
            [
                "An estimated 100,000 people were displaced.",
                "Average drought duration increased by 3 months.",
                "Maize production declined by 18 percent.",
                "Maize production declined severely.",
            ],
            start=1,
        )
    ]
    model = ScriptedModel(
        {
            "ClaimUsefulnessList": [
                ClaimUsefulnessList(
                    verdicts=[
                        ClaimUsefulnessVerdict(
                            claim_id="claim_001",
                            verdict="reject",
                            reason="Number concerns displaced people.",
                        ),
                        ClaimUsefulnessVerdict(
                            claim_id="claim_002",
                            verdict="context",
                            reason="Quantifies a related drought characteristic.",
                        ),
                        ClaimUsefulnessVerdict(
                            claim_id="claim_003",
                            verdict="direct_answer",
                            reason="Directly reports the requested change.",
                        ),
                        ClaimUsefulnessVerdict(
                            claim_id="claim_004",
                            verdict="direct_answer",
                            reason="Describes the requested change without a value.",
                        ),
                    ]
                )
            ]
        }
    )

    accepted = asyncio.run(
        _judge_claim_usefulness(state, claims, model=model)  # type: ignore[arg-type]
    )

    assert [claim.claim_id for claim in accepted] == [
        "claim_001",
        "claim_002",
        "claim_003",
        "claim_004",
    ]
    assert [claim.statement_type for claim in accepted] == [
        "context",
        "context",
        "answer",
        "answer",
    ]
    assert accepted[0].answer_fit == "quantitative_proxy"
    assert accepted[3].answer_fit == "direct_qualitative"


def test_related_unit_quantitative_metric_claim_is_direct_evidence() -> None:
    """Metric-subject quantities remain direct answers even in a related unit."""
    state = ResearchState(
        metric=_metric(),
        country_iso3="KEN",
        country_name="Kenya",
    )
    claims = [
        EvidenceClaim(
            claim_id="claim_001",
            source_type="vectorstore",
            source_id="1",
            quoted_text=(
                "some areas experiencing between 50 and 90 percent crop loss."
            ),
            country="Kenya",
            relevance="crop loss magnitude for the metric subject",
            answer_fit="direct_related_measure",
            url="kenya.pdf",
        ),
        EvidenceClaim(
            claim_id="claim_002",
            source_type="vectorstore",
            source_id="2",
            quoted_text="Average drought duration increased by 3 months.",
            country="Kenya",
            relevance="related hazard only",
            answer_fit="quantitative_proxy",
            url="kenya.pdf",
        ),
    ]
    model = ScriptedModel(
        {
            "ClaimUsefulnessList": [
                ClaimUsefulnessList(
                    verdicts=[
                        ClaimUsefulnessVerdict(
                            claim_id="claim_001",
                            verdict="context",
                            reason=(
                                "Reports crop loss percent rather than the "
                                "exact requested unit."
                            ),
                        ),
                        ClaimUsefulnessVerdict(
                            claim_id="claim_002",
                            verdict="context",
                            reason="Quantifies a related drought characteristic.",
                        ),
                    ]
                )
            ]
        }
    )

    accepted = asyncio.run(
        _judge_claim_usefulness(state, claims, model=model)  # type: ignore[arg-type]
    )

    assert [claim.claim_id for claim in accepted] == ["claim_001", "claim_002"]
    assert [claim.statement_type for claim in accepted] == ["answer", "context"]
    assert accepted[0].answer_fit == "direct_related_measure"
    assert is_direct_evidence_claim(accepted[0])
    assert not is_direct_evidence_claim(accepted[1])


def test_claim_with_only_unsupported_explicit_event_year_is_rejected() -> None:
    assert _has_only_ineligible_explicit_years("The 2014 map shows affected crops.")
    assert not _has_only_ineligible_explicit_years(
        "Published in 2025 using results from the 2023 El Nino event."
    )
    assert not _has_only_ineligible_explicit_years(
        "Floods destroyed 53 percent of cultivated area."
    )


def test_explicit_global_result_cannot_be_localized_by_country_metadata() -> None:
    state = ResearchState(
        metric=_metric(),
        country_iso3="SOM",
        country_name="Federal Republic of Somalia",
    )
    quote = "El Nino 1997/98 affected only four percent of the total agricultural area."
    assert _has_explicit_scope_conflict(
        state,
        quote,
        source_text=(
            "The cycles dominated by El Nino were associated with more area "
            "affected by drought at the global agricultural level. "
            f"{quote}"
        ),
    )
    # Aggregate-total wording is enough even when the nearby chunk omits "global".
    assert _has_explicit_scope_conflict(
        state,
        "affecting only four percent of the total agricultural area.",
        source_text=(
            "On the other hand, an El Nino year that takes place during La Nina "
            "dominance seems to have less impact on crop areas. This could "
            "explain why El Nino 1997/98 did not produce the impacts anticipated, "
            "affecting only four percent of the total agricultural area."
        ),
    )
    assert not _has_explicit_scope_conflict(
        state,
        "In Somalia, drought affected four percent of the total agricultural area.",
        source_text=(
            "In Somalia, drought affected four percent of the total agricultural area."
        ),
    )
    assert not _has_explicit_scope_conflict(
        state,
        "Production declined by 18 percent during the season.",
        source_text="Production declined by 18 percent during the season.",
    )


def test_research_keeps_global_scope_as_context_without_country_attribution() -> None:
    quote = (
        "This could explain why El Nino 1997/98 did not produce the impacts "
        "anticipated, affecting only four percent of the total agricultural area."
    )
    source_text = (
        "The cycles dominated by El Nino were associated with more area affected "
        "by drought at the global agricultural level. "
        f"{quote}"
    )
    hit = _hit(0, source_text, countries_iso3=["ETH"])
    bad_statement = (
        "During the 1997/98 El Nino, 4% of Ethiopia's total agricultural area "
        "was affected."
    )
    model = ScriptedModel(
        {
            "ExtractedClaimList": [
                ExtractedClaimList(
                    claims=[_candidate(hit, quote, fit="direct_requested_unit")]
                )
            ],
            "AnswerStatementList": [
                AnswerStatementList(
                    statements=[
                        AnswerStatement(
                            statement_id="stmt_001",
                            text=bad_statement,
                            statement_type="answer",
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
                    reasoning="Should be overridden by geography check.",
                )
            ],
        }
    )

    result = asyncio.run(
        research(
            metric=_metric(),
            country_iso3="ETH",
            vector_store=FakeVectorStore([hit]),
            config=_config(target_pdf_claims_per_metric=1),
            model=model,  # type: ignore[arg-type]
            verifier_model=model,  # type: ignore[arg-type]
            web_research_enabled=False,
            generate_research_queries_fn=_one_pdf_query,
        )
    )

    assert len(result.claims) == 1
    assert result.claims[0].statement_type == "context"
    assert result.claims[0].quoted_text == quote
    assert len(result.statements) == 1
    assert result.statements[0].statement_type == "context"
    assert "Ethiopia" not in result.statements[0].text
    assert "four percent of the total agricultural area" in result.statements[0].text
    assert result.status == "cannot_answer"
    assert "Ethiopia's total agricultural area" not in result.final_summary


def test_research_returns_twenty_country_filtered_pdf_claims() -> None:
    quotes = [
        f"Maize production declined by {index + 10} percent." for index in range(22)
    ]
    hits = [_hit(index, quote) for index, quote in enumerate(quotes)]
    candidates = [
        _candidate(hit, quote, fit="direct_requested_unit")
        for hit, quote in zip(hits, quotes, strict=True)
    ]
    model = ScriptedModel(
        {
            "ExtractedClaimList": [
                ExtractedClaimList(claims=candidates[:20]),
                ExtractedClaimList(claims=candidates[20:]),
            ],
            "ClaimUsefulnessList": [
                _usefulness_verdicts(20),
                _usefulness_verdicts(2, start=21),
            ],
            "AnswerStatementList": [AnswerStatementList()],
        }
    )
    store = FakeVectorStore(hits)

    result = asyncio.run(
        research(
            metric=_metric(),
            country_iso3="KEN",
            vector_store=store,
            config=_config(),
            model=model,  # type: ignore[arg-type]
            verifier_model=model,  # type: ignore[arg-type]
            web_research_enabled=False,
            generate_research_queries_fn=_one_pdf_query,
        )
    )

    assert len(result.claims) == 20
    assert len(result.statements) == 20
    assert all(claim.source_type == "vectorstore" for claim in result.claims)
    assert store.calls == [
        {
            "query": "Kenya El Nino maize production percentage loss table",
            "countries_iso3": ["KEN"],
            "limit": 20,
        }
    ]
    assert result.open_gaps == []
    assert result.status == "answered"
    assert len(result.query_runs) == 1
    assert result.query_runs[0].destination == "vectorstore"
    assert result.query_runs[0].query == (
        "Kenya El Nino maize production percentage loss table"
    )
    assert result.query_runs[0].results_returned == 20
    assert result.query_runs[0].results_accepted == 20


def test_context_is_retained_but_does_not_mark_metric_answered() -> None:
    quote = "Average drought duration increased by 3 months."
    hit = _hit(0, quote)
    model = ScriptedModel(
        {
            "ExtractedClaimList": [
                ExtractedClaimList(
                    claims=[_candidate(hit, quote, fit="supporting_context")]
                )
            ],
            "ClaimUsefulnessList": [
                ClaimUsefulnessList(
                    verdicts=[
                        ClaimUsefulnessVerdict(
                            claim_id="claim_001",
                            verdict="context",
                            reason="Relevant drought context, not production change.",
                        )
                    ]
                )
            ],
            "AnswerStatementList": [AnswerStatementList()],
        }
    )

    result = asyncio.run(
        research(
            metric=_metric(),
            country_iso3="KEN",
            vector_store=FakeVectorStore([hit]),
            config=_config(),
            model=model,  # type: ignore[arg-type]
            verifier_model=model,  # type: ignore[arg-type]
            web_research_enabled=False,
            generate_research_queries_fn=_one_pdf_query,
        )
    )

    assert result.status == "cannot_answer"
    assert len(result.statements) == 1
    assert result.statements[0].statement_type == "context"
    assert len(result.open_gaps) == 1
    assert "no evidence directly answers" in result.open_gaps[0].description


def test_research_requests_quantitative_pdf_queries_using_example_shape() -> None:
    captured: dict[str, Any] = {}

    async def capture_queries(**kwargs: Any) -> list[ResearchQuery]:
        captured.update(kwargs)
        return await _one_pdf_query()

    model = ScriptedModel({})
    asyncio.run(
        research(
            metric=_metric(),
            country_iso3="KEN",
            vector_store=FakeVectorStore([]),
            config=_config(max_pdf_queries_per_metric=5),
            model=model,  # type: ignore[arg-type]
            verifier_model=model,  # type: ignore[arg-type]
            web_research_enabled=False,
            generate_research_queries_fn=capture_queries,
        )
    )

    assert captured["country_iso3"] == "KEN"
    assert captured["preferred_destinations"] == ["vectorstore"]
    assert captured["min_queries"] == 3
    assert captured["max_queries"] == 5
    assert captured["example"] == _metric().example
    assert "unit" not in captured
    assert "Prioritize quantitative evidence" in captured["explanation"]
    assert "1997-98, 2015-16, 2018-19, 2023-24, and 2026-27" in captured["explanation"]


def test_web_runs_after_ten_pdf_claims_and_appends_five() -> None:
    pdf_quotes = [
        f"Maize production declined by {index + 10} percent." for index in range(10)
    ]
    hits = [_hit(index, quote) for index, quote in enumerate(pdf_quotes)]
    pdf_candidates = [
        _candidate(hit, quote, fit="direct_requested_unit")
        for hit, quote in zip(hits, pdf_quotes, strict=True)
    ]
    web_contents = [
        (
            "https://fao.org/web-1",
            (
                "Kenya evidence. Maize production declined by 10 percent. "
                "Maize production declined by 41 percent in another region. "
                "Maize yield declined by 12 percent."
            ),
        ),
        (
            "https://un.org/web-2",
            (
                "Kenya evidence. Crop losses reached 22 percent. "
                "The affected-area production loss was 35 percent. "
                "Yield declined by 16 percent."
            ),
        ),
    ]
    web_candidates: list[ExtractedClaimCandidate] = []
    web_quotes = [
        "Maize production declined by 10 percent.",
        "Maize production declined by 41 percent in another region.",
        "Maize yield declined by 12 percent.",
        "Crop losses reached 22 percent.",
        "The affected-area production loss was 35 percent.",
        "Yield declined by 16 percent.",
    ]
    for index, quote in enumerate(web_quotes):
        source_id = "web:001" if index < 3 else "web:002"
        web_candidates.append(
            ExtractedClaimCandidate(
                source_id=source_id,
                quoted_text=quote,
                country="Kenya",
                relevance="quantitative metric evidence",
                answer_fit="direct_related_measure",
            )
        )
    model = ScriptedModel(
        {
            "ExtractedClaimList": [
                ExtractedClaimList(claims=pdf_candidates),
                ExtractedClaimList(claims=web_candidates),
            ],
            "ClaimUsefulnessList": [
                _usefulness_verdicts(10),
                _usefulness_verdicts(6, start=11),
            ],
            "AnswerStatementList": [AnswerStatementList()],
        }
    )
    web_call: dict[str, Any] = {}

    async def fake_web_research(query: str, **kwargs: Any) -> Any:
        web_call["query"] = query
        web_call.update(kwargs)
        return SimpleNamespace(
            scraped=[
                SimpleNamespace(url=url, content=content, title=f"Web {index}")
                for index, (url, content) in enumerate(web_contents)
            ],
            queries=[
                SimpleNamespace(query=f"search {index}", num_results_returned=4)
                for index in range(5)
            ],
            snippet_only=[],
            scrape_failed=[],
            blocked_by_policy=[],
            source_http_error=[],
            bot_detected=[],
            scraped_irrelevant=[],
        )

    result = asyncio.run(
        research(
            metric=_metric(),
            country_iso3="KEN",
            vector_store=FakeVectorStore(hits),
            config=_config(),
            model=model,  # type: ignore[arg-type]
            verifier_model=model,  # type: ignore[arg-type]
            web_research_fn=fake_web_research,
            generate_research_queries_fn=_one_pdf_query,
        )
    )

    assert (
        len([claim for claim in result.claims if claim.source_type == "vectorstore"])
        == 10
    )
    assert len([claim for claim in result.claims if claim.source_type == "web"]) == 5
    assert web_call["research_depth"] == {
        "max_iterations": 2,
        "queries_first": 3,
        "queries_followup": 2,
        "urls_first": 3,
        "urls_followup": 2,
        "hub_deepening_cap": 5,
        "evaluator_extra_prompt": (
            "Prioritize numerical results that directly answer the metric: "
            "values, units, numerators, denominators, affected area or "
            "population, magnitude of change, geography, event and period."
        ),
    }
    assert "numerator, denominator" in web_call["query"]
    assert "FEWS NET" in web_call["domain_expertise"]
    assert "FSNAU" in web_call["domain_expertise"]
    assert result.research_iterations == 2
    assert len(result.statements) == 14  # one web claim corroborates a PDF finding
    assert any(len(statement.citations) == 2 for statement in result.statements)
    assert [run.destination for run in result.query_runs] == [
        "vectorstore",
        "web",
        "web",
        "web",
        "web",
        "web",
    ]
    assert result.query_runs[0].results_returned == 10
    assert result.query_runs[0].results_accepted == 10
    assert [run.query for run in result.query_runs[1:]] == [
        "search 0",
        "search 1",
        "search 2",
        "search 3",
        "search 4",
    ]
    assert all(run.results_returned == 4 for run in result.query_runs[1:])
    # Scrapes are not attributed per search query; accepted sources land on first.
    assert result.query_runs[1].results_accepted == 2
    assert all(run.results_accepted == 0 for run in result.query_runs[2:])


def test_failed_statement_verification_falls_back_to_exact_claim() -> None:
    quote = "Maize production declined by 18 percent."
    hit = _hit(0, quote)
    model = ScriptedModel(
        {
            "ExtractedClaimList": [
                ExtractedClaimList(
                    claims=[_candidate(hit, quote, fit="direct_requested_unit")]
                )
            ],
            "ClaimUsefulnessList": [_usefulness_verdicts(1)],
            "AnswerStatementList": [
                AnswerStatementList(
                    statements=[
                        AnswerStatement(
                            statement_id="draft",
                            text="Production collapsed because of El Nino.",
                            supporting_claim_ids=[f"claim_{1:03d}"],
                        )
                    ]
                )
            ],
        }
    )
    verifier = ScriptedModel(
        {
            "StatementVerification": [
                StatementVerification(
                    statement_id="stmt_001",
                    verdict="insufficient",
                    reasoning="Unsupported causation.",
                )
            ]
        }
    )

    result = asyncio.run(
        research(
            metric=_metric(),
            country_iso3="KEN",
            vector_store=FakeVectorStore([hit]),
            config=_config(target_pdf_claims_per_metric=1),
            model=model,  # type: ignore[arg-type]
            verifier_model=verifier,  # type: ignore[arg-type]
            web_research_enabled=False,
            generate_research_queries_fn=_one_pdf_query,
        )
    )

    assert result.statements[0].text == quote
    assert result.claims[0].quoted_text == quote


def test_answer_drafting_can_synthesize_multiple_compatible_claims() -> None:
    first = "Maize production declined by 18 percent."
    second = "Crop losses reached 24 percent in the eastern region."
    hit = _hit(0, f"{first} {second}")
    model = ScriptedModel(
        {
            "ExtractedClaimList": [
                ExtractedClaimList(
                    claims=[
                        _candidate(hit, first, fit="direct_requested_unit"),
                        _candidate(hit, second, fit="direct_related_measure"),
                    ]
                )
            ],
            "ClaimUsefulnessList": [_usefulness_verdicts(2)],
            "AnswerStatementList": [
                AnswerStatementList(
                    statements=[
                        AnswerStatement(
                            statement_id="draft",
                            text=(
                                "Maize production declined by 18 percent, while "
                                "crop losses reached 24 percent in the eastern region."
                            ),
                            supporting_claim_ids=["claim_001", "claim_002"],
                        )
                    ]
                )
            ],
        }
    )
    verifier = ScriptedModel(
        {
            "StatementVerification": [
                StatementVerification(
                    statement_id="stmt_001",
                    verdict="entailed",
                    reasoning="Both quantities and qualifiers are preserved.",
                )
            ]
        }
    )

    result = asyncio.run(
        research(
            metric=_metric(),
            country_iso3="KEN",
            vector_store=FakeVectorStore([hit]),
            config=_config(target_pdf_claims_per_metric=2),
            model=model,  # type: ignore[arg-type]
            verifier_model=verifier,  # type: ignore[arg-type]
            web_research_enabled=False,
            generate_research_queries_fn=_one_pdf_query,
        )
    )

    assert len(result.statements) == 1
    assert result.statements[0].supporting_claim_ids == ["claim_001", "claim_002"]
    assert "18 percent" in result.statements[0].text
    assert "24 percent" in result.statements[0].text


def test_no_evidence_returns_one_blocking_gap() -> None:
    model = ScriptedModel({})
    result = asyncio.run(
        research(
            metric=_metric(),
            country_iso3="KEN",
            vector_store=FakeVectorStore([]),
            config=_config(),
            model=model,  # type: ignore[arg-type]
            verifier_model=model,  # type: ignore[arg-type]
            web_research_enabled=False,
            generate_research_queries_fn=_one_pdf_query,
        )
    )
    assert result.status == "cannot_answer"
    assert len(result.open_gaps) == 1
    assert "No country-specific evidence" in result.open_gaps[0].description


def test_visual_evidence_is_loaded_verified_and_added_as_claim_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fao_impact_monitor.agent.researcher_agent as researcher_module

    image_path = tmp_path / "chart.png"
    image_bytes = b"test-png-bytes"
    image_path.write_bytes(image_bytes)
    artifact = VisualArtifact(
        artifact_id="region-1",
        path=str(image_path),
        sha256=hashlib.sha256(image_bytes).hexdigest(),
        media_type="image/png",
        physical_page=2,
    )
    chunk = RetrievedChunk(
        source_id="vs:507f1f77bcf86cd799439011:1",
        document_id="507f1f77bcf86cd799439011",
        chunk_index=1,
        document_url="somalia.pdf",
        chunk_text="[TARGET SOURCE EVIDENCE] Flood-risk context.",
        retrieval_query="flood risk",
        page_number=2,
        visual_artifacts=[artifact],
    )
    state = ResearchState(metric=_metric(), country_iso3="KEN", country_name="Kenya")
    insight_text = "The chart shows maize production declining by 12 percent."
    reader = ScriptedModel(
        {
            "VisualInsightList": [
                VisualInsightList(
                    insights=[
                        VisualInsightCandidate(
                            text=insight_text,
                            artifact_ids=["region-1"],
                            relevance="metric value",
                        )
                    ]
                )
            ]
        }
    )
    verifier = ScriptedModel(
        {
            "VisualInsightVerdictList": [
                VisualInsightVerdictList(
                    verdicts=[
                        VisualInsightVerdict(
                            insight_index=0,
                            verdict="entailed",
                            reasoning="Visible in the chart.",
                        )
                    ]
                )
            ]
        }
    )
    monkeypatch.setattr(
        researcher_module,
        "get_config",
        lambda: SimpleNamespace(pdf_pipeline=SimpleNamespace(artifact_dir=tmp_path)),
    )

    asyncio.run(
        _enrich_visual_chunk(
            state,
            chunk,
            model=reader,  # type: ignore[arg-type]
            verifier_model=verifier,  # type: ignore[arg-type]
            max_artifacts=1,
        )
    )

    assert chunk.research_visual_facts == [
        VerifiedResearchVisualFact(text=insight_text, artifact_ids=["region-1"])
    ]
    assert match_quoted_text(insight_text, chunk.chunk_text) == "exact"
    assert state.analyzed_visual_artifact_ids == {"region-1"}


def test_quantitative_status_classification() -> None:
    qualitative = AnswerStatement(statement_id="a", text="Flooding affected farms.")
    quantitative = AnswerStatement(
        statement_id="b",
        text="Flooding affected 20 percent of cropland.",
    )
    quantitative_context = AnswerStatement(
        statement_id="c",
        text="Average flood duration increased by 20 percent.",
        statement_type="context",
    )
    assert not statements_have_quantitative_evidence([qualitative])
    assert statements_have_quantitative_evidence([quantitative])
    assert classify_researcher_status([]) == "cannot_answer"
    assert classify_researcher_status([qualitative]) == "high_level_answer"
    assert classify_researcher_status([quantitative]) == "answered"
    assert classify_researcher_status([quantitative_context]) == "cannot_answer"
