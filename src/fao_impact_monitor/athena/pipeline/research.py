"""Research over admissible evidence. Excluded evidence never enters the pack."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from fao_impact_monitor.athena.agents.schemas import ResearchPlan, SubQuestion
from fao_impact_monitor.athena.context import AthenaContext
from fao_impact_monitor.athena.model.filters import FilterSpec, admit, without_fields
from fao_impact_monitor.athena.model.records import (
    CitedStatement,
    ClaimRecord,
    ContradictionView,
    EvidenceRecord,
    EvidenceView,
    RelationAssertionRecord,
    ResearchResult,
)
from fao_impact_monitor.athena.model.status import ScientificStatus
from fao_impact_monitor.athena.ns import (
    GRAPH_OPERATIONS,
    PROVENANCE_FIELDS,
    relation,
    term,
)
from fao_impact_monitor.athena.pipeline.discover import discover_for_claim
from fao_impact_monitor.athena.pipeline.ingest import RawClaim, ingest_raw_claim
from fao_impact_monitor.athena.pipeline.load import load_evidence
from fao_impact_monitor.athena.retrieval.admissibility import admissible_evidence
from fao_impact_monitor.athena.retrieval.candidates import retrieve_candidates
from fao_impact_monitor.athena.retrieval.relations import query_relations
from fao_impact_monitor.athena.storage.ontology import LocationHierarchy
from fao_impact_monitor.athena.storage.rdf import digest, n3_iri, n3_str


async def run_research(
    ctx: AthenaContext,
    question: str,
    spec: FilterSpec,
    *,
    rounds: int,
    external: bool,
) -> ResearchResult:
    gaps: list[str] = []
    if external and not ctx.external.enabled:
        gaps.append("External retrieval is disabled.")
    elif external and ctx.external.enabled:
        for hit in await ctx.external.search(question):
            await ingest_raw_claim(
                ctx,
                RawClaim(
                    wording=hit.text,
                    excerpt=hit.text,
                    tags={**hit.tags, "acquisition_channel": [hit.acquisition_channel]},
                ),
            )
    plan = await ctx.researcher.plan(question)
    strict_spec = spec.model_copy(update={"transferability": False})
    strict_claims, strict_assertions = await _explore(
        ctx, question, plan, strict_spec, rounds
    )
    hierarchy = LocationHierarchy.from_graph(ctx.graph)
    strict_view = _view(
        strict_claims,
        strict_assertions,
        load_evidence(ctx.graph),
        strict_spec,
        hierarchy.ancestors,
        gaps,
        plan.assumptions,
    )
    extrapolation: EvidenceView | None = None
    if spec.transferability:
        broad_claims, broad_assertions = await _explore(
            ctx, question, plan, spec, rounds
        )
        extrapolation = _view(
            broad_claims,
            broad_assertions,
            load_evidence(ctx.graph),
            spec,
            hierarchy.ancestors,
            gaps,
            plan.assumptions,
        )
    pack = strict_view.model_dump_json()
    synthesis = await ctx.researcher.synthesize(question, pack)
    if synthesis.quantitative_model and synthesis.uncertainty:
        strict_view.qualitative = False
        strict_view.quantitative_model = synthesis.quantitative_model
        strict_view.uncertainty = synthesis.uncertainty
    else:
        strict_view.qualitative = True
        strict_view.quantitative_model = None
        strict_view.uncertainty = None
    strict_view.assumptions = list(
        dict.fromkeys([*strict_view.assumptions, *synthesis.assumptions])
    )
    strict_view.gaps = list(dict.fromkeys([*strict_view.gaps, *synthesis.gaps]))
    _drop_indirect_causes(strict_view, strict_assertions)
    result = ResearchResult(
        question=question,
        strict=strict_view,
        extrapolation=extrapolation,
    )
    _write_cache(
        ctx,
        question,
        spec,
        rounds,
        result,
        [item.claim_iri for item in strict_view.statements],
    )
    return result


async def _explore(
    ctx: AthenaContext,
    question: str,
    plan: ResearchPlan,
    spec: FilterSpec,
    rounds: int,
) -> tuple[dict[str, ClaimRecord], dict[str, RelationAssertionRecord]]:
    hierarchy = LocationHierarchy.from_graph(ctx.graph)
    evidence = load_evidence(ctx.graph)
    found: dict[str, ClaimRecord] = {}
    assertions: dict[str, RelationAssertionRecord] = {}
    questions = plan.subquestions or [SubQuestion(text=question)]
    for sub in questions:
        for _round in range(rounds):
            claims = await retrieve_candidates(
                ctx.graph,
                ctx.index,
                ctx.embedder,
                query=sub.text,
                entities=sub.variables,
                spec=spec,
                evidence=evidence,
                ancestors_of=hierarchy.ancestors,
                limit=ctx.config.candidate_limit,
            )
            for claim in claims:
                found[claim.iri] = claim
            if len(found) >= 2:
                seed = next(iter(found))
                await discover_for_claim(
                    ctx, seed, spec=spec, extra=list(found.values())
                )
            evidence = load_evidence(ctx.graph)
            known = dict(found)
            for claim in list(found.values()):
                for parent in ("causes", "correlatesWith", "supports", "contradicts"):
                    views = query_relations(
                        ctx.graph,
                        claim.iri,
                        parent,
                        "both",
                        spec,
                        hierarchy.ancestors,
                        known,
                        evidence,
                    )
                    for view in views:
                        if view.extrapolation and not spec.transferability:
                            continue
                        assertions[view.assertion.iri] = view.assertion
                        for endpoint in (
                            view.assertion.subject_iri,
                            view.assertion.object_iri,
                        ):
                            if endpoint in known:
                                found[endpoint] = known[endpoint]
    return found, assertions


def _view(
    claims: dict[str, ClaimRecord],
    assertions: dict[str, RelationAssertionRecord],
    evidence: dict[str, EvidenceRecord],
    spec: FilterSpec,
    ancestors_of: Callable[[str], set[str]],
    gaps: list[str],
    assumptions: list[str],
) -> EvidenceView:
    statements: list[CitedStatement] = []
    for claim in claims.values():
        kept = admissible_evidence(claim.evidence_iris, evidence, spec, ancestors_of)
        if not kept:
            continue
        decision = admit(
            without_fields(spec, set(PROVENANCE_FIELDS)),
            claim.tags,
            claim.unknown_tags,
            ancestors_of,
        )
        statements.append(
            CitedStatement(
                text=claim.normalized,
                claim_iri=claim.iri,
                evidence_iris=[item.iri for item in kept],
                scientific_status=claim.scientific_status.value,
                extrapolation=decision.extrapolation,
            )
        )
    contradictions: list[ContradictionView] = []
    for assertion in assertions.values():
        if not is_contradiction(assertion):
            continue
        kept = admissible_evidence(
            assertion.evidence_iris, evidence, spec, ancestors_of
        )
        contradictions.append(
            ContradictionView(
                assertion_iri=assertion.iri,
                summary=assertion.relation_iri.rsplit("/", 1)[-1],
                claim_iris=[assertion.subject_iri, assertion.object_iri],
                evidence_iris=[item.iri for item in kept],
            )
        )
    mode: Literal["strict", "extrapolation"] = (
        "extrapolation" if spec.transferability else "strict"
    )
    return EvidenceView(
        mode=mode,
        statements=statements,
        contradictions=contradictions,
        gaps=list(gaps),
        assumptions=list(assumptions),
        qualitative=True,
    )


def is_contradiction(assertion: RelationAssertionRecord) -> bool:
    local = assertion.relation_iri.rsplit("/", 1)[-1]
    return local in {"contradicts", "oppositeEffectDirection", "incompatibleEstimate"}


def _drop_indirect_causes(
    view: EvidenceView,
    assertions: dict[str, RelationAssertionRecord],
) -> None:
    """A multi-hop chain is not reported as a direct causal finding."""
    direct: set[str] = set()
    for assertion in assertions.values():
        if assertion.scientific_status != ScientificStatus.SOURCE_REPORTED:
            continue
        if assertion.derivation is not None:
            continue
        local = assertion.relation_iri.rsplit("/", 1)[-1]
        if local in {"causes", "increases", "decreases", "yieldLoss"}:
            direct.add(assertion.subject_iri)
            direct.add(assertion.object_iri)
    for statement in view.statements:
        if statement.scientific_status == ScientificStatus.DERIVED.value:
            statement.relation_iri = relation("causes")
        if statement.claim_iri not in direct:
            statement.relation_iri = None


def _write_cache(
    ctx: AthenaContext,
    question: str,
    spec: FilterSpec,
    rounds: int,
    result: ResearchResult,
    depends_on: list[str],
) -> None:
    key = digest(question, spec.model_dump_json(), str(rounds))
    iri = term(f"cache/{key}")
    payload = result.model_dump_json()
    depends = "\n".join(
        f"{n3_iri(iri)} {n3_iri(term('dependsOn'))} {n3_iri(claim)} ."
        for claim in depends_on
    )
    ctx.graph.update(
        f"""
        DELETE {{ GRAPH {n3_iri(GRAPH_OPERATIONS)} {{ {n3_iri(iri)} ?p ?o }} }}
        WHERE {{ GRAPH {n3_iri(GRAPH_OPERATIONS)} {{ {n3_iri(iri)} ?p ?o }} }};
        INSERT DATA {{
          GRAPH {n3_iri(GRAPH_OPERATIONS)} {{
            {n3_iri(iri)} a {n3_iri(term("CacheEntry"))} ;
                {n3_iri(term("cacheKey"))} {n3_str(key)} ;
                {n3_iri(term("recordJson"))} {n3_str(payload)} .
            {depends}
          }}
        }}
        """
    )
