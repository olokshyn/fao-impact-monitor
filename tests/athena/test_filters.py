from __future__ import annotations

from fao_impact_monitor.athena.model.filters import (
    FilterExpr,
    FilterSpec,
    compile_mongo,
    compile_sparql,
    flatten_tags,
    matches,
    mongo_matches,
    without_fields,
)
from fao_impact_monitor.athena.model.status import UnknownPolicy
from fao_impact_monitor.athena.ns import PROVENANCE_FIELDS, tag_predicate
from fao_impact_monitor.athena.storage.graph import MemoryGraph
from fao_impact_monitor.athena.storage.rdf import n3_iri, n3_str


def _country(
    value: str, *, unknown: UnknownPolicy = UnknownPolicy.EXCLUDE
) -> FilterSpec:
    return FilterSpec(
        root=FilterExpr(
            op="tag",
            field="country",
            values=[value],
            comparison="include",
            unknown=unknown,
        )
    )


def test_unknown_country_is_excluded_unless_requested() -> None:
    known = {"country": ["KEN"]}
    unknown: dict[str, list[str]] = {}
    spec = _country("KEN")
    assert matches(spec, known, [])
    assert not matches(spec, unknown, ["country"])
    include_unknown = _country("KEN", unknown=UnknownPolicy.INCLUDE)
    assert matches(include_unknown, unknown, ["country"])
    only_unknown = spec.model_copy(
        update={
            "root": spec.root.model_copy(update={"unknown": UnknownPolicy.ONLY})  # type: ignore[union-attr]
        }
    )
    assert matches(only_unknown, unknown, ["country"])
    assert not matches(only_unknown, known, [])


def test_and_or_and_exclusion() -> None:
    tags = {"country": ["KEN"], "source_type": ["peer-reviewed"]}
    spec = FilterSpec(
        root=FilterExpr(
            op="and",
            children=[
                FilterExpr(op="tag", field="country", values=["KEN"]),
                FilterExpr(
                    op="tag",
                    field="source_type",
                    values=["web-page"],
                    comparison="exclude",
                ),
            ],
        )
    )
    assert matches(spec, tags, [])
    web = {"country": ["KEN"], "source_type": ["web-page"]}
    assert not matches(spec, web, [])
    either = FilterSpec(
        root=FilterExpr(
            op="or",
            children=[
                FilterExpr(op="tag", field="country", values=["KEN"]),
                FilterExpr(op="tag", field="country", values=["ETH"]),
            ],
        )
    )
    assert matches(either, {"country": ["ETH"]}, [])
    assert not matches(either, {"country": ["UGA"]}, [])


def test_mongo_and_sparql_agree_with_matcher() -> None:
    tags = {"country": ["KEN"], "source_type": ["peer-reviewed"]}
    unknown = ["region"]
    spec = FilterSpec(
        root=FilterExpr(
            op="and",
            children=[
                FilterExpr(op="tag", field="country", values=["KEN"]),
                FilterExpr(
                    op="not",
                    children=[FilterExpr(op="tag", field="region", values=["x"])],
                ),
            ],
        )
    )
    document = flatten_tags(tags, unknown)
    assert mongo_matches(document, compile_mongo(spec)) is matches(spec, tags, unknown)
    graph = MemoryGraph()
    triples = " ".join(
        f"{n3_iri('http://example.org/c')} {n3_iri(tag_predicate(field))} {n3_str(value)} ."
        for field, values in tags.items()
        for value in values
    )
    triples += (
        f" {n3_iri('http://example.org/c')} {n3_iri('https://fao.org/athena/unknownTag')} "
        f"{n3_str('region')} ."
    )
    graph.update("INSERT DATA { GRAPH <http://example.org/g> { " + triples + " } }")
    expression = compile_sparql(spec, "?s")
    found = graph.ask(
        "ASK { GRAPH <http://example.org/g> { ?s ?p ?o FILTER(" + expression + ") } }"
    )
    assert found is matches(spec, tags, unknown)


def test_provenance_filters_can_be_removed_for_claim_applicability() -> None:
    spec = FilterSpec(
        root=FilterExpr(
            op="and",
            children=[
                FilterExpr(op="tag", field="country", values=["KEN"]),
                FilterExpr(
                    op="tag",
                    field="source_type",
                    values=["web-page"],
                    comparison="exclude",
                ),
            ],
        )
    )
    applicability = without_fields(spec, set(PROVENANCE_FIELDS))
    assert matches(applicability, {"country": ["KEN"], "source_type": ["web-page"]}, [])
    assert not matches(spec, {"country": ["KEN"], "source_type": ["web-page"]}, [])
