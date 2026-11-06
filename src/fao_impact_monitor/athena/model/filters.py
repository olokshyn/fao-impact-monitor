"""Filter compilation for Mongo pre-filters, SPARQL, and in-memory checks."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from pydantic import BaseModel, Field, model_validator
from rdflib import Literal as RdfLiteral
from rdflib import URIRef

from fao_impact_monitor.athena.model.status import UnknownPolicy
from fao_impact_monitor.athena.ns import GEO_FIELDS, tag_predicate, term


class FilterExpr(BaseModel):
    """One inclusion, exclusion, AND, OR, or NOT node."""

    op: Literal["and", "or", "not", "tag"]
    children: list[FilterExpr] = Field(default_factory=list)
    field: str | None = None
    values: list[str] = Field(default_factory=list)
    comparison: Literal["include", "exclude"] = "include"
    unknown: UnknownPolicy = UnknownPolicy.EXCLUDE

    @model_validator(mode="after")
    def check_shape(self) -> FilterExpr:
        if self.op == "tag" and not self.field:
            raise ValueError("tag filter requires a field")
        if self.op == "not" and len(self.children) != 1:
            raise ValueError("not filter requires one child")
        if self.op in {"and", "or"} and not self.children:
            raise ValueError(f"{self.op} filter requires children")
        return self


class FilterSpec(BaseModel):
    """Filters applied before ranking and on every graph expansion."""

    root: FilterExpr | None = None
    strict: bool = True
    transferability: bool = False


def is_unknown(
    field: str, tags: Mapping[str, Sequence[str]], unknown: Sequence[str]
) -> bool:
    if field in unknown:
        return True
    return len(tags.get(field) or []) == 0


def matches(
    spec: FilterSpec,
    tags: Mapping[str, Sequence[str]],
    unknown: Sequence[str],
) -> bool:
    if spec.root is None:
        return True
    return _matches_expr(spec.root, tags, unknown)


def _matches_expr(
    expr: FilterExpr,
    tags: Mapping[str, Sequence[str]],
    unknown: Sequence[str],
) -> bool:
    if expr.op == "and":
        return all(_matches_expr(child, tags, unknown) for child in expr.children)
    if expr.op == "or":
        return any(_matches_expr(child, tags, unknown) for child in expr.children)
    if expr.op == "not":
        return not _matches_expr(expr.children[0], tags, unknown)
    return _tag_matches(expr, tags, unknown)


def _tag_matches(
    expr: FilterExpr,
    tags: Mapping[str, Sequence[str]],
    unknown: Sequence[str],
) -> bool:
    field = expr.field or ""
    unknown_field = is_unknown(field, tags, unknown)
    if expr.unknown == UnknownPolicy.ONLY:
        return unknown_field
    values = set(tags.get(field) or [])
    intersects = any(value in values for value in expr.values)
    if expr.comparison == "include":
        known_hit = intersects and not unknown_field
    else:
        known_hit = (not intersects) and not unknown_field
    if expr.unknown == UnknownPolicy.INCLUDE and unknown_field:
        return True
    if expr.unknown == UnknownPolicy.EXCLUDE and unknown_field:
        return False
    return known_hit


def compile_sparql(spec: FilterSpec, subject: str = "?s") -> str:
    """Return a SPARQL boolean expression. Empty filters are ``true``."""
    if spec.root is None:
        return "true"
    return _sparql_expr(spec.root, subject)


def _sparql_expr(expr: FilterExpr, subject: str) -> str:
    if expr.op == "and":
        parts = " && ".join(_sparql_expr(child, subject) for child in expr.children)
        return f"({parts})"
    if expr.op == "or":
        parts = " || ".join(_sparql_expr(child, subject) for child in expr.children)
        return f"({parts})"
    if expr.op == "not":
        return f"(!{_sparql_expr(expr.children[0], subject)})"
    return _sparql_tag(expr, subject)


def _sparql_tag(expr: FilterExpr, subject: str) -> str:
    field = expr.field or ""
    predicate = URIRef(tag_predicate(field)).n3()
    unknown_predicate = URIRef(term("unknownTag")).n3()
    value_terms = " || ".join(
        f"EXISTS {{ {subject} {predicate} {RdfLiteral(value).n3()} }}"
        for value in expr.values
    )
    unknown_hit = f"EXISTS {{ {subject} {unknown_predicate} {RdfLiteral(field).n3()} }}"
    if expr.comparison == "include":
        known = f"({value_terms})" if value_terms else "false"
    elif value_terms:
        known = f"(!({value_terms}) && !({unknown_hit}))"
    else:
        known = f"!({unknown_hit})"
    if expr.unknown == UnknownPolicy.ONLY:
        return f"({unknown_hit})"
    if expr.unknown == UnknownPolicy.INCLUDE:
        return f"({known} || {unknown_hit})"
    return f"({known} && !({unknown_hit}))"


def compile_mongo(spec: FilterSpec) -> dict[str, Any]:
    """MongoDB filter document applied inside vector and lexical search."""
    if spec.root is None:
        return {}
    return _mongo_expr(spec.root)


def _mongo_expr(expr: FilterExpr) -> dict[str, Any]:
    if expr.op == "and":
        return {"$and": [_mongo_expr(child) for child in expr.children]}
    if expr.op == "or":
        return {"$or": [_mongo_expr(child) for child in expr.children]}
    if expr.op == "not":
        return {"$nor": [_mongo_expr(expr.children[0])]}
    return _mongo_tag(expr)


def _mongo_tag(expr: FilterExpr) -> dict[str, Any]:
    field = expr.field or ""
    unknown: dict[str, Any] = {"unknown_tags": field}
    not_unknown: dict[str, Any] = {"unknown_tags": {"$nin": [field]}}
    if expr.comparison == "include":
        known: dict[str, Any] = (
            {"$and": [{field: {"$in": list(expr.values)}}, not_unknown]}
            if expr.values
            else {"_id": {"$exists": False}}
        )
    else:
        known = {
            "$and": [
                {field: {"$exists": True, "$nin": list(expr.values)}},
                not_unknown,
            ]
        }
    if expr.unknown == UnknownPolicy.ONLY:
        return unknown
    if expr.unknown == UnknownPolicy.INCLUDE:
        return {"$or": [known, unknown]}
    return {"$and": [known, not_unknown]}


def mongo_matches(document: Mapping[str, Any], query: Mapping[str, Any]) -> bool:
    """Evaluate the Mongo filter subset Athena compiles."""
    if not query:
        return True
    if "$and" in query:
        return all(mongo_matches(document, child) for child in query["$and"])
    if "$or" in query:
        return any(mongo_matches(document, child) for child in query["$or"])
    if "$nor" in query:
        return not any(mongo_matches(document, child) for child in query["$nor"])
    for key, expected in query.items():
        if not _field_matches(document.get(key), expected):
            return False
    return True


def _field_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        if "$in" in expected:
            return _contains_any(actual, expected["$in"])
        if "$nin" in expected:
            if "$exists" in expected and expected["$exists"] and actual is None:
                return False
            return not _contains_any(actual, expected["$nin"])
        if "$exists" in expected:
            return (actual is not None) is bool(expected["$exists"])
        return False
    if isinstance(actual, list):
        return expected in actual
    return cast(bool, actual == expected)


def _contains_any(actual: Any, values: Sequence[Any]) -> bool:
    if isinstance(actual, list):
        return any(value in actual for value in values)
    return actual in values


def flatten_tags(
    tags: Mapping[str, Sequence[str]], unknown: Sequence[str]
) -> dict[str, Any]:
    document: dict[str, Any] = {field: list(values) for field, values in tags.items()}
    document["unknown_tags"] = list(unknown)
    return document


@dataclass(frozen=True)
class Admission:
    allowed: bool
    extrapolation: bool


def broaden_spec(
    spec: FilterSpec,
    ancestors_of: Callable[[str], set[str]],
) -> FilterSpec:
    """Add ancestor places to inclusive geography filters."""
    if spec.root is None:
        return spec
    return spec.model_copy(
        update={
            "root": _broaden_expr(spec.root, ancestors_of),
            "transferability": False,
        }
    )


def _broaden_expr(
    expr: FilterExpr,
    ancestors_of: Callable[[str], set[str]],
) -> FilterExpr:
    if expr.op != "tag":
        return expr.model_copy(
            update={
                "children": [
                    _broaden_expr(child, ancestors_of) for child in expr.children
                ]
            }
        )
    if expr.field not in GEO_FIELDS or expr.comparison != "include":
        return expr
    extra: list[str] = []
    for value in expr.values:
        for ancestor in sorted(ancestors_of(value)):
            if ancestor not in expr.values and ancestor not in extra:
                extra.append(ancestor)
    if not extra:
        return expr
    return expr.model_copy(update={"values": [*expr.values, *extra]})


def admit(
    spec: FilterSpec,
    tags: Mapping[str, Sequence[str]],
    unknown: Sequence[str],
    ancestors_of: Callable[[str], set[str]],
) -> Admission:
    """Match a record. Transferability is explicit and labeled."""
    if matches(spec, tags, unknown):
        return Admission(True, False)
    if not spec.transferability:
        return Admission(False, False)
    broadened = broaden_spec(spec, ancestors_of)
    if matches(broadened, tags, unknown):
        return Admission(True, True)
    return Admission(False, False)


def without_fields(spec: FilterSpec, fields: set[str]) -> FilterSpec:
    """Drop tag constraints on ``fields``. A missing root matches everything."""
    if spec.root is None:
        return spec
    rewritten = _without_fields(spec.root, fields)
    return spec.model_copy(update={"root": rewritten})


def _without_fields(expr: FilterExpr, fields: set[str]) -> FilterExpr | None:
    if expr.op == "tag":
        if expr.field in fields:
            return None
        return expr
    if expr.op == "not":
        child = _without_fields(expr.children[0], fields)
        if child is None:
            return None
        return expr.model_copy(update={"children": [child]})
    rebuilt: list[FilterExpr] = []
    for child in expr.children:
        stripped = _without_fields(child, fields)
        if stripped is None:
            if expr.op == "or":
                return None
            continue
        rebuilt.append(stripped)
    if not rebuilt:
        return None
    if len(rebuilt) == 1:
        return rebuilt[0]
    return expr.model_copy(update={"children": rebuilt})
