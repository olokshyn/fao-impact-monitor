"""Graph store protocol, in-memory SPARQL, and Fuseki."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, cast

import httpx
from rdflib import Dataset, Graph, URIRef
from rdflib.query import Result

from fao_impact_monitor.athena.ns import GRAPH_ONTOLOGY, term

SEED_PATH = Path(__file__).resolve().parents[1] / "ontology" / "seed.ttl"


class GraphError(RuntimeError):
    """Raised when a SPARQL request fails."""


class GraphStore(Protocol):
    def update(self, sparql: str) -> None: ...

    def query(self, sparql: str) -> list[dict[str, str]]: ...

    def ask(self, sparql: str) -> bool: ...


def seed_turtle() -> str:
    return SEED_PATH.read_text(encoding="utf-8")


def turtle_to_insert(graph_iri: str, turtle: str) -> str:
    parsed = Graph()
    parsed.parse(data=turtle, format="turtle")
    lines = [
        f"{subject.n3()} {predicate.n3()} {obj.n3()} ."
        for subject, predicate, obj in parsed
    ]
    body = "\n".join(lines)
    return f"INSERT DATA {{ GRAPH {URIRef(graph_iri).n3()} {{\n{body}\n}} }}"


def ensure_ontology(store: GraphStore) -> None:
    registry = URIRef(term("tagRegistry/v1")).n3()
    exists = store.ask(
        "ASK { GRAPH "
        + URIRef(GRAPH_ONTOLOGY).n3()
        + " { "
        + registry
        + " a "
        + URIRef(term("TagRegistry")).n3()
        + " } }"
    )
    if not exists:
        store.update(turtle_to_insert(GRAPH_ONTOLOGY, seed_turtle()))


def _rows(result: Result) -> list[dict[str, str]]:
    variables = [str(variable) for variable in (result.vars or [])]
    rows: list[dict[str, str]] = []
    for row in result:
        item: dict[str, str] = {}
        raw = cast(Any, row)
        for index, name in enumerate(variables):
            cell = raw[name] if name in raw else raw[index]
            item[name] = "" if cell is None else str(cell)
        rows.append(item)
    return rows


class MemoryGraph:
    """rdflib named-graph store used by unit tests and local runs."""

    def __init__(self) -> None:
        self._dataset = Dataset()

    def update(self, sparql: str) -> None:
        self._dataset.update(sparql)

    def query(self, sparql: str) -> list[dict[str, str]]:
        return _rows(self._dataset.query(sparql))

    def ask(self, sparql: str) -> bool:
        result = self._dataset.query(sparql)
        return bool(result.askAnswer)


class FusekiGraph:
    """SPARQL client for Jena Fuseki. One update request is one transaction."""

    def __init__(
        self,
        query_url: str,
        update_url: str,
        client: httpx.Client | None = None,
    ) -> None:
        self._query_url = query_url
        self._update_url = update_url
        self._client = client or httpx.Client(timeout=60.0)

    def update(self, sparql: str) -> None:
        response = self._client.post(
            self._update_url,
            content=sparql.encode("utf-8"),
            headers={"Content-Type": "application/sparql-update"},
        )
        if response.status_code >= 400:
            raise GraphError(
                f"Fuseki update failed ({response.status_code}): {response.text}"
            )

    def query(self, sparql: str) -> list[dict[str, str]]:
        payload = self._post_query(sparql)
        head = payload.get("head", {})
        variables = (
            [str(name) for name in head.get("vars", [])]
            if isinstance(head, dict)
            else []
        )
        rows: list[dict[str, str]] = []
        results = payload.get("results", {})
        bindings = results.get("bindings", []) if isinstance(results, dict) else []
        for binding in bindings:
            if not isinstance(binding, dict):
                continue
            row: dict[str, str] = {}
            for name in variables:
                cell = binding.get(name)
                if isinstance(cell, dict) and "value" in cell:
                    row[name] = str(cell["value"])
                else:
                    row[name] = ""
            rows.append(row)
        return rows

    def ask(self, sparql: str) -> bool:
        payload = self._post_query(sparql)
        return bool(payload.get("boolean", False))

    def probe(self) -> None:
        self.ask("ASK { ?s ?p ?o }")

    def _post_query(self, sparql: str) -> dict[str, object]:
        response = self._client.post(
            self._query_url,
            content=sparql.encode("utf-8"),
            headers={
                "Content-Type": "application/sparql-query",
                "Accept": "application/sparql-results+json",
            },
        )
        if response.status_code >= 400:
            raise GraphError(
                f"Fuseki query failed ({response.status_code}): {response.text}"
            )
        data: dict[str, object] = response.json()
        return data
