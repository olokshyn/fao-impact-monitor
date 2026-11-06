"""Athena IRIs and named graphs."""

from __future__ import annotations

NS = "https://fao.org/athena/"
GRAPH_ACCEPTED = f"{NS}graph/accepted"
GRAPH_CANDIDATES = f"{NS}graph/candidates"
GRAPH_ONTOLOGY = f"{NS}graph/ontology"
GRAPH_OPERATIONS = f"{NS}graph/operations"

RDFS_SUB_PROPERTY = "http://www.w3.org/2000/01/rdf-schema#subPropertyOf"
RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
OWL_OBJECT_PROPERTY = "http://www.w3.org/2002/07/owl#ObjectProperty"

TAG_FIELDS: tuple[str, ...] = (
    "country",
    "region",
    "broader_region",
    "source_type",
    "publishing_group",
    "publisher",
    "journal",
    "acquisition_channel",
)
GEO_FIELDS: frozenset[str] = frozenset({"country", "region", "broader_region"})
PROVENANCE_FIELDS: frozenset[str] = frozenset(
    {
        "source_type",
        "publishing_group",
        "publisher",
        "journal",
        "acquisition_channel",
    }
)


def term(local: str) -> str:
    return f"{NS}{local}"


def relation(local: str) -> str:
    return f"{NS}relation/{local}"


def tag_predicate(field: str) -> str:
    return f"{NS}tag/{field}"


def location(code: str) -> str:
    return f"{NS}loc/{code}"
