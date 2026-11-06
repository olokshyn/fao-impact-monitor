"""Read the versioned tag registry, relation hierarchy, and location hierarchy."""

from __future__ import annotations

from collections import defaultdict

from fao_impact_monitor.athena.model.records import RelationDefinition, TagField
from fao_impact_monitor.athena.ns import (
    GRAPH_ONTOLOGY,
    RDFS_LABEL,
    RDFS_SUB_PROPERTY,
    term,
)
from fao_impact_monitor.athena.storage.graph import GraphStore
from fao_impact_monitor.athena.storage.rdf import n3_iri


class LocationHierarchy:
    """Explicit broader-place links. Transferability uses ancestors only."""

    def __init__(self, broader: dict[str, set[str]]) -> None:
        self._broader = broader

    def ancestors(self, code: str) -> set[str]:
        found: set[str] = set()
        stack = list(self._broader.get(code, set()))
        while stack:
            current = stack.pop()
            if current in found:
                continue
            found.add(current)
            stack.extend(self._broader.get(current, set()))
        return found

    @classmethod
    def from_graph(cls, store: GraphStore) -> LocationHierarchy:
        rows = store.query(
            f"""
            SELECT ?code ?parent WHERE {{
              GRAPH {n3_iri(GRAPH_ONTOLOGY)} {{
                ?place a {n3_iri(term("Location"))} ;
                       {n3_iri(term("code"))} ?code ;
                       {n3_iri(term("broader"))} ?broader .
                ?broader {n3_iri(term("code"))} ?parent .
              }}
            }}
            """
        )
        broader: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            broader[row["code"]].add(row["parent"])
        return cls(broader)


def tag_registry(store: GraphStore) -> dict[str, TagField]:
    rows = store.query(
        f"""
        SELECT ?name ?valueType ?multiplicity ?extraction ?inheritance ?inherits ?vocab WHERE {{
          GRAPH {n3_iri(GRAPH_ONTOLOGY)} {{
            ?field a {n3_iri(term("TagField"))} ;
                   {n3_iri(term("fieldName"))} ?name ;
                   {n3_iri(term("valueType"))} ?valueType ;
                   {n3_iri(term("multiplicity"))} ?multiplicity ;
                   {n3_iri(term("extractionRule"))} ?extraction ;
                   {n3_iri(term("inheritanceRule"))} ?inheritance ;
                   {n3_iri(term("inherits"))} ?inherits .
            OPTIONAL {{ ?field {n3_iri(term("vocabulary"))} ?vocab }}
          }}
        }}
        """
    )
    grouped: dict[str, TagField] = {}
    vocab: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row["vocab"] and row["vocab"] not in vocab[row["name"]]:
            vocab[row["name"]].append(row["vocab"])
        grouped[row["name"]] = TagField(
            name=row["name"],
            value_type=row["valueType"],
            multiplicity="many" if row["multiplicity"] == "many" else "one",
            vocabulary=vocab[row["name"]],
            extraction_rule=row["extraction"],
            inheritance_rule=row["inheritance"],
            inherits=row["inherits"] == "true",
        )
    for field in grouped.values():
        field.vocabulary = vocab[field.name]
    return grouped


def relation_definitions(store: GraphStore) -> dict[str, RelationDefinition]:
    rows = store.query(
        f"""
        SELECT ?rel ?label ?definition ?parent ?direction ?subject ?object ?version ?example ?alias
        WHERE {{
          GRAPH {n3_iri(GRAPH_ONTOLOGY)} {{
            ?rel a {n3_iri("http://www.w3.org/2002/07/owl#ObjectProperty")} ;
                 {n3_iri(RDFS_LABEL)} ?label ;
                 {n3_iri(term("definition"))} ?definition ;
                 {n3_iri(term("directionality"))} ?direction ;
                 {n3_iri(term("subjectType"))} ?subject ;
                 {n3_iri(term("objectType"))} ?object ;
                 {n3_iri(term("version"))} ?version .
            OPTIONAL {{ ?rel {n3_iri(RDFS_SUB_PROPERTY)} ?parent }}
            OPTIONAL {{ ?rel {n3_iri(term("example"))} ?example }}
            OPTIONAL {{ ?rel {n3_iri(term("alias"))} ?alias }}
          }}
        }}
        """
    )
    found: dict[str, RelationDefinition] = {}
    for row in rows:
        current = found.get(row["rel"])
        if current is None:
            current = RelationDefinition(
                iri=row["rel"],
                label=row["label"],
                definition=row["definition"],
                subject_type=row["subject"],
                object_type=row["object"],
                directionality=row["direction"],
                version=row["version"],
            )
            found[row["rel"]] = current
        if row["parent"] and row["parent"] not in current.parents:
            current.parents.append(row["parent"])
        if row["example"] and row["example"] not in current.examples:
            current.examples.append(row["example"])
        if row["alias"] and row["alias"] not in current.aliases:
            current.aliases.append(row["alias"])
    return found


def is_descendant(store: GraphStore, relation: str, ancestor: str) -> bool:
    return store.ask(
        f"""
        ASK {{
          GRAPH {n3_iri(GRAPH_ONTOLOGY)} {{
            {n3_iri(relation)} {n3_iri(RDFS_SUB_PROPERTY)}* {n3_iri(ancestor)}
          }}
        }}
        """
    )


def find_relation(store: GraphStore, name_or_alias: str) -> RelationDefinition | None:
    needle = name_or_alias.strip().lower()
    for definition in relation_definitions(store).values():
        names = {definition.label.lower(), definition.iri.rsplit("/", 1)[-1].lower()}
        names.update(alias.lower() for alias in definition.aliases)
        if needle in names:
            return definition
    return None
