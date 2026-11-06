"""Load authoritative records out of named graphs."""

from __future__ import annotations

import json

from pydantic import BaseModel

from fao_impact_monitor.athena.model.records import (
    ClaimRecord,
    EvidenceRecord,
    RelationAssertionRecord,
)
from fao_impact_monitor.athena.ns import term
from fao_impact_monitor.athena.storage.graph import GraphStore
from fao_impact_monitor.athena.storage.rdf import n3_iri


def load_models[T: BaseModel](
    store: GraphStore, rdf_type: str, model: type[T]
) -> dict[str, T]:
    rows = store.query(
        f"""
        SELECT ?json WHERE {{
          GRAPH ?g {{
            ?s a {n3_iri(rdf_type)} ;
               {n3_iri(term("recordJson"))} ?json .
          }}
        }}
        """
    )
    loaded: dict[str, T] = {}
    for row in rows:
        payload = json.loads(row["json"])
        record = model.model_validate(payload)
        loaded[str(payload["iri"])] = record
    return loaded


def load_claims(store: GraphStore) -> dict[str, ClaimRecord]:
    return load_models(store, term("Claim"), ClaimRecord)


def load_evidence(store: GraphStore) -> dict[str, EvidenceRecord]:
    return load_models(store, term("Evidence"), EvidenceRecord)


def load_assertions(store: GraphStore) -> dict[str, RelationAssertionRecord]:
    return load_models(store, term("RelationAssertion"), RelationAssertionRecord)
