from __future__ import annotations

import pytest

from fao_impact_monitor.athena.config import get_athena_config
from fao_impact_monitor.athena.ns import relation
from fao_impact_monitor.athena.storage.graph import FusekiGraph, ensure_ontology
from fao_impact_monitor.athena.storage.ontology import is_descendant


@pytest.mark.integration
def test_fuseki_accepts_the_seed_ontology() -> None:
    settings = get_athena_config()
    graph = FusekiGraph(settings.fuseki_query_url, settings.fuseki_update_url)
    ensure_ontology(graph)
    assert is_descendant(graph, relation("yieldLoss"), relation("causes"))
