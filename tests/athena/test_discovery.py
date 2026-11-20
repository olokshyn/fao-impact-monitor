from fao_impact_monitor.athena.agents.discovery import proposals_from_choices
from fao_impact_monitor.athena.agents.schemas import DiscoveryRequest


def _request() -> DiscoveryRequest:
    return DiscoveryRequest(
        claim_text="focus\nRain failed.",
        candidates=["other\nYield fell."],
        existing_relations=["causes", "correlates with"],
    )


def test_confident_forward_relation_is_kept() -> None:
    result = proposals_from_choices(_request(), [("causes [forward]", 0.91)])
    assert len(result.relations) == 1
    relation = result.relations[0]
    assert relation.subject_iri == "focus"
    assert relation.object_iri == "other"
    assert relation.relation == "causes"


def test_uncertain_or_empty_relation_is_dropped() -> None:
    request = _request()
    assert proposals_from_choices(request, [("causes [reverse]", 0.4)]).relations == []
    assert proposals_from_choices(request, [("none", 0.99)]).relations == []
