from fao_impact_monitor.athena.agents.verification import (
    _flag,
    judgment_from_scores,
)


def test_uncertain_probability_stays_unresolved() -> None:
    assert _flag(0.91) is True
    assert _flag(0.1) is False
    assert _flag(0.5) is None


def test_scores_become_the_existing_verdict() -> None:
    judgment = judgment_from_scores(
        {"faithful": 0.97, "tentative": 0.04, "relation_supported": 0.55},
        study_design="observational",
        study_confidence=0.93,
    )
    assert judgment.faithful is True
    assert judgment.tentative is False
    assert judgment.relation_supported is None
    assert judgment.study_design == "observational"
    assert "faithful=0.97" in judgment.justification


def test_unstated_or_uncertain_design_is_omitted() -> None:
    missing = judgment_from_scores(
        {"faithful": 0.9, "tentative": 0.0},
        study_design="not stated",
        study_confidence=0.99,
    )
    shaky = judgment_from_scores(
        {"faithful": 0.9, "tentative": 0.0},
        study_design="experimental",
        study_confidence=0.4,
    )
    assert missing.study_design is None
    assert shaky.study_design is None
