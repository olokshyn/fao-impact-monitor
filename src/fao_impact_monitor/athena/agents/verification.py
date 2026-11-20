"""Evidence checks as Jev judgments. Citation matching stays in code."""

from __future__ import annotations

from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul

from fao_impact_monitor.athena.agents.schemas import (
    VerificationJudgment,
    VerificationPacket,
)
from fao_impact_monitor.athena.observe import (
    DecisionLog,
    DecisionRecord,
    input_hash,
)
from fao_impact_monitor.athena.storage.rdf import utc_stamp

# A Noul near 0.5 is unresolved, so the existing review status still applies.
_YES = 0.8
_NO = 0.2

_STUDY_DESIGNS = {
    "not stated": "The excerpt does not name a study design.",
    "observational": "The excerpt describes observation without an assigned intervention.",
    "experimental": "The excerpt describes an intervention assigned by the study.",
    "model or simulation": "The excerpt describes a model, simulation, or scenario.",
    "assessment or survey": "The excerpt describes an assessment, census, or survey.",
    "review or synthesis": "The excerpt describes a review or synthesis of other studies.",
}


def _flag(probability: float) -> bool | None:
    if probability >= _YES:
        return True
    if probability <= _NO:
        return False
    return None


def judgment_from_scores(
    scores: dict[str, float],
    *,
    study_design: str | None,
    study_confidence: float,
) -> VerificationJudgment:
    """Turn Jev probabilities into the verdict the pipeline already combines."""
    design = study_design
    if design == "not stated" or study_confidence < _YES:
        design = None
    parts = [f"{name}={score:.2f}" for name, score in scores.items()]
    if design is not None:
        parts.append(f"study_design={design}")
    return VerificationJudgment(
        faithful=_flag(scores["faithful"]) if "faithful" in scores else None,
        relation_supported=(
            _flag(scores["relation_supported"])
            if "relation_supported" in scores
            else None
        ),
        context_compatible=(
            _flag(scores["context_compatible"])
            if "context_compatible" in scores
            else None
        ),
        derivation_valid=(
            _flag(scores["derivation_valid"]) if "derivation_valid" in scores else None
        ),
        expresses_causality=(
            _flag(scores["expresses_causality"])
            if "expresses_causality" in scores
            else None
        ),
        tentative=_flag(scores.get("tentative", 0.0)) is True,
        study_design=design,
        justification="; ".join(parts),
    )


def _questions(packet: VerificationPacket) -> dict[str, Noul | Choice]:
    questions: dict[str, Noul | Choice] = {
        "faithful": Noul(
            instructions=(
                "Does `excerpt` state `claim`, including its quantities, units, "
                "negation, and scope? A wording that changes any of those is no."
            )
        ),
        "tentative": Noul(
            instructions=(
                "Does `excerpt` present `claim` as tentative, provisional, or uncertain?"
            )
        ),
        "study_design": Choice(
            instructions="What study design does `excerpt` state for `claim`?",
            criteria=_STUDY_DESIGNS,
        ),
    }
    if packet.relation is not None:
        questions["relation_supported"] = Noul(
            instructions=(
                "Does `excerpt` state `relation` in that direction? "
                "Correlation, co-occurrence, and similarity are not support for it."
            )
        )
        questions["context_compatible"] = Noul(
            instructions=(
                "Do the population, place, time, and conditions in `excerpt` "
                "match `claim`?"
            )
        )
        questions["expresses_causality"] = Noul(
            instructions=(
                "Does `excerpt` state a cause, rather than a correlation "
                "or a co-occurrence?"
            )
        )
    if packet.derivation_rule is not None or packet.premises or packet.assumptions:
        questions["derivation_valid"] = Noul(
            instructions=(
                "Does `excerpt` identify every premise in `premises`, the rule "
                "`derivation_rule`, and every assumption in `assumptions`?"
            )
        )
    return questions


class SdkEvidenceVerifier:
    def __init__(self, model_name: str, decisions: DecisionLog) -> None:
        self._model = model_name
        self._decisions = decisions
        self._client: AsyncTypeSafeClient | None = None

    def _typesafe(self) -> AsyncTypeSafeClient:
        if self._client is None:
            self._client = AsyncTypeSafeClient(timeout=60.0)
        return self._client

    async def judge(self, packet: VerificationPacket) -> VerificationJudgment:
        response = await self._typesafe().system_one(
            state=packet.model_dump(mode="json"),
            questions=_questions(packet),
            model=self._model,
        )
        scores = {name: answer.noul for name, answer in response.nouls.items()}
        design = response.choices["study_design"]
        judgment = judgment_from_scores(
            scores,
            study_design=str(design.choice),
            study_confidence=float(design.confidence),
        )
        answered_by = response.model
        self._decisions.write(
            DecisionRecord(
                agent="verification",
                model=answered_by,
                stage="verification",
                input_hash=input_hash(packet.model_dump_json()),
                output=judgment.model_dump(mode="json"),
                timestamp=utc_stamp(),
            )
        )
        return judgment
