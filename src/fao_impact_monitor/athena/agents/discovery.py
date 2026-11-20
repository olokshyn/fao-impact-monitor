"""Relation discovery over the relations that already exist."""

from __future__ import annotations

from typesafe_sdk import AsyncTypeSafeClient, Choice

from fao_impact_monitor.athena.agents.schemas import (
    DiscoveryRequest,
    DiscoveryResult,
    ProposedRelation,
)
from fao_impact_monitor.athena.model.status import ScientificStatus
from fao_impact_monitor.athena.observe import DecisionLog, DecisionRecord, input_hash
from fao_impact_monitor.athena.storage.rdf import utc_stamp

_CONFIDENT = 0.8
_NONE = "none"


def _split(line: str) -> tuple[str, str]:
    iri, separator, text = line.partition("\n")
    if not separator:
        return iri, iri
    return iri, text


def _option(label: str, *, forward: bool) -> str:
    direction = "forward" if forward else "reverse"
    return f"{label} [{direction}]"


def _criteria(relations: list[str]) -> dict[str, str]:
    criteria = {
        _NONE: "The evidence does not state a relation between the focus claim and this candidate.",
    }
    for label in relations:
        criteria[_option(label, forward=True)] = (
            f"The evidence states that the focus claim {label} the candidate."
        )
        criteria[_option(label, forward=False)] = (
            f"The evidence states that the candidate {label} the focus claim."
        )
    return criteria


def proposals_from_choices(
    request: DiscoveryRequest,
    choices: list[tuple[str, float]],
) -> DiscoveryResult:
    """Keep a relation only when Jev selects it with enough confidence."""
    focus, _text = _split(request.claim_text)
    options = _criteria(request.existing_relations)
    relations: list[ProposedRelation] = []
    for (iri, _candidate), (choice, confidence) in zip(
        (_split(line) for line in request.candidates),
        choices,
        strict=True,
    ):
        if choice not in options or choice == _NONE or confidence < _CONFIDENT:
            continue
        label, _bracket, direction = choice.partition(" [")
        forward = direction == "forward]"
        relations.append(
            ProposedRelation(
                subject_iri=focus if forward else iri,
                object_iri=iri if forward else focus,
                relation=label,
                scientific_status=ScientificStatus.SOURCE_REPORTED,
            )
        )
    return DiscoveryResult(relations=relations)


class SdkRelationProposer:
    def __init__(self, model_name: str, decisions: DecisionLog) -> None:
        self._model = model_name
        self._decisions = decisions
        self._client: AsyncTypeSafeClient | None = None

    def _typesafe(self) -> AsyncTypeSafeClient:
        if self._client is None:
            self._client = AsyncTypeSafeClient(timeout=60.0)
        return self._client

    async def propose(self, request: DiscoveryRequest) -> DiscoveryResult:
        if not request.candidates or not request.existing_relations:
            return DiscoveryResult()
        criteria = _criteria(request.existing_relations)
        focus_iri, focus_text = _split(request.claim_text)
        candidates = [
            {"iri": iri, "text": text}
            for iri, text in (_split(line) for line in request.candidates)
        ]
        questions = {
            f"candidate-{index}": Choice(
                instructions=(
                    "Which relation does the evidence state between `focus` and "
                    f"`candidates[{index}]`? A difference of place, time, "
                    "population, or conditions is not a relation. "
                    "Correlation is not causation."
                ),
                criteria=criteria,
            )
            for index in range(len(candidates))
        }
        response = await self._typesafe().system_one(
            state={
                "focus": {"iri": focus_iri, "text": focus_text},
                "candidates": candidates,
            },
            questions=questions,
            model=self._model,
        )
        chosen = [
            (
                response.choices[f"candidate-{index}"].choice,
                response.choices[f"candidate-{index}"].confidence,
            )
            for index in range(len(candidates))
        ]
        result = proposals_from_choices(request, chosen)
        self._decisions.write(
            DecisionRecord(
                agent="relation-discovery",
                model=response.model,
                stage="relation-discovery",
                input_hash=input_hash(request.model_dump_json()),
                output=result.model_dump(mode="json"),
                timestamp=utc_stamp(),
            )
        )
        return result
