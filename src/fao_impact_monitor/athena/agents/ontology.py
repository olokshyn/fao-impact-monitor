"""Ontology agent. Reuse a definition before proposing a subtype."""

from __future__ import annotations

from fao_impact_monitor.athena.agents.runner import run_structured
from fao_impact_monitor.athena.agents.schemas import OntologyDecision, OntologyRequest
from fao_impact_monitor.athena.observe import DecisionLog

INSTRUCTIONS = """
If an existing relation name or alias already covers the proposal, return it as reuse_name.
Otherwise return a definition, one parent, endpoint types, directionality, and cited examples.
Creating a type does not assert that any particular pair is related.
Increases and decreases describe effect direction, not benefit or harm.
""".strip()


class SdkOntologyProposer:
    def __init__(self, model_name: str, decisions: DecisionLog) -> None:
        self._model = model_name
        self._decisions = decisions

    async def propose(self, request: OntologyRequest) -> OntologyDecision:
        return await run_structured(
            agent="ontology",
            instructions=INSTRUCTIONS,
            output_type=OntologyDecision,
            model_name=self._model,
            prompt=request.model_dump_json(),
            decisions=self._decisions,
        )
