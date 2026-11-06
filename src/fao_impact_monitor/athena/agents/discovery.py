"""Relation discovery. Proposals are not knowledge until verification accepts them."""

from __future__ import annotations

from fao_impact_monitor.athena.agents.runner import run_structured
from fao_impact_monitor.athena.agents.schemas import DiscoveryRequest, DiscoveryResult
from fao_impact_monitor.athena.observe import DecisionLog

INSTRUCTIONS = """
Propose relations between the focus claim and the candidates, in either direction.
Respect population, geography, time, and conditions. A context difference is not a contradiction.
Use an existing relation name when one fits. Propose a new type only with a definition,
parent, endpoint types, and a cited example.
Mark source-reported only when a cited excerpt states that relation.
A chain of links is derived, not a directly reported cause.
Do not treat causality or correlation as transitive.
""".strip()


class SdkRelationProposer:
    def __init__(self, model_name: str, decisions: DecisionLog) -> None:
        self._model = model_name
        self._decisions = decisions

    async def propose(self, request: DiscoveryRequest) -> DiscoveryResult:
        return await run_structured(
            agent="relation-discovery",
            instructions=INSTRUCTIONS,
            output_type=DiscoveryResult,
            model_name=self._model,
            prompt=request.model_dump_json(),
            decisions=self._decisions,
        )
