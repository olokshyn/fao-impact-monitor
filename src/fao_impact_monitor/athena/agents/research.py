"""Research agent. Paths are candidate mechanisms, not automatic causal effects."""

from __future__ import annotations

from fao_impact_monitor.athena.agents.runner import run_structured
from fao_impact_monitor.athena.agents.schemas import ResearchPlan, Synthesis
from fao_impact_monitor.athena.observe import DecisionLog

PLAN_INSTRUCTIONS = """
Decompose the question into variables, contexts, and subquestions.
Do not assume a universal direction such as an El Niño drought for every place.
Name the baseline when the question asks for a comparison.
""".strip()

SYNTHESIS_INSTRUCTIONS = """
Answer only from the supplied admissible pack.
Do not mention evidence that is absent from the pack.
Keep contradictory evidence and its context.
A multi-hop path is not a directly reported causal finding.
Do not multiply scores along a path.
Set qualitative to false only when you also give an explicit quantitative model and an uncertainty range.
Otherwise leave the assessment qualitative and list the missing data or model.
Label extrapolated geography as extrapolation. Do not present it as local evidence.
""".strip()


class SdkResearchReasoner:
    def __init__(self, model_name: str, decisions: DecisionLog) -> None:
        self._model = model_name
        self._decisions = decisions

    async def plan(self, question: str) -> ResearchPlan:
        return await run_structured(
            agent="research-plan",
            instructions=PLAN_INSTRUCTIONS,
            output_type=ResearchPlan,
            model_name=self._model,
            prompt=question,
            decisions=self._decisions,
        )

    async def synthesize(self, question: str, pack: str) -> Synthesis:
        return await run_structured(
            agent="research-synthesis",
            instructions=SYNTHESIS_INSTRUCTIONS,
            output_type=Synthesis,
            model_name=self._model,
            prompt=question + "\n\n" + pack,
            decisions=self._decisions,
        )
