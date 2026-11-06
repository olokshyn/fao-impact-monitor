"""Evidence-reading verifier. It does not replace deterministic citation checks."""

from __future__ import annotations

from fao_impact_monitor.athena.agents.runner import run_structured
from fao_impact_monitor.athena.agents.schemas import (
    VerificationJudgment,
    VerificationPacket,
)
from fao_impact_monitor.athena.observe import DecisionLog

INSTRUCTIONS = """
Check the recorded statement against the excerpt only.
Set faithful to false when wording, quantities, units, negation, uncertainty, or context differ.
For a relation, relation_supported is true only when the excerpt supports that relation,
its direction, and its subtype. Correlation, co-occurrence, and similarity are not causality.
Set expresses_causality only when the excerpt states a cause, and keep tentative wording.
If the excerpt is ambiguous, use null instead of guessing.
derivation_valid is true only when every premise, rule, and assumption is identified.
""".strip()


class SdkEvidenceVerifier:
    def __init__(self, model_name: str, decisions: DecisionLog) -> None:
        self._model = model_name
        self._decisions = decisions

    async def judge(self, packet: VerificationPacket) -> VerificationJudgment:
        return await run_structured(
            agent="verification",
            instructions=INSTRUCTIONS,
            output_type=VerificationJudgment,
            model_name=self._model,
            prompt=packet.model_dump_json(),
            decisions=self._decisions,
        )
