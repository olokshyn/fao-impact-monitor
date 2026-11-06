"""Ingestion agent: atomic claims, entities, quantities, passages, and tags."""

from __future__ import annotations

from fao_impact_monitor.athena.agents.runner import run_structured
from fao_impact_monitor.athena.agents.schemas import ExtractionRequest, ExtractionResult
from fao_impact_monitor.athena.observe import DecisionLog

INSTRUCTIONS = """
Extract atomic claims from the source text.
Copy excerpts exactly from the text. Do not invent quantities, units, negation, or scope.
Put a place in applicable_tags only when the evidence applies there. Places that are
only mentioned go in mentioned_places.
Leave unknown tags unknown. Do not guess a country, journal, or publisher.
Keep acquisition channel out of source type.
Record an explicit relation only when the source states it. Correlation is not causality.
Preserve tentative wording and the study design when the source gives them.
""".strip()


class SdkClaimExtractor:
    def __init__(self, model_name: str, decisions: DecisionLog) -> None:
        self._model = model_name
        self._decisions = decisions

    async def extract(self, request: ExtractionRequest) -> ExtractionResult:
        return await run_structured(
            agent="ingestion",
            instructions=INSTRUCTIONS,
            output_type=ExtractionResult,
            model_name=self._model,
            prompt=request.model_dump_json(),
            decisions=self._decisions,
        )
