"""External retrieval is off unless explicitly enabled. Hits use normal ingestion."""

from __future__ import annotations

from fao_impact_monitor.athena.agents.schemas import ExternalHit


class DisabledExternalRetriever:
    enabled = False

    async def search(self, query: str) -> list[ExternalHit]:
        del query
        return []
