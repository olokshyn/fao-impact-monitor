"""Checkpointed backfill for a newly activated relation type."""

from __future__ import annotations

from fao_impact_monitor.athena.context import AthenaContext
from fao_impact_monitor.athena.model.filters import FilterSpec
from fao_impact_monitor.athena.model.records import RetrievalRecipe
from fao_impact_monitor.athena.ns import GRAPH_OPERATIONS, term
from fao_impact_monitor.athena.pipeline.discover import discover_for_claim
from fao_impact_monitor.athena.pipeline.load import load_claims
from fao_impact_monitor.athena.storage.rdf import n3_iri, n3_str


def _cursor_iri(relation_iri: str) -> str:
    local = relation_iri.rsplit("/", 1)[-1]
    return term(f"backfill/{local}")


def read_cursor(ctx: AthenaContext, relation_iri: str) -> str:
    rows = ctx.graph.query(
        f"""
        SELECT ?cursor WHERE {{
          GRAPH {n3_iri(GRAPH_OPERATIONS)} {{
            {n3_iri(_cursor_iri(relation_iri))} {n3_iri(term("cursor"))} ?cursor
          }}
        }}
        """
    )
    if not rows:
        return ""
    return rows[0]["cursor"]


def write_cursor(ctx: AthenaContext, relation_iri: str, cursor: str) -> None:
    iri = _cursor_iri(relation_iri)
    ctx.graph.update(
        f"""
        DELETE {{
          GRAPH {n3_iri(GRAPH_OPERATIONS)} {{ {n3_iri(iri)} {n3_iri(term("cursor"))} ?cursor }}
        }}
        WHERE {{
          GRAPH {n3_iri(GRAPH_OPERATIONS)} {{ {n3_iri(iri)} {n3_iri(term("cursor"))} ?cursor }}
        }};
        INSERT DATA {{
          GRAPH {n3_iri(GRAPH_OPERATIONS)} {{
            {n3_iri(iri)} {n3_iri(term("cursor"))} {n3_str(cursor)}
          }}
        }}
        """
    )


def read_recipe(ctx: AthenaContext, relation_iri: str) -> RetrievalRecipe | None:
    local = relation_iri.rsplit("/", 1)[-1]
    rows = ctx.graph.query(
        f"""
        SELECT ?json WHERE {{
          GRAPH ?g {{ {n3_iri(term("recipe/" + local))} {n3_iri(term("recordJson"))} ?json }}
        }}
        """
    )
    if not rows:
        return None
    return RetrievalRecipe.model_validate_json(rows[0]["json"])


async def run_backfill(ctx: AthenaContext, relation_iri: str) -> dict[str, object]:
    """Evaluate older claims, including old-to-old pairs, without re-embedding the corpus."""
    claims = sorted(load_claims(ctx.graph).values(), key=lambda claim: claim.iri)
    cursor = read_cursor(ctx, relation_iri)
    pending = [claim for claim in claims if claim.iri > cursor]
    window = pending[: ctx.config.backfill_batch_size]
    if not window:
        return {"done": True, "cursor": cursor}
    created: list[str] = []
    for claim in window:
        created.extend(
            await discover_for_claim(
                ctx,
                claim.iri,
                spec=FilterSpec(),
                extra=window,
            )
        )
    new_cursor = window[-1].iri
    write_cursor(ctx, relation_iri, new_cursor)
    if len(pending) > len(window):
        ctx.gate.submit("backfill", {"relation_iri": relation_iri})
    return {"done": False, "cursor": new_cursor, "assertions": created}
