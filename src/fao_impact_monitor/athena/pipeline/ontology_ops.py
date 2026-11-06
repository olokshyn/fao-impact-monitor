"""Activate a relation subtype without creating an assertion."""

from __future__ import annotations

from fao_impact_monitor.athena.agents.schemas import NewTypeDraft, OntologyRequest
from fao_impact_monitor.athena.context import AthenaContext
from fao_impact_monitor.athena.model.records import (
    RelationDefinition,
    RetrievalRecipe,
    TypeProfile,
)
from fao_impact_monitor.athena.ns import GRAPH_ONTOLOGY, term
from fao_impact_monitor.athena.observe import DecisionRecord, input_hash
from fao_impact_monitor.athena.storage.ontology import (
    find_relation,
    relation_definitions,
)
from fao_impact_monitor.athena.storage.rdf import (
    n3_iri,
    n3_str,
    utc_stamp,
    write_definition,
    write_record,
)


def validate_definition(
    definition: RelationDefinition,
    known: dict[str, RelationDefinition],
) -> list[str]:
    errors: list[str] = []
    if not definition.definition.strip():
        errors.append("definition is required")
    if not definition.parents:
        errors.append("parent is required")
    if not definition.examples:
        errors.append("cited examples are required")
    if not definition.subject_type or not definition.object_type:
        errors.append("endpoint constraints are required")
    for parent in definition.parents:
        if parent not in known:
            errors.append(f"parent is not in the ontology: {parent}")
        if parent == definition.iri:
            errors.append("relation cannot be its own parent")
    if definition.iri in known:
        errors.append("relation already exists")
    return errors


async def activate_type(ctx: AthenaContext, draft: NewTypeDraft) -> str | None:
    """Reuse an existing type or validate and activate a new one. No assertion is written."""
    existing = find_relation(ctx.graph, draft.name)
    if existing is not None:
        return existing.iri
    for alias in draft.aliases:
        existing = find_relation(ctx.graph, alias)
        if existing is not None:
            return existing.iri
    known = relation_definitions(ctx.graph)
    decision = await ctx.ontologist.propose(
        OntologyRequest(
            proposal=draft,
            existing=[item.label for item in known.values()],
        )
    )
    if decision.reuse_name:
        reused = find_relation(ctx.graph, decision.reuse_name)
        if reused is not None:
            return reused.iri
    parent = find_relation(ctx.graph, decision.parent or draft.parent)
    if parent is None:
        ctx.decisions.write(
            DecisionRecord(
                agent="ontology",
                model=ctx.config.ontology_model,
                stage="ontology",
                input_hash=input_hash(draft.name),
                output={"error": "missing parent"},
                timestamp=utc_stamp(),
            )
        )
        return None
    local = _local_name(decision.reuse_name or draft.name)
    definition = RelationDefinition(
        iri=term(f"relation/{local}"),
        label=local,
        definition=decision.definition or draft.definition,
        parents=[parent.iri],
        subject_type=_type_iri(decision.subject_type or draft.subject_type),
        object_type=_type_iri(decision.object_type or draft.object_type),
        directionality=decision.directionality or draft.directionality,
        examples=decision.examples or draft.examples,
        aliases=decision.aliases or draft.aliases,
        version="1",
    )
    errors = validate_definition(definition, known)
    if errors:
        ctx.decisions.write(
            DecisionRecord(
                agent="ontology",
                model=ctx.config.ontology_model,
                stage="ontology",
                input_hash=input_hash(draft.model_dump_json()),
                output={"errors": errors},
                timestamp=utc_stamp(),
            )
        )
        return None
    write_definition(ctx.graph, definition)
    recipe = RetrievalRecipe(
        relation_iri=definition.iri,
        subject_type=definition.subject_type,
        object_type=definition.object_type,
        aliases=definition.aliases,
        examples=definition.examples,
        terms=[definition.label, *definition.aliases, *definition.examples],
    )
    ctx.graph.update(
        f"""
        INSERT DATA {{
          GRAPH {n3_iri(GRAPH_ONTOLOGY)} {{
            {n3_iri(term("recipe/" + local))} {n3_iri(term("recordJson"))} {n3_str(recipe.model_dump_json())} .
          }}
        }}
        """
    )
    profile = TypeProfile(
        iri=term(f"profile/{local}"),
        relation_iri=definition.iri,
        text="\n".join(
            [definition.definition, *definition.examples, *definition.aliases]
        ),
        aliases=definition.aliases,
        examples=definition.examples,
    )
    write_record(ctx.graph, profile, GRAPH_ONTOLOGY, index=True)
    ctx.gate.submit("backfill", {"relation_iri": definition.iri})
    return definition.iri


def _local_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() else " " for char in value.strip())
    parts = cleaned.split()
    if not parts:
        return "relation"
    return parts[0].lower() + "".join(part[:1].upper() + part[1:] for part in parts[1:])


def _type_iri(value: str) -> str:
    if value.startswith(("http://", "https://")):
        return value
    return term(value)
