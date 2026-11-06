"""Tag inheritance and explicit geographic derivations."""

from __future__ import annotations

from fao_impact_monitor.athena.model.records import TagDerivation, TagField
from fao_impact_monitor.athena.storage.ontology import LocationHierarchy


def merge_tags(
    registry: dict[str, TagField],
    source_tags: dict[str, list[str]],
    extracted: dict[str, list[str]],
    extracted_unknown: list[str],
) -> tuple[dict[str, list[str]], list[str]]:
    """Preserve supplied metadata, inherit publication fields, and keep unknowns."""
    tags: dict[str, list[str]] = {}
    unknown: list[str] = []
    for name, field in registry.items():
        values = [value for value in extracted.get(name, []) if value]
        if values:
            tags[name] = values
            continue
        if field.inherits and source_tags.get(name):
            tags[name] = list(source_tags[name])
            continue
        unknown.append(name)
    for name in extracted_unknown:
        if name not in tags and name not in unknown:
            unknown.append(name)
    return tags, unknown


def derive_broader_regions(
    tags: dict[str, list[str]],
    unknown: list[str],
    hierarchy: LocationHierarchy,
) -> tuple[dict[str, list[str]], list[str], list[TagDerivation]]:
    """Add ancestor places only with a recorded derivation."""
    broader = list(tags.get("broader_region", []))
    derivations: list[TagDerivation] = []
    for code in tags.get("country", []):
        for ancestor in sorted(hierarchy.ancestors(code)):
            if ancestor in broader:
                continue
            broader.append(ancestor)
            derivations.append(
                TagDerivation(
                    field="broader_region",
                    value=ancestor,
                    rule="location-hierarchy",
                    premises=[code],
                )
            )
    updated = dict(tags)
    remaining = [field for field in unknown if field != "broader_region" or not broader]
    if broader:
        updated["broader_region"] = broader
        remaining = [field for field in remaining if field != "broader_region"]
    return updated, remaining, derivations
