"""Precision of proposed relations and recall of candidate retrieval."""

from __future__ import annotations


def relation_precision(
    predicted: set[tuple[str, str, str]],
    gold: set[tuple[str, str, str]],
) -> float:
    if not predicted:
        return 1.0 if not gold else 0.0
    return len(predicted & gold) / len(predicted)


def candidate_recall(retrieved: set[str], relevant: set[str]) -> float:
    if not relevant:
        return 1.0
    return len(retrieved & relevant) / len(relevant)
