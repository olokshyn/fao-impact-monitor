"""Scientific status is independent of verification status."""

from __future__ import annotations

from enum import StrEnum


class ScientificStatus(StrEnum):
    SOURCE_REPORTED = "source-reported"
    DERIVED = "derived"
    HYPOTHETICAL = "hypothetical"


class VerificationStatus(StrEnum):
    PENDING = "pending"
    VERIFIED = "verified"
    REJECTED = "rejected"
    REVIEW = "review"


class Directionality(StrEnum):
    DIRECTIONAL = "directional"
    SYMMETRIC = "symmetric"


class UnknownPolicy(StrEnum):
    EXCLUDE = "exclude"
    INCLUDE = "include"
    ONLY = "only"
