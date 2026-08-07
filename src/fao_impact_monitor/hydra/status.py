"""Shared lifecycle status for Task, StageResult, and Run."""

from enum import StrEnum, auto


class Status(StrEnum):
    CREATED = auto()
    SCHEDULED = auto()
    RUNNING = auto()
    RETRYING = auto()
    COMPLETED = auto()
    FAILED = auto()
