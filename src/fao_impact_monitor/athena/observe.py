"""Decision and claim logs. Every agent decision is recorded."""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Protocol

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DecisionRecord(BaseModel):
    agent: str
    model: str
    stage: str
    input_hash: str
    output: dict[str, Any] = Field(default_factory=dict)
    claim_ids: list[str] = Field(default_factory=list)
    timestamp: str


class DecisionLog(Protocol):
    def write(self, record: DecisionRecord) -> None: ...


class MemoryDecisionLog:
    """Logger plus an in-memory sink used by tests and as a wrapper."""

    def __init__(self, sink: DecisionLog | None = None) -> None:
        self.records: list[DecisionRecord] = []
        self._sink = sink

    def write(self, record: DecisionRecord) -> None:
        logger.info(
            "athena decision agent=%s stage=%s claims=%s",
            record.agent,
            record.stage,
            ",".join(record.claim_ids),
        )
        self.records.append(record)
        if self._sink is not None:
            self._sink.write(record)


class MongoDecisionLog:
    """Durable decision log in the rebuildable Mongo database."""

    def __init__(self, collection: Any) -> None:
        self._collection = collection

    def write(self, record: DecisionRecord) -> None:
        self._collection.insert_one(record.model_dump())


def input_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def log_claim(
    decisions: DecisionLog,
    *,
    claim_id: str,
    stage: str,
    status: str,
    detail: str,
    timestamp: str,
) -> None:
    decisions.write(
        DecisionRecord(
            agent="pipeline",
            model="",
            stage=stage,
            input_hash=input_hash(claim_id + stage + status),
            output={"status": status, "detail": detail},
            claim_ids=[claim_id],
            timestamp=timestamp,
        )
    )
