"""Mongo initialization and idempotent document claims for the PDF package."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

from beanie import init_beanie
from pymongo import ReturnDocument
from pymongo.asynchronous.database import AsyncDatabase

from fao_impact_monitor.config import MongoConfig, get_config
from fao_impact_monitor.data_lake.mongo import create_async_mongo_client
from fao_impact_monitor.pdf_pipeline.models import (
    EvidenceUnit,
    PdfDocumentRecord,
    PdfEmbeddingRecord,
    SectionContextCard,
)

PDF_PIPELINE_MODELS: list[type[Any]] = [
    PdfDocumentRecord,
    SectionContextCard,
    EvidenceUnit,
    PdfEmbeddingRecord,
]


async def init_pdf_pipeline_beanie(database: AsyncDatabase[Any] | Any) -> None:
    await init_beanie(database=database, document_models=PDF_PIPELINE_MODELS)


async def connect_pdf_pipeline(
    config: MongoConfig | None = None,
) -> Any:
    cfg = config or get_config().mongo
    client = create_async_mongo_client(cfg)
    await init_pdf_pipeline_beanie(client[cfg.db_name])
    return client


async def claim_document(
    *,
    content_sha256: str,
    path: str,
    seed: dict[str, Any],
) -> tuple[dict[str, Any] | None, bool]:
    """Create or reopen a document; completed content remains immutable."""
    now = datetime.now(UTC)
    collection = PdfDocumentRecord.get_pymongo_collection()
    existing = await collection.find_one({"content_sha256": content_sha256})
    if existing is not None and existing.get("status") == "completed":
        await collection.update_one(
            {"_id": existing["_id"]}, {"$addToSet": {"discovered_paths": path}}
        )
        return existing, False
    query: dict[str, Any] = {"content_sha256": content_sha256}
    # These fields are owned by the atomic state transition below. MongoDB
    # rejects a field that appears in both $setOnInsert and $set.
    insert_seed = {
        key: value
        for key, value in seed.items()
        if key
        not in {
            "status",
            "failure",
            "completed_at",
            "updated_at",
            "discovered_paths",
        }
    }
    update = {
        "$setOnInsert": insert_seed,
        "$set": {
            "status": "processing",
            "failure": None,
            "updated_at": now,
        },
        "$unset": {"completed_at": ""},
        "$addToSet": {"discovered_paths": path},
    }
    doc = await collection.find_one_and_update(
        query,
        update,
        upsert=existing is None,
        return_document=ReturnDocument.AFTER,
    )
    return doc, doc is not None


async def delete_document_embeddings(content_sha256: str) -> None:
    """Remove only vector representations owned by one PDF content hash."""
    prefix = {"$regex": f"^{re.escape(content_sha256)}:"}
    await PdfEmbeddingRecord.get_pymongo_collection().delete_many(
        {
            "pipeline": "pdf_pipeline",
            "$or": [
                {"section_id": prefix},
                {"owner_id": prefix},
                {"evidence_id": prefix},
            ],
        }
    )


async def reset_pdf_pipeline() -> None:
    """Delete only data owned by this package, preserving all other embeddings."""
    for model in (
        SectionContextCard,
        EvidenceUnit,
        PdfDocumentRecord,
        PdfEmbeddingRecord,
    ):
        await model.get_pymongo_collection().delete_many({"pipeline": "pdf_pipeline"})


async def purge_pdf_document(content_sha256: str) -> None:
    """Remove and verify every pipeline record for one source hash."""
    prefix = {"$regex": f"^{re.escape(content_sha256)}:"}
    section_filter = {
        "$or": [
            {"document_sha256": content_sha256},
            {"section_id": prefix},
        ]
    }
    evidence_filter = {
        "$or": [
            {"document_sha256": content_sha256},
            {"section_id": prefix},
            {"evidence_id": prefix},
        ]
    }
    embedding_filter = {
        "pipeline": "pdf_pipeline",
        "$or": [
            {"section_id": prefix},
            {"owner_id": prefix},
            {"evidence_id": prefix},
        ],
    }
    await PdfEmbeddingRecord.get_pymongo_collection().delete_many(embedding_filter)
    await EvidenceUnit.get_pymongo_collection().delete_many(evidence_filter)
    await SectionContextCard.get_pymongo_collection().delete_many(section_filter)
    await PdfDocumentRecord.get_pymongo_collection().delete_many(
        {"content_sha256": content_sha256}
    )
    remaining = {
        "documents": await PdfDocumentRecord.get_pymongo_collection().count_documents(
            {"content_sha256": content_sha256}
        ),
        "sections": await SectionContextCard.get_pymongo_collection().count_documents(
            section_filter
        ),
        "evidence": await EvidenceUnit.get_pymongo_collection().count_documents(
            evidence_filter
        ),
        "embeddings": await PdfEmbeddingRecord.get_pymongo_collection().count_documents(
            embedding_filter
        ),
    }
    if any(remaining.values()):
        raise RuntimeError(f"PDF cleanup incomplete for {content_sha256}: {remaining}")
