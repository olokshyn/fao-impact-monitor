"""Athena service settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from fao_impact_monitor.config import DATA_DIR


class AthenaConfig(BaseSettings):
    """Environment-backed settings for the Athena service."""

    model_config = SettingsConfigDict(
        env_prefix="ATHENA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    fuseki_query_url: str = "http://127.0.0.1:3030/ds/query"
    fuseki_update_url: str = "http://127.0.0.1:3030/ds/update"
    object_store: str = "local"
    object_store_root: Path = DATA_DIR / "athena"
    gcs_bucket: str = ""
    mongo_db_name: str = "athena"
    broker_url: str = "amqp://guest:guest@127.0.0.1:5672//"
    ingest_batch_size: int = 8
    ingest_batch_wait_seconds: float = 30
    verify_batch_size: int = 8
    verify_batch_wait_seconds: float = 30
    discover_batch_size: int = 4
    discover_batch_wait_seconds: float = 60
    backfill_batch_size: int = 16
    backfill_batch_wait_seconds: float = 60
    index_batch_size: int = 16
    index_batch_wait_seconds: float = 15
    research_batch_size: int = 1
    research_batch_wait_seconds: float = 0
    maintain_batch_size: int = 4
    maintain_batch_wait_seconds: float = 60
    ingestion_model: str = "openai.gpt-5.6-luna"
    verifier_model: str = "openai.gpt-5.6-luna"
    discovery_model: str = "openai.gpt-5.6-luna"
    ontology_model: str = "openai.gpt-5.6-luna"
    research_model: str = "openai.gpt-5.6-terra"
    candidate_limit: int = 20
    research_max_rounds: int = 4
    api_host: str = "127.0.0.1"
    api_port: int = 8088
    ensure_indexes: bool = False
    vector_index_name: str = "athena_vector_index"
    text_index_name: str = "athena_text_index"
    external_retrieval_enabled: bool = False


class BatchPolicy(BaseModel):
    """Run a task type once enough items arrive or the oldest item is due."""

    task_type: str
    min_batch_size: int = Field(ge=1)
    max_wait_seconds: float = Field(ge=0)


def policies_from_config(config: AthenaConfig) -> dict[str, BatchPolicy]:
    pairs = (
        ("ingest", config.ingest_batch_size, config.ingest_batch_wait_seconds),
        ("verify", config.verify_batch_size, config.verify_batch_wait_seconds),
        ("discover", config.discover_batch_size, config.discover_batch_wait_seconds),
        ("backfill", config.backfill_batch_size, config.backfill_batch_wait_seconds),
        ("index_sync", config.index_batch_size, config.index_batch_wait_seconds),
        ("research", config.research_batch_size, config.research_batch_wait_seconds),
        ("maintain", config.maintain_batch_size, config.maintain_batch_wait_seconds),
    )
    return {
        task_type: BatchPolicy(
            task_type=task_type,
            min_batch_size=size,
            max_wait_seconds=wait,
        )
        for task_type, size, wait in pairs
    }


@lru_cache
def get_athena_config() -> AthenaConfig:
    return AthenaConfig()
