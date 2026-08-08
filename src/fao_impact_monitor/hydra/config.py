"""Hydra settings. Instantiate only from the host app config, not here."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class FetchConfig(BaseSettings):
    """Settings for FetchStage blob persistence (fsspec-compatible paths)."""

    model_config = SettingsConfigDict(
        env_prefix="HYDRA_FETCH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Relative to cwd by default; may be an object-storage URL (s3://…).
    body_save_dir: str = "fetched_data/fetch"


class HydraConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HYDRA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    max_attempts: int = 3
    heartbeat_interval_minutes: float = 5.0
    stale_multiplier: float = 3.0
    wait_poll_interval_seconds: float = 1.0
    claim_idle_sleep_seconds: float = 0.05
    fetch: FetchConfig = Field(default_factory=FetchConfig)
