"""Hydra settings. Instantiate only from the host app config, not here."""

from pydantic_settings import BaseSettings, SettingsConfigDict


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
