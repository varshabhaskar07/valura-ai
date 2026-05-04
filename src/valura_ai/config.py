"""Runtime configuration.

Loaded once at import; tests override via env vars or by constructing
``Settings`` directly. We deliberately keep this small — extending the
config surface area is cheap, but unused knobs cost reviewers attention.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        extra="ignore",
    )

    # --- LLM ---
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    model: str = Field(default="gpt-4o-mini", alias="VALURA_MODEL")

    # --- Pipeline timeouts (seconds) ---
    # Total budget for safety + classifier + agent.
    # 12s gives ~6s headroom over the p95 e2e target so a slow tail
    # surfaces as a structured timeout event rather than a hang.
    request_timeout_s: float = Field(default=12.0, alias="VALURA_REQUEST_TIMEOUT_S")
    # Classifier budget. On expiry we fall back to the heuristic router
    # so a slow LLM never blocks routing.
    classifier_timeout_s: float = Field(default=4.0, alias="VALURA_CLASSIFIER_TIMEOUT_S")

    # --- Memory ---
    history_turns: int = Field(default=3, alias="VALURA_HISTORY_TURNS")

    # --- Server ---
    host: str = Field(default="127.0.0.1", alias="VALURA_HOST")
    port: int = Field(default=8000, alias="VALURA_PORT")
    log_level: str = Field(default="INFO", alias="VALURA_LOG_LEVEL")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
