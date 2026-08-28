import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


def runtime_env_file() -> Path:
    """Return the editable .env beside the source tree or packaged executable."""

    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / ".env"
    return Path(__file__).resolve().parents[1] / ".env"


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    llm_provider: Literal["deepseek", "qwen", "custom"] = "custom"
    """Explicit request dialect; old configurations deliberately default to custom."""
    llm_api_style: Literal["chat", "responses"] = "chat"
    """OpenAI-compatible endpoint family used by the configured provider."""
    llm_base_url: str = ""
    llm_api_key: SecretStr = SecretStr("")
    llm_model: str = ""
    llm_reasoning_effort: Literal[
        "", "none", "minimal", "low", "medium", "high", "xhigh", "max"
    ] = ""
    """Optional backend-only effort override; intentionally absent from the desktop dialog."""
    llm_timeout_seconds: float = 120.0
    """Backend request timeout; intentionally hidden from the end-user dialog."""
    llm_max_retries: int = 1
    llm_thinking_policy: Literal["strip", "reject"] = "strip"
    """``strip`` discards provider reasoning; ``reject`` fails closed when it is present."""
    llm_history_messages: int = Field(default=8, ge=1, le=100)
    """Backend conversation window; intentionally hidden from the end-user dialog."""
    skill_paths: str = ""
    model_config = SettingsConfigDict(
        env_file=str(runtime_env_file()),
        env_prefix="VPP_",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
