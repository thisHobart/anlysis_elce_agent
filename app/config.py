import sys
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


def runtime_env_file() -> Path:
    """Return the editable .env beside the source tree or packaged executable."""

    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / ".env"
    return Path(__file__).resolve().parents[1] / ".env"


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    llm_base_url: str = ""
    llm_api_key: SecretStr = SecretStr("")
    llm_model: str = ""
    llm_timeout_seconds: float = 20.0
    llm_max_retries: int = 1
    llm_history_messages: int = Field(default=8, ge=1, le=100)
    """Conversation turns sent to the model; the desktop dialog writes this value."""
    skill_paths: str = ""
    model_config = SettingsConfigDict(
        env_file=str(runtime_env_file()),
        env_prefix="VPP_",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
