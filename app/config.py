import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


def runtime_env_file() -> Path:
    """Return the editable .env beside the source tree or packaged executable."""

    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / ".env"
    return Path(__file__).resolve().parents[1] / ".env"


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    llm_enabled: bool = False
    llm_base_url: str = ""
    llm_api_key: SecretStr = SecretStr("")
    llm_model: str = ""
    llm_timeout_seconds: float = 20.0
    llm_max_retries: int = 1
    llm_structured_mode: Literal["json_prompt", "native"] = "native"
    llm_history_messages: int = 8
    skill_paths: str = ""
    model_config = SettingsConfigDict(
        env_file=str(runtime_env_file()),
        env_prefix="VPP_",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
