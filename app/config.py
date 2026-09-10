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

    llm_provider: Literal["deepseek", "qwen", "gemini", "custom"] = "custom"
    """Explicit provider adapter; old configurations deliberately default to custom."""
    llm_api_style: Literal["chat", "responses"] = "chat"
    """OpenAI-compatible endpoint family used by the configured provider."""
    llm_structured_output_method: Literal[
        "function_calling",
        "json_schema",
        "json_mode",
        "prompt_json",
    ] = "function_calling"
    """Native structured-output protocol used by OpenAI-compatible adapters.

    ``prompt_json`` is an explicit compatibility mode for proxies such as Cherry Studio:
    the schema is included in a system message and enforced locally after generation.
    The native Gemini adapter always uses Gemini ``json_schema`` and ignores this switch.
    """
    llm_base_url: str = ""
    llm_api_key: SecretStr = SecretStr("")
    llm_model: str = ""
    llm_google_vertexai: bool = False
    """Use Vertex AI rather than the Gemini Developer API in the native Gemini adapter."""
    llm_google_project: str = ""
    llm_google_location: str = ""
    llm_reasoning_effort: Literal[
        "", "none", "minimal", "low", "medium", "high", "xhigh", "max"
    ] = ""
    """Optional backend-only effort override; intentionally absent from the desktop dialog."""
    llm_timeout_seconds: float = 120.0
    """Backend request timeout; intentionally hidden from the end-user dialog."""
    llm_max_retries: int = 1
    llm_context_window_tokens: int = Field(default=0, ge=0)
    """Configured model context window; 0 leaves preflight enforcement to the caller."""
    llm_max_output_tokens: int = Field(default=4096, ge=256)
    """Maximum generated tokens reserved by context preflight and sent to providers."""
    llm_context_safety_tokens: int = Field(default=1024, ge=0)
    """Additional headroom for provider-specific message framing and tokenization drift."""
    llm_thinking_policy: Literal["strip", "reject"] = "strip"
    """``strip`` discards provider reasoning; ``reject`` fails closed when it is present."""
    llm_history_messages: int = Field(default=8, ge=1, le=100)
    """Approximate recent-message budget, selected as half as many complete turns."""
    llm_retrieved_turns: int = Field(default=3, ge=0, le=10)
    """Earlier related conversation turns retrieved beyond the recent window; 0 disables retrieval."""
    skill_paths: str = ""
    model_config = SettingsConfigDict(
        env_file=str(runtime_env_file()),
        env_prefix="VPP_",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
