import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.llm.model_profiles import ModelCapabilities, ModelProfile, resolve_model_profile, upsert_user_model_profile
from app.llm.runtime_settings import (
    ApplicationSettings,
    load_application_settings,
    migrated_application_settings,
    save_application_settings,
)
from app.runtime_paths import application_settings_path


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
    the schema and research-function whitelist are included in a system message and
    enforced locally after generation. No model proposal executes before normal plan approval.
    The native Gemini adapter always uses Gemini ``json_schema`` and ignores this switch.
    """
    llm_base_url: str = ""
    llm_api_key: SecretStr = SecretStr("")
    llm_model: str = ""
    llm_google_vertexai: bool = False
    """Use Vertex AI rather than the Gemini Developer API in the native Gemini adapter."""
    llm_google_project: str = ""
    llm_google_location: str = ""
    llm_reasoning_effort: Literal["", "none", "minimal", "low", "medium", "high", "xhigh", "max"] = ""
    """Optional backend-only effort override; intentionally absent from the desktop dialog."""
    llm_timeout_seconds: float = 120.0
    """Backend request timeout; intentionally hidden from the end-user dialog."""
    llm_max_retries: int = 1
    llm_context_window_tokens: int | None = Field(default=None, ge=1)
    """Resolved model capability; absent means the route is not verified."""
    llm_max_output_tokens: int | None = Field(default=None, ge=256)
    """Resolved model capability; task-specific output limits are computed per call."""
    llm_context_safety_tokens: int = Field(default=1024, ge=0)
    """Additional headroom for provider-specific message framing and tokenization drift."""
    llm_thinking_policy: Literal["strip", "reject"] = "strip"
    """``strip`` discards provider reasoning; ``reject`` fails closed when it is present."""
    llm_history_messages: int = Field(default=8, ge=1, le=100)
    """Approximate recent-message budget, selected as half as many complete turns."""
    llm_retrieved_turns: int = Field(default=3, ge=0, le=10)
    """Earlier related conversation turns retrieved beyond the recent window; 0 disables retrieval."""
    p2_news_path: str = ""
    """Optional frozen JSONL corpus used by the desktop P1→P2→P3 workflow."""
    skill_paths: str = ""
    model_config = SettingsConfigDict(
        env_file=str(runtime_env_file()),
        env_prefix="VPP_",
        extra="ignore",
    )

    @field_validator("llm_context_window_tokens", "llm_max_output_tokens", mode="before")
    @classmethod
    def empty_or_zero_capacity_is_unverified(cls, value):
        if value in (None, "", 0, "0"):
            return None
        return value


@lru_cache
def get_settings() -> Settings:
    connection = Settings()
    settings_path = application_settings_path()
    application = load_application_settings(settings_path)
    if not application.legacy_llm_env_migrated:
        explicit = {
            str(key): str(value) for key, value in dotenv_values(runtime_env_file()).items() if value is not None
        }
        explicit.update({key: value for key, value in os.environ.items() if key.startswith("VPP_LLM_")})
        application = migrated_application_settings(explicit)
        save_application_settings(application, settings_path)
        context_value = explicit.get("VPP_LLM_CONTEXT_WINDOW_TOKENS", "").strip()
        output_value = explicit.get("VPP_LLM_MAX_OUTPUT_TOKENS", "").strip()
        if connection.llm_model and context_value and output_value:
            method = connection.llm_structured_output_method
            legacy_route = (
                f"vertex://{application.llm.gemini.project}/{application.llm.gemini.location}"
                if connection.llm_provider == "gemini" and application.llm.gemini.vertexai
                else connection.llm_base_url
            )
            upsert_user_model_profile(
                ModelProfile(
                    provider=connection.llm_provider,
                    base_url=legacy_route,
                    model=connection.llm_model,
                    context_window_tokens=int(context_value),
                    max_output_tokens=int(output_value),
                    api_style=connection.llm_api_style,
                    structured_output_method=method,
                    capabilities=ModelCapabilities(
                        function_calling=method == "function_calling",
                        json_schema=method == "json_schema",
                        json_mode=method in {"json_mode", "prompt_json"},
                        parallel_tool_calls=method == "function_calling",
                    ),
                )
            )

    runtime = application.llm
    route_url = connection.llm_base_url
    if connection.llm_provider == "gemini":
        if runtime.gemini.vertexai:
            route_url = f"vertex://{runtime.gemini.project}/{runtime.gemini.location}"
        else:
            route_url = ""
    resolved = resolve_model_profile(
        provider=connection.llm_provider,
        base_url=route_url,
        model=connection.llm_model,
    )
    profile = resolved.profile
    return Settings(
        _env_file=None,
        llm_provider=connection.llm_provider,
        llm_base_url=connection.llm_base_url,
        llm_api_key=connection.llm_api_key,
        llm_model=connection.llm_model,
        llm_api_style=profile.api_style if profile is not None else runtime.unverified_api_style,
        llm_structured_output_method=(
            profile.structured_output_method if profile is not None else runtime.unverified_structured_output_method
        ),
        llm_context_window_tokens=(profile.context_window_tokens if profile is not None else None),
        llm_max_output_tokens=profile.max_output_tokens if profile is not None else None,
        llm_timeout_seconds=runtime.timeout_seconds,
        llm_max_retries=runtime.max_retries,
        llm_reasoning_effort=runtime.reasoning_effort,
        llm_thinking_policy=runtime.thinking_policy,
        llm_history_messages=runtime.history_messages,
        llm_retrieved_turns=runtime.retrieved_turns,
        llm_google_vertexai=runtime.gemini.vertexai,
        llm_google_project=runtime.gemini.project,
        llm_google_location=runtime.gemini.location,
        p2_news_path=connection.p2_news_path,
        skill_paths=connection.skill_paths,
    )


@lru_cache
def get_application_settings() -> ApplicationSettings:
    get_settings()
    return load_application_settings()
