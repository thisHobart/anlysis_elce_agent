"""Per-user, non-secret LLM runtime policy stored outside ``.env``."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.runtime_paths import application_settings_path


class OperationTokenPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_output_tokens: int = Field(ge=256)
    minimum_output_tokens: int = Field(ge=1)


class LLMOperationPolicies(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dialogue: OperationTokenPolicy = Field(
        default_factory=lambda: OperationTokenPolicy(target_output_tokens=1024, minimum_output_tokens=384)
    )
    eda_planning: OperationTokenPolicy = Field(
        default_factory=lambda: OperationTokenPolicy(target_output_tokens=2048, minimum_output_tokens=768)
    )
    eda_planning_recovery: OperationTokenPolicy = Field(
        default_factory=lambda: OperationTokenPolicy(target_output_tokens=1024, minimum_output_tokens=384)
    )
    eda_analysis: OperationTokenPolicy = Field(
        default_factory=lambda: OperationTokenPolicy(target_output_tokens=2048, minimum_output_tokens=384)
    )
    result_explanation: OperationTokenPolicy = Field(
        default_factory=lambda: OperationTokenPolicy(target_output_tokens=2048, minimum_output_tokens=512)
    )
    news_extraction: OperationTokenPolicy = Field(
        default_factory=lambda: OperationTokenPolicy(target_output_tokens=4096, minimum_output_tokens=1024)
    )
    generic: OperationTokenPolicy = Field(
        default_factory=lambda: OperationTokenPolicy(target_output_tokens=2048, minimum_output_tokens=512)
    )


class GeminiTransportSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vertexai: bool = False
    project: str = ""
    location: str = ""


class LLMRuntimeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timeout_seconds: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=1, ge=0, le=10)
    reasoning_effort: Literal["", "none", "minimal", "low", "medium", "high", "xhigh", "max"] = ""
    thinking_policy: Literal["strip", "reject"] = "strip"
    history_messages: int = Field(default=8, ge=1, le=100)
    retrieved_turns: int = Field(default=3, ge=0, le=10)
    unverified_api_style: Literal["chat", "responses"] = "chat"
    unverified_structured_output_method: Literal["function_calling", "json_schema", "json_mode", "prompt_json"] = (
        "prompt_json"
    )
    unverified_input_tokens: int = Field(default=8192, ge=2048)
    operation_policies: LLMOperationPolicies = Field(default_factory=LLMOperationPolicies)
    gemini: GeminiTransportSettings = Field(default_factory=GeminiTransportSettings)


class ApplicationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    legacy_llm_env_migrated: bool = False
    legacy_llm_env_migration_version: Literal[1] | None = None
    llm: LLMRuntimeSettings = Field(default_factory=LLMRuntimeSettings)


def load_application_settings(path: str | Path | None = None) -> ApplicationSettings:
    target = Path(path) if path is not None else application_settings_path()
    if not target.is_file():
        return ApplicationSettings()
    return ApplicationSettings.model_validate_json(target.read_text(encoding="utf-8"))


def save_application_settings(settings: ApplicationSettings, path: str | Path | None = None) -> None:
    target = Path(path) if path is not None else application_settings_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(settings.model_dump(mode="json"), stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def migrated_application_settings(values: dict[str, str]) -> ApplicationSettings:
    """Build the first JSON settings document from explicitly present legacy keys."""

    runtime: dict[str, object] = {}
    mappings = {
        "VPP_LLM_TIMEOUT_SECONDS": ("timeout_seconds", float),
        "VPP_LLM_MAX_RETRIES": ("max_retries", int),
        "VPP_LLM_REASONING_EFFORT": ("reasoning_effort", str),
        "VPP_LLM_THINKING_POLICY": ("thinking_policy", str),
        "VPP_LLM_HISTORY_MESSAGES": ("history_messages", int),
        "VPP_LLM_RETRIEVED_TURNS": ("retrieved_turns", int),
        "VPP_LLM_API_STYLE": ("unverified_api_style", str),
        "VPP_LLM_STRUCTURED_OUTPUT_METHOD": ("unverified_structured_output_method", str),
    }
    for environment_name, (field_name, converter) in mappings.items():
        if environment_name in values and values[environment_name].strip():
            runtime[field_name] = converter(values[environment_name].strip())
    gemini = {
        "vertexai": values.get("VPP_LLM_GOOGLE_VERTEXAI", "false").strip().casefold() in {"1", "true", "yes", "on"},
        "project": values.get("VPP_LLM_GOOGLE_PROJECT", "").strip(),
        "location": values.get("VPP_LLM_GOOGLE_LOCATION", "").strip(),
    }
    runtime["gemini"] = gemini
    return ApplicationSettings(
        legacy_llm_env_migrated=True,
        legacy_llm_env_migration_version=1,
        llm=LLMRuntimeSettings.model_validate(runtime),
    )
