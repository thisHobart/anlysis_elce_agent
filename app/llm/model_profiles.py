"""Versioned model-route capabilities resolved independently from secrets."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import date
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field

from app.runtime_paths import user_model_profiles_path

ApiStyle = Literal["chat", "responses"]
StructuredOutputMethod = Literal["function_calling", "json_schema", "json_mode", "prompt_json"]
ProfileSource = Literal["builtin", "user", "legacy", "unverified"]


class ModelCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid")

    function_calling: bool = False
    json_schema: bool = False
    json_mode: bool = False
    parallel_tool_calls: bool = False


class ModelProfile(BaseModel):
    """Capabilities for one effective provider endpoint and model pair."""

    model_config = ConfigDict(extra="forbid")

    provider: Literal["deepseek", "qwen", "gemini", "custom"]
    base_url: str
    model: str = Field(min_length=1)
    context_window_tokens: int = Field(gt=0)
    max_output_tokens: int = Field(ge=256)
    api_style: ApiStyle = "chat"
    structured_output_method: StructuredOutputMethod = "prompt_json"
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)
    reasoning_uses_output_budget: bool | None = None
    source_url: str = ""
    reviewed_at: date | None = None
    deprecated: bool = False

    @property
    def route_key(self) -> str:
        return model_route_key(self.provider, self.base_url, self.model)


class ModelProfileCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    profiles: list[ModelProfile] = Field(default_factory=list)


class ResolvedModelProfile(BaseModel):
    """One lookup result. Unverified routes deliberately have no invented limits."""

    model_config = ConfigDict(extra="forbid")

    route_key: str
    source: ProfileSource
    profile: ModelProfile | None = None

    @property
    def verified(self) -> bool:
        return self.profile is not None and self.source != "unverified"


def normalize_base_url(value: str, *, provider: str = "custom") -> str:
    """Normalize a route without incorporating credentials or query secrets."""

    raw = value.strip()
    if not raw:
        return "gemini-developer-api" if provider == "gemini" else ""
    if raw.startswith("vertex://"):
        return raw.rstrip("/")
    parsed = urlsplit(raw)
    scheme = parsed.scheme.casefold()
    hostname = (parsed.hostname or "").casefold()
    port = parsed.port
    if port is not None and not ((scheme == "https" and port == 443) or (scheme == "http" and port == 80)):
        hostname = f"{hostname}:{port}"
    path = parsed.path.rstrip("/")
    return urlunsplit((scheme, hostname, path, "", ""))


def model_route_key(provider: str, base_url: str, model: str) -> str:
    return "|".join(
        (
            provider.strip().casefold(),
            normalize_base_url(base_url, provider=provider),
            model.strip(),
        )
    )


def builtin_model_profiles_path() -> Path:
    return Path(__file__).with_name("model_profiles.json")


def load_model_profile_catalog(path: str | Path) -> ModelProfileCatalog:
    candidate = Path(path)
    if not candidate.is_file():
        return ModelProfileCatalog()
    return ModelProfileCatalog.model_validate_json(candidate.read_text(encoding="utf-8"))


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_model_profile_catalog(catalog: ModelProfileCatalog, path: str | Path) -> None:
    payload = json.dumps(catalog.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"
    _atomic_write(Path(path), payload)


def upsert_user_model_profile(profile: ModelProfile, path: str | Path | None = None) -> None:
    target = Path(path) if path is not None else user_model_profiles_path()
    catalog = load_model_profile_catalog(target)
    retained = [item for item in catalog.profiles if item.route_key != profile.route_key]
    save_model_profile_catalog(
        ModelProfileCatalog(profiles=[*retained, profile]),
        target,
    )


def resolve_model_profile(
    *,
    provider: str,
    base_url: str,
    model: str,
    user_path: str | Path | None = None,
    builtin_path: str | Path | None = None,
) -> ResolvedModelProfile:
    route_key = model_route_key(provider, base_url, model)
    sources = (
        ("user", Path(user_path) if user_path is not None else user_model_profiles_path()),
        ("builtin", Path(builtin_path) if builtin_path is not None else builtin_model_profiles_path()),
    )
    for source, path in sources:
        catalog = load_model_profile_catalog(path)
        match = next((item for item in catalog.profiles if item.route_key == route_key), None)
        if match is not None:
            return ResolvedModelProfile(route_key=route_key, source=source, profile=match)
    return ResolvedModelProfile(route_key=route_key, source="unverified")


def resolve_effective_model_profile(
    *,
    provider: str,
    base_url: str,
    model: str,
    context_window_tokens: int | None = None,
    max_output_tokens: int | None = None,
    api_style: ApiStyle = "chat",
    structured_output_method: StructuredOutputMethod = "prompt_json",
    user_path: str | Path | None = None,
    builtin_path: str | Path | None = None,
) -> ResolvedModelProfile:
    """Resolve catalogs first, then honor explicit legacy/test capabilities."""

    resolved = resolve_model_profile(
        provider=provider,
        base_url=base_url,
        model=model,
        user_path=user_path,
        builtin_path=builtin_path,
    )
    if resolved.profile is not None or not context_window_tokens or not max_output_tokens:
        return resolved
    profile = ModelProfile(
        provider=provider,
        base_url=base_url,
        model=model,
        context_window_tokens=context_window_tokens,
        max_output_tokens=max_output_tokens,
        api_style=api_style,
        structured_output_method=structured_output_method,
    )
    return ResolvedModelProfile(route_key=profile.route_key, source="legacy", profile=profile)
