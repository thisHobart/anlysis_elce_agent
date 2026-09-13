"""Model-route profiles, request budgets, and prompt-free call auditing."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.config import get_application_settings, get_settings
from app.llm.audit import ModelCallAudit
from app.llm.budget import ModelBudgetManager, ModelRequestPurpose
from app.llm.gateway import ModelContextLimitError, ModelMessage
from app.llm.model_profiles import (
    ModelProfile,
    ModelProfileCatalog,
    ResolvedModelProfile,
    model_route_key,
    normalize_base_url,
    resolve_model_profile,
    save_model_profile_catalog,
)
from app.llm.runtime_settings import LLMRuntimeSettings


def _profile(*, context: int = 128_000, output: int = 16_384) -> ModelProfile:
    return ModelProfile(
        provider="custom",
        base_url="HTTPS://Model.Example:443/v1/",
        model="research-model",
        context_window_tokens=context,
        max_output_tokens=output,
        api_style="chat",
        structured_output_method="prompt_json",
    )


def test_route_key_normalizes_endpoint_without_credentials() -> None:
    assert normalize_base_url("HTTPS://Model.Example:443/v1/?token=secret") == "https://model.example/v1"
    key = model_route_key("Custom", "HTTPS://Model.Example:443/v1/?token=secret", "research-model")
    assert key == "custom|https://model.example/v1|research-model"
    assert "secret" not in key


def test_user_profile_wholly_overrides_builtin_profile(tmp_path: Path) -> None:
    builtin = tmp_path / "builtin.json"
    user = tmp_path / "user.json"
    save_model_profile_catalog(ModelProfileCatalog(profiles=[_profile(output=4096)]), builtin)
    save_model_profile_catalog(ModelProfileCatalog(profiles=[_profile(output=8192)]), user)

    resolved = resolve_model_profile(
        provider="custom",
        base_url="https://model.example/v1",
        model="research-model",
        user_path=user,
        builtin_path=builtin,
    )

    assert resolved.source == "user"
    assert resolved.profile is not None
    assert resolved.profile.max_output_tokens == 8192


def test_invalid_profile_catalog_is_rejected_instead_of_silently_overwritten(tmp_path: Path) -> None:
    invalid = tmp_path / "profiles.json"
    invalid.write_text('{"schema_version":1,"profiles":[{"provider":"custom"}]}', encoding="utf-8")

    with pytest.raises(ValidationError):
        resolve_model_profile(
            provider="custom",
            base_url="https://model.example/v1",
            model="research-model",
            user_path=invalid,
            builtin_path=tmp_path / "missing.json",
        )


def test_budget_counts_tool_schema_and_clamps_to_task_target() -> None:
    profile = _profile()
    manager = ModelBudgetManager(
        resolved_profile=ResolvedModelProfile(
            route_key=profile.route_key,
            source="user",
            profile=profile,
        )
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "analyze",
                "description": "x" * 1000,
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    budget = manager.budget(
        [ModelMessage(role="user", content="分析")],
        purpose=ModelRequestPurpose.EDA_PLANNING,
        tools=tools,
    )

    assert budget.tool_tokens > 200
    assert budget.input_tokens > budget.message_tokens
    assert budget.effective_output_tokens == 2048
    assert budget.safety_tokens == 4096


def test_unverified_route_uses_short_input_limit_without_inventing_capacity() -> None:
    manager = ModelBudgetManager(
        resolved_profile=ResolvedModelProfile(
            route_key="custom|https://unknown.invalid/v1|unknown",
            source="unverified",
        ),
        runtime=LLMRuntimeSettings(unverified_input_tokens=2048),
    )

    with pytest.raises(ModelContextLimitError, match="未验证模型"):
        manager.budget([ModelMessage(role="user", content="中" * 2500)])


def test_verified_route_fails_when_minimum_answer_cannot_fit() -> None:
    profile = _profile(context=3000, output=2048)
    manager = ModelBudgetManager(
        resolved_profile=ResolvedModelProfile(
            route_key=profile.route_key,
            source="user",
            profile=profile,
        )
    )

    with pytest.raises(ModelContextLimitError, match="minimum_output"):
        manager.budget(
            [ModelMessage(role="user", content="中" * 2500)],
            purpose=ModelRequestPurpose.EDA_PLANNING,
        )


def test_call_audit_keeps_usage_but_not_prompt_or_api_key(tmp_path: Path) -> None:
    profile = _profile()
    manager = ModelBudgetManager(
        resolved_profile=ResolvedModelProfile(
            route_key=profile.route_key,
            source="user",
            profile=profile,
        )
    )
    budget = manager.budget([ModelMessage(role="user", content="private prompt")])
    target = tmp_path / "model-calls.jsonl"
    ModelCallAudit(target).record(
        budget,
        response=SimpleNamespace(
            response_metadata={
                "finish_reason": "stop",
                "token_usage": {"prompt_tokens": 123, "completion_tokens": 45},
            }
        ),
        outcome="completed",
    )

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["actual_input_tokens"] == 123
    assert payload["actual_output_tokens"] == 45
    assert payload["finish_reason"] == "stop"
    assert "private prompt" not in target.read_text(encoding="utf-8")
    assert "api_key" not in payload


def test_planning_recovery_is_marked_as_compacted_in_budget_and_audit(tmp_path: Path) -> None:
    profile = _profile()
    manager = ModelBudgetManager(
        resolved_profile=ResolvedModelProfile(
            route_key=profile.route_key,
            source="user",
            profile=profile,
        )
    )
    budget = manager.budget(
        [ModelMessage(role="user", content="minimal")],
        purpose=ModelRequestPurpose.EDA_PLANNING_RECOVERY,
    )
    target = tmp_path / "model-calls.jsonl"

    ModelCallAudit(target).record(budget, outcome="completed")

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert budget.effective_output_tokens == 1024
    assert payload["compaction_level"] == 1
    assert payload["recovery"] == "minimal_eda_plan_intent"


def test_legacy_llm_environment_migrates_once_without_storing_secret(monkeypatch, tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "VPP_LLM_PROVIDER=custom\n"
        "VPP_LLM_BASE_URL=https://legacy.example/v1\n"
        "VPP_LLM_API_KEY=do-not-copy\n"
        "VPP_LLM_MODEL=legacy-model\n"
        "VPP_LLM_API_STYLE=responses\n"
        "VPP_LLM_STRUCTURED_OUTPUT_METHOD=json_schema\n"
        "VPP_LLM_CONTEXT_WINDOW_TOKENS=64000\n"
        "VPP_LLM_MAX_OUTPUT_TOKENS=8000\n"
        "VPP_LLM_TIMEOUT_SECONDS=180\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PRICE_RESEARCH_APP_DATA_DIRECTORY", str(tmp_path / "app-data"))
    monkeypatch.setenv("VPP_LLM_PROVIDER", "custom")
    monkeypatch.setenv("VPP_LLM_BASE_URL", "https://legacy.example/v1")
    monkeypatch.setenv("VPP_LLM_API_KEY", "do-not-copy")
    monkeypatch.setenv("VPP_LLM_MODEL", "legacy-model")
    monkeypatch.setenv("VPP_LLM_API_STYLE", "responses")
    monkeypatch.setenv("VPP_LLM_STRUCTURED_OUTPUT_METHOD", "json_schema")
    monkeypatch.setenv("VPP_LLM_CONTEXT_WINDOW_TOKENS", "64000")
    monkeypatch.setenv("VPP_LLM_MAX_OUTPUT_TOKENS", "8000")
    monkeypatch.setenv("VPP_LLM_TIMEOUT_SECONDS", "180")
    monkeypatch.setattr("app.config.runtime_env_file", lambda: env_path)
    get_settings.cache_clear()
    get_application_settings.cache_clear()
    try:
        settings = get_settings()
        app_content = (tmp_path / "app-data" / "settings.json").read_text(encoding="utf-8")
        profile_content = (tmp_path / "app-data" / "model_profiles.json").read_text(encoding="utf-8")

        assert settings.llm_context_window_tokens == 64000
        assert settings.llm_max_output_tokens == 8000
        assert settings.llm_timeout_seconds == 180
        assert settings.llm_api_style == "responses"
        assert get_application_settings().legacy_llm_env_migration_version == 1
        assert "do-not-copy" not in app_content
        assert "do-not-copy" not in profile_content
    finally:
        get_settings.cache_clear()
        get_application_settings.cache_clear()
