"""Offline contract checks for the opt-in live protocol acceptance probe."""

from __future__ import annotations

import json

import pytest

from app.config import Settings
from app.llm.gateway import ModelToolCall, ModelToolTurn
from scripts.probe_dynamic_tool_turn import _turn_payload, _validate_case, _write_report


def _settings(**updates) -> Settings:
    values = {
        "llm_provider": "custom",
        "llm_base_url": "https://model.invalid/v1",
        "llm_api_key": "secret",
        "llm_model": "test-model",
        "llm_structured_output_method": "function_calling",
    }
    values.update(updates)
    return Settings(_env_file=None, **values)


def test_live_probe_cases_require_the_expected_transport_mode():
    _validate_case("openai-native", _settings())
    _validate_case("gemini-native", _settings(llm_provider="gemini"))
    _validate_case("prompt-json", _settings(llm_structured_output_method="prompt_json"))

    with pytest.raises(ValueError, match="gemini-native"):
        _validate_case("gemini-native", _settings())
    with pytest.raises(ValueError, match="openai-native"):
        _validate_case("openai-native", _settings(llm_structured_output_method="prompt_json"))
    with pytest.raises(ValueError, match="prompt-json"):
        _validate_case("prompt-json", _settings())


def test_live_probe_report_is_deliberately_redacted(tmp_path):
    turn = ModelToolTurn(
        content="provider response that must not be persisted",
        tool_calls=[ModelToolCall(name="price_descriptive_distribution", arguments={}, call_id="call-1")],
    )
    payload = _turn_payload(turn)

    assert "content" not in payload
    assert payload["content_present"] is True
    assert payload["content_length"] > 0

    destination = tmp_path / "probe.json"
    _write_report(destination, {"turn": payload})
    persisted = json.loads(destination.read_text(encoding="utf-8"))
    assert "provider response" not in json.dumps(persisted)
