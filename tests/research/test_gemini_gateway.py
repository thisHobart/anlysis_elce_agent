"""Verify native Gemini routing and structured output behavior."""

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from app.config import Settings
from app.llm.factory import build_model_gateway
from app.llm.gateway import ModelConfigurationError, ModelMessage
from app.llm.gemini import GeminiModelGateway
from app.llm.openai_compatible import ResearchModelGateway


class StructuredAnswer(BaseModel):
    value: str


def _settings(**overrides) -> Settings:
    return Settings(
        _env_file=None,
        llm_provider=overrides.pop("llm_provider", "gemini"),
        llm_base_url=overrides.pop("llm_base_url", "http://127.0.0.1:23333/v1"),
        llm_api_key=overrides.pop("llm_api_key", "test-key"),
        llm_model=overrides.pop("llm_model", "gemini-test"),
        **overrides,
    )


class RecordingGeminiModel:
    def __init__(self) -> None:
        self.method = None
        self.include_raw = None
        self.last_messages = None

    def with_structured_output(self, schema, *, method, include_raw):
        self.method = method
        self.include_raw = include_raw

        class Bound:
            def invoke(_self, messages):
                self.last_messages = messages
                return {
                    "raw": SimpleNamespace(
                        content='{"value":"ok"}',
                        additional_kwargs={},
                        tool_calls=[],
                        response_metadata={"finish_reason": "STOP", "secret": "discard"},
                    ),
                    "parsed": schema(value="ok"),
                    "parsing_error": None,
                }

        return Bound()


def test_factory_selects_native_gemini_only_for_gemini_provider():
    assert isinstance(build_model_gateway(_settings()), GeminiModelGateway)
    assert isinstance(
        build_model_gateway(_settings(llm_provider="custom")),
        ResearchModelGateway,
    )


def test_gemini_client_does_not_reuse_openai_proxy_url():
    gateway = GeminiModelGateway(_settings())

    options = gateway._model_options()

    assert "base_url" not in options
    assert options["vertexai"] is False
    assert options["api_key"] == "test-key"


def test_gemini_structured_output_always_uses_native_json_schema_and_exposes_raw():
    observed: list[dict] = []
    gateway = GeminiModelGateway(
        _settings(llm_structured_output_method="function_calling"),
        structured_output_observer=observed.append,
    )
    model = RecordingGeminiModel()
    gateway._model = model

    result = gateway.invoke_structured(
        messages=[ModelMessage(role="user", content="test")],
        schema=StructuredAnswer,
    )

    assert result.value == "ok"
    assert model.method == "json_schema"
    assert model.include_raw is True
    assert observed == [
        {
            "provider": "gemini",
            "transport": "google_native",
            "model": "gemini-test",
            "structured_output_method": "json_schema",
            "schema_enforcement": "provider",
            "raw_content": '{"value":"ok"}',
            "tool_calls": [],
            "parsed": {"value": "ok"},
            "parsing_error": None,
            "response_metadata": {"finish_reason": "STOP"},
        }
    ]


def test_gemini_rejects_cherry_model_prefix_with_actionable_error():
    gateway = GeminiModelGateway(_settings(llm_model="vertexai:gemini-test"))

    with pytest.raises(ModelConfigurationError, match="vertexai: 模型前缀"):
        gateway._model_options()


def test_vertex_configuration_uses_project_and_adc_instead_of_api_key():
    gateway = GeminiModelGateway(
        _settings(
            llm_google_vertexai=True,
            llm_google_project="example-project",
            llm_google_location="global",
        )
    )

    options = gateway._model_options()

    assert options["vertexai"] is True
    assert options["project"] == "example-project"
    assert options["location"] == "global"
    assert "api_key" not in options
