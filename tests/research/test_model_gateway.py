"""Verify the native model protocol, optional degradation, and reasoning policy."""

from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from app.config import Settings
from app.llm import compat
from app.llm.budget import ModelRequestPurpose
from app.llm.gateway import (
    ModelConfigurationError,
    ModelMessage,
    ModelOutputTruncatedError,
    ModelProtocolError,
    ModelResponseError,
    ModelTransientError,
)
from app.llm.openai_compatible import ResearchModelGateway

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "price_descriptive_distribution",
            "description": "计算电价分布。",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


class StructuredAnswer(BaseModel):
    value: str


class RejectedRequest(Exception):
    """Stand-in for the provider 400 raised when a request field is unsupported."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.status_code = 400


def _settings(**overrides) -> Settings:
    return Settings(
        _env_file=None,
        llm_base_url=overrides.pop("llm_base_url", "http://model.invalid/v1"),
        llm_model=overrides.pop("llm_model", "test-model"),
        **overrides,
    )


def _messages(content: str = "test") -> list[ModelMessage]:
    return [ModelMessage(role="user", content=content)]


class BoundModel:
    def __init__(self, schema, owner):
        self.schema = schema
        self.owner = owner

    def invoke(self, messages):
        if self.owner.structured_error is not None:
            raise self.owner.structured_error
        self.owner.last_messages = messages
        return {
            "raw": self.owner.response,
            "parsed": self.schema(value="ok"),
            "parsing_error": None,
        }


class RecordingModel:
    def __init__(self, *, reasoning_content: str = "", content: str = "") -> None:
        self.method = None
        self.include_raw = None
        self.bound_tools = None
        self.tool_choice = None
        self.parallel_tool_calls = None
        self.last_messages = None
        self.response = SimpleNamespace(
            content=content,
            additional_kwargs={"reasoning_content": reasoning_content},
            tool_calls=[
                {
                    "name": "price_descriptive_distribution",
                    "args": {},
                    "id": "call-1",
                }
            ],
            response_metadata={},
        )
        self.structured_error = None

    def with_structured_output(self, schema, *, method, include_raw):
        self.method = method
        self.include_raw = include_raw
        return BoundModel(schema, self)

    def invoke(self, messages):
        self.last_messages = messages
        return self.response

    def bind_tools(self, tools, **options):
        self.bound_tools = tools
        self.tool_choice = options.get("tool_choice")
        self.parallel_tool_calls = options.get("parallel_tool_calls")

        class ProposedCalls:
            def invoke(_self, messages):
                self.last_messages = messages
                return self.response

        return ProposedCalls()


class BindableRecordingModel(RecordingModel):
    def __init__(self, *, content: str) -> None:
        super().__init__(content=content)
        self.bound_options: list[dict[str, int]] = []

    def bind(self, **options):
        self.bound_options.append(options)
        return self


def test_structured_output_uses_native_function_calling_and_typed_messages():
    gateway = ResearchModelGateway(_settings())
    model = RecordingModel()
    gateway._model = model

    result = gateway.invoke_structured(messages=_messages(), schema=StructuredAnswer)

    assert result.value == "ok"
    assert model.method == "function_calling"
    assert model.include_raw is True
    assert isinstance(model.last_messages[0], HumanMessage)


def test_parseable_structured_output_is_rejected_when_provider_reports_truncation():
    gateway = ResearchModelGateway(_settings())
    model = RecordingModel()
    model.response.response_metadata = {"finish_reason": "length"}
    gateway._model = model

    with pytest.raises(ModelOutputTruncatedError, match="响应不完整"):
        gateway.invoke_structured(messages=_messages(), schema=StructuredAnswer)


def test_text_output_is_rejected_when_provider_reports_truncation():
    gateway = ResearchModelGateway(_settings())
    model = RecordingModel(content="partial answer")
    model.response.response_metadata = {"finish_reason": "length"}
    gateway._model = model

    with pytest.raises(ModelOutputTruncatedError, match="响应不完整"):
        gateway.invoke_text(messages=_messages())


def test_text_output_limit_is_bound_per_request_purpose() -> None:
    gateway = ResearchModelGateway(_settings())
    model = BindableRecordingModel(content="ok")
    gateway._model = model

    result = gateway.invoke_text(
        messages=_messages(),
        purpose=ModelRequestPurpose.DIALOGUE,
    )

    assert result == "ok"
    assert model.bound_options == [{"max_tokens": 1024}]
    assert "max_tokens" not in gateway._model_options()


def test_selector_purpose_overrides_enabled_qwen_reasoning_per_request() -> None:
    gateway = ResearchModelGateway(_settings(llm_provider="qwen", llm_reasoning_effort="high"))
    model = BindableRecordingModel(content="ok")
    gateway._model = model

    gateway.invoke_text(
        messages=_messages(),
        purpose=ModelRequestPurpose.EDA_PLANNING,
    )

    assert gateway._model_options()["extra_body"] == {"enable_thinking": True}
    assert model.bound_options == [{"max_tokens": 2048, "extra_body": {"enable_thinking": False}}]


def test_tool_calls_are_rejected_as_a_batch_when_provider_reports_truncation():
    gateway = ResearchModelGateway(_settings())
    model = RecordingModel()
    model.response.response_metadata = {"finish_reason": "length"}
    gateway._model = model

    with pytest.raises(ModelOutputTruncatedError, match="响应不完整"):
        gateway.invoke_tool_calls(messages=_messages(), tools=TOOLS)


def test_sdk_length_exception_is_exposed_as_output_truncation() -> None:
    LengthFinishReasonError = type("LengthFinishReasonError", (RuntimeError,), {})
    gateway = ResearchModelGateway(_settings())
    model = RecordingModel()
    model.structured_error = LengthFinishReasonError("length limit was reached")
    gateway._model = model

    with pytest.raises(ModelOutputTruncatedError, match="token 限制"):
        gateway.invoke_structured(messages=_messages(), schema=StructuredAnswer)


def test_provider_500_is_exposed_as_invocation_scoped_transient_failure() -> None:
    server_error = RuntimeError("internal server error")
    server_error.status_code = 500
    gateway = ResearchModelGateway(_settings())
    model = RecordingModel()
    model.structured_error = server_error
    gateway._model = model

    with pytest.raises(ModelTransientError, match="暂时不可用"):
        gateway.invoke_structured(messages=_messages(), schema=StructuredAnswer)


def test_structured_output_method_follows_the_configured_native_protocol():
    """A Gemini/Claude endpoint reached via an OpenAI-compatible proxy can select json_schema."""

    gateway = ResearchModelGateway(_settings(llm_structured_output_method="json_schema"))
    model = RecordingModel()
    gateway._model = model

    result = gateway.invoke_structured(messages=_messages(), schema=StructuredAnswer)

    assert result.value == "ok"
    assert model.method == "json_schema"


def test_structured_output_observer_receives_function_arguments_without_reasoning():
    observed = []
    gateway = ResearchModelGateway(
        _settings(),
        structured_output_observer=observed.append,
    )
    model = RecordingModel(reasoning_content="private reasoning")
    gateway._model = model

    result = gateway.invoke_structured(messages=_messages(), schema=StructuredAnswer)

    assert result.value == "ok"
    assert observed == [
        {
            "provider": "custom",
            "transport": "openai_compatible",
            "model": "test-model",
            "structured_output_method": "function_calling",
            "schema_enforcement": "provider",
            "raw_content": "",
            "tool_calls": [
                {
                    "name": "price_descriptive_distribution",
                    "arguments": {},
                    "call_id": "call-1",
                }
            ],
            "parsed": {"value": "ok"},
            "parsing_error": None,
            "response_metadata": {},
        }
    ]
    assert "private reasoning" not in str(observed)


def test_prompt_json_injects_schema_and_uses_json_mode_for_cherry_compatibility():
    gateway = ResearchModelGateway(_settings(llm_structured_output_method="prompt_json"))
    model = RecordingModel()
    gateway._model = model

    result = gateway.invoke_structured(messages=_messages(), schema=StructuredAnswer)

    assert result.value == "ok"
    assert model.method == "json_mode"
    assert isinstance(model.last_messages[0], SystemMessage)
    assert '"value"' in model.last_messages[0].content
    assert "不要返回 Markdown" in model.last_messages[0].content
    assert isinstance(model.last_messages[-1], HumanMessage)


def test_prompt_json_schema_failure_is_repairable_response_error_not_protocol_error():
    class InvalidJsonShapeModel(RecordingModel):
        def with_structured_output(self, schema, *, method, include_raw):
            self.method = method

            class Bound:
                def invoke(_self, messages):
                    self.last_messages = messages
                    return {
                        "raw": SimpleNamespace(
                            content='{"invented":"field"}',
                            additional_kwargs={},
                            tool_calls=[],
                        ),
                        "parsed": None,
                        "parsing_error": "value is required",
                    }

            return Bound()

    gateway = ResearchModelGateway(_settings(llm_structured_output_method="prompt_json"))
    model = InvalidJsonShapeModel()
    gateway._model = model

    with pytest.raises(ModelResponseError, match="Cherry 兼容 JSON"):
        gateway.invoke_structured(messages=_messages(), schema=StructuredAnswer)
    assert model.method == "json_mode"


def test_research_function_calls_are_proposed_without_execution():
    gateway = ResearchModelGateway(_settings())
    model = RecordingModel()
    gateway._model = model

    calls = gateway.invoke_tool_calls(messages=_messages("分析电价分布"), tools=TOOLS)

    assert [call.name for call in calls] == ["price_descriptive_distribution"]
    assert calls[0].call_id == "call-1"
    assert model.bound_tools == TOOLS
    assert model.tool_choice == "required"
    assert model.parallel_tool_calls is True


def test_prompt_json_applies_local_schema_and_whitelist_to_research_function_selection():
    class PromptToolModel(RecordingModel):
        def with_structured_output(self, schema, *, method, include_raw):
            self.method = method
            self.include_raw = include_raw

            class Bound:
                def invoke(_self, messages):
                    self.last_messages = messages
                    return {
                        "raw": SimpleNamespace(
                            content=('{"calls":[{"name":"price_descriptive_distribution","arguments":{}}]}'),
                            additional_kwargs={},
                            tool_calls=[],
                            response_metadata={},
                        ),
                        "parsed": schema.model_validate(
                            {
                                "calls": [
                                    {
                                        "name": "price_descriptive_distribution",
                                        "arguments": {},
                                    }
                                ]
                            }
                        ),
                        "parsing_error": None,
                    }

            return Bound()

    gateway = ResearchModelGateway(_settings(llm_structured_output_method="prompt_json"))
    model = PromptToolModel()
    gateway._model = model

    calls = gateway.invoke_tool_calls(messages=_messages("分析电价分布"), tools=TOOLS)

    assert [call.name for call in calls] == ["price_descriptive_distribution"]
    assert calls[0].call_id == "prompt-json-1"
    assert model.bound_tools is None
    assert model.method == "json_mode"
    assert isinstance(model.last_messages[0], SystemMessage)
    assert "研究函数选择兼容协议" in model.last_messages[0].content
    assert "price_descriptive_distribution" in model.last_messages[0].content


def test_prompt_json_research_function_selection_rejects_names_outside_whitelist():
    class UnknownPromptToolModel(RecordingModel):
        def with_structured_output(self, schema, *, method, include_raw):
            class Bound:
                def invoke(_self, messages):
                    return {
                        "raw": SimpleNamespace(
                            content='{"calls":[{"name":"delete_database","arguments":{}}]}',
                            additional_kwargs={},
                            tool_calls=[],
                            response_metadata={},
                        ),
                        "parsed": schema.model_validate({"calls": [{"name": "delete_database", "arguments": {}}]}),
                        "parsing_error": None,
                    }

            return Bound()

    gateway = ResearchModelGateway(_settings(llm_structured_output_method="prompt_json"))
    gateway._model = UnknownPromptToolModel()

    with pytest.raises(ModelResponseError, match="未提供的研究函数"):
        gateway.invoke_tool_calls(messages=_messages("分析电价分布"), tools=TOOLS)


def test_custom_endpoint_sends_no_reasoning_control_by_default():
    gateway = ResearchModelGateway(_settings())

    options = gateway._model_options()
    assert options["use_responses_api"] is False
    assert "extra_body" not in options
    assert "reasoning_effort" not in options
    assert "reasoning" not in options


@pytest.mark.parametrize("model_name", ["deepseek-chat", "deepseek-reasoner", "future-deepseek-model"])
def test_all_deepseek_models_disable_thinking_with_request_parameter(model_name: str):
    gateway = ResearchModelGateway(_settings(llm_provider="deepseek", llm_model=model_name))

    assert gateway._model_options()["extra_body"] == {"thinking": {"type": "disabled"}}


def test_qwen_chat_uses_its_documented_thinking_toggle():
    gateway = ResearchModelGateway(_settings(llm_provider="qwen"))

    assert gateway._model_options()["extra_body"] == {"enable_thinking": False}


@pytest.mark.parametrize("provider", ["deepseek", "qwen"])
def test_provider_responses_api_uses_reasoning_effort(provider: str):
    gateway = ResearchModelGateway(_settings(llm_provider=provider, llm_api_style="responses"))

    options = gateway._model_options()
    assert options["use_responses_api"] is True
    assert options["reasoning"] == {"effort": "none"}
    assert "extra_body" not in options


@pytest.mark.parametrize(
    ("api_style", "expected"),
    [
        ("chat", {"reasoning_effort": "none"}),
        ("responses", {"reasoning": {"effort": "none"}}),
    ],
)
def test_custom_reasoning_effort_is_only_sent_when_backend_configures_it(
    api_style: str,
    expected: dict,
):
    gateway = ResearchModelGateway(
        _settings(
            llm_provider="custom",
            llm_api_style=api_style,
            llm_reasoning_effort="none",
        )
    )

    options = gateway._model_options()
    for key, value in expected.items():
        assert options[key] == value


def test_strict_policy_rejects_reasoning_content_even_when_tool_call_is_valid():
    gateway = ResearchModelGateway(_settings(llm_thinking_policy="reject"))
    gateway._model = RecordingModel(reasoning_content="private reasoning")

    with pytest.raises(ModelResponseError, match="思考模式禁用失败"):
        gateway.invoke_tool_calls(messages=_messages(), tools=TOOLS)


def test_strict_policy_rejects_nonempty_think_block_in_structured_response():
    gateway = ResearchModelGateway(_settings(llm_thinking_policy="reject"))
    gateway._model = RecordingModel(content="<think>still thinking</think>")

    with pytest.raises(ModelResponseError, match="思考模式禁用失败"):
        gateway.invoke_structured(messages=_messages(), schema=StructuredAnswer)


def test_default_policy_discards_reasoning_instead_of_failing():
    gateway = ResearchModelGateway(_settings())
    gateway._model = RecordingModel(
        reasoning_content="private reasoning",
        content="<think>still thinking</think>可以开始分析。",
    )

    assert gateway.invoke_text(messages=_messages()) == "可以开始分析。"


def test_structured_output_does_not_switch_protocol_when_function_calling_is_refused():
    class NoStructuredToolsModel(RecordingModel):
        def with_structured_output(self, schema, *, method, include_raw):
            raise RejectedRequest("Thinking mode does not support this tool_choice")

    gateway = ResearchModelGateway(_settings(llm_api_style="responses"))
    gateway._model = NoStructuredToolsModel()

    with pytest.raises(ModelProtocolError, match="API Style=responses"):
        gateway.invoke_structured(messages=_messages(), schema=StructuredAnswer)


def test_structured_output_never_recovers_json_from_prose():
    class ProseModel(RecordingModel):
        def with_structured_output(self, schema, *, method, include_raw):
            class Bound:
                def invoke(_self, messages):
                    return {
                        "raw": SimpleNamespace(
                            content='<think>weighing</think>\n```json\n{"value": "ok"}\n```',
                            additional_kwargs={},
                            tool_calls=[],
                        ),
                        "parsed": None,
                        "parsing_error": "no tool call returned",
                    }

            return Bound()

    gateway = ResearchModelGateway(_settings())
    gateway._model = ProseModel()

    with pytest.raises(ModelProtocolError, match="只有显式配置 prompt_json"):
        gateway.invoke_structured(messages=_messages(), schema=StructuredAnswer)


def test_refused_provider_reasoning_option_is_dropped_and_retried():
    """An auto-added provider option may degrade; an explicit custom option may not."""

    class PickyModel(RecordingModel):
        def __init__(self, gateway) -> None:
            super().__init__()
            self.gateway = gateway
            self.attempts = 0

        def with_structured_output(self, schema, *, method, include_raw):
            self.attempts += 1
            if "provider_reasoning" not in self.gateway._disabled:
                raise RejectedRequest("Unrecognized request argument supplied: enable_thinking")
            return BoundModel(schema, self)

    gateway = ResearchModelGateway(_settings(llm_provider="qwen"))
    model = PickyModel(gateway)
    gateway._get_model = lambda: model

    result = gateway.invoke_structured(messages=_messages(), schema=StructuredAnswer)

    assert result.value == "ok"
    assert model.attempts == 2
    assert "extra_body" not in gateway._model_options()


def test_parallel_tool_calls_is_optional_but_tool_choice_remains_required():
    class ParallelPickyModel(RecordingModel):
        def __init__(self) -> None:
            super().__init__()
            self.bind_options: list[dict] = []

        def bind_tools(self, tools, **options):
            self.bind_options.append(options)
            if "parallel_tool_calls" in options:
                raise RejectedRequest("parallel_tool_calls is not supported")
            return super().bind_tools(tools, **options)

    gateway = ResearchModelGateway(_settings())
    model = ParallelPickyModel()
    gateway._model = model

    calls = gateway.invoke_tool_calls(messages=_messages(), tools=TOOLS)

    assert len(calls) == 1
    assert model.bind_options == [
        {"tool_choice": "required", "parallel_tool_calls": True},
        {"tool_choice": "required"},
    ]


def test_tool_calls_fail_closed_when_native_tools_are_unsupported():
    class NoToolsModel(RecordingModel):
        def bind_tools(self, tools, **options):
            raise RejectedRequest("This model does not support tools or tool_choice")

    gateway = ResearchModelGateway(_settings(llm_provider="custom"))
    gateway._model = NoToolsModel(content='{"calls": []}')

    with pytest.raises(ModelProtocolError, match="Provider=custom"):
        gateway.invoke_tool_calls(messages=_messages("分析电价分布"), tools=TOOLS)
    assert "tools" not in gateway._disabled
    assert "tool_choice" not in gateway._disabled


def test_tool_calls_reject_function_names_not_in_the_request():
    gateway = ResearchModelGateway(_settings())
    model = RecordingModel()
    model.response.tool_calls[0]["name"] = "unregistered_function"
    gateway._model = model

    with pytest.raises(ModelResponseError, match="未提供的研究函数"):
        gateway.invoke_tool_calls(messages=_messages(), tools=TOOLS)


def test_provider_shaped_tuple_messages_are_rejected_at_the_boundary():
    gateway = ResearchModelGateway(_settings())
    gateway._model = RecordingModel()

    with pytest.raises(ModelConfigurationError, match="ModelMessage"):
        gateway.invoke_text(messages=[("human", "test")])  # type: ignore[list-item]


def test_double_encoded_response_body_is_unwrapped():
    assert compat.unwrap_double_encoded_body(b'"{\\"choices\\": []}"') == b'{"choices": []}'
    assert compat.unwrap_double_encoded_body(b'{"choices": []}') is None
    assert compat.unwrap_double_encoded_body(b"not json") is None


def test_unsupported_feature_only_reports_optional_fields():
    rejected_core = RejectedRequest("Thinking mode does not support this tool_choice")
    rejected_optional = RejectedRequest("parallel_tool_calls is not supported")

    assert (
        compat.unsupported_feature(
            rejected_core,
            set(),
            compat.OPTIONAL_TOOL_CALL_FEATURES,
        )
        is None
    )
    assert (
        compat.unsupported_feature(
            rejected_optional,
            set(),
            compat.OPTIONAL_TOOL_CALL_FEATURES,
        )
        == "parallel_tool_calls"
    )
