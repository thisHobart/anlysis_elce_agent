"""Verify structured output uses provider function calling."""

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from app.config import Settings
from app.llm.gateway import ModelResponseError
from app.llm.openai_compatible import OpenAICompatibleGateway


class StructuredAnswer(BaseModel):
    value: str


class BoundModel:
    def __init__(self, schema, owner):
        self.schema = schema
        self.owner = owner

    def invoke(self, messages):
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
        )

    def with_structured_output(self, schema, *, method, include_raw):
        self.method = method
        self.include_raw = include_raw
        return BoundModel(schema, self)

    def invoke(self, messages):
        self.last_messages = messages
        return self.response

    def bind_tools(self, tools, *, tool_choice, parallel_tool_calls):
        self.bound_tools = tools
        self.tool_choice = tool_choice
        self.parallel_tool_calls = parallel_tool_calls

        class ProposedCalls:
            def invoke(_self, messages):
                self.last_messages = messages
                return self.response

        return ProposedCalls()


def test_structured_output_uses_function_calling():
    gateway = OpenAICompatibleGateway(
        Settings(
            llm_base_url="http://model.invalid/v1",
            llm_model="test-model",
        )
    )
    model = RecordingModel()
    gateway._model = model

    result = gateway.invoke_structured(messages=[("human", "test")], schema=StructuredAnswer)

    assert result.value == "ok"
    assert model.method == "function_calling"
    assert model.include_raw is True


def test_research_function_calls_are_proposed_without_execution():
    gateway = OpenAICompatibleGateway(
        Settings(
            llm_base_url="http://model.invalid/v1",
            llm_model="test-model",
        )
    )
    model = RecordingModel()
    gateway._model = model
    tools = [
        {
            "type": "function",
            "function": {
                "name": "price_descriptive_distribution",
                "description": "计算电价分布。",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    calls = gateway.invoke_tool_calls(messages=[("human", "分析电价分布")], tools=tools)

    assert [call.name for call in calls] == ["price_descriptive_distribution"]
    assert calls[0].call_id == "call-1"
    assert model.bound_tools == tools
    assert model.tool_choice == "required"
    assert model.parallel_tool_calls is True


def test_custom_openai_compatible_endpoint_disables_thinking_in_request_body():
    gateway = OpenAICompatibleGateway(
        Settings(
            llm_base_url="http://127.0.0.1:8000/v1",
            llm_model="Qwen3-8B",
        )
    )

    assert gateway._model_options()["extra_body"] == {
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }


@pytest.mark.parametrize("model_name", ["deepseek-chat", "deepseek-reasoner", "future-deepseek-model"])
def test_all_deepseek_models_disable_thinking_with_request_parameter(model_name: str):
    gateway = OpenAICompatibleGateway(
        Settings(
            llm_base_url="https://api.deepseek.com/v1",
            llm_model=model_name,
        )
    )

    assert gateway._model_options()["extra_body"] == {"enable_thinking": False}


def test_gateway_rejects_reasoning_content_even_when_tool_call_is_valid():
    gateway = OpenAICompatibleGateway(
        Settings(
            llm_base_url="http://model.invalid/v1",
            llm_model="test-model",
        )
    )
    gateway._model = RecordingModel(reasoning_content="private reasoning")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "price_descriptive_distribution",
                "description": "计算电价分布。",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    with pytest.raises(ModelResponseError, match="思考模式禁用失败"):
        gateway.invoke_tool_calls(messages=[("human", "test")], tools=tools)


def test_gateway_rejects_nonempty_think_block_in_structured_response():
    gateway = OpenAICompatibleGateway(
        Settings(
            llm_base_url="http://model.invalid/v1",
            llm_model="test-model",
        )
    )
    gateway._model = RecordingModel(content="<think>still thinking</think>")

    with pytest.raises(ModelResponseError, match="思考模式禁用失败"):
        gateway.invoke_structured(messages=[("human", "test")], schema=StructuredAnswer)
