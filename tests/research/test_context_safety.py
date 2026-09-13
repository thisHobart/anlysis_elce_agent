import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from app.llm.context_safety import (
    conservative_token_count,
    ensure_complete_response,
    is_output_truncation_error,
    structured_request_budget,
)
from app.llm.gateway import (
    ModelContextLimitError,
    ModelMessage,
    ModelOutputTruncatedError,
    ModelResponseError,
    ModelTransientError,
)
from app.research.agent.errors import (
    ResearchModelContextLimitError,
    ResearchModelOutputTruncatedError,
    ResearchModelSchemaError,
    ResearchModelTransientError,
)
from app.research.agent.orchestrator import DialogueDecision, ModelResearchDialogue


class _Schema(BaseModel):
    value: str


def test_budget_includes_messages_schema_output_and_safety_reserves() -> None:
    counter = lambda value: len(value)
    messages = [ModelMessage(role="system", content="abc"), ModelMessage(role="user", content="中文")]

    measured = structured_request_budget(
        messages,
        _Schema,
        context_window_tokens=100_000,
        reserved_output_tokens=500,
        safety_tokens=100,
        token_counter=counter,
    )

    assert measured.input_tokens > len("abc中文")
    assert measured.required_tokens == measured.input_tokens + 600
    assert measured.fits


def test_budget_rejects_complete_request_instead_of_silently_clipping_content() -> None:
    content = "人工智能" * 100

    with pytest.raises(ModelContextLimitError, match="完整结构化请求超过上下文限制"):
        structured_request_budget(
            [ModelMessage(role="user", content=content)],
            _Schema,
            context_window_tokens=100,
            reserved_output_tokens=20,
            safety_tokens=10,
            token_counter=len,
        )

    assert content.endswith("智能")


@pytest.mark.parametrize("reason", ["length", "MAX_TOKENS", "max-output-tokens"])
def test_truncated_finish_reason_is_rejected_even_if_payload_was_parseable(reason: str) -> None:
    response = SimpleNamespace(response_metadata={"finish_reason": reason})

    with pytest.raises(ModelOutputTruncatedError, match="响应不完整"):
        ensure_complete_response(response)


def test_normal_finish_reason_is_accepted() -> None:
    ensure_complete_response(SimpleNamespace(response_metadata={"finish_reason": "STOP"}))


def test_conservative_counter_does_not_split_or_mutate_unicode() -> None:
    text = "人工智能⚡AEMO-123"
    assert conservative_token_count(text) > 0
    assert text == "人工智能⚡AEMO-123"


def test_sdk_length_exception_is_recognized_before_a_response_envelope_exists() -> None:
    LengthFinishReasonError = type("LengthFinishReasonError", (RuntimeError,), {})

    assert is_output_truncation_error(
        LengthFinishReasonError("Could not parse response content as the length limit was reached")
    )
    assert not is_output_truncation_error(RuntimeError("connection reset"))


def _dialogue_kwargs() -> dict[str, object]:
    return {
        "question": "继续分析",
        "status": "awaiting_user",
        "config": None,
        "plan": None,
        "data_profile": None,
        "quality_report": None,
        "summary": None,
        "evaluation": None,
        "history": [],
        "available_skills": [],
    }


@pytest.mark.parametrize(
    ("gateway_error", "research_error"),
    [
        (ModelContextLimitError("too large"), ResearchModelContextLimitError),
        (ModelResponseError("bad schema"), ResearchModelSchemaError),
        (ModelTransientError("try later"), ResearchModelTransientError),
    ],
)
def test_dialogue_preserves_model_failure_categories(gateway_error, research_error) -> None:
    class Gateway:
        enabled = True
        model_name = "test"

        def invoke_structured(self, **_kwargs):
            raise gateway_error

    dialogue = ModelResearchDialogue(gateway=Gateway())

    with pytest.raises(research_error):
        dialogue.decide(**_dialogue_kwargs())


def test_dialogue_retries_a_truncated_response_once_with_changed_bounded_context() -> None:
    requests: list[dict[str, object]] = []

    class Gateway:
        enabled = True
        model_name = "test"

        def invoke_structured(self, *, messages, schema):
            assert schema is DialogueDecision
            requests.append(json.loads(messages[-1].content))
            if len(requests) == 1:
                raise ModelOutputTruncatedError("length")
            return DialogueDecision(intent="discussion", response="可以继续。")

    dialogue = ModelResearchDialogue(gateway=Gateway())
    result = dialogue.decide(**_dialogue_kwargs())

    assert result.response == "可以继续。"
    assert len(requests) == 2
    assert "retry_reason" not in requests[0]
    assert "retry_reason" in requests[1]
    assert "allowed_functions" in requests[0]
    assert "allowed_functions" not in requests[1]
    assert "allowed_function_names" in requests[1]


def test_dialogue_reports_truncation_after_its_single_changed_retry() -> None:
    class Gateway:
        enabled = True
        model_name = "test"

        def invoke_structured(self, **_kwargs):
            raise ModelOutputTruncatedError("length")

    dialogue = ModelResearchDialogue(gateway=Gateway())

    with pytest.raises(ResearchModelOutputTruncatedError):
        dialogue.decide(**_dialogue_kwargs())
