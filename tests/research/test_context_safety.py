from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from app.llm.context_safety import (
    conservative_token_count,
    ensure_complete_response,
    structured_request_budget,
)
from app.llm.gateway import ModelContextLimitError, ModelMessage, ModelOutputTruncatedError


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
