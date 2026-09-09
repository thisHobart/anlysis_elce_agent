"""Provider-neutral preflight accounting and completion guards for model calls."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from app.llm.gateway import (
    ModelContextLimitError,
    ModelMessage,
    ModelOutputTruncatedError,
)

TokenCounter = Callable[[str], int]


def conservative_token_count(text: str) -> int:
    """Estimate tokens without assuming that a custom endpoint uses a known tokenizer.

    ASCII runs are approximated at four bytes per token while every non-ASCII code point
    counts as one. The explicit safety reserve covers message framing and tokenizer drift.
    Provider adapters may inject an exact counter into ``structured_request_budget``.
    """

    ascii_bytes = sum(1 for character in text if ord(character) < 128)
    non_ascii = len(text) - ascii_bytes
    return max(1, math.ceil(ascii_bytes / 4) + non_ascii)


@dataclass(frozen=True)
class RequestBudget:
    input_tokens: int
    reserved_output_tokens: int
    safety_tokens: int
    context_window_tokens: int

    @property
    def required_tokens(self) -> int:
        return self.input_tokens + self.reserved_output_tokens + self.safety_tokens

    @property
    def fits(self) -> bool:
        return self.context_window_tokens <= 0 or self.required_tokens <= self.context_window_tokens


def structured_request_budget(
    messages: Sequence[ModelMessage],
    schema: type[BaseModel],
    *,
    context_window_tokens: int,
    reserved_output_tokens: int,
    safety_tokens: int,
    token_counter: TokenCounter = conservative_token_count,
) -> RequestBudget:
    """Count the complete semantic request, including its structured-output contract."""

    message_tokens = sum(token_counter(message.content) + 8 for message in messages)
    schema_text = json.dumps(
        schema.model_json_schema(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    budget = RequestBudget(
        input_tokens=message_tokens + token_counter(schema_text) + 16,
        reserved_output_tokens=reserved_output_tokens,
        safety_tokens=safety_tokens,
        context_window_tokens=context_window_tokens,
    )
    if not budget.fits:
        raise ModelContextLimitError(
            "完整结构化请求超过上下文限制："
            f"input={budget.input_tokens}, output_reserve={reserved_output_tokens}, "
            f"safe_buffer={safety_tokens}, context_window={context_window_tokens}"
        )
    return budget


_TRUNCATED_FINISH_REASONS = frozenset(
    {"length", "max_tokens", "max_token", "max_output_tokens", "token_limit"}
)


def response_finish_reason(response: Any) -> str:
    """Read common finish-reason shapes without retaining provider response bodies."""

    metadata = getattr(response, "response_metadata", None)
    if not isinstance(metadata, dict):
        return ""
    value = metadata.get("finish_reason") or metadata.get("finishReason")
    if isinstance(value, dict):
        value = value.get("type") or value.get("reason")
    return str(value or "").strip().casefold()


def ensure_complete_response(response: Any) -> None:
    """Fail closed even when a truncated response happens to satisfy the local schema."""

    reason = response_finish_reason(response)
    normalized = reason.replace("-", "_").replace(" ", "_")
    if normalized in _TRUNCATED_FINISH_REASONS:
        raise ModelOutputTruncatedError(
            f"模型输出达到 token 限制，响应不完整（finish_reason={reason}）"
        )


def is_output_truncation_error(error: BaseException) -> bool:
    """Recognize SDK exceptions raised before a provider response envelope is returned."""

    name = type(error).__name__.casefold()
    message = str(error).casefold()
    return name in {"lengthfinishreasonerror", "maxtokenserror"} or any(
        marker in message
        for marker in (
            "length limit was reached",
            "finish_reason=max_tokens",
            "finish reason: max_tokens",
            "finishreason.max_tokens",
        )
    )
