"""Protocol contracts required by the research Agent."""

from __future__ import annotations

import inspect
import json
from typing import Any, Literal, Protocol, TypeVar

from pydantic import BaseModel, Field, field_validator, model_validator

StructuredResult = TypeVar("StructuredResult", bound=BaseModel)


class ModelToolCall(BaseModel):
    """Provider-neutral proposed function call; execution remains application-controlled."""

    name: str
    arguments: dict[str, Any]
    call_id: str | None = None

    @field_validator("arguments", mode="before")
    @classmethod
    def _decode_arguments(cls, value: Any) -> Any:
        """Accept the JSON-encoded argument string some endpoints return verbatim."""

        if isinstance(value, str):
            try:
                return json.loads(value or "{}")
            except ValueError:
                return value
        return value


class ModelMessage(BaseModel):
    """Canonical message owned by the Agent, independent of an API wire format."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_call_id: str | None = None
    tool_calls: list[ModelToolCall] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_tool_message(self) -> ModelMessage:
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("tool 消息必须包含 tool_call_id")
        if self.role != "tool" and self.tool_call_id is not None:
            raise ValueError("只有 tool 消息可以包含 tool_call_id")
        if self.tool_calls and self.role != "assistant":
            raise ValueError("只有 assistant 消息可以包含 tool_calls")
        if self.role == "assistant" and self.tool_calls and any(not call.call_id for call in self.tool_calls):
            raise ValueError("assistant 工具调用必须保留 provider call_id")
        if self.role != "assistant" and not self.content:
            raise ValueError(f"{self.role} 消息内容不能为空")
        if self.role == "assistant" and not self.content and not self.tool_calls:
            raise ValueError("assistant 消息必须包含正文或工具调用")
        return self


class ModelToolTurn(BaseModel):
    """One assistant turn that requests tools or completes with visible text."""

    content: str = ""
    tool_calls: list[ModelToolCall] = Field(default_factory=list)
    finish_reason: str | None = None

    @model_validator(mode="after")
    def _validate_turn(self) -> ModelToolTurn:
        if not self.content.strip() and not self.tool_calls:
            raise ValueError("模型工具回合必须包含正文或至少一个工具调用")
        if self.tool_calls and any(not call.call_id for call in self.tool_calls):
            raise ValueError("模型工具回合中的调用必须包含 provider call_id")
        return self


class ModelGatewayError(RuntimeError):
    """Base error raised by the shared model boundary."""


class ModelConfigurationError(ModelGatewayError):
    """The configured model endpoint cannot be used."""


class ModelProtocolError(ModelConfigurationError):
    """The endpoint does not implement a core protocol required by the Agent."""


class ModelResponseError(ModelGatewayError):
    """The model returned an invalid or empty response."""


class ModelTransientError(ModelGatewayError):
    """A retryable transport or provider failure affected one invocation."""


class ModelContextLimitError(ModelResponseError):
    """A complete request cannot fit in the configured model context window."""


class ModelOutputTruncatedError(ModelResponseError):
    """The provider reports that generation stopped at its output limit."""


class ModelThinkingError(ModelResponseError):
    """The endpoint kept its reasoning trace while the strict policy is active.

    No other request shape fixes this, so callers stop instead of degrading.
    """


class ModelGateway(Protocol):
    """One injectable model boundary shared by the main Agent and Subagents."""

    @property
    def enabled(self) -> bool: ...

    @property
    def model_name(self) -> str: ...

    def invoke_structured(
        self,
        *,
        messages: list[ModelMessage],
        schema: type[StructuredResult],
        purpose: Any = "generic",
    ) -> StructuredResult: ...

    def invoke_text(self, *, messages: list[ModelMessage], purpose: Any = "generic") -> str: ...

    def invoke_tool_calls(
        self,
        *,
        messages: list[ModelMessage],
        tools: list[dict[str, Any]],
        purpose: Any = "generic",
    ) -> list[ModelToolCall]: ...

    def invoke_tool_turn(
        self,
        *,
        messages: list[ModelMessage],
        tools: list[dict[str, Any]],
        purpose: Any = "generic",
    ) -> ModelToolTurn: ...


def invoke_structured_for_purpose(
    gateway: ModelGateway,
    *,
    messages: list[ModelMessage],
    schema: type[StructuredResult],
    purpose: Any,
) -> StructuredResult:
    """Pass purpose to the new boundary while preserving injected legacy doubles."""

    invoke = gateway.invoke_structured
    parameters = inspect.signature(invoke).parameters.values()
    supports_purpose = any(item.name == "purpose" or item.kind == inspect.Parameter.VAR_KEYWORD for item in parameters)
    kwargs: dict[str, Any] = {"messages": messages, "schema": schema}
    if supports_purpose:
        kwargs["purpose"] = purpose
    return invoke(**kwargs)
