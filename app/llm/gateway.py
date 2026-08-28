"""Protocol contracts required by the research Agent."""

from __future__ import annotations

import json
from typing import Any, Literal, Protocol, TypeVar

from pydantic import BaseModel, field_validator, model_validator

StructuredResult = TypeVar("StructuredResult", bound=BaseModel)


class ModelMessage(BaseModel):
    """Canonical message owned by the Agent, independent of an API wire format."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_call_id: str | None = None

    @model_validator(mode="after")
    def _validate_tool_message(self) -> ModelMessage:
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("tool 消息必须包含 tool_call_id")
        if self.role != "tool" and self.tool_call_id is not None:
            raise ValueError("只有 tool 消息可以包含 tool_call_id")
        return self


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


class ModelGatewayError(RuntimeError):
    """Base error raised by the shared model boundary."""


class ModelConfigurationError(ModelGatewayError):
    """The configured model endpoint cannot be used."""


class ModelProtocolError(ModelConfigurationError):
    """The endpoint does not implement a core protocol required by the Agent."""


class ModelResponseError(ModelGatewayError):
    """The model returned an invalid or empty response."""


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
    ) -> StructuredResult: ...

    def invoke_text(self, *, messages: list[ModelMessage]) -> str: ...

    def invoke_tool_calls(
        self,
        *,
        messages: list[ModelMessage],
        tools: list[dict[str, Any]],
    ) -> list[ModelToolCall]: ...
