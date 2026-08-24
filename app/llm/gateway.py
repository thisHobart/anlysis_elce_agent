"""Provider-neutral model contracts."""

from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import BaseModel

StructuredResult = TypeVar("StructuredResult", bound=BaseModel)
ModelMessage = tuple[str, str]


class ModelGatewayError(RuntimeError):
    """Base error raised by the shared model boundary."""


class ModelConfigurationError(ModelGatewayError):
    """The configured model endpoint cannot be used."""


class ModelResponseError(ModelGatewayError):
    """The model returned an invalid or empty response."""


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
