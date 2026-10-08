"""Shared model infrastructure used by every research Agent.

Exports are lazy so low-level configuration modules can load model-profile data
without importing provider adapters back through :mod:`app.config`.
"""

from importlib import import_module
from typing import Any

__all__ = [
    "GeminiModelGateway",
    "ModelConfigurationError",
    "ModelContextLimitError",
    "ModelGateway",
    "ModelGatewayError",
    "ModelMessage",
    "ModelOutputTruncatedError",
    "ModelProtocolError",
    "ModelResponseError",
    "ModelToolCall",
    "ModelToolTurn",
    "ModelTransientError",
    "ResearchModelGateway",
    "build_model_gateway",
]


def __getattr__(name: str) -> Any:
    if name == "build_model_gateway":
        return getattr(import_module("app.llm.factory"), name)
    if name == "GeminiModelGateway":
        return getattr(import_module("app.llm.gemini"), name)
    if name == "ResearchModelGateway":
        return getattr(import_module("app.llm.openai_compatible"), name)
    if name in {
        "ModelConfigurationError",
        "ModelContextLimitError",
        "ModelGateway",
        "ModelGatewayError",
        "ModelMessage",
        "ModelOutputTruncatedError",
        "ModelProtocolError",
        "ModelResponseError",
        "ModelToolCall",
        "ModelToolTurn",
        "ModelTransientError",
    }:
        return getattr(import_module("app.llm.gateway"), name)
    raise AttributeError(name)
