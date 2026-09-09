"""Shared model infrastructure used by every research Agent."""

from app.llm.factory import build_model_gateway
from app.llm.gateway import (
    ModelConfigurationError,
    ModelContextLimitError,
    ModelGateway,
    ModelGatewayError,
    ModelMessage,
    ModelOutputTruncatedError,
    ModelProtocolError,
    ModelResponseError,
)
from app.llm.gemini import GeminiModelGateway
from app.llm.openai_compatible import ResearchModelGateway

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
    "ResearchModelGateway",
    "build_model_gateway",
]
