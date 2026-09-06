"""Composition helper for the shared model gateway."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.config import Settings, get_settings
from app.llm.gateway import ModelGateway
from app.llm.gemini import GeminiModelGateway
from app.llm.openai_compatible import ResearchModelGateway


def build_model_gateway(
    settings: Settings | None = None,
    *,
    structured_output_observer: Callable[[dict[str, Any]], None] | None = None,
) -> ModelGateway:
    """Select one provider adapter while preserving the shared Agent boundary."""

    configured = settings or get_settings()
    if configured.llm_provider == "gemini":
        return GeminiModelGateway(
            configured,
            structured_output_observer=structured_output_observer,
        )
    return ResearchModelGateway(
        configured,
        structured_output_observer=structured_output_observer,
    )
