"""Composition helper for the shared model gateway."""

from __future__ import annotations

from app.config import Settings
from app.llm.gateway import ModelGateway
from app.llm.openai_compatible import ResearchModelGateway


def build_model_gateway(settings: Settings | None = None) -> ModelGateway:
    """Build the single model gateway injected into all research Agent roles."""

    return ResearchModelGateway(settings)
