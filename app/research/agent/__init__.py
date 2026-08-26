"""Main research Agent and domain Subagents.

Keep this package initializer free of eager imports.  Planning contracts import
``app.research.agent.errors`` and ``app.research.agent.schemas``; importing the
concrete Agent roles here would load the EDA planner in the opposite direction
and make otherwise valid submodule imports depend on import order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.research.agent.orchestrator import MainResearchAgent
    from app.research.agent.subagents.eda import EDASubagent

__all__ = ["EDASubagent", "MainResearchAgent"]


def __getattr__(name: str) -> Any:
    """Preserve package-level imports without loading concrete roles eagerly."""

    if name == "MainResearchAgent":
        from app.research.agent.orchestrator import MainResearchAgent

        return MainResearchAgent
    if name == "EDASubagent":
        from app.research.agent.subagents.eda import EDASubagent

        return EDASubagent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
