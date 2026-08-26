"""Domain Subagents called by the main research Agent.

Concrete roles are imported from their defining modules so this package does
not introduce import-time dependencies on planning implementations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.research.agent.subagents.eda import EDASubagent, ModelEDAPlanner

__all__ = ["EDASubagent", "ModelEDAPlanner"]


def __getattr__(name: str) -> Any:
    """Resolve compatibility exports only when a caller requests them."""

    if name in __all__:
        from app.research.agent.subagents.eda import EDASubagent, ModelEDAPlanner

        return {"EDASubagent": EDASubagent, "ModelEDAPlanner": ModelEDAPlanner}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
