"""Runtime permission policy independent from model and Skill instructions."""

from __future__ import annotations

from dataclasses import dataclass

from app.research.tools.contracts import ToolCall


class ToolPermissionError(PermissionError):
    """A Skill or approved plan is not permitted to call a tool."""


@dataclass(frozen=True)
class ToolPolicy:
    allowed_functions: frozenset[str]

    def authorize(self, call: ToolCall) -> None:
        if call.name not in self.allowed_functions:
            raise ToolPermissionError(f"当前 Skill 未授权工具：{call.name}")
