"""Validated runtime representation of an Agent Skill."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class SkillDefinition(BaseModel):
    """Instructions and tool permissions loaded from one SKILL.md bundle."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    name: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    description: str = Field(min_length=1, max_length=1024)
    version: str = Field(default="unversioned", min_length=1)
    domain: str = Field(default="general", min_length=1)
    allowed_tools: list[str] = Field(default_factory=list)
    instructions: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    root: Path
    source: Literal["builtin", "external"]

    def prompt_context(self) -> dict[str, Any]:
        """Return the bounded context supplied to a model after activation."""

        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "domain": self.domain,
            "allowed_tools": self.allowed_tools,
            "instructions": self.instructions,
            "source": self.source,
        }
