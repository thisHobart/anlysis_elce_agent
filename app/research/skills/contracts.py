"""Validated runtime representation of an Agent Skill."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.research.tools.catalog import TOOL_CATALOG


class ResearchProtocolStage(BaseModel):
    """One observable stage in a domain research evidence ladder."""

    model_config = ConfigDict(extra="forbid")

    stage_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    title: str = Field(min_length=1, max_length=128)
    objective: str = Field(min_length=1)
    functions: list[str] = Field(default_factory=list)
    function_rules: dict[str, str] = Field(default_factory=dict)
    exit_gate: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_functions(self) -> ResearchProtocolStage:
        if len(self.functions) != len(set(self.functions)):
            raise ValueError(f"研究协议阶段 {self.stage_id} 包含重复函数")
        unknown = sorted(set(self.functions).difference(TOOL_CATALOG))
        if unknown:
            raise ValueError(f"研究协议阶段 {self.stage_id} 包含未知函数：{', '.join(unknown)}")
        unmatched_rules = sorted(set(self.function_rules).difference(self.functions))
        if unmatched_rules:
            raise ValueError(
                f"研究协议阶段 {self.stage_id} 的函数规则没有对应函数：{', '.join(unmatched_rules)}"
            )
        return self


class ResearchProtocol(BaseModel):
    """Versioned local decision protocol used instead of a model reasoning trace."""

    model_config = ConfigDict(extra="forbid")

    protocol_id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    version: str = Field(min_length=1)
    kind: Literal["domain-evidence-ladder"] = "domain-evidence-ladder"
    objective: str = Field(min_length=1)
    invariants: list[str] = Field(min_length=1)
    stop_conditions: list[str] = Field(min_length=1)
    stages: list[ResearchProtocolStage] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_stages(self) -> ResearchProtocol:
        stage_ids = [stage.stage_id for stage in self.stages]
        if len(stage_ids) != len(set(stage_ids)):
            raise ValueError("研究协议阶段 ID 必须唯一")
        functions = [name for stage in self.stages for name in stage.functions]
        if len(functions) != len(set(functions)):
            raise ValueError("一个研究函数只能属于一个领域协议阶段")
        if "data_quality" not in functions:
            raise ValueError("领域研究协议必须包含 data_quality 门禁")
        return self

    @property
    def function_order(self) -> tuple[str, ...]:
        """Return the stable domain order used by the local plan compiler."""

        return tuple(name for stage in self.stages for name in stage.functions)

    def compact_context(self) -> dict[str, Any]:
        """Return the complete, bounded protocol supplied to the function selector."""

        return self.model_dump(mode="json")


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
    research_protocol: ResearchProtocol | None = None
    root: Path
    source: Literal["builtin", "external"]

    @model_validator(mode="after")
    def validate_protocol_permissions(self) -> SkillDefinition:
        if self.research_protocol is None:
            return self
        protocol_functions = set(self.research_protocol.function_order)
        allowed = set(self.allowed_tools)
        if protocol_functions != allowed:
            missing = sorted(allowed.difference(protocol_functions))
            extra = sorted(protocol_functions.difference(allowed))
            details = []
            if missing:
                details.append(f"协议未覆盖授权函数：{', '.join(missing)}")
            if extra:
                details.append(f"协议包含未授权函数：{', '.join(extra)}")
            raise ValueError("；".join(details))
        return self

    def order_function_names(self, names: list[str]) -> list[str]:
        """Order proposed functions by the local domain protocol when present."""

        if self.research_protocol is None:
            return list(names)
        rank = {name: index for index, name in enumerate(self.research_protocol.function_order)}
        return sorted(names, key=rank.__getitem__)

    def prompt_context(self) -> dict[str, Any]:
        """Return the bounded context supplied to a model after activation."""

        context = {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "domain": self.domain,
            "allowed_tools": self.allowed_tools,
            "instructions": self.instructions,
            "source": self.source,
        }
        if self.research_protocol is not None:
            context["research_protocol"] = self.research_protocol.compact_context()
        return context
