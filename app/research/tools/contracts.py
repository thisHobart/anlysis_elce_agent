"""Function-calling contracts shared by local and future MCP tool providers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from app.research.schemas.results import DataQualityReport
from app.research.schemas.study import StudyConfig


class ToolArguments(BaseModel):
    """Base arguments visible in a model function schema."""

    model_config = ConfigDict(extra="forbid")


class DataQualityArguments(ToolArguments):
    pass


class PriceProfileArguments(ToolArguments):
    methods: list[str] = Field(min_length=1)
    max_lag: int = Field(ge=0)
    spike_iqr_multiplier: float = Field(gt=0)


class ExogenousProfileArguments(ToolArguments):
    methods: list[str] = Field(min_length=1)
    variables: list[str] = Field(min_length=1)
    outlier_iqr_multiplier: float = Field(gt=0)


class RelationshipArguments(ToolArguments):
    methods: list[str] = Field(min_length=1)
    variables: list[str] = Field(min_length=1)
    max_lag: int = Field(ge=0)
    min_observations: int = Field(ge=3)


class ToolCall(BaseModel):
    """One validated function call compiled from an approved research plan."""

    model_config = ConfigDict(extra="forbid")

    call_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolOutput(BaseModel):
    """Structured output returned by a deterministic tool function."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    result_key: str
    value: Any = None


class ToolResult(BaseModel):
    """Auditable outcome of one controlled function call."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    call: ToolCall
    status: Literal["completed"] = "completed"
    output: ToolOutput
    duration_ms: float = Field(ge=0)
    provider: str = Field(min_length=1)
    tool_version: str = Field(min_length=1)
    data_fingerprint: str = Field(pattern=r"^[a-f0-9]{12}$")
    output_hash: str = Field(pattern=r"^[a-f0-9]{64}$")


@dataclass(frozen=True)
class ToolContext:
    """Trusted runtime data hidden from model-visible function arguments."""

    config: StudyConfig
    frame: pd.DataFrame
    quality: DataQualityReport


ToolHandler = Callable[[ToolContext, ToolArguments], ToolOutput]


@dataclass(frozen=True)
class ToolSpec:
    """One registered local function with a model-readable JSON schema."""

    name: str
    version: str
    description: str
    arguments_model: type[ToolArguments]
    handler: ToolHandler
    provider: str = "local"

    def function_schema(self) -> dict[str, Any]:
        """Return an OpenAI-compatible function definition."""

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.arguments_model.model_json_schema(),
            },
        }
