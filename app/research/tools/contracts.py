"""Function-calling contracts shared by local and future MCP tool providers.

Vocabulary: a **function** is a research function in the domain sense - what a Skill
authorizes, what a protocol stage orders, and what a plan step names. A **tool** is the
OpenAI tool-call protocol boundary: ``ToolCall``, ``ToolOutput``, ``ToolResult``, the
registry, policy and executor that carry them, and the ``tool_*`` loop state that holds
them. Anything naming a research function uses `function`; anything naming the transport
uses `tool`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.research.schemas.results import DataQualityReport
from app.research.schemas.study import StudyConfig


class ToolArguments(BaseModel):
    """Base arguments visible in a model function schema."""

    model_config = ConfigDict(extra="forbid")


class NoArguments(ToolArguments):
    pass


class DataQualityArguments(NoArguments):
    pass


class PriceExtremeArguments(ToolArguments):
    spike_iqr_multiplier: float = Field(gt=0)


class PriceAutocorrelationArguments(ToolArguments):
    max_lag: int = Field(ge=0)


class VariablesArguments(ToolArguments):
    variables: list[str] = Field(min_length=1)


class ExogenousOutlierArguments(VariablesArguments):
    outlier_iqr_multiplier: float = Field(gt=0)


class RelationshipArguments(VariablesArguments):
    min_observations: int = Field(ge=3)


class RelationshipLagArguments(RelationshipArguments):
    max_lag: int = Field(ge=0)


class SegmentDefinition(BaseModel):
    """One safe, reproducible subset selector; arbitrary expressions are forbidden."""

    model_config = ConfigDict(extra="forbid")

    segment_id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,31}$")
    label: str = Field(min_length=1, max_length=64)
    kind: Literal["hours", "months", "time_range"]
    hours: list[int] | None = None
    months: list[int] | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None

    @model_validator(mode="after")
    def validate_selector(self) -> SegmentDefinition:
        if self.kind == "hours":
            if not self.hours or any(value < 0 or value > 23 for value in self.hours):
                raise ValueError("hours segment requires values from 0 through 23")
            if len(self.hours) != len(set(self.hours)):
                raise ValueError("hours segment must not contain duplicate hours")
            if self.months is not None or self.start_time is not None or self.end_time is not None:
                raise ValueError("hours segment accepts only hours")
        elif self.kind == "months":
            if not self.months or any(value < 1 or value > 12 for value in self.months):
                raise ValueError("months segment requires values from 1 through 12")
            if len(self.months) != len(set(self.months)):
                raise ValueError("months segment must not contain duplicate months")
            if self.hours is not None or self.start_time is not None or self.end_time is not None:
                raise ValueError("months segment accepts only months")
        else:
            if self.start_time is None or self.end_time is None or self.start_time > self.end_time:
                raise ValueError("time_range segment requires start_time not after end_time")
            if self.hours is not None or self.months is not None:
                raise ValueError("time_range segment accepts only start_time and end_time")
        return self


class SegmentComparisonArguments(ToolArguments):
    comparison_id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,31}$")
    segments: list[SegmentDefinition] = Field(min_length=2, max_length=12)
    min_observations: int = Field(default=3, ge=3)

    @model_validator(mode="after")
    def validate_segment_ids(self) -> SegmentComparisonArguments:
        ids = [segment.segment_id for segment in self.segments]
        if len(ids) != len(set(ids)):
            raise ValueError("segment_id values must be unique within one comparison")
        return self


class RelationshipSegmentComparisonArguments(SegmentComparisonArguments):
    variables: list[str] = Field(min_length=1)
    min_observations: int = Field(ge=3)


class ToolCall(BaseModel):
    """One validated function call compiled from an approved research plan."""

    model_config = ConfigDict(extra="forbid")

    call_id: str = Field(min_length=1)
    work_id: str = Field(pattern=r"^[a-f0-9]{24}$")
    step_id: str = Field(pattern=r"^S[1-9][0-9]*$")
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolOutput(BaseModel):
    """Structured output returned by a deterministic tool function."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    result_key: str
    value: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Auditable outcome of one controlled function call."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    call: ToolCall
    status: Literal["completed"] = "completed"
    output: ToolOutput
    started_at: str | None = None
    finished_at: str | None = None
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
    display_name: str = ""
    result_key: str = ""

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
