"""Public contracts for the persistent LangGraph research loop."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.research.schemas.feedback import FeedbackPacket

LoopPhase = Literal[
    "idle",
    "understanding",
    "planning",
    "validating_plan",
    "awaiting_approval",
    "executing_tools",
    "validating_result",
    "evaluating",
    "awaiting_user",
    "completed",
    "stopped",
    "failed",
]
InterruptKind = Literal["plan_approval", "need_user", "result"]
ResumeAction = Literal[
    "approve",
    "timeout_accept",
    "modify",
    "reject",
    "followup",
    "next_round",
    "accept_limitations",
    "stop",
]


class LoopBudget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_plan_repairs: int = 2
    max_tool_retries: int = 1
    max_research_iterations: int = 2
    max_tool_calls: int = 8
    max_active_seconds: float = 600.0
    plan_repairs_used: int = 0
    tool_calls_used: int = 0
    research_iterations_used: int = 0
    active_seconds_used: float = 0.0
    tool_retry_counts: dict[str, int] = Field(default_factory=dict)


class ApprovalState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["none", "waiting", "paused", "expired", "approved", "rejected"] = "none"
    plan_id: str | None = None
    deadline: str | None = None
    remaining_seconds: int | None = None
    decision: ResumeAction | None = None
    origin: Literal["initial", "user_revision", "evaluation_auto"] = "initial"


class AuthorizationEnvelope(BaseModel):
    """Maximum scope approved by the user for automatic evaluation revisions."""

    model_config = ConfigDict(extra="forbid")

    skill_name: str
    skill_version: str
    data_fingerprint: str
    tools: list[str]
    variables: list[str]
    methods: dict[str, list[str]]
    max_lag_by_tool: dict[str, int]


class ToolCallRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    call: dict[str, Any]
    status: Literal["pending", "running", "completed", "failed", "reused"] = "pending"
    attempts: int = 0
    started_at: str | None = None
    finished_at: str | None = None
    result: dict[str, Any] | None = None
    error: FeedbackPacket | None = None


class InterruptPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: InterruptKind
    phase: LoopPhase
    message: str
    choices: list[ResumeAction]
    plan: dict[str, Any] | None = None
    evaluation: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    deadline: str | None = None
    remaining_seconds: int | None = None


class ResumePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: ResumeAction
    message: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class ResearchLoopSnapshot(BaseModel):
    """Desktop-facing serializable view of one persistent graph thread."""

    model_config = ConfigDict(extra="forbid")

    thread_id: str
    phase: LoopPhase
    values: dict[str, Any]
    interrupt: InterruptPayload | None = None
    events: list[dict[str, Any]] = Field(default_factory=list)
