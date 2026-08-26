"""Public contracts for the persistent LangGraph research loop."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
InterruptKind = Literal[
    "plan_approval",
    "plan_error",
    "result_limitations",
    "result_rejected",
    "result",
    "response_error",
    "finalization_error",
]
ResumeAction = Literal[
    "approve",
    "timeout_accept",
    "modify",
    "reject",
    "followup",
    "next_round",
    "accept_limitations",
    "retry",
    "clarify",
    "stop",
]


class EpisodeBudget(BaseModel):
    """Safety limits scoped to one research episode, never the whole conversation."""

    model_config = ConfigDict(extra="forbid")

    max_plan_attempts_per_iteration: int = 3
    max_function_attempts_per_call: int = 2
    max_evaluated_iterations: int = 4
    plan_attempts_in_iteration: int = 0
    evaluated_iterations: int = 0

    @model_validator(mode="before")
    @classmethod
    def migrate_thread_scoped_budget(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        migrated = dict(value)
        aliases = {
            "max_research_iterations": "max_evaluated_iterations",
            "research_iterations_used": "evaluated_iterations",
        }
        for old, new in aliases.items():
            if old in migrated and new not in migrated:
                migrated[new] = migrated[old]
            migrated.pop(old, None)
        if "max_plan_repairs" in migrated and "max_plan_attempts_per_iteration" not in migrated:
            migrated["max_plan_attempts_per_iteration"] = int(migrated["max_plan_repairs"]) + 1
        if "max_tool_retries" in migrated and "max_function_attempts_per_call" not in migrated:
            migrated["max_function_attempts_per_call"] = int(migrated["max_tool_retries"]) + 1
        if "plan_repairs_used" in migrated and "plan_attempts_in_iteration" not in migrated:
            migrated["plan_attempts_in_iteration"] = int(migrated["plan_repairs_used"])
        migrated.pop("max_plan_repairs", None)
        migrated.pop("max_tool_retries", None)
        migrated.pop("plan_repairs_used", None)
        for obsolete in (
            "max_tool_calls",
            "tool_calls_used",
            "max_function_calls",
            "function_calls_used",
            "max_active_seconds",
            "active_seconds_used",
            "tool_retry_counts",
            "function_attempts",
        ):
            migrated.pop(obsolete, None)
        return migrated

    def reset_for_episode(self) -> EpisodeBudget:
        return self.model_copy(
            update={
                "plan_attempts_in_iteration": 0,
                "evaluated_iterations": 0,
            }
        )

    def reset_for_iteration(self) -> EpisodeBudget:
        return self.model_copy(update={"plan_attempts_in_iteration": 0})


# Compatibility import name; serialized state uses the episode-scoped field names above.
LoopBudget = EpisodeBudget


EpisodeStatus = Literal[
    "idle",
    "planning",
    "awaiting_approval",
    "executing",
    "evaluating",
    "accepted",
    "limited",
    "plan_error",
    "rejected",
    "stopped",
    "failed",
]


class LoopCursor(BaseModel):
    """Durable location within session → episode → iteration → stage-attempt."""

    model_config = ConfigDict(extra="forbid")

    episode_id: str | None = None
    episode_number: int = 0
    episode_goal: str = ""
    episode_status: EpisodeStatus = "idle"
    iteration_id: str | None = None
    iteration_number: int = 0
    plan_attempt_number: int = 0
    call_attempt_number: int = 0
    current_call_id: str | None = None
    stage: str = "idle"
    terminal_reason: str | None = None

    @model_validator(mode="before")
    @classmethod
    def migrate_ambiguous_attempt(cls, value: Any) -> Any:
        if not isinstance(value, dict) or "attempt_number" not in value:
            return value
        migrated = dict(value)
        attempt = int(migrated.pop("attempt_number", 0))
        if str(migrated.get("stage", "")).startswith("function:"):
            migrated.setdefault("call_attempt_number", attempt)
        else:
            migrated.setdefault("plan_attempt_number", attempt)
        return migrated

    @classmethod
    def start_episode(cls, *, number: int, goal: str) -> LoopCursor:
        return cls(
            episode_id=uuid4().hex[:12],
            episode_number=number,
            episode_goal=goal,
            episode_status="planning",
            iteration_id=uuid4().hex[:12],
            iteration_number=1,
            plan_attempt_number=0,
            call_attempt_number=0,
            current_call_id=None,
            stage="planning",
        )

    def next_iteration(self) -> LoopCursor:
        return self.model_copy(
            update={
                "episode_status": "planning",
                "iteration_id": uuid4().hex[:12],
                "iteration_number": self.iteration_number + 1,
                "plan_attempt_number": 0,
                "call_attempt_number": 0,
                "current_call_id": None,
                "stage": "planning",
                "terminal_reason": None,
            }
        )


class ApprovalState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["none", "waiting", "expired", "approved", "rejected"] = "none"
    plan_id: str | None = None
    plan_fingerprint: str | None = None
    approved_at: str | None = None
    deadline: str | None = None
    remaining_seconds: int | None = None
    decision: ResumeAction | None = None
    origin: Literal["initial", "user_revision"] = "initial"


class AuthorizationEnvelope(BaseModel):
    """Maximum scope approved by the user for automatic evaluation revisions."""

    model_config = ConfigDict(extra="forbid")

    approved_plan_id: str
    approved_plan_fingerprint: str
    approved_at: str
    question_hash: str
    max_steps: int = Field(ge=1)
    skill_name: str
    skill_version: str
    data_fingerprint: str = Field(pattern=r"^[a-f0-9]{12}$")
    functions: list[str]
    variables: list[str]
    max_lag_by_function: dict[str, int]
    segment_parameters_by_function: dict[str, dict[str, Any]]
    approved_parameters_by_function: dict[str, dict[str, Any]]


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
    automatic_timeout: bool = False
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class ResearchLoopSnapshot(BaseModel):
    """Desktop-facing serializable view of one persistent graph thread."""

    model_config = ConfigDict(extra="forbid")

    thread_id: str
    phase: LoopPhase
    values: dict[str, Any]
    interrupt: InterruptPayload | None = None
    events: list[dict[str, Any]] = Field(default_factory=list)
