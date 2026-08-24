"""Stable structured feedback exchanged across the research loop."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

FeedbackSource = Literal[
    "user",
    "skill_validator",
    "plan_validator",
    "tool_executor",
    "tool_result_validator",
    "evaluator",
    "budget_guard",
]
FeedbackSeverity = Literal["info", "warning", "error", "fatal"]


class FeedbackPacket(BaseModel):
    """Machine-readable correction or stop signal; never contains hidden reasoning."""

    model_config = ConfigDict(extra="forbid")

    feedback_id: str = Field(default_factory=lambda: uuid4().hex)
    source: FeedbackSource
    code: str = Field(min_length=1)
    severity: FeedbackSeverity
    message: str = Field(min_length=1)
    step_id: str | None = None
    observed: Any = None
    expected: Any = None
    recommendation: str = ""
    retryable: bool = False
    requires_user: bool = False
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    evidence_fingerprint: str | None = None
