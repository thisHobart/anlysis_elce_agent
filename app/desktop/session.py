"""Persistent conversation state for the desktop research workspace."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field
from PySide6.QtCore import QStandardPaths

SessionStatus = Literal[
    "idle",
    "understanding",
    "inspecting_data",
    "awaiting_plan_approval",
    "awaiting_user",
    "running",
    "evaluating",
    "completed",
    "failed",
    "stopped",
]
InputRole = Literal["config", "target", "actuals", "forecasts"]
InputStatus = Literal["empty", "selected", "loading", "ready", "warning", "failed", "changed"]
MessageKind = Literal["text", "notice", "thinking", "tool", "plan", "result", "error"]
TraceCategory = Literal["session", "user", "agent", "input", "plan", "tool", "evaluation", "artifact", "error"]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class VariableEvidence(BaseModel):
    """Variable-level readiness evidence displayed below one input file."""

    model_config = ConfigDict(extra="forbid")

    name: str
    coverage_rate: float | None = None
    status: Literal["ready", "warning", "failed"] = "ready"
    detail: str = ""


class SessionInputFile(BaseModel):
    """One user-managed file slot with a fixed analytical role."""

    model_config = ConfigDict(extra="forbid")

    role: InputRole
    label: str
    expected_name: str = ""
    path: str = ""
    status: InputStatus = "empty"
    detail: str = "尚未选择"
    variables: list[VariableEvidence] = Field(default_factory=list)


def default_input_files() -> dict[InputRole, SessionInputFile]:
    return {
        "config": SessionInputFile(
            role="config",
            label="研究配置",
        ),
        "target": SessionInputFile(
            role="target",
            label="目标电价",
        ),
        "actuals": SessionInputFile(
            role="actuals",
            label="实际外生变量",
        ),
        "forecasts": SessionInputFile(
            role="forecasts",
            label="预测外生变量",
        ),
    }


class SessionMessage(BaseModel):
    """One typed item in the continuous conversation timeline."""

    model_config = ConfigDict(extra="forbid")

    message_id: str = Field(default_factory=lambda: uuid4().hex)
    role: Literal["user", "assistant", "system"]
    kind: MessageKind = "text"
    content: str = ""
    created_at: str = Field(default_factory=utc_now)
    payload: dict[str, Any] = Field(default_factory=dict)


class TraceEvent(BaseModel):
    """Observable Agent action; never stores hidden model reasoning."""

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(default_factory=lambda: uuid4().hex)
    created_at: str = Field(default_factory=utc_now)
    category: TraceCategory
    name: str
    status: Literal["info", "running", "completed", "warning", "failed", "stopped"] = "info"
    duration_ms: float | None = None
    summary: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class SessionRunRecord(BaseModel):
    """Compact lineage for one completed plan execution in a conversation."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    plan_id: str
    parent_run_id: str | None = None
    question: str
    artifact_directory: str
    report_path: str
    created_at: str = Field(default_factory=utc_now)
    evaluation: dict[str, Any] = Field(default_factory=dict)


class ResearchSession(BaseModel):
    """Single source of truth shared by history, conversation and context panes."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 7
    session_id: str = Field(default_factory=lambda: uuid4().hex)
    title: str = "新会话"
    status: SessionStatus = "idle"
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    inputs: dict[InputRole, SessionInputFile] = Field(default_factory=default_input_files)
    messages: list[SessionMessage] = Field(default_factory=list)
    trace: list[TraceEvent] = Field(default_factory=list)
    current_plan: dict[str, Any] | None = None
    plan_stale: bool = False
    plan_feedback_deadline: str | None = None
    plan_feedback_remaining_seconds: int | None = None
    data_profile: dict[str, Any] | None = None
    quality_report: dict[str, Any] | None = None
    latest_eda_summary: dict[str, Any] | None = None
    latest_evaluation: dict[str, Any] | None = None
    runs: list[SessionRunRecord] = Field(default_factory=list)
    run_id: str | None = None
    artifact_directory: str | None = None
    report_path: str | None = None
    graph_event_count: int = 0
    graph_event_sequence: int = 0

    @property
    def can_analyze(self) -> bool:
        return bool(self.inputs["target"].path or self.inputs["config"].path)

    def touch(self) -> None:
        self.updated_at = utc_now()

    def derive_title(self, question: str) -> None:
        if self.title == "新会话":
            compact = " ".join(question.split())
            self.title = compact[:28] + ("…" if len(compact) > 28 else "")
        self.touch()


class SessionStore:
    """Atomic JSON persistence under the platform application-data directory."""

    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            root = Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation))
            path = root / "research_sessions.json"
        self.path = Path(path)

    def load(self) -> list[ResearchSession]:
        if not self.path.is_file():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                return []
            sessions = [ResearchSession.model_validate(item) for item in payload]
            for session in sessions:
                session.schema_version = 7
                if session.graph_event_sequence == 0 and session.graph_event_count:
                    session.graph_event_sequence = session.graph_event_count
                if session.current_plan and not self._plan_has_required_versions(session.current_plan):
                    session.current_plan = None
                    session.plan_stale = True
                    session.status = "idle"
                for message in session.messages:
                    stored_plan = message.payload.get("plan") if message.kind == "plan" else None
                    if isinstance(stored_plan, dict) and not self._plan_has_required_versions(stored_plan):
                        message.kind = "notice"
                        message.content = "旧版本研究方案缺少 Skill 或工具版本，已失效，请重新向大模型提出问题。"
                        message.payload = {}
            return sessions
        except (OSError, ValueError, TypeError):
            return []

    @staticmethod
    def _plan_has_required_versions(plan: dict[str, Any]) -> bool:
        if plan.get("planner") != "llm" or not plan.get("skill_name") or not plan.get("skill_version"):
            return False
        steps = plan.get("steps")
        return isinstance(steps, list) and bool(steps) and all(
            isinstance(step, dict) and bool(step.get("tool_version")) for step in steps
        )

    def save(self, sessions: list[ResearchSession]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary.write_text(
            json.dumps([session.model_dump(mode="json") for session in sessions], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, self.path)
