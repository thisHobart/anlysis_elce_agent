"""Persistent conversation state for the desktop research workspace."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.research.agent.schemas import rename_legacy_step_keys
from app.research.data.sources.summary import DataSummary
from app.research.tools.catalog import FUNCTION_CATALOG
from app.runtime_paths import application_data_directory

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
InputRole = Literal["target", "actuals", "forecasts"]
InputStatus = Literal["empty", "selected", "loading", "ready", "warning", "failed", "changed"]
DataPanelState = Literal["empty", "exploring", "ready", "unavailable"]
DataSourceKind = Literal["database", "file"]
MessageKind = Literal["text", "notice", "thinking", "tool", "plan", "data_plan", "result", "error"]
TraceCategory = Literal["session", "user", "agent", "input", "plan", "tool", "evaluation", "artifact", "error"]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class VariableEvidence(BaseModel):
    """Variable-level readiness evidence displayed below one input file."""

    model_config = ConfigDict(extra="ignore")

    name: str
    coverage_rate: float | None = None
    status: Literal["ready", "warning", "failed"] = "ready"
    detail: str = ""


class SessionInputFile(BaseModel):
    """One user-managed file slot with a fixed analytical role."""

    model_config = ConfigDict(extra="ignore")

    role: InputRole
    label: str
    expected_name: str = ""
    path: str = ""
    status: InputStatus = "empty"
    detail: str = "尚未选择"
    variables: list[VariableEvidence] = Field(default_factory=list)


def default_input_files() -> dict[InputRole, SessionInputFile]:
    return {
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

    model_config = ConfigDict(extra="ignore")

    message_id: str = Field(default_factory=lambda: uuid4().hex)
    turn_id: str | None = None
    episode_id: str | None = None
    role: Literal["user", "assistant", "system"]
    kind: MessageKind = "text"
    content: str = ""
    created_at: str = Field(default_factory=utc_now)
    payload: dict[str, Any] = Field(default_factory=dict)


class TraceEvent(BaseModel):
    """Observable Agent action; never stores hidden model reasoning."""

    model_config = ConfigDict(extra="ignore")

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

    model_config = ConfigDict(extra="ignore")

    run_id: str
    episode_id: str | None = None
    plan_id: str
    parent_run_id: str | None = None
    question: str
    artifact_directory: str
    report_path: str
    created_at: str = Field(default_factory=utc_now)
    evaluation: dict[str, Any] = Field(default_factory=dict)
    data_fingerprint: str | None = Field(default=None, pattern=r"^[a-f0-9]{12}$")
    study_name: str | None = None
    target_name: str | None = None
    study_start_time: str | None = None
    study_end_time: str | None = None
    skill_name: str | None = None
    skill_version: str | None = None
    memory_status: Literal["active", "stale"] = "active"


SESSION_SCHEMA_VERSION = 12
"""Projection schema written by this build; bump it whenever stored sessions change shape."""


class SessionMigrationError(ValueError):
    """A stored session cannot be migrated onto the current projection schema."""


class ResearchSession(BaseModel):
    """Single source of truth shared by history, conversation and context panes."""

    model_config = ConfigDict(extra="ignore")

    schema_version: int = SESSION_SCHEMA_VERSION
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
    source_kind: DataSourceKind = "database"
    data_state: DataPanelState = "empty"
    data_summary: DataSummary | None = None
    dataset_fingerprint: str | None = None

    @field_validator("inputs", mode="before")
    @classmethod
    def discard_removed_input_roles(cls, value: Any) -> Any:
        """Open older sessions after the retired YAML configuration slot is removed."""

        if not isinstance(value, dict):
            return value
        current = default_input_files()
        for role in current:
            if role in value:
                current[role] = value[role]
        return current

    @property
    def can_analyze(self) -> bool:
        return bool(self.inputs["target"].path)

    def touch(self) -> None:
        self.updated_at = utc_now()

    def derive_title(self, question: str) -> None:
        if self.title == "新会话":
            compact = " ".join(question.split())
            self.title = compact[:28] + ("…" if len(compact) > 28 else "")
        self.touch()


def _carry_forward(session: ResearchSession) -> None:
    """Migration step for revisions that only added optional fields with defaults."""


def _stored_plans(session: ResearchSession) -> list[dict[str, Any]]:
    """Return every plan payload a session keeps, in the conversation and in its state."""

    plans = [session.current_plan] if isinstance(session.current_plan, dict) else []
    plans.extend(
        message.payload["plan"]
        for message in session.messages
        if message.kind in {"plan", "data_plan"} and isinstance(message.payload.get("plan"), dict)
    )
    return plans


def _migrate_7_to_8(session: ResearchSession) -> None:
    """A plan step's `tool` became `function`; rewrite the payloads written before that."""

    for plan in _stored_plans(session):
        steps = plan.get("steps")
        if not isinstance(steps, list):
            continue
        plan["steps"] = [rename_legacy_step_keys(step) for step in steps]


def _migrate_8_to_9(session: ResearchSession) -> None:
    """Version the removal of the user-facing research-configuration file slot."""


def _migrate_9_to_10(session: ResearchSession) -> None:
    """Version durable turn IDs and Episode linkage added to the UI projection."""


def _migrate_10_to_11(session: ResearchSession) -> None:
    """Version data-scoped run memory and explicit stale-state tracking."""


def _migrate_11_to_12(session: ResearchSession) -> None:
    """Version the data panel replacing the three user-managed file slots.

    A session saved before the panel existed carries no dataset description, so it
    opens on the empty panel and needs one new question before it can continue.
    The retired ``inputs`` payload stays on disk and is simply ignored.
    """


SESSION_MIGRATIONS: dict[int, Callable[[ResearchSession], None]] = {
    **{version: _carry_forward for version in range(1, 7)},
    7: _migrate_7_to_8,
    8: _migrate_8_to_9,
    9: _migrate_9_to_10,
    10: _migrate_10_to_11,
    11: _migrate_11_to_12,
}


STALE_PLAN_NOTICE = "这份分析方案是用旧版本的研究方法生成的，已经作废。重新提问一次，我会按当前版本给方案。"


def _apply_session_invariants(
    session: ResearchSession,
    skill_versions: dict[str, str] | None = None,
) -> None:
    """Repair state that depends on the current catalog rather than on the schema version."""

    if session.graph_event_sequence == 0 and session.graph_event_count:
        session.graph_event_sequence = session.graph_event_count
    if session.current_plan and not plan_is_executable(session.current_plan, skill_versions):
        session.current_plan = None
        session.plan_stale = True
        session.status = "idle"
    for message in session.messages:
        stored_plan = message.payload.get("plan") if message.kind in {"plan", "data_plan"} else None
        if isinstance(stored_plan, dict) and not plan_is_executable(stored_plan, skill_versions):
            message.kind = "notice"
            message.content = STALE_PLAN_NOTICE
            message.payload = {}


def plan_is_executable(plan: dict[str, Any], skill_versions: dict[str, str] | None = None) -> bool:
    """Check a stored plan against what is installed now, so it fails on open rather than on run.

    The execution gate rejects the same plans, but only once the user has already started a run;
    ``skill_versions`` is injected because the Skill registry lives outside this projection.
    """

    if plan.get("planner") != "llm" or not plan.get("skill_name") or not plan.get("skill_version"):
        return False
    if skill_versions is not None:
        installed = skill_versions.get(str(plan["skill_name"]))
        if installed is None or installed != plan["skill_version"]:
            return False
    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps:
        return False
    for step in steps:
        if not isinstance(step, dict) or not step.get("function_version"):
            return False
        spec = FUNCTION_CATALOG.get(str(step.get("function", "")))
        if spec is None or spec.version != step["function_version"]:
            return False
    return True


def migrate_session(
    session: ResearchSession,
    skill_versions: dict[str, str] | None = None,
) -> ResearchSession:
    """Walk one stored session up to the current schema, refusing unknown jumps."""

    version = session.schema_version
    if version > SESSION_SCHEMA_VERSION:
        raise SessionMigrationError(
            f"会话由更新版本的程序写入（schema {version} > {SESSION_SCHEMA_VERSION}）"
        )
    while version < SESSION_SCHEMA_VERSION:
        migration = SESSION_MIGRATIONS.get(version)
        if migration is None:
            raise SessionMigrationError(f"缺少 schema {version} 到 {version + 1} 的迁移步骤")
        migration(session)
        version += 1
    session.schema_version = SESSION_SCHEMA_VERSION
    _apply_session_invariants(session, skill_versions)
    return session


class SessionStore:
    """Atomic JSON persistence that never trades existing history for a failed read."""

    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            path = application_data_directory() / "research_sessions.json"
        self.path = Path(path)
        self.recovery_notices: list[str] = []
        self._write_path: Path | None = None

    @property
    def write_path(self) -> Path:
        """Return where save() writes; redirected when the main file must stay untouched."""

        return self._write_path or self.path

    def load(self, skill_versions: dict[str, str] | None = None) -> list[ResearchSession]:
        """Load every readable session; pass installed Skill versions to retire stale plans."""

        self.recovery_notices = []
        self._write_path = None
        payload = self._read_payload()
        if payload is None:
            return []
        sessions: list[ResearchSession] = []
        damaged: list[dict[str, Any]] = []
        for index, item in enumerate(payload):
            try:
                sessions.append(migrate_session(ResearchSession.model_validate(item), skill_versions))
            except (ValueError, TypeError) as exc:
                damaged.append({"index": index, "error": f"{type(exc).__name__}: {exc}", "record": item})
        if damaged:
            self._quarantine(damaged)
        return sessions

    def _read_payload(self) -> list[Any] | None:
        source = self.path
        if not source.is_file():
            backup = self._backup_path(self.path)
            if not backup.is_file():
                return None
            # A crash between the two atomic renames in save() leaves only the backup.
            source = backup
            self.recovery_notices.append(f"主会话文件缺失，已从备份 {backup.name} 恢复。")
        try:
            raw = source.read_text(encoding="utf-8")
        except OSError as exc:
            self._redirect_writes(f"无法读取历史会话文件（{type(exc).__name__}: {exc}）")
            return None
        try:
            payload = json.loads(raw)
        except ValueError as exc:
            self._set_aside(source, reason=f"{type(exc).__name__}: {exc}")
            return None
        if not isinstance(payload, list):
            self._set_aside(source, reason="顶层不是会话数组")
            return None
        return payload

    def _set_aside(self, source: Path, *, reason: str) -> None:
        """Keep an unparseable file on disk under a new name instead of overwriting it."""

        target = self._sidecar("corrupt")
        try:
            os.replace(source, target)
        except OSError as exc:
            self._redirect_writes(f"历史会话文件无法解析（{reason}），且无法移开（{type(exc).__name__}）")
            return
        self.recovery_notices.append(
            f"历史会话文件无法解析（{reason}），原文件已保留为 {target.name}，程序从空白历史继续。"
        )

    def _redirect_writes(self, reason: str) -> None:
        self._write_path = self._sidecar("recovered")
        self.recovery_notices.append(
            f"{reason}；本次运行的会话改写到 {self._write_path.name}，原文件保持不动。"
        )

    def _quarantine(self, damaged: list[dict[str, Any]]) -> None:
        target = self._sidecar("damaged")
        try:
            target.write_text(json.dumps(damaged, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as exc:
            self.recovery_notices.append(
                f"{len(damaged)} 条会话记录无法解析，隔离文件写入失败（{type(exc).__name__}），这些记录未载入。"
            )
            return
        self.recovery_notices.append(
            f"{len(damaged)} 条会话记录无法解析，已隔离到 {target.name}，其余会话正常载入。"
        )

    def _sidecar(self, marker: str) -> Path:
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        return self.path.with_name(f"{self.path.stem}.{marker}-{stamp}{self.path.suffix}")

    @staticmethod
    def _backup_path(target: Path) -> Path:
        return target.with_name(f"{target.name}.bak")

    def save(self, sessions: list[ResearchSession]) -> None:
        target = self.write_path
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f"{target.name}.tmp")
        temporary.write_text(
            json.dumps([session.model_dump(mode="json") for session in sessions], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if target.is_file():
            # Renaming is atomic and free, so the previous good file always survives one write.
            os.replace(target, self._backup_path(target))
        os.replace(temporary, target)
