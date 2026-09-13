"""Versioned contracts for one bounded P1→P2→P1→P3 research flow."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

FlowPhase = Literal[
    "p1_initial_complete",
    "p2_ready",
    "p2_needs_review",
    "p1_synthesis_complete",
    "awaiting_forecast_approval",
    "forecast_running",
    "forecast_verified",
    "feedback_complete",
    "failed",
    "stopped",
]
ResumableFlowPhase = Literal[
    "p1_initial_complete",
    "p2_needs_review",
    "p2_ready",
    "p1_synthesis_complete",
    "awaiting_forecast_approval",
    "forecast_running",
]
FlowStage = Literal[
    "p1_initial",
    "p2",
    "p1_synthesis",
    "p3_prepare",
    "p3_execute",
    "p1_feedback",
]
FlowAction = Literal[
    "continue",
    "retry",
    "stop",
    "review_p2",
    "approve_forecast",
    "new_goal",
]
P2ReviewDisposition = Literal["accepted", "corrected", "rejected"]
P2ReviewUse = Literal["analysis", "background_only"]


class P2ReviewItem(BaseModel):
    """One authoritative extraction and the review action currently applicable to it."""

    model_config = ConfigDict(extra="forbid")

    cache_key: str
    document_version_id: str
    revision: str
    title: str
    source_name: str
    source_ref: str
    body: str
    result: dict[str, Any]
    blocking_reasons: tuple[str, ...] = ()
    review_status: Literal["unreviewed", "accepted", "corrected", "rejected"] = "unreviewed"
    review_use: P2ReviewUse | None = None
    reviewer: str | None = None
    review_reason: str | None = None
    required: bool = False
    allowed_dispositions: tuple[P2ReviewDisposition, ...] = (
        "accepted",
        "corrected",
        "rejected",
    )
    allows_background_only: bool = False


class P2ReviewDecision(BaseModel):
    """Version-bound and idempotent P2 review command accepted by the application service."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    flow_id: str
    cache_key: str
    expected_revision: str
    decision: P2ReviewDisposition
    use: P2ReviewUse = "analysis"
    reviewer: str = Field(min_length=1, max_length=128)
    reason: str = Field(min_length=1, max_length=2048)
    request_id: str = Field(min_length=8, max_length=128)
    corrected: dict[str, Any] | None = None


class P2ReviewSummary(BaseModel):
    """Review-gate projection; SQLite remains the underlying source of truth."""

    model_config = ConfigDict(extra="forbid")

    flow_id: str
    phase: FlowPhase
    revision: str
    items: tuple[P2ReviewItem, ...] = ()
    required_pending: int = Field(ge=0)
    required_resolved: int = Field(ge=0)
    optional_unreviewed: int = Field(ge=0)
    quality_failures: tuple[str, ...] = ()
    report_path: Path | None = None

    @property
    def can_revalidate(self) -> bool:
        return self.required_pending == 0


class FlowRunReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_kind: Literal["eda", "news", "forecast", "feedback"]
    run_id: str
    parent_run_id: str | None = None
    artifact_directory: Path
    report_path: Path
    data_fingerprint: str | None = None


class P2ResearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    region_id: Literal["shandong"] = "shandong"
    market: str = "CN-SHANDONG"
    timezone: str = "Asia/Shanghai"
    interval_minutes: int = 15
    price_start: datetime
    price_end: datetime
    information_cutoff: datetime
    anomaly_evidence: dict[str, object] = Field(default_factory=dict)
    anomaly_windows: tuple[dict[str, object], ...] = ()
    questions: tuple[str, ...]
    search_topics: tuple[str, ...]
    required_source_fields: tuple[str, ...] = (
        "source_url",
        "title",
        "content",
        "published_at",
        "fetched_at",
        "content_scope",
    )
    maximum_supplemental_documents: Literal[20] = 20


class FeatureDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    decision: Literal["selected", "excluded", "insufficient_evidence"]
    reason: str
    evidence: dict[str, object] = Field(default_factory=dict)


class ForecastFeedback(BaseModel):
    model_config = ConfigDict(extra="forbid")

    forecast_run_id: str
    plan_id: str
    status: Literal["verified", "not_verified", "not_evaluable"]
    fold_common_observations: tuple[int, int, int]
    aggregate_metrics: dict[str, dict[str, object]]
    error_by_hour: dict[str, float] = Field(default_factory=dict)
    used_news_features: tuple[str, ...] = ()
    excluded_news_features: dict[str, str] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    diagnosis: tuple[str, ...] = ()
    next_steps: tuple[str, ...] = ()
    prediction_path: Path | None = None
    backtest_path: Path | None = None
    metrics_path: Path | None = None
    report_path: Path | None = None
    artifact_directory: Path | None = None
    figure_paths: dict[str, Path] = Field(default_factory=dict)
    output_hash: str | None = None


class FullFlowState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[2] = 2
    flow_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    phase: FlowPhase = "p1_initial_complete"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    request_text: str = ""
    requested_forecast: bool = False
    durable_goal: str = ""
    requested_stages: tuple[FlowStage, ...] = ()
    current_stage: FlowStage | None = "p2"
    last_completed_stage: FlowStage = "p1_initial"
    blocked_reason: str | None = None
    allowed_actions: tuple[FlowAction, ...] = ("continue", "stop")
    stage_attempts: dict[FlowStage, int] = Field(default_factory=dict)
    input_fingerprints: dict[str, str] = Field(default_factory=dict)
    artifact_references: dict[str, Path] = Field(default_factory=dict)
    approved_analysis_scope: tuple[
        Literal["p1_initial", "p2", "p1_synthesis", "p1_feedback"], ...
    ] = ("p1_initial", "p2", "p1_synthesis", "p1_feedback")
    analysis_scope_approved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    p1_request: P2ResearchRequest
    runs: tuple[FlowRunReference, ...]
    p2_news_path: Path | None = None
    p2_feature_path: Path | None = None
    p2_manifest_path: Path | None = None
    p2_knowledge_cutoff: datetime | None = None
    feature_decisions: tuple[FeatureDecision, ...] = ()
    forecast_plan: dict[str, object] | None = None
    forecast_plan_fingerprint: str | None = None
    approved_forecast_plan_id: str | None = None
    forecast_approved_at: datetime | None = None
    max_forecast_runs: Literal[1] = 1
    max_feedback_analysis_runs: Literal[1] = 1
    forecast_runs_used: int = Field(default=0, ge=0, le=1)
    feedback_runs_used: int = Field(default=0, ge=0, le=1)
    feedback: ForecastFeedback | None = None
    stop_reason: str | None = None
    error: str | None = None
    resume_from_phase: ResumableFlowPhase | None = None
    failed_stage: Literal["p2", "p1_synthesis", "p3_prepare", "p3_execute"] | None = None


def synchronize_flow_lifecycle(state: FullFlowState) -> FullFlowState:
    """Derive the durable cursor fields from the authoritative phase and artifacts."""

    requested_stages = state.requested_stages or (
        ("p2", "p1_synthesis", "p3_prepare", "p3_execute", "p1_feedback")
        if state.requested_forecast
        else ("p2", "p1_synthesis")
    )
    stage_by_phase: dict[FlowPhase, FlowStage | None] = {
        "p1_initial_complete": "p2",
        "p2_needs_review": "p2",
        "p2_ready": "p1_synthesis",
        "p1_synthesis_complete": "p3_prepare" if state.requested_forecast else None,
        "awaiting_forecast_approval": "p3_execute",
        "forecast_running": "p3_execute",
        "forecast_verified": None,
        "feedback_complete": None,
        "failed": state.failed_stage,
        "stopped": None,
    }
    last_by_phase: dict[FlowPhase, FlowStage] = {
        "p1_initial_complete": "p1_initial",
        "p2_needs_review": "p1_initial",
        "p2_ready": "p2",
        "p1_synthesis_complete": "p1_synthesis",
        "awaiting_forecast_approval": "p3_prepare",
        "forecast_running": "p3_prepare",
        "forecast_verified": "p3_execute",
        "feedback_complete": "p1_feedback",
        "failed": state.last_completed_stage,
        "stopped": state.last_completed_stage,
    }
    actions_by_phase: dict[FlowPhase, tuple[FlowAction, ...]] = {
        "p1_initial_complete": ("continue", "stop"),
        "p2_needs_review": ("review_p2", "retry", "stop"),
        "p2_ready": ("continue", "stop"),
        "p1_synthesis_complete": (("continue", "stop") if state.requested_forecast else ("new_goal",)),
        "awaiting_forecast_approval": ("approve_forecast", "stop"),
        "forecast_running": ("approve_forecast", "stop"),
        "forecast_verified": ("new_goal",),
        "feedback_complete": ("new_goal",),
        "failed": ("retry", "stop"),
        "stopped": ("new_goal",),
    }
    blocked_reason = None
    if state.phase == "p2_needs_review":
        blocked_reason = state.stop_reason or "P2存在未解决复核项"
    elif state.phase == "awaiting_forecast_approval":
        blocked_reason = "等待用户单独确认P3预测方案"
    elif state.phase == "failed":
        blocked_reason = state.stop_reason or state.error or "当前阶段失败"
    elif state.phase == "stopped":
        blocked_reason = state.stop_reason or "用户停止当前全流程"

    references = dict(state.artifact_references)
    for run in state.runs:
        if run.run_kind == "news":
            stage = "p2"
        elif run.run_kind == "forecast":
            stage = "p3"
        elif run.run_kind == "feedback":
            stage = "p1_feedback"
        elif run.parent_run_id is None:
            stage = "p1_initial"
        else:
            stage = "p1_synthesis"
        references[f"{stage}_artifact_directory"] = run.artifact_directory
        references[f"{stage}_report"] = run.report_path
    for name, path in (
        ("p2_news", state.p2_news_path),
        ("p2_features", state.p2_feature_path),
        ("p2_manifest", state.p2_manifest_path),
    ):
        if path is not None:
            references[name] = path

    return FullFlowState.model_validate(
        {
            **state.model_dump(),
            "schema_version": 2,
            "durable_goal": state.durable_goal or state.request_text,
            "requested_stages": requested_stages,
            "current_stage": stage_by_phase[state.phase],
            "last_completed_stage": last_by_phase[state.phase],
            "blocked_reason": blocked_reason,
            "allowed_actions": actions_by_phase[state.phase],
            "artifact_references": references,
        }
    )
