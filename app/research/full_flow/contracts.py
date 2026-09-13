"""Versioned contracts for one bounded P1→P2→P1→P3 research flow."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
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

    schema_version: Literal[1] = 1
    flow_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    phase: FlowPhase = "p1_initial_complete"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    request_text: str = ""
    requested_forecast: bool = False
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
