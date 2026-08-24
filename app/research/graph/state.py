"""Serializable state for the persistent research loop."""

from __future__ import annotations

from typing import Any, TypedDict


class ResearchLoopState(TypedDict, total=False):
    graph_schema_version: int
    thread_id: str
    session_id: str
    phase: str
    pending_user_message: str
    user_request: str
    messages: list[dict[str, Any]]
    study_config: dict[str, Any] | None
    data_profile: dict[str, Any] | None
    quality_report: dict[str, Any] | None
    data_fingerprint: str | None
    available_skills: list[dict[str, Any]]
    active_skill: dict[str, Any] | None
    decision: dict[str, Any] | None
    current_plan: dict[str, Any] | None
    plan_history: list[dict[str, Any]]
    plan_fingerprints: list[str]
    plan_origin: str
    authorization_envelope: dict[str, Any] | None
    approval_state: dict[str, Any]
    approval_timeout_seconds: int
    tool_queue: list[dict[str, Any]]
    tool_cursor: int
    tool_records: dict[str, dict[str, Any]]
    tool_results: list[dict[str, Any]]
    pending_tool_result: dict[str, Any] | None
    tool_outcome: str
    evaluation: dict[str, Any] | None
    eda_summary: dict[str, Any] | None
    feedback_packets: list[dict[str, Any]]
    budget: dict[str, Any]
    evidence_fingerprints: list[str]
    run_history: list[dict[str, Any]]
    latest_run: dict[str, Any] | None
    loop_records: list[str]
    assistant_message: str
    stop_reason: str | None
    events: list[dict[str, Any]]
