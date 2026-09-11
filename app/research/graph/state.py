"""Serializable state for the persistent research loop."""

from __future__ import annotations

from typing import Any, TypedDict


class ResearchLoopState(TypedDict, total=False):
    graph_schema_version: int
    schema_upgrade_required: bool
    thread_id: str
    session_id: str
    phase: str
    control: str
    return_to_gate: str | None
    loop_cursor: dict[str, Any]
    episode_history: list[dict[str, Any]]
    episode_summaries: list[dict[str, Any]]
    pending_user_message: str
    pending_message_id: str | None
    pending_turn_id: str | None
    active_turn_id: str | None
    user_request: str
    latest_turn: str
    explanation_request: str
    messages: list[dict[str, Any]]
    study_config: dict[str, Any] | None
    data_profile: dict[str, Any] | None
    data_summary: dict[str, Any] | None
    quality_report: dict[str, Any] | None
    active_skill: dict[str, Any] | None
    decision: dict[str, Any] | None
    current_plan: dict[str, Any] | None
    plan_history: list[dict[str, Any]]
    plan_fingerprints: list[str]
    planning_failure_fingerprints: list[str]
    user_interrupt_kind: str | None
    plan_origin: str
    revision_cycle_id: str
    progress_records: list[dict[str, Any]]
    authorization_envelope: dict[str, Any] | None
    approval_state: dict[str, Any]
    approval_timeout_seconds: int
    automatic_approval_enabled: bool
    tool_queue: list[dict[str, Any]]
    tool_cursor: int
    tool_records: dict[str, dict[str, Any]]
    tool_result_cache: dict[str, dict[str, Any]]
    tool_results: list[dict[str, Any]]
    pending_tool_result: dict[str, Any] | None
    evaluation: dict[str, Any] | None
    eda_summary: dict[str, Any] | None
    feedback_packets: list[dict[str, Any]]
    feedback_history: list[dict[str, Any]]
    budget: dict[str, Any]
    evidence_fingerprints: list[str]
    agenda_fingerprints: list[str]
    run_history: list[dict[str, Any]]
    latest_run: dict[str, Any] | None
    loop_records: list[str]
    assistant_message: str
    stop_reason: str | None
    events: list[dict[str, Any]]
