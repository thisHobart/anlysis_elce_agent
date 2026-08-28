"""Safe compatibility boundaries for persisted research-loop checkpoints."""

from __future__ import annotations

from typing import Any


def safe_incompatible_state(values: dict[str, Any], *, target_version: int) -> dict[str, Any]:
    """Project an old checkpoint without validating versioned nested contracts."""

    source_version = values.get("graph_schema_version")
    message = (
        f"研究循环 checkpoint 版本 {source_version!r} 与当前版本 {target_version} 不兼容；"
        "旧状态和此前报告仍然保留。请新建对话并重新提交研究问题。"
    )
    raw_events = values.get("events")
    events = [item for item in raw_events if isinstance(item, dict)][-40:] if isinstance(raw_events, list) else []
    return {
        "graph_schema_version": source_version,
        "thread_id": str(values.get("thread_id") or values.get("session_id") or ""),
        "session_id": str(values.get("session_id") or values.get("thread_id") or ""),
        "phase": "stopped",
        "control": "schema_upgrade_required",
        "user_request": str(values.get("user_request") or ""),
        "latest_turn": str(values.get("latest_turn") or values.get("user_request") or ""),
        "assistant_message": message,
        "stop_reason": message,
        "schema_upgrade_required": True,
        "events": events,
    }
