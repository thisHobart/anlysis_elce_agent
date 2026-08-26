"""Safe compatibility boundaries for persisted research-loop checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def safe_incompatible_state(values: dict[str, Any], *, target_version: int) -> dict[str, Any]:
    """Project an old checkpoint without validating versioned nested contracts."""

    source_version = values.get("graph_schema_version")
    message = (
        f"研究循环 checkpoint 版本 {source_version!r} 与当前版本 {target_version} 不兼容；"
        "原始状态已保留，下一条研究消息会安全重建循环。"
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


def archive_incompatible_state(
    *,
    thread_id: str,
    values: dict[str, Any],
    checkpoint_path: Path | None,
) -> Path | None:
    """Write a best-effort JSON backup beside a persistent SQLite checkpoint."""

    if checkpoint_path is None:
        return None
    safe_thread = hashlib.sha256(thread_id.encode("utf-8")).hexdigest()[:16]
    created_at = datetime.now(UTC)
    root = checkpoint_path.parent / "checkpoint_backups" / safe_thread
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"schema-{values.get('graph_schema_version', 'unknown')}-{created_at:%Y%m%dT%H%M%SZ}.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(values, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return destination
