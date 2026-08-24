"""Persist positive and negative loop outcomes as idempotent JSON records."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def write_loop_record(*, thread_id: str, state: dict[str, Any], outcome: str) -> Path:
    if not SAFE_ID.fullmatch(thread_id):
        raise ValueError("invalid research thread id")
    payload = {
        "graph_schema_version": state.get("graph_schema_version"),
        "thread_id": thread_id,
        "outcome": outcome,
        "phase": state.get("phase"),
        "active_skill": state.get("active_skill"),
        "current_plan": state.get("current_plan"),
        "plan_history": state.get("plan_history", []),
        "feedback_packets": state.get("feedback_packets", []),
        "tool_records": state.get("tool_records", {}),
        "evaluation": state.get("evaluation"),
        "run_history": state.get("run_history", []),
        "budget": state.get("budget", {}),
        "stop_reason": state.get("stop_reason"),
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    fingerprint = hashlib.sha256(canonical).hexdigest()[:16]
    root = Path(__file__).resolve().parents[3] / "artifacts" / "research" / "loops" / thread_id
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"{fingerprint}.json"
    if destination.is_file():
        return destination
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(destination)
    return destination
