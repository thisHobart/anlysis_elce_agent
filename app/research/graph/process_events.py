"""Versioned events for the user-visible think, act, observe timeline."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

PROCESS_EVENT_PREFIX = "__VPP_PROCESS_EVENT__"
MAX_PROCESS_EVENTS = 320

ProcessEventType = Literal[
    "thinking_started",
    "thinking_ready",
    "action_started",
    "action_completed",
    "action_failed",
    "step_completed",
]
ProcessSource = Literal["model", "system"]


class ProcessEvent(BaseModel):
    """One idempotent update to a visible decision step."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    event_id: str = Field(default_factory=lambda: uuid4().hex)
    session_id: str
    flow_id: str
    round: int = Field(ge=0)
    step_id: str = Field(min_length=1)
    sequence: int = Field(ge=1)
    event_type: ProcessEventType
    phase: str
    source: ProcessSource
    title: str = ""
    content: str = ""
    action_id: str | None = None
    tool_name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: str = ""
    report_path: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


def append_process_events(
    existing: list[dict[str, Any]],
    *events: ProcessEvent,
) -> list[dict[str, Any]]:
    """Append new events while retaining bounded, unique history."""

    merged = [item for item in existing if isinstance(item, dict)]
    known = {str(item.get("event_id", "")) for item in merged}
    for event in events:
        if event.event_id in known:
            continue
        merged.append(event.model_dump(mode="json"))
        known.add(event.event_id)
    return merged[-MAX_PROCESS_EVENTS:]


def process_event_message(event: ProcessEvent) -> str:
    """Encode an event for the existing desktop worker progress signal."""

    return PROCESS_EVENT_PREFIX + json.dumps(
        event.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def split_process_event_message(message: str) -> ProcessEvent | None:
    """Decode a process event, returning ``None`` for legacy progress text."""

    if not message.startswith(PROCESS_EVENT_PREFIX):
        return None
    return ProcessEvent.model_validate_json(message[len(PROCESS_EVENT_PREFIX) :])
