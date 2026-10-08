"""Shared context-window and durable episode-memory policy."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeVar

MAX_PERSISTED_CONVERSATION_MESSAGES = 120
MAX_EPISODE_SUMMARIES = 48
DEFAULT_HISTORY_CHARACTER_BUDGET = 24_000
DEFAULT_EPISODE_CONTEXT_ITEMS = 8

T = TypeVar("T")


def _message_content(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("content") or "")
    return str(getattr(item, "content", "") or "")


def _with_message_content(item: T, content: str) -> T:
    """Copy one supported message value with bounded content."""

    if isinstance(item, dict):
        return {**item, "content": content}  # type: ignore[return-value]
    model_copy = getattr(item, "model_copy", None)
    if callable(model_copy):
        return model_copy(update={"content": content})
    raise TypeError(f"unsupported conversation message type: {type(item).__name__}")


def _truncate_recent_content(content: str, limit: int) -> str:
    """Keep the recent end of an oversized message inside a hard budget."""

    if len(content) <= limit:
        return content
    marker = "…[前文已截断]"
    if limit <= len(marker):
        return marker[-limit:]
    return marker + content[-(limit - len(marker)) :]


def bounded_recent_history(
    history: Sequence[T],
    *,
    max_messages: int,
    max_characters: int = DEFAULT_HISTORY_CHARACTER_BUDGET,
) -> list[T]:
    """Return a recent window bounded by both messages and approximate tokens.

    Character count is deliberately provider-neutral.  It prevents one unusually
    large pasted message from consuming an otherwise message-count-only window.
    """

    if max_characters < 1:
        return []
    selected: list[T] = []
    used = 0
    for item in reversed(history[-max(1, max_messages) :]):
        content = _message_content(item)
        remaining = max_characters - used
        if remaining <= 0:
            break
        if len(content) > remaining:
            # The newest message is still useful, but it must not be allowed to
            # defeat the advertised context budget by itself.
            if not selected:
                selected.append(_with_message_content(item, _truncate_recent_content(content, remaining)))
            break
        selected.append(item)
        used += len(content)
    selected.reverse()
    return selected


def compact_episode_context(
    summaries: Sequence[dict[str, Any]],
    *,
    limit: int = DEFAULT_EPISODE_CONTEXT_ITEMS,
    data_fingerprint: str | None = None,
) -> list[dict[str, Any]]:
    """Project active, data-scoped historical evidence into model context.

    When a data fingerprint is supplied, only evidence produced from that
    exact immutable study snapshot is eligible.  This prevents a restored UI
    session from silently treating results from replaced input files as facts
    about the current data.
    """

    keys = (
        "episode_id",
        "episode_number",
        "goal",
        "status",
        "run_id",
        "plan_id",
        "evaluation_decision",
        "summary",
        "findings",
        "warnings",
        "report_path",
        "figure_count",
        "figure_keys",
        "data_fingerprint",
        "study_name",
        "target_name",
        "study_start_time",
        "study_end_time",
        "skill_name",
        "skill_version",
    )
    eligible = [
        item
        for item in summaries
        if item.get("memory_status", "active") == "active"
        and (data_fingerprint is None or item.get("data_fingerprint") == data_fingerprint)
    ]
    result: list[dict[str, Any]] = []
    for item in eligible[-max(1, limit) :]:
        compact = {key: item.get(key) for key in keys}
        for key, maximum in (("goal", 512), ("summary", 1200), ("report_path", 512)):
            value = compact.get(key)
            if value is not None:
                compact[key] = str(value)[:maximum]
        for key in ("findings", "warnings"):
            values = compact.get(key)
            compact[key] = [str(value)[:512] for value in (values or [])[:8]]
        compact["figure_keys"] = [str(value)[:128] for value in (compact.get("figure_keys") or [])[:32]]
        result.append(compact)
    return result
