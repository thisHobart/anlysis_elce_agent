"""Shared bounded conversation and Episode-memory policy."""

from __future__ import annotations

from app.research.agent.context import (
    DEFAULT_HISTORY_CHARACTER_BUDGET,
    bounded_recent_history,
    compact_episode_context,
)
from app.research.agent.schemas import ConversationMessage


def test_recent_history_is_bounded_by_count_and_character_budget() -> None:
    history = [
        ConversationMessage(message_id=f"m-{index}", role="user", content=str(index) * 10)
        for index in range(6)
    ]

    assert [item.message_id for item in bounded_recent_history(history, max_messages=3)] == [
        "m-3",
        "m-4",
        "m-5",
    ]
    assert [
        item.message_id
        for item in bounded_recent_history(history, max_messages=6, max_characters=15)
    ] == ["m-5"]


def test_episode_context_keeps_recent_evidence_without_unbounded_payloads() -> None:
    summaries = [
        {
            "episode_id": f"episode-{index}",
            "episode_number": index,
            "goal": f"目标 {index}",
            "status": "accepted",
            "findings": [f"发现 {index}"],
            "heavy_internal_field": {"raw": list(range(100))},
        }
        for index in range(1, 11)
    ]

    compact = compact_episode_context(summaries, limit=3)
    assert [item["episode_id"] for item in compact] == ["episode-8", "episode-9", "episode-10"]
    assert all("heavy_internal_field" not in item for item in compact)


def test_single_oversized_message_is_hard_truncated_without_mutating_source() -> None:
    source = ConversationMessage(role="user", content="x" * (DEFAULT_HISTORY_CHARACTER_BUDGET + 500))

    selected = bounded_recent_history([source], max_messages=8)

    assert len(selected[0].content) == DEFAULT_HISTORY_CHARACTER_BUDGET
    assert selected[0].content.startswith("…[前文已截断]")
    assert len(source.content) == DEFAULT_HISTORY_CHARACTER_BUDGET + 500


def test_episode_context_rejects_stale_or_different_data() -> None:
    summaries = [
        {
            "episode_id": "matching",
            "episode_number": 1,
            "memory_status": "active",
            "data_fingerprint": "a" * 12,
        },
        {
            "episode_id": "stale",
            "episode_number": 2,
            "memory_status": "stale",
            "data_fingerprint": "a" * 12,
        },
        {
            "episode_id": "different",
            "episode_number": 3,
            "memory_status": "active",
            "data_fingerprint": "b" * 12,
        },
    ]

    compact = compact_episode_context(summaries, data_fingerprint="a" * 12)

    assert [item["episode_id"] for item in compact] == ["matching"]
