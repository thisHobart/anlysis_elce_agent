"""Leak-free P2 event features for historical and future P3 decision rows."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import datetime

import pandas as pd

from app.research.news.clock import MarketClock
from app.research.news.contracts import MergedEvent
from app.research.news.features import (
    CAPACITY_EFFECTS,
    FEATURE_EVENT_TYPES,
    _is_active,
    _known_state,
)


def build_forecast_news_features(
    events: Sequence[MergedEvent],
    *,
    clock: MarketClock,
    start_at: datetime,
    end_at: datetime,
    knowledge_cutoff: datetime,
) -> pd.DataFrame:
    """Project event features while freezing knowledge at each prediction origin.

    Historical rows use the information known at their own timestamp. Future rows
    use only event states visible at ``knowledge_cutoff``. Their ``available_at``
    records that fixed cutoff so the forecast adapter can enforce it again.
    """

    cutoff = knowledge_cutoff.astimezone(clock.zone)
    rows: list[dict[str, object]] = []
    usable = [
        event
        for event in events
        if event.announcement_available_at <= knowledge_cutoff
        and (
            event.relevance == "short_term"
            or any(revision.relevance == "short_term" for revision in event.state_history)
        )
    ]
    for interval_start in clock.grid(start_at, end_at):
        knowledge_at = min(interval_start, cutoff)
        interval_end = interval_start + clock.interval
        active = []
        upcoming = []
        announced = []
        for event in usable:
            known = _known_state(event, knowledge_at)
            if known is None:
                continue
            state, state_available_at = known
            if state.relevance != "short_term":
                continue
            if state.analysis_eligibility != "eligible":
                continue
            if state.entity_resolution is not None and not state.entity_resolution.matches_market(clock.market):
                continue
            if state.status == "cancelled" or state.review_status == "rejected":
                continue
            if interval_start <= cutoff and interval_start - clock.interval < event.announcement_available_at <= knowledge_at:
                announced.append(event.event_id)
            if state.effective_start_at is not None and state.effective_start_at > interval_start:
                upcoming.append((event, state))
            if _is_active(
                event,
                state,
                state_available_at,
                interval_start,
                interval_end,
                events=usable,
                knowledge_at=knowledge_at,
            ):
                active.append((event, state))
        counts = {name: 0 for name in FEATURE_EVENT_TYPES}
        for _event, state in active:
            if state.event_type in counts:
                counts[state.event_type] += 1
        known_capacities = [state.capacity_mw for _event, state in active if state.capacity_mw is not None]
        active_capacity = (
            0.0
            if not active
            else (float(sum(known_capacities)) if len(known_capacities) == len(active) else None)
        )
        by_effect: dict[str, float | None] = {}
        for effect in CAPACITY_EFFECTS:
            values = [state.capacity_mw for _event, state in active if state.physical_effect == effect]
            by_effect[effect] = None if any(value is None for value in values) else float(sum(values))
        rows.append(
            {
                "timestamp": interval_start,
                "available_at": knowledge_at,
                "active_event_count": len(active),
                "active_capacity_mw": active_capacity,
                "direction_up_count": sum(state.direction == "up" for _event, state in active),
                "direction_down_count": sum(state.direction == "down" for _event, state in active),
                **{f"count_{name}": value for name, value in counts.items()},
                "new_announcement_count": len(announced),
                "upcoming_event_count": len(upcoming),
                "next_effective_in_hours": min(
                    (
                        (state.effective_start_at - interval_start).total_seconds() / 3600
                        for _event, state in upcoming
                        if state.effective_start_at is not None
                    ),
                    default=None,
                ),
                **{f"capacity_{name}_mw": value for name, value in by_effect.items()},
            }
        )
    return pd.DataFrame(rows)


def forecast_news_feature_hash(frame: pd.DataFrame) -> str:
    """Stable content hash used by P2/P3 hand-off manifests."""

    payload = frame.copy()
    for column in ("timestamp", "available_at"):
        payload[column] = pd.to_datetime(payload[column]).map(lambda value: value.isoformat())
    payload = payload.astype(object).where(pd.notna(payload), None)
    encoded = json.dumps(payload.to_dict(orient="records"), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
