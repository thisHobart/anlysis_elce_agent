"""Event-derived numeric features, and the leak-free export P1 is allowed to read.

The hand-off rule this module enforces: a feature row for interval T may only use events
that were already announced by T. P2 does the gating here, before export, because the P1
loader treats `available_at` as reporting evidence rather than a row-level filter.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import datetime

from app.research.news.clock import MarketClock
from app.research.news.contracts import (
    EventFeatureRow,
    EventFeatureSnapshot,
    EventStateRevision,
    MergedEvent,
)

FEATURE_VERSION = "1.1.0"

FEATURE_EVENT_TYPES: tuple[str, ...] = (
    "generation_outage",
    "generation_restore",
    "transmission_constraint",
    "demand_shock",
    "renewable_supply_change",
    "fuel_supply_change",
)

# A restoration is an event impulse, not a state that stays "active" forever. If its
# source gives no end, it contributes to exactly the settlement interval in which it
# occurs. Persistent event types retain their open-ended semantics until a later update.
POINT_EVENT_TYPES = frozenset({"generation_restore"})


class EventFeatureError(ValueError):
    """Raised when features cannot be built without violating the availability rule."""


def _known_state(
    event: MergedEvent, knowledge_at: datetime
) -> tuple[EventStateRevision, datetime] | None:
    visible = [item for item in event.state_history if item.available_at <= knowledge_at]
    if visible:
        return visible[-1], visible[-1].available_at
    if event.announcement_available_at > knowledge_at:
        return None
    return (
        EventStateRevision(
            available_at=event.announcement_available_at,
            relevance=event.relevance,
            event_type=event.event_type,
            affected_regions=event.affected_regions,
            affected_assets=event.affected_assets,
            capacity_mw=event.capacity_mw,
            effective_start_at=event.effective_start_at,
            effective_end_at=event.effective_end_at,
            direction=event.direction,
            status=event.status,
            physical_effect=event.physical_effect,
        ),
        event.announcement_available_at,
    )


def _restoration_end(
    outage_asset_keys: tuple[str, ...],
    outage_start: datetime,
    events: Sequence[MergedEvent],
    knowledge_at: datetime,
) -> datetime | None:
    """Conservatively pair a restoration with an outage only on the same named asset."""

    if not outage_asset_keys:
        return None
    ends: list[datetime] = []
    for candidate in events:
        known = _known_state(candidate, knowledge_at)
        if known is None:
            continue
        state, state_available_at = known
        if (
            state.event_type == "generation_restore"
            and state.status != "cancelled"
            and state.asset_keys == outage_asset_keys
            and state.effective_start_at is not None
            and state.effective_start_at >= outage_start
        ):
            ends.append(max(state.effective_start_at, state_available_at))
    return min(ends) if ends else None


def _is_active(
    event: MergedEvent,
    state: EventStateRevision,
    state_available_at: datetime,
    interval_start: datetime,
    interval_end: datetime,
    *,
    events: Sequence[MergedEvent],
    knowledge_at: datetime,
) -> bool:
    """Active means the event is both already announced and currently in effect."""

    if state.status == "cancelled" or state.effective_start_at is None:
        return False
    if state.event_type in POINT_EVENT_TYPES:
        # A point event contributes once, at the first decision interval where it is both
        # effective and knowable. This preserves a late-arriving restoration without
        # leaking it into the interval that began before the bulletin arrived.
        observable_at = max(state.effective_start_at, state_available_at)
        duration = interval_end - interval_start
        return observable_at <= interval_start < observable_at + duration
    if state.effective_start_at >= interval_end:
        return False
    effective_end = state.effective_end_at
    restored_at = (
        _restoration_end(state.asset_keys, state.effective_start_at, events, knowledge_at)
        if state.event_type == "generation_outage"
        else None
    )
    if restored_at is not None and (effective_end is None or restored_at < effective_end):
        effective_end = restored_at
    return not (effective_end is not None and effective_end <= interval_start)


def build_event_features(
    events: Sequence[MergedEvent],
    *,
    clock: MarketClock,
    start_at: datetime,
    end_at: datetime,
    as_of: datetime,
    feature_version: str = FEATURE_VERSION,
) -> EventFeatureSnapshot:
    """One row per settlement interval, counting only what was knowable at that interval."""

    usable = [event for event in events if event.announcement_available_at <= as_of]
    later = [event for event in events if event.announcement_available_at > as_of]
    if later:
        raise EventFeatureError(
            f"{len(later)} 个事件的公告时间晚于 as_of；事件视图必须先按 as_of 重建再构造特征"
        )

    short_term = [
        event
        for event in usable
        if event.relevance == "short_term"
        or any(revision.relevance == "short_term" for revision in event.state_history)
    ]
    # A historical snapshot must not contain rows whose decision timestamp is later than
    # its information cutoff.  Otherwise the CSV export would label a future row as
    # ``available_at == timestamp`` even though it was produced using an earlier snapshot.
    snapshot_end_at = min(clock.floor(end_at), clock.floor(as_of))
    if snapshot_end_at < clock.floor(start_at):
        raise EventFeatureError("as_of 早于特征时间范围，无法构造非空的历史快照")
    rows: list[EventFeatureRow] = []
    for interval_start in clock.grid(start_at, snapshot_end_at):
        interval_end = interval_start + clock.interval
        knowledge_at = min(interval_start, as_of)
        active_with_state: list[tuple[MergedEvent, EventStateRevision]] = []
        for event in short_term:
            known = _known_state(event, knowledge_at)
            if known is None:
                continue
            state, state_available_at = known
            if state.relevance != "short_term":
                continue
            if _is_active(
                event,
                state,
                state_available_at,
                interval_start,
                interval_end,
                events=short_term,
                knowledge_at=knowledge_at,
            ):
                active_with_state.append((event, state))
        active = [event for event, _ in active_with_state]
        counts = {event_type: 0 for event_type in FEATURE_EVENT_TYPES}
        for _, state in active_with_state:
            if state.event_type in counts:
                counts[state.event_type] += 1

        capacities = [state.capacity_mw for _, state in active_with_state if state.capacity_mw is not None]
        if not active:
            capacity: float | None = 0.0
        elif len(capacities) == len(active):
            capacity = round(float(sum(capacities)), 6)
        else:
            # At least one active event has unknown size, so the total is not a known number.
            capacity = None

        rows.append(
            EventFeatureRow(
                interval_start=interval_start,
                active_event_count=len(active),
                active_capacity_mw=capacity,
                direction_up_count=sum(1 for _, state in active_with_state if state.direction == "up"),
                direction_down_count=sum(1 for _, state in active_with_state if state.direction == "down"),
                event_type_counts=counts,
                source_event_ids=tuple(sorted(event.event_id for event in active)),
            )
        )

    snapshot_rows = tuple(rows)
    return EventFeatureSnapshot(
        feature_version=feature_version,
        market=clock.market,
        market_timezone=clock.timezone,
        interval_minutes=clock.interval_minutes,
        as_of=as_of,
        rows=snapshot_rows,
        source_event_ids=tuple(sorted({event.event_id for event in short_term})),
        content_hash=_snapshot_hash(snapshot_rows, clock=clock, as_of=as_of, feature_version=feature_version),
    )


def _snapshot_hash(
    rows: tuple[EventFeatureRow, ...], *, clock: MarketClock, as_of: datetime, feature_version: str
) -> str:
    payload = json.dumps(
        {
            "as_of": as_of.isoformat(),
            "feature_version": feature_version,
            "interval_minutes": clock.interval_minutes,
            "market": clock.market,
            "rows": [
                {
                    "active_capacity_mw": row.active_capacity_mw,
                    "active_event_count": row.active_event_count,
                    "direction_down_count": row.direction_down_count,
                    "direction_up_count": row.direction_up_count,
                    "event_type_counts": row.event_type_counts,
                    "interval_start": row.interval_start.isoformat(),
                    "source_event_ids": list(row.source_event_ids),
                }
                for row in rows
            ],
            "timezone": clock.timezone,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def snapshot_to_csv_rows(snapshot: EventFeatureSnapshot) -> tuple[tuple[str, ...], ...]:
    """Flat numeric table for the P1 hand-off.

    `available_at` equals `timestamp` by construction: the gating already happened, so a
    downstream reader that ignores the column still cannot see the future through this file.
    """

    header = (
        "timestamp",
        "available_at",
        "active_event_count",
        "active_capacity_mw",
        "direction_up_count",
        "direction_down_count",
        *(f"count_{event_type}" for event_type in FEATURE_EVENT_TYPES),
    )
    body = tuple(
        (
            row.interval_start.isoformat().replace("+00:00", "Z"),
            row.interval_start.isoformat().replace("+00:00", "Z"),
            str(row.active_event_count),
            "" if row.active_capacity_mw is None else f"{row.active_capacity_mw:g}",
            str(row.direction_up_count),
            str(row.direction_down_count),
            *(str(row.event_type_counts.get(event_type, 0)) for event_type in FEATURE_EVENT_TYPES),
        )
        for row in snapshot.rows
    )
    return (header, *body)
