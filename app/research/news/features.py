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
from app.research.news.contracts import EventFeatureRow, EventFeatureSnapshot, MergedEvent

FEATURE_VERSION = "1.0.0"

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


def _is_active(event: MergedEvent, interval_start: datetime, interval_end: datetime) -> bool:
    """Active means the event is both already announced and currently in effect."""

    if event.effective_start_at is None:
        return False
    if event.event_type in POINT_EVENT_TYPES:
        # A point event contributes once, at the first decision interval where it is both
        # effective and knowable. This preserves a late-arriving restoration without
        # leaking it into the interval that began before the bulletin arrived.
        observable_at = max(event.effective_start_at, event.announcement_available_at)
        duration = interval_end - interval_start
        return observable_at <= interval_start < observable_at + duration
    if event.announcement_available_at > interval_start:
        return False
    if event.effective_start_at >= interval_end:
        return False
    effective_end = event.effective_end_at
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

    short_term = [event for event in usable if event.relevance == "short_term"]
    rows: list[EventFeatureRow] = []
    for interval_start in clock.grid(start_at, end_at):
        interval_end = interval_start + clock.interval
        active = [event for event in short_term if _is_active(event, interval_start, interval_end)]
        counts = {event_type: 0 for event_type in FEATURE_EVENT_TYPES}
        for event in active:
            if event.event_type in counts:
                counts[event.event_type] += 1

        capacities = [event.capacity_mw for event in active if event.capacity_mw is not None]
        if not active:
            # No active event is a known zero, not an unknown; the contract keeps them apart.
            capacity: float | None = None
        elif capacities:
            capacity = round(float(sum(capacities)), 6)
        else:
            capacity = 0.0

        rows.append(
            EventFeatureRow(
                interval_start=interval_start,
                active_event_count=len(active),
                active_capacity_mw=capacity,
                direction_up_count=sum(1 for event in active if event.direction == "up"),
                direction_down_count=sum(1 for event in active if event.direction == "down"),
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
