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

FEATURE_VERSION = "1.4.0"
CAPACITY_EFFECTS = ("supply_up", "supply_down", "demand_up", "demand_down", "transfer_up", "transfer_down",
                    "mixed", "unknown")

FEATURE_EVENT_TYPES: tuple[str, ...] = (
    "generation_outage",
    "generation_restore",
    "transmission_constraint",
    "demand_shock",
    "storage_dispatch",
    "renewable_supply_change",
    "fuel_supply_change",
)

# These event types describe an occurrence rather than an open-ended operating state
# when the source gives no explicit end.  Treating a load record or one dispatch action
# as permanently active would turn one historical bulletin into a constant future
# predictor.  An explicit end still wins and represents a real duration.
POINT_EVENT_TYPES = frozenset({"generation_restore", "demand_shock", "storage_dispatch"})


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
            entity_resolution=event.entity_resolution,
            capacity_mw=event.capacity_mw,
            effective_start_at=event.effective_start_at,
            effective_end_at=event.effective_end_at,
            direction=event.direction,
            status=event.status,
            physical_effect=event.physical_effect,
            review_status=event.review_status,
        ),
        event.announcement_available_at,
    )


def _restoration_end(
    outage_asset_keys: tuple[str, ...],
    outage_start: datetime,
    events: Sequence[MergedEvent],
    knowledge_at: datetime,
    market_keys: tuple[str, ...] = (),
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
            and state.review_status != "rejected"
            and state.asset_keys == outage_asset_keys
            and (not market_keys or state.market_keys == market_keys)
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
    if state.event_type in POINT_EVENT_TYPES and state.effective_end_at is None:
        if state.event_type == "generation_restore":
            # A late restoration becomes actionable when it is known because it closes
            # an open outage state at that decision time.
            observable_at = max(state.effective_start_at, state_available_at)
            duration = interval_end - interval_start
            return observable_at <= interval_start < observable_at + duration
        # A historical demand peak or dispatch action is not moved to publication time.
        # If it arrived after its settlement interval, the announcement pulse carries
        # the newly available information while the physical-event feature stays zero.
        return state_available_at <= interval_start <= state.effective_start_at < interval_end
    if state.effective_start_at >= interval_end:
        return False
    effective_end = state.effective_end_at
    restored_at = (
        _restoration_end(state.asset_keys, state.effective_start_at, events, knowledge_at, state.market_keys)
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
        upcoming: list[tuple[MergedEvent, EventStateRevision]] = []
        announced: list[str] = []
        for event in short_term:
            known = _known_state(event, knowledge_at)
            if known is None:
                continue
            state, state_available_at = known
            if state.relevance != "short_term":
                continue
            if state.entity_resolution is not None and not state.entity_resolution.matches_market(clock.market):
                continue
            if state.status == "cancelled" or state.review_status == "rejected":
                continue
            if interval_start - clock.interval < event.announcement_available_at <= knowledge_at:
                announced.append(event.event_id)
            if state.effective_start_at is not None and state.effective_start_at > interval_start:
                upcoming.append((event, state))
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

        by_effect = {}
        for effect in CAPACITY_EFFECTS:
            values = [state.capacity_mw for _, state in active_with_state if state.physical_effect == effect]
            by_effect[effect] = None if any(value is None for value in values) else round(sum(values), 6)

        rows.append(
            EventFeatureRow(
                interval_start=interval_start,
                active_event_count=len(active),
                active_capacity_mw=capacity,
                direction_up_count=sum(1 for _, state in active_with_state if state.direction == "up"),
                direction_down_count=sum(1 for _, state in active_with_state if state.direction == "down"),
                event_type_counts=counts,
                source_event_ids=tuple(sorted(event.event_id for event in active)),
                new_announcement_count=len(announced),
                new_announcement_event_ids=tuple(sorted(announced)),
                upcoming_event_count=len(upcoming),
                upcoming_event_ids=tuple(sorted(event.event_id for event, _ in upcoming)),
                next_effective_in_hours=min(
                    ((state.effective_start_at - interval_start).total_seconds() / 3600 for _, state in upcoming),
                    default=None,
                ),
                capacity_by_effect_mw=by_effect,
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
            "rows": [row.model_dump(mode="json") for row in rows],
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
        "new_announcement_count", "upcoming_event_count", "next_effective_in_hours",
        *(f"capacity_{effect}_mw" for effect in CAPACITY_EFFECTS),
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
            str(row.new_announcement_count), str(row.upcoming_event_count),
            "" if row.next_effective_in_hours is None else f"{row.next_effective_in_hours:g}",
            *("" if row.capacity_by_effect_mw.get(effect) is None else f"{row.capacity_by_effect_mw[effect]:g}"
              for effect in CAPACITY_EFFECTS),
        )
        for row in snapshot.rows
    )
    return (header, *body)
