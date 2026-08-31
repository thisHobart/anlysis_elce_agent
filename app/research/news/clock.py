"""One market clock for news, events and prices, plus the three time axes P2 analyses on."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.research.news.contracts import MergedEvent

TimeAxis = Literal["announcement", "effective"]

CLOCK_VERSION = "1.0.0"


class MarketClockError(ValueError):
    """Raised when an instant cannot be placed on the market's settlement grid."""


@dataclass(frozen=True)
class MarketClock:
    """The single grid every P2 timestamp is placed on.

    Keeping this in one object is what stops a study from silently mixing a publication
    stamp, an event stamp and a settlement stamp that were each rounded differently.
    """

    market: str
    timezone: str = "UTC"
    interval_minutes: int = 30
    clock_version: str = CLOCK_VERSION

    def __post_init__(self) -> None:
        if not self.market:
            raise MarketClockError("market 不能为空")
        if self.interval_minutes <= 0 or 1440 % self.interval_minutes != 0:
            raise MarketClockError("结算间隔必须为正数且能整除一天")
        try:
            ZoneInfo(self.timezone)
        except ZoneInfoNotFoundError as exc:
            raise MarketClockError(f"未知市场时区：{self.timezone}") from exc

    @property
    def zone(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    @property
    def interval(self) -> timedelta:
        return timedelta(minutes=self.interval_minutes)

    def local(self, instant: datetime) -> datetime:
        return self._utc(instant).astimezone(self.zone)

    def floor(self, instant: datetime) -> datetime:
        """Snap down to the settlement interval this instant falls inside."""

        local = self.local(instant)
        minute_of_day = local.hour * 60 + local.minute
        floored_minute = (minute_of_day // self.interval_minutes) * self.interval_minutes
        floored_local = local.replace(
            hour=floored_minute // 60,
            minute=floored_minute % 60,
            second=0,
            microsecond=0,
        )
        return floored_local.astimezone(UTC)

    def ceil(self, instant: datetime) -> datetime:
        moment = self._utc(instant)
        floored = self.floor(moment)
        return floored if floored == moment else floored + self.interval

    def grid(self, start: datetime, end: datetime) -> tuple[datetime, ...]:
        first, last = self.floor(start), self.floor(end)
        if last < first:
            raise MarketClockError("网格结束时间不能早于开始时间")
        count = int((last - first) / self.interval) + 1
        return tuple(first + index * self.interval for index in range(count))

    def _utc(self, instant: datetime) -> datetime:
        if instant.tzinfo is None or instant.utcoffset() is None:
            raise MarketClockError("时间必须带时区")
        return instant.astimezone(UTC)


@dataclass(frozen=True)
class LeadTimeReading:
    """How far ahead of the event the market could have known, per the doc's third axis."""

    event_id: str
    announcement_available_at: datetime
    effective_start_at: datetime | None
    lead_time_hours: float | None

    @property
    def is_after_the_fact(self) -> bool:
        """True when the news only became available after the event had already started."""

        return self.lead_time_hours is not None and self.lead_time_hours < 0

    @property
    def is_advance_notice(self) -> bool:
        return self.lead_time_hours is not None and self.lead_time_hours > 0


def zero_point(event: MergedEvent, axis: TimeAxis) -> datetime | None:
    """The instant an analysis measures from; declaring it is mandatory, never implied."""

    if axis == "announcement":
        return event.announcement_available_at
    if axis == "effective":
        return event.effective_start_at
    raise MarketClockError(f"未知时间轴：{axis}")


def lead_time_reading(event: MergedEvent) -> LeadTimeReading:
    return LeadTimeReading(
        event_id=event.event_id,
        announcement_available_at=event.announcement_available_at,
        effective_start_at=event.effective_start_at,
        lead_time_hours=event.lead_time_hours,
    )


def lead_time_table(events: tuple[MergedEvent, ...]) -> tuple[LeadTimeReading, ...]:
    return tuple(lead_time_reading(event) for event in events)


def is_usable_at(event: MergedEvent, decision_time: datetime) -> bool:
    """Whether an event may inform a decision taken at `decision_time`, with no exceptions."""

    if decision_time.tzinfo is None or decision_time.utcoffset() is None:
        raise MarketClockError("决策时间必须带时区")
    return event.announcement_available_at <= decision_time.astimezone(UTC)
