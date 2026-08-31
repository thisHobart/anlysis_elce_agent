"""A provenance-neutral price series shared by the fixture generator and the analysis.

The analysis reads this shape and nothing else. It carries no notion of an injected
effect, which is what lets the statistics be tested against an answer key they cannot see.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from app.research.news.clock import MarketClock


class PriceSeriesError(ValueError):
    """Raised when a price series cannot be placed on the market clock safely."""


@dataclass(frozen=True)
class PriceObservations:
    """Settlement-interval prices on one market clock.

    Prices may legitimately be zero or negative, so nothing here may take a logarithm or
    compute a percentage error against them.
    """

    clock: MarketClock
    timestamps: tuple[datetime, ...]
    prices: tuple[float, ...]
    provenance: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.timestamps) != len(self.prices):
            raise PriceSeriesError("价格与时间戳长度必须一致")
        if not self.timestamps:
            raise PriceSeriesError("价格序列不能为空")
        normalized: list[datetime] = []
        for moment in self.timestamps:
            if moment.tzinfo is None or moment.utcoffset() is None:
                raise PriceSeriesError("价格时间戳必须带时区")
            normalized.append(moment.astimezone(UTC))
        normalized_timestamps = tuple(normalized)
        object.__setattr__(self, "timestamps", normalized_timestamps)

        if list(normalized_timestamps) != sorted(normalized_timestamps):
            raise PriceSeriesError("价格时间戳必须递增")
        if len(set(normalized_timestamps)) != len(normalized_timestamps):
            raise PriceSeriesError("价格时间戳不能重复")
        for moment in normalized_timestamps:
            if self.clock.floor(moment) != moment:
                raise PriceSeriesError("价格时间戳必须落在结算网格上")
        for previous, current in itertools.pairwise(normalized_timestamps):
            if current - previous != self.clock.interval:
                raise PriceSeriesError("价格时间戳必须按结算间隔连续，不能存在缺口")
        for value in self.prices:
            try:
                finite = math.isfinite(value)
            except TypeError as exc:
                raise PriceSeriesError("价格必须是有限数值") from exc
            if not finite:
                raise PriceSeriesError("价格必须是有限数值，不能包含 NaN 或 Inf")

    @property
    def start_at(self) -> datetime:
        return self.timestamps[0]

    @property
    def end_at(self) -> datetime:
        return self.timestamps[-1]

    def as_mapping(self) -> dict[datetime, float]:
        return dict(zip(self.timestamps, self.prices, strict=True))

    def index_of(self, moment: datetime) -> int | None:
        floored = self.clock.floor(moment)
        position = int((floored - self.start_at) / self.clock.interval)
        if position < 0 or position >= len(self.timestamps):
            return None
        return position if self.timestamps[position] == floored else None

    def slice_window(self, start_at: datetime, end_at: datetime) -> tuple[tuple[datetime, ...], tuple[float, ...]]:
        """Half-open [start, end): the interval an event ends on belongs to the next window."""

        selected = [
            (moment, price)
            for moment, price in zip(self.timestamps, self.prices, strict=True)
            if start_at <= moment < end_at
        ]
        if not selected:
            return (), ()
        moments, values = zip(*selected, strict=True)
        return tuple(moments), tuple(values)

    def content_hash(self) -> str:
        payload = json.dumps(
            {
                "interval_minutes": self.clock.interval_minutes,
                "market": self.clock.market,
                "prices": [round(price, 6) for price in self.prices],
                "timestamps": [moment.isoformat() for moment in self.timestamps],
                "timezone": self.clock.timezone,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_csv_rows(self) -> tuple[tuple[str, str], ...]:
        header = ("timestamp", "price")
        rows = tuple(
            (moment.isoformat().replace("+00:00", "Z"), f"{price:.4f}")
            for moment, price in zip(self.timestamps, self.prices, strict=True)
        )
        return (header, *rows)


def load_price_csv(path: Path, clock: MarketClock) -> PriceObservations:
    """Read a committed price fixture. Carries no knowledge of how it was produced."""

    if not path.is_file():
        raise PriceSeriesError(f"价格文件不存在：{path}")
    timestamps: list[datetime] = []
    prices: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = {"timestamp", "price"}.difference(reader.fieldnames or ())
        if missing:
            raise PriceSeriesError(f"价格文件缺少列：{', '.join(sorted(missing))}")
        for line_number, row in enumerate(reader, start=2):
            try:
                moment = datetime.fromisoformat(row["timestamp"])
                prices.append(float(row["price"]))
            except (TypeError, ValueError) as exc:
                raise PriceSeriesError(f"价格文件第 {line_number} 行无效：{exc}") from exc
            if moment.tzinfo is None:
                raise PriceSeriesError(f"价格文件第 {line_number} 行缺少时区")
            timestamps.append(moment.astimezone(UTC))
    return PriceObservations(
        clock=clock,
        timestamps=tuple(timestamps),
        prices=tuple(prices),
        provenance={"source": str(path.name)},
    )
