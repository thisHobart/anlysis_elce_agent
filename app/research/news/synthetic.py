"""Fixture-only synthetic price generator.

This module writes the answer key. Analysis code must never import it: if the statistics
could read `EffectSpec`, recovering the injected effect would prove nothing at all. A
boundary test enforces that separation.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.research.news.clock import MarketClock
from app.research.news.prices import PriceObservations

GENERATOR_VERSION = "1.0.0"


@dataclass(frozen=True)
class EffectSpec:
    """One deliberately injected price effect, expressed in absolute price units.

    Absolute, never multiplicative: the generator is allowed to produce zero and negative
    prices, so anything log-shaped would be undefined exactly where the interesting cases are.
    """

    label: str
    start_at: datetime
    end_at: datetime
    delta: float
    volatility_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if self.end_at <= self.start_at:
            raise ValueError(f"{self.label}: 效应结束必须晚于开始")
        if self.volatility_multiplier <= 0:
            raise ValueError(f"{self.label}: 波动倍数必须为正")

    def covers(self, instant: datetime) -> bool:
        return self.start_at <= instant < self.end_at


@dataclass(frozen=True)
class SyntheticPriceConfig:
    """Shape of the baseline price path before any event effect is injected."""

    clock: MarketClock
    start_at: datetime
    weeks: int = 8
    base_level: float = 60.0
    daily_amplitude: float = 18.0
    weekly_amplitude: float = 6.0
    noise_scale: float = 4.0
    seed: int = 20260112
    effects: tuple[EffectSpec, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.weeks <= 0:
            raise ValueError("周数必须为正")
        if self.noise_scale < 0:
            raise ValueError("噪声幅度不能为负")


def _deterministic_noise(seed: int, index: int) -> float:
    """A reproducible pseudo-noise draw that does not depend on any global RNG state."""

    digest = hashlib.sha256(f"{seed}:{index}".encode()).digest()
    raw = int.from_bytes(digest[:8], "big") / float(1 << 64)
    # Box-Muller on two decorrelated uniforms keeps the tail behaviour roughly normal.
    companion = int.from_bytes(digest[8:16], "big") / float(1 << 64)
    first = min(max(raw, 1e-12), 1 - 1e-12)
    return math.sqrt(-2.0 * math.log(first)) * math.cos(2.0 * math.pi * companion)


def generate_price_series(config: SyntheticPriceConfig) -> PriceObservations:
    """Build a fixed baseline path, then add each injected effect on top of it."""

    clock = config.clock
    start = clock.floor(config.start_at)
    intervals_per_day = 1440 // clock.interval_minutes
    total = config.weeks * 7 * intervals_per_day
    timestamps = tuple(start + index * clock.interval for index in range(total))

    prices: list[float] = []
    for index, moment in enumerate(timestamps):
        local = clock.local(moment)
        minutes_of_day = local.hour * 60 + local.minute
        daily = config.daily_amplitude * math.sin(2 * math.pi * (minutes_of_day / 1440.0) - math.pi / 2)
        weekly = config.weekly_amplitude * math.cos(2 * math.pi * (local.weekday() / 7.0))
        noise = config.noise_scale * _deterministic_noise(config.seed, index)

        level = config.base_level + daily + weekly
        volatility = 1.0
        for effect in config.effects:
            if effect.covers(moment):
                level += effect.delta
                volatility *= effect.volatility_multiplier
        prices.append(round(level + noise * volatility, 4))

    return PriceObservations(
        clock=clock,
        timestamps=timestamps,
        prices=tuple(prices),
        provenance={
            "generator_version": GENERATOR_VERSION,
            "seed": str(config.seed),
            "weeks": str(config.weeks),
            "effect_labels": ",".join(effect.label for effect in config.effects),
        },
    )


def effect_specs_from_events(
    windows: Sequence[tuple[str, datetime, datetime, float, float]],
) -> tuple[EffectSpec, ...]:
    """Convenience builder so a fixture manifest can list windows as plain tuples."""

    return tuple(
        EffectSpec(
            label=label,
            start_at=start.astimezone(UTC),
            end_at=end.astimezone(UTC),
            delta=delta,
            volatility_multiplier=volatility,
        )
        for label, start, end, delta, volatility in windows
    )


def config_from_manifest(manifest: Mapping[str, Any]) -> SyntheticPriceConfig:
    """Rebuild the exact generator configuration from the committed fixture manifest."""

    clock_spec = manifest["clock"]
    baseline = manifest["baseline"]
    clock = MarketClock(
        market=clock_spec["market"],
        timezone=clock_spec["timezone"],
        interval_minutes=int(clock_spec["interval_minutes"]),
    )
    effects = tuple(
        EffectSpec(
            label=str(item["label"]),
            start_at=_parse_instant(item["start_at"]),
            end_at=_parse_instant(item["end_at"]),
            delta=float(item["delta"]),
            volatility_multiplier=float(item.get("volatility_multiplier", 1.0)),
        )
        for item in manifest.get("effects", ())
    )
    return SyntheticPriceConfig(
        clock=clock,
        start_at=_parse_instant(baseline["start_at"]),
        weeks=int(baseline["weeks"]),
        base_level=float(baseline["base_level"]),
        daily_amplitude=float(baseline["daily_amplitude"]),
        weekly_amplitude=float(baseline["weekly_amplitude"]),
        noise_scale=float(baseline["noise_scale"]),
        seed=int(baseline["seed"]),
        effects=effects,
    )


def _parse_instant(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        raise ValueError(f"manifest 时间必须带时区：{value}")
    return parsed.astimezone(UTC)
