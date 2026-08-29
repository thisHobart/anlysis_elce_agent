"""Deterministic event-window price analysis.

Everything here is pre-registered: the windows, the baseline definition, the control pool,
the placebo offsets, the permutation count and the correction method are fixed before any
result is seen. The module deliberately cannot see how the fixture prices were built.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

import numpy as np

from app.research.news.clock import MarketClock, TimeAxis, zero_point
from app.research.news.contracts import MergedEvent
from app.research.news.prices import PriceObservations

ANALYSIS_VERSION = "1.0.0"

Conclusion = Literal[
    "association_consistent_with_expected_direction",
    "not_supported_by_current_data",
    "insufficient_sample",
    "contradicts_expected_direction",
]

DEFAULT_WINDOW_HOURS: tuple[float, ...] = (1.0, 6.0, 24.0)
DEFAULT_PLACEBO_OFFSET_DAYS: tuple[int, ...] = (-14, -7, 7, 14)
DEFAULT_PERMUTATION_SAMPLES = 400
DEFAULT_SIGNIFICANCE = 0.05
MINIMUM_WINDOW_INTERVALS = 2


class EventAnalysisError(ValueError):
    """Raised when an analysis cannot be run under its own pre-registered rules."""


@dataclass(frozen=True)
class WindowSpec:
    """One pre-registered observation window measured forward from the zero point."""

    hours: float

    def __post_init__(self) -> None:
        if self.hours <= 0:
            raise EventAnalysisError("窗口长度必须为正")

    @property
    def label(self) -> str:
        return f"[0,{self.hours:g}h]"

    @property
    def duration(self) -> timedelta:
        return timedelta(hours=self.hours)


@dataclass(frozen=True)
class WindowMetrics:
    """What the prices did inside one window, in absolute units only."""

    window_label: str
    start_at: datetime
    end_at: datetime
    interval_count: int
    mean_price: float
    baseline_price: float
    deviation: float
    absolute_deviation: float
    volatility: float
    spike_rate: float


@dataclass(frozen=True)
class EventWindowResult:
    """One (event, window) test with its controls, placebos and corrected significance."""

    event_id: str
    event_type: str
    axis: TimeAxis
    zero_point_at: datetime
    expected_direction: str
    metrics: WindowMetrics
    control_sample_size: int
    control_deviation_mean: float
    control_deviation_stdev: float
    placebo_deviations: tuple[float, ...]
    permutation_samples: int
    permutation_p_value: float
    corrected_p_value: float
    test_sidedness: Literal["one_sided", "two_sided"]
    conclusion: Conclusion
    conclusion_reason: str


@dataclass(frozen=True)
class AnalysisMethod:
    """The pre-registered rules; changing any of these changes the result fingerprint."""

    analysis_version: str = ANALYSIS_VERSION
    window_hours: tuple[float, ...] = DEFAULT_WINDOW_HOURS
    placebo_offset_days: tuple[int, ...] = DEFAULT_PLACEBO_OFFSET_DAYS
    permutation_samples: int = DEFAULT_PERMUTATION_SAMPLES
    significance: float = DEFAULT_SIGNIFICANCE
    seed: int = 20260829
    correction: Literal["holm"] = "holm"
    baseline: Literal["same_hour_of_week_event_free"] = "same_hour_of_week_event_free"
    spike_quantile: float = 0.95

    def as_dict(self) -> dict[str, object]:
        return {
            "analysis_version": self.analysis_version,
            "baseline": self.baseline,
            "correction": self.correction,
            "permutation_samples": self.permutation_samples,
            "placebo_offset_days": list(self.placebo_offset_days),
            "seed": self.seed,
            "significance": self.significance,
            "spike_quantile": self.spike_quantile,
            "window_hours": list(self.window_hours),
        }


@dataclass(frozen=True)
class AnalysisResult:
    """The full machine-readable outcome plus the fingerprint that makes it reproducible."""

    axis: TimeAxis
    method: AnalysisMethod
    clock: MarketClock
    results: tuple[EventWindowResult, ...]
    excluded_events: tuple[tuple[str, str], ...] = ()
    price_hash: str = ""
    event_hash: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)

    def for_event(self, event_id: str) -> tuple[EventWindowResult, ...]:
        return tuple(result for result in self.results if result.event_id == event_id)

    def content_hash(self) -> str:
        payload = json.dumps(
            {
                "axis": self.axis,
                "event_hash": self.event_hash,
                "method": self.method.as_dict(),
                "price_hash": self.price_hash,
                "results": [
                    {
                        "conclusion": result.conclusion,
                        "corrected_p_value": round(result.corrected_p_value, 6),
                        "deviation": round(result.metrics.deviation, 6),
                        "event_id": result.event_id,
                        "permutation_p_value": round(result.permutation_p_value, 6),
                        "window": result.metrics.window_label,
                    }
                    for result in self.results
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _hour_of_week(clock: MarketClock, moment: datetime) -> int:
    local = clock.local(moment)
    return local.weekday() * 24 + local.hour


def _event_occupied_intervals(
    prices: PriceObservations,
    events: Sequence[MergedEvent],
    axis: TimeAxis,
    max_window: timedelta,
) -> set[datetime]:
    """Intervals any event could plausibly touch; these can never serve as a control."""

    occupied: set[datetime] = set()
    clock = prices.clock
    for event in events:
        spans: list[tuple[datetime, datetime]] = []
        anchor = zero_point(event, axis)
        if anchor is not None:
            spans.append((anchor, anchor + max_window))
        if event.effective_start_at is not None:
            end = event.effective_end_at or (event.effective_start_at + max_window)
            spans.append((event.effective_start_at, end + max_window))
        spans.append((event.announcement_available_at, event.announcement_available_at + max_window))
        for start, end in spans:
            moment = clock.floor(start)
            limit = clock.ceil(end)
            while moment < limit:
                occupied.add(moment)
                moment += clock.interval
    return occupied


@dataclass(frozen=True)
class _Baseline:
    """Event-free reference level per hour-of-week, plus the spike threshold."""

    by_hour_of_week: dict[int, float]
    overall: float
    spike_threshold: float
    free_indices: tuple[int, ...]

    def expected(self, clock: MarketClock, moments: Sequence[datetime]) -> float:
        if not moments:
            return self.overall
        values = [self.by_hour_of_week.get(_hour_of_week(clock, moment), self.overall) for moment in moments]
        return float(statistics.fmean(values))


def _build_baseline(
    prices: PriceObservations, occupied: set[datetime], method: AnalysisMethod
) -> _Baseline:
    clock = prices.clock
    buckets: dict[int, list[float]] = {}
    free_values: list[float] = []
    free_indices: list[int] = []
    for index, (moment, price) in enumerate(zip(prices.timestamps, prices.prices, strict=True)):
        if moment in occupied:
            continue
        buckets.setdefault(_hour_of_week(clock, moment), []).append(price)
        free_values.append(price)
        free_indices.append(index)
    if not free_values:
        raise EventAnalysisError("没有任何无事件区间可用于构造基线")
    return _Baseline(
        by_hour_of_week={key: float(statistics.fmean(values)) for key, values in buckets.items()},
        overall=float(statistics.fmean(free_values)),
        spike_threshold=float(np.quantile(np.asarray(free_values, dtype=float), method.spike_quantile)),
        free_indices=tuple(free_indices),
    )


def _window_metrics(
    prices: PriceObservations,
    baseline: _Baseline,
    window: WindowSpec,
    start_at: datetime,
) -> WindowMetrics | None:
    end_at = start_at + window.duration
    moments, values = prices.slice_window(start_at, end_at)
    if len(values) < MINIMUM_WINDOW_INTERVALS:
        return None
    mean_price = float(statistics.fmean(values))
    expected = baseline.expected(prices.clock, moments)
    differences = [later - earlier for earlier, later in itertools.pairwise(values)]
    return WindowMetrics(
        window_label=window.label,
        start_at=start_at,
        end_at=end_at,
        interval_count=len(values),
        mean_price=round(mean_price, 6),
        baseline_price=round(expected, 6),
        deviation=round(mean_price - expected, 6),
        absolute_deviation=round(abs(mean_price - expected), 6),
        volatility=round(float(statistics.pstdev(differences)) if len(differences) > 1 else 0.0, 6),
        spike_rate=round(sum(1 for value in values if value > baseline.spike_threshold) / len(values), 6),
    )


def _control_deviations(
    prices: PriceObservations,
    baseline: _Baseline,
    window: WindowSpec,
    occupied: set[datetime],
) -> tuple[float, ...]:
    """Same clock position on every event-free day: what "no event" normally looks like."""

    clock = prices.clock
    deviations: list[float] = []
    for index in baseline.free_indices:
        start_at = prices.timestamps[index]
        end_at = start_at + window.duration
        moment, limit = start_at, clock.ceil(end_at)
        if any(step in occupied for step in _iterate(moment, limit, clock.interval)):
            continue
        metrics = _window_metrics(prices, baseline, window, start_at)
        if metrics is not None:
            deviations.append(metrics.deviation)
    return tuple(deviations)


def _iterate(start: datetime, end: datetime, step: timedelta):
    moment = start
    while moment < end:
        yield moment
        moment += step


def _permutation_p_value(
    observed: float,
    null_deviations: Sequence[float],
    *,
    expected_direction: str,
    samples: int,
    seed: int,
    event_id: str,
    window_label: str,
) -> tuple[float, int, Literal["one_sided", "two_sided"]]:
    """Empirical p-value against event-free windows; the seed is part of the fingerprint."""

    if not null_deviations:
        return 1.0, 0, "two_sided"
    # Python's builtin hash() is salted per process, so a stable digest is required for the
    # result fingerprint to survive a restart.
    digest = hashlib.sha256(f"{seed}:{event_id}:{window_label}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:4], "big"))
    pool = np.asarray(null_deviations, dtype=float)
    draw_count = min(samples, pool.size)
    draws = rng.choice(pool, size=draw_count, replace=pool.size < samples)

    if expected_direction == "up":
        extreme = int(np.sum(draws >= observed))
        sidedness: Literal["one_sided", "two_sided"] = "one_sided"
    elif expected_direction == "down":
        extreme = int(np.sum(draws <= observed))
        sidedness = "one_sided"
    else:
        extreme = int(np.sum(np.abs(draws) >= abs(observed)))
        sidedness = "two_sided"
    return (extreme + 1) / (draw_count + 1), int(draw_count), sidedness


def _holm_correction(p_values: Sequence[float]) -> tuple[float, ...]:
    """Holm-Bonferroni: control the family-wise error across every test we actually ran."""

    if not p_values:
        return ()
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    total = len(p_values)
    corrected = [0.0] * total
    running = 0.0
    for rank, index in enumerate(order):
        adjusted = min(1.0, (total - rank) * p_values[index])
        running = max(running, adjusted)
        corrected[index] = round(running, 6)
    return tuple(corrected)


def _classify(
    *,
    metrics: WindowMetrics,
    expected_direction: str,
    corrected_p_value: float,
    control_sample_size: int,
    placebo_deviations: Sequence[float],
    method: AnalysisMethod,
) -> tuple[Conclusion, str]:
    """Three outcome classes only; "not significant" never becomes "no effect"."""

    if control_sample_size < 10 or metrics.interval_count < MINIMUM_WINDOW_INTERVALS:
        return (
            "insufficient_sample",
            (
                f"对照窗口仅 {control_sample_size} 个、事件窗口 {metrics.interval_count} 个区间，"
                "样本量不足以支持判断。"
            ),
        )

    has_expectation = expected_direction in {"up", "down"}
    matches_direction = has_expectation and (
        (expected_direction == "up" and metrics.deviation > 0)
        or (expected_direction == "down" and metrics.deviation < 0)
    )
    opposes_direction = has_expectation and not matches_direction and metrics.deviation != 0
    significant = corrected_p_value <= method.significance
    placebo_exceeds = sum(1 for value in placebo_deviations if abs(value) >= abs(metrics.deviation))

    # An insignificant result is never promoted to a finding, whichever way it happens to
    # point. Direction and placebo stability only decide how a SIGNIFICANT result is read.
    if not significant:
        return (
            "not_supported_by_current_data",
            (
                f"事件窗偏离基线 {metrics.deviation:+.2f}，校正后 p={corrected_p_value:.4f} 未达到 "
                f"{method.significance:g} 阈值，当前数据未提供支持；这不等于证明没有影响。"
            ),
        )
    if opposes_direction:
        return (
            "contradicts_expected_direction",
            f"事件窗偏离基线 {metrics.deviation:+.2f}，方向与事件语义预期相反，校正后 p={corrected_p_value:.4f}。",
        )
    if matches_direction and placebo_exceeds:
        return (
            "contradicts_expected_direction",
            (
                f"事件窗偏离基线 {metrics.deviation:+.2f}，但 {placebo_exceeds}/{len(placebo_deviations)} "
                "个安慰剂窗口达到了同等幅度，效应不稳定。"
            ),
        )
    if matches_direction:
        return (
            "association_consistent_with_expected_direction",
            (
                f"事件窗偏离基线 {metrics.deviation:+.2f}，校正后 p={corrected_p_value:.4f}，"
                f"{len(placebo_deviations)} 个安慰剂窗口均未达到同等幅度。"
            ),
        )
    return (
        "not_supported_by_current_data",
        (
            f"事件窗偏离基线 {metrics.deviation:+.2f}（校正后 p={corrected_p_value:.4f}），"
            "但事件语义没有给出方向预期，因此不作方向性结论。"
        ),
    )


class EventPriceAnalyzer:
    """Run the pre-registered event-window analysis for one time axis."""

    def __init__(self, method: AnalysisMethod | None = None) -> None:
        self.method = method or AnalysisMethod()

    def analyze(
        self,
        prices: PriceObservations,
        events: Sequence[MergedEvent],
        *,
        axis: TimeAxis = "announcement",
    ) -> AnalysisResult:
        method = self.method
        windows = tuple(WindowSpec(hours) for hours in method.window_hours)
        max_window = max(window.duration for window in windows)
        occupied = _event_occupied_intervals(prices, events, axis, max_window)
        baseline = _build_baseline(prices, occupied, method)

        raw: list[tuple[EventWindowResult, float]] = []
        excluded: list[tuple[str, str]] = []

        # The control pool depends only on the window and the event-free mask, so it is
        # built once per window instead of once per (event, window) pair.
        controls_by_window = {
            window.label: _control_deviations(prices, baseline, window, occupied) for window in windows
        }

        for event in sorted(events, key=lambda item: (item.announcement_available_at, item.event_id)):
            anchor = zero_point(event, axis)
            if anchor is None:
                excluded.append((event.event_id, f"事件在 {axis} 时间轴上没有零点"))
                continue
            if prices.index_of(anchor) is None:
                excluded.append((event.event_id, "事件零点落在价格序列覆盖范围之外"))
                continue

            for window in windows:
                controls = controls_by_window[window.label]
                # Ceil, never floor: a window that began before the announcement would be
                # reading prices the news could not yet have influenced.
                metrics = _window_metrics(prices, baseline, window, prices.clock.ceil(anchor))
                if metrics is None:
                    excluded.append((event.event_id, f"{window.label} 窗口内价格区间不足"))
                    continue
                placebos = self._placebo_deviations(prices, baseline, window, anchor)
                p_value, samples, sidedness = _permutation_p_value(
                    metrics.deviation,
                    controls,
                    expected_direction=event.direction,
                    samples=method.permutation_samples,
                    seed=method.seed,
                    event_id=event.event_id,
                    window_label=window.label,
                )
                raw.append(
                    (
                        EventWindowResult(
                            event_id=event.event_id,
                            event_type=event.event_type,
                            axis=axis,
                            zero_point_at=anchor,
                            expected_direction=event.direction,
                            metrics=metrics,
                            control_sample_size=len(controls),
                            control_deviation_mean=round(
                                float(statistics.fmean(controls)) if controls else 0.0, 6
                            ),
                            control_deviation_stdev=round(
                                float(statistics.pstdev(controls)) if len(controls) > 1 else 0.0, 6
                            ),
                            placebo_deviations=placebos,
                            permutation_samples=samples,
                            permutation_p_value=round(p_value, 6),
                            corrected_p_value=round(p_value, 6),
                            test_sidedness=sidedness,
                            conclusion="not_supported_by_current_data",
                            conclusion_reason="",
                        ),
                        p_value,
                    )
                )

        corrected = _holm_correction([p_value for _, p_value in raw])
        finalized: list[EventWindowResult] = []
        for (result, _), corrected_p in zip(raw, corrected, strict=True):
            conclusion, reason = _classify(
                metrics=result.metrics,
                expected_direction=result.expected_direction,
                corrected_p_value=corrected_p,
                control_sample_size=result.control_sample_size,
                placebo_deviations=result.placebo_deviations,
                method=method,
            )
            finalized.append(
                EventWindowResult(
                    **{
                        **result.__dict__,
                        "corrected_p_value": corrected_p,
                        "conclusion": conclusion,
                        "conclusion_reason": reason,
                    }
                )
            )

        return AnalysisResult(
            axis=axis,
            method=method,
            clock=prices.clock,
            results=tuple(finalized),
            excluded_events=tuple(excluded),
            price_hash=prices.content_hash(),
            event_hash=_event_hash(events),
            notes=(
                f"零点时间轴：{axis}；窗口在事件零点之后前视，绝不回看。",
                "价格可能为零或负，因此只使用绝对差值，不使用对数收益或 MAPE。",
                f"多重比较采用 {method.correction}，同时报告原始与校正后 p 值。",
                "“未提供支持”不等于“证明没有影响”，事件关联也不等于因果效应。",
            ),
        )

    def _placebo_deviations(
        self,
        prices: PriceObservations,
        baseline: _Baseline,
        window: WindowSpec,
        anchor: datetime,
    ) -> tuple[float, ...]:
        """The same window shifted whole weeks away, where no such event happened."""

        deviations: list[float] = []
        for offset in self.method.placebo_offset_days:
            shifted = prices.clock.floor(anchor + timedelta(days=offset))
            if prices.index_of(shifted) is None:
                continue
            metrics = _window_metrics(prices, baseline, window, shifted)
            if metrics is not None:
                deviations.append(metrics.deviation)
        return tuple(deviations)


def _event_hash(events: Sequence[MergedEvent]) -> str:
    rows = sorted(
        (
            {
                "announcement_available_at": event.announcement_available_at.isoformat(),
                "direction": event.direction,
                "effective_start_at": (
                    event.effective_start_at.isoformat() if event.effective_start_at else None
                ),
                "event_id": event.event_id,
                "event_type": event.event_type,
            }
            for event in events
        ),
        key=lambda item: item["event_id"],
    )
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
