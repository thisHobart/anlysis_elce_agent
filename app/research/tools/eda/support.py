"""Shared numeric, timing, and metadata helpers for deterministic EDA functions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def number(value: Any, decimals: int = 6) -> float | None:
    """Round to a JSON-stable float, or return None for missing/infinite values."""

    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return round(parsed, decimals) if np.isfinite(parsed) else None


def share(count: int, total: int) -> float:
    return round(count / total, 6) if total else 0.0


def intervals_per_day(frequency: str) -> int:
    """Convert a canonical pandas frequency into whole intervals per calendar day."""

    offset = pd.tseries.frequencies.to_offset(frequency)
    seconds = offset.nanos / 1_000_000_000
    return max(1, round(86_400 / seconds))


def seasonal_periods(frequency: str) -> dict[str, int]:
    daily = intervals_per_day(frequency)
    return {"day": daily, "week": daily * 7}


def base_metadata(values: pd.Series, *, unit: str) -> dict[str, Any]:
    """Return the shared price-series header every target function reports identically."""

    clean = values.dropna().astype(float)
    return {
        "unit": unit,
        "observations": int(clean.size),
        "missing_observations": int(values.isna().sum()),
        "coverage_rate": round(float(values.notna().mean()), 6),
        "start_time": clean.index.min().isoformat() if not clean.empty else None,
        "end_time": clean.index.max().isoformat() if not clean.empty else None,
    }


def require_datetime_index(values: pd.Series | pd.DataFrame, *, what: str) -> None:
    if not isinstance(values.index, pd.DatetimeIndex):
        raise TypeError(f"{what} requires a DatetimeIndex")


def evenly_spaced_values(values: pd.Series) -> pd.Series:
    """Return the numeric series on its canonical axis with the original gaps preserved."""

    return values.astype(float)


def robust_scale(values: np.ndarray) -> tuple[float, float]:
    """Return the median and the MAD-based robust standard-deviation estimate."""

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return median, mad * 1.482602218505602
