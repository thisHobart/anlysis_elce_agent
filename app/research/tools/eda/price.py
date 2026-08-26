"""Describe electricity-price distribution, seasonality, volatility, and extremes."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.research.tools.eda.support import base_metadata
from app.research.tools.eda.support import number as _number


def _group_profile(values: pd.Series, grouper: pd.Index, labels: dict[int, str] | None = None) -> list[dict[str, Any]]:
    frame = pd.DataFrame({"value": values, "group": grouper}).dropna(subset=["value"])
    result: list[dict[str, Any]] = []
    for group, subset in frame.groupby("group", observed=True):
        group_key = int(group)
        result.append(
            {
                "group": group_key,
                "label": labels.get(group_key, str(group_key)) if labels else str(group_key),
                "observations": len(subset),
                "mean": _number(subset["value"].mean()),
                "median": _number(subset["value"].median()),
                "std": _number(subset["value"].std()),
            }
        )
    return result


def _autocorrelation(values: pd.Series, max_lag: int) -> list[dict[str, Any]]:
    clean_count = int(values.notna().sum())
    maximum = min(max_lag, max(clean_count - 2, 0))
    return [{"lag": lag, "correlation": _number(values.autocorr(lag=lag))} for lag in range(1, maximum + 1)]


def _rolling_profiles(values: pd.Series, frequency: str) -> dict[str, Any]:
    offset = pd.tseries.frequencies.to_offset(frequency)
    intervals_per_day = max(1, round(86_400 / (offset.nanos / 1_000_000_000)))
    profiles: dict[str, Any] = {}
    for label, window in {"one_day": intervals_per_day, "seven_days": intervals_per_day * 7}.items():
        minimum = max(2, window // 2)
        rolling_mean = values.rolling(window=window, min_periods=minimum).mean()
        rolling_std = values.rolling(window=window, min_periods=minimum).std()
        profiles[label] = {
            "window_intervals": window,
            "latest_mean": _number(rolling_mean.dropna().iloc[-1]) if rolling_mean.notna().any() else None,
            "latest_std": _number(rolling_std.dropna().iloc[-1]) if rolling_std.notna().any() else None,
            "minimum_mean": _number(rolling_mean.min()),
            "maximum_mean": _number(rolling_mean.max()),
            "maximum_std": _number(rolling_std.max()),
        }
    return profiles


def analyze_price(
    values: pd.Series,
    *,
    unit: str,
    frequency: str,
    max_lag: int,
    spike_iqr_multiplier: float,
    methods: set[str] | None = None,
) -> dict[str, Any]:
    """Return JSON-ready descriptive evidence without making causal claims."""

    if not isinstance(values.index, pd.DatetimeIndex):
        raise TypeError("price series must use a DatetimeIndex")
    clean = values.dropna().astype(float)
    if clean.empty:
        raise ValueError("price EDA requires at least one numeric observation")

    allowed_methods = {"distribution", "volatility", "extremes", "autocorrelation", "seasonality"}
    selected_methods = allowed_methods if methods is None else set(methods)
    unknown_methods = selected_methods.difference(allowed_methods)
    if unknown_methods:
        raise ValueError(f"unknown price methods: {', '.join(sorted(unknown_methods))}")
    if not selected_methods:
        raise ValueError("price analysis requires at least one method")

    quantiles = clean.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    q1 = float(quantiles.loc[0.25])
    q3 = float(quantiles.loc[0.75])
    iqr = q3 - q1
    upper_threshold = q3 + spike_iqr_multiplier * iqr
    lower_threshold = q1 - spike_iqr_multiplier * iqr
    high_spikes = clean.loc[clean > upper_threshold]
    extreme_lows = clean.loc[clean < lower_threshold]

    positions = np.arange(len(values), dtype=float)[values.notna().to_numpy()]
    trend_slope = float(np.polyfit(positions, clean.to_numpy(), 1)[0]) if len(clean) >= 2 else None
    changes = values.astype(float).diff().dropna()
    weekday_labels = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}

    def extreme_examples(series: pd.Series, *, ascending: bool) -> list[dict[str, Any]]:
        ordered = series.sort_values(ascending=ascending).head(10)
        return [{"timestamp": timestamp.isoformat(), "value": _number(value)} for timestamp, value in ordered.items()]

    result: dict[str, Any] = {
        "methods": sorted(selected_methods),
        **base_metadata(values, unit=unit),
    }
    if "distribution" in selected_methods:
        result["distribution"] = {
            "mean": _number(clean.mean()),
            "median": _number(clean.median()),
            "std": _number(clean.std()),
            "min": _number(clean.min()),
            "max": _number(clean.max()),
            "skewness": _number(clean.skew()),
            "kurtosis": _number(clean.kurt()),
            "quantiles": {str(key): _number(value) for key, value in quantiles.items()},
        }
    if "extremes" in selected_methods:
        result["signs"] = {
            "negative_count": int((clean < 0).sum()),
            "negative_rate": round(float((clean < 0).mean()), 6),
            "zero_count": int((clean == 0).sum()),
            "zero_rate": round(float((clean == 0).mean()), 6),
        }
        result["extremes"] = {
            "method": "Tukey outer fence",
            "iqr_multiplier": spike_iqr_multiplier,
            "upper_threshold": _number(upper_threshold),
            "lower_threshold": _number(lower_threshold),
            "high_spike_count": len(high_spikes),
            "high_spike_rate": round(len(high_spikes) / len(clean), 6),
            "extreme_low_count": len(extreme_lows),
            "extreme_low_rate": round(len(extreme_lows) / len(clean), 6),
            "highest_examples": extreme_examples(high_spikes, ascending=False),
            "lowest_examples": extreme_examples(extreme_lows, ascending=True),
        }
    if "volatility" in selected_methods:
        result["volatility"] = {
            "trend_slope_per_interval": _number(trend_slope),
            "first_difference_std": _number(changes.std()),
            "mean_absolute_change": _number(changes.abs().mean()),
        }
        result["rolling_statistics"] = _rolling_profiles(values, frequency)
    if "autocorrelation" in selected_methods:
        result["autocorrelation"] = _autocorrelation(values, max_lag)
    if "seasonality" in selected_methods:
        result["seasonality"] = {
            "hour_of_day": _group_profile(values, values.index.hour),
            "day_of_week": _group_profile(values, values.index.dayofweek, weekday_labels),
            "day_type": _group_profile(
                values, pd.Index(np.where(values.index.dayofweek < 5, 0, 1)), {0: "Weekday", 1: "Weekend"}
            ),
            "month": _group_profile(values, values.index.month),
        }
    return result
