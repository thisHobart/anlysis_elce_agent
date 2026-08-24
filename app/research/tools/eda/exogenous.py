"""Profile exogenous variables and identify strong pairwise collinearity."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, 6) if np.isfinite(number) else None


def _series_summary(
    values: pd.Series,
    unit: str,
    outlier_iqr_multiplier: float,
    methods: set[str],
) -> dict[str, Any]:
    clean = values.dropna().astype(float)
    if clean.empty:
        return {
            "unit": unit,
            "observations": 0,
            "missing_observations": int(values.isna().sum()),
            "coverage_rate": 0.0,
        }
    q1, median, q3 = clean.quantile([0.25, 0.5, 0.75])
    iqr = q3 - q1
    lower = q1 - outlier_iqr_multiplier * iqr
    upper = q3 + outlier_iqr_multiplier * iqr
    outlier_count = int(((clean < lower) | (clean > upper)).sum()) if iqr else 0
    positions = np.arange(len(values), dtype=float)[values.notna().to_numpy()]
    trend_slope = float(np.polyfit(positions, clean.to_numpy(), 1)[0]) if len(clean) >= 2 else None
    result: dict[str, Any] = {
        "unit": unit,
        "observations": len(clean),
        "missing_observations": int(values.isna().sum()),
        "coverage_rate": round(float(values.notna().mean()), 6),
    }
    if "distribution" in methods:
        result.update(
            {
                "mean": _number(clean.mean()),
                "median": _number(median),
                "std": _number(clean.std()),
                "min": _number(clean.min()),
                "q1": _number(q1),
                "q3": _number(q3),
                "max": _number(clean.max()),
                "skewness": _number(clean.skew()),
                "zero_rate": round(float((clean == 0).mean()), 6),
                "negative_rate": round(float((clean < 0).mean()), 6),
            }
        )
    if "trend" in methods:
        result.update(
            {
                "trend_slope_per_interval": _number(trend_slope),
                "first_difference_std": _number(values.astype(float).diff().std()),
            }
        )
    if "outliers" in methods:
        result.update(
            {
                "outlier_count": outlier_count,
                "outlier_rate": round(outlier_count / len(clean), 6),
            }
        )
    return result


def analyze_exogenous(
    frame: pd.DataFrame,
    *,
    names: list[str],
    units: dict[str, str],
    outlier_iqr_multiplier: float,
    strong_correlation_threshold: float = 0.8,
    methods: set[str] | None = None,
) -> dict[str, Any]:
    """Describe every explanatory series and its contemporaneous collinearity."""

    allowed_methods = {"distribution", "outliers", "trend", "collinearity"}
    selected_methods = allowed_methods if methods is None else set(methods)
    unknown_methods = selected_methods.difference(allowed_methods)
    if unknown_methods:
        raise ValueError(f"unknown exogenous methods: {', '.join(sorted(unknown_methods))}")
    if not selected_methods:
        raise ValueError("exogenous analysis requires at least one method")
    summaries = {
        name: _series_summary(frame[name], units.get(name, ""), outlier_iqr_multiplier, selected_methods)
        for name in names
    }
    if not names:
        return {
            "methods": sorted(selected_methods),
            "series": summaries,
            "correlation_matrix": {},
            "strong_collinearity_pairs": [],
        }

    correlations = frame[names].corr(method="pearson", min_periods=3) if "collinearity" in selected_methods else None
    matrix = (
        {row: {column: _number(correlations.loc[row, column]) for column in names} for row in names}
        if correlations is not None
        else {}
    )
    strong_pairs: list[dict[str, Any]] = []
    if correlations is not None:
        for left_index, left in enumerate(names):
            for right in names[left_index + 1 :]:
                correlation = correlations.loc[left, right]
                if pd.notna(correlation) and abs(float(correlation)) >= strong_correlation_threshold:
                    observations = int(frame[[left, right]].dropna().shape[0])
                    strong_pairs.append(
                        {
                            "left": left,
                            "right": right,
                            "correlation": _number(correlation),
                            "observations": observations,
                        }
                    )
    strong_pairs.sort(key=lambda item: abs(item["correlation"] or 0), reverse=True)
    return {
        "methods": sorted(selected_methods),
        "series": summaries,
        "correlation_matrix": matrix,
        "strong_collinearity_threshold": strong_correlation_threshold,
        "strong_collinearity_pairs": strong_pairs,
    }
