"""Measure contemporaneous, lagged, grouped, and nonlinear price relationships."""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, 6) if np.isfinite(number) else None


def _correlation(
    target: pd.Series,
    feature: pd.Series,
    *,
    method: Literal["pearson", "spearman"],
    min_observations: int,
) -> dict[str, Any]:
    paired = pd.concat([target.rename("target"), feature.rename("feature")], axis=1).dropna()
    observations = len(paired)
    if observations < min_observations or paired["target"].nunique() < 2 or paired["feature"].nunique() < 2:
        return {"correlation": None, "p_value": None, "observations": observations}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if method == "pearson":
            result = stats.pearsonr(paired["target"], paired["feature"])
        else:
            result = stats.spearmanr(paired["target"], paired["feature"])
    return {
        "correlation": _number(result.statistic),
        "p_value": _number(result.pvalue),
        "observations": observations,
    }


def _lag_profile(
    target: pd.Series,
    feature: pd.Series,
    *,
    max_lag: int,
    min_observations: int,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    rows: list[dict[str, Any]] = []
    for lag in range(max_lag + 1):
        shifted = feature.shift(lag)
        paired = pd.concat([target, shifted], axis=1).dropna()
        correlation = None
        if len(paired) >= min_observations and paired.iloc[:, 0].nunique() >= 2 and paired.iloc[:, 1].nunique() >= 2:
            correlation = _number(paired.iloc[:, 0].corr(paired.iloc[:, 1]))
        rows.append({"lag": lag, "correlation": correlation, "observations": len(paired)})

    usable = [row for row in rows if row["correlation"] is not None]
    if not usable:
        return rows, None
    best = max(usable, key=lambda row: abs(row["correlation"]))
    significance = _correlation(
        target,
        feature.shift(best["lag"]),
        method="pearson",
        min_observations=min_observations,
    )
    return rows, {**best, "p_value": significance["p_value"]}


def _grouped_correlations(
    target: pd.Series,
    feature: pd.Series,
    *,
    group_values: pd.Index,
    min_observations: int,
) -> list[dict[str, Any]]:
    frame = pd.DataFrame({"target": target, "feature": feature, "group": group_values})
    rows: list[dict[str, Any]] = []
    for group, subset in frame.groupby("group", observed=True):
        result = _correlation(
            subset["target"],
            subset["feature"],
            method="pearson",
            min_observations=min_observations,
        )
        if result["correlation"] is not None:
            rows.append({"group": int(group), **result})
    return rows


def _quantile_response(target: pd.Series, feature: pd.Series, min_observations: int) -> list[dict[str, Any]]:
    paired = pd.concat([target.rename("target"), feature.rename("feature")], axis=1).dropna()
    if len(paired) < max(min_observations, 8) or paired["feature"].nunique() < 4:
        return []
    try:
        paired["bucket"] = pd.qcut(paired["feature"], q=4, labels=False, duplicates="drop")
    except ValueError:
        return []
    rows: list[dict[str, Any]] = []
    for bucket, subset in paired.groupby("bucket", observed=True):
        rows.append(
            {
                "bucket": int(bucket) + 1,
                "observations": len(subset),
                "feature_min": _number(subset["feature"].min()),
                "feature_max": _number(subset["feature"].max()),
                "target_mean": _number(subset["target"].mean()),
                "target_median": _number(subset["target"].median()),
            }
        )
    return rows


def analyze_relationships(
    frame: pd.DataFrame,
    *,
    target_name: str,
    exogenous_names: list[str],
    max_lag: int,
    min_observations: int,
    methods: set[str] | None = None,
) -> dict[str, Any]:
    """Analyze each feature independently using pairwise-complete observations.

    A positive lag means ``feature[t-lag]`` is compared with ``target[t]``;
    therefore the feature leads the target by that many canonical intervals.
    """

    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("relationship analysis requires a DatetimeIndex")
    allowed_methods = {
        "pearson",
        "spearman",
        "lag_scan",
        "hour_segments",
        "month_segments",
        "quantile_response",
    }
    selected_methods = allowed_methods if methods is None else set(methods)
    unknown_methods = selected_methods.difference(allowed_methods)
    if unknown_methods:
        raise ValueError(f"unknown relationship methods: {', '.join(sorted(unknown_methods))}")
    if not selected_methods:
        raise ValueError("relationship analysis requires at least one method")
    target = frame[target_name]
    relationships: dict[str, Any] = {}
    for name in exogenous_names:
        feature = frame[name]
        result: dict[str, Any] = {"lag_semantics": "positive lag means the exogenous variable leads the target"}
        contemporaneous: dict[str, Any] = {}
        if "pearson" in selected_methods:
            contemporaneous["pearson"] = _correlation(
                    target,
                    feature,
                    method="pearson",
                    min_observations=min_observations,
                )
        if "spearman" in selected_methods:
            contemporaneous["spearman"] = _correlation(
                    target,
                    feature,
                    method="spearman",
                    min_observations=min_observations,
                )
        if contemporaneous:
            result["contemporaneous"] = contemporaneous
        if "lag_scan" in selected_methods:
            lag_profile, best_lag = _lag_profile(
                target,
                feature,
                max_lag=max_lag,
                min_observations=min_observations,
            )
            result["lag_profile"] = lag_profile
            result["best_absolute_lag"] = best_lag
        if "hour_segments" in selected_methods:
            result["by_hour"] = _grouped_correlations(
                target,
                feature,
                group_values=frame.index.hour,
                min_observations=min_observations,
            )
        if "month_segments" in selected_methods:
            result["by_month"] = _grouped_correlations(
                target,
                feature,
                group_values=frame.index.month,
                min_observations=min_observations,
            )
        if "quantile_response" in selected_methods:
            result["feature_quantile_response"] = _quantile_response(target, feature, min_observations)
        relationships[name] = result
    return {
        "methods": sorted(selected_methods),
        "target": target_name,
        "pairwise_complete_analysis": True,
        "minimum_observations": min_observations,
        "correlation_is_not_causation": True,
        "series": relationships,
    }
