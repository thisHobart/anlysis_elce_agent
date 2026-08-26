"""Nonlinear, predictive-precedence, and time-stability views of price relationships."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from app.research.tools.eda.support import intervals_per_day, number, require_datetime_index, share

MUTUAL_INFORMATION_BINS = 12
MIN_GRANGER_MULTIPLE = 20
MAX_GRANGER_LAG = 96
ROLLING_WINDOW_DAYS = 30
MIN_ROLLING_WINDOWS = 3


def _rank_bins(values: np.ndarray, bins: int) -> np.ndarray:
    """Assign equal-frequency bin labels so mutual information is scale invariant."""

    ranks = stats.rankdata(values, method="average")
    edges = np.linspace(0, len(values), bins + 1)[1:-1]
    return np.searchsorted(edges, ranks - 0.5)


def _mutual_information(left: np.ndarray, right: np.ndarray, *, bins: int) -> dict[str, Any]:
    joint = np.histogram2d(_rank_bins(left, bins), _rank_bins(right, bins), bins=bins)[0]
    total = joint.sum()
    if total <= 0:
        return {"mutual_information": None, "normalized_mutual_information": None}
    probability = joint / total
    left_marginal = probability.sum(axis=1, keepdims=True)
    right_marginal = probability.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        contribution = probability * np.log(probability / (left_marginal * right_marginal))
    information = float(np.nansum(contribution))

    def entropy(marginal: np.ndarray) -> float:
        flat = marginal.ravel()
        flat = flat[flat > 0]
        return float(-np.sum(flat * np.log(flat)))

    reference = min(entropy(left_marginal), entropy(right_marginal))
    return {
        "mutual_information": number(max(0.0, information)),
        "normalized_mutual_information": number(max(0.0, information) / reference) if reference > 0 else None,
    }


def analyze_mutual_information(
    frame: pd.DataFrame,
    *,
    target_name: str,
    variables: list[str],
    max_lag: int,
    min_observations: int,
) -> dict[str, Any]:
    """Score nonlinear dependence that a linear correlation coefficient can miss."""

    require_datetime_index(frame, what="mutual information analysis")
    target = frame[target_name]
    series: dict[str, Any] = {}
    for name in variables:
        feature = frame[name]
        rows: list[dict[str, Any]] = []
        for lag in range(max_lag + 1):
            paired = pd.concat([target, feature.shift(lag)], axis=1).dropna()
            if len(paired) < min_observations or paired.iloc[:, 1].nunique() < MUTUAL_INFORMATION_BINS:
                continue
            scores = _mutual_information(
                paired.iloc[:, 0].to_numpy(),
                paired.iloc[:, 1].to_numpy(),
                bins=MUTUAL_INFORMATION_BINS,
            )
            rows.append({"lag": lag, "observations": len(paired), **scores})
        if not rows:
            series[name] = {"evaluated_lags": 0, "note": "成对有效样本不足，未计算互信息。"}
            continue
        best = max(rows, key=lambda row: row["normalized_mutual_information"] or 0)
        contemporaneous = next((row for row in rows if row["lag"] == 0), None)

        def absolute_pearson(lag: int, series_at_hand: pd.Series = feature) -> float | None:
            paired = pd.concat([target, series_at_hand.shift(lag)], axis=1).dropna()
            if len(paired) < min_observations:
                return None
            return number(abs(float(paired.iloc[:, 0].corr(paired.iloc[:, 1]))))

        linear_at_best = absolute_pearson(int(best["lag"]))
        normalized = best.get("normalized_mutual_information")
        series[name] = {
            "evaluated_lags": len(rows),
            "contemporaneous": contemporaneous,
            "best_lag": best,
            "absolute_pearson_at_lag_0": absolute_pearson(0),
            "absolute_pearson_at_best_lag": linear_at_best,
            "nonlinearity_flagged": bool(
                normalized is not None
                and linear_at_best is not None
                and normalized >= 0.05
                and linear_at_best < 0.1
            ),
            "profile": rows,
        }
    flagged = [name for name, item in series.items() if item.get("nonlinearity_flagged")]
    return {
        "methods": ["relationship_mutual_information_scan"],
        "target": target_name,
        "mutual_information": {
            "bins": MUTUAL_INFORMATION_BINS,
            "binning": "等频秩分箱，结果对单调变换不敏感",
            "lag_semantics": "正滞后表示 feature[t-lag] 与 price[t] 比较",
            "series": series,
            "nonlinear_candidates": flagged,
            "caveat": "互信息衡量统计依赖强度，既不指示方向，也不构成因果或样本外预测增益。",
        },
    }


def _ordinary_least_squares_residual_sum(design: np.ndarray, response: np.ndarray) -> float:
    coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
    residual = response - design @ coefficients
    return float(np.sum(residual**2))


def _granger_at_lag(target: np.ndarray, feature: np.ndarray, *, lag: int) -> dict[str, Any] | None:
    rows = len(target) - lag
    if rows <= 2 * lag + 2:
        return None
    response = target[lag:]
    intercept = np.ones((rows, 1))
    target_lags = np.column_stack([target[lag - offset : len(target) - offset] for offset in range(1, lag + 1)])
    feature_lags = np.column_stack([feature[lag - offset : len(feature) - offset] for offset in range(1, lag + 1)])
    restricted = np.column_stack([intercept, target_lags])
    unrestricted = np.column_stack([restricted, feature_lags])
    restricted_error = _ordinary_least_squares_residual_sum(restricted, response)
    unrestricted_error = _ordinary_least_squares_residual_sum(unrestricted, response)
    degrees_of_freedom = rows - unrestricted.shape[1]
    if degrees_of_freedom <= 0 or unrestricted_error <= 0:
        return None
    statistic = ((restricted_error - unrestricted_error) / lag) / (unrestricted_error / degrees_of_freedom)
    if not np.isfinite(statistic) or statistic < 0:
        return None
    p_value = float(stats.f.sf(statistic, lag, degrees_of_freedom))
    return {
        "lag": lag,
        "f_statistic": number(statistic),
        "p_value": number(p_value),
        "observations": rows,
        "rejects_no_precedence_at_5_percent": bool(p_value < 0.05),
        "residual_variance_reduction": number(max(0.0, 1 - unrestricted_error / restricted_error)),
    }


def analyze_granger_precedence(
    frame: pd.DataFrame,
    *,
    target_name: str,
    variables: list[str],
    max_lag: int,
    frequency: str,
    min_observations: int,
) -> dict[str, Any]:
    """Test whether a driver's own history improves an autoregressive price fit."""

    require_datetime_index(frame, what="granger precedence analysis")
    daily = intervals_per_day(frequency)
    series: dict[str, Any] = {}
    for name in variables:
        paired = pd.concat([frame[target_name].rename("target"), frame[name].rename("feature")], axis=1).dropna()
        if len(paired) < min_observations:
            series[name] = {"tested_lags": [], "note": "成对有效样本不足，未执行前置性检验。"}
            continue
        ceiling = int(min(max_lag, MAX_GRANGER_LAG, max(1, len(paired) // MIN_GRANGER_MULTIPLE)))
        candidates = sorted({lag for lag in (1, 2, daily, ceiling) if 1 <= lag <= ceiling})
        target_values = paired["target"].to_numpy(dtype=float)
        feature_values = paired["feature"].to_numpy(dtype=float)
        rows = [
            row
            for lag in candidates
            if (row := _granger_at_lag(target_values, feature_values, lag=lag)) is not None
        ]
        if not rows:
            series[name] = {"tested_lags": [], "note": "样本长度不足以拟合受限与非受限回归。"}
            continue
        best = min(rows, key=lambda row: row["p_value"] if row["p_value"] is not None else 1.0)
        series[name] = {
            "tested_lags": rows,
            "strongest": best,
            "precedes_price": bool(best["rejects_no_precedence_at_5_percent"]),
            "max_evaluated_lag": ceiling,
        }
    preceding = [name for name, item in series.items() if item.get("precedes_price")]
    return {
        "methods": ["relationship_granger_causality_scan"],
        "target": target_name,
        "granger_precedence": {
            "series": series,
            "variables_with_precedence": preceding,
            "share_with_precedence": share(len(preceding), len(series)),
            "caveat": (
                "Granger 检验只说明变量历史在样本内改善自回归拟合，"
                "既不是物理因果，也不等于按时间顺序切分后的样本外增益。"
            ),
        },
    }


def analyze_rolling_stability(
    frame: pd.DataFrame,
    *,
    target_name: str,
    variables: list[str],
    frequency: str,
    min_observations: int,
) -> dict[str, Any]:
    """Check whether each contemporaneous relationship holds across the study window."""

    require_datetime_index(frame, what="rolling correlation stability")
    daily = intervals_per_day(frequency)
    window = int(min(max(daily, len(frame) // 6), daily * ROLLING_WINDOW_DAYS))
    if window < min_observations or len(frame) < window * MIN_ROLLING_WINDOWS:
        window = max(min_observations, len(frame) // MIN_ROLLING_WINDOWS)
    target = frame[target_name]
    series: dict[str, Any] = {}
    for name in variables:
        rolling = target.rolling(window=window, min_periods=min_observations).corr(frame[name])
        usable = rolling.dropna()
        stride = max(1, window // 2)
        sampled = usable.iloc[::stride]
        if len(sampled) < MIN_ROLLING_WINDOWS:
            series[name] = {"windows": 0, "note": "可用滚动窗口不足，未评估关系稳定性。"}
            continue
        values = sampled.to_numpy(dtype=float)
        positive = int((values > 0).sum())
        negative = int((values < 0).sum())
        series[name] = {
            "windows": len(sampled),
            "mean_correlation": number(float(values.mean())),
            "std_correlation": number(float(values.std())),
            "min_correlation": number(float(values.min())),
            "max_correlation": number(float(values.max())),
            "sign_flip_share": share(min(positive, negative), len(values)),
            "share_above_absolute_0_1": share(int((np.abs(values) >= 0.1).sum()), len(values)),
            "stability": (
                "stable"
                if float(values.std()) < 0.15 and min(positive, negative) == 0
                else "sign_unstable"
                if min(positive, negative) > 0
                else "magnitude_unstable"
            ),
            "timeline": [
                {"timestamp": timestamp.isoformat(), "correlation": number(value)}
                for timestamp, value in sampled.items()
            ],
        }
    unstable = [name for name, item in series.items() if item.get("stability") in {"sign_unstable", "magnitude_unstable"}]
    return {
        "methods": ["relationship_rolling_correlation_stability"],
        "target": target_name,
        "rolling_stability": {
            "window_intervals": window,
            "window_days": number(window / daily, 2),
            "series": series,
            "unstable_variables": unstable,
            "caveat": "滚动相关反映关系随时间的漂移；单一全样本系数可能掩盖阶段性反号。",
        },
    }
