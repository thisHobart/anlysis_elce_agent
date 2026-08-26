"""Target-series structure diagnostics: stationarity, seasonality, memory, regimes."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import MSTL, STL
from statsmodels.tsa.stattools import adfuller, kpss, pacf

from app.research.tools.eda.support import (
    base_metadata,
    intervals_per_day,
    number,
    require_datetime_index,
    robust_scale,
    seasonal_periods,
    share,
)

MIN_UNIT_ROOT_OBSERVATIONS = 32
MIN_DECOMPOSITION_CYCLES = 2
MAX_TEST_POINTS = 8_760
MAX_DECOMPOSITION_POINTS = 20_000
RESAMPLE_LADDER = ("1h", "2h", "3h", "6h", "12h", "1D")


def _clean(values: pd.Series, *, what: str) -> pd.Series:
    require_datetime_index(values, what=what)
    series = values.astype(float)
    if series.dropna().empty:
        raise ValueError(f"{what} requires at least one numeric observation")
    return series


def reduce_resolution(values: pd.Series, *, frequency: str, max_points: int) -> tuple[pd.Series, str, bool]:
    """Aggregate a long series onto a coarser grid so tests stay fast and interpretable."""

    if len(values) <= max_points:
        return values, frequency, False
    for rule in RESAMPLE_LADDER:
        reduced = values.resample(rule).mean()
        if len(reduced) <= max_points:
            return reduced, rule, True
    return values.resample("1D").mean(), "1D", True


def fill_for_spectral_methods(values: pd.Series) -> tuple[pd.Series, float]:
    """Interpolate only for methods that cannot accept gaps, and report how much was filled."""

    missing = int(values.isna().sum())
    filled = values.interpolate(method="time", limit_direction="both").dropna()
    return filled, share(missing, len(values))


def augmented_dickey_fuller(series: pd.Series, *, max_lag: int) -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        statistic, p_value, used_lag, observations, critical, _ = adfuller(
            series.to_numpy(),
            maxlag=max_lag,
            regression="c",
            autolag="AIC",
        )
    return {
        "test": "Augmented Dickey-Fuller",
        "null_hypothesis": "序列存在单位根（非平稳）",
        "statistic": number(statistic),
        "p_value": number(p_value),
        "used_lag": int(used_lag),
        "observations": int(observations),
        "critical_values": {key: number(value) for key, value in critical.items()},
        "rejects_null_at_5_percent": bool(p_value < 0.05),
    }


def kpss_test(series: pd.Series) -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        statistic, p_value, lags, critical = kpss(series.to_numpy(), regression="c", nlags="auto")
    return {
        "test": "KPSS",
        "null_hypothesis": "序列围绕常数水平平稳",
        "statistic": number(statistic),
        "p_value": number(p_value),
        "p_value_at_table_bound": bool(p_value in (0.01, 0.1)),
        "used_lag": int(lags),
        "critical_values": {key: number(value) for key, value in critical.items()},
        "rejects_null_at_5_percent": bool(p_value < 0.05),
    }


VERDICT_TEXT = {
    "stationary": "两项检验一致支持围绕常数水平的平稳性。",
    "unit_root": "两项检验一致提示存在单位根，需要差分或去趋势。",
    "trend_or_break_suspected": "两项检验结论冲突，通常对应趋势项或结构突变。",
    "inconclusive": "两项检验均未给出明确结论，样本可能偏短或噪声偏大。",
}


def stationarity_verdict(adf: dict[str, Any], kpss_result: dict[str, Any]) -> str:
    adf_stationary = bool(adf["rejects_null_at_5_percent"])
    kpss_nonstationary = bool(kpss_result["rejects_null_at_5_percent"])
    if adf_stationary and not kpss_nonstationary:
        return "stationary"
    if not adf_stationary and kpss_nonstationary:
        return "unit_root"
    if adf_stationary and kpss_nonstationary:
        return "trend_or_break_suspected"
    return "inconclusive"


def analyze_stationarity(values: pd.Series, *, unit: str, frequency: str) -> dict[str, Any]:
    """Run the ADF/KPSS battery on the level and the first difference of the target."""

    series = _clean(values, what="stationarity testing")
    reduced, resolution, downsampled = reduce_resolution(series, frequency=frequency, max_points=MAX_TEST_POINTS)
    tested, imputed = fill_for_spectral_methods(reduced)
    if len(tested) < MIN_UNIT_ROOT_OBSERVATIONS:
        raise ValueError(f"stationarity testing requires at least {MIN_UNIT_ROOT_OBSERVATIONS} aligned observations")
    max_lag = int(min(max(4, len(tested) // 20), intervals_per_day(resolution) * 2, 48))
    level_adf = augmented_dickey_fuller(tested, max_lag=max_lag)
    level_kpss = kpss_test(tested)
    differenced = tested.diff().dropna()
    difference_adf = augmented_dickey_fuller(differenced, max_lag=max_lag)
    difference_kpss = kpss_test(differenced)
    level_verdict = stationarity_verdict(level_adf, level_kpss)
    difference_verdict = stationarity_verdict(difference_adf, difference_kpss)
    if level_verdict == "stationary":
        transform = "none"
    elif difference_verdict == "stationary":
        transform = "first_difference"
    else:
        transform = "seasonal_or_structural_review"
    return {
        "methods": ["price_stationarity_tests"],
        **base_metadata(values, unit=unit),
        "stationarity": {
            "analysis_resolution": resolution,
            "downsampled_for_tests": downsampled,
            "tested_observations": len(tested),
            "interpolated_share_for_tests": imputed,
            "level": {"adf": level_adf, "kpss": level_kpss, "verdict": level_verdict},
            "first_difference": {"adf": difference_adf, "kpss": difference_kpss, "verdict": difference_verdict},
            "verdict": level_verdict,
            "verdict_text": VERDICT_TEXT[level_verdict],
            "recommended_transform": transform,
            "caveat": "单位根检验对结构突变敏感；结论只用于建模前的差分与去季节决策，不代表预测能力。",
        },
    }


def _component_strength(residual: np.ndarray, component: np.ndarray) -> float | None:
    denominator = float(np.var(residual + component))
    if denominator <= 0:
        return None
    return number(max(0.0, 1.0 - float(np.var(residual)) / denominator))


def _daily_shape(seasonal: pd.Series) -> dict[str, Any]:
    profile = seasonal.groupby(seasonal.index.hour).mean()
    if profile.empty:
        return {}
    return {
        "peak_hour": int(profile.idxmax()),
        "trough_hour": int(profile.idxmin()),
        "peak_to_trough": number(float(profile.max() - profile.min())),
        "hourly_profile": [{"hour": int(hour), "value": number(value)} for hour, value in profile.sort_index().items()],
    }


def analyze_seasonal_decomposition(values: pd.Series, *, unit: str, frequency: str) -> dict[str, Any]:
    """Split the target into trend, daily/weekly seasonality, and remainder."""

    series = _clean(values, what="seasonal decomposition")
    reduced, resolution, downsampled = reduce_resolution(
        series, frequency=frequency, max_points=MAX_DECOMPOSITION_POINTS
    )
    prepared, imputed = fill_for_spectral_methods(reduced)
    periods = seasonal_periods(resolution)
    usable = [
        (label, length)
        for label, length in (("day", periods["day"]), ("week", periods["week"]))
        if length >= 4 and len(prepared) >= MIN_DECOMPOSITION_CYCLES * length
    ]
    if not usable:
        raise ValueError("seasonal decomposition requires at least two complete daily cycles")
    if len(usable) > 1:
        fitted = MSTL(prepared, periods=[length for _label, length in usable]).fit()
        seasonal_frame = fitted.seasonal
        components = {label: seasonal_frame.iloc[:, index] for index, (label, _length) in enumerate(usable)}
    else:
        fitted = STL(prepared, period=usable[0][1], robust=True).fit()
        components = {usable[0][0]: pd.Series(fitted.seasonal, index=prepared.index)}
    trend = pd.Series(np.asarray(fitted.trend), index=prepared.index)
    residual = np.asarray(fitted.resid)

    variances = {
        "trend": float(np.var(trend.to_numpy())),
        **{f"seasonal_{label}": float(np.var(item.to_numpy())) for label, item in components.items()},
        "remainder": float(np.var(residual)),
    }
    total_variance = sum(variances.values()) or 1.0
    seasonal_strength = {label: _component_strength(residual, item.to_numpy()) for label, item in components.items()}
    dominant = max(seasonal_strength.items(), key=lambda item: item[1] or 0.0)[0] if seasonal_strength else None
    return {
        "methods": ["price_seasonal_decomposition"],
        **base_metadata(values, unit=unit),
        "decomposition": {
            "algorithm": "MSTL" if len(usable) > 1 else "STL",
            "analysis_resolution": resolution,
            "downsampled_for_decomposition": downsampled,
            "interpolated_share_for_decomposition": imputed,
            "periods": {label: int(length) for label, length in usable},
            "trend_strength": _component_strength(residual, trend.to_numpy()),
            "seasonal_strength": seasonal_strength,
            "dominant_seasonality": dominant,
            "variance_share": {key: number(value / total_variance) for key, value in variances.items()},
            "remainder_std": number(float(np.std(residual))),
            "daily_shape": _daily_shape(components["day"]) if "day" in components else {},
            "caveat": "分解为描述性成分拆分；季节强度高不代表外生变量没有增量价值。",
        },
    }


def analyze_partial_autocorrelation(
    values: pd.Series,
    *,
    unit: str,
    frequency: str,
    max_lag: int,
) -> dict[str, Any]:
    """Measure direct lag-by-lag memory (PACF) and residual autocorrelation (Ljung-Box)."""

    series = _clean(values, what="partial autocorrelation")
    prepared, imputed = fill_for_spectral_methods(series)
    limit = int(min(max_lag, max(1, len(prepared) // 2 - 1), 400))
    if limit < 1:
        raise ValueError("partial autocorrelation requires more aligned observations")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coefficients = pacf(prepared.to_numpy(), nlags=limit, method="ywm")
    band = float(1.96 / np.sqrt(len(prepared)))
    rows = [
        {"lag": index, "partial_correlation": number(value), "significant": bool(abs(value) > band)}
        for index, value in enumerate(coefficients)
        if index >= 1
    ]
    significant = [row for row in rows if row["significant"]]
    daily = intervals_per_day(frequency)
    candidate_lags = sorted({lag for lag in (1, 2, daily, daily * 7, limit) if 1 <= lag <= limit})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ljung = acorr_ljungbox(prepared.to_numpy(), lags=candidate_lags, return_df=True)
    return {
        "methods": ["price_partial_autocorrelation"],
        **base_metadata(values, unit=unit),
        "partial_autocorrelation": {
            "max_lag": limit,
            "interpolated_share_for_tests": imputed,
            "significance_band": number(band),
            "series": rows,
            "significant_lag_count": len(significant),
            "strongest_lags": sorted(significant, key=lambda row: abs(row["partial_correlation"] or 0), reverse=True)[
                :8
            ],
            "suggested_autoregressive_order": max((row["lag"] for row in significant), default=0),
            "ljung_box": [
                {
                    "lag": int(lag),
                    "statistic": number(row["lb_stat"]),
                    "p_value": number(row["lb_pvalue"]),
                    "rejects_independence_at_5_percent": bool(row["lb_pvalue"] < 0.05),
                }
                for lag, row in ljung.iterrows()
            ],
            "caveat": "PACF 反映线性直接记忆，是自回归基线的候选阶数依据，不代表外生变量的增量价值。",
        },
    }


def _runs(mask: np.ndarray) -> list[int]:
    """Return the length of every contiguous True run."""

    lengths: list[int] = []
    current = 0
    for flag in mask:
        if flag:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return lengths


def _episode_summary(mask: np.ndarray) -> dict[str, Any]:
    lengths = _runs(mask)
    return {
        "episode_count": len(lengths),
        "total_intervals": int(sum(lengths)),
        "mean_duration_intervals": number(float(np.mean(lengths))) if lengths else None,
        "max_duration_intervals": int(max(lengths)) if lengths else 0,
    }


def _top_groups(index: pd.DatetimeIndex, mask: np.ndarray, *, attribute: str, limit: int) -> list[dict[str, Any]]:
    if not mask.any():
        return []
    counts = pd.Series(getattr(index, attribute))[mask].value_counts().head(limit)
    total = int(mask.sum())
    return [
        {"group": int(key), "spike_count": int(count), "share_of_spikes": share(int(count), total)}
        for key, count in counts.items()
    ]


def analyze_spike_regime(values: pd.Series, *, unit: str, spike_iqr_multiplier: float) -> dict[str, Any]:
    """Describe negative, zero, and spike regimes, their clustering, and when they occur."""

    series = _clean(values, what="spike regime profiling")
    clean = series.dropna()
    q1, q3 = clean.quantile([0.25, 0.75])
    iqr = float(q3 - q1)
    upper = float(q3) + spike_iqr_multiplier * iqr
    lower = float(q1) - spike_iqr_multiplier * iqr
    high_mask = (clean > upper).to_numpy()
    low_mask = (clean < lower).to_numpy()
    total = int(clean.size)
    if total > 1 and high_mask[:-1].any():
        conditional = share(int((high_mask[1:] & high_mask[:-1]).sum()), int(high_mask[:-1].sum()))
    else:
        conditional = 0.0
    unconditional = share(int(high_mask.sum()), total)
    calm = _runs(~(high_mask | low_mask))
    return {
        "methods": ["price_spike_regime_profile"],
        **base_metadata(values, unit=unit),
        "spike_regime": {
            "method": "Tukey outer fence",
            "iqr_multiplier": spike_iqr_multiplier,
            "upper_threshold": number(upper),
            "lower_threshold": number(lower),
            "negative_share": share(int((clean < 0).sum()), total),
            "zero_share": share(int((clean == 0).sum()), total),
            "high_spike_share": unconditional,
            "low_spike_share": share(int(low_mask.sum()), total),
            "high_episodes": _episode_summary(high_mask),
            "low_episodes": _episode_summary(low_mask),
            "clustering": {
                "probability_spike_follows_spike": conditional,
                "unconditional_spike_probability": unconditional,
                "persistence_ratio": number(conditional / unconditional) if unconditional else None,
            },
            "concentration": {
                "top_hours": _top_groups(clean.index, high_mask, attribute="hour", limit=5),
                "top_months": _top_groups(clean.index, high_mask, attribute="month", limit=5),
            },
            "longest_calm_streak_intervals": int(max(calm)) if calm else 0,
            "caveat": "尖峰与负价是市场结构证据，本函数只识别和描述，不做删除或截尾。",
        },
    }


DURATION_PERCENTILES = (0, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98, 99, 100)


def analyze_duration_curve(values: pd.Series, *, unit: str) -> dict[str, Any]:
    """Build the price duration curve and the concentration of extreme intervals."""

    series = _clean(values, what="duration curve")
    clean = series.dropna()
    total = int(clean.size)
    descending = clean.sort_values(ascending=False)
    ordered = descending.to_numpy()
    points = [
        {
            "exceedance_share": round(percentile / 100, 6),
            "price": number(float(ordered[min(total - 1, max(0, round(percentile / 100 * (total - 1))))])),
        }
        for percentile in DURATION_PERCENTILES
    ]
    tail_count = max(1, round(total * 0.05))
    positive_sum = float(clean[clean > 0].sum())
    top_positive = float(descending.head(tail_count).clip(lower=0).sum())
    median = float(clean.median())
    p95 = float(clean.quantile(0.95))
    p05 = float(clean.quantile(0.05))
    return {
        "methods": ["price_duration_curve"],
        **base_metadata(values, unit=unit),
        "duration_curve": {
            "points": points,
            "share_above_zero": share(int((clean > 0).sum()), total),
            "share_below_zero": share(int((clean < 0).sum()), total),
            "share_above_mean": share(int((clean > clean.mean()).sum()), total),
            "top_5_percent_mean": number(float(descending.head(tail_count).mean())),
            "bottom_5_percent_mean": number(float(descending.tail(tail_count).mean())),
            "top_5_percent_share_of_positive_value": (
                number(top_positive / positive_sum) if positive_sum > 0 else None
            ),
            "spread_p95_minus_p05": number(p95 - p05),
            "ratio_p95_to_median": number(p95 / median) if median > 0 else None,
            "caveat": "持续曲线描述价格水平的时间分布，不包含时间顺序信息。",
        },
    }


def _error_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, Any] | None:
    paired = pd.concat([actual.rename("actual"), predicted.rename("predicted")], axis=1).dropna()
    if paired.empty:
        return None
    error = paired["actual"] - paired["predicted"]
    return {
        "observations": len(paired),
        "mae": number(float(error.abs().mean())),
        "rmse": number(float(np.sqrt((error**2).mean()))),
        "median_absolute_error": number(float(error.abs().median())),
        "bias": number(float(error.mean())),
    }


def analyze_naive_baselines(values: pd.Series, *, unit: str, frequency: str) -> dict[str, Any]:
    """Compute the naive error floor every future forecast has to beat."""

    series = _clean(values, what="naive baseline benchmarking")
    periods = seasonal_periods(frequency)
    candidates: list[tuple[str, str, int]] = [("persistence", "上一时刻价格", 1)]
    if len(series) > periods["day"] * 2:
        candidates.append(("daily_naive", "前一日同一时刻价格", periods["day"]))
    if len(series) > periods["week"] * 2:
        candidates.append(("weekly_naive", "上周同一时刻价格", periods["week"]))
    rows: list[dict[str, Any]] = []
    for name, label, lag in candidates:
        metrics = _error_metrics(series, series.shift(lag))
        if metrics is not None:
            rows.append({"baseline": name, "label": label, "lag_intervals": lag, **metrics})
    if not rows:
        raise ValueError("naive baseline benchmarking requires at least two aligned observations")
    reference = next((row for row in rows if row["baseline"] == "persistence"), rows[0])
    for row in rows:
        row["mae_ratio_to_persistence"] = number(row["mae"] / reference["mae"]) if reference["mae"] else None
    best = min(rows, key=lambda row: row["mae"] if row["mae"] is not None else float("inf"))
    clean = series.dropna()
    scale = float(clean.abs().median()) or 1.0
    near_zero_share = share(int((clean.abs() < 0.01 * scale).sum()), int(clean.size))
    return {
        "methods": ["price_naive_baseline_benchmark"],
        **base_metadata(values, unit=unit),
        "naive_baselines": {
            "baselines": rows,
            "best_baseline": best["baseline"],
            "best_baseline_label": best["label"],
            "best_mae": best["mae"],
            "best_rmse": best["rmse"],
            "near_zero_share": near_zero_share,
            "percentage_errors_reliable": bool(near_zero_share < 0.01),
            "error_floor_note": (
                f"后续模型必须在同一时间切分下优于“{best['label']}”的 MAE {best['mae']:,.2f}，"
                "否则不构成预测增益。"
            ),
            "caveat": "本基线为样本内全期计算，正式实验需改为按时间顺序滚动重估。",
        },
    }


def analyze_variance_stabilization(values: pd.Series, *, unit: str) -> dict[str, Any]:
    """Check whether a robust asinh transform would tame the target's tails."""

    series = _clean(values, what="variance stabilization check")
    clean = series.dropna().to_numpy()
    if clean.size < 8:
        raise ValueError("variance stabilization check requires at least eight observations")
    center, scale = robust_scale(clean)
    if scale <= 0:
        raise ValueError("variance stabilization check requires a non-degenerate target series")
    standardized = (clean - center) / scale
    transformed = np.arcsinh(standardized)

    def describe(sample: np.ndarray) -> dict[str, Any]:
        jarque = stats.jarque_bera(sample)
        return {
            "skewness": number(float(stats.skew(sample))),
            "excess_kurtosis": number(float(stats.kurtosis(sample, fisher=True))),
            "jarque_bera_statistic": number(float(jarque.statistic)),
            "jarque_bera_p_value": number(float(jarque.pvalue)),
            "share_beyond_5_robust_sd": share(int((np.abs(sample) > 5).sum()), int(sample.size)),
        }

    raw = describe(standardized)
    stabilized = describe(transformed)
    improved = abs(stabilized["excess_kurtosis"] or 0) < abs(raw["excess_kurtosis"] or 0) * 0.8
    return {
        "methods": ["price_variance_stabilization_check"],
        **base_metadata(values, unit=unit),
        "variance_stabilization": {
            "robust_center": number(center),
            "robust_scale": number(scale),
            "standardization": "median 与 1.4826×MAD 的稳健标准化",
            "raw_standardized": raw,
            "asinh_transformed": stabilized,
            "recommended_transform": "asinh_median_mad" if improved else "none",
            "rationale": (
                "asinh 变换明显压缩了尾部厚度，建议在后续建模阶段作为方差稳定候选。"
                if improved
                else "稳健标准化后的尾部厚度已可接受，暂不需要额外方差稳定变换。"
            ),
            "caveat": "变换建议只适用于建模预处理；报告展示的统计量仍基于原始价格。",
        },
    }
