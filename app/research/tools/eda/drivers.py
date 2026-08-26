"""Driver-side diagnostics: multicollinearity strength and per-variable stationarity."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.research.tools.eda.structure import (
    MAX_TEST_POINTS,
    MIN_UNIT_ROOT_OBSERVATIONS,
    VERDICT_TEXT,
    augmented_dickey_fuller,
    fill_for_spectral_methods,
    kpss_test,
    reduce_resolution,
    stationarity_verdict,
)
from app.research.tools.eda.support import intervals_per_day, number, require_datetime_index, share

MIN_VIF_OBSERVATIONS = 24
VIF_MODERATE_THRESHOLD = 5.0
VIF_SEVERE_THRESHOLD = 10.0


def _variance_inflation(design: np.ndarray, column: int) -> float | None:
    """Return one variance inflation factor computed from an intercept-augmented design."""

    target = design[:, column]
    others = np.delete(design, column, axis=1)
    others = np.column_stack([np.ones(len(others)), others])
    centered = target - target.mean()
    total = float(np.sum(centered**2))
    if total <= 0:
        return None
    coefficients, *_ = np.linalg.lstsq(others, target, rcond=None)
    residual = target - others @ coefficients
    unexplained = float(np.sum(residual**2))
    r_squared = 1.0 - unexplained / total
    if r_squared >= 1 - 1e-12:
        return None
    return float(1.0 / (1.0 - r_squared))


def analyze_variance_inflation(frame: pd.DataFrame, *, names: list[str]) -> dict[str, Any]:
    """Quantify how much each driver is explained by the remaining selected drivers."""

    require_datetime_index(frame, what="variance inflation analysis")
    if len(names) < 2:
        raise ValueError("variance inflation analysis requires at least two selected variables")
    complete = frame[names].dropna()
    rows: list[dict[str, Any]] = []
    if len(complete) >= MIN_VIF_OBSERVATIONS:
        design = complete.to_numpy(dtype=float)
        for index, name in enumerate(names):
            value = _variance_inflation(design, index)
            rows.append(
                {
                    "variable": name,
                    "variance_inflation_factor": number(value) if value is not None else None,
                    "explained_by_others_r_squared": number(1 - 1 / value) if value else None,
                    "severity": (
                        "unavailable"
                        if value is None
                        else "severe"
                        if value >= VIF_SEVERE_THRESHOLD
                        else "moderate"
                        if value >= VIF_MODERATE_THRESHOLD
                        else "acceptable"
                    ),
                }
            )
    rows.sort(key=lambda row: row["variance_inflation_factor"] or 0, reverse=True)
    severe = [row["variable"] for row in rows if row["severity"] == "severe"]
    moderate = [row["variable"] for row in rows if row["severity"] == "moderate"]
    return {
        "methods": ["exogenous_variance_inflation"],
        "multicollinearity": {
            "complete_case_observations": len(complete),
            "sufficient_observations": bool(len(complete) >= MIN_VIF_OBSERVATIONS),
            "moderate_threshold": VIF_MODERATE_THRESHOLD,
            "severe_threshold": VIF_SEVERE_THRESHOLD,
            "variables": rows,
            "severe_variables": severe,
            "moderate_variables": moderate,
            "verdict": (
                "severe_redundancy"
                if severe
                else "moderate_redundancy"
                if moderate
                else "acceptable"
                if rows
                else "insufficient_observations"
            ),
            "caveat": "VIF 反映线性冗余程度；高 VIF 表示变量之间信息重叠，不代表变量与电价无关。",
        },
    }


def analyze_driver_stationarity(frame: pd.DataFrame, *, names: list[str], frequency: str) -> dict[str, Any]:
    """Run the ADF/KPSS battery on every selected driver."""

    require_datetime_index(frame, what="driver stationarity testing")
    if not names:
        raise ValueError("driver stationarity testing requires at least one selected variable")
    series_results: dict[str, Any] = {}
    for name in names:
        values = frame[name].astype(float)
        reduced, resolution, downsampled = reduce_resolution(
            values, frequency=frequency, max_points=MAX_TEST_POINTS
        )
        tested, imputed = fill_for_spectral_methods(reduced)
        if len(tested) < MIN_UNIT_ROOT_OBSERVATIONS:
            series_results[name] = {
                "tested_observations": len(tested),
                "verdict": "insufficient_observations",
                "verdict_text": "有效样本不足，未执行单位根检验。",
            }
            continue
        max_lag = int(min(max(4, len(tested) // 20), intervals_per_day(resolution) * 2, 48))
        adf = augmented_dickey_fuller(tested, max_lag=max_lag)
        kpss_result = kpss_test(tested)
        verdict = stationarity_verdict(adf, kpss_result)
        series_results[name] = {
            "analysis_resolution": resolution,
            "downsampled_for_tests": downsampled,
            "tested_observations": len(tested),
            "interpolated_share_for_tests": imputed,
            "adf": adf,
            "kpss": kpss_result,
            "verdict": verdict,
            "verdict_text": VERDICT_TEXT[verdict],
        }
    non_stationary = [
        name
        for name, item in series_results.items()
        if item["verdict"] in {"unit_root", "trend_or_break_suspected"}
    ]
    return {
        "methods": ["exogenous_stationarity_tests"],
        "driver_stationarity": {
            "series": series_results,
            "non_stationary_variables": non_stationary,
            "non_stationary_share": share(len(non_stationary), len(series_results)),
            "caveat": "驱动与电价同时非平稳时，同期相关可能来自共同趋势而非真实关系。",
        },
    }
