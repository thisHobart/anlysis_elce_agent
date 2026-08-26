"""Deterministic comparisons across explicitly declared time subsets."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from app.research.tools.contracts import SegmentDefinition
from app.research.tools.eda.relationships import _correlation


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, 6) if np.isfinite(number) else None


def _bound(value: Any, index: pd.DatetimeIndex) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if index.tz is None:
        return timestamp.tz_localize(None) if timestamp.tzinfo is not None else timestamp
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(index.tz)
    return timestamp.tz_convert(index.tz)


def segment_mask(index: pd.DatetimeIndex, segment: SegmentDefinition) -> np.ndarray:
    """Return a boolean mask on the canonical market-time index."""

    if segment.kind == "hours":
        return np.asarray(index.hour.isin(segment.hours or []), dtype=bool)
    if segment.kind == "months":
        return np.asarray(index.month.isin(segment.months or []), dtype=bool)
    start = _bound(segment.start_time, index)
    end = _bound(segment.end_time, index)
    return np.asarray((index >= start) & (index <= end), dtype=bool)


def _selector_payload(segment: SegmentDefinition) -> dict[str, Any]:
    return segment.model_dump(mode="json", exclude_none=True)


def _price_statistics(values: pd.Series, mask: np.ndarray, *, minimum: int) -> dict[str, Any]:
    selected = values.loc[mask]
    clean = selected.dropna().astype(float)
    quantiles = clean.quantile([0.25, 0.75]) if not clean.empty else pd.Series(dtype=float)
    return {
        "selected_rows": int(mask.sum()),
        "observations": int(clean.size),
        "coverage_rate": round(float(clean.size / mask.sum()), 6) if mask.sum() else 0.0,
        "sufficient_observations": bool(clean.size >= minimum),
        "mean": _number(clean.mean()) if not clean.empty else None,
        "median": _number(clean.median()) if not clean.empty else None,
        "std": _number(clean.std()) if not clean.empty else None,
        "minimum": _number(clean.min()) if not clean.empty else None,
        "maximum": _number(clean.max()) if not clean.empty else None,
        "q25": _number(quantiles.get(0.25)) if not clean.empty else None,
        "q75": _number(quantiles.get(0.75)) if not clean.empty else None,
    }


def compare_price_segments(
    values: pd.Series,
    *,
    comparison_id: str,
    segments: list[SegmentDefinition],
    min_observations: int,
) -> dict[str, Any]:
    """Compare price distributions for every segment in one auditable result."""

    if not isinstance(values.index, pd.DatetimeIndex):
        raise TypeError("segment comparison requires a DatetimeIndex")
    masks = {segment.segment_id: segment_mask(values.index, segment) for segment in segments}
    rows = {
        segment.segment_id: {
            "label": segment.label,
            "selector": _selector_payload(segment),
            **_price_statistics(values, masks[segment.segment_id], minimum=min_observations),
        }
        for segment in segments
    }
    contrasts: list[dict[str, Any]] = []
    for left, right in combinations(segments, 2):
        left_row = rows[left.segment_id]
        right_row = rows[right.segment_id]
        left_mean, right_mean = left_row["mean"], right_row["mean"]
        left_median, right_median = left_row["median"], right_row["median"]
        contrasts.append(
            {
                "left_segment_id": left.segment_id,
                "right_segment_id": right.segment_id,
                "overlap_rows": int((masks[left.segment_id] & masks[right.segment_id]).sum()),
                "mean_difference_left_minus_right": (
                    _number(left_mean - right_mean) if left_mean is not None and right_mean is not None else None
                ),
                "median_difference_left_minus_right": (
                    _number(left_median - right_median)
                    if left_median is not None and right_median is not None
                    else None
                ),
            }
        )
    return {
        "methods": ["price_segment_distribution_comparison"],
        "price": {
            comparison_id: {
                "comparison_id": comparison_id,
                "segments": rows,
                "contrasts": contrasts,
            }
        },
    }


def compare_relationship_segments(
    frame: pd.DataFrame,
    *,
    target_name: str,
    variables: list[str],
    comparison_id: str,
    segments: list[SegmentDefinition],
    min_observations: int,
) -> dict[str, Any]:
    """Compare contemporaneous Pearson relationships across declared segments."""

    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("segment comparison requires a DatetimeIndex")
    masks = {segment.segment_id: segment_mask(frame.index, segment) for segment in segments}
    series: dict[str, Any] = {}
    for variable in variables:
        segment_rows: dict[str, Any] = {}
        for segment in segments:
            mask = masks[segment.segment_id]
            segment_rows[segment.segment_id] = {
                "label": segment.label,
                "selector": _selector_payload(segment),
                **_correlation(
                    frame.loc[mask, target_name],
                    frame.loc[mask, variable],
                    method="pearson",
                    min_observations=min_observations,
                ),
            }
        contrasts: list[dict[str, Any]] = []
        for left, right in combinations(segments, 2):
            left_value = segment_rows[left.segment_id]["correlation"]
            right_value = segment_rows[right.segment_id]["correlation"]
            contrasts.append(
                {
                    "left_segment_id": left.segment_id,
                    "right_segment_id": right.segment_id,
                    "overlap_rows": int((masks[left.segment_id] & masks[right.segment_id]).sum()),
                    "correlation_difference_left_minus_right": (
                        _number(left_value - right_value)
                        if left_value is not None and right_value is not None
                        else None
                    ),
                }
            )
        series[variable] = {"segments": segment_rows, "contrasts": contrasts}
    return {
        "methods": ["relationship_pearson_segment_comparison"],
        "relationships": {
            comparison_id: {
                "comparison_id": comparison_id,
                "series": series,
            }
        },
    }
