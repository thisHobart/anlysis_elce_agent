"""Profile time-series quality and translate anomalies into analytical risks."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

import numpy as np
import pandas as pd

from app.research.data.alignment import AlignmentResult
from app.research.data.loader import LoadedSeries
from app.research.schemas.results import (
    AlignmentQualityReport,
    DataQualityReport,
    QualityIssue,
    SeriesQualityReport,
)
from app.research.schemas.study import StudyConfig


def _iso(value: pd.Timestamp | None) -> str | None:
    return value.isoformat() if value is not None and not pd.isna(value) else None


def _inferred_frequency(frame: pd.DataFrame) -> str | None:
    timestamps = frame["timestamp"].drop_duplicates().sort_values()
    if len(timestamps) < 2:
        return None
    seconds = timestamps.diff().dropna().dt.total_seconds().round(6)
    if seconds.empty:
        return None
    most_common_seconds = Counter(seconds.tolist()).most_common(1)[0][0]
    return str(pd.to_timedelta(most_common_seconds, unit="s"))


def _outlier_count(values: pd.Series, multiplier: float) -> int:
    clean = values.dropna()
    if len(clean) < 4:
        return 0
    q1, q3 = clean.quantile([0.25, 0.75])
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr == 0:
        return 0
    return int(((clean < q1 - multiplier * iqr) | (clean > q3 + multiplier * iqr)).sum())


def _availability_metrics(loaded: LoadedSeries) -> tuple[float | None, float | None, float | None]:
    if "available_at" not in loaded.frame:
        return None, None, None
    usable = loaded.frame.dropna(subset=["timestamp", "available_at"])
    if usable.empty:
        return None, None, None
    lag_hours = (usable["available_at"] - usable["timestamp"]).dt.total_seconds() / 3600
    return (
        round(float(lag_hours.median()), 6),
        round(float(lag_hours.max()), 6),
        round(float((lag_hours > 0).mean()), 6),
    )


def _series_report(
    loaded: LoadedSeries,
    aligned: AlignmentResult,
    config: StudyConfig,
) -> SeriesQualityReport:
    values = aligned.frame[loaded.spec.name]
    outlier_count = _outlier_count(values, config.analysis.outlier_iqr_multiplier)
    clean_count = int(values.notna().sum())
    median_lag, max_lag, after_rate = _availability_metrics(loaded)
    return SeriesQualityReport(
        name=loaded.spec.name,
        path=str(loaded.spec.path),
        unit=loaded.spec.unit,
        raw_rows=loaded.raw_rows,
        valid_timestamp_rows=len(loaded.frame),
        valid_value_rows=int(loaded.frame["value"].notna().sum()),
        invalid_timestamp_rows=loaded.invalid_timestamp_rows,
        non_numeric_rows=loaded.non_numeric_rows,
        duplicate_timestamp_rows=loaded.duplicate_timestamp_rows,
        duplicate_timestamp_keys=loaded.duplicate_timestamp_keys,
        first_timestamp=_iso(loaded.frame["timestamp"].min()),
        last_timestamp=_iso(loaded.frame["timestamp"].max()),
        inferred_frequency=_inferred_frequency(loaded.frame),
        expected_frequency=config.study.frequency,
        aligned_rows=len(values),
        aligned_non_null_rows=clean_count,
        aligned_coverage_rate=round(float(values.notna().mean()), 6),
        missing_interval_count=int(values.isna().sum()),
        outlier_count=outlier_count,
        outlier_rate=round(outlier_count / clean_count, 6) if clean_count else 0.0,
        availability_type=loaded.spec.availability_type,
        availability_timestamp_present=loaded.spec.available_at_column is not None,
        median_availability_lag_hours=median_lag,
        max_availability_lag_hours=max_lag,
        available_after_observation_rate=after_rate,
    )


def _quality_issues(
    reports: Iterable[SeriesQualityReport],
    *,
    target_name: str,
) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    for report in reports:
        is_target = report.name == target_name
        if report.aligned_non_null_rows == 0:
            issues.append(
                QualityIssue(
                    severity="critical",
                    code="series_has_no_values",
                    series=report.name,
                    message="No numeric observations remain on the canonical time grid.",
                )
            )
        elif report.aligned_coverage_rate < 0.5:
            issues.append(
                QualityIssue(
                    severity="critical" if is_target else "high",
                    code="very_low_coverage",
                    series=report.name,
                    message="Less than half of the canonical time grid has usable values.",
                    evidence={"coverage_rate": report.aligned_coverage_rate},
                )
            )
        elif report.aligned_coverage_rate < 0.9:
            issues.append(
                QualityIssue(
                    severity="high" if is_target else "medium",
                    code="low_coverage",
                    series=report.name,
                    message="Missing intervals may bias time-series and relationship analysis.",
                    evidence={"coverage_rate": report.aligned_coverage_rate},
                )
            )
        elif report.aligned_coverage_rate < 0.98:
            issues.append(
                QualityIssue(
                    severity="medium",
                    code="incomplete_coverage",
                    series=report.name,
                    message="The series has gaps on the canonical time grid.",
                    evidence={"coverage_rate": report.aligned_coverage_rate},
                )
            )
        if report.invalid_timestamp_rows:
            issues.append(
                QualityIssue(
                    severity="high",
                    code="invalid_timestamps",
                    series=report.name,
                    message="Rows with invalid timestamps were excluded before alignment.",
                    evidence={"rows": report.invalid_timestamp_rows},
                )
            )
        if report.non_numeric_rows:
            issues.append(
                QualityIssue(
                    severity="high",
                    code="non_numeric_values",
                    series=report.name,
                    message="Non-empty values could not be converted to numbers.",
                    evidence={"rows": report.non_numeric_rows},
                )
            )
        if report.duplicate_timestamp_rows:
            issues.append(
                QualityIssue(
                    severity="medium",
                    code="duplicate_timestamps",
                    series=report.name,
                    message="Duplicate timestamps were aggregated using the configured rule.",
                    evidence={
                        "affected_rows": report.duplicate_timestamp_rows,
                        "duplicate_keys": report.duplicate_timestamp_keys,
                    },
                )
            )
        if not is_target and report.availability_type == "unknown":
            issues.append(
                QualityIssue(
                    severity="medium",
                    code="availability_unknown",
                    series=report.name,
                    message="Feature availability at prediction time is unknown; relationships are descriptive only.",
                )
            )
        if not is_target and report.availability_type == "forecast" and not report.availability_timestamp_present:
            issues.append(
                QualityIssue(
                    severity="high",
                    code="forecast_vintage_unknown",
                    series=report.name,
                    message="Forecast values have no issue/availability timestamp, so their prediction-time availability is unverified.",
                )
            )
        if not is_target and report.availability_type == "observed_only":
            issues.append(
                QualityIssue(
                    severity="high",
                    code="lookahead_risk",
                    series=report.name,
                    message="Observed-only values may not be available at a future prediction origin.",
                )
            )
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    return sorted(issues, key=lambda issue: (severity_order[issue.severity], issue.series or "", issue.code))


def build_quality_report(
    target: LoadedSeries,
    exogenous: list[LoadedSeries],
    aligned: AlignmentResult,
    config: StudyConfig,
) -> DataQualityReport:
    """Build a compact profile and decide whether descriptive EDA may proceed."""

    loaded_series = [target, *exogenous]
    reports = {item.spec.name: _series_report(item, aligned, config) for item in loaded_series}
    complete_case_rows = int(aligned.frame.dropna().shape[0])
    alignment = AlignmentQualityReport(
        timezone=config.study.timezone,
        frequency=config.study.frequency,
        start_time=aligned.start_time.isoformat(),
        end_time=aligned.end_time.isoformat(),
        expected_rows=len(aligned.frame),
        complete_case_rows=complete_case_rows,
        complete_case_rate=round(complete_case_rows / len(aligned.frame), 6),
    )
    issues = _quality_issues(reports.values(), target_name=target.spec.name)
    target_report = reports[target.spec.name]
    usable = (
        target_report.aligned_non_null_rows >= config.analysis.min_relationship_observations
        and target_report.aligned_coverage_rate >= config.analysis.min_target_coverage_rate
    )
    return DataQualityReport(usable_for_eda=usable, series=reports, alignment=alignment, issues=issues)
