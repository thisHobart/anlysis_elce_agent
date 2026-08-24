"""Serializable result contracts for quality checks and EDA runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class QualityIssue(BaseModel):
    """One actionable data-quality or model-readiness concern."""

    model_config = ConfigDict(extra="forbid")

    severity: Literal["critical", "high", "medium", "low"]
    code: str
    series: str | None = None
    message: str
    evidence: dict[str, Any] = Field(default_factory=dict)


class SeriesQualityReport(BaseModel):
    """Profile one source before and after canonical time alignment."""

    model_config = ConfigDict(extra="forbid")

    name: str
    path: str
    unit: str
    raw_rows: int
    valid_timestamp_rows: int
    valid_value_rows: int
    invalid_timestamp_rows: int
    non_numeric_rows: int
    duplicate_timestamp_rows: int
    duplicate_timestamp_keys: int
    first_timestamp: str | None
    last_timestamp: str | None
    inferred_frequency: str | None
    expected_frequency: str
    aligned_rows: int
    aligned_non_null_rows: int
    aligned_coverage_rate: float
    missing_interval_count: int
    outlier_count: int
    outlier_rate: float
    availability_type: str
    availability_timestamp_present: bool = False
    median_availability_lag_hours: float | None = None
    max_availability_lag_hours: float | None = None
    available_after_observation_rate: float | None = None


class AlignmentQualityReport(BaseModel):
    """Describe the common analysis window and join coverage."""

    model_config = ConfigDict(extra="forbid")

    timezone: str
    frequency: str
    start_time: str
    end_time: str
    expected_rows: int
    complete_case_rows: int
    complete_case_rate: float


class DataQualityReport(BaseModel):
    """Aggregate source profiles, alignment coverage, and prioritized issues."""

    model_config = ConfigDict(extra="forbid")

    usable_for_eda: bool
    series: dict[str, SeriesQualityReport]
    alignment: AlignmentQualityReport
    issues: list[QualityIssue] = Field(default_factory=list)


class PipelineRunResult(BaseModel):
    """Public return value for one completed EDA pipeline run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    status: Literal["completed"] = "completed"
    artifact_directory: Path
    report_path: Path
    aligned_rows: int
    quality_report: DataQualityReport
    eda_summary: dict[str, Any]
