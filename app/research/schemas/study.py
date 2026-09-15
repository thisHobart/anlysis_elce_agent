"""Configuration contracts for a reproducible price/exogenous EDA study."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SeriesSpec(BaseModel):
    """Describe one timestamped numeric series and how it may be aggregated."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, pattern=r"^[A-Za-z][A-Za-z0-9_\-]*$")
    path: Path
    timestamp_column: str = Field(default="timestamp", min_length=1)
    value_column: str = Field(min_length=1)
    unit: str = Field(default="", max_length=64)
    timezone: str | None = None
    file_format: Literal["auto", "csv", "parquet"] = "auto"
    aggregation: Literal["mean", "median", "sum", "min", "max", "first", "last"] = "mean"
    available_at_column: str | None = None
    availability_type: Literal["known_at_timestamp", "forecast", "observed_only", "unknown"] = "unknown"
    csv_options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str | None) -> str | None:
        if value is None:
            return value
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"unknown timezone: {value}") from exc
        return value

    def resolved(self, base_directory: Path) -> SeriesSpec:
        path = self.path if self.path.is_absolute() else base_directory / self.path
        return self.model_copy(update={"path": path.resolve()})


class StudyDefinition(BaseModel):
    """Define market scope and the canonical analysis time axis."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=128)
    market: str = Field(min_length=1, max_length=128)
    timezone: str = "Asia/Shanghai"
    frequency: str = "1h"
    start_time: datetime | None = None
    end_time: datetime | None = None

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"unknown timezone: {value}") from exc
        return value

    @field_validator("frequency")
    @classmethod
    def validate_frequency(cls, value: str) -> str:
        try:
            offset = pd.tseries.frequencies.to_offset(value)
        except ValueError as exc:
            raise ValueError(f"invalid pandas frequency: {value}") from exc
        if offset.nanos <= 0:
            raise ValueError("frequency must be positive")
        return value

    @model_validator(mode="after")
    def validate_window(self) -> StudyDefinition:
        if self.start_time and self.end_time:
            start = pd.Timestamp(self.start_time)
            end = pd.Timestamp(self.end_time)
            if start.tzinfo is None:
                start = start.tz_localize(self.timezone)
            else:
                start = start.tz_convert(self.timezone)
            if end.tzinfo is None:
                end = end.tz_localize(self.timezone)
            else:
                end = end.tz_convert(self.timezone)
            if start > end:
                raise ValueError("start_time must not be after end_time")
        return self


class AnalysisSettings(BaseModel):
    """Control EDA sensitivity, minimum evidence, and artifact output."""

    model_config = ConfigDict(extra="forbid")

    max_lag: int = Field(default=72, ge=0, le=24 * 31)
    min_relationship_observations: int = Field(default=12, ge=3)
    min_target_coverage_rate: float = Field(default=0.5, ge=0, le=1)
    outlier_iqr_multiplier: float = Field(default=1.5, gt=0)
    spike_iqr_multiplier: float = Field(default=3.0, gt=0)
    output_directory: Path = Path("artifacts/research")

    def resolved(self, base_directory: Path) -> AnalysisSettings:
        path = self.output_directory
        if not path.is_absolute():
            path = base_directory / path
        return self.model_copy(update={"output_directory": path.resolve()})


class StudyInputDescriptor(BaseModel):
    """Lightweight, serializable input selection that never reads data contents."""

    model_config = ConfigDict(extra="forbid")

    target_path: Path
    actuals_path: Path | None = None
    forecasts_path: Path | None = None
    output_directory: Path | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    study_name: str | None = Field(default=None, min_length=1, max_length=128)
    market: str | None = Field(default=None, min_length=1, max_length=128)
    timezone: str | None = None
    frequency: str | None = None

    @field_validator("target_path", "actuals_path", "forecasts_path")
    @classmethod
    def validate_input_file(cls, value: Path | None) -> Path | None:
        if value is None:
            return None
        resolved = value.resolve()
        if not resolved.is_file():
            raise ValueError(f"文件不存在：{resolved}")
        if resolved.suffix.casefold() not in {".csv", ".parquet", ".pq"}:
            raise ValueError(f"只支持 CSV/Parquet 文件：{resolved}")
        return resolved

    @field_validator("output_directory")
    @classmethod
    def resolve_output_directory(cls, value: Path | None) -> Path | None:
        return value.resolve() if value is not None else None

    @field_validator("timezone")
    @classmethod
    def validate_optional_timezone(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"unknown timezone: {value}") from exc
        return value

    @field_validator("frequency")
    @classmethod
    def validate_optional_frequency(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            offset = pd.tseries.frequencies.to_offset(value)
        except ValueError as exc:
            raise ValueError(f"invalid pandas frequency: {value}") from exc
        if offset.nanos <= 0:
            raise ValueError("frequency must be positive")
        return value

    @model_validator(mode="after")
    def validate_window(self) -> StudyInputDescriptor:
        if self.start_time is not None and self.end_time is not None and self.start_time > self.end_time:
            raise ValueError("start_time must not be after end_time")
        return self


class StudyConfig(BaseModel):
    """Top-level contract consumed by the EDA pipeline."""

    model_config = ConfigDict(extra="forbid")

    study: StudyDefinition
    target: SeriesSpec
    exogenous: list[SeriesSpec] = Field(default_factory=list)
    analysis: AnalysisSettings = Field(default_factory=AnalysisSettings)

    @model_validator(mode="after")
    def validate_series_names(self) -> StudyConfig:
        names = [self.target.name, *(series.name for series in self.exogenous)]
        if len(names) != len(set(names)):
            raise ValueError("target and exogenous series names must be unique")
        return self

    def resolved(self, base_directory: Path) -> StudyConfig:
        return self.model_copy(
            update={
                "target": self.target.resolved(base_directory),
                "exogenous": [series.resolved(base_directory) for series in self.exogenous],
                "analysis": self.analysis.resolved(base_directory),
            }
        )


def load_study_config(path: str | Path) -> StudyConfig:
    """Load YAML configuration and resolve all relative paths beside that file."""

    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"research configuration not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise TypeError("research configuration must contain a YAML mapping")
    return StudyConfig.model_validate(raw).resolved(config_path.parent)
