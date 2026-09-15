"""Infer the internal study contract directly from selected data files."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from app.research.schemas.study import AnalysisSettings, SeriesSpec, StudyConfig, StudyDefinition, StudyInputDescriptor
from app.runtime_paths import default_research_output_directory


def _read_sample(path: Path) -> pd.DataFrame:
    try:
        if path.suffix.casefold() == ".csv":
            return pd.read_csv(path, nrows=2500)
        return pd.read_parquet(path).head(2500)
    except Exception as exc:
        raise ValueError(f"无法读取 {path.name}：{exc}") from exc


def _timestamp_column(frame: pd.DataFrame, filename: str) -> str:
    preferred = ("datetime", "timestamp", "time", "date", "日期", "时间")
    by_lower = {str(column).casefold(): str(column) for column in frame.columns}
    for name in preferred:
        if name.casefold() in by_lower:
            return by_lower[name.casefold()]
    for column in frame.columns:
        values = frame[column].dropna()
        if values.empty or pd.api.types.is_numeric_dtype(values):
            continue
        parsed = pd.to_datetime(values, errors="coerce")
        if float(parsed.notna().mean()) >= 0.9:
            return str(column)
    raise ValueError(f"{filename} 中无法自动识别时间列；请确认文件包含可解析的时间列")


def _numeric_columns(
    frame: pd.DataFrame,
    timestamp_column: str,
    filename: str,
    *,
    excluded: set[str] | None = None,
) -> list[str]:
    result: list[str] = []
    ignored = excluded or set()
    for column in frame.columns:
        name = str(column)
        if name == timestamp_column or name in ignored:
            continue
        source = frame[column].dropna()
        if source.empty:
            # Parquet preserves a numeric schema even when an early sample has no
            # values. Keep that declared series: a later part of a sparse outer
            # join may contain the observations (for example a newly launched
            # solar forecast).
            if pd.api.types.is_numeric_dtype(frame[column].dtype):
                result.append(name)
            continue
        converted = pd.to_numeric(source, errors="coerce")
        if float(converted.notna().mean()) >= 0.9:
            result.append(name)
    if not result:
        raise ValueError(f"{filename} 中没有可用数值列")
    return result


def _availability_column(frame: pd.DataFrame, timestamp_column: str) -> str | None:
    """Recognize common forecast publication-time fields without user configuration."""

    candidates = (
        "available_at",
        "availability_time",
        "published_at",
        "publication_time",
        "issued_at",
        "issue_time",
        "forecast_created_at",
    )
    by_lower = {str(column).casefold(): str(column) for column in frame.columns}
    return next(
        (
            by_lower[name]
            for name in candidates
            if name in by_lower and by_lower[name] != timestamp_column
        ),
        None,
    )


def _target_value_column(columns: list[str]) -> str:
    priorities = ("rt_node_price_2b", "rt_price", "real_time_price", "price", "电价")
    lowered = {column.casefold(): column for column in columns}
    for name in priorities:
        if name in lowered:
            return lowered[name]
    contains_price = [column for column in columns if "price" in column.casefold() or "电价" in column]
    return contains_price[0] if contains_price else columns[0]


def _safe_name(value: str, *, prefix: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip()).strip("_-")
    if not normalized or not normalized[0].isalpha():
        normalized = f"{prefix}_{normalized or 'value'}"
    return normalized


def _frequency(frame: pd.DataFrame, timestamp_column: str) -> str:
    timestamps = pd.to_datetime(frame[timestamp_column], errors="coerce").dropna().drop_duplicates().sort_values()
    if len(timestamps) >= 3:
        try:
            inferred = pd.infer_freq(timestamps)
        except ValueError:
            inferred = None
        if inferred:
            return inferred
    differences = timestamps.diff().dropna().dt.total_seconds()
    if differences.empty:
        return "1h"
    seconds = max(1, round(float(differences.median())))
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    if seconds % 60 == 0:
        return f"{seconds // 60}min"
    return f"{seconds}s"


def infer_study_context(
    *,
    target_path: str | Path,
    actuals_path: str | Path | None = None,
    forecasts_path: str | Path | None = None,
    output_directory: str | Path | None = None,
) -> StudyConfig:
    """Create a deterministic runtime contract from the three supported data roles."""

    target_source = Path(target_path).resolve()
    target_frame = _read_sample(target_source)
    target_timestamp = _timestamp_column(target_frame, target_source.name)
    target_columns = _numeric_columns(target_frame, target_timestamp, target_source.name)
    target_value = _target_value_column(target_columns)
    target = SeriesSpec(
        name=_safe_name(target_value, prefix="price"),
        path=target_source,
        timestamp_column=target_timestamp,
        value_column=target_value,
        unit="unknown",
        availability_type="observed_only",
    )

    exogenous: list[SeriesSpec] = []
    used_names = {target.name}
    for source_value, availability, prefix in (
        (actuals_path, "observed_only", "actual"),
        (forecasts_path, "forecast", "forecast"),
    ):
        if source_value is None:
            continue
        source_path = Path(source_value).resolve()
        frame = _read_sample(source_path)
        timestamp = _timestamp_column(frame, source_path.name)
        # Availability metadata can accompany both observations and forecasts.
        # Excluding it from numeric inference prevents Parquet datetime columns
        # from being interpreted as nanosecond-valued business variables.
        available_at = _availability_column(frame, timestamp)
        for column in _numeric_columns(
            frame,
            timestamp,
            source_path.name,
            excluded={available_at} if available_at else None,
        ):
            base_name = _safe_name(column, prefix=prefix)
            if base_name in used_names:
                base_name = _safe_name(f"{prefix}_{base_name}", prefix=prefix)
            name = base_name
            counter = 2
            while name in used_names:
                name = f"{base_name}_{counter}"
                counter += 1
            used_names.add(name)
            exogenous.append(
                SeriesSpec(
                    name=name,
                    path=source_path,
                    timestamp_column=timestamp,
                    value_column=column,
                    unit="unknown",
                    available_at_column=available_at,
                    availability_type=availability,
                )
            )

    artifact_root = (
        Path(output_directory).resolve() if output_directory is not None else default_research_output_directory()
    )
    return StudyConfig(
        study=StudyDefinition(
            name="inferred_price_study",
            market="unspecified",
            timezone="Asia/Shanghai",
            frequency=_frequency(target_frame, target_timestamp),
        ),
        target=target,
        exogenous=exogenous,
        analysis=AnalysisSettings(output_directory=artifact_root),
    )


def resolve_study_input(descriptor: StudyInputDescriptor) -> StudyConfig:
    """Read and infer a full execution contract after the dialogue route requires data."""

    config = infer_study_context(
        target_path=descriptor.target_path,
        actuals_path=descriptor.actuals_path,
        forecasts_path=descriptor.forecasts_path,
        output_directory=descriptor.output_directory,
    )
    study_updates = {
        "start_time": descriptor.start_time,
        "end_time": descriptor.end_time,
    }
    for name in ("name", "market", "timezone", "frequency"):
        descriptor_name = "study_name" if name == "name" else name
        value = getattr(descriptor, descriptor_name)
        if value is not None:
            study_updates[name] = value
    return config.model_copy(update={"study": config.study.model_copy(update=study_updates)})
