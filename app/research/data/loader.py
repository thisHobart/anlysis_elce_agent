"""Load timestamped numeric series while preserving pre-cleaning quality evidence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from app.research.schemas.study import SeriesSpec


class ResearchDataError(ValueError):
    """Raised when a configured research dataset cannot be used safely."""


@dataclass(frozen=True)
class LoadedSeries:
    """Normalized source values plus evidence collected before aggregation."""

    spec: SeriesSpec
    frame: pd.DataFrame
    raw_rows: int
    invalid_timestamp_rows: int
    non_numeric_rows: int
    duplicate_timestamp_rows: int
    duplicate_timestamp_keys: int


def _read_frame(spec: SeriesSpec) -> pd.DataFrame:
    if not spec.path.is_file():
        raise ResearchDataError(f"data file not found for {spec.name}: {spec.path}")
    file_format = spec.file_format
    if file_format == "auto":
        suffix = spec.path.suffix.casefold()
        if suffix == ".csv":
            file_format = "csv"
        elif suffix in {".parquet", ".pq"}:
            file_format = "parquet"
        else:
            raise ResearchDataError(f"cannot infer file format for {spec.name}: {spec.path.suffix}")
    try:
        if file_format == "csv":
            return pd.read_csv(spec.path, **spec.csv_options)
        return pd.read_parquet(spec.path)
    except Exception as exc:
        raise ResearchDataError(f"failed to read {spec.name} from {spec.path}: {exc}") from exc


def _parse_timestamp(values: pd.Series, source_timezone: str, target_timezone: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    try:
        timezone = parsed.dt.tz
    except AttributeError as exc:
        raise ResearchDataError("timestamp column contains incompatible mixed timezone values") from exc
    if timezone is None:
        parsed = parsed.dt.tz_localize(source_timezone, ambiguous="NaT", nonexistent="NaT")
    return parsed.dt.tz_convert(target_timezone)


def _boundary(value: datetime | None, timezone: str) -> pd.Timestamp | None:
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(timezone)
    return timestamp.tz_convert(timezone)


def load_series(
    spec: SeriesSpec,
    *,
    study_timezone: str,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
) -> LoadedSeries:
    """Read one source and normalize timestamps/numbers without hiding bad rows."""

    raw = _read_frame(spec)
    required = {spec.timestamp_column, spec.value_column}
    if spec.available_at_column:
        required.add(spec.available_at_column)
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ResearchDataError(f"{spec.name} is missing required columns: {', '.join(missing)}")

    source_timezone = spec.timezone or study_timezone
    timestamp = _parse_timestamp(raw[spec.timestamp_column], source_timezone, study_timezone)
    value = pd.to_numeric(raw[spec.value_column], errors="coerce")
    invalid_timestamp_rows = int(timestamp.isna().sum())
    non_numeric_rows = int((raw[spec.value_column].notna() & value.isna()).sum())

    frame = pd.DataFrame({"timestamp": timestamp, "value": value})
    if spec.available_at_column:
        frame["available_at"] = _parse_timestamp(raw[spec.available_at_column], source_timezone, study_timezone)

    frame = frame.loc[frame["timestamp"].notna()].copy()
    start = _boundary(start_time, study_timezone)
    end = _boundary(end_time, study_timezone)
    if start is not None:
        frame = frame.loc[frame["timestamp"] >= start]
    if end is not None:
        frame = frame.loc[frame["timestamp"] <= end]

    duplicate_mask = frame["timestamp"].duplicated(keep=False)
    duplicate_timestamp_rows = int(duplicate_mask.sum())
    duplicate_timestamp_keys = int(frame.loc[duplicate_mask, "timestamp"].nunique())
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    if frame.empty:
        raise ResearchDataError(f"{spec.name} has no valid rows in the configured study window")

    return LoadedSeries(
        spec=spec,
        frame=frame,
        raw_rows=len(raw),
        invalid_timestamp_rows=invalid_timestamp_rows,
        non_numeric_rows=non_numeric_rows,
        duplicate_timestamp_rows=duplicate_timestamp_rows,
        duplicate_timestamp_keys=duplicate_timestamp_keys,
    )
