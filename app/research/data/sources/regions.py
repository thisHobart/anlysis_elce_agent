"""Region-aware, read-only electricity-price sources for desktop sessions.

Region definitions contain no credentials.  Operators add a region to the YAML
catalog and provide the referenced environment-variable group beside the app.
The desktop receives a normalized Parquet file, so the existing evidence,
fingerprint, and immutable-snapshot boundaries stay unchanged.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yaml
from dotenv import dotenv_values

from app.config import runtime_env_file
from app.research.data.sources.summary import DataSummary, build_summary
from app.runtime_paths import application_data_directory, source_worktree

REGION_CATALOG_FILENAME = "region_databases.yaml"
REGION_CATALOG_ENVIRONMENT_VARIABLE = "PRICE_RESEARCH_REGION_DATABASES"
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class RegionSourceError(RuntimeError):
    """A configured regional price source could not be read safely."""


@dataclass(frozen=True)
class RegionalSeries:
    """One business-clock series selected from a region's feature database."""

    name: str
    table: str
    type_value: int | None = None
    value_column: str = "quantity"

    def __post_init__(self) -> None:
        for value in (self.name, self.table, self.value_column):
            if not _IDENTIFIER.fullmatch(value):
                raise ValueError(f"unsafe regional series identifier: {value}")


@dataclass(frozen=True)
class RegionProfile:
    """Non-secret contract for one regional actual-price source."""

    region_id: str
    label: str
    market: str
    province_value: str
    timezone: str
    frequency: str
    credential_prefix: str
    price_table: str
    price_label: str
    analytics_credential_prefix: str | None = None
    weather_actual_table: str | None = None
    weather_forecast_table: str | None = None
    weather_daily_table: str | None = None
    weather_columns: tuple[str, ...] = ()
    price_history_table: str | None = None
    actual_series: tuple[RegionalSeries, ...] = ()
    forecast_series: tuple[RegionalSeries, ...] = ()
    target_name: str = "rt_price"
    province_column: str = "province"
    type_column: str = "type"
    type_value: int = 1
    date_column: str = "ts_day"
    hour_column: str = "hour"
    minute_column: str = "min"
    value_column: str = "quantity"
    available_at_column: str = "create_time"

    def __post_init__(self) -> None:
        object.__setattr__(self, "weather_columns", tuple(self.weather_columns))
        for field_name in ("actual_series", "forecast_series"):
            raw_series = getattr(self, field_name)
            converted = tuple(
                item if isinstance(item, RegionalSeries) else RegionalSeries(**item)
                for item in raw_series
            )
            object.__setattr__(self, field_name, converted)
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", self.region_id):
            raise ValueError(f"invalid region id: {self.region_id}")
        ZoneInfo(self.timezone)
        for value in (
            self.price_table,
            self.target_name,
            self.province_column,
            self.type_column,
            self.date_column,
            self.hour_column,
            self.minute_column,
            self.value_column,
            self.available_at_column,
            *(value for value in self.weather_columns),
            *(
                value
                for value in (
                    self.weather_actual_table,
                    self.weather_forecast_table,
                    self.weather_daily_table,
                    self.price_history_table,
                )
                if value is not None
            ),
        ):
            if not _IDENTIFIER.fullmatch(value):
                raise ValueError(f"unsafe database identifier in region {self.region_id}: {value}")
        configured_weather = (
            self.analytics_credential_prefix,
            self.weather_actual_table,
            self.weather_forecast_table,
            self.weather_columns,
        )
        if any(configured_weather) and not all(configured_weather):
            raise ValueError(f"incomplete weather source in region {self.region_id}")
        if self.analytics_credential_prefix and not _IDENTIFIER.fullmatch(self.analytics_credential_prefix):
            raise ValueError(f"unsafe credential prefix in region {self.region_id}")
        names = [item.name for item in (*self.actual_series, *self.forecast_series)]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate regional series name in region {self.region_id}")


@dataclass(frozen=True)
class DatabaseCredentials:
    """One connection secret group; repr deliberately omits the password."""

    host: str
    port: int
    user: str
    database: str
    password: str = field(repr=False)


@dataclass(frozen=True)
class RegionPriceFetch:
    """Result of one bounded regional dataset fetch for a desktop session."""

    profile: RegionProfile
    path: Path
    fetched_at: datetime
    row_count: int
    start_at: datetime
    end_at: datetime
    duplicate_timestamp_rows: int
    invalid_rows: int
    database_name: str
    actuals_path: Path | None = None
    forecasts_path: Path | None = None
    actual_variable_names: tuple[str, ...] = ()
    forecast_variable_names: tuple[str, ...] = ()
    series_details: dict[str, Any] = field(default_factory=dict)
    weather_details: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> DataSummary:
        table_names = {self.profile.target_name: self.profile.price_table}
        table_names.update({item.name: item.table for item in self.profile.actual_series})
        table_names.update({item.name: item.table for item in self.profile.forecast_series})
        return build_summary(
            market=self.profile.market,
            target_name=self.profile.target_name,
            exogenous_names=[*self.actual_variable_names, *self.forecast_variable_names],
            start_time=self.start_at,
            end_time=self.end_at,
            frequency=self.profile.frequency,
            fetched_at=self.fetched_at,
            table_names=table_names,
        )

    def safe_details(self) -> dict[str, Any]:
        """Return audit details that contain neither an address nor credentials."""

        return {
            "region_id": self.profile.region_id,
            "region_label": self.profile.label,
            "market": self.profile.market,
            "database": self.database_name,
            "table": self.profile.price_table,
            "price_kind": self.profile.price_label,
            "frequency": self.profile.frequency,
            "fetched_at": self.fetched_at.isoformat(),
            "rows": self.row_count,
            "start_at": self.start_at.isoformat(),
            "end_at": self.end_at.isoformat(),
            "duplicate_timestamp_rows": self.duplicate_timestamp_rows,
            "invalid_rows": self.invalid_rows,
            "series": self.series_details,
            "weather": self.weather_details,
        }


def region_catalog_path() -> Path | None:
    """Locate the non-secret catalog in a checkout, bundle, or operator override."""

    configured = os.environ.get(REGION_CATALOG_ENVIRONMENT_VARIABLE)
    env_path = runtime_env_file()
    if configured is None and env_path.is_file():
        configured = dotenv_values(env_path).get(REGION_CATALOG_ENVIRONMENT_VARIABLE)
    if configured:
        candidate = Path(configured).expanduser()
        if not candidate.is_absolute():
            candidate = env_path.parent / candidate
        return candidate if candidate.is_file() else None
    roots = [Path(__file__).resolve().parent]
    worktree = source_worktree()
    if worktree is not None:
        roots.append(worktree / "configs")
    for root in roots:
        candidate = root / REGION_CATALOG_FILENAME
        if candidate.is_file():
            return candidate
    return None


def load_region_profiles(path: str | Path | None = None) -> dict[str, RegionProfile]:
    """Load and validate every selectable region without touching its database."""

    source = Path(path) if path is not None else region_catalog_path()
    if source is None:
        raise RegionSourceError("找不到地区数据配置")
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RegionSourceError(f"地区数据配置无法读取：{exc}") from exc
    rows = payload.get("regions") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise RegionSourceError("地区数据配置没有可用地区")
    profiles: dict[str, RegionProfile] = {}
    try:
        for row in rows:
            if not isinstance(row, dict):
                raise TypeError("region entry must be a mapping")
            profile = RegionProfile(**row)
            if profile.region_id in profiles:
                raise ValueError(f"duplicate region id: {profile.region_id}")
            profiles[profile.region_id] = profile
    except (TypeError, ValueError) as exc:
        raise RegionSourceError(f"地区数据配置无效：{exc}") from exc
    return profiles


def credentials_for(
    profile: RegionProfile,
    credential_prefix: str | None = None,
) -> DatabaseCredentials:
    """Read one region's secrets with process environment taking precedence over .env."""

    file_values = dotenv_values(runtime_env_file()) if runtime_env_file().is_file() else {}
    values: dict[str, str] = {}
    prefix = credential_prefix or profile.credential_prefix
    for suffix in ("HOST", "PORT", "USER", "PASSWORD", "NAME"):
        key = f"{prefix}_{suffix}"
        value = os.environ.get(key)
        if value is None:
            value = file_values.get(key)
        values[suffix] = str(value or "").strip()
    missing = [suffix for suffix, value in values.items() if not value]
    if missing:
        names = "、".join(f"{prefix}_{suffix}" for suffix in missing)
        raise RegionSourceError(f"{profile.label}的数据连接还没配置完整：{names}")
    try:
        port = int(values["PORT"])
    except ValueError as exc:
        raise RegionSourceError(f"{prefix}_PORT 必须是整数") from exc
    return DatabaseCredentials(
        host=values["HOST"],
        port=port,
        user=values["USER"],
        database=values["NAME"],
        password=values["PASSWORD"],
    )


def _connect(credentials: DatabaseCredentials):
    try:
        import pymysql
    except ImportError as exc:  # pragma: no cover - dependency is declared by the project
        raise RegionSourceError("缺少地区数据连接组件") from exc
    connection = pymysql.connect(
        host=credentials.host,
        port=credentials.port,
        user=credentials.user,
        password=credentials.password,
        database=credentials.database,
        charset="utf8mb4",
        connect_timeout=10,
        read_timeout=120,
        write_timeout=10,
        autocommit=False,
    )
    with connection.cursor() as cursor:
        cursor.execute("SET SESSION TRANSACTION READ ONLY")
        cursor.execute("START TRANSACTION READ ONLY")
    return connection


def _business_timestamp(day: date | datetime | str, hour: Any, minute: Any) -> datetime:
    hour_value, minute_value = int(hour), int(minute)
    if not 0 <= hour_value <= 24 or not 0 <= minute_value < 60 or (hour_value == 24 and minute_value != 0):
        raise ValueError("invalid business clock")
    return datetime.fromisoformat(str(day)[:10]) + timedelta(hours=hour_value, minutes=minute_value)


def _business_time_sql(profile: RegionProfile) -> str:
    def quote(value: str) -> str:
        return f"`{value}`"

    return (
        f"TIMESTAMP(DATE_ADD({quote(profile.date_column)}, "
        f"INTERVAL ({quote(profile.hour_column)}*60+{quote(profile.minute_column)}) MINUTE))"
    )


def _price_query(profile: RegionProfile, table: str | None = None) -> str:
    def quote(value: str) -> str:
        return f"`{value}`"

    business_time = _business_time_sql(profile)
    columns = ",".join(
        quote(value)
        for value in (
            profile.date_column,
            profile.hour_column,
            profile.minute_column,
            profile.value_column,
            profile.available_at_column,
        )
    )
    return (
        f"SELECT {columns} FROM {quote(table or profile.price_table)} "
        f"WHERE {quote(profile.province_column)}=%s AND {quote(profile.type_column)}=%s "
        f"AND {quote(profile.available_at_column)}<=%s AND {business_time}<=%s "
        f"ORDER BY {quote(profile.date_column)},{quote(profile.hour_column)},{quote(profile.minute_column)},"
        f"{quote(profile.available_at_column)}"
    )


def _series_query(
    profile: RegionProfile,
    series: RegionalSeries,
    *,
    future: bool,
    cutoff: datetime,
) -> tuple[str, tuple[Any, ...]]:
    type_filter = "" if series.type_value is None else f" AND `{profile.type_column}`=%s"
    future_filter = "" if future else f" AND {_business_time_sql(profile)}<=%s"
    sql = (
        f"SELECT `{profile.date_column}`,`{profile.hour_column}`,`{profile.minute_column}`,"
        f"`{series.value_column}`,`{profile.available_at_column}` FROM `{series.table}` "
        f"WHERE `{profile.province_column}`=%s{type_filter} "
        f"AND `{profile.available_at_column}`<=%s{future_filter} "
        f"ORDER BY `{profile.date_column}`,`{profile.hour_column}`,`{profile.minute_column}`,"
        f"`{profile.available_at_column}`"
    )
    parameters: list[Any] = [profile.province_value]
    if series.type_value is not None:
        parameters.append(series.type_value)
    parameters.append(cutoff)
    if not future:
        parameters.append(cutoff)
    return sql, tuple(parameters)


def _normalize_business_series(
    raw: pd.DataFrame,
    profile: RegionProfile,
    *,
    name: str,
) -> tuple[pd.DataFrame, int, int]:
    timestamps: list[datetime | None] = []
    for row in raw.itertuples(index=False):
        try:
            timestamps.append(
                _business_timestamp(
                    getattr(row, profile.date_column),
                    getattr(row, profile.hour_column),
                    getattr(row, profile.minute_column),
                )
            )
        except (TypeError, ValueError):
            timestamps.append(None)
    normalized = pd.DataFrame(
        {
            "timestamp": timestamps,
            name: pd.to_numeric(raw.iloc[:, 3], errors="coerce"),
            "available_at": pd.to_datetime(raw[profile.available_at_column], errors="coerce"),
        }
    )
    invalid = int(normalized[["timestamp", name, "available_at"]].isna().any(axis=1).sum())
    normalized = normalized.dropna(subset=["timestamp", name, "available_at"])
    duplicates = int(normalized.duplicated("timestamp", keep=False).sum())
    normalized = (
        normalized.sort_values(["timestamp", "available_at"], kind="stable")
        .drop_duplicates("timestamp", keep="last")
        .sort_values("timestamp", kind="stable")
        .reset_index(drop=True)
    )
    return normalized, duplicates, invalid


def _combine_series_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Outer-join named series and retain the most conservative row availability."""

    if not frames:
        return pd.DataFrame(columns=["timestamp", "available_at"])
    combined: pd.DataFrame | None = None
    availability_columns: list[str] = []
    for index, frame in enumerate(frames):
        value_columns = [column for column in frame if column not in {"timestamp", "available_at"}]
        if not value_columns:
            continue
        available_name = f"_available_{index}"
        prepared = frame.rename(columns={"available_at": available_name})
        availability_columns.append(available_name)
        combined = prepared if combined is None else combined.merge(prepared, on="timestamp", how="outer")
    if combined is None:
        return pd.DataFrame(columns=["timestamp", "available_at"])
    combined["available_at"] = combined[availability_columns].max(axis=1)
    return combined.drop(columns=availability_columns).sort_values("timestamp", kind="stable").reset_index(drop=True)


def _query_frame(connection: Any, sql: str, parameters: tuple[Any, ...]) -> pd.DataFrame:
    if not re.match(r"^\s*SELECT\b", sql, re.IGNORECASE):
        raise RegionSourceError("地区数据源拒绝非只读查询")
    with connection.cursor() as cursor:
        cursor.execute(sql, parameters)
        rows = cursor.fetchall()
        columns = [str(item[0]) for item in cursor.description]
    return pd.DataFrame(rows, columns=columns)


def _weather_query(profile: RegionProfile, table: str, *, future: bool) -> str:
    columns = ",".join(f"`{value}`" for value in profile.weather_columns)
    future_limit = "" if future else " AND `forecast_time`<=%s"
    return (
        f"SELECT `forecast_time`,{columns},`create_time` FROM `{table}` "
        f"WHERE `province_name`=%s AND `create_time`<=%s{future_limit} "
        "ORDER BY `forecast_time`,`create_time`"
    )


def _daily_weather_profile(connection: Any, profile: RegionProfile, cutoff: datetime) -> dict[str, Any]:
    if profile.weather_daily_table is None:
        return {}
    frame = _query_frame(
        connection,
        f"SELECT COUNT(*) rows_total,MIN(`forecast_date`) start_date,MAX(`forecast_date`) end_date "
        f"FROM `{profile.weather_daily_table}` WHERE `province_name`=%s AND `create_time`<=%s",
        (profile.province_value, cutoff),
    )
    if frame.empty:
        return {"table": profile.weather_daily_table, "rows": 0}
    row = frame.iloc[0]
    return {
        "table": profile.weather_daily_table,
        "rows": int(row["rows_total"] or 0),
        "start_date": str(row["start_date"] or ""),
        "end_date": str(row["end_date"] or ""),
        "used_on_15_minute_axis": False,
    }


def _normalize_weather(frame: pd.DataFrame, profile: RegionProfile) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["timestamp", *profile.weather_columns, "available_at"])
    normalized = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(frame["forecast_time"], errors="coerce"),
            "available_at": pd.to_datetime(frame["create_time"], errors="coerce"),
        }
    )
    for column in profile.weather_columns:
        normalized[column] = pd.to_numeric(frame[column], errors="coerce")
    normalized = normalized.dropna(subset=["timestamp", "available_at"])
    return (
        normalized.sort_values(["timestamp", "available_at"], kind="stable")
        .drop_duplicates("timestamp", keep="last")
        .set_index("timestamp")
        .sort_index()
    )


def _expand_hourly_weather(frame: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Repeat each hourly value only within its own hour on the research grid."""

    if frame.empty:
        return frame.reset_index()
    offset = pd.tseries.frequencies.to_offset(frequency)
    offset_delta = pd.Timedelta(offset.nanos, unit="ns")
    hour = pd.Timedelta(hours=1)
    steps = max(1, int(hour / offset_delta) - 1)
    grid = pd.date_range(frame.index.min(), frame.index.max() + hour - offset_delta, freq=frequency)
    return frame.reindex(grid).ffill(limit=steps).rename_axis("timestamp").reset_index()


def _expand_hour_ending_series(frame: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Expand an hourly value over the quarter-hours ending at its business hour."""

    if frame.empty:
        return frame.reset_index()
    offset = pd.tseries.frequencies.to_offset(frequency)
    offset_delta = pd.Timedelta(offset.nanos, unit="ns")
    periods = max(1, int(pd.Timedelta(hours=1) / offset_delta))
    expanded = []
    for shift in range(periods):
        copy = frame.copy()
        copy.index = copy.index - shift * offset_delta
        expanded.append(copy)
    return (
        pd.concat(expanded)
        .sort_index(kind="stable")
        .rename_axis("timestamp")
        .reset_index()
    )


def _merge_actual_weather(
    era5: pd.DataFrame,
    gfs: pd.DataFrame,
    profile: RegionProfile,
) -> tuple[pd.DataFrame, int]:
    """Prefer ERA5 at each field and use GFS only where the historical value is absent."""

    era5_values = _normalize_weather(era5, profile)
    gfs_values = _normalize_weather(gfs, profile)
    era5_times = set(era5_values.index)
    gfs_fill_times = len(set(gfs_values.index).difference(era5_times))
    merged = era5_values.combine_first(gfs_values).sort_index()
    return _expand_hourly_weather(merged, profile.frequency), gfs_fill_times


def _write_parquet_atomic(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def fetch_region_price(
    profile: RegionProfile,
    *,
    output_directory: str | Path | None = None,
    progress: Callable[[int, str], None] | None = None,
    now: datetime | None = None,
    connection_factory: Callable[[DatabaseCredentials], Any] = _connect,
) -> RegionPriceFetch:
    """Fetch one region's price target and configured exogenous factors atomically."""

    fetched_at = now or datetime.now(UTC).astimezone()
    if fetched_at.tzinfo is None or fetched_at.utcoffset() is None:
        raise RegionSourceError("取数时间必须包含时区")
    local_cutoff = fetched_at.astimezone(ZoneInfo(profile.timezone)).replace(tzinfo=None)
    credentials = credentials_for(profile)
    if progress:
        progress(10, "正在连接所选地区的数据")
    connection = connection_factory(credentials)
    try:
        raw = _query_frame(
            connection,
            _price_query(profile),
            (profile.province_value, profile.type_value, local_cutoff, local_cutoff),
        )
        history_raw = (
            _query_frame(
                connection,
                _price_query(profile, profile.price_history_table),
                (profile.province_value, profile.type_value, local_cutoff, local_cutoff),
            )
            if profile.price_history_table
            else pd.DataFrame()
        )
        actual_raw: list[tuple[RegionalSeries, pd.DataFrame]] = []
        forecast_raw: list[tuple[RegionalSeries, pd.DataFrame]] = []
        for series in profile.actual_series:
            query, parameters = _series_query(profile, series, future=False, cutoff=local_cutoff)
            actual_raw.append((series, _query_frame(connection, query, parameters)))
        for series in profile.forecast_series:
            query, parameters = _series_query(profile, series, future=True, cutoff=local_cutoff)
            forecast_raw.append((series, _query_frame(connection, query, parameters)))
    except Exception as exc:
        raise RegionSourceError(f"{profile.label}电价或影响因素读取失败：{exc}") from exc
    finally:
        try:
            connection.rollback()
        finally:
            connection.close()
    if progress:
        progress(45, "正在整理电价时间点")
    if raw.empty:
        raise RegionSourceError(f"{profile.label}当前没有可用的实际电价")
    normalized, duplicate_rows, invalid_rows = _normalize_business_series(
        raw,
        profile,
        name=profile.target_name,
    )
    if normalized.empty:
        raise RegionSourceError(f"{profile.label}实际电价没有有效时间点或数值")
    if not history_raw.empty:
        history, history_duplicates, history_invalid = _normalize_business_series(
            history_raw,
            profile,
            name=profile.target_name,
        )
        duplicate_rows += history_duplicates
        invalid_rows += history_invalid
        if not history.empty:
            expanded_history = _expand_hour_ending_series(
                history.set_index("timestamp"),
                profile.frequency,
            ).set_index("timestamp")
            normalized = (
                normalized.set_index("timestamp")
                .combine_first(expanded_history)
                .reset_index()
            )
            normalized = normalized.loc[normalized["timestamp"] <= local_cutoff]
            normalized = normalized.sort_values("timestamp", kind="stable").reset_index(drop=True)
    root = Path(output_directory) if output_directory is not None else application_data_directory() / "region-data"
    destination = (root / profile.region_id / "actual-realtime-price.parquet").resolve()
    _write_parquet_atomic(normalized, destination)

    actuals_path: Path | None = None
    forecasts_path: Path | None = None
    actual_names: tuple[str, ...] = ()
    forecast_names: tuple[str, ...] = ()
    weather_details: dict[str, Any] = {}
    series_details: dict[str, Any] = {"actual": {}, "forecast": {}}
    actual_frames: list[pd.DataFrame] = []
    forecast_frames: list[pd.DataFrame] = []
    for group, rows, destination_frames in (
        ("actual", actual_raw, actual_frames),
        ("forecast", forecast_raw, forecast_frames),
    ):
        for series, series_raw in rows:
            frame, duplicates, invalid = _normalize_business_series(
                series_raw,
                profile,
                name=series.name,
            )
            if not frame.empty:
                destination_frames.append(frame)
            series_details[group][series.name] = {
                "table": series.table,
                "source_rows": len(series_raw),
                "usable_rows": len(frame),
                "start_at": frame["timestamp"].iloc[0].isoformat() if not frame.empty else "",
                "end_at": frame["timestamp"].iloc[-1].isoformat() if not frame.empty else "",
                "duplicate_timestamp_rows": duplicates,
                "invalid_rows": invalid,
            }
    weather_ready = all(
        (
            profile.analytics_credential_prefix,
            profile.weather_actual_table,
            profile.weather_forecast_table,
            profile.weather_columns,
        )
    )
    if weather_ready:
        if progress:
            progress(55, "正在读取历史和预测影响因素")
        analytics_credentials = credentials_for(profile, profile.analytics_credential_prefix)
        analytics_connection = connection_factory(analytics_credentials)
        try:
            era5 = _query_frame(
                analytics_connection,
                _weather_query(profile, str(profile.weather_actual_table), future=False),
                (profile.province_value, local_cutoff, local_cutoff),
            )
            gfs = _query_frame(
                analytics_connection,
                _weather_query(profile, str(profile.weather_forecast_table), future=True),
                (profile.province_value, local_cutoff),
            )
            daily_profile = _daily_weather_profile(analytics_connection, profile, local_cutoff)
        except Exception as exc:
            raise RegionSourceError(f"{profile.label}影响因素读取失败：{exc}") from exc
        finally:
            try:
                analytics_connection.rollback()
            finally:
                analytics_connection.close()
        gfs_timestamps = pd.to_datetime(gfs.get("forecast_time"), errors="coerce")
        gfs_history = gfs.loc[gfs_timestamps <= local_cutoff].copy()
        actual_weather, gfs_fill_times = _merge_actual_weather(era5, gfs_history, profile)
        forecast_weather = _expand_hourly_weather(_normalize_weather(gfs, profile), profile.frequency)
        weather_actual_names = tuple(
            column for column in profile.weather_columns if actual_weather[column].notna().any()
        )
        forecast_source_names = tuple(
            column for column in profile.weather_columns if forecast_weather[column].notna().any()
        )
        weather_forecast_names = tuple(
            f"forecast_{column}" if column in weather_actual_names else column
            for column in forecast_source_names
        )
        if weather_actual_names:
            actual_frames.append(actual_weather[["timestamp", *weather_actual_names, "available_at"]])
        if forecast_source_names:
            forecast_frames.append(
                forecast_weather[["timestamp", *forecast_source_names, "available_at"]]
            )
        weather_details = {
            "database": analytics_credentials.database,
            "actual_table": profile.weather_actual_table,
            "forecast_table": profile.weather_forecast_table,
            "actual_source_rows": len(era5),
            "forecast_source_rows": len(gfs),
            "actual_15_minute_rows": len(actual_weather),
            "forecast_15_minute_rows": len(forecast_weather),
            "gfs_history_fill_hours": gfs_fill_times,
            "actual_variables": list(weather_actual_names),
            "forecast_variables": list(weather_forecast_names),
            "daily_forecast": daily_profile,
        }
    combined_actuals = _combine_series_frames(actual_frames)
    combined_forecasts = _combine_series_frames(forecast_frames)
    actual_names = tuple(
        column for column in combined_actuals if column not in {"timestamp", "available_at"}
    )
    forecast_source_names = tuple(
        column for column in combined_forecasts if column not in {"timestamp", "available_at"}
    )
    forecast_names = tuple(
        f"forecast_{column}" if column in actual_names else column
        for column in forecast_source_names
    )
    if actual_names:
        actuals_path = (root / profile.region_id / "actual-factors.parquet").resolve()
        _write_parquet_atomic(combined_actuals, actuals_path)
    if forecast_source_names:
        forecasts_path = (root / profile.region_id / "forecast-factors.parquet").resolve()
        _write_parquet_atomic(combined_forecasts, forecasts_path)
    if progress:
        progress(100, "所选地区的电价和影响因素已经就绪")
    return RegionPriceFetch(
        profile=profile,
        path=destination,
        fetched_at=fetched_at,
        row_count=len(normalized),
        start_at=normalized["timestamp"].iloc[0],
        end_at=normalized["timestamp"].iloc[-1],
        duplicate_timestamp_rows=duplicate_rows,
        invalid_rows=invalid_rows,
        database_name=credentials.database,
        actuals_path=actuals_path,
        forecasts_path=forecasts_path,
        actual_variable_names=actual_names,
        forecast_variable_names=forecast_names,
        series_details=series_details,
        weather_details=weather_details,
    )
