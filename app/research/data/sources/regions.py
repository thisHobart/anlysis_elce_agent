"""Region-aware, read-only electricity-price sources for desktop sessions.

Region definitions contain no credentials.  Operators add a region to the YAML
catalog and provide the referenced environment-variable group beside the app.
The desktop receives a normalized Parquet file, so the existing evidence,
fingerprint, and immutable-snapshot boundaries stay unchanged.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4
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
REGION_CACHE_FORMAT_VERSION = 1
REGION_CACHE_POINTER = "current.json"
REGION_FETCH_MANIFEST = "fetch.json"


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
class RegionalTimePointSeries:
    """One categorical series stored as date + HH:mm text."""

    name: str
    table: str
    category_value: str
    category_column: str = "lx"
    date_column: str = "trade_date"
    time_column: str = "time_point"
    value_column: str = "quantity"
    available_at_column: str | None = "create_time"

    def __post_init__(self) -> None:
        for value in (
            self.name,
            self.table,
            self.category_column,
            self.date_column,
            self.time_column,
            self.value_column,
            *(value for value in (self.available_at_column,) if value is not None),
        ):
            if not _IDENTIFIER.fullmatch(value):
                raise ValueError(f"unsafe regional time-point series identifier: {value}")
        if not self.category_value.strip():
            raise ValueError("regional time-point category value cannot be empty")


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
    history_start_date: date = date(1970, 1, 1)
    refresh_lookback_days: int = 30
    analytics_credential_prefix: str | None = None
    weather_actual_table: str | None = None
    weather_forecast_table: str | None = None
    weather_daily_table: str | None = None
    weather_columns: tuple[str, ...] = ()
    price_history_table: str | None = None
    price_credential_prefix: str | None = None
    market_credential_prefix: str | None = None
    actual_series: tuple[RegionalSeries, ...] = ()
    forecast_series: tuple[RegionalSeries, ...] = ()
    market_series: tuple[RegionalTimePointSeries, ...] = ()
    target_name: str = "rt_price"
    province_column: str = "province"
    type_column: str = "type"
    type_value: int = 1
    date_column: str = "ts_day"
    hour_column: str = "hour"
    minute_column: str = "min"
    value_column: str = "quantity"
    available_at_column: str = "create_time"
    price_layout: str = "business_clock"
    price_date_column: str = "trade_date"
    price_time_column: str = "time_point"
    price_value_column: str = "price"
    price_available_at_column: str | None = "create_time"
    price_filter_province: bool = True
    price_filter_type: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.history_start_date, str):
            object.__setattr__(self, "history_start_date", date.fromisoformat(self.history_start_date))
        if not isinstance(self.history_start_date, date):
            raise TypeError(f"invalid history start date in region {self.region_id}")
        try:
            lookback_days = int(self.refresh_lookback_days)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid refresh lookback in region {self.region_id}") from exc
        object.__setattr__(self, "refresh_lookback_days", lookback_days)
        if lookback_days < 1:
            raise ValueError(f"refresh lookback must be positive in region {self.region_id}")
        object.__setattr__(self, "weather_columns", tuple(self.weather_columns))
        for field_name in ("actual_series", "forecast_series"):
            raw_series = getattr(self, field_name)
            converted = tuple(
                item if isinstance(item, RegionalSeries) else RegionalSeries(**item)
                for item in raw_series
            )
            object.__setattr__(self, field_name, converted)
        object.__setattr__(
            self,
            "market_series",
            tuple(
                item if isinstance(item, RegionalTimePointSeries) else RegionalTimePointSeries(**item)
                for item in self.market_series
            ),
        )
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", self.region_id):
            raise ValueError(f"invalid region id: {self.region_id}")
        if self.price_layout not in {"business_clock", "time_point"}:
            raise ValueError(f"invalid price layout in region {self.region_id}: {self.price_layout}")
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
            self.price_date_column,
            self.price_time_column,
            self.price_value_column,
            *(value for value in (self.price_available_at_column,) if value is not None),
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
        for prefix in (self.price_credential_prefix, self.market_credential_prefix):
            if prefix and not _IDENTIFIER.fullmatch(prefix):
                raise ValueError(f"unsafe credential prefix in region {self.region_id}")
        names = [
            item.name
            for item in (*self.actual_series, *self.forecast_series, *self.market_series)
        ]
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
    fetch_mode: str = "full"
    query_start_at: datetime | None = None
    query_metrics: tuple[dict[str, Any], ...] = ()
    cache_generation: str = ""

    def summary(self) -> DataSummary:
        table_names = {self.profile.target_name: self.profile.price_table}
        table_names.update({item.name: item.table for item in self.profile.actual_series})
        table_names.update({item.name: item.table for item in self.profile.forecast_series})
        table_names.update({item.name: item.table for item in self.profile.market_series})
        variable_kinds = {
            **{name: "actual" for name in self.actual_variable_names},
            **{name: "forecast" for name in self.forecast_variable_names},
        }
        return build_summary(
            market=self.profile.market,
            target_name=self.profile.target_name,
            exogenous_names=[*self.actual_variable_names, *self.forecast_variable_names],
            start_time=self.start_at,
            end_time=self.end_at,
            frequency=self.profile.frequency,
            fetched_at=self.fetched_at,
            table_names=table_names,
            variable_kinds=variable_kinds,
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
            "price_availability": (
                self.profile.price_available_at_column or "business_timestamp_assumption"
            ),
            "frequency": self.profile.frequency,
            "fetched_at": self.fetched_at.isoformat(),
            "rows": self.row_count,
            "start_at": self.start_at.isoformat(),
            "end_at": self.end_at.isoformat(),
            "duplicate_timestamp_rows": self.duplicate_timestamp_rows,
            "invalid_rows": self.invalid_rows,
            "series": self.series_details,
            "weather": self.weather_details,
            "fetch_mode": self.fetch_mode,
            "query_start_at": self.query_start_at.isoformat() if self.query_start_at else "",
            "query_metrics": list(self.query_metrics),
            "cache_generation": self.cache_generation,
        }


@dataclass(frozen=True)
class RegionalSnapshot:
    """Immutable, read-only point-in-time view used by forecasting workflows."""

    profile: RegionProfile
    as_of: datetime
    path: Path
    actuals_path: Path | None
    forecasts_path: Path | None
    source: RegionPriceFetch = field(repr=False)

    def read_target(self) -> pd.DataFrame:
        return pd.read_parquet(self.path)

    def read_actuals(self) -> pd.DataFrame:
        return pd.read_parquet(self.actuals_path) if self.actuals_path else pd.DataFrame()

    def read_forecasts(self) -> pd.DataFrame:
        return pd.read_parquet(self.forecasts_path) if self.forecasts_path else pd.DataFrame()


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


def _text_time_sql(date_column: str, time_column: str) -> str:
    """Build a timestamp expression that also accepts the market value 24:00."""

    return (
        f"TIMESTAMP(DATE_ADD(`{date_column}`, INTERVAL ("
        f"CAST(SUBSTRING_INDEX(`{time_column}`, ':', 1) AS UNSIGNED)*60+"
        f"CAST(SUBSTRING_INDEX(`{time_column}`, ':', -1) AS UNSIGNED)) MINUTE))"
    )


def _price_query(profile: RegionProfile, table: str | None = None) -> str:
    def quote(value: str) -> str:
        return f"`{value}`"

    if profile.price_layout == "time_point":
        market_time = _text_time_sql(profile.price_date_column, profile.price_time_column)
        price_columns = [
            profile.price_date_column,
            profile.price_time_column,
            profile.price_value_column,
        ]
        if profile.price_available_at_column:
            price_columns.append(profile.price_available_at_column)
        columns = ",".join(quote(value) for value in price_columns)
        filters: list[str] = []
        if profile.price_filter_province:
            filters.append(f"{quote(profile.province_column)}=%s")
        if profile.price_filter_type:
            filters.append(f"{quote(profile.type_column)}=%s")
        filters.extend(
            (
                f"{quote(profile.price_date_column)}>=%s",
                f"{quote(profile.price_date_column)}<=%s",
            )
        )
        if profile.price_available_at_column:
            filters.append(f"{quote(profile.price_available_at_column)}<=%s")
        filters.extend((f"{market_time}>=%s", f"{market_time}<=%s"))
        order_columns = [profile.price_date_column, profile.price_time_column]
        if profile.price_available_at_column:
            order_columns.append(profile.price_available_at_column)
        return (
            f"SELECT {columns} FROM {quote(table or profile.price_table)} "
            f"WHERE {' AND '.join(filters)} "
            f"ORDER BY {','.join(quote(value) for value in order_columns)}"
        )

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
        f"AND {quote(profile.date_column)}>=%s AND {quote(profile.date_column)}<=%s "
        f"AND {quote(profile.available_at_column)}<=%s "
        f"AND {business_time}>=%s AND {business_time}<=%s "
        f"ORDER BY {quote(profile.date_column)},{quote(profile.hour_column)},{quote(profile.minute_column)},"
        f"{quote(profile.available_at_column)}"
    )


def _price_parameters(
    profile: RegionProfile,
    *,
    start: datetime,
    cutoff: datetime,
) -> tuple[Any, ...]:
    if profile.price_layout == "time_point":
        parameters: list[Any] = []
        if profile.price_filter_province:
            parameters.append(profile.province_value)
        if profile.price_filter_type:
            parameters.append(profile.type_value)
        parameters.extend((start.date(), cutoff.date()))
        if profile.price_available_at_column:
            parameters.append(cutoff)
        parameters.extend((start, cutoff))
        return tuple(parameters)
    return (
        profile.province_value,
        profile.type_value,
        start.date(),
        cutoff.date(),
        cutoff,
        start,
        cutoff,
    )


def _series_group_query(
    profile: RegionProfile,
    series_group: tuple[RegionalSeries, ...],
    *,
    future: bool,
    start: datetime,
    cutoff: datetime,
) -> tuple[str, tuple[Any, ...]]:
    first = series_group[0]
    type_values = tuple(item.type_value for item in series_group if item.type_value is not None)
    type_filter = ""
    selected_type = ""
    if type_values:
        placeholders = ",".join("%s" for _value in type_values)
        type_filter = f" AND `{profile.type_column}` IN ({placeholders})"
        selected_type = f",`{profile.type_column}`"
    upper_date_filter = "" if future else f" AND `{profile.date_column}`<=%s"
    upper_time_filter = "" if future else f" AND {_business_time_sql(profile)}<=%s"
    sql = (
        f"SELECT `{profile.date_column}`,`{profile.hour_column}`,`{profile.minute_column}`,"
        f"`{first.value_column}`,`{profile.available_at_column}`{selected_type} FROM `{first.table}` "
        f"WHERE `{profile.province_column}`=%s{type_filter} "
        f"AND `{profile.date_column}`>=%s{upper_date_filter} "
        f"AND `{profile.available_at_column}`<=%s "
        f"AND {_business_time_sql(profile)}>=%s{upper_time_filter} "
        f"ORDER BY `{profile.date_column}`,`{profile.hour_column}`,`{profile.minute_column}`,"
        f"`{profile.available_at_column}`"
    )
    parameters: list[Any] = [profile.province_value, *type_values, start.date()]
    if not future:
        parameters.append(cutoff.date())
    parameters.extend((cutoff, start))
    if not future:
        parameters.append(cutoff)
    return sql, tuple(parameters)


def _series_query(
    profile: RegionProfile,
    series: RegionalSeries,
    *,
    future: bool,
    cutoff: datetime,
    start: datetime | None = None,
) -> tuple[str, tuple[Any, ...]]:
    """Compatibility wrapper for callers that need a single configured series."""

    return _series_group_query(
        profile,
        (series,),
        future=future,
        start=start or datetime.combine(profile.history_start_date, time.min),
        cutoff=cutoff,
    )


def _market_series_groups(
    series: tuple[RegionalTimePointSeries, ...],
) -> list[tuple[RegionalTimePointSeries, ...]]:
    grouped: dict[
        tuple[str, str, str, str, str, str | None],
        list[RegionalTimePointSeries],
    ] = {}
    for item in series:
        key = (
            item.table,
            item.category_column,
            item.date_column,
            item.time_column,
            item.value_column,
            item.available_at_column,
        )
        grouped.setdefault(key, []).append(item)
    return [tuple(items) for items in grouped.values()]


def _market_series_group_query(
    series_group: tuple[RegionalTimePointSeries, ...],
    *,
    start: datetime,
    cutoff: datetime,
) -> tuple[str, tuple[Any, ...]]:
    first = series_group[0]
    placeholders = ",".join("%s" for _item in series_group)
    market_time = _text_time_sql(first.date_column, first.time_column)
    selected_columns = [
        first.date_column,
        first.time_column,
        first.value_column,
        first.category_column,
    ]
    if first.available_at_column:
        selected_columns.insert(3, first.available_at_column)
    filters = [
        f"`{first.category_column}` IN ({placeholders})",
        f"`{first.date_column}`>=%s",
        f"`{first.date_column}`<=%s",
    ]
    if first.available_at_column:
        filters.append(f"`{first.available_at_column}`<=%s")
    filters.extend((f"{market_time}>=%s", f"{market_time}<=%s"))
    order_columns = [first.date_column, first.time_column]
    if first.available_at_column:
        order_columns.append(first.available_at_column)
    sql = (
        f"SELECT {','.join(f'`{value}`' for value in selected_columns)} "
        f"FROM `{first.table}` WHERE {' AND '.join(filters)} "
        f"ORDER BY {','.join(f'`{value}`' for value in order_columns)}"
    )
    parameters: tuple[Any, ...] = (
        *(item.category_value for item in series_group),
        start.date(),
        cutoff.date(),
    )
    if first.available_at_column:
        parameters = (*parameters, cutoff)
    parameters = (*parameters, start, cutoff)
    return sql, parameters


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


def _normalize_time_point_series(
    raw: pd.DataFrame,
    *,
    name: str,
    date_column: str,
    time_column: str,
    value_column: str,
    available_at_column: str | None,
) -> tuple[pd.DataFrame, int, int]:
    timestamps: list[datetime | None] = []
    for day_value, time_value in zip(raw.get(date_column, ()), raw.get(time_column, ()), strict=False):
        try:
            hour_text, minute_text = str(time_value).strip().split(":", 1)
            timestamps.append(_business_timestamp(day_value, hour_text, minute_text))
        except (AttributeError, TypeError, ValueError):
            timestamps.append(None)
    availability = (
        pd.to_datetime(raw.get(available_at_column), errors="coerce")
        if available_at_column
        else pd.to_datetime(timestamps, errors="coerce")
    )
    normalized = pd.DataFrame(
        {
            "timestamp": timestamps,
            name: pd.to_numeric(raw.get(value_column), errors="coerce"),
            "available_at": availability,
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


def _normalize_price_series(
    raw: pd.DataFrame,
    profile: RegionProfile,
) -> tuple[pd.DataFrame, int, int]:
    if profile.price_layout == "time_point":
        return _normalize_time_point_series(
            raw,
            name=profile.target_name,
            date_column=profile.price_date_column,
            time_column=profile.price_time_column,
            value_column=profile.price_value_column,
            available_at_column=profile.price_available_at_column,
        )
    return _normalize_business_series(raw, profile, name=profile.target_name)


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
    availability = combined[availability_columns].apply(
        lambda column: pd.to_datetime(column, errors="coerce")
    )
    combined["available_at"] = availability.max(axis=1)
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
        f"WHERE `province_name`=%s AND `forecast_time`>=%s "
        f"AND `create_time`<=%s{future_limit} "
        "ORDER BY `forecast_time`,`create_time`"
    )


def _daily_weather_profile(
    connection: Any,
    profile: RegionProfile,
    start: datetime,
    cutoff: datetime,
    metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    if profile.weather_daily_table is None:
        return {}
    frame = _timed_query(
        connection,
        label=f"weather_daily:{profile.weather_daily_table}",
        sql=(
            f"SELECT COUNT(*) rows_total,MIN(`forecast_date`) start_date,MAX(`forecast_date`) end_date "
            f"FROM `{profile.weather_daily_table}` WHERE `province_name`=%s "
            "AND `forecast_date`>=%s AND `create_time`<=%s"
        ),
        parameters=(profile.province_value, start.date(), cutoff),
        metrics=metrics,
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
        return pd.DataFrame(
            columns=[*profile.weather_columns, "available_at"],
            index=pd.DatetimeIndex([], name="timestamp"),
        )
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


def _write_json_atomic(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, default=str),
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _profile_cache_key(profile: RegionProfile) -> str:
    encoded = json.dumps(asdict(profile), sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


@dataclass(frozen=True)
class _CachedRegionData:
    generation: str
    cutoff: datetime
    target_path: Path
    actuals_path: Path | None
    forecasts_path: Path | None
    actual_names: tuple[str, ...]
    forecast_names: tuple[str, ...]


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _load_cached_region(root: Path, profile: RegionProfile) -> _CachedRegionData | None:
    pointer = _read_json(root / REGION_CACHE_POINTER)
    if pointer is None or pointer.get("format_version") != REGION_CACHE_FORMAT_VERSION:
        return None
    generation = str(pointer.get("generation") or "")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", generation):
        return None
    directory = root / "generations" / generation
    manifest = _read_json(directory / REGION_FETCH_MANIFEST)
    if (
        manifest is None
        or manifest.get("format_version") != REGION_CACHE_FORMAT_VERSION
        or manifest.get("profile_key") != _profile_cache_key(profile)
    ):
        return None
    files = manifest.get("files")
    if not isinstance(files, dict):
        return None

    def resolved(name: str) -> Path | None:
        filename = files.get(name)
        if not filename:
            return None
        if filename not in {
            "actual-realtime-price.parquet",
            "actual-factors.parquet",
            "forecast-factors.parquet",
        }:
            return None
        path = directory / filename
        try:
            return path if path.is_file() and path.stat().st_size > 0 else None
        except OSError:
            return None

    target_path = resolved("target")
    actuals_path = resolved("actuals")
    forecasts_path = resolved("forecasts")
    if target_path is None:
        return None
    if files.get("actuals") and actuals_path is None:
        return None
    if files.get("forecasts") and forecasts_path is None:
        return None
    try:
        cutoff = datetime.fromisoformat(str(manifest["cutoff"]))
    except (KeyError, ValueError):
        return None
    return _CachedRegionData(
        generation=generation,
        cutoff=cutoff,
        target_path=target_path,
        actuals_path=actuals_path,
        forecasts_path=forecasts_path,
        actual_names=tuple(str(value) for value in manifest.get("actual_names", [])),
        forecast_names=tuple(str(value) for value in manifest.get("forecast_names", [])),
    )


def _merge_incremental(existing: pd.DataFrame | None, refreshed: pd.DataFrame) -> pd.DataFrame:
    if existing is None or existing.empty:
        return refreshed.sort_values("timestamp", kind="stable").reset_index(drop=True)
    if refreshed.empty:
        return existing.sort_values("timestamp", kind="stable").reset_index(drop=True)
    old = existing.set_index("timestamp")
    new = refreshed.set_index("timestamp")
    combined = new.combine_first(old)
    if "available_at" in old or "available_at" in new:
        availability = pd.concat(
            [
                frame["available_at"]
                for frame in (old, new)
                if "available_at" in frame
            ],
            axis=1,
        ).max(axis=1)
        combined["available_at"] = availability
    return combined.sort_index().rename_axis("timestamp").reset_index()


def _series_groups(series: tuple[RegionalSeries, ...]) -> list[tuple[RegionalSeries, ...]]:
    grouped: dict[tuple[str, str, bool], list[RegionalSeries]] = {}
    for item in series:
        key = (item.table, item.value_column, item.type_value is None)
        grouped.setdefault(key, []).append(item)
    return [tuple(items) for items in grouped.values()]


def _timed_query(
    connection: Any,
    *,
    label: str,
    sql: str,
    parameters: tuple[Any, ...],
    metrics: list[dict[str, Any]],
) -> pd.DataFrame:
    started = perf_counter()
    frame = _query_frame(connection, sql, parameters)
    metrics.append(
        {
            "label": label,
            "duration_ms": round((perf_counter() - started) * 1000, 1),
            "rows": len(frame),
        }
    )
    return frame


def fetch_region_price(
    profile: RegionProfile,
    *,
    output_directory: str | Path | None = None,
    progress: Callable[[int, str], None] | None = None,
    now: datetime | None = None,
    connection_factory: Callable[[DatabaseCredentials], Any] = _connect,
) -> RegionPriceFetch:
    """Fetch one full or incremental regional generation and publish it atomically."""

    fetched_at = now or datetime.now(UTC).astimezone()
    if fetched_at.tzinfo is None or fetched_at.utcoffset() is None:
        raise RegionSourceError("取数时间必须包含时区")
    local_cutoff = fetched_at.astimezone(ZoneInfo(profile.timezone)).replace(tzinfo=None)
    history_start = datetime.combine(profile.history_start_date, time.min)
    base = Path(output_directory) if output_directory is not None else application_data_directory() / "region-data"
    root = (base / profile.region_id).resolve()
    cached = _load_cached_region(root, profile)
    old_target: pd.DataFrame | None = None
    old_actuals: pd.DataFrame | None = None
    old_forecasts: pd.DataFrame | None = None
    if cached is not None:
        try:
            old_target = pd.read_parquet(cached.target_path)
            old_actuals = pd.read_parquet(cached.actuals_path) if cached.actuals_path else None
            old_forecasts = pd.read_parquet(cached.forecasts_path) if cached.forecasts_path else None
        except (OSError, ValueError, ImportError):
            cached = None
            old_target = old_actuals = old_forecasts = None
    fetch_mode = "incremental" if cached is not None else "full"
    query_start = (
        max(history_start, cached.cutoff - timedelta(days=profile.refresh_lookback_days))
        if cached is not None
        else history_start
    )
    metrics: list[dict[str, Any]] = []
    source_credentials: dict[str, DatabaseCredentials] = {}
    source_connections: dict[str, Any] = {}

    def connection_for(prefix: str) -> tuple[DatabaseCredentials, Any]:
        credentials = source_credentials.get(prefix)
        if credentials is None:
            credentials = credentials_for(profile, prefix)
            source_credentials[prefix] = credentials
        connection = source_connections.get(prefix)
        if connection is None:
            connection = connection_factory(credentials)
            source_connections[prefix] = connection
        return credentials, connection

    price_prefix = profile.price_credential_prefix or profile.credential_prefix
    market_prefix = profile.market_credential_prefix or price_prefix
    price_credentials, price_connection = connection_for(price_prefix)
    if progress:
        progress(10, "正在增量更新所选地区的数据" if cached else "正在连接所选地区的数据")
    try:
        raw = _timed_query(
            price_connection,
            label=f"price:{profile.price_table}",
            sql=_price_query(profile),
            parameters=_price_parameters(profile, start=query_start, cutoff=local_cutoff),
            metrics=metrics,
        )
        history_raw = (
            _timed_query(
                price_connection,
                label=f"price_history:{profile.price_history_table}",
                sql=_price_query(profile, profile.price_history_table),
                parameters=_price_parameters(profile, start=query_start, cutoff=local_cutoff),
                metrics=metrics,
            )
            if profile.price_history_table
            else pd.DataFrame()
        )
        grouped_rows: dict[str, list[tuple[RegionalSeries, pd.DataFrame]]] = {
            "actual": [],
            "forecast": [],
        }
        feature_connection = None
        if profile.actual_series or profile.forecast_series:
            _feature_credentials, feature_connection = connection_for(profile.credential_prefix)
        for group_name, configured, future in (
            ("actual", profile.actual_series, False),
            ("forecast", profile.forecast_series, True),
        ):
            for series_group in _series_groups(configured):
                query, parameters = _series_group_query(
                    profile,
                    series_group,
                    future=future,
                    start=query_start,
                    cutoff=local_cutoff,
                )
                first = series_group[0]
                type_values = [item.type_value for item in series_group if item.type_value is not None]
                label_types = ",".join(str(value) for value in type_values) or "all"
                group_raw = _timed_query(
                    feature_connection,
                    label=f"{group_name}:{first.table}:{label_types}",
                    sql=query,
                    parameters=parameters,
                    metrics=metrics,
                )
                for series in series_group:
                    if series.type_value is None:
                        series_raw = group_raw
                    else:
                        series_raw = group_raw.loc[
                            group_raw[profile.type_column] == series.type_value
                        ].copy()
                    grouped_rows[group_name].append((series, series_raw))

        market_rows: list[tuple[RegionalTimePointSeries, pd.DataFrame]] = []
        if profile.market_series:
            _market_credentials, market_connection = connection_for(market_prefix)
            for market_group in _market_series_groups(profile.market_series):
                query, parameters = _market_series_group_query(
                    market_group,
                    start=query_start,
                    cutoff=local_cutoff,
                )
                first = market_group[0]
                market_raw = _timed_query(
                    market_connection,
                    label=f"market:{first.table}",
                    sql=query,
                    parameters=parameters,
                    metrics=metrics,
                )
                for series in market_group:
                    market_rows.append(
                        (
                            series,
                            market_raw.loc[
                                market_raw[series.category_column] == series.category_value
                            ].copy(),
                        )
                    )
    except Exception as exc:
        raise RegionSourceError(f"{profile.label}电价或影响因素读取失败：{exc}") from exc
    finally:
        for connection in source_connections.values():
            try:
                connection.rollback()
            finally:
                connection.close()

    if progress:
        progress(45, "正在整理电价时间点")
    normalized, duplicate_rows, invalid_rows = _normalize_price_series(raw, profile)
    if not history_raw.empty:
        history, history_duplicates, history_invalid = _normalize_price_series(history_raw, profile)
        duplicate_rows += history_duplicates
        invalid_rows += history_invalid
        if not history.empty:
            expanded_history = _expand_hour_ending_series(
                history.set_index("timestamp"),
                profile.frequency,
            ).set_index("timestamp")
            normalized = normalized.set_index("timestamp").combine_first(expanded_history).reset_index()
    normalized = _merge_incremental(old_target, normalized)
    normalized = normalized.loc[
        (normalized["timestamp"] >= history_start) & (normalized["timestamp"] <= local_cutoff)
    ].sort_values("timestamp", kind="stable").reset_index(drop=True)
    if normalized.empty:
        raise RegionSourceError(f"{profile.label}当前没有可用的实际电价")

    weather_details: dict[str, Any] = {}
    series_details: dict[str, Any] = {"actual": {}, "forecast": {}}
    actual_frames: list[pd.DataFrame] = []
    forecast_frames: list[pd.DataFrame] = []
    for group_name, destination_frames in (
        ("actual", actual_frames),
        ("forecast", forecast_frames),
    ):
        for series, series_raw in grouped_rows[group_name]:
            frame, duplicates, invalid = _normalize_business_series(
                series_raw,
                profile,
                name=series.name,
            )
            if not frame.empty:
                destination_frames.append(frame)
            series_details[group_name][series.name] = {
                "table": series.table,
                "source_rows": len(series_raw),
                "usable_rows": len(frame),
                "start_at": frame["timestamp"].iloc[0].isoformat() if not frame.empty else "",
                "end_at": frame["timestamp"].iloc[-1].isoformat() if not frame.empty else "",
                "duplicate_timestamp_rows": duplicates,
                "invalid_rows": invalid,
            }

    for series, series_raw in market_rows:
        frame, duplicates, invalid = _normalize_time_point_series(
            series_raw,
            name=series.name,
            date_column=series.date_column,
            time_column=series.time_column,
            value_column=series.value_column,
            available_at_column=series.available_at_column,
        )
        if not frame.empty:
            actual_frames.append(frame)
        series_details["actual"][series.name] = {
            "table": series.table,
            "source_rows": len(series_raw),
            "usable_rows": len(frame),
            "start_at": frame["timestamp"].iloc[0].isoformat() if not frame.empty else "",
            "end_at": frame["timestamp"].iloc[-1].isoformat() if not frame.empty else "",
            "duplicate_timestamp_rows": duplicates,
            "invalid_rows": invalid,
            "category": series.category_value,
            "availability": series.available_at_column or "business_timestamp_assumption",
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
            era5 = _timed_query(
                analytics_connection,
                label=f"weather_actual:{profile.weather_actual_table}",
                sql=_weather_query(profile, str(profile.weather_actual_table), future=False),
                parameters=(profile.province_value, query_start, local_cutoff, local_cutoff),
                metrics=metrics,
            )
            gfs = _timed_query(
                analytics_connection,
                label=f"weather_forecast:{profile.weather_forecast_table}",
                sql=_weather_query(profile, str(profile.weather_forecast_table), future=True),
                parameters=(profile.province_value, query_start, local_cutoff),
                metrics=metrics,
            )
            daily_profile = _daily_weather_profile(
                analytics_connection,
                profile,
                query_start,
                local_cutoff,
                metrics,
            )
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
            "forecast_variables": [
                f"forecast_{column}" if column in weather_actual_names else column
                for column in forecast_source_names
            ],
            "daily_forecast": daily_profile,
        }

    combined_actuals = _merge_incremental(old_actuals, _combine_series_frames(actual_frames))
    combined_forecasts = _merge_incremental(old_forecasts, _combine_series_frames(forecast_frames))
    combined_actuals = combined_actuals.loc[combined_actuals["timestamp"] >= history_start]
    combined_forecasts = combined_forecasts.loc[combined_forecasts["timestamp"] >= history_start]
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

    generation = f"{local_cutoff:%Y%m%dT%H%M%S}-{uuid4().hex[:8]}"
    generations = root / "generations"
    staging = generations / f".staging-{uuid4().hex}"
    final = generations / generation
    files: dict[str, str | None] = {
        "target": "actual-realtime-price.parquet",
        "actuals": "actual-factors.parquet" if actual_names else None,
        "forecasts": "forecast-factors.parquet" if forecast_source_names else None,
    }
    try:
        staging.mkdir(parents=True, exist_ok=False)
        _write_parquet_atomic(normalized, staging / str(files["target"]))
        if files["actuals"]:
            _write_parquet_atomic(combined_actuals, staging / str(files["actuals"]))
        if files["forecasts"]:
            _write_parquet_atomic(combined_forecasts, staging / str(files["forecasts"]))
        _write_json_atomic(
            {
                "format_version": REGION_CACHE_FORMAT_VERSION,
                "profile_key": _profile_cache_key(profile),
                "generation": generation,
                "fetched_at": fetched_at.isoformat(),
                "cutoff": local_cutoff.isoformat(),
                "query_start_at": query_start.isoformat(),
                "fetch_mode": fetch_mode,
                "files": files,
                "actual_names": list(actual_names),
                "forecast_names": list(forecast_names),
                "query_metrics": metrics,
            },
            staging / REGION_FETCH_MANIFEST,
        )
        generations.mkdir(parents=True, exist_ok=True)
        os.replace(staging, final)
        _write_json_atomic(
            {"format_version": REGION_CACHE_FORMAT_VERSION, "generation": generation},
            root / REGION_CACHE_POINTER,
        )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    target_path = final / str(files["target"])
    actuals_path = final / str(files["actuals"]) if files["actuals"] else None
    forecasts_path = final / str(files["forecasts"]) if files["forecasts"] else None
    if progress:
        progress(100, "所选地区的数据已经增量更新" if cached else "所选地区的数据已经就绪")
    return RegionPriceFetch(
        profile=profile,
        path=target_path,
        fetched_at=fetched_at,
        row_count=len(normalized),
        start_at=normalized["timestamp"].iloc[0],
        end_at=normalized["timestamp"].iloc[-1],
        duplicate_timestamp_rows=duplicate_rows,
        invalid_rows=invalid_rows,
        database_name=price_credentials.database,
        actuals_path=actuals_path,
        forecasts_path=forecasts_path,
        actual_variable_names=actual_names,
        forecast_variable_names=forecast_names,
        series_details=series_details,
        weather_details=weather_details,
        fetch_mode=fetch_mode,
        query_start_at=query_start,
        query_metrics=tuple(metrics),
        cache_generation=generation,
    )


def fetch_regional_snapshot(
    profile: RegionProfile,
    *,
    as_of: datetime,
    output_directory: str | Path,
    progress: Callable[[int, str], None] | None = None,
    connection_factory: Callable[[DatabaseCredentials], Any] = _connect,
) -> RegionalSnapshot:
    """Freeze a historical database view without mutating the desktop cache pointer."""

    fetched = fetch_region_price(
        profile,
        output_directory=output_directory,
        progress=progress,
        now=as_of,
        connection_factory=connection_factory,
    )
    return RegionalSnapshot(
        profile=profile,
        as_of=as_of,
        path=fetched.path,
        actuals_path=fetched.actuals_path,
        forecasts_path=fetched.forecasts_path,
        source=fetched,
    )
