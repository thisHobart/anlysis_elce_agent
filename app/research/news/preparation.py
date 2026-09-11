"""Reproducible Shandong database profiling and collected-news preparation.

Credentials stay in environment variables. The database boundary exposes only a
small fixed query catalogue and starts every connection in read-only mode.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from zoneinfo import ZoneInfo

import pandas as pd

from app.research.news.contracts import CollectedNewsRecord
from app.research.news.normalization import NewsNormalizer

PROFILE_VERSION = "1.0.0"
SEARCH_CONFIG_VERSION = "1.0.0"
MARKET = "SHANDONG"
MARKET_TIMEZONE = "Asia/Shanghai"
INTERVAL_MINUTES = 15
CONTENT_SCOPES = frozenset({"full_text", "excerpt", "summary"})

PRICE_TABLES = {
    "day_ahead_actual_price": "t_data_province_days_cleared_price",
    "real_time_actual_price": "t_data_province_real_time_cleared_price",
    "real_time_actual_price_hourly": "t_data_province_real_time_cleared_price_hour",
}
FEATURE_TABLES = {
    "actual_load": "t_data_dr_province_actual_load",
    "actual_renewable_output": "t_data_dr_province_actual_new_energy_exert",
    "actual_generation": "t_data_dr_province_power_generation",
    "system_reserve": "t_data_dr_province_system_info",
    "generator_maintenance_plan": "t_data_sdc_days_ago_generator_unit_maintenance_plan",
    "actual_contact_transmission": "t_data_dr_province_contact_transmission",
    "forecast_contact_transmission": "t_data_sdc_province_days_forecast_contact_transmission",
    "coal_fired_plan": "t_data_sdc_province_coal_fired_plan",
}
ANALYTICS_TABLES = {
    "day_ahead_price_forecast": "t_data_province_days_forecase_production_price",
    "era5_hourly_weather": "t_spo_aiweather_province_hourly_forecast_ERA5",
    "gfs_hourly_weather": "t_spo_aiweather_province_hourly_forecast",
    "gfs_daily_weather": "t_spo_aiweather_province_daily_forecast",
    "coal_index": "t_direct_coal_index_data",
}


class PreparationError(RuntimeError):
    """Raised when a safe and reproducible preparation run cannot continue."""


@dataclass(frozen=True)
class MySQLCredentials:
    host: str
    port: int
    user: str
    password: str = field(repr=False)
    database: str

    @classmethod
    def from_env(cls, role: str) -> MySQLCredentials:
        prefix = f"VPP_SHANDONG_DB_{role.upper()}_"
        values = {key: os.getenv(prefix + key, "") for key in ("HOST", "PORT", "USER", "PASSWORD", "NAME")}
        missing = [key for key, value in values.items() if not value]
        if missing:
            raise PreparationError(f"缺少 {role} 数据库环境变量：{', '.join(prefix + key for key in missing)}")
        try:
            port = int(values["PORT"])
        except ValueError as exc:
            raise PreparationError(f"{prefix}PORT 必须是整数") from exc
        return cls(values["HOST"], port, values["USER"], values["PASSWORD"], values["NAME"])

    @property
    def safe_identity(self) -> str:
        return self.database


def _connect(credentials: MySQLCredentials):
    try:
        import pymysql
    except ImportError as exc:  # pragma: no cover
        raise PreparationError("缺少 PyMySQL；请安装项目声明的运行依赖") from exc
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


def business_timestamp(day: date | datetime | str, hour: int, minute: int) -> datetime:
    """Convert the source business clock, including hour=24, deterministically."""

    if not 0 <= hour <= 24 or not 0 <= minute < 60 or (hour == 24 and minute != 0):
        raise ValueError("business_date clock must be 00:00..24:00 and hour=24 requires minute=0")
    base = datetime.fromisoformat(str(day)[:10])
    return base + timedelta(hours=hour, minutes=minute)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "item"):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _query_frame(connection, sql: str, parameters: tuple[Any, ...] = ()) -> pd.DataFrame:
    """Execute one internally defined SELECT and return a bounded analytical frame."""

    if not re.match(r"^\s*SELECT\b", sql, flags=re.IGNORECASE):
        raise PreparationError("只读边界拒绝非 SELECT 语句")
    with connection.cursor() as cursor:
        cursor.execute(sql, parameters)
        rows = cursor.fetchall()
        columns = [item[0] for item in cursor.description]
    return pd.DataFrame(rows, columns=columns)


def _table_exists(connection, database: str, table: str) -> bool:
    frame = _query_frame(
        connection,
        "SELECT COUNT(*) AS n FROM information_schema.TABLES WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s",
        (database, table),
    )
    return bool(frame.iloc[0]["n"])


def _columns(connection, database: str, table: str) -> list[dict[str, Any]]:
    frame = _query_frame(
        connection,
        "SELECT COLUMN_NAME,COLUMN_TYPE,IS_NULLABLE,COLUMN_KEY,COLUMN_COMMENT "
        "FROM information_schema.COLUMNS WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s ORDER BY ORDINAL_POSITION",
        (database, table),
    )
    return [
        {
            "name": row.COLUMN_NAME,
            "type": row.COLUMN_TYPE,
            "nullable": row.IS_NULLABLE == "YES",
            "key": row.COLUMN_KEY,
            "comment": row.COLUMN_COMMENT,
        }
        for row in frame.itertuples(index=False)
    ]


def _database_catalog(connection, database: str) -> dict[str, Any]:
    databases = _query_frame(
        connection,
        "SELECT SCHEMA_NAME FROM information_schema.SCHEMATA ORDER BY SCHEMA_NAME",
    )
    tables = _query_frame(
        connection,
        "SELECT TABLE_NAME,TABLE_TYPE,TABLE_ROWS FROM information_schema.TABLES "
        "WHERE TABLE_SCHEMA=%s ORDER BY TABLE_NAME",
        (database,),
    )
    keywords = re.compile(
        r"price|load|weather|power|wind|solar|photo|coal|unit|province|market|trade|forecast|forecase|transmission",
        re.IGNORECASE,
    )
    table_rows = [
        {"table": row.TABLE_NAME, "type": row.TABLE_TYPE, "estimated_rows": row.TABLE_ROWS}
        for row in tables.itertuples(index=False)
    ]
    return {
        "database": database,
        "accessible_databases": databases.SCHEMA_NAME.tolist(),
        "table_count": len(table_rows),
        "tables": table_rows,
        "discovered_candidate_tables": [item["table"] for item in table_rows if keywords.search(item["table"])],
    }


_BUSINESS_TS_SQL = "DATE_ADD(DATE(ts_day), INTERVAL (hour*60+min) MINUTE)"


def _local_cutoff(as_of: datetime | None) -> datetime | None:
    if as_of is None:
        return None
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise PreparationError("database as_of requires a timezone offset")
    return as_of.astimezone(ZoneInfo(MARKET_TIMEZONE)).replace(tzinfo=None)


def _profile_business_table(connection, table: str, *, cutoff: datetime | None = None) -> dict[str, Any]:
    where = ""
    parameters: tuple[Any, ...] = ()
    if cutoff is not None:
        where = f" WHERE create_time<=%s AND {_BUSINESS_TS_SQL}<=%s"
        parameters = (cutoff, cutoff)
    frame = _query_frame(
        connection,
        f"SELECT COUNT(*) rows_total,COUNT(DISTINCT {_BUSINESS_TS_SQL}) distinct_times,"
        f"MIN({_BUSINESS_TS_SQL}) min_time,MAX({_BUSINESS_TS_SQL}) max_time,"
        "SUM(ts_day IS NULL) null_time,SUM(hour=24) hour_24_rows,"
        "SUM(hour<0 OR hour>24 OR min<0 OR min>=60 OR (hour=24 AND min<>0)) invalid_clock_rows,"
        "MIN(create_time) min_created_at,MAX(create_time) max_created_at FROM `" + table + "`" + where,
        parameters,
    )
    return _jsonable(frame.iloc[0].to_dict())


def _profile_special_table(connection, table: str, *, cutoff: datetime | None = None) -> dict[str, Any]:
    schema = _query_frame(
        connection,
        "SELECT COLUMN_NAME,DATA_TYPE FROM information_schema.COLUMNS "
        "WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=%s ORDER BY ORDINAL_POSITION",
        (table,),
    )
    names = set(schema.COLUMN_NAME)
    if {"ts_day", "hour", "min"}.issubset(names):
        return _profile_business_table(connection, table, cutoff=cutoff)
    time_col = next(
        (name for name in ("forecast_time", "forecast_date", "publish_time", "p_date", "ts_day") if name in names),
        None,
    )
    if time_col is None:
        frame = _query_frame(connection, f"SELECT COUNT(*) rows_total FROM `{table}`")
        return _jsonable(frame.iloc[0].to_dict())
    created = (
        "MIN(create_time) min_created_at,MAX(create_time) max_created_at"
        if "create_time" in names
        else "NULL min_created_at,NULL max_created_at"
    )
    filters: list[str] = []
    parameters: list[Any] = []
    if cutoff is not None:
        filters.append(f"`{time_col}`<=%s")
        parameters.append(cutoff)
        if "create_time" in names:
            filters.append("create_time<=%s")
            parameters.append(cutoff)
    where = " WHERE " + " AND ".join(filters) if filters else ""
    frame = _query_frame(
        connection,
        f"SELECT COUNT(*) rows_total,COUNT(DISTINCT `{time_col}`) distinct_times,"
        f"MIN(`{time_col}`) min_time,MAX(`{time_col}`) max_time,SUM(`{time_col}` IS NULL) null_time,"
        f"{created} FROM `{table}`{where}",
        tuple(parameters),
    )
    return _jsonable(frame.iloc[0].to_dict())


def profile_databases(
    feature: MySQLCredentials,
    analytics: MySQLCredentials,
    *,
    as_of: datetime | None = None,
) -> dict[str, Any]:
    """Profile the allow-listed tables without exporting raw schema or row samples."""

    cutoff = _local_cutoff(as_of)
    result: dict[str, Any] = {
        "profile_version": PROFILE_VERSION,
        "market": MARKET,
        "market_timezone": MARKET_TIMEZONE,
        "as_of": _jsonable(as_of.astimezone(UTC)) if as_of is not None else None,
        "connections": {"feature": feature.safe_identity, "analytics": analytics.safe_identity},
        "catalogs": {},
        "datasets": [],
    }
    for role, credentials, datasets in (
        ("feature", feature, {**PRICE_TABLES, **FEATURE_TABLES}),
        ("analytics", analytics, ANALYTICS_TABLES),
    ):
        connection = _connect(credentials)
        try:
            result["catalogs"][role] = _database_catalog(connection, credentials.database)
            for purpose, table in datasets.items():
                exists = _table_exists(connection, credentials.database, table)
                item: dict[str, Any] = {"role": role, "purpose": purpose, "table": table, "exists": exists}
                if exists:
                    item["columns"] = _columns(connection, credentials.database, table)
                    item["profile"] = _profile_special_table(connection, table, cutoff=cutoff)
                result["datasets"].append(item)
        finally:
            connection.rollback()
            connection.close()
    actual = [
        item
        for item in result["datasets"]
        if item["purpose"] in {"day_ahead_actual_price", "real_time_actual_price"}
        and item["exists"]
        and item["profile"]["rows_total"] > 0
    ]
    result["actual_price_conclusion"] = {
        "exists": bool(actual),
        "tables": [item["table"] for item in actual],
        "forecast_not_used_as_actual": True,
    }
    return result


def load_p1_frames(
    feature: MySQLCredentials,
    analytics: MySQLCredentials,
    *,
    as_of: datetime | None = None,
) -> dict[str, pd.DataFrame]:
    """Load only the columns needed for deterministic P1 profiling."""

    cutoff = _local_cutoff(as_of)
    frames: dict[str, pd.DataFrame] = {}
    connection = _connect(feature)
    try:
        specs = {
            "real_time_price": (PRICE_TABLES["real_time_actual_price"], "type=1"),
            "day_ahead_price": (PRICE_TABLES["day_ahead_actual_price"], "type=1"),
            "load": (FEATURE_TABLES["actual_load"], "type=11"),
            "wind": (FEATURE_TABLES["actual_renewable_output"], "type=4"),
            "solar": (FEATURE_TABLES["actual_renewable_output"], "type=12"),
            "generation": (FEATURE_TABLES["actual_generation"], "type=1"),
            "positive_reserve": (FEATURE_TABLES["system_reserve"], "type=2"),
            "negative_reserve": (FEATURE_TABLES["system_reserve"], "type=3"),
        }
        for name, (table, predicate) in specs.items():
            cutoff_sql = f" AND create_time<=%s AND {_BUSINESS_TS_SQL}<=%s" if cutoff is not None else ""
            parameters = (cutoff, cutoff) if cutoff is not None else ()
            frames[name] = _query_frame(
                connection,
                f"SELECT {_BUSINESS_TS_SQL} timestamp,quantity value,create_time available_at "
                f"FROM `{table}` WHERE province='山东省' AND {predicate}{cutoff_sql} ORDER BY ts_day,hour,min",
                parameters,
            )
        maintenance_cutoff = " AND create_time<=%s AND bgn_time<=%s" if cutoff is not None else ""
        frames["maintenance"] = _query_frame(
            connection,
            "SELECT p_date,bgn_time,end_time,dev_name,rep_type,votage_level,create_time available_at "
            "FROM t_data_sdc_days_ago_generator_unit_maintenance_plan "
            f"WHERE province='山东省'{maintenance_cutoff} ORDER BY p_date,bgn_time,dev_name",
            (cutoff, cutoff) if cutoff is not None else (),
        )
    finally:
        connection.rollback()
        connection.close()
    connection = _connect(analytics)
    try:
        weather_cutoff = " AND create_time<=%s AND forecast_time<=%s" if cutoff is not None else ""
        frames["weather"] = _query_frame(
            connection,
            "SELECT forecast_time timestamp,temperature,wind_speed_eighty,hour_precipitation,"
            "solar_radiation,create_time available_at FROM t_spo_aiweather_province_hourly_forecast_ERA5 "
            f"WHERE province_name='山东省'{weather_cutoff} ORDER BY forecast_time",
            (cutoff, cutoff) if cutoff is not None else (),
        )
        coal_cutoff = " WHERE create_time<=%s AND publish_time<=%s" if cutoff is not None else ""
        frames["coal"] = _query_frame(
            connection,
            "SELECT publish_time timestamp,num value,create_time available_at,type_name,heat_value "
            f"FROM t_direct_coal_index_data{coal_cutoff} ORDER BY publish_time",
            (cutoff, cutoff) if cutoff is not None else (),
        )
    finally:
        connection.rollback()
        connection.close()
    return frames


def _series_profile(frame: pd.DataFrame, *, expected_per_day: int) -> dict[str, Any]:
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    duplicate_rows = int(data.duplicated("timestamp", keep=False).sum())
    distinct = data.drop_duplicates("timestamp")
    intervals = distinct.timestamp.sort_values().diff().dt.total_seconds().div(60).value_counts().head(5)
    daily_counts = distinct.groupby(distinct.timestamp.dt.date).size()
    return _jsonable(
        {
            "rows": len(data),
            "distinct_times": distinct.timestamp.nunique(),
            "start_at": distinct.timestamp.min(),
            "end_at": distinct.timestamp.max(),
            "null_values": data.value.isna().sum(),
            "duplicate_time_rows": duplicate_rows,
            "min": data.value.min(),
            "max": data.value.max(),
            "negative_rows": (data.value < 0).sum(),
            "dominant_intervals_minutes": {str(int(key)): int(value) for key, value in intervals.items()},
            "complete_day_rate": float((daily_counts >= expected_per_day).mean()) if len(daily_counts) else 0.0,
        }
    )


def _daily_anomalies(
    frame: pd.DataFrame,
    *,
    expected_per_day: int,
    high: bool = True,
    low: bool = True,
    negative: bool = False,
    limit: int = 5,
) -> list[dict[str, Any]]:
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    data = data.dropna(subset=["timestamp", "value"]).drop_duplicates("timestamp", keep="last")
    grouped = data.groupby(data.timestamp.dt.date).value
    daily = grouped.agg(["count", "min", "max", "mean"])
    daily["negative_intervals"] = grouped.apply(lambda values: int((values < 0).sum()))
    daily = daily[daily["count"] >= max(1, int(expected_per_day * 0.8))]
    median = daily["mean"].median()
    mad = (daily["mean"] - median).abs().median()
    daily["robust_z"] = 0.0 if not mad else (daily["mean"] - median) / (1.4826 * mad)
    selections: list[tuple[str, pd.DataFrame]] = []
    if high:
        selections.append(("high", daily.nlargest(limit, "robust_z")))
    if low:
        selections.append(("low", daily.nsmallest(limit, "robust_z")))
    if negative:
        selections.append(("negative", daily[daily.negative_intervals > 0].nlargest(limit, "negative_intervals")))
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for kind, selected in selections:
        for day, row in selected.iterrows():
            key = (kind, day.isoformat())
            if key in seen:
                continue
            seen.add(key)
            rows.append({"kind": kind, "date": day.isoformat(), **_jsonable(row.to_dict())})
    return rows


def _coal_profile_and_anomalies(frame: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    key = ["timestamp", "type_name", "heat_value"]
    duplicates = int(data.duplicated(key, keep=False).sum())
    clean = data.dropna(subset=["timestamp", "value"]).drop_duplicates(key, keep="last")
    clean["series"] = clean.type_name.fillna("").astype(str) + "|" + clean.heat_value.fillna("").astype(str)
    clean = clean.sort_values(["series", "timestamp"])
    clean["previous_value"] = clean.groupby("series").value.shift()
    clean["absolute_change"] = clean.value - clean.previous_value
    changes = clean.dropna(subset=["previous_value"]).copy()
    changes["absolute_change_rank"] = changes.absolute_change.abs()
    changes = changes.nlargest(10, "absolute_change_rank")
    anomalies = [
        {
            "kind": "change",
            "date": row.timestamp.date().isoformat(),
            "series": row.series,
            "previous_value": float(row.previous_value),
            "value": float(row.value),
            "absolute_change": float(row.absolute_change),
        }
        for row in changes.itertuples(index=False)
    ]
    profile = {
        "rows": len(data),
        "distinct_times": int(clean.timestamp.nunique()),
        "distinct_series": int(clean.series.nunique()),
        "start_at": _jsonable(clean.timestamp.min()),
        "end_at": _jsonable(clean.timestamp.max()),
        "null_values": int(data.value.isna().sum()),
        "duplicate_composite_key_rows": duplicates,
        "min": _jsonable(clean.value.min()),
        "max": _jsonable(clean.value.max()),
        "anomaly_sample_is_small": clean.timestamp.nunique() < 12,
    }
    return profile, anomalies


def _maintenance_summary(frame: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = frame.copy()
    for column in ("p_date", "bgn_time", "end_time", "available_at"):
        data[column] = pd.to_datetime(data[column], errors="coerce")
    invalid = data.bgn_time.notna() & data.end_time.notna() & (data.end_time < data.bgn_time)
    starts = data.dropna(subset=["bgn_time"]).groupby(data.bgn_time.dt.date).size().nlargest(10)
    windows = [
        {"kind": "planned_start_cluster", "date": day.isoformat(), "plan_count": int(count)}
        for day, count in starts.items()
    ]
    profile = {
        "rows": len(data),
        "distinct_business_dates": int(data.p_date.nunique()),
        "distinct_assets": int(data.dev_name.nunique()),
        "start_at": _jsonable(data.bgn_time.min()),
        "end_at": _jsonable(data.end_time.max()),
        "missing_start_rows": int(data.bgn_time.isna().sum()),
        "missing_end_rows": int(data.end_time.isna().sum()),
        "invalid_interval_rows": int(invalid.sum()),
    }
    return profile, windows


def build_p1_summary(frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    profiles = {
        name: _series_profile(frame, expected_per_day=96)
        for name, frame in frames.items()
        if name not in {"weather", "coal", "maintenance"}
    }
    profiles["coal"], coal_anomalies = _coal_profile_and_anomalies(frames["coal"])
    profiles["maintenance"], maintenance_anomalies = _maintenance_summary(frames["maintenance"])
    weather = frames["weather"].copy()
    weather["timestamp"] = pd.to_datetime(weather["timestamp"])
    weather_metrics: dict[str, Any] = {}
    anomalies: dict[str, Any] = {}
    for column, label in (
        ("temperature", "temperature"),
        ("wind_speed_eighty", "weather_wind"),
        ("hour_precipitation", "precipitation"),
        ("solar_radiation", "solar_radiation"),
    ):
        series = weather[["timestamp", column, "available_at"]].rename(columns={column: "value"})
        weather_metrics[label] = _series_profile(series, expected_per_day=24)
        anomalies[label] = _daily_anomalies(series, expected_per_day=24, negative=False)
    profiles.update(weather_metrics)
    for name in ("real_time_price", "day_ahead_price"):
        anomalies[name] = _daily_anomalies(frames[name], expected_per_day=96, negative=True)
    for name in ("load", "wind", "solar", "generation", "positive_reserve", "negative_reserve"):
        anomalies[name] = _daily_anomalies(frames[name], expected_per_day=96)
    anomalies["coal"] = coal_anomalies
    anomalies["maintenance"] = maintenance_anomalies
    price = profiles["real_time_price"]
    return {
        "summary_version": PROFILE_VERSION,
        "market": MARKET,
        "region": "山东省",
        "market_timezone": MARKET_TIMEZONE,
        "settlement_interval_minutes": INTERVAL_MINUTES,
        "primary_actual_price": "实时省级出清电价",
        "study_start_at": price["start_at"],
        "study_end_at": price["end_at"],
        "series_profiles": profiles,
        "anomaly_windows": anomalies,
        "quality_notes": [
            "hour=24 已确定性转换为次日 00:00",
            "异常日要求至少达到预期日内点数的 80%，避免把残缺日误判为异常",
            "create_time 仅作为数据库记录可获得时间画像；不能自动等同于市场首次公开时间",
        ],
    }


def _query_id(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "qry_" + hashlib.sha256(raw).hexdigest()[:16]


def build_search_plan(summary: dict[str, Any], *, as_of: datetime) -> dict[str, Any]:
    start = str(summary["study_start_at"])[:10]
    end = str(summary["study_end_at"])[:10]
    base = [
        ("市场与政策", "现货市场 规则 新能源入市 容量电价 辅助服务", "policy_long_horizon", "market_rules"),
        ("发电供应", "机组 停运 跳闸 故障 检修 恢复 并网", "generation_outage", "generation"),
        ("输电与外来电", "银东直流 扎青直流 特高压 外来电 检修 断面", "transmission_constraint", "transmission"),
        ("燃料与成本", "动力煤 电厂存煤 港口库存 封航 进口煤 碳价", "fuel_supply_change", "coal"),
        ("负荷与需求", "统调负荷 创新高 高温 寒潮 有序用电 需求响应", "demand_shock", "load"),
        ("新能源出力", "风电 光伏 出力 限电 弃风 弃光 储能", "renewable_supply_change", "renewables"),
    ]
    queries: list[dict[str, Any]] = []
    for topic, terms, event_type, trigger in base:
        item = {
            "layer": "continuous",
            "topic": topic,
            "keywords": f"山东 {terms}",
            "region_terms": ["山东", "山东省", "山东电网"],
            "start_date": start,
            "end_date": end,
            "trigger": trigger,
            "expected_event_type": event_type,
        }
        item["query_id"] = _query_id(item)
        queries.append(item)
    trigger_specs = [
        ("real_time_price", "实时电价 尖峰 负电价", "market_price_context"),
        ("load", "用电负荷 高温 寒潮", "demand_shock"),
        ("generation", "机组 出力 检修 停运", "generation_outage"),
        ("wind", "风电 出力 大风", "renewable_supply_change"),
        ("solar", "光伏 出力 云量", "renewable_supply_change"),
        ("temperature", "寒潮 高温 电力保供", "demand_shock"),
        ("precipitation", "暴雨 洪水 电网", "extreme_weather"),
    ]
    study_start = date.fromisoformat(start)
    study_end = date.fromisoformat(end)
    candidates: list[tuple[str, str, str, str]] = []
    for variable, terms, event_type in trigger_specs:
        desired_kinds = (
            ("high", "negative")
            if variable == "real_time_price"
            else (("high",) if variable == "precipitation" else ("high", "low"))
        )
        anomalies = summary["anomaly_windows"].get(variable, [])
        for kind in desired_kinds:
            anomaly = next(
                (
                    item
                    for item in anomalies
                    if item["kind"] == kind
                    and study_start <= date.fromisoformat(item["date"]) <= study_end
                ),
                None,
            )
            if anomaly is not None:
                candidates.append((anomaly["date"], variable, terms, event_type))
    for variable, terms, event_type in (
        ("coal", "动力煤 煤价 库存 供应 运输", "fuel_supply_change"),
        ("maintenance", "机组 设备 检修 停运 恢复", "generation_outage"),
    ):
        added = 0
        for anomaly in summary["anomaly_windows"].get(variable, []):
            anomaly_date = date.fromisoformat(anomaly["date"])
            if study_start <= anomaly_date <= study_end:
                candidates.append((anomaly["date"], variable, terms, event_type))
                added += 1
                if added == 2:
                    break
    seen_days: set[tuple[str, str]] = set()
    for day_text, variable, terms, event_type in candidates:
        key = (day_text, event_type)
        if key in seen_days:
            continue
        seen_days.add(key)
        day = date.fromisoformat(day_text)
        item = {
            "layer": "anomaly_window",
            "topic": f"{variable} 异常窗口",
            "keywords": f"山东 {terms} {day_text}",
            "region_terms": ["山东", "山东省", "山东电网"],
            "start_date": (day - timedelta(days=3)).isoformat(),
            "end_date": (day + timedelta(days=3)).isoformat(),
            "trigger": f"{variable}:{day_text}",
            "expected_event_type": event_type,
        }
        item["query_id"] = _query_id(item)
        queries.append(item)
    return {
        "search_config_version": SEARCH_CONFIG_VERSION,
        "market": MARKET,
        "as_of": as_of.astimezone(UTC).isoformat(),
        "source_priority": [
            "government_regulator",
            "grid_market_operator",
            "meteorological",
            "company_official",
            "authoritative_media",
            "industry_media",
            "aggregator",
        ],
        "queries": queries,
    }


def canonical_url(value: str) -> str:
    parts = urlsplit(value.strip())
    query = [
        (key, item)
        for key, item in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith("utm_")
    ]
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, urlencode(query), ""))


def _content_hash(title: str, content: str) -> str:
    normalized = " ".join((title + "\n" + content).split()).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def prepare_news_snapshot(source: Path, *, output: Path, query_ids: set[str]) -> dict[str, Any]:
    """Validate, quarantine, deduplicate and normalize collected search results."""

    raw_dir = output / "raw_news"
    raw_dir.mkdir(parents=True, exist_ok=True)
    valid: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                required = {
                    "source_name",
                    "source_url",
                    "source_type",
                    "title",
                    "content",
                    "published_at",
                    "fetched_at",
                    "search_query_id",
                }
                missing = sorted(key for key in required if not item.get(key))
                if missing:
                    raise ValueError("missing fields: " + ", ".join(missing))
                if item["search_query_id"] not in query_ids:
                    raise ValueError("unknown search_query_id")
                published = datetime.fromisoformat(item["published_at"])
                fetched = datetime.fromisoformat(item["fetched_at"])
                if published.tzinfo is None or fetched.tzinfo is None:
                    raise ValueError("published_at and fetched_at require timezone offsets")
                if fetched < published:
                    raise ValueError("fetched_at cannot precede published_at")
                content_scope = item.get("content_scope", "excerpt")
                if content_scope not in CONTENT_SCOPES:
                    raise ValueError(f"unsupported content_scope: {content_scope}")
                item["content_scope"] = content_scope
                item["source_url"] = canonical_url(item["source_url"])
                item["content_hash"] = _content_hash(item["title"], item["content"])
                valid.append(item)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                quarantine.append({"line_number": line_number, "reason": str(exc)})
    source_order = [
        "government_regulator",
        "grid_market_operator",
        "meteorological",
        "company_official",
        "authoritative_media",
        "industry_media",
        "aggregator",
    ]
    priority = {name: index for index, name in enumerate(source_order)}
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    duplicates = 0
    for item in sorted(valid, key=lambda row: (priority.get(row["source_type"], 99), row["published_at"])):
        key = (item["source_url"], item["content_hash"])
        if key in unique:
            duplicates += 1
            continue
        unique[key] = item
    normalized: list[CollectedNewsRecord] = []
    for item in unique.values():
        raw_name = item["content_hash"] + ".json"
        raw_path = raw_dir / raw_name
        raw_payload = {
            "snapshot_kind": "full_text_record"
            if item["content_scope"] == "full_text"
            else "curated_text_record",
            **item,
        }
        raw_path.write_text(json.dumps(raw_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        normalized.append(
            CollectedNewsRecord(
                source_name=item["source_name"],
                source_document_id=item.get("source_document_id")
                or hashlib.sha256(item["source_url"].encode()).hexdigest()[:24],
                source_ref=item["source_url"],
                version=int(item.get("version", 1)),
                title=item["title"],
                body=item["content"],
                published_at=datetime.fromisoformat(item["published_at"]),
                collected_at=datetime.fromisoformat(item["fetched_at"]),
                updated_at=datetime.fromisoformat(item["updated_at"]) if item.get("updated_at") else None,
                language=item.get("language", "zh-CN"),
                market_tags=(MARKET,),
                metadata={
                    "source_type": item["source_type"],
                    "search_query_id": item["search_query_id"],
                    "region_tags": item.get("region_tags", ["山东省"]),
                    "fetched_at": item["fetched_at"],
                    "raw_snapshot_path": f"raw_news/{raw_name}",
                    "published_at_precision": item.get("published_at_precision", "instant"),
                    "content_scope": item["content_scope"],
                    "raw_snapshot_kind": raw_payload["snapshot_kind"],
                },
            )
        )
    documents = NewsNormalizer().normalize_many(normalized)
    with (output / "normalized_news.jsonl").open("w", encoding="utf-8") as stream:
        for document in documents:
            stream.write(document.model_dump_json() + "\n")
    with (output / "collected_news.jsonl").open("w", encoding="utf-8") as stream:
        for record in normalized:
            stream.write(record.model_dump_json() + "\n")
    with (output / "quarantined_news.jsonl").open("w", encoding="utf-8") as stream:
        for item in quarantine:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    scope_distribution = pd.Series([item["content_scope"] for item in unique.values()]).value_counts().to_dict()
    return {
        "search_result_count": len(valid),
        "normalized_news_count": len(documents),
        "duplicate_count": duplicates,
        "input_quarantine_count": len(quarantine),
        "source_distribution": pd.Series([item["source_type"] for item in unique.values()]).value_counts().to_dict(),
        "query_result_counts": pd.Series([item["search_query_id"] for item in unique.values()]).value_counts().to_dict(),
        "content_scope_distribution": scope_distribution,
        "full_text_count": int(scope_distribution.get("full_text", 0)),
    }


def apply_search_execution(
    plan: dict[str, Any],
    *,
    imported_result_counts: dict[str, int],
    execution_log: Path | None = None,
    import_service: str | None = None,
) -> dict[str, Any]:
    """Attach execution evidence without claiming that an unlogged query ran."""

    query_ids = {item["query_id"] for item in plan["queries"]}
    executions: dict[str, dict[str, Any]] = {}
    if execution_log is not None:
        with execution_log.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    query_id = item["query_id"]
                    if query_id not in query_ids:
                        raise ValueError("unknown query_id")
                    if query_id in executions:
                        raise ValueError("duplicate query execution")
                    executed_at = datetime.fromisoformat(item["executed_at"])
                    if executed_at.tzinfo is None:
                        raise ValueError("executed_at requires a timezone offset")
                    returned = int(item["returned_result_count"])
                    if returned < 0:
                        raise ValueError("returned_result_count must not be negative")
                    service = str(item["search_service"]).strip()
                    if not service:
                        raise ValueError("search_service is required")
                    executions[query_id] = {
                        "status": "executed",
                        "executed_at": executed_at.astimezone(UTC).isoformat(),
                        "search_service": service,
                        "returned_result_count": returned,
                    }
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    raise PreparationError(f"搜索执行日志第 {line_number} 行无效：{exc}") from exc
    for query in plan["queries"]:
        query_id = query["query_id"]
        imported = int(imported_result_counts.get(query_id, 0))
        evidence = executions.get(query_id)
        if evidence is not None:
            query.update(evidence)
        elif imported:
            query.update(
                {
                    "status": "results_imported_without_execution_log",
                    "executed_at": None,
                    "search_service": import_service,
                    "returned_result_count": None,
                }
            )
        else:
            query.update(
                {
                    "status": "not_run",
                    "executed_at": None,
                    "search_service": None,
                    "returned_result_count": None,
                }
            )
        query["imported_result_count"] = imported
    plan["execution_summary"] = {
        "executed": sum(item["status"] == "executed" for item in plan["queries"]),
        "results_imported_without_execution_log": sum(
            item["status"] == "results_imported_without_execution_log" for item in plan["queries"]
        ),
        "not_run": sum(item["status"] == "not_run" for item in plan["queries"]),
    }
    return plan


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
