"""Run read-only EXPLAIN diagnostics for one configured regional data source."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, time
from time import perf_counter
from typing import Any
from zoneinfo import ZoneInfo

from app.research.data.sources.regions import (
    RegionProfile,
    RegionSourceError,
    _connect,
    _price_parameters,
    _price_query,
    _series_group_query,
    _series_groups,
    _weather_query,
    credentials_for,
    load_region_profiles,
)


def _explain(connection: Any, label: str, sql: str, parameters: tuple[Any, ...]) -> dict[str, Any]:
    started = perf_counter()
    with connection.cursor() as cursor:
        cursor.execute(f"EXPLAIN {sql}", parameters)
        columns = [str(item[0]) for item in cursor.description]
        rows = [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]
    return {
        "label": label,
        "duration_ms": round((perf_counter() - started) * 1000, 1),
        "plan": rows,
    }


def _feature_queries(
    profile: RegionProfile,
    start: datetime,
    cutoff: datetime,
) -> list[tuple[str, str, tuple[Any, ...]]]:
    queries = [
        (
            f"price:{profile.price_table}",
            _price_query(profile),
            _price_parameters(profile, start=start, cutoff=cutoff),
        )
    ]
    if profile.price_history_table:
        queries.append(
            (
                f"price_history:{profile.price_history_table}",
                _price_query(profile, profile.price_history_table),
                _price_parameters(profile, start=start, cutoff=cutoff),
            )
        )
    for group_name, configured, future in (
        ("actual", profile.actual_series, False),
        ("forecast", profile.forecast_series, True),
    ):
        for group in _series_groups(configured):
            sql, parameters = _series_group_query(
                profile,
                group,
                future=future,
                start=start,
                cutoff=cutoff,
            )
            types = ",".join(str(item.type_value) for item in group if item.type_value is not None) or "all"
            queries.append((f"{group_name}:{group[0].table}:{types}", sql, parameters))
    return queries


def _index_suggestions(profile: RegionProfile) -> list[dict[str, Any]]:
    suggestions: dict[str, list[str]] = {}
    suggestions[profile.price_table] = [
        profile.province_column,
        profile.type_column,
        profile.date_column,
        profile.hour_column,
        profile.minute_column,
        profile.available_at_column,
    ]
    if profile.price_history_table:
        suggestions[profile.price_history_table] = list(suggestions[profile.price_table])
    for series in (*profile.actual_series, *profile.forecast_series):
        columns = [profile.province_column]
        if series.type_value is not None:
            columns.append(profile.type_column)
        columns.extend(
            (
                profile.date_column,
                profile.hour_column,
                profile.minute_column,
                profile.available_at_column,
            )
        )
        suggestions.setdefault(series.table, columns)
    for table in (profile.weather_actual_table, profile.weather_forecast_table):
        if table:
            suggestions[table] = ["province_name", "forecast_time", "create_time"]
    if profile.weather_daily_table:
        suggestions[profile.weather_daily_table] = ["province_name", "forecast_date", "create_time"]
    return [{"table": table, "candidate_columns": columns} for table, columns in suggestions.items()]


def diagnose(profile: RegionProfile) -> dict[str, Any]:
    cutoff = datetime.now(UTC).astimezone(ZoneInfo(profile.timezone)).replace(tzinfo=None)
    start = datetime.combine(profile.history_start_date, time.min)
    feature_connection = _connect(credentials_for(profile))
    try:
        plans = [
            _explain(feature_connection, label, sql, parameters)
            for label, sql, parameters in _feature_queries(profile, start, cutoff)
        ]
    finally:
        try:
            feature_connection.rollback()
        finally:
            feature_connection.close()
    if profile.analytics_credential_prefix and profile.weather_actual_table and profile.weather_forecast_table:
        analytics_connection = _connect(credentials_for(profile, profile.analytics_credential_prefix))
        try:
            plans.extend(
                [
                    _explain(
                        analytics_connection,
                        f"weather_actual:{profile.weather_actual_table}",
                        _weather_query(profile, profile.weather_actual_table, future=False),
                        (profile.province_value, start, cutoff, cutoff),
                    ),
                    _explain(
                        analytics_connection,
                        f"weather_forecast:{profile.weather_forecast_table}",
                        _weather_query(profile, profile.weather_forecast_table, future=True),
                        (profile.province_value, start, cutoff),
                    ),
                ]
            )
            if profile.weather_daily_table:
                plans.append(
                    _explain(
                        analytics_connection,
                        f"weather_daily:{profile.weather_daily_table}",
                        (
                            "SELECT COUNT(*) rows_total,MIN(`forecast_date`) start_date,"
                            f"MAX(`forecast_date`) end_date FROM `{profile.weather_daily_table}` "
                            "WHERE `province_name`=%s AND `forecast_date`>=%s AND `create_time`<=%s"
                        ),
                        (profile.province_value, start.date(), cutoff),
                    )
                )
        finally:
            try:
                analytics_connection.rollback()
            finally:
                analytics_connection.close()
    return {
        "region": profile.label,
        "range": {"start": start.isoformat(), "cutoff": cutoff.isoformat()},
        "explains": plans,
        "index_suggestions": _index_suggestions(profile),
        "note": "索引列仅为候选；请由 DBA 结合 EXPLAIN、现有索引和写入负载决定。",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="输出地区数据库查询的只读 EXPLAIN 和候选索引列")
    parser.add_argument("region", help="region_databases.yaml 中的 region_id，例如 shandong")
    parser.add_argument("--catalog", help="可选的地区配置文件路径")
    arguments = parser.parse_args()
    profiles = load_region_profiles(arguments.catalog)
    if arguments.region not in profiles:
        parser.error(f"未知地区：{arguments.region}")
    try:
        result = diagnose(profiles[arguments.region])
    except RegionSourceError as exc:
        parser.exit(2, f"无法运行诊断：{exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
