"""Read-only and normalization coverage for session-scoped regional price data."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import ClassVar

import pandas as pd

from app.research.data.sources.regions import (
    RegionProfile,
    _expand_hour_ending_series,
    _merge_actual_weather,
    credentials_for,
    fetch_region_price,
    load_region_profiles,
)


def _profile() -> RegionProfile:
    return RegionProfile(
        region_id="shandong",
        label="山东",
        market="山东电网",
        province_value="山东省",
        timezone="Asia/Shanghai",
        frequency="15min",
        credential_prefix="VPP_SHANDONG_DB_FEATURE",
        price_table="t_data_province_real_time_cleared_price",
        price_label="实时省级出清电价",
    )


class _Cursor:
    description: ClassVar[list[tuple[str]]] = [
        ("ts_day",),
        ("hour",),
        ("min",),
        ("quantity",),
        ("create_time",),
    ]

    def __init__(self) -> None:
        self.sql = ""
        self.parameters = ()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, sql, parameters):
        self.sql = sql
        self.parameters = parameters

    def fetchall(self):
        return [
            (date(2026, 9, 9), 23, 45, Decimal("321.5"), "2026-09-09T23:50:00"),
            (date(2026, 9, 9), 24, 0, Decimal("330.0"), "2026-09-10T00:05:00"),
        ]


class _Connection:
    def __init__(self) -> None:
        self.cursor_value = _Cursor()
        self.rolled_back = False
        self.closed = False

    def cursor(self):
        return self.cursor_value

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = True


def _configure(monkeypatch) -> None:
    for suffix, value in {
        "HOST": "db.invalid",
        "PORT": "3306",
        "USER": "reader",
        "PASSWORD": "do-not-print",
        "NAME": "prices",
    }.items():
        monkeypatch.setenv(f"VPP_SHANDONG_DB_FEATURE_{suffix}", value)


def test_builtin_catalog_contains_shandong() -> None:
    profiles = load_region_profiles()
    assert list(profiles) == ["shandong"]
    profile = profiles["shandong"]
    assert profile.market == "山东电网"
    assert profile.price_history_table == "t_data_province_real_time_cleared_price_hour"
    assert [item.name for item in profile.actual_series] == [
        "actual_load",
        "total_generation",
        "non_market_output",
        "thermal_power",
        "renewable_output",
        "actual_wind",
        "actual_hydro",
        "actual_solar",
        "positive_reserve",
        "negative_reserve",
    ]
    assert [item.name for item in profile.forecast_series] == [
        "forecast_load",
        "forecast_generation",
        "forecast_wind",
        "forecast_solar",
        "forecast_hydro",
    ]


def test_hour_ending_price_expands_back_over_its_four_quarters() -> None:
    hourly = pd.DataFrame(
        {
            "rt_price": [501.074],
            "available_at": [pd.Timestamp("2026-01-02")],
        },
        index=pd.DatetimeIndex(["2026-01-01T01:00:00"], name="timestamp"),
    )

    expanded = _expand_hour_ending_series(hourly, "15min")

    assert expanded["timestamp"].tolist() == list(
        pd.date_range("2026-01-01T00:15:00", "2026-01-01T01:00:00", freq="15min")
    )
    assert expanded["rt_price"].tolist() == [501.074] * 4


def test_catalog_can_add_regions_without_changing_the_desktop(tmp_path: Path) -> None:
    source = tmp_path / "regions.yaml"
    source.write_text(
        """regions:
  - region_id: shandong
    label: 山东
    market: 山东电网
    province_value: 山东省
    timezone: Asia/Shanghai
    frequency: 15min
    credential_prefix: VPP_SHANDONG_DB_FEATURE
    price_table: t_data_province_real_time_cleared_price
    price_label: 实时电价
  - region_id: hebei
    label: 河北
    market: 河北电网
    province_value: 河北省
    timezone: Asia/Shanghai
    frequency: 15min
    credential_prefix: VPP_HEBEI_DB_FEATURE
    price_table: t_price
    price_label: 实时电价
""",
        encoding="utf-8",
    )

    profiles = load_region_profiles(source)
    assert list(profiles) == ["shandong", "hebei"]
    assert profiles["hebei"].credential_prefix == "VPP_HEBEI_DB_FEATURE"


def test_credentials_never_print_password(monkeypatch) -> None:
    _configure(monkeypatch)
    credentials = credentials_for(_profile())
    assert "do-not-print" not in repr(credentials)


def test_fetch_normalizes_hour_24_and_writes_session_file(monkeypatch, tmp_path: Path) -> None:
    _configure(monkeypatch)
    connection = _Connection()
    result = fetch_region_price(
        _profile(),
        output_directory=tmp_path,
        now=datetime.fromisoformat("2026-09-10T08:00:00+08:00"),
        connection_factory=lambda _credentials: connection,
    )

    frame = pd.read_parquet(result.path)
    assert frame["timestamp"].tolist() == [
        pd.Timestamp("2026-09-09T23:45:00"),
        pd.Timestamp("2026-09-10T00:00:00"),
    ]
    assert frame["rt_price"].tolist() == [321.5, 330.0]
    assert result.row_count == 2
    assert connection.rolled_back and connection.closed
    assert connection.cursor_value.parameters[:2] == ("山东省", 1)
    assert "INSERT" not in connection.cursor_value.sql.upper()
    assert "UPDATE" not in connection.cursor_value.sql.upper()


def test_actual_weather_prefers_era5_and_uses_gfs_for_missing_hours() -> None:
    profile = RegionProfile(
        region_id="shandong",
        label="山东",
        market="山东电网",
        province_value="山东省",
        timezone="Asia/Shanghai",
        frequency="15min",
        credential_prefix="VPP_SHANDONG_DB_FEATURE",
        price_table="t_price",
        price_label="实时电价",
        analytics_credential_prefix="VPP_SHANDONG_DB_ANALYTICS",
        weather_actual_table="t_weather_actual",
        weather_forecast_table="t_weather_forecast",
        weather_columns=("temperature",),
    )
    era5 = pd.DataFrame(
        {
            "forecast_time": ["2026-09-01T00:00:00"],
            "temperature": [20],
            "create_time": ["2026-09-02T00:00:00"],
        }
    )
    gfs = pd.DataFrame(
        {
            "forecast_time": ["2026-09-01T00:00:00", "2026-09-01T01:00:00"],
            "temperature": [99, 21],
            "create_time": ["2026-08-31T20:00:00", "2026-08-31T20:00:00"],
        }
    )

    merged, fill_hours = _merge_actual_weather(era5, gfs, profile)

    assert fill_hours == 1
    assert len(merged) == 8
    assert merged["temperature"].tolist() == [20] * 4 + [21] * 4
