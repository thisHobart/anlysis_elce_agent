"""Point-in-time snapshot preparation for the Shandong forecast workflow."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from app.research.data.sources.regions import (
    RegionalSnapshot,
    RegionPriceFetch,
    RegionProfile,
    fetch_regional_snapshot,
)
from app.research.forecasting.contracts import ForecastPlan, ForecastSnapshotSpec

CORE_FORECAST_COLUMNS = (
    "forecast_load",
    "forecast_generation",
    "forecast_wind",
    "forecast_solar",
)
WEATHER_COLUMNS = (
    "temperature",
    "wind_speed_eighty",
    "cloud_cover",
    "solar_radiation",
    "hour_precipitation",
)


class ForecastDataError(ValueError):
    """The selected regional data cannot support the fixed P3 forecast contract."""


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _short_hash(values: list[str]) -> str:
    encoded = json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


def _read_series_file(path: str | Path | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=["timestamp", "available_at"])
    source = Path(path)
    if not source.is_file():
        raise ForecastDataError(f"预测输入文件不存在：{source}")
    frame = pd.read_parquet(source)
    if "timestamp" not in frame:
        raise ForecastDataError(f"预测输入缺少 timestamp：{source.name}")
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    if "available_at" in frame:
        frame["available_at"] = pd.to_datetime(frame["available_at"], errors="coerce")
    else:
        frame["available_at"] = frame["timestamp"]
    return (
        frame.dropna(subset=["timestamp"])
        .sort_values(["timestamp", "available_at"], kind="stable")
    )


def _local_naive(value: datetime | pd.Timestamp) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result.tzinfo is not None:
        result = result.tz_convert("Asia/Shanghai").tz_localize(None)
    return result


def _normalize_time_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for name in ("timestamp", "available_at"):
        if name not in result:
            continue
        values = pd.to_datetime(result[name], errors="coerce")
        if getattr(values.dt, "tz", None) is not None:
            values = values.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
        result[name] = values
    return result


def _visible(frame: pd.DataFrame, as_of: pd.Timestamp, *, future: bool) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = _normalize_time_columns(frame)
    visible = frame.loc[frame["available_at"].le(as_of)].copy()
    if not future:
        visible = visible.loc[visible["timestamp"].le(as_of)]
    return visible.sort_values(["timestamp", "available_at"], kind="stable").drop_duplicates(
        "timestamp", keep="last"
    )


def _join_values(parts: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    result: pd.DataFrame | None = None
    for label, frame in parts:
        if frame.empty:
            continue
        values = frame.set_index("timestamp").drop(columns=["available_at"], errors="ignore")
        values = values.rename(columns={column: f"{label}{column}" for column in values})
        result = values if result is None else result.join(values, how="outer")
    return result.sort_index() if result is not None else pd.DataFrame()


def adapt_regional_frames(
    target: pd.DataFrame,
    actuals: pd.DataFrame,
    forecasts: pd.DataFrame,
    *,
    as_of: datetime,
    target_end: datetime,
) -> pd.DataFrame:
    """Map the regional cache schema onto the isolated CTM input schema."""

    cutoff = _local_naive(as_of)
    end = _local_naive(target_end)
    target = _visible(target, cutoff, future=False)
    actuals = _visible(actuals, cutoff, future=False)
    forecasts = _visible(forecasts, cutoff, future=True)
    joined = _join_values((("target__", target), ("actual__", actuals), ("forecast__", forecasts)))
    if joined.empty:
        raise ForecastDataError("山东预测快照为空")
    start = min(joined.index.min(), cutoff - pd.Timedelta(days=260))
    grid = pd.date_range(start.floor("15min"), end.ceil("15min"), freq="15min")
    source = joined.reindex(grid)
    output = pd.DataFrame(index=grid)
    output.index.name = "timestamp"
    output["rt_price"] = pd.to_numeric(source.get("target__rt_price"), errors="coerce")

    actual_map = {
        "actual_load": "actual_load",
        "total_generation": "act_total_gen",
        "non_market_output": "non_market_output",
        "renewable_output": "act_new_energy_total",
        "actual_wind": "act_new_energy_wind",
        "actual_solar": "act_new_energy_solar",
        "actual_hydro": "act_hydro",
        "positive_reserve": "positive_reserve",
        "negative_reserve": "negative_reserve",
    }
    for source_name, destination in actual_map.items():
        key = f"actual__{source_name}"
        if key in source:
            output[destination] = pd.to_numeric(source[key], errors="coerce")

    forecast_map = {
        "forecast_load": "fcst_load_type_11",
        "forecast_generation": "fcst_total_gen",
        "forecast_hydro": "fcst_water",
    }
    for source_name, destination in forecast_map.items():
        key = f"forecast__{source_name}"
        if key in source:
            output[destination] = pd.to_numeric(source[key], errors="coerce")
    wind = pd.to_numeric(source.get("forecast__forecast_wind"), errors="coerce")
    solar = pd.to_numeric(source.get("forecast__forecast_solar"), errors="coerce")
    if isinstance(wind, pd.Series):
        output["fcst_new_energy_type_4"] = wind
    if isinstance(solar, pd.Series):
        output["fcst_new_energy_pv_unified"] = solar
    if isinstance(wind, pd.Series) or isinstance(solar, pd.Series):
        wind_series = wind if isinstance(wind, pd.Series) else pd.Series(index=grid, dtype=float)
        solar_series = solar if isinstance(solar, pd.Series) else pd.Series(index=grid, dtype=float)
        output["fcst_re_total"] = wind_series.add(solar_series, fill_value=0.0)
    if "fcst_load_type_11" in output and "fcst_re_total" in output:
        output["fcst_net_load"] = output["fcst_load_type_11"] - output["fcst_re_total"]

    for name in WEATHER_COLUMNS:
        historical = source.get(f"actual__{name}")
        predicted = source.get(f"forecast__{name}")
        values = pd.Series(index=grid, dtype=float)
        if isinstance(historical, pd.Series):
            values.loc[values.index <= cutoff] = pd.to_numeric(historical, errors="coerce")
        if isinstance(predicted, pd.Series):
            values.loc[values.index > cutoff] = pd.to_numeric(predicted, errors="coerce")
        output[f"wx_{name}"] = values
    output["wx_source_hist"] = output.index.to_series().le(cutoff).astype(float)
    output["wx_source_forecast"] = output.index.to_series().gt(cutoff).astype(float)
    output["wx_weather_available_mask"] = (
        output[[f"wx_{name}" for name in WEATHER_COLUMNS]].notna().any(axis=1).astype(float)
    )
    output["wx_weather_core_coverage"] = (
        output[[f"wx_{name}" for name in WEATHER_COLUMNS]].notna().mean(axis=1)
    )
    return output.replace([float("inf"), float("-inf")], np.nan)


def _complete_target_days(target: pd.DataFrame, *, before: pd.Timestamp) -> list[pd.Timestamp]:
    values = target.copy()
    values["timestamp"] = pd.to_datetime(values["timestamp"], errors="coerce")
    values["rt_price"] = pd.to_numeric(values["rt_price"], errors="coerce")
    values = values.dropna(subset=["timestamp", "rt_price"])
    values = values.loc[values["timestamp"].dt.normalize() < before.normalize()]
    counts = values.groupby(values["timestamp"].dt.normalize())["rt_price"].count()
    return sorted((pd.Timestamp(day) for day, count in counts.items() if count == 96), reverse=True)


def select_backtest_anchors(target: pd.DataFrame, *, now: datetime) -> list[datetime]:
    local_now = pd.Timestamp(now)
    if local_now.tzinfo is not None:
        local_now = local_now.tz_convert("Asia/Shanghai").tz_localize(None)
    days = _complete_target_days(target, before=local_now)
    anchors: list[pd.Timestamp] = []
    for day in days:
        if (local_now.normalize() - day).days > 90:
            continue
        if not anchors or (anchors[-1] - day).days >= 7:
            anchors.append(day)
        if len(anchors) == 3:
            break
    if len(anchors) != 3:
        raise ForecastDataError("最近90天内找不到3个相隔至少7天且具有96点真实电价的回测日")
    zone = ZoneInfo("Asia/Shanghai")
    return [day.to_pydatetime().replace(tzinfo=zone) for day in reversed(anchors)]


def _write_snapshot(frame: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.reset_index().to_parquet(path, index=False)
    return _hash_file(path)[:12]


def _current_frames(
    target_path: str | Path,
    actuals_path: str | Path | None,
    forecasts_path: str | Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        _read_series_file(target_path),
        _read_series_file(actuals_path),
        _read_series_file(forecasts_path),
    )


SnapshotFetcher = Callable[..., RegionPriceFetch]


def prepare_forecast_plan(
    *,
    question: str,
    profile: RegionProfile,
    target_path: str | Path,
    actuals_path: str | Path | None,
    forecasts_path: str | Path | None,
    output_directory: str | Path,
    source_data_fingerprint: str | None = None,
    now: datetime | None = None,
    progress: Callable[[int, str], None] | None = None,
    snapshot_fetcher: SnapshotFetcher | None = None,
) -> ForecastPlan:
    """Freeze three strict historical folds plus one current future snapshot."""

    if profile.region_id != "shandong":
        raise ForecastDataError("P3最小预测当前只支持山东")
    instant = now or datetime.now(UTC).astimezone(ZoneInfo(profile.timezone))
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=ZoneInfo(profile.timezone))
    instant = instant.astimezone(ZoneInfo(profile.timezone))
    target, actuals, forecasts = _current_frames(target_path, actuals_path, forecasts_path)
    last_price = pd.to_datetime(target.loc[target["rt_price"].notna(), "timestamp"], errors="coerce").max()
    if pd.isna(last_price):
        raise ForecastDataError("山东实际电价没有可用历史")
    stale = _local_naive(instant) - _local_naive(last_price)
    if stale > pd.Timedelta(hours=24):
        raise ForecastDataError(f"山东实际电价已超过24小时未更新（最后时点 {last_price}）")
    warnings: list[str] = []
    if stale > pd.Timedelta(hours=2):
        warnings.append(f"实际电价已超过2小时未更新（最后时点 {last_price}）")
    anchors = select_backtest_anchors(target, now=instant)
    plan_id = uuid4().hex[:12]
    root = Path(output_directory).resolve() / f"forecast-plan-{plan_id}"
    data_root = root / "data"
    specs: list[ForecastSnapshotSpec] = []
    truth = target.set_index("timestamp")[["rt_price"]].sort_index()
    for index, anchor in enumerate(anchors, start=1):
        if progress:
            progress(5 + index * 12, f"正在冻结第 {index}/3 个历史回测锚点")
        source_root = root / "sources" / f"fold-{index}"
        fetched: RegionPriceFetch | RegionalSnapshot
        if snapshot_fetcher is None:
            fetched = fetch_regional_snapshot(
                profile,
                output_directory=source_root,
                progress=None,
                as_of=anchor,
            )
        else:
            fetched = snapshot_fetcher(
                profile,
                output_directory=source_root,
                progress=None,
                now=anchor,
            )
        fold_target, fold_actuals, fold_forecasts = _current_frames(
            fetched.path,
            fetched.actuals_path,
            fetched.forecasts_path,
        )
        target_start = datetime.combine(anchor.date(), time.min, tzinfo=anchor.tzinfo)
        target_end = target_start + timedelta(hours=23, minutes=45)
        model_frame = adapt_regional_frames(
            fold_target,
            fold_actuals,
            fold_forecasts,
            as_of=anchor,
            target_end=target_end,
        )
        if int(model_frame.loc[model_frame.index < pd.Timestamp(anchor).tz_localize(None), "rt_price"].notna().sum()) < 2016:
            raise ForecastDataError(f"回测锚点 {anchor.date()} 之前不足2016个有效电价点")
        model_path = data_root / f"backtest-{anchor:%Y%m%d}.parquet"
        fingerprint = _write_snapshot(model_frame, model_path)
        truth_path = data_root / f"truth-{anchor:%Y%m%d}.parquet"
        truth.loc[str(anchor.date())].reset_index().to_parquet(truth_path, index=False)
        truth_fingerprint = _hash_file(truth_path)[:12]
        specs.append(
            ForecastSnapshotSpec(
                role="backtest",
                as_of=anchor,
                target_start=target_start,
                target_end=target_end,
                path=model_path,
                truth_path=truth_path,
                fingerprint=fingerprint,
                truth_fingerprint=truth_fingerprint,
            )
        )

    tomorrow = instant.date() + timedelta(days=1)
    future_start = datetime.combine(tomorrow, time.min, tzinfo=ZoneInfo(profile.timezone))
    future_end = future_start + timedelta(hours=23, minutes=45)
    future_frame = adapt_regional_frames(
        target,
        actuals,
        forecasts,
        as_of=instant,
        target_end=future_end,
    )
    horizon = future_frame.loc[str(tomorrow)]
    core = [
        column
        for column in ("fcst_load_type_11", "fcst_total_gen", "fcst_new_energy_type_4", "fcst_new_energy_pv_unified")
        if column in horizon
    ]
    if not core or max(int(horizon[column].notna().sum()) for column in core) < 72:
        raise ForecastDataError("次日至少需要一个负荷、发电、风电或光伏预测变量覆盖72/96点")
    weather_columns = [f"wx_{name}" for name in WEATHER_COLUMNS if f"wx_{name}" in horizon]
    weather_coverage = (
        max(int(horizon[column].notna().sum()) for column in weather_columns)
        if weather_columns
        else 0
    )
    if weather_coverage < 72:
        warnings.append("次日天气覆盖不足72/96点，模型将使用缺失掩码，不填入未来实况")
    future_path = data_root / f"future-{tomorrow:%Y%m%d}.parquet"
    future_fingerprint = _write_snapshot(future_frame, future_path)
    specs.append(
        ForecastSnapshotSpec(
            role="future",
            as_of=instant,
            target_start=future_start,
            target_end=future_end,
            path=future_path,
            fingerprint=future_fingerprint,
        )
    )
    composite = _short_hash(
        [
            value
            for item in specs
            for value in (item.fingerprint, item.truth_fingerprint)
            if value is not None
        ]
    )
    plan = ForecastPlan(
        plan_id=plan_id,
        question=question,
        forecast_start=future_start,
        forecast_end=future_end,
        snapshots=specs,
        data_fingerprint=composite,
        source_data_fingerprint=source_data_fingerprint,
        preflight_warnings=warnings,
        created_at=instant,
    )
    (root / "provenance").mkdir(parents=True, exist_ok=True)
    (root / "provenance" / "forecast_plan.json").write_text(
        plan.model_dump_json(indent=2), encoding="utf-8"
    )
    shutil.rmtree(root / "sources", ignore_errors=True)
    if progress:
        progress(100, "山东预测输入已经冻结")
    return plan
