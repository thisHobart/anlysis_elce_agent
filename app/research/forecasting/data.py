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
from app.research.forecasting.contracts import (
    ForecastNewsFeatureSource,
    ForecastPlan,
    ForecastSnapshotSpec,
)

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

    def local_wall_time(value: object) -> pd.Timestamp:
        if pd.isna(value):
            return pd.NaT
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError):
            return pd.NaT
        if timestamp.tzinfo is None:
            return timestamp
        return timestamp.tz_convert("Asia/Shanghai").tz_localize(None)

    for name in ("timestamp", "available_at"):
        if name not in result:
            continue
        # Persisted CSVs may mix naive market times, UTC intervals and +08:00 review
        # cutoffs, with or without fractional seconds. Parse each value independently;
        # naive values already represent the market wall clock, while aware values are
        # converted to that same wall clock before removing the timezone.
        result[name] = pd.to_datetime(result[name].map(local_wall_time), errors="coerce")
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


def _operational_backtest_origin(target_start: datetime, *, current_origin: datetime) -> datetime:
    """Replay the same lead time as the current next-day forecast."""

    zone = ZoneInfo("Asia/Shanghai")
    origin = current_origin.astimezone(zone)
    next_target_start = datetime.combine(origin.date() + timedelta(days=1), time.min, tzinfo=zone)
    lead_time = next_target_start - origin
    return target_start.astimezone(zone) - lead_time


def _baseline_common_observations(frame: pd.DataFrame, forecast_index: pd.DatetimeIndex) -> int:
    """Count points available to both the day and week baselines at the frozen origin."""

    price = pd.to_numeric(frame["rt_price"], errors="coerce")
    day = price.reindex(forecast_index - pd.Timedelta(days=1)).set_axis(forecast_index)
    week = price.reindex(forecast_index - pd.Timedelta(days=7)).set_axis(forecast_index)
    return int(pd.concat([day.rename("day"), week.rename("week")], axis=1).dropna().shape[0])


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


def _read_news_features(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        raise ForecastDataError(f"新闻特征文件不存在：{source}")
    frame = pd.read_parquet(source) if source.suffix.casefold() in {".parquet", ".pq"} else pd.read_csv(source)
    missing = {"timestamp", "available_at"}.difference(frame.columns)
    if missing:
        raise ForecastDataError(f"新闻特征缺少列：{', '.join(sorted(missing))}")
    frame = _normalize_time_columns(frame)
    if frame[["timestamp", "available_at"]].isna().any().any():
        raise ForecastDataError("新闻特征包含无效 timestamp 或 available_at")
    if frame.duplicated(["timestamp", "available_at"]).any():
        raise ForecastDataError("新闻特征包含重复 timestamp + available_at")
    return frame.sort_values(["timestamp", "available_at"], kind="stable")


def _news_name(name: str) -> str:
    normalized = "".join(character if character.isalnum() or character == "_" else "_" for character in name)
    return normalized if normalized.startswith("news_") else f"news_{normalized}"


def select_news_forecast_features(
    path: str | Path,
    *,
    selection_cutoff: datetime,
    minimum_observations: int = 96,
) -> tuple[tuple[str, ...], dict[str, str]]:
    """Select varying numeric news fields using only pre-evaluation observations."""

    frame = _read_news_features(path)
    cutoff = _local_naive(selection_cutoff)
    eligible = frame.loc[frame["available_at"].le(cutoff) & frame["timestamp"].lt(cutoff)]
    selected: list[str] = []
    excluded: dict[str, str] = {}
    for column in frame.columns:
        if column in {"timestamp", "available_at"}:
            continue
        values = pd.to_numeric(eligible[column], errors="coerce")
        name = _news_name(str(column))
        if int(values.notna().sum()) < minimum_observations:
            excluded[name] = f"选择截止点前有效观测少于{minimum_observations}个"
        elif int(values.nunique(dropna=True)) < 2:
            excluded[name] = "选择截止点前没有变化"
        else:
            selected.append(name)
    return tuple(selected), excluded


def _attach_news_features(
    model_frame: pd.DataFrame,
    news_frame: pd.DataFrame,
    *,
    as_of: datetime,
    selected_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Join only feature versions knowable at this exact forecast origin."""

    if not selected_columns:
        return model_frame
    cutoff = _local_naive(as_of)
    visible = news_frame.loc[news_frame["available_at"].le(cutoff)].copy()
    visible = visible.sort_values(["timestamp", "available_at"], kind="stable").drop_duplicates("timestamp", keep="last")
    by_output = {_news_name(str(column)): column for column in news_frame.columns if column not in {"timestamp", "available_at"}}
    output = model_frame.copy()
    values = visible.set_index("timestamp")
    for selected in selected_columns:
        source = by_output.get(selected)
        if source is None:
            raise ForecastDataError(f"找不到已批准新闻特征：{selected}")
        output[selected] = pd.to_numeric(values[source], errors="coerce").reindex(output.index)
    return output


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
    news_features_path: str | Path | None = None,
    news_source_run_id: str | None = None,
    news_source_p1_run_id: str | None = None,
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
    target_days = select_backtest_anchors(target, now=instant)
    historical_origins = [
        _operational_backtest_origin(target_day, current_origin=instant)
        for target_day in target_days
    ]
    news_frame = _read_news_features(news_features_path) if news_features_path else None
    selected_news: tuple[str, ...] = ()
    excluded_news: dict[str, str] = {}
    news_sources: tuple[ForecastNewsFeatureSource, ...] = ()
    if news_frame is not None:
        selected_news, excluded_news = select_news_forecast_features(
            news_features_path,
            selection_cutoff=historical_origins[0],
        )
        source_path = Path(news_features_path).resolve()
        news_sources = (
            ForecastNewsFeatureSource(
                source_run_id=news_source_run_id or source_path.parent.name,
                source_p1_run_id=news_source_p1_run_id,
                source_path=source_path,
                source_sha256=_hash_file(source_path),
                selected_columns=selected_news,
                excluded_columns=excluded_news,
                selection_cutoff=historical_origins[0],
                selection_reason="只使用最早回测锚点之前的观测筛选非恒定数值新闻特征",
            ),
        )
        if not selected_news:
            warnings.append("没有新闻特征通过预测输入筛选；P2证据仍保留用于综合分析")
    plan_id = uuid4().hex[:12]
    root = Path(output_directory).resolve() / f"forecast-plan-{plan_id}"
    data_root = root / "data"
    specs: list[ForecastSnapshotSpec] = []
    truth = target.set_index("timestamp")[["rt_price"]].sort_index()
    for index, (target_day, anchor) in enumerate(
        zip(target_days, historical_origins, strict=True), start=1
    ):
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
        target_start = datetime.combine(target_day.date(), time.min, tzinfo=target_day.tzinfo)
        target_end = target_start + timedelta(hours=23, minutes=45)
        model_frame = adapt_regional_frames(
            fold_target,
            fold_actuals,
            fold_forecasts,
            as_of=anchor,
            target_end=target_end,
        )
        if news_frame is not None:
            model_frame = _attach_news_features(
                model_frame, news_frame, as_of=anchor, selected_columns=selected_news
            )
        if int(model_frame.loc[model_frame.index < pd.Timestamp(anchor).tz_localize(None), "rt_price"].notna().sum()) < 2016:
            raise ForecastDataError(f"回测锚点 {anchor.date()} 之前不足2016个有效电价点")
        forecast_index = pd.date_range(
            pd.Timestamp(target_start).tz_localize(None), periods=96, freq="15min"
        )
        baseline_common = _baseline_common_observations(model_frame, forecast_index)
        if baseline_common < 72:
            raise ForecastDataError(
                f"回测目标日 {target_start.date()} 在预测起点 {anchor.isoformat()} "
                f"只有{baseline_common}/96个日、周基线共同有效点；至少需要72个"
            )
        model_path = data_root / f"backtest-{target_start:%Y%m%d}.parquet"
        fingerprint = _write_snapshot(model_frame, model_path)
        truth_path = data_root / f"truth-{target_start:%Y%m%d}.parquet"
        truth.loc[str(target_start.date())].reset_index().to_parquet(truth_path, index=False)
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
                news_feature_columns=selected_news,
                expected_baseline_common_observations=baseline_common,
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
    if news_frame is not None:
        future_frame = _attach_news_features(
            future_frame, news_frame, as_of=instant, selected_columns=selected_news
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
            news_feature_columns=selected_news,
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
        news_feature_sources=news_sources,
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
