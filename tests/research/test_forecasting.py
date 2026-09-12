"""P3 Shandong point-forecast contracts, leakage gates, and CPU model."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from app.research.forecasting.contracts import (
    CTMTrainingConfig,
    ForecastPlan,
    ForecastSnapshotSpec,
)
from app.research.forecasting.data import adapt_regional_frames, select_backtest_anchors
from app.research.forecasting.model import forecast_ctm_similar_day
from app.research.forecasting.service import run_forecast_plan
from app.research.forecasting.workflow import execute_forecast_workflow

ZONE = ZoneInfo("Asia/Shanghai")


def _snapshot(role: str, anchor: datetime, path: Path) -> ForecastSnapshotSpec:
    return ForecastSnapshotSpec(
        role=role,
        as_of=anchor,
        target_start=anchor,
        target_end=anchor + timedelta(hours=23, minutes=45),
        path=path,
        truth_path=path if role == "backtest" else None,
        truth_fingerprint="c" * 12 if role == "backtest" else None,
        fingerprint="a" * 12,
    )


def test_forecast_plan_has_three_spaced_folds_and_one_future(tmp_path: Path):
    anchors = [datetime(2026, 8, day, tzinfo=ZONE) for day in (1, 8, 15)]
    future = datetime(2026, 9, 13, tzinfo=ZONE)

    plan = ForecastPlan(
        question="预测山东未来24小时",
        forecast_start=future,
        forecast_end=future + timedelta(hours=23, minutes=45),
        snapshots=[
            *(
                _snapshot("backtest", value, tmp_path / f"{index}.parquet")
                for index, value in enumerate(anchors)
            ),
            _snapshot("future", future, tmp_path / "future.parquet"),
        ],
        data_fingerprint="b" * 12,
    )

    assert plan.read_only
    assert plan.horizon_steps == 96
    assert plan.training.seed == 42


def test_anchor_selection_uses_complete_days_at_least_a_week_apart():
    days = pd.to_datetime(["2026-08-01", "2026-08-08", "2026-08-15", "2026-08-22"])
    index = days[0] + pd.to_timedelta(np.arange(96) * 15, unit="min")
    frames = []
    for day in days:
        frame = pd.DataFrame({"timestamp": day + (index - days[0]), "rt_price": range(96)})
        frames.append(frame)

    anchors = select_backtest_anchors(
        pd.concat(frames, ignore_index=True),
        now=datetime(2026, 8, 23, 12, tzinfo=ZONE),
    )

    assert [value.date().isoformat() for value in anchors] == ["2026-08-08", "2026-08-15", "2026-08-22"]


def test_adapter_does_not_expose_actual_values_after_cutoff():
    index = pd.date_range("2026-08-01", periods=16, freq="15min")
    cutoff = datetime(2026, 8, 1, 1, 45, tzinfo=ZONE)
    index = index.tz_localize(ZONE)
    target = pd.DataFrame(
        {"timestamp": index, "rt_price": range(16), "available_at": index + pd.Timedelta(minutes=1)}
    )
    actual = pd.DataFrame(
        {"timestamp": index, "actual_load": range(16), "available_at": index + pd.Timedelta(minutes=1)}
    )
    forecast = pd.DataFrame(
        {"timestamp": index, "forecast_load": range(16), "available_at": pd.Timestamp(cutoff)}
    )

    adapted = adapt_regional_frames(
        target,
        actual,
        forecast,
        as_of=cutoff,
        target_end=datetime(2026, 8, 1, 3, 45, tzinfo=ZONE),
    )

    local_cutoff = pd.Timestamp(cutoff).tz_localize(None)
    assert adapted.loc[adapted.index > local_cutoff, "rt_price"].isna().all()
    assert adapted.loc[adapted.index > local_cutoff, "actual_load"].isna().all()
    assert adapted.loc[adapted.index > local_cutoff, "fcst_load_type_11"].notna().all()


def test_adapter_uses_latest_version_visible_at_cutoff():
    timestamp = pd.Timestamp("2026-08-01 00:15")
    cutoff = datetime(2026, 8, 1, 1, tzinfo=ZONE)
    target = pd.DataFrame(
        {
            "timestamp": [timestamp, timestamp],
            "rt_price": [100.0, 999.0],
            "available_at": [timestamp + pd.Timedelta(minutes=5), timestamp + pd.Timedelta(hours=2)],
        }
    )

    adapted = adapt_regional_frames(
        target,
        pd.DataFrame(),
        pd.DataFrame(),
        as_of=cutoff,
        target_end=cutoff,
    )

    assert adapted.loc[timestamp, "rt_price"] == 100.0


def _model_frame() -> pd.DataFrame:
    index = pd.date_range("2026-06-01", periods=96 * 24, freq="15min")
    slot = index.hour * 4 + index.minute // 15
    day = np.asarray((index.normalize() - index.normalize().min()).days, dtype=float)
    load = 60000 + 4000 * np.sin(2 * np.pi * slot / 96)
    solar = np.maximum(0, 9000 * np.sin(np.pi * (slot - 24) / 48))
    price = 260 + 0.006 * (load - solar - 55000) + 8 * np.sin(day / 3)
    frame = pd.DataFrame(index=index)
    frame["rt_price"] = price
    frame["actual_load"] = load * 1.01
    frame["fcst_load_type_11"] = load
    frame["fcst_new_energy_pv_unified"] = solar
    frame["fcst_re_total"] = solar + 3000
    frame["fcst_net_load"] = load - frame["fcst_re_total"]
    frame["wx_temperature"] = 22 + 6 * np.sin(2 * np.pi * slot / 96)
    frame["wx_solar_radiation"] = np.maximum(0, 800 * np.sin(np.pi * (slot - 24) / 48))
    frame["wx_cloud_cover"] = 40
    frame["wx_wind_speed_eighty"] = 5
    frame["wx_hour_precipitation"] = 0
    return frame


def test_cpu_ctm_and_similar_day_are_deterministic_and_ignore_future_truth():
    frame = _model_frame()
    forecast_index = frame.index[-96:]
    train_end = forecast_index[0] - pd.Timedelta(minutes=15)
    config = CTMTrainingConfig(
        lookback_steps=8,
        internal_ticks=2,
        memory_length=2,
        hidden_size=8,
        sync_pairs=4,
        dropout=0,
        max_epochs=1,
        batch_size=16,
        early_stopping_patience=1,
        minimum_training_rows=96 * 10,
        maximum_training_samples=24,
    )
    clean = frame.copy()
    clean.loc[forecast_index, "rt_price"] = np.nan
    poisoned = clean.copy()
    poisoned.loc[forecast_index, "rt_price"] = 9999.0
    poisoned.loc[forecast_index, "actual_load"] = 999999.0

    first = forecast_ctm_similar_day(
        clean,
        train_end=train_end,
        forecast_index=forecast_index,
        config=config,
    )
    second = forecast_ctm_similar_day(
        poisoned,
        train_end=train_end,
        forecast_index=forecast_index,
        config=config,
    )

    assert len(first) == 96
    assert np.isfinite(first["predicted_rt_price"]).all()
    assert np.allclose(first["predicted_rt_price"], second["predicted_rt_price"], atol=1e-5)


def test_forecast_service_writes_research_package_and_resumes(tmp_path: Path):
    source = _model_frame()
    root = tmp_path / "forecast-plan-test"
    data_root = root / "data"
    data_root.mkdir(parents=True)
    start = source.index[0]
    anchor_starts = [start + pd.Timedelta(days=day) for day in (7, 14, 21)]
    snapshots = []
    for number, anchor in enumerate(anchor_starts, start=1):
        frozen = source.copy()
        frozen.loc[frozen.index >= anchor, "rt_price"] = np.nan
        path = data_root / f"backtest-{number}.parquet"
        frozen.reset_index(names="timestamp").to_parquet(path, index=False)
        truth_path = data_root / f"truth-{number}.parquet"
        truth = source.loc[
            anchor : anchor + pd.Timedelta(hours=23, minutes=45), ["rt_price"]
        ]
        truth.reset_index(names="timestamp").to_parquet(truth_path, index=False)
        snapshots.append(
            ForecastSnapshotSpec(
                role="backtest",
                as_of=anchor.to_pydatetime().replace(tzinfo=ZONE),
                target_start=anchor.to_pydatetime().replace(tzinfo=ZONE),
                target_end=(anchor + pd.Timedelta(hours=23, minutes=45))
                .to_pydatetime()
                .replace(tzinfo=ZONE),
                path=path,
                truth_path=truth_path,
                fingerprint=hashlib.sha256(path.read_bytes()).hexdigest()[:12],
                truth_fingerprint=hashlib.sha256(truth_path.read_bytes()).hexdigest()[:12],
            )
        )
    future_start = start + pd.Timedelta(days=23)
    future_frame = source.copy()
    future_frame.loc[future_frame.index >= future_start, "rt_price"] = np.nan
    future_path = data_root / "future.parquet"
    future_frame.reset_index(names="timestamp").to_parquet(future_path, index=False)
    snapshots.append(
        ForecastSnapshotSpec(
            role="future",
            as_of=(future_start - pd.Timedelta(minutes=15)).to_pydatetime().replace(tzinfo=ZONE),
            target_start=future_start.to_pydatetime().replace(tzinfo=ZONE),
            target_end=(future_start + pd.Timedelta(hours=23, minutes=45))
            .to_pydatetime()
            .replace(tzinfo=ZONE),
            path=future_path,
            fingerprint=hashlib.sha256(future_path.read_bytes()).hexdigest()[:12],
        )
    )
    plan = ForecastPlan(
        question="预测山东明天实时电价",
        forecast_start=future_start.to_pydatetime().replace(tzinfo=ZONE),
        forecast_end=(future_start + pd.Timedelta(hours=23, minutes=45))
        .to_pydatetime()
        .replace(tzinfo=ZONE),
        snapshots=snapshots,
        data_fingerprint="d" * 12,
        training=CTMTrainingConfig(
            lookback_steps=8,
            internal_ticks=2,
            memory_length=2,
            hidden_size=8,
            sync_pairs=4,
            max_epochs=1,
            batch_size=4,
            minimum_training_rows=96,
            training_stride=8,
            maximum_training_samples=12,
            early_stopping_patience=1,
        ),
    )

    class Interrupted(RuntimeError):
        pass

    def interrupt_after_first_fold(_value: int, message: str) -> None:
        if "第 2/3 个历史回测" in message:
            raise Interrupted

    with pytest.raises(Interrupted):
        run_forecast_plan(plan, progress=interrupt_after_first_fold)
    first_fold = root / "backtest-1.parquet"
    first_fold_mtime = first_fold.stat().st_mtime_ns

    first = execute_forecast_workflow(plan)
    second = run_forecast_plan(plan)

    assert second is not None
    assert second.run_id == first.run_id
    assert first_fold.stat().st_mtime_ns == first_fold_mtime
    assert first.prediction_path.is_file()
    assert first.backtest_path.is_file()
    assert first.report_path.is_file()
    assert (root / "provenance" / "checkpoint.json").is_file()
    assert (root / "provenance" / "snapshot_manifest.json").is_file()
    assert (root / "provenance" / "environment.json").is_file()
    prediction = pd.read_csv(first.prediction_path)
    assert len(prediction) == 96
    assert {
        "timestamp",
        "predicted_rt_price",
        "ctm_prediction",
        "similar_day_prior",
        "ctm_weight",
        "ctm_certainty",
        "data_cutoff",
        "algorithm_version",
    }.issubset(prediction.columns)
