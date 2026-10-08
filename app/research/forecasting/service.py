"""Application service for three-fold backtesting and the next-day forecast."""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from app.research.forecasting.artifacts import (
    artifact_hash,
    write_backtest_figure,
    write_forecast_figure,
    write_metrics,
    write_report,
)
from app.research.forecasting.contracts import (
    FORECAST_ALGORITHM_VERSION,
    ForecastFoldResult,
    ForecastMetricSet,
    ForecastPlan,
    ForecastRunResult,
)


def _load_snapshot(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "timestamp" not in frame:
        raise ValueError(f"预测快照缺少 timestamp：{path.name}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    return frame.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unavailable"


def _manifest_is_valid(root: Path) -> bool:
    try:
        payload = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        files = payload["files"]
        return isinstance(files, dict) and bool(files) and all(
            (root / relative).is_file() and _sha256(root / relative) == digest
            for relative, digest in files.items()
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _read_checkpoint(path: Path, plan: ForecastPlan) -> set[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    if payload.get("plan_id") != plan.plan_id or payload.get("data_fingerprint") != plan.data_fingerprint:
        return set()
    return {str(item) for item in payload.get("completed_work_items", [])}


def _write_checkpoint(path: Path, plan: ForecastPlan, completed: set[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "plan_id": plan.plan_id,
                "data_fingerprint": plan.data_fingerprint,
                "algorithm_version": plan.algorithm_version,
                "completed_work_items": sorted(completed),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _metrics(actual: pd.Series, predicted: pd.Series) -> ForecastMetricSet:
    aligned = pd.concat(
        [pd.to_numeric(actual, errors="coerce"), pd.to_numeric(predicted, errors="coerce")], axis=1
    ).dropna()
    if aligned.empty:
        raise ValueError("预测指标没有共同有效点")
    error = aligned.iloc[:, 1] - aligned.iloc[:, 0]
    return ForecastMetricSet(
        observations=len(aligned),
        mae=float(error.abs().mean()),
        rmse=float(np.sqrt(error.pow(2).mean())),
        bias=float(error.mean()),
    )


def _baseline_frame(frame: pd.DataFrame, forecast_index: pd.DatetimeIndex) -> pd.DataFrame:
    price = pd.to_numeric(frame["rt_price"], errors="coerce")
    last = price.loc[price.index < forecast_index[0]].dropna()
    if last.empty:
        raise ValueError("朴素基线没有锚点前实际电价")
    persistence = pd.Series(float(last.iloc[-1]), index=forecast_index)
    return pd.DataFrame(
        {
            "persistence": persistence,
            "day_naive": price.reindex(forecast_index - pd.Timedelta(days=1)).set_axis(forecast_index),
            "week_naive": price.reindex(forecast_index - pd.Timedelta(days=7)).set_axis(forecast_index),
        }
    )


def _aggregate(backtest: pd.DataFrame) -> dict[str, ForecastMetricSet]:
    common = backtest[
        [
            "actual_rt_price",
            "predicted_rt_price",
            "persistence",
            "day_naive",
            "week_naive",
        ]
    ].dropna()
    actual = common["actual_rt_price"]
    return {
        "model": _metrics(actual, common["predicted_rt_price"]),
        "persistence": _metrics(actual, common["persistence"]),
        "day_naive": _metrics(actual, common["day_naive"]),
        "week_naive": _metrics(actual, common["week_naive"]),
    }


def run_forecast_plan(
    plan: ForecastPlan,
    *,
    progress: Callable[[int, str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    stop_after_work_item: str | None = None,
) -> ForecastRunResult | None:
    """Execute immutable snapshots only; no database or publisher is reachable here."""

    from app.research.forecasting.model import ForecastCancelled, forecast_ctm_similar_day

    callback = progress or (lambda _value, _message: None)
    is_cancelled = cancelled or (lambda: False)
    if plan.algorithm_version != FORECAST_ALGORITHM_VERSION:
        raise ValueError(
            f"预测算法版本已变化：方案={plan.algorithm_version}，当前={FORECAST_ALGORITHM_VERSION}；请重新确认新方案"
        )
    root = next(item.path.parent.parent for item in plan.snapshots if item.role == "future")
    figures = root / "figures"
    provenance = root / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    checkpoint_path = provenance / "checkpoint.json"
    completed = _read_checkpoint(checkpoint_path, plan)
    stored_result_path = provenance / "run_result.json"
    for snapshot in plan.snapshots:
        if _sha256(snapshot.path)[:12] != snapshot.fingerprint:
            raise ValueError(f"冻结输入指纹已经变化：{snapshot.path.name}")
        if (
            snapshot.truth_path
            and snapshot.truth_fingerprint
            and _sha256(snapshot.truth_path)[:12] != snapshot.truth_fingerprint
        ):
            raise ValueError(f"回测真实值指纹已经变化：{snapshot.truth_path.name}")
    for source in plan.news_feature_sources:
        if not source.source_path.is_file() or _sha256(source.source_path) != source.source_sha256:
            raise ValueError(f"新闻特征来源指纹已经变化：{source.source_path.name}")
    if "future" in completed and stored_result_path.is_file():
        try:
            stored_result = ForecastRunResult.model_validate_json(
                stored_result_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            completed.discard("future")
        else:
            required = (
                stored_result.prediction_path,
                stored_result.backtest_path,
                stored_result.metrics_path,
                stored_result.report_path,
                *stored_result.figure_paths.values(),
            )
            if all(path.is_file() for path in required) and _manifest_is_valid(root):
                callback(100, "已从检查点恢复完整预测结果")
                return stored_result
            completed.discard("future")
    fold_results: list[ForecastFoldResult] = []
    fold_frames: list[pd.DataFrame] = []
    backtests = [item for item in plan.snapshots if item.role == "backtest"]
    for number, snapshot in enumerate(backtests, start=1):
        if is_cancelled():
            raise ForecastCancelled("预测已取消")
        if _sha256(snapshot.path)[:12] != snapshot.fingerprint:
            raise ValueError(f"回测 {number}/3 的冻结输入指纹已经变化")
        if (
            snapshot.truth_path
            and snapshot.truth_fingerprint
            and _sha256(snapshot.truth_path)[:12] != snapshot.truth_fingerprint
        ):
            raise ValueError(f"回测 {number}/3 的真实值指纹已经变化")
        work_item = f"backtest-{number}"
        fold_path = root / f"backtest-{number}.parquet"
        fold_result_path = provenance / f"fold-{number}.json"
        if work_item in completed and fold_path.is_file() and fold_result_path.is_file():
            try:
                result = ForecastFoldResult.model_validate_json(
                    fold_result_path.read_text(encoding="utf-8")
                )
                points = pd.read_parquet(fold_path)
                if result.snapshot_fingerprint != snapshot.fingerprint:
                    raise ValueError("snapshot fingerprint changed")
            except (OSError, ValueError):
                completed.discard(work_item)
            else:
                common_columns = [
                    "actual_rt_price",
                    "predicted_rt_price",
                    "persistence",
                    "day_naive",
                    "week_naive",
                ]
                result = result.model_copy(
                    update={
                        "common_observations": len(points[common_columns].dropna()),
                        "news_feature_columns": snapshot.news_feature_columns,
                    }
                )
                fold_results.append(result)
                fold_frames.append(points)
                callback(5 + number * 22, f"已恢复第 {number}/3 个历史回测")
                if stop_after_work_item == work_item:
                    return None
                continue
        callback(5 + (number - 1) * 22, f"正在运行第 {number}/3 个历史回测")
        frame = _load_snapshot(snapshot.path)
        forecast_index = pd.date_range(
            pd.Timestamp(snapshot.target_start).tz_localize(None), periods=96, freq="15min"
        )

        def epoch_progress(epoch: int, message: str, *, fold_number: int = number) -> None:
            callback(
                5 + (fold_number - 1) * 22 + int(18 * epoch / plan.training.max_epochs),
                f"回测 {fold_number}/3 · {message}",
            )

        predicted = forecast_ctm_similar_day(
            frame,
            train_end=forecast_index[0] - pd.Timedelta(minutes=15),
            forecast_index=forecast_index,
            config=plan.training,
            progress=epoch_progress,
            cancelled=is_cancelled,
        )
        truth = pd.read_parquet(snapshot.truth_path) if snapshot.truth_path else pd.DataFrame()
        truth["timestamp"] = pd.to_datetime(truth.get("timestamp"), errors="coerce")
        actual = truth.set_index("timestamp")["rt_price"].reindex(forecast_index)
        if int(actual.notna().sum()) != 96:
            raise ValueError(f"回测 {number}/3 的真实电价不是完整96点")
        baselines = _baseline_frame(frame, forecast_index)
        points = predicted.join(baselines)
        points.insert(0, "actual_rt_price", actual)
        common = points[
            [
                "actual_rt_price",
                "predicted_rt_price",
                "persistence",
                "day_naive",
                "week_naive",
            ]
        ].dropna()
        if common.empty:
            raise ValueError(f"回测 {number}/3 没有可供模型和三种基线共同比较的点")
        points.insert(0, "anchor", snapshot.as_of.isoformat())
        points.insert(0, "timestamp", points.index)
        points.to_parquet(fold_path, index=False)
        result = ForecastFoldResult(
            anchor=snapshot.as_of,
            target_start=snapshot.target_start,
            target_end=snapshot.target_end,
            model=_metrics(common["actual_rt_price"], common["predicted_rt_price"]),
            persistence=_metrics(common["actual_rt_price"], common["persistence"]),
            day_naive=_metrics(common["actual_rt_price"], common["day_naive"]),
            week_naive=_metrics(common["actual_rt_price"], common["week_naive"]),
            prediction_path=fold_path,
            snapshot_fingerprint=snapshot.fingerprint,
            common_observations=len(common),
            news_feature_columns=snapshot.news_feature_columns,
        )
        fold_results.append(result)
        fold_frames.append(points)
        fold_result_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        completed.add(work_item)
        _write_checkpoint(checkpoint_path, plan, completed)
        if stop_after_work_item == work_item:
            callback(5 + number * 22, f"第 {number}/3 个历史回测已写入检查点")
            return None

    backtest = pd.concat(fold_frames, ignore_index=True)
    backtest_path = root / "backtest_predictions.parquet"
    backtest.to_parquet(backtest_path, index=False)
    aggregate = _aggregate(backtest)
    warnings: list[str] = list(plan.preflight_warnings)
    enough_comparable_points = all(item.common_observations >= 72 for item in fold_results)
    if not enough_comparable_points:
        warnings.append(
            "基线可比点不足：三个回测折未分别达到72个共同有效点，不能验证预测增益"
        )
    if not enough_comparable_points or aggregate["model"].mae >= min(
        aggregate["day_naive"].mae,
        aggregate["week_naive"].mae,
    ):
        warnings.append("未验证出预测增益：没有在足够共同有效点上优于前一日/一周前朴素基线")
    fold_maes = [item.model.mae for item in fold_results]
    if min(fold_maes) > 0 and max(fold_maes) > 2 * min(fold_maes):
        warnings.append("三折表现不稳定：最差折MAE超过最佳折的两倍")

    callback(75, "正在生成次日96点预测")
    future = next(item for item in plan.snapshots if item.role == "future")
    if _sha256(future.path)[:12] != future.fingerprint:
        raise ValueError("次日预测的冻结输入指纹已经变化")
    frame = _load_snapshot(future.path)
    forecast_index = pd.date_range(pd.Timestamp(plan.forecast_start).tz_localize(None), periods=96, freq="15min")
    train_end = pd.to_numeric(frame["rt_price"], errors="coerce").dropna().index.max()

    def future_progress(epoch: int, message: str) -> None:
        callback(75 + int(20 * epoch / plan.training.max_epochs), f"次日预测 · {message}")

    prediction = forecast_ctm_similar_day(
        frame,
        train_end=train_end,
        forecast_index=forecast_index,
        config=plan.training,
        progress=future_progress,
        cancelled=is_cancelled,
    )
    prediction.insert(0, "timestamp", prediction.index)
    prediction["data_cutoff"] = future.as_of.isoformat()
    prediction_path = root / "prediction.csv"
    prediction.to_csv(prediction_path, index=False, encoding="utf-8-sig")
    forecast_figure = figures / "forecast_curve.svg"
    backtest_figure = figures / "backtest_comparison.svg"
    write_forecast_figure(prediction, forecast_figure)
    write_backtest_figure(aggregate, backtest_figure)
    metrics_path = root / "metrics.json"
    write_metrics(fold_results, aggregate, warnings, metrics_path)
    report_path = root / "report.md"
    write_report(
        plan=plan,
        folds=fold_results,
        aggregate=aggregate,
        warnings=warnings,
        destination=report_path,
    )
    snapshot_manifest_path = provenance / "snapshot_manifest.json"
    snapshot_manifest_path.write_text(
        json.dumps(
            {
                "plan_id": plan.plan_id,
                "data_fingerprint": plan.data_fingerprint,
                "snapshots": [
                    {
                        **item.model_dump(mode="json"),
                        "sha256": _sha256(item.path),
                        "truth_sha256": _sha256(item.truth_path) if item.truth_path else None,
                    }
                    for item in plan.snapshots
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    environment_path = provenance / "environment.json"
    environment_path.write_text(
        json.dumps(
            {
                "algorithm_id": plan.algorithm_id,
                "algorithm_version": plan.algorithm_version,
                "python": platform.python_version(),
                "dependencies": {
                    name: _package_version(name)
                    for name in ("numpy", "pandas", "scikit-learn", "torch")
                },
                "device": "cpu",
                "deterministic": True,
                "seed": plan.training.seed,
                "business_database_writes": 0,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    artifact_files = (
        prediction_path,
        backtest_path,
        metrics_path,
        report_path,
        forecast_figure,
        backtest_figure,
        snapshot_manifest_path,
        environment_path,
    )
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "plan_id": plan.plan_id,
                "algorithm": plan.algorithm_version,
                "data_fingerprint": plan.data_fingerprint,
                "read_only": True,
                "status": "completed",
                "business_database_writes": 0,
                "files": {
                    str(path.relative_to(root)): _sha256(path) for path in artifact_files
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    digest = artifact_hash([*artifact_files, manifest_path])
    result = ForecastRunResult(
        run_id=uuid4().hex[:12],
        plan=plan,
        folds=fold_results,
        aggregate=aggregate,
        diagnostics={
            "database_mode": "read_only",
            "business_database_writes": 0,
            "leakage_gate": "passed",
            "backtest_folds": 3,
            "forecast_points": 96,
            "data_cutoff": future.as_of.isoformat(),
            "baseline_verified": aggregate["model"].mae
            < min(aggregate["day_naive"].mae, aggregate["week_naive"].mae)
            and enough_comparable_points,
            "evaluation_status": (
                "not_evaluable"
                if not enough_comparable_points
                else (
                    "verified"
                    if aggregate["model"].mae
                    < min(aggregate["day_naive"].mae, aggregate["week_naive"].mae)
                    else "not_verified"
                )
            ),
            "fold_common_observations": [item.common_observations for item in fold_results],
            "news_feature_columns": sorted(
                {column for item in plan.snapshots for column in item.news_feature_columns}
            ),
            "news_feature_sources": [
                item.model_dump(mode="json") for item in plan.news_feature_sources
            ],
        },
        warnings=warnings,
        prediction_path=prediction_path,
        backtest_path=backtest_path,
        metrics_path=metrics_path,
        report_path=report_path,
        artifact_directory=root,
        figure_paths={"forecast": forecast_figure, "backtest": backtest_figure},
        output_hash=digest,
    )
    stored_result_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    completed.add("future")
    _write_checkpoint(checkpoint_path, plan, completed)
    callback(100, "山东次日预测已经完成")
    return result
