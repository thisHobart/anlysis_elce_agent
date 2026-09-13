"""Cross-stage contracts, P2 hand-off and bounded feedback behavior."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from app.research.forecasting.contracts import (
    ForecastFoldResult,
    ForecastMetricSet,
    ForecastPlan,
    ForecastRunResult,
    ForecastSnapshotSpec,
)
from app.research.full_flow import FlowRunReference, FullFlowStore, FullResearchFlow
from app.research.news import (
    JsonlCollectedNewsAdapter,
    MarketClock,
    ObviousNewsEventExtractor,
    build_forecast_news_features,
    load_price_csv,
    run_news_price_study,
)
from app.research.news.workspace import NewsWorkspace

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "news_price"
pytestmark = pytest.mark.integration


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def test_future_news_features_freeze_knowledge_at_prediction_origin():
    clock = MarketClock(market="TEST_MARKET", timezone="UTC", interval_minutes=30)
    prices = load_price_csv(FIXTURES / "synthetic_prices.csv", clock)
    study = run_news_price_study(
        adapter=JsonlCollectedNewsAdapter(FIXTURES / "synthetic_news.jsonl"),
        prices=prices,
        as_of=datetime.fromisoformat("2026-03-01T23:30:00+00:00"),
        axis="effective",
    )
    cutoff = datetime.fromisoformat("2026-01-12T10:10:00+00:00")
    frame = build_forecast_news_features(
        study.view.events,
        clock=clock,
        start_at=cutoff,
        end_at=cutoff + timedelta(hours=12),
        knowledge_cutoff=cutoff,
    )

    future = frame.loc[pd.to_datetime(frame.timestamp, utc=True) > pd.Timestamp(cutoff)]
    assert not future.empty
    assert (pd.to_datetime(future.available_at, utc=True) == pd.Timestamp(cutoff)).all()
    late_event = next(event for event in study.view.events if event.announcement_available_at > cutoff)
    assert late_event.event_id not in str(frame.to_dict(orient="records"))


def test_p2_workspace_handoff_and_p1_feature_decisions_are_persistent(tmp_path: Path):
    prices = pd.read_csv(FIXTURES / "synthetic_prices.csv").rename(columns={"price": "rt_price"})
    target = tmp_path / "target.csv"
    prices.to_csv(target, index=False)
    p1_report = _touch(tmp_path / "p1" / "report.md")
    flow = FullResearchFlow(
        store=FullFlowStore(tmp_path / "flow.json"),
        output_directory=tmp_path / "runs",
    )
    state = flow.start(
        p1_run=FlowRunReference(
            run_kind="eda",
            run_id="p1",
            artifact_directory=p1_report.parent,
            report_path=p1_report,
        ),
        eda_summary={
            "price": {
                "start_time": "2026-01-05T00:00:00+00:00",
                "end_time": "2026-03-01T23:30:00+00:00",
                "distribution": {"mean": 100.0, "min": -20.0, "max": 300.0},
            }
        },
        information_cutoff=datetime.fromisoformat("2026-03-01T23:30:00+00:00"),
    )
    state = state.model_copy(
        update={
            "p1_request": state.p1_request.model_copy(
                update={"market": "TEST_MARKET", "timezone": "UTC", "interval_minutes": 30}
            )
        }
    )
    flow.store.save(state)

    state = flow.run_p2(
        state=state,
        news_path=FIXTURES / "synthetic_news.jsonl",
        target_path=target,
        extractor=ObviousNewsEventExtractor(market_timezone="UTC"),
    )
    assert state.phase == "p2_ready"
    assert state.p2_feature_path and state.p2_feature_path.is_file()
    assert state.p2_manifest_path and state.p2_manifest_path.is_file()
    review_queue = json.loads(
        (state.runs[-1].artifact_directory / "review-queue.json").read_text(encoding="utf-8")
    )
    assert len(review_queue["extractions"]) == 10
    assert all(item["result"]["pass_results"] == [] for item in review_queue["extractions"])
    market = json.loads(
        (state.runs[-1].artifact_directory.parents[1] / "market.json").read_text(encoding="utf-8")
    )
    assert market == {"market": "TEST_MARKET", "timezone": "UTC", "interval_minutes": 30}

    workspace = NewsWorkspace(state.runs[-1].artifact_directory.parents[1])
    with workspace.connect() as database:
        cache_key = database.execute("SELECT cache_key FROM extractions ORDER BY created_at LIMIT 1").fetchone()[0]
    workspace.review(
        cache_key,
        decision="accepted",
        reviewer="integration-test",
        reason="确认抽取事实，验证复核后的可用时间不会被回写",
        market_timezone="UTC",
    )
    retry = state.model_copy(update={"phase": "p2_needs_review"})
    state = flow.run_p2(
        state=retry,
        news_path=FIXTURES / "synthetic_news.jsonl",
        target_path=target,
        extractor=ObviousNewsEventExtractor(market_timezone="UTC"),
    )
    assert state.phase == "p2_ready"
    assert state.p2_knowledge_cutoff and state.p2_knowledge_cutoff > state.p1_request.information_cutoff
    resumed_manifest = json.loads(state.p2_manifest_path.read_text(encoding="utf-8"))
    assert datetime.fromisoformat(resumed_manifest["information_cutoff"]) == state.p2_knowledge_cutoff

    state = flow.synthesize_p1(state=state)
    assert state.phase == "p1_synthesis_complete"
    assert any(item.decision == "selected" for item in state.feature_decisions)
    assert flow.store.load().runs[-1].parent_run_id == state.runs[-2].run_id


def test_forecast_gate_requires_72_points_in_every_fold_and_feedback_stops(tmp_path: Path):
    moment = datetime(2026, 9, 13, tzinfo=UTC)
    specs = []
    for number in range(4):
        path = _touch(tmp_path / "forecast" / "data" / f"{number}.parquet")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        truth = _touch(tmp_path / "forecast" / "data" / f"truth-{number}.parquet") if number < 3 else None
        specs.append(
            ForecastSnapshotSpec(
                role="backtest" if number < 3 else "future",
                as_of=moment + timedelta(days=number * 7),
                target_start=moment + timedelta(days=number * 7),
                target_end=moment + timedelta(days=number * 7, hours=23, minutes=45),
                path=path,
                fingerprint=digest,
                truth_path=truth,
                truth_fingerprint=hashlib.sha256(truth.read_bytes()).hexdigest()[:12] if truth else None,
            )
        )
    plan = ForecastPlan(
        question="预测山东明天实时电价",
        forecast_start=specs[-1].target_start,
        forecast_end=specs[-1].target_end,
        snapshots=specs,
        data_fingerprint="d" * 12,
    )
    metric = lambda observations, mae: ForecastMetricSet(
        observations=observations, mae=mae, rmse=mae, bias=0
    )
    folds = [
        ForecastFoldResult(
            anchor=spec.as_of,
            target_start=spec.target_start,
            target_end=spec.target_end,
            model=metric(points, 1),
            persistence=metric(points, 4),
            day_naive=metric(points, 3),
            week_naive=metric(points, 2),
            prediction_path=spec.path,
            snapshot_fingerprint=spec.fingerprint,
            common_observations=points,
        )
        for spec, points in zip(specs[:3], (71, 73, 72), strict=True)
    ]
    report = _touch(tmp_path / "forecast" / "report.md")
    result = ForecastRunResult(
        run_id="p3",
        plan=plan,
        folds=folds,
        aggregate={
            "model": metric(216, 1),
            "persistence": metric(216, 4),
            "day_naive": metric(216, 3),
            "week_naive": metric(216, 2),
        },
        prediction_path=_touch(tmp_path / "forecast" / "prediction.csv"),
        backtest_path=_touch(tmp_path / "forecast" / "backtests.parquet"),
        metrics_path=_touch(tmp_path / "forecast" / "metrics.json"),
        report_path=report,
        artifact_directory=report.parent,
        figure_paths={},
        output_hash="a" * 64,
    )
    assert result.evaluation_status == "not_evaluable"
    assert not result.baseline_verified

    p1_report = _touch(tmp_path / "p1" / "report.md")
    flow = FullResearchFlow(store=FullFlowStore(tmp_path / "flow.json"), output_directory=tmp_path / "runs")
    state = flow.start(
        p1_run=FlowRunReference(run_kind="eda", run_id="p1", artifact_directory=p1_report.parent, report_path=p1_report),
        eda_summary={"price": {"start_time": moment.isoformat(), "end_time": moment.isoformat(), "distribution": {}}},
        information_cutoff=moment,
    ).model_copy(update={"phase": "forecast_running", "forecast_runs_used": 1})
    finished = flow._finish_forecast(state, result)
    assert finished.phase == "feedback_complete"
    assert finished.feedback_runs_used == 1
    assert finished.forecast_runs_used == 1
    assert finished.feedback and finished.feedback.status == "not_evaluable"
    with pytest.raises(ValueError, match="没有可批准"):
        flow.approve_and_run_p3(
            state=finished,
            plan_id=plan.plan_id,
            plan_fingerprint="irrelevant",
        )


def test_forecast_execution_failure_is_persisted_without_becoming_bad_performance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    moment = datetime(2026, 9, 13, tzinfo=UTC)
    specs = []
    for number in range(4):
        path = _touch(tmp_path / "forecast" / "data" / f"failure-{number}.parquet")
        truth = _touch(tmp_path / "forecast" / "data" / f"failure-truth-{number}.parquet") if number < 3 else None
        specs.append(
            ForecastSnapshotSpec(
                role="backtest" if number < 3 else "future",
                as_of=moment + timedelta(days=number * 7),
                target_start=moment + timedelta(days=number * 7),
                target_end=moment + timedelta(days=number * 7, hours=23, minutes=45),
                path=path,
                fingerprint=hashlib.sha256(path.read_bytes()).hexdigest()[:12],
                truth_path=truth,
                truth_fingerprint=hashlib.sha256(truth.read_bytes()).hexdigest()[:12] if truth else None,
            )
        )
    plan = ForecastPlan(
        plan_id="plan-failure",
        question="预测山东明天实时电价",
        forecast_start=specs[-1].target_start,
        forecast_end=specs[-1].target_end,
        snapshots=specs,
        data_fingerprint="f" * 12,
    )
    p1_report = _touch(tmp_path / "p1" / "report.md")
    flow = FullResearchFlow(store=FullFlowStore(tmp_path / "flow.json"), output_directory=tmp_path / "runs")
    state = flow.start(
        p1_run=FlowRunReference(run_kind="eda", run_id="p1", artifact_directory=p1_report.parent, report_path=p1_report),
        eda_summary={"price": {"start_time": moment.isoformat(), "end_time": moment.isoformat(), "distribution": {}}},
        information_cutoff=moment,
    ).model_copy(
        update={
            "phase": "awaiting_forecast_approval",
            "forecast_plan": plan.model_dump(mode="json"),
            "forecast_plan_fingerprint": "fingerprint",
        }
    )
    monkeypatch.setattr(
        "app.research.full_flow.service.execute_forecast_workflow",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("broken input")),
    )

    with pytest.raises(RuntimeError, match="broken input"):
        flow.approve_and_run_p3(
            state=state,
            plan_id="plan-failure",
            plan_fingerprint="fingerprint",
        )

    persisted = flow.store.load()
    assert persisted.phase == "failed"
    assert persisted.forecast_runs_used == 1
    assert persisted.feedback_runs_used == 0
    assert persisted.feedback is None
    assert "RuntimeError" in (persisted.error or "")


def test_frozen_shandong_news_corpus_has_auditable_collection_metadata():
    corpus = Path(__file__).resolve().parents[2] / "data" / "news" / "shandong_p2_test_news.jsonl"
    records = JsonlCollectedNewsAdapter(corpus).load()
    assert 1 <= len(records) <= 20
    assert all("CN-SHANDONG" in record.market_tags for record in records)
    for record in records:
        assert record.source_ref.startswith("https://")
        assert record.metadata["query_source"]
        assert record.metadata["content_scope"]
        expected = hashlib.sha256(record.body.encode("utf-8")).hexdigest()
        assert record.metadata["content_sha256"] == expected
    manifest = json.loads(corpus.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert manifest["record_count"] == len(records)
    assert manifest["source_file_sha256"] == hashlib.sha256(corpus.read_bytes()).hexdigest()
    assert manifest["browser_mcp_changed"] is False
