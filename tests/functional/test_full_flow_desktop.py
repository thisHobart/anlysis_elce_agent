"""Functional coverage for the persisted flow as a desktop user sees it."""

from __future__ import annotations

import hashlib
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from app.desktop.main_window import MainWindow
from app.desktop.session import SessionRunRecord, SessionStore
from app.research.forecasting.contracts import ForecastPlan, ForecastSnapshotSpec
from app.research.full_flow import (
    FlowRunReference,
    ForecastFeedback,
    FullFlowStore,
    FullResearchFlow,
)

pytestmark = pytest.mark.functional


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def _plan(tmp_path: Path) -> ForecastPlan:
    start = datetime(2026, 9, 13, tzinfo=UTC)
    snapshots = []
    for number in range(4):
        path = _touch(tmp_path / "snapshots" / f"snapshot-{number}.parquet")
        truth = _touch(tmp_path / "snapshots" / f"truth-{number}.parquet") if number < 3 else None
        snapshots.append(
            ForecastSnapshotSpec(
                role="backtest" if number < 3 else "future",
                as_of=start + timedelta(days=number * 7),
                target_start=start + timedelta(days=number * 7),
                target_end=start + timedelta(days=number * 7, hours=23, minutes=45),
                path=path,
                fingerprint=hashlib.sha256(path.read_bytes()).hexdigest()[:12],
                truth_path=truth,
                truth_fingerprint=hashlib.sha256(truth.read_bytes()).hexdigest()[:12] if truth else None,
            )
        )
    return ForecastPlan(
        question="预测山东明天实时电价",
        forecast_start=snapshots[-1].target_start,
        forecast_end=snapshots[-1].target_end,
        snapshots=snapshots,
        data_fingerprint="d" * 12,
    )


def _awaiting_state(tmp_path: Path, plan: ForecastPlan):
    p1_report = _touch(tmp_path / "p1" / "report.md")
    flow = FullResearchFlow(
        store=FullFlowStore(tmp_path / "flow.json"),
        output_directory=tmp_path / "full-flow",
    )
    moment = datetime(2026, 9, 12, tzinfo=UTC)
    state = flow.start(
        p1_run=FlowRunReference(
            run_kind="eda",
            run_id="p1",
            artifact_directory=p1_report.parent,
            report_path=p1_report,
        ),
        eda_summary={
            "price": {
                "start_time": moment.isoformat(),
                "end_time": moment.isoformat(),
                "distribution": {},
            }
        },
        information_cutoff=moment,
    ).model_copy(
        update={
            "phase": "awaiting_forecast_approval",
            "forecast_plan": plan.model_dump(mode="json"),
            "forecast_plan_fingerprint": "flow-plan-fingerprint",
        }
    )
    flow.store.save(state)
    return flow, state


def test_plan_card_click_runs_owned_flow_once_and_result_questions_are_read_only(
    qt_app: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    plan = _plan(tmp_path)
    flow, state = _awaiting_state(tmp_path, plan)
    window = MainWindow(session_store=SessionStore(tmp_path / "sessions.json"))
    workspace = window.workspace
    calls: list[dict[str, Any]] = []

    def complete(self, *, state, plan_id, plan_fingerprint, progress=None):
        calls.append(
            {
                "phase": state.phase,
                "plan_id": plan_id,
                "fingerprint": plan_fingerprint,
            }
        )
        artifact = tmp_path / "forecast"
        feedback = ForecastFeedback(
            forecast_run_id="p3-functional",
            plan_id=plan_id,
            status="not_verified",
            fold_common_observations=(96, 96, 96),
            aggregate_metrics={
                "model": {"observations": 288, "mae": 5.0, "rmse": 6.0, "bias": 0.0},
                "day_naive": {"observations": 288, "mae": 4.0, "rmse": 5.0, "bias": 0.0},
                "week_naive": {"observations": 288, "mae": 4.5, "rmse": 5.5, "bias": 0.0},
            },
            used_news_features=("news_active_event_count",),
            diagnosis=("日前同刻基线更强。",),
            prediction_path=_touch(artifact / "prediction.csv"),
            backtest_path=_touch(artifact / "backtest.parquet"),
            metrics_path=_touch(artifact / "metrics.json"),
            report_path=_touch(artifact / "report.md"),
            artifact_directory=artifact,
            output_hash="a" * 64,
        )
        forecast_run = FlowRunReference(
            run_kind="forecast",
            run_id=feedback.forecast_run_id,
            parent_run_id=state.runs[-1].run_id,
            artifact_directory=artifact,
            report_path=artifact / "report.md",
            data_fingerprint=plan.data_fingerprint,
        )
        feedback_root = tmp_path / "feedback"
        feedback_run = FlowRunReference(
            run_kind="feedback",
            run_id="p1-feedback-functional",
            parent_run_id=feedback.forecast_run_id,
            artifact_directory=feedback_root,
            report_path=_touch(feedback_root / "report.md"),
            data_fingerprint=plan.data_fingerprint,
        )
        finished = state.model_copy(
            update={
                "phase": "feedback_complete",
                "forecast_runs_used": 1,
                "feedback_runs_used": 1,
                "approved_forecast_plan_id": plan_id,
                "runs": (*state.runs, forecast_run, feedback_run),
                "feedback": feedback,
                "stop_reason": "已完成一次P1反馈分析并停止",
            }
        )
        self.store.save(finished)
        return finished

    def run_immediately(*, kind, operation, success_handler, failure_handler=None):
        del failure_handler
        calls.append({"worker_kind": kind})
        success_handler(operation(lambda *_args: None))

    monkeypatch.setattr(FullResearchFlow, "approve_and_run_p3", complete)
    monkeypatch.setattr(workspace, "_start_worker", run_immediately)
    try:
        workspace.update_full_flow_state(
            state,
            state_path=flow.store.path,
            output_directory=flow.output_directory,
        )
        workspace.update_full_flow_state(
            state,
            state_path=flow.store.path,
            output_directory=flow.output_directory,
        )

        plan_messages = [message for message in workspace.current_session.messages if message.kind == "forecast_plan"]
        assert len(plan_messages) == 1
        assert workspace.current_session.status == "awaiting_plan_approval"
        assert workspace.current_session.full_flow_output_directory == str(flow.output_directory.resolve())

        widget = workspace.conversation.current_plan_widget
        assert widget is not None
        widget.run_button.click()

        assert {"worker_kind": "full_flow_forecast_execute"} in calls
        assert any(
            call.get("plan_id") == plan.plan_id
            and call.get("fingerprint") == "flow-plan-fingerprint"
            and call.get("phase") == "awaiting_forecast_approval"
            for call in calls
        )
        assert workspace.current_session.status == "completed"
        assert flow.store.load().forecast_runs_used == 1
        assert sum(message.kind == "forecast_result" for message in workspace.current_session.messages) == 1
        assert [run.run_kind for run in workspace.current_session.runs] == ["eda", "forecast", "feedback"]
        assert workspace.current_session.run_id == "p3-functional"

        calls_before_repeat = len(calls)
        workspace.run_plan(plan)
        assert len(calls) == calls_before_repeat
        assert "不能再次执行" in workspace.current_session.messages[-1].content

        before = len(workspace.current_session.messages)
        workspace.submit_question("这次预测结果和限制是什么？")
        assert len(calls) == 2
        assert len(workspace.current_session.messages) == before + 2
        assert "模型聚合 MAE 为 5.00" in workspace.current_session.messages[-1].content
    finally:
        window.close()
        qt_app.processEvents()


def test_persisted_running_flow_is_reconfirmed_without_resetting_budget(
    qt_app: QApplication,
    tmp_path: Path,
):
    plan = _plan(tmp_path)
    flow, state = _awaiting_state(tmp_path, plan)
    running = state.model_copy(
        update={
            "phase": "forecast_running",
            "forecast_runs_used": 1,
            "approved_forecast_plan_id": plan.plan_id,
        }
    )
    flow.store.save(running)
    session_store = SessionStore(tmp_path / "sessions.json")
    window = MainWindow(session_store=session_store)
    try:
        window.workspace.update_full_flow_state(
            running,
            state_path=flow.store.path,
            output_directory=flow.output_directory,
        )
        session_id = window.workspace.current_session.session_id
    finally:
        window.close()
        qt_app.processEvents()

    reopened = MainWindow(session_store=SessionStore(tmp_path / "sessions.json"))
    try:
        reopened.workspace.select_session(session_id)
        session = reopened.workspace.current_session
        assert session.status == "awaiting_plan_approval"
        assert session.current_plan and session.current_plan["plan_id"] == plan.plan_id
        resolved = reopened.workspace._full_flow_for_plan(plan)
        assert resolved is not None
        assert resolved[1].phase == "forecast_running"
        assert resolved[1].forecast_runs_used == 1
        assert reopened.workspace.conversation.current_plan_widget is not None
    finally:
        reopened.close()
        qt_app.processEvents()


def test_chat_request_reuses_limited_p1_and_reaches_p3_card(
    qt_app: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    window = MainWindow(session_store=SessionStore(tmp_path / "sessions.json"))
    workspace = window.workspace
    session = workspace.current_session
    p1_root = tmp_path / "p1-existing"
    p1_report = _touch(p1_root / "report.md")
    target = _touch(tmp_path / "target.parquet")
    session.region_id = "shandong"
    session.region_timezone = "Asia/Shanghai"
    session.source_kind = "database"
    session.inputs["target"].path = str(target)
    session.dataset_fingerprint = "d" * 12
    session.status = "awaiting_user"
    session.current_plan = {
        "plan_id": "screening-plan",
        "plan_kind": "research",
        "variable_selection_stage": "screening",
    }
    session.latest_eda_summary = {
        "price": {
            "start_time": "2026-01-01T00:00:00+08:00",
            "end_time": "2026-09-12T23:45:00+08:00",
            "distribution": {"mean": 100.0},
        }
    }
    session.runs.append(
        SessionRunRecord(
            run_id="p1-limited",
            plan_id="screening-plan",
            question="分析山东电价以及相关数据",
            artifact_directory=str(p1_root),
            report_path=str(p1_report),
            evaluation={"decision": "need_user", "summary": "等待确认推荐变量"},
            data_fingerprint="d" * 12,
        )
    )
    calls: list[str] = []
    forecast_plan = _plan(tmp_path)

    def run_p2(self, *, state, **_kwargs):
        calls.append("p2")
        package = tmp_path / "p2-package"
        feature_path = _touch(package / "forecast_features.csv")
        report = _touch(package / "report.md")
        news_run = FlowRunReference(
            run_kind="news",
            run_id="p2-chat",
            parent_run_id=state.runs[-1].run_id,
            artifact_directory=package,
            report_path=report,
            data_fingerprint="a" * 12,
        )
        updated = state.model_copy(
            update={
                "phase": "p2_ready",
                "runs": (*state.runs, news_run),
                "p2_feature_path": feature_path,
            }
        )
        self.store.save(updated)
        return updated

    def synthesize(self, *, state):
        calls.append("p1_synthesis")
        package = tmp_path / "p1-synthesis"
        synthesis_run = FlowRunReference(
            run_kind="eda",
            run_id="p1-synthesis-chat",
            parent_run_id=state.runs[-1].run_id,
            artifact_directory=package,
            report_path=_touch(package / "report.md"),
            data_fingerprint="b" * 12,
        )
        updated = state.model_copy(
            update={"phase": "p1_synthesis_complete", "runs": (*state.runs, synthesis_run)}
        )
        self.store.save(updated)
        return updated

    def prepare(self, *, state, **_kwargs):
        calls.append("p3_prepare")
        updated = state.model_copy(
            update={
                "phase": "awaiting_forecast_approval",
                "forecast_plan": forecast_plan.model_dump(mode="json"),
                "forecast_plan_fingerprint": "chat-flow-fingerprint",
            }
        )
        self.store.save(updated)
        return updated

    def run_immediately(*, kind, operation, success_handler, failure_handler=None):
        del failure_handler
        assert kind in {"full_flow_p2", "full_flow_p1_synthesis", "full_flow_p3_prepare"}
        success_handler(operation(lambda *_args: None))
        if workspace._pending_full_flow_resume:
            workspace._pending_full_flow_resume = False
            workspace._resume_full_flow()

    monkeypatch.setattr("app.desktop.workspace.build_model_gateway", lambda _settings: object())
    monkeypatch.setattr("app.desktop.workspace.StructuredNewsEventExtractor", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(FullResearchFlow, "run_p2", run_p2)
    monkeypatch.setattr(FullResearchFlow, "synthesize_p1", synthesize)
    monkeypatch.setattr(FullResearchFlow, "prepare_p3", prepare)
    monkeypatch.setattr(workspace, "_start_worker", run_immediately)
    try:
        workspace.submit_question("根据已经分析的数据，开始新闻分析最后电价预测")

        assert calls == ["p2", "p1_synthesis", "p3_prepare"]
        assert session.status == "awaiting_plan_approval"
        assert session.pending_forecast_question is None
        assert [run.run_kind for run in session.runs] == ["eda", "news", "eda"]
        assert session.current_plan and session.current_plan["plan_id"] == forecast_plan.plan_id
        assert workspace.conversation.current_plan_widget is not None
        assert not any("确认推荐变量" in message.content for message in session.messages)
    finally:
        window.close()
        qt_app.processEvents()
