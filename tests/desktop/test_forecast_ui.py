"""P3 forecast cards remain explicit, typed, and restorable."""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QLabel

from app.desktop.main_window import MainWindow
from app.desktop.message_widgets import ForecastPlanMessageWidget, ForecastResultMessageWidget
from app.desktop.panes import ConversationPane
from app.desktop.session import (
    ResearchSession,
    SessionMessage,
    SessionRunRecord,
    SessionStore,
    plan_is_executable,
)
from app.desktop.workspace import (
    _is_forecast_request,
    _is_news_analysis_request,
    _requests_news_forecast,
)
from app.research.agent.orchestrator import DialogueDecision, MainResearchAgent
from app.research.agent.subagents.eda import EDASubagent
from app.research.application.coordinator import ResearchCoordinator
from app.research.forecasting.contracts import ForecastPlan, ForecastSnapshotSpec
from app.research.graph.contracts import InterruptPayload, ResearchLoopSnapshot


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _plan(tmp_path: Path) -> ForecastPlan:
    zone = ZoneInfo("Asia/Shanghai")
    anchors = [datetime(2026, 8, day, tzinfo=zone) for day in (1, 8, 15)]
    future = datetime(2026, 9, 13, tzinfo=zone)
    snapshots = []
    for number, anchor in enumerate(anchors, start=1):
        path = tmp_path / f"backtest-{number}.parquet"
        truth = tmp_path / f"truth-{number}.parquet"
        path.touch()
        truth.touch()
        snapshots.append(
            ForecastSnapshotSpec(
                role="backtest",
                as_of=anchor,
                target_start=anchor,
                target_end=anchor + timedelta(hours=23, minutes=45),
                path=path,
                truth_path=truth,
                truth_fingerprint="a" * 12,
                fingerprint=f"{number}" * 12,
            )
        )
    future_path = tmp_path / "future.parquet"
    future_path.touch()
    snapshots.append(
        ForecastSnapshotSpec(
            role="future",
            as_of=future - timedelta(hours=1),
            target_start=future,
            target_end=future + timedelta(hours=23, minutes=45),
            path=future_path,
            fingerprint="f" * 12,
        )
    )
    return ForecastPlan(
        question="预测山东明天实时电价",
        forecast_start=future,
        forecast_end=future + timedelta(hours=23, minutes=45),
        snapshots=snapshots,
        data_fingerprint="d" * 12,
    )


def test_forecast_plan_card_requires_explicit_click_and_restores(
    qt_app: QApplication, tmp_path: Path
):
    plan = _plan(tmp_path)
    pane = ConversationPane()
    message = SessionMessage(
        role="assistant",
        kind="forecast_plan",
        payload={"plan": plan.model_dump(mode="json"), "state": "awaiting"},
    )

    widget = pane.render_message(message)

    assert isinstance(widget, ForecastPlanMessageWidget)
    assert widget.status_label.text() == "等待你确认"
    assert not widget.run_button.isHidden()
    assert plan_is_executable(plan.model_dump(mode="json"))
    pane.close()
    qt_app.processEvents()


def test_forecast_result_card_highlights_unverified_gain(qt_app: QApplication):
    widget = ForecastResultMessageWidget(
        {
            "aggregate": {
                "model": {"mae": 32.0},
                "day_naive": {"mae": 28.0},
                "week_naive": {"mae": 30.0},
            },
            "warnings": ["未验证出预测增益"],
        }
    )

    labels = [child.text() for child in widget.findChildren(QLabel)]
    assert any("模型 32.00" in text for text in labels)
    assert any("未验证出预测增益" in text for text in labels)
    widget.close()
    qt_app.processEvents()


def test_forecast_dialogue_intents_are_versioned():
    assert DialogueDecision(intent="new_forecast_plan").intent == "new_forecast_plan"
    assert DialogueDecision(intent="execute_forecast_plan").intent == "execute_forecast_plan"


def test_forecast_wording_matches_one_day_but_combined_analysis_stays_in_p1():
    assert _is_forecast_request("开始预测一天的电价", has_price_context=True)
    assert not _is_forecast_request(
        "分析山东电价以及相关数据，开始预测一天的电价",
        has_price_context=True,
    )


def test_news_then_forecast_wording_enters_the_owned_full_flow():
    question = "根据已经分析的数据，开始新闻分析最后电价预测"

    assert _is_news_analysis_request(question)
    assert _requests_news_forecast(question)
    assert not _is_forecast_request(question, has_price_context=True)
    assert not _is_news_analysis_request("这个工作台支持新闻分析吗？")
    assert not _is_news_analysis_request("解释一下已有的新闻分析结果")


@pytest.mark.functional
def test_model_forecast_handoff_prepares_a_card_before_showing_confirmation(
    qt_app: QApplication,
    tmp_path: Path,
):
    window = MainWindow(session_store=SessionStore(tmp_path / "sessions.json"))
    workspace = window.workspace
    snapshot = ResearchLoopSnapshot(
        thread_id=workspace.current_session.session_id,
        phase="awaiting_user",
        values={
            "control": "forecast_plan_request",
            "latest_turn": "开始预测一天的电价",
            "assistant_message": "请确认是否执行该预测方案。",
        },
    )
    try:
        with patch.object(workspace, "_submit_forecast_request") as prepare:
            workspace._loop_completed(snapshot)

            assert workspace._pending_forecast_request == "开始预测一天的电价"
            assert not any("请确认是否执行" in message.content for message in workspace.current_session.messages)

            workspace._thread_finished()
            qt_app.processEvents()

        prepare.assert_called_once_with("开始预测一天的电价", record_user_message=False)
    finally:
        window.close()
        qt_app.processEvents()


@pytest.mark.functional
def test_combined_request_prepares_p3_only_after_p1_result(
    qt_app: QApplication,
    tmp_path: Path,
):
    window = MainWindow(session_store=SessionStore(tmp_path / "sessions.json"))
    workspace = window.workspace
    session = workspace.current_session
    question = "分析山东电价以及相关数据，开始预测一天的电价"
    session.pending_forecast_question = question
    snapshot = ResearchLoopSnapshot(
        thread_id=session.session_id,
        phase="completed",
        values={
            "control": "result",
            "current_plan": {"plan_id": "p1-plan", "question": question},
            "latest_run": {
                "run_id": "p1-run",
                "plan_id": "p1-plan",
                "artifact_directory": str(tmp_path / "p1"),
                "report_path": str(tmp_path / "p1" / "report.md"),
                "evaluation": {"decision": "accept", "summary": "P1分析完成。"},
                "figure_paths": {},
            },
            "assistant_message": "P1分析完成。",
        },
        interrupt=InterruptPayload(
            kind="result",
            phase="completed",
            message="P1分析完成。",
            choices=["followup", "next_round", "stop"],
            result={},
        ),
    )
    try:
        with patch.object(workspace, "_submit_forecast_request") as prepare:
            workspace._loop_completed(snapshot)

            assert workspace._pending_forecast_request == question
            assert session.pending_forecast_question is None

            workspace._thread_finished()
            qt_app.processEvents()

        prepare.assert_called_once_with(question, record_user_message=False)
    finally:
        window.close()
        qt_app.processEvents()


def test_old_forecast_algorithm_plan_requires_a_new_plan(tmp_path: Path):
    payload = _plan(tmp_path).model_dump(mode="json")
    payload["algorithm_version"] = "1.0.0+ctm_base_v4_weather_interaction_source"
    assert not plan_is_executable(payload)


def test_unselected_region_never_enters_forecast_branch(qt_app: QApplication, tmp_path: Path):
    class ForecastDialogue:
        enabled = True
        model_name = "forecast-route-test"

        def decide(self, **_kwargs):
            return DialogueDecision(intent="new_forecast_plan")

    class UnusedPlanner:
        enabled = True
        model_name = "forecast-route-test"

    coordinator = ResearchCoordinator(
        main_agent=MainResearchAgent(model_dialogue=ForecastDialogue()),
        eda_subagent=EDASubagent(model_planner=UnusedPlanner()),
    )
    window = MainWindow(
        agent=coordinator,
        session_store=SessionStore(tmp_path / "sessions.json"),
        region_profiles={},
    )
    try:
        window.workspace.submit_question("预测山东明天实时电价")

        deadline = time.monotonic() + 5
        while window.workspace.current_session.status != "failed" and time.monotonic() < deadline:
            qt_app.processEvents()
            time.sleep(0.01)

        session = window.workspace.current_session
        assert session.current_plan is None
        assert session.status == "failed"
        assert "还没有选择地区" in session.messages[-1].content
        assert not window.workspace.is_busy
    finally:
        window.close()
        coordinator.close()
        qt_app.processEvents()


def test_text_approval_uses_the_same_forecast_execution_path(qt_app: QApplication, tmp_path: Path):
    window = MainWindow(session_store=SessionStore(tmp_path / "sessions.json"))
    try:
        plan = _plan(tmp_path)
        session = window.workspace.current_session
        session.current_plan = plan.model_dump(mode="json")
        session.status = "awaiting_plan_approval"
        session.messages.append(
            SessionMessage(
                role="assistant",
                kind="forecast_plan",
                payload={"plan": session.current_plan, "state": "awaiting"},
            )
        )
        window.workspace._render_current()
        with patch.object(window.workspace, "_run_forecast_plan") as execute:
            window.workspace.submit_question("可以开始")
        execute.assert_called_once()
        assert execute.call_args.args[0].plan_id == plan.plan_id
        assert session.current_plan is not None
    finally:
        window.close()
        qt_app.processEvents()


def test_forecast_result_question_does_not_prepare_another_plan(qt_app: QApplication, tmp_path: Path):
    window = MainWindow(session_store=SessionStore(tmp_path / "sessions.json"))
    try:
        plan = _plan(tmp_path)
        session = window.workspace.current_session
        session.current_plan = plan.model_dump(mode="json")
        session.status = "completed"
        session.messages.append(
            SessionMessage(
                role="assistant",
                kind="forecast_result",
                payload={
                    "aggregate": {
                        "model": {"mae": 30.0},
                        "day_naive": {"mae": 35.0},
                        "week_naive": {"mae": 40.0},
                    },
                    "diagnostics": {"evaluation_status": "verified"},
                    "warnings": [],
                },
            )
        )
        session.runs.append(
            SessionRunRecord(
                run_id="forecast-run",
                run_kind="forecast",
                plan_id=plan.plan_id,
                question=plan.question,
                artifact_directory=str(tmp_path),
                report_path=str(tmp_path / "report.md"),
            )
        )
        with patch.object(window.workspace, "_submit_forecast_request") as prepare:
            window.workspace.submit_question("这次明天电价预测的结果说明什么？请说明回测表现和限制。")
        prepare.assert_not_called()
        assert session.status == "completed"
        assert "模型聚合 MAE 为 30.00" in session.messages[-1].content
    finally:
        window.close()
        qt_app.processEvents()


def test_reloaded_running_forecast_requires_new_confirmation(qt_app: QApplication, tmp_path: Path):
    plan = _plan(tmp_path)
    session_path = tmp_path / "sessions.json"
    store = SessionStore(session_path)
    store.save(
        [
            ResearchSession(
                status="running",
                current_plan=plan.model_dump(mode="json"),
                messages=[
                    SessionMessage(
                        role="assistant",
                        kind="forecast_plan",
                        payload={"plan": plan.model_dump(mode="json"), "state": "running"},
                    )
                ],
            )
        ]
    )

    window = MainWindow(session_store=SessionStore(session_path))
    try:
        restored = next(item for item in window.workspace.sessions if item.current_plan)
        window.workspace.select_session(restored.session_id)
        assert restored.status == "awaiting_plan_approval"
        assert restored.messages[0].payload["state"] == "awaiting"
        assert "上次预测运行已中断" in restored.messages[-1].content
        assert window.workspace.conversation.current_plan_widget is not None
        assert not window.workspace.conversation.current_plan_widget.run_button.isHidden()
    finally:
        window.close()
        qt_app.processEvents()


def test_legacy_import_binds_forecast_summary_to_forecast_run(qt_app: QApplication, tmp_path: Path):
    window = MainWindow(session_store=SessionStore(tmp_path / "sessions.json"))
    try:
        plan = _plan(tmp_path)
        session = window.workspace.current_session
        session.current_plan = plan.model_dump(mode="json")
        session.latest_eda_summary = {"price": {"distribution": {"mean": 123}}}
        session.latest_evaluation = {"decision": "need_user", "summary": "old P1"}
        session.runs.extend(
            [
                SessionRunRecord(
                    run_id="p1",
                    run_kind="eda",
                    plan_id="p1-plan",
                    question="分析",
                    artifact_directory=str(tmp_path / "p1"),
                    report_path=str(tmp_path / "p1" / "report.md"),
                    evaluation={"decision": "need_user", "summary": "old P1"},
                ),
                SessionRunRecord(
                    run_id="p3",
                    run_kind="forecast",
                    plan_id=plan.plan_id,
                    question=plan.question,
                    artifact_directory=str(tmp_path / "p3"),
                    report_path=str(tmp_path / "p3" / "report.md"),
                    evaluation={"decision": "need_user", "summary": "P3 not evaluable"},
                ),
            ]
        )
        session.run_id = "p3"
        session.artifact_directory = str(tmp_path / "p3")
        session.report_path = str(tmp_path / "p3" / "report.md")

        imported = window.workspace._legacy_graph_import(session)

        assert imported["latest_run"]["evaluation"]["summary"] == "P3 not evaluable"
        assert imported["evaluation"]["summary"] == "P3 not evaluable"
        assert imported["eda_summary"] is None
    finally:
        window.close()
        qt_app.processEvents()
