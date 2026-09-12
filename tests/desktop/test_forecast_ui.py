"""P3 forecast cards remain explicit, typed, and restorable."""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QLabel

from app.desktop.main_window import MainWindow
from app.desktop.message_widgets import ForecastPlanMessageWidget, ForecastResultMessageWidget
from app.desktop.panes import ConversationPane
from app.desktop.session import SessionMessage, SessionStore, plan_is_executable
from app.research.agent.orchestrator import DialogueDecision
from app.research.forecasting.contracts import ForecastPlan, ForecastSnapshotSpec


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


def test_unselected_region_never_enters_forecast_branch(qt_app: QApplication, tmp_path: Path):
    window = MainWindow(
        session_store=SessionStore(tmp_path / "sessions.json"),
        region_profiles={},
    )
    try:
        window.workspace.submit_question("预测山东明天实时电价")

        session = window.workspace.current_session
        assert session.current_plan is None
        assert session.status == "failed"
        assert "还没有选择地区" in session.messages[-1].content
        assert not window.workspace.is_busy
    finally:
        window.close()
        qt_app.processEvents()
