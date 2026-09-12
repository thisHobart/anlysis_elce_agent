"""Exercise a small P1 -> P3 scenario with real data and the configured model.

Only the test-process forecast trainer is reduced to two epochs / 64 samples.
Production source, model settings, user sessions, and database data are untouched.
"""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd
from PySide6.QtWidgets import QApplication

from app.desktop.main_window import MainWindow
from app.desktop.session import SessionStore
from app.desktop.theme import APP_STYLE
from app.research.forecasting.data import prepare_forecast_plan
from scripts.validate_desktop_phase1 import _shutdown_window, _wait
from scripts.validate_phase1 import validation_settings


def main() -> int:
    settings = validation_settings(120)
    root = Path(__file__).resolve().parents[1] / "artifacts" / "acceptance"
    output = root / ("p1-to-p3-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S"))
    output.mkdir(parents=True)
    os.environ["PRICE_RESEARCH_APP_DATA_DIRECTORY"] = str(output / "app-data")
    evidence: dict = {"model": settings.llm_model, "training": "2 epochs, 64 samples; flow test only"}
    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(APP_STYLE)
    window = MainWindow(session_store=SessionStore(output / "sessions.json"))
    window.show()
    workspace = window.workspace

    def save(stage: str) -> None:
        session = workspace.current_session
        evidence[stage] = {
            "status": session.status,
            "data_state": session.data_state,
            "plan_kind": (session.current_plan or {}).get("plan_kind", "eda") if session.current_plan else None,
            "run_kinds": [run.run_kind for run in session.runs],
            "report_path": session.report_path,
            "gate": workspace._active_interrupt_kind,
            "last_messages": [{"kind": m.kind, "content": m.content} for m in session.messages[-4:]],
        }
        (output / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
        workspace.store.save(workspace.sessions)
        app.processEvents()
        window.grab().save(str(output / f"{stage}.png"))
        print(json.dumps({"stage": stage, **evidence[stage]}, ensure_ascii=True), flush=True)

    def wait(label: str, timeout: int = 300) -> None:
        _wait(app, lambda: not workspace.is_busy, timeout=timeout, label=label)
        save(label)

    def short_prepare(**kwargs):
        plan = prepare_forecast_plan(**kwargs)
        plan = plan.model_copy(update={"training": plan.training.model_copy(update={
            "max_epochs": 2, "maximum_training_samples": 64,
        })})
        location = plan.snapshots[-1].path.parent.parent / "provenance" / "forecast_plan.json"
        location.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
        return plan

    def diagnostic_clone(name: str, source):
        clone = source.model_copy(deep=True)
        clone.session_id = name + "-" + source.session_id
        clone.title = "test: " + name
        workspace.sessions.append(clone)
        workspace.current_session_id = clone.session_id
        workspace._render_current()
        return clone

    try:
        save("00_new_session")
        workspace.select_region("shandong")
        wait("01_shandong_data", 240)
        if workspace.current_session.data_state != "ready":
            raise RuntimeError("Shandong fetch failed")
        workspace.submit_question("只分析山东实时电价的分布，给出均值、最低价、最高价；这是一个简单测试，不做外生变量分析。")
        wait("02_p1_plan")
        if workspace.current_session.status != "awaiting_plan_approval":
            raise RuntimeError("P1 did not reach approval")
        workspace.conversation.current_plan_widget.run_button.click()
        wait("03_p1_result")
        if not workspace.current_session.report_path:
            raise RuntimeError("P1 produced no report")
        p1 = workspace.current_session.model_copy(deep=True)
        with patch("app.desktop.workspace.prepare_forecast_plan", short_prepare):
            workspace.submit_question("预测山东明天实时电价")
            wait("04_p3_plan", 420)
        if workspace.current_session.status != "awaiting_plan_approval":
            raise RuntimeError("P3 did not reach approval")
        approved_state = workspace.current_session.model_copy(deep=True)
        evidence["p1_to_p3"] = {
            "p1_plan_state_after_transition": [m.payload.get("state") for m in workspace.current_session.messages if m.kind == "data_plan"],
            "p1_graph_exists": workspace.agent.has_thread(p1.session_id),
            "p3_uses_p1_references": [key for key in workspace.current_session.current_plan if "eda" in key or "parent" in key],
        }
        workspace.conversation.current_plan_widget.run_button.click()
        wait("05_p3_result", 240)
        completed = workspace.current_session.model_copy(deep=True)
        if completed.status != "completed" or not any(r.run_kind == "forecast" for r in completed.runs):
            raise RuntimeError("P3 execution failed")
        forecast_message = next(m for m in reversed(completed.messages) if m.kind == "forecast_result")
        prediction = pd.read_csv(forecast_message.payload["prediction_path"])
        evidence["forecast"] = {
            "rows": len(prediction),
            "finite_predictions": int(pd.to_numeric(prediction.predicted_rt_price).notna().sum()),
            "aggregate": forecast_message.payload["aggregate"],
            "warnings": forecast_message.payload["warnings"],
        }

        # Follow-up uses the real dialogue gateway after the end-to-end run.
        workspace.submit_question("这次明天电价预测的结果说明什么？请说明回测表现和限制。")
        wait("06_explain_forecast", 300)
        evidence["explanation_import"] = {
            "eda_summary_still_present": workspace.current_session.latest_eda_summary is not None,
            "forecast_plan_still_present": (workspace.current_session.current_plan or {}).get("plan_kind") == "forecast",
        }

        # Capture dispatch without starting extra model/database work.
        for name, state, question in (
            ("07_text_approve", approved_state, "可以开始"),
            ("08_short_followup", p1, "预测山东未来24小时"),
        ):
            clone = diagnostic_clone(name, state)
            with patch.object(workspace, "_start_worker") as dispatched:
                workspace.submit_question(question)
                evidence[name + "_dispatch"] = {
                    "task_kind": dispatched.call_args.kwargs["kind"] if dispatched.called else None,
                    "current_plan": (clone.current_plan or {}).get("plan_kind"),
                }
            save(name)

        # Simulate a process exit during forecast execution using its persisted state.
        interrupted = diagnostic_clone("09_restart_running", approved_state)
        interrupted.status = "running"
        for message in interrupted.messages:
            if message.kind == "forecast_plan":
                message.payload["state"] = "running"
        restart_store = SessionStore(output / "restart-sessions.json")
        restart_store.save([interrupted])
        reopened = MainWindow(session_store=restart_store)
        reopened.workspace.select_session(interrupted.session_id)
        card = reopened.workspace.conversation.current_plan_widget
        evidence["restart_running"] = {
            "status": reopened.workspace.current_session.status,
            "busy": reopened.workspace.is_busy,
            "confirm_button_hidden": card.run_button.isHidden(),
            "card_status": card.status_label.text(),
        }
        reopened.close()
        workspace.current_session_id = completed.session_id
        workspace._render_current()
        save("10_finished")
        print(json.dumps({"output": str(output)}, ensure_ascii=True), flush=True)
        return 0
    except Exception as exc:
        evidence["error"] = {"type": type(exc).__name__, "message": str(exc)}
        save("failed")
        raise
    finally:
        _shutdown_window(app, window, timeout=240)
        print(json.dumps({"output": str(output)}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
