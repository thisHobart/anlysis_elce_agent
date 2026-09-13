"""Exercise a small P1 -> P3 scenario with the configured model.

Real data is the default; --synthetic explicitly selects a toy regional source.
Only the test-process forecast trainer is reduced to two epochs / 64 samples.
All outputs are isolated. Exit 0 means checks passed, 2 means observed issues,
and an exception means the main flow could not finish.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication

from app.desktop.main_window import MainWindow
from app.desktop.session import SessionStore
from app.desktop.theme import APP_STYLE
from app.research.data.sources.regions import RegionPriceFetch
from app.research.forecasting.data import prepare_forecast_plan
from scripts.validate_desktop_phase1 import _shutdown_window, _wait
from scripts.validate_phase1 import validation_settings


def summarize_evidence(evidence: dict, output: Path) -> list[str]:
    """Summarize observed failures without hiding them behind a completed forecast."""
    issues = []
    if evidence.get("error"):
        issues.append("主流程中断：" + evidence["error"]["message"])
    p1 = evidence.get("03_p1_result", {})
    if p1.get("gate") == "result_limitations":
        issues.append("P1 已生成统计报告，但仍停留在 result_limitations；需核对假设与已计算统计量的映射。")
    explanation = evidence.get("06_explain_forecast", {})
    if explanation.get("status") == "awaiting_plan_approval":
        issues.append("追问已有预测结果被路由成新预测方案，没有回答回测表现与限制。")
    approval = evidence.get("07_text_approve_dispatch", {})
    if approval and approval.get("current_plan") is None:
        issues.append("输入“可以开始”清空了待确认的预测方案，并转入普通 dialogue 分支。")
    short = evidence.get("08_short_followup_dispatch", {})
    if short and short.get("task_kind") != "forecast_prepare":
        issues.append("“预测山东未来24小时”未进入预测准备分支；本项仅验证桌面分派。")
    restart = evidence.get("restart_running", {})
    if restart.get("status") == "running" and not restart.get("busy") and restart.get("confirm_button_hidden"):
        issues.append("重载 running 会话后仍显示执行中，无后台任务且确认按钮隐藏，无法从卡片继续。")
    forecast_result = evidence.get("05_p3_result", {})
    if forecast_result.get("status") == "completed" and forecast_result.get("gate"):
        issues.append("P3 已完成但仍保留 P1 的交互门禁：" + forecast_result["gate"])
    imported = evidence.get("explanation_import", {}).get("latest_run", {})
    if imported and evidence.get("explanation_import", {}).get("imported_eda_summary_present"):
        issues.append("旧图导入同时携带 P3 latest_run 和 P1 eda_summary；需核对评价与计划是否属于同一轮。")
    evidence["issues"] = issues
    evidence["validation_status"] = "blocked" if evidence.get("error") else ("issues_found" if issues else "passed")
    (output / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# P1 → P3 简单案例验证", "", f"数据模式：{evidence.get('data_mode', 'real')}；状态：{evidence['validation_status']}。",
             "训练仅2轮、最多64个样本，本次结果用于流程检查。", "", "## 问题", ""]
    lines.extend(f"- {issue}" for issue in issues)
    lines.extend(["", "完整状态、指标和分派记录见 [evidence.json](evidence.json)。"])
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return issues


def synthetic_fetcher(instant: datetime):
    """Provide a declared toy source; preserve point-in-time availability gates."""
    start = pd.Timestamp(instant).tz_localize(None).normalize() - pd.Timedelta(days=60)

    def fetch(profile, *, output_directory, now=None, progress=None, **_kwargs):
        cutoff = pd.Timestamp(now or instant).tz_localize(None)
        index = pd.date_range(start, cutoff.normalize() + pd.Timedelta(days=2) - pd.Timedelta(minutes=15), freq="15min")
        phase = 2 * np.pi * (index.hour * 4 + index.minute // 15) / 96
        price = 300 + 50 * np.sin(phase)
        target = pd.DataFrame({"timestamp": index, "rt_price": price, "available_at": index + pd.Timedelta(minutes=15)})
        actuals = pd.DataFrame({"timestamp": index, "actual_load": 60000 + 4000 * np.sin(phase), "available_at": index + pd.Timedelta(minutes=15)})
        forecasts = pd.DataFrame({"timestamp": index, "forecast_load": 60000 + 4000 * np.sin(phase), "available_at": index.normalize() - pd.Timedelta(days=1)})
        root = Path(output_directory)
        root.mkdir(parents=True, exist_ok=True)
        frames = [frame.loc[frame.available_at <= cutoff].copy() for frame in (target, actuals, forecasts)]
        for name, frame in zip(("target", "actuals", "forecasts"), frames, strict=True):
            frame.to_parquet(root / f"{name}.parquet", index=False)
        return RegionPriceFetch(
            profile=profile, path=root / "target.parquet", fetched_at=now or instant,
            row_count=len(frames[0]), start_at=frames[0].timestamp.min().to_pydatetime(),
            end_at=frames[0].timestamp.max().to_pydatetime(), duplicate_timestamp_rows=0,
            invalid_rows=0, database_name="SYNTHETIC_NO_DATABASE",
            actuals_path=root / "actuals.parquet", forecasts_path=root / "forecasts.parquet",
            actual_variable_names=("actual_load",), forecast_variable_names=("forecast_load",),
        )

    return fetch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true", help="Use a toy regional source; keep real configured LLM and computation")
    args = parser.parse_args()
    settings = validation_settings(120)
    root = Path(__file__).resolve().parents[1] / "artifacts" / "acceptance"
    output = root / ("p1-to-p3-" + ("synthetic-" if args.synthetic else "real-") + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f"))
    output.mkdir(parents=True)
    os.environ["PRICE_RESEARCH_APP_DATA_DIRECTORY"] = str(output / "app-data")
    os.environ["PRICE_RESEARCH_OUTPUT_DIRECTORY"] = str(output / "research")
    instant = datetime.now(ZoneInfo("Asia/Shanghai"))
    evidence: dict = {"model": settings.llm_model, "data_mode": "synthetic" if args.synthetic else "real",
                      "training": "2 epochs, 64 samples; flow test only", "stage_seconds": {}}
    app = QApplication.instance() or QApplication([])
    # Windows offscreen Qt can have an empty font database.
    font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "msyh.ttc"
    if font_path.is_file():
        QFontDatabase.addApplicationFont(str(font_path))
    app.setFont(QFont("Microsoft YaHei UI", 10))
    app.setStyleSheet(APP_STYLE)
    window = MainWindow(session_store=SessionStore(output / "sessions.json"),
                        region_fetcher=synthetic_fetcher(instant) if args.synthetic else None)
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
        started = time.monotonic()
        _wait(app, lambda: not workspace.is_busy, timeout=timeout, label=label)
        evidence["stage_seconds"][label] = round(time.monotonic() - started, 2)
        save(label)

    def short_prepare(**kwargs):
        if args.synthetic:
            kwargs["now"] = instant
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
        target = pd.read_parquet(workspace.current_session.inputs["target"].path)
        evidence["p1_input_statistics"] = target.rt_price.agg(["count", "mean", "min", "max"]).to_dict()
        distribution = (workspace.current_session.latest_eda_summary or {}).get("price", {}).get("distribution", {})
        evidence["p1_statistics_match"] = all(
            key in distribution and np.isclose(distribution[key], evidence["p1_input_statistics"][key], atol=1e-5)
            for key in ("mean", "min", "max")
        )
        if not evidence["p1_statistics_match"]:
            raise RuntimeError("P1 mean/min/max disagree with source data or are missing")
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
            "finite_predictions": int(np.isfinite(pd.to_numeric(prediction.predicted_rt_price, errors="coerce")).sum()),
            "unique_timestamps": int(prediction.timestamp.nunique()),
            "aggregate": forecast_message.payload["aggregate"],
            "warnings": forecast_message.payload["warnings"],
        }
        if evidence["forecast"]["rows"] != 96 or evidence["forecast"]["finite_predictions"] != 96:
            raise RuntimeError("P3 must produce exactly 96 finite predictions")
        expected_index = pd.date_range(pd.Timestamp(completed.current_plan["forecast_start"]).tz_localize(None), periods=96, freq="15min")
        evidence["forecast"]["correct_time_grid"] = bool(
            pd.DatetimeIndex(pd.to_datetime(prediction.timestamp)).equals(expected_index)
        )
        package = Path(forecast_message.payload["artifact_directory"])
        manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
        evidence["forecast"]["manifest_files_checked"] = len(manifest["files"])
        evidence["forecast"]["manifest_hashes_match"] = all(
            hashlib.sha256((package / name).read_bytes()).hexdigest() == digest
            for name, digest in manifest["files"].items()
        )
        evidence["forecast"]["parent_run_id"] = completed.runs[-1].parent_run_id
        if not evidence["forecast"]["correct_time_grid"] or not evidence["forecast"]["manifest_hashes_match"]:
            raise RuntimeError("P3 timestamp grid or artifact hashes failed validation")

        # Exercise actual desktop routing: this may incorrectly create a new plan.
        workspace.submit_question("这次明天电价预测的结果说明什么？请说明回测表现和限制。")
        wait("06_explain_forecast", 300)
        imported = workspace._legacy_graph_import(workspace.current_session)
        evidence["explanation_import"] = {
            "eda_summary_still_present": workspace.current_session.latest_eda_summary is not None,
            "imported_eda_summary_present": imported.get("eda_summary") is not None,
            "forecast_plan_still_present": (workspace.current_session.current_plan or {}).get("plan_kind") == "forecast",
            "previous_plan_id": completed.current_plan["plan_id"],
            "current_plan_id": (workspace.current_session.current_plan or {}).get("plan_id"),
        }
        evidence["explanation_import"]["latest_run"] = imported.get("latest_run")

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
        issues = summarize_evidence(evidence, output)
        print(json.dumps({"output": str(output)}, ensure_ascii=True), flush=True)
        return 2 if issues else 0
    except Exception as exc:
        evidence["error"] = {"type": type(exc).__name__, "message": str(exc)}
        save("failed")
        summarize_evidence(evidence, output)
        raise
    finally:
        _shutdown_window(app, window, timeout=240)
        print(json.dumps({"output": str(output)}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
