"""Exercise the real Phase 1 user flow through the PySide6 workspace."""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from app.desktop.main_window import MainWindow
from app.desktop.session import SessionStore
from app.desktop.theme import APP_STYLE
from scripts.validate_phase1 import VALIDATION_MINIMUM_TIMEOUT_SECONDS, validation_settings


def _wait(app: QApplication, predicate, *, timeout: float, label: str) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(0.02)
    raise TimeoutError(f"Timed out waiting for {label}")


def _last_error(window: MainWindow) -> str:
    errors = [message.content for message in window.workspace.current_session.messages if message.kind == "error"]
    return errors[-1] if errors else "unknown desktop failure"


def _desktop_progress(message: str) -> None:
    print(json.dumps({"desktop_status": message}, ensure_ascii=True), flush=True)


def _shutdown_window(
    app: QApplication,
    window: MainWindow,
    *,
    timeout: float,
) -> None:
    """Cooperatively finish a busy Worker before invoking the modal close guard."""

    if window.workspace.is_busy:
        window.workspace.cancel_current_task()
        _wait(
            app,
            lambda: not window.workspace.is_busy,
            timeout=timeout,
            label="desktop worker cancellation before shutdown",
        )
    if window.isVisible():
        window.close()
    app.processEvents()


def _retry_interrupt(
    app: QApplication,
    window: MainWindow,
    *,
    kind: str,
    timeout: float,
    label: str,
) -> None:
    if window.workspace._active_interrupt_kind != kind:
        return
    window.workspace.submit_question("重试")
    _wait(app, lambda: not window.workspace.is_busy, timeout=timeout, label=label)


def main() -> int:
    settings = validation_settings(VALIDATION_MINIMUM_TIMEOUT_SECONDS)
    root = Path(__file__).resolve().parents[1]
    acceptance_root = root / "artifacts" / "acceptance"
    acceptance_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    store = SessionStore(acceptance_root / f"desktop-sessions-{stamp}.json")
    app = QApplication.instance() or QApplication([])
    app.setFont(QFont("Microsoft YaHei UI", 10))
    app.setStyleSheet(APP_STYLE)
    window = MainWindow(session_store=store, plan_feedback_seconds=30)
    window.workspace.status_changed.connect(_desktop_progress)
    window.show()
    workspace = window.workspace
    workspace.set_input_file("target", str(root / "data" / "target_rt_price.csv"))
    workspace.set_input_file("actuals", str(root / "data" / "feature_actuals.csv"))
    workspace.set_input_file("forecasts", str(root / "data" / "feature_forecasts.csv"))
    session_id = workspace.current_session.session_id
    restored: MainWindow | None = None
    try:
        workspace.submit_question(
            "分析 actual_load、forecast_total_generation 与实时电价的同期和领先滞后关系。"
        )
        _wait(
            app,
            lambda: not workspace.is_busy,
            timeout=180,
            label="real model proposal",
        )
        _retry_interrupt(
            app,
            window,
            kind="plan_error",
            timeout=180,
            label="real model proposal retry",
        )
        if workspace.current_session.status == "failed":
            raise RuntimeError(_last_error(window))
        if workspace.current_session.status != "awaiting_plan_approval":
            raise RuntimeError(f"Expected plan approval state, got {workspace.current_session.status}")

        workspace.submit_question(
            "把最大滞后改为48个15分钟间隔，只分析actual_load和forecast_total_generation，"
            "关系分析保留Pearson、Spearman和领先滞后。"
        )
        _wait(app, lambda: not workspace.is_busy, timeout=180, label="real model revision")
        _retry_interrupt(
            app,
            window,
            kind="plan_error",
            timeout=180,
            label="real model revision retry",
        )
        if workspace.current_session.status == "failed":
            raise RuntimeError(_last_error(window))
        revised = workspace.current_session.current_plan
        if revised is None or int(revised["revision"]) < 2:
            raise RuntimeError("The desktop did not persist a revised plan.")

        workspace.submit_question("按这个执行")
        _wait(app, lambda: not workspace.is_busy, timeout=240, label="approved function execution")
        if workspace.current_session.status == "awaiting_user":
            if workspace._active_interrupt_kind != "result_limitations":
                raise RuntimeError(
                    f"Unexpected user gate after execution: {workspace._active_interrupt_kind}"
                )
            if not workspace.current_session.report_path:
                raise RuntimeError("A limited desktop result did not preserve its report path.")
            workspace.submit_question("这些现有结果说明什么？只引用真实执行过的证据。")
            _wait(app, lambda: not workspace.is_busy, timeout=180, label="limited-result explanation")
            _retry_interrupt(
                app,
                window,
                kind="response_error",
                timeout=180,
                label="limited-result explanation retry",
            )
            if (
                workspace.current_session.status != "awaiting_user"
                or workspace._active_interrupt_kind != "result_limitations"
            ):
                raise RuntimeError("A limited-result follow-up escaped its result gate.")
            first_explanation = workspace.current_session.messages[-1].content
            workspace.submit_question("接受当前限制")
            _wait(
                app,
                lambda: not workspace.is_busy and workspace.current_session.status == "stopped",
                timeout=180,
                label="stop after preserving bounded-loop evidence",
            )
        elif workspace.current_session.status == "completed":
            workspace.submit_question("这些结果说明什么？请明确证据限制。")
            _wait(app, lambda: not workspace.is_busy, timeout=180, label="result explanation")
            _retry_interrupt(
                app,
                window,
                kind="response_error",
                timeout=180,
                label="result explanation retry",
            )
            if workspace.current_session.status == "failed":
                raise RuntimeError(_last_error(window))
            if workspace.current_session.status != "completed":
                raise RuntimeError(
                    f"Result explanation did not return to the completed result gate: "
                    f"{workspace.current_session.status}"
                )
            first_explanation = workspace.current_session.messages[-1].content
        else:
            raise RuntimeError(
                f"Expected completed or limited result state, got {workspace.current_session.status}"
            )
        completed = workspace.current_session
        if not completed.report_path or not Path(completed.report_path).is_file():
            raise RuntimeError("The desktop did not produce a report.")
        if not Path(completed.report_path).with_name("methods.md").is_file():
            raise RuntimeError("The desktop did not produce methods.md beside report.md.")
        report_path = workspace.current_session.report_path
        run_id = workspace.current_session.run_id
        terminal_status = workspace.current_session.status
        window.close()
        app.processEvents()

        restored = MainWindow(session_store=store)
        restored.workspace.status_changed.connect(_desktop_progress)
        restored.show()
        restored.workspace.select_session(session_id)
        restored_session = restored.workspace.current_session
        if restored_session.status != terminal_status or restored_session.latest_eda_summary is None:
            raise RuntimeError("Completed evidence was not restored from the desktop session store.")
        restored.workspace.submit_question("基于已保存结果，用一句话总结最重要的限制。")
        _wait(app, lambda: not restored.workspace.is_busy, timeout=180, label="restored-session follow-up")
        _retry_interrupt(
            app,
            restored,
            kind="response_error",
            timeout=180,
            label="restored-session follow-up retry",
        )
        if restored.workspace.current_session.status == "failed":
            raise RuntimeError(_last_error(restored))
        restored_explanation = restored.workspace.current_session.messages[-1].content
        final_plan = restored.workspace.current_session.current_plan or revised

        evidence = {
            "status": "passed",
            "model": settings.llm_model,
            "structured_mode": "native",
            "session_id": session_id,
            "plan_revision": final_plan["revision"],
            "skill": {"name": final_plan["skill_name"], "version": final_plan["skill_version"]},
            "run_id": run_id,
            "terminal_status": terminal_status,
            "report_path": report_path,
            "first_explanation": first_explanation,
            "restored_explanation": restored_explanation,
            "session_store": str(store.path),
        }
        evidence_path = acceptance_root / f"desktop-acceptance-{stamp}.json"
        evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({**evidence, "first_explanation": first_explanation[:160]}, ensure_ascii=True, indent=2))
        restored.close()
        app.processEvents()
        return 0
    finally:
        cleanup_timeout = max(
            60.0,
            settings.llm_timeout_seconds * (settings.llm_max_retries + 1) + 30.0,
        )
        if restored is not None:
            _shutdown_window(app, restored, timeout=cleanup_timeout)
        _shutdown_window(app, window, timeout=cleanup_timeout)


if __name__ == "__main__":
    raise SystemExit(main())
