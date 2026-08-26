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

from app.config import get_settings
from app.desktop.main_window import MainWindow
from app.desktop.session import SessionStore
from app.desktop.theme import APP_STYLE


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


def main() -> int:
    settings = get_settings()
    root = Path(__file__).resolve().parents[1]
    acceptance_root = root / "artifacts" / "acceptance"
    acceptance_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    store = SessionStore(acceptance_root / f"desktop-sessions-{stamp}.json")
    config_path = root / "configs" / "research" / "price_exogenous_eda.yaml"
    app = QApplication.instance() or QApplication([])
    app.setFont(QFont("Microsoft YaHei UI", 10))
    app.setStyleSheet(APP_STYLE)
    window = MainWindow(config_path=config_path, session_store=store, plan_feedback_seconds=30)
    window.show()
    workspace = window.workspace
    session_id = workspace.current_session.session_id
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
        if workspace.current_session.status == "failed":
            raise RuntimeError(_last_error(window))
        if workspace.current_session.status != "awaiting_plan_approval":
            raise RuntimeError(f"Expected plan approval state, got {workspace.current_session.status}")

        workspace.submit_question(
            "把最大滞后改为48个15分钟间隔，只分析actual_load和forecast_total_generation，"
            "关系分析保留Pearson、Spearman和领先滞后。"
        )
        _wait(app, lambda: not workspace.is_busy, timeout=180, label="real model revision")
        if workspace.current_session.status == "failed":
            raise RuntimeError(_last_error(window))
        revised = workspace.current_session.current_plan
        if revised is None or int(revised["revision"]) < 2:
            raise RuntimeError("The desktop did not persist a revised plan.")

        workspace.submit_question("按这个执行")
        _wait(app, lambda: not workspace.is_busy, timeout=240, label="approved function execution")
        if workspace.current_session.status == "awaiting_user":
            workspace.submit_question("接受当前限制")
            _wait(
                app,
                lambda: not workspace.is_busy and workspace.current_session.status == "completed",
                timeout=180,
                label="accept bounded-loop limitations",
            )
        if workspace.current_session.status != "completed":
            raise RuntimeError(f"Expected completed result state, got {workspace.current_session.status}")
        completed = workspace.current_session
        if not completed.report_path or not Path(completed.report_path).is_file():
            raise RuntimeError("The desktop did not produce a report.")

        workspace.submit_question("这些结果说明什么？请明确证据限制。")
        _wait(app, lambda: not workspace.is_busy, timeout=180, label="result explanation")
        if workspace.current_session.status == "failed":
            raise RuntimeError(_last_error(window))
        first_explanation = workspace.current_session.messages[-1].content
        report_path = workspace.current_session.report_path
        run_id = workspace.current_session.run_id
        window.close()
        app.processEvents()

        restored = MainWindow(session_store=store)
        restored.show()
        restored.workspace.select_session(session_id)
        restored_session = restored.workspace.current_session
        if restored_session.status != "completed" or restored_session.latest_eda_summary is None:
            raise RuntimeError("Completed evidence was not restored from the desktop session store.")
        restored.workspace.submit_question("基于已保存结果，用一句话总结最重要的限制。")
        _wait(app, lambda: not restored.workspace.is_busy, timeout=180, label="restored-session follow-up")
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
        if window.isVisible():
            window.close()
        app.processEvents()


if __name__ == "__main__":
    raise SystemExit(main())
