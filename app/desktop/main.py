"""PySide6 desktop application bootstrap."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from app.desktop.main_window import MainWindow
from app.desktop.theme import APP_STYLE
from app.runtime_paths import default_p2_news_path


def _smoke_result_argument(argv: Sequence[str]) -> tuple[list[str], Path | None]:
    """Remove the internal EXE smoke option before Qt parses arguments."""

    cleaned: list[str] = []
    result: Path | None = None
    index = 0
    while index < len(argv):
        argument = str(argv[index])
        if argument == "--smoke-test":
            if index + 1 >= len(argv):
                raise ValueError("--smoke-test requires a result path")
            result = Path(str(argv[index + 1])).resolve()
            index += 2
            continue
        if argument.startswith("--smoke-test="):
            value = argument.partition("=")[2].strip()
            if not value:
                raise ValueError("--smoke-test requires a result path")
            result = Path(value).resolve()
            index += 1
            continue
        cleaned.append(argument)
        index += 1
    return cleaned, result


def _desktop_smoke_payload(window: MainWindow) -> dict[str, Any]:
    """Describe production resources loaded by a fully constructed desktop."""

    metadata = window.workspace.agent.skills.metadata()
    builtin_skills = sorted(
        (
            {"name": str(item["name"]), "version": str(item["version"])}
            for item in metadata
            if item.get("source") == "builtin"
        ),
        key=lambda item: item["name"],
    )
    skill_names = {item["name"] for item in builtin_skills}
    required = {"price-exogenous-eda", "price-forecastability-audit"}
    missing = sorted(required - skill_names)
    function_names = sorted(window.workspace.agent.tools.names)
    if missing:
        raise RuntimeError(f"Built desktop is missing builtin Skills: {', '.join(missing)}")
    if len(function_names) != 30:
        raise RuntimeError(f"Built desktop loaded {len(function_names)} research functions instead of 30")
    from app.research.forecasting.contracts import CTMTrainingConfig
    from app.research.forecasting.model import LightweightCTM

    forecast_model = LightweightCTM(3, CTMTrainingConfig(hidden_size=4, internal_ticks=2))
    forecast_runtime = {
        "model": type(forecast_model).__name__,
        "device": str(next(forecast_model.parameters()).device),
        "torch_available": True,
        "sklearn_available": True,
    }
    p2_news_path = default_p2_news_path()
    if p2_news_path is None:
        raise RuntimeError("Built desktop is missing the audited P2 news corpus")
    return {
        "status": "passed",
        "window_constructed": True,
        "builtin_skills": builtin_skills,
        "function_count": len(function_names),
        "function_names": function_names,
        "skill_load_errors": list(window.workspace.agent.skill_load_errors),
        "session_store_path": str(window.workspace.store.path.resolve()),
        "research_output_directory": str(window.workspace.research_output_directory.resolve()),
        "forecast_runtime": forecast_runtime,
        "p2_news_path": str(p2_news_path),
    }


def _write_smoke_result(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _run_smoke(app: QApplication, result_path: Path) -> int:
    try:
        window = MainWindow()
        app.processEvents()
        payload = _desktop_smoke_payload(window)
        window.close()
        app.processEvents()
    except Exception as exc:  # noqa: BLE001 - smoke boundary records the real startup failure
        _write_smoke_result(
            result_path,
            {"status": "failed", "error_type": type(exc).__name__, "message": str(exc)},
        )
        return 1
    _write_smoke_result(result_path, payload)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Create and run the desktop UI directly."""

    arguments, smoke_result = _smoke_result_argument(list(argv) if argv is not None else sys.argv)
    app = QApplication.instance() or QApplication(arguments)
    app.setApplicationName("电价与外生变量研究 Agent")
    app.setOrganizationName("Price Research")
    app.setFont(QFont("Microsoft YaHei UI", 10))
    app.setStyleSheet(APP_STYLE)
    if smoke_result is not None:
        return _run_smoke(app, smoke_result)
    window = MainWindow()
    window.show()
    return app.exec()
