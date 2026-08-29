"""Production-boundary checks used by the freshly built Windows executable."""

from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from app.desktop.main import _desktop_smoke_payload, _smoke_result_argument
from app.desktop.main_window import MainWindow
from app.desktop.session import SessionStore
from scripts.build_desktop_exe import sanitized_build_path
from scripts.validate_desktop_phase1 import _shutdown_window


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_smoke_argument_is_removed_before_qt_parses_it(tmp_path: Path) -> None:
    result = tmp_path / "result.json"

    cleaned, parsed = _smoke_result_argument(["PriceResearchAgent.exe", f"--smoke-test={result}"])

    assert cleaned == ["PriceResearchAgent.exe"]
    assert parsed == result.resolve()


def test_desktop_smoke_payload_loads_both_skills_and_all_functions(
    qt_app: QApplication,
    tmp_path: Path,
) -> None:
    window = MainWindow(session_store=SessionStore(tmp_path / "sessions.json"))
    try:
        payload = _desktop_smoke_payload(window)
    finally:
        window.close()

    assert payload["status"] == "passed"
    assert {item["name"] for item in payload["builtin_skills"]} == {
        "price-exogenous-eda",
        "price-forecastability-audit",
    }
    assert payload["function_count"] == 30
    assert Path(payload["session_store_path"]) == (tmp_path / "sessions.json").resolve()
    assert Path(payload["research_output_directory"]).is_absolute()


def test_validation_shutdown_cancels_busy_worker_before_closing() -> None:
    events: list[str] = []

    class FakeWorkspace:
        is_busy = True

        def cancel_current_task(self) -> None:
            events.append("cancel")
            self.is_busy = False

    class FakeWindow:
        workspace = FakeWorkspace()

        def isVisible(self) -> bool:
            return True

        def close(self) -> None:
            assert not self.workspace.is_busy
            events.append("close")

    class FakeApp:
        def processEvents(self) -> None:
            events.append("events")

    _shutdown_window(FakeApp(), FakeWindow(), timeout=0.1)  # type: ignore[arg-type]

    assert events[0:2] == ["cancel", "events"]
    assert "close" in events


def test_build_path_drops_foreign_icu_but_keeps_windows_and_unrelated_tools(tmp_path: Path) -> None:
    windows_root = tmp_path / "Windows"
    system32 = windows_root / "System32"
    foreign = tmp_path / "foreign-runtime"
    unrelated = tmp_path / "ordinary-tool"
    for directory in (system32, foreign, unrelated):
        directory.mkdir(parents=True)
    (system32 / "icuuc.dll").write_bytes(b"windows shim")
    (foreign / "icuuc.dll").write_bytes(b"incompatible private ICU")

    sanitized = sanitized_build_path(
        ";".join((str(foreign), str(system32), str(unrelated))),
        windows_root=windows_root,
    ).split(";")

    assert str(foreign) not in sanitized
    assert str(system32) in sanitized
    assert str(unrelated) in sanitized


def test_default_session_store_uses_the_shared_runtime_app_data_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app_data = tmp_path / "configured-app-data"
    monkeypatch.setenv("PRICE_RESEARCH_APP_DATA_DIRECTORY", str(app_data))

    assert SessionStore().path == (app_data / "research_sessions.json").resolve()
