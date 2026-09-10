"""The right-hand data panel, its wording, and when an approved plan survives."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from app.desktop.main_window import MainWindow
from app.desktop.message_widgets import DataPlanMessageWidget, summary_fields
from app.desktop.panes import DataPanel
from app.desktop.session import ResearchSession, SessionStore
from app.research.application.coordinator import ResearchCoordinator
from app.research.data.sources import naming
from app.research.data.sources.summary import DataSummary, VariableLabel, build_summary
from tests.desktop.test_desktop import select_desktop_data, wait_until

# Database and programming words the panel must never show. The one screen that
# may show them is the data-details dialog, which takes two clicks to reach.
FORBIDDEN_WORDS = (
    "SQL",
    "WHERE",
    "JOIN",
    "unpivot",
    "指纹",
    "快照",
    "宽表",
    "长表",
    "连接串",
    "隧道",
    "表名",
    "字段名",
    "max_lag",
)
TERMINOLOGY_EXEMPT = {"DataDetailsDialog", "_dataset_details"}


def visible_strings(path: Path) -> list[tuple[int, str]]:
    """Collect every string literal that is not a docstring and not exempt."""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings: set[int] = set()
    skipped: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                docstrings.add(id(body[0].value))
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in TERMINOLOGY_EXEMPT:
            skipped.update(id(child) for child in ast.walk(node))
    return [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
        and id(node) not in skipped
    ]


@pytest.mark.parametrize(
    "module",
    ["panes.py", "message_widgets.py", "workspace.py", "session.py", "input_config.py"],
)
def test_desktop_strings_never_use_database_words(module: str):
    path = Path("app/desktop") / module
    offences = [
        (line, text, word)
        for line, text in visible_strings(path)
        for word in FORBIDDEN_WORDS
        if word in text
    ]
    assert offences == []


def test_empty_panel_offers_nothing_to_click(qt_app: QApplication, tmp_path: Path):
    window = MainWindow(session_store=SessionStore(tmp_path / "sessions.json"))
    try:
        panel = window.workspace.context.data_panel
        assert panel.state == "empty"
        assert panel.empty_label.text() == DataPanel.EMPTY_TEXT
        for view in (panel.exploring_view, panel.ready_view, panel.unavailable_view):
            assert not view.isVisibleTo(panel)
    finally:
        window.close()


def test_unavailable_panel_says_the_same_thing_whatever_broke(qt_app: QApplication, tmp_path: Path):
    window = MainWindow(session_store=SessionStore(tmp_path / "sessions.json"))
    try:
        panel = window.workspace.context.data_panel
        panel.set_state("unavailable", None)

        assert panel.unavailable_view.isVisibleTo(panel)
        assert "取不到" in panel.status_label.text()
        # A first-ever failure has no earlier dataset to fall back on.
        assert not panel.history_container.isVisibleTo(panel)
        assert panel.retry_button.text() == "再试一次"
        assert panel.local_file_button.text() == "用本地文件"

        earlier = build_summary(
            market="山东电网",
            target_name="rt_price",
            exogenous_names=["wind_power"],
            start_time="2026-01-01T00:00:00",
            end_time="2026-09-09T23:45:00",
            frequency="15T",
        )
        panel.set_state("unavailable", earlier)
        assert panel.history_container.isVisibleTo(panel)
        assert "山东电网 实时电价" in panel.history_card.rows["电价"].value_label.text()
    finally:
        window.close()


def test_ready_panel_reads_only_the_prepared_summary(qt_app: QApplication, tmp_path: Path):
    window = MainWindow(session_store=SessionStore(tmp_path / "sessions.json"))
    try:
        panel = window.workspace.context.data_panel
        summary = DataSummary(
            price_label="山东电网 实时电价",
            variables=[VariableLabel("风电出力"), VariableLabel("p001", resolved=False)],
            start_date="2026年1月1日",
            end_date="2026年9月9日",
            granularity_text="每 15 分钟一个点",
            gap_text="缺 47 个点",
            fetched_at_text="今天 15:32",
        )
        panel.set_state("ready", summary)

        rows = panel.ready_card.rows
        assert rows["电价"].value_label.text() == "山东电网 实时电价"
        assert "风电出力、" in rows["影响因素"].value_label.text()
        # An unnamed column keeps its stored name rather than becoming 「变量 1」.
        assert "p001" in rows["影响因素"].value_label.text()
        assert "#B45309" in rows["影响因素"].value_label.text()
        assert rows["影响因素"].value_label.toolTip() == "这项数据还没配中文名"
        assert rows["时间范围"].value_label.text() == "2026年1月1日 — 2026年9月9日"
        assert rows["时间范围"].sub_label.text() == "每 15 分钟一个点，缺 47 个点"
        assert panel.fetched_label.text() == "数据取自今天 15:32"
        assert panel.refetch_button.text() == "取最新的"
        assert panel.reselect_button.text() == "换一批数据"
        assert panel.details_button.text() == "查看取数细节"
    finally:
        window.close()


def test_gapless_data_shows_only_the_sampling_interval(qt_app: QApplication, tmp_path: Path):
    window = MainWindow(session_store=SessionStore(tmp_path / "sessions.json"))
    try:
        panel = window.workspace.context.data_panel
        panel.set_state(
            "ready",
            DataSummary(
                price_label="实时电价",
                start_date="2026年1月1日",
                end_date="2026年9月9日",
                granularity_text="每 15 分钟一个点",
            ),
        )
        assert panel.ready_card.rows["时间范围"].sub_label.text() == "每 15 分钟一个点"
    finally:
        window.close()


def test_missing_word_list_entry_never_blocks_the_research(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(naming, "word_list_path", lambda: None)
    naming.reload_word_list()
    try:
        summary = build_summary(
            market="unspecified",
            target_name="rt_price",
            exogenous_names=["wind_power"],
            start_time="2026-01-01T00:00:00",
            end_time="2026-01-02T00:00:00",
            frequency="15T",
        )
        assert summary.price_label == "rt_price"
        assert summary.variables == [VariableLabel("wind_power", resolved=False)]
    finally:
        naming.reload_word_list()


def test_word_list_matches_on_table_then_column_then_shape():
    naming.reload_word_list()
    assert naming.label_variable("wind_power").display == "风电出力"
    assert naming.label_variable("Wind_Power").display == "风电出力"
    assert naming.label_variable("windpower").display == "风电出力"
    assert naming.label_variable("p001").resolved is False


def test_confirmation_card_and_panel_word_the_dataset_identically(
    qt_app: QApplication,
    desktop_study: Path,
    model_agent: ResearchCoordinator,
    tmp_path: Path,
):
    window = MainWindow(agent=model_agent, session_store=SessionStore(tmp_path / "sessions.json"))
    select_desktop_data(window, desktop_study)
    try:
        workspace = window.workspace
        workspace.submit_question("分析 actual_load 与电价 2 小时的滞后关系")
        wait_until(qt_app, lambda: not workspace.is_busy)

        session = workspace.current_session
        assert session.data_state == "ready"
        assert session.data_summary is not None
        assert session.messages[-1].kind == "data_plan"

        card = workspace.conversation.current_plan_widget
        assert isinstance(card, DataPlanMessageWidget)
        assert card.TITLE == "开始之前，跟你确认一下"
        assert card.run_button.text() == "可以开始"
        assert card.revise_button.text() == "我想改改"
        assert card.reject_button.text() == "先不做"

        panel = workspace.context.data_panel
        for key, value, sub in summary_fields(session.data_summary):
            assert panel.ready_card.rows[key].value_label.text() == value
            assert panel.ready_card.rows[key].sub_label.text() == sub
    finally:
        window.close()


def test_countdown_chip_counts_in_minutes_and_seconds(
    qt_app: QApplication,
    desktop_study: Path,
    model_agent: ResearchCoordinator,
    tmp_path: Path,
):
    window = MainWindow(agent=model_agent, session_store=SessionStore(tmp_path / "sessions.json"))
    select_desktop_data(window, desktop_study)
    try:
        workspace = window.workspace
        workspace.submit_question("分析电价分布")
        wait_until(qt_app, lambda: not workspace.is_busy)
        card = workspace.conversation.current_plan_widget
        assert card is not None

        card.set_feedback_countdown(95)
        assert card.status_label.text() == "1 分 35 秒后自动开始"
    finally:
        window.close()


def test_taking_the_same_data_again_keeps_an_approved_plan(
    qt_app: QApplication,
    desktop_study: Path,
    model_agent: ResearchCoordinator,
    tmp_path: Path,
):
    window = MainWindow(agent=model_agent, session_store=SessionStore(tmp_path / "sessions.json"))
    select_desktop_data(window, desktop_study)
    try:
        workspace = window.workspace
        workspace.submit_question("分析电价分布")
        wait_until(qt_app, lambda: not workspace.is_busy)
        session = workspace.current_session
        assert session.current_plan is not None
        plan_id = session.current_plan["plan_id"]

        workspace.refetch_dataset()

        assert workspace.current_session.current_plan is not None
        assert workspace.current_session.current_plan["plan_id"] == plan_id
        assert not workspace.current_session.plan_stale
        assert workspace.current_session.data_state == "ready"
    finally:
        window.close()


def test_changed_data_voids_the_plan_with_the_agreed_notice(
    qt_app: QApplication,
    desktop_study: Path,
    model_agent: ResearchCoordinator,
    tmp_path: Path,
):
    import pandas as pd

    from app.desktop.workspace import DATA_CHANGED_NOTICE

    window = MainWindow(agent=model_agent, session_store=SessionStore(tmp_path / "sessions.json"))
    select_desktop_data(window, desktop_study)
    try:
        workspace = window.workspace
        workspace.submit_question("分析电价分布")
        wait_until(qt_app, lambda: not workspace.is_busy)
        assert workspace.current_session.current_plan is not None

        target = Path(workspace.current_session.inputs["target"].path)
        frame = pd.read_csv(target)
        frame.iloc[0, -1] = float(frame.iloc[0, -1]) + 11.0
        frame.to_csv(target, index=False)

        workspace.refetch_dataset()

        assert workspace.current_session.current_plan is None
        assert workspace.current_session.plan_stale
        assert workspace.current_session.messages[-1].content == DATA_CHANGED_NOTICE
    finally:
        window.close()


def test_a_session_saved_before_the_panel_still_opens(qt_app: QApplication, tmp_path: Path):
    store = SessionStore(tmp_path / "sessions.json")
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text(
        '[{"schema_version": 11, "session_id": "old", "title": "旧研究", '
        '"inputs": {"target": {"role": "target", "label": "目标电价", "path": ""}}}]',
        encoding="utf-8",
    )

    window = MainWindow(session_store=store)
    try:
        restored = next(item for item in window.workspace.sessions if item.session_id == "old")
        assert isinstance(restored, ResearchSession)
        assert restored.schema_version == 12
        assert restored.data_state == "empty"
        assert restored.data_summary is None
        assert restored.source_kind == "database"
    finally:
        window.close()
