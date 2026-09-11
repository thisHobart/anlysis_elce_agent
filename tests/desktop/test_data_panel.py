"""The right-hand data panel, its wording, and when an approved plan survives."""

from __future__ import annotations

import ast
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication, QLabel

from app.desktop.input_config import build_runtime_study
from app.desktop.main_window import MainWindow
from app.desktop.message_widgets import DataPlanMessageWidget, summary_fields
from app.desktop.panes import DataPanel, FactorListDialog
from app.desktop.session import SESSION_SCHEMA_VERSION, ResearchSession, SessionStore
from app.research.application.coordinator import ResearchCoordinator
from app.research.data.sources import naming
from app.research.data.sources.materialize import snapshot_path
from app.research.data.sources.regions import RegionPriceFetch, RegionProfile, RegionSourceError
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
    "端口",
    "取数契约",
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


def test_ready_panel_compacts_and_exposes_all_factors(qt_app: QApplication):
    panel = DataPanel()
    summary = DataSummary(
        price_label="山东电网 实时电价",
        variables=[
            VariableLabel("统调负荷"),
            VariableLabel("发电总出力"),
            VariableLabel("风电出力"),
            VariableLabel("水电出力"),
            VariableLabel("光伏出力"),
            VariableLabel("负荷预测"),
        ],
        start_date="2025年11月1日",
        end_date="2026年9月11日",
        granularity_text="每 15 分钟一个点",
    )
    try:
        panel.show()
        panel.set_state("ready", summary)
        qt_app.processEvents()

        compact = panel.ready_card.rows["影响因素"].value_label.text()
        assert "统调负荷、发电总出力、风电出力、水电出力" in compact
        assert "共 6 项" in compact
        assert "光伏出力" not in compact
        assert panel.variables_button.isVisibleTo(panel)
        assert panel.variables_button.text() == "查看全部 6 项影响因素"

        dialog = FactorListDialog(summary.variables, panel)
        assert [
            dialog.tree.topLevelItem(index).text(0)
            for index in range(dialog.tree.topLevelItemCount())
        ] == ["实际值（5）", "预测值（1）"]
        dialog.search.setText("负荷")
        actual_group = dialog.tree.topLevelItem(0)
        assert [
            actual_group.child(index).text(0)
            for index in range(actual_group.childCount())
            if not actual_group.child(index).isHidden()
        ] == ["统调负荷"]
        assert not dialog.tree.topLevelItem(1).isHidden()
        dialog.close()
    finally:
        panel.close()


def test_region_button_fetches_shandong_for_only_the_current_session(
    qt_app: QApplication,
    tmp_path: Path,
):
    profile = RegionProfile(
        region_id="shandong",
        label="山东",
        market="山东电网",
        province_value="山东省",
        timezone="Asia/Shanghai",
        frequency="15min",
        credential_prefix="VPP_SHANDONG_DB_FEATURE",
        price_table="t_data_province_real_time_cleared_price",
        price_label="实时省级出清电价",
    )

    def fetcher(selected, *, output_directory, progress):
        assert selected is profile
        progress(20, "正在取数")
        path = Path(output_directory) / "shandong" / "actual-realtime-price.parquet"
        path.parent.mkdir(parents=True)
        pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-09-01", periods=8, freq="15min"),
                "rt_price": range(8),
                "available_at": pd.date_range("2026-09-01", periods=8, freq="15min"),
            }
        ).to_parquet(path, index=False)
        actuals_path = path.with_name("actual-weather.parquet")
        forecasts_path = path.with_name("forecast-weather.parquet")
        pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-09-01", periods=8, freq="15min"),
                "temperature": range(8),
                "available_at": pd.date_range("2026-09-02", periods=8, freq="15min"),
            }
        ).to_parquet(actuals_path, index=False)
        pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-09-01", periods=8, freq="15min"),
                "temperature": range(8),
                "available_at": pd.date_range("2026-08-31", periods=8, freq="15min"),
            }
        ).to_parquet(forecasts_path, index=False)
        return RegionPriceFetch(
            profile=profile,
            path=path,
            fetched_at=datetime.fromisoformat("2026-09-01T02:00:00+08:00"),
            row_count=8,
            start_at=datetime(2026, 9, 1, 0, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
            end_at=datetime(2026, 9, 1, 1, 45, tzinfo=ZoneInfo("Asia/Shanghai")),
            duplicate_timestamp_rows=0,
            invalid_rows=0,
            database_name="prices",
            actuals_path=actuals_path,
            forecasts_path=forecasts_path,
            actual_variable_names=("temperature",),
            forecast_variable_names=("forecast_temperature",),
        )

    window = MainWindow(
        session_store=SessionStore(tmp_path / "sessions.json"),
        region_profiles={"shandong": profile},
        region_fetcher=fetcher,
    )
    try:
        workspace = window.workspace
        original_session_id = workspace.current_session.session_id
        panel = workspace.context.data_panel
        assert panel.region_button.text() == "地区：山东 ▾"
        panel.region_menu.actions()[0].trigger()
        wait_until(qt_app, lambda: not workspace.is_busy)

        session = workspace.current_session
        assert session.session_id == original_session_id
        assert session.region_id == "shandong"
        assert session.source_kind == "database"
        assert session.data_state == "ready"
        assert session.can_analyze
        assert session.data_summary is not None
        assert session.data_summary.price_label == "山东电网 实时电价"
        assert [item.display for item in session.data_summary.variables] == ["气温", "气温预测"]
        assert session.database_fetch_details["rows"] == 8
        assert panel.ready_card.rows["电价"].value_label.text() == "山东电网 实时电价"
        assert panel.ready_card.rows["影响因素"].value_label.text() == "气温、气温预测"
        config = build_runtime_study(session)
        assert [item.name for item in config.exogenous] == ["temperature", "forecast_temperature"]
        assert all(item.available_at_column == "available_at" for item in config.exogenous)
    finally:
        window.close()


def test_region_fetch_failure_only_marks_data_unavailable(qt_app: QApplication, tmp_path: Path):
    profile = RegionProfile(
        region_id="shandong",
        label="山东",
        market="山东电网",
        province_value="山东省",
        timezone="Asia/Shanghai",
        frequency="15min",
        credential_prefix="VPP_SHANDONG_DB_FEATURE",
        price_table="t_data_province_real_time_cleared_price",
        price_label="实时省级出清电价",
    )

    def unavailable(*_args, **_kwargs):
        raise RegionSourceError("测试连接不可用")

    window = MainWindow(
        session_store=SessionStore(tmp_path / "sessions.json"),
        region_profiles={"shandong": profile},
        region_fetcher=unavailable,
    )
    try:
        panel = window.workspace.context.data_panel
        panel.region_menu.actions()[0].trigger()
        wait_until(qt_app, lambda: not window.workspace.is_busy)

        session = window.workspace.current_session
        assert session.data_state == "unavailable"
        assert session.status == "idle"
        assert not session.can_analyze
        assert panel.unavailable_view.isVisibleTo(panel)
        assert session.messages[-1].kind == "error"
        assert "测试连接不可用" in session.messages[-1].content
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


def test_exploring_panel_lights_up_what_is_settled_and_says_nothing_is_taken_yet(
    qt_app: QApplication,
    tmp_path: Path,
):
    window = MainWindow(session_store=SessionStore(tmp_path / "sessions.json"))
    try:
        panel = window.workspace.context.data_panel
        panel.set_state(
            "exploring",
            DataSummary(
                price_label="山东电网 实时电价",
                variables=[VariableLabel("风电出力"), VariableLabel("光伏出力")],
            ),
        )

        assert panel.exploring_view.isVisibleTo(panel)
        assert "正在找数据" in panel.status_label.text()
        rows = panel.exploring_card.rows
        assert rows["电价"].mark.text() == "✓"
        assert rows["影响因素"].value_label.text() == "正在核对：风电出力、光伏出力"
        # Nothing is claimed about a field the exploration has not reached.
        assert rows["时间范围"].value_label.text() == "待定"
        assert rows["时间范围"].mark.text() == "○"
        labels = [item.text() for item in panel.exploring_view.findChildren(QLabel)]
        assert DataPanel.EXPLORING_TEXT in labels
        assert DataPanel.EXPLORING_NOTE in labels
    finally:
        window.close()


def test_asking_a_question_lights_up_what_is_already_known(
    qt_app: QApplication,
    desktop_study: Path,
    model_agent: ResearchCoordinator,
    tmp_path: Path,
):
    """§9.2: the card fills in as the exploration learns, instead of staying blank."""

    window = MainWindow(agent=model_agent, session_store=SessionStore(tmp_path / "sessions.json"))
    select_desktop_data(window, desktop_study)
    try:
        workspace = window.workspace
        panel = workspace.context.data_panel

        # The work runs on a worker thread, so reading the panel straight after the
        # call catches it mid-exploration, before any answer comes back.
        workspace.submit_question("分析电价分布")
        exploring_state = panel.state
        exploring = {
            key: (row.value_label.text(), row.mark.text(), row.sub_label.text())
            for key, row in panel.exploring_card.rows.items()
        }
        wait_until(qt_app, lambda: not workspace.is_busy)

        assert exploring_state == "exploring"
        # The price is named the same way before and after the data is read.
        assert exploring["电价"] == (panel.ready_card.rows["电价"].value_label.text(), "✓", "")
        assert exploring["电价"][0] == "实时电价"
        assert exploring["影响因素"][0].startswith("正在核对：")
        assert "统调负荷" in exploring["影响因素"][0]
        assert exploring["时间范围"][:2] == ("待定", "○")
        assert exploring["时间范围"][2] == "每小时一个点"

        assert workspace.current_session.data_state == "ready"
        assert panel.ready_card.rows["时间范围"].value_label.text() != "待定"
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


def test_chat_selected_time_range_reaches_the_plan_and_data_panel(
    qt_app: QApplication,
    desktop_study: Path,
    model_agent: ResearchCoordinator,
    tmp_path: Path,
):
    window = MainWindow(agent=model_agent, session_store=SessionStore(tmp_path / "sessions.json"))
    select_desktop_data(window, desktop_study)
    try:
        workspace = window.workspace
        workspace.submit_question("仅分析 2026-01-02 到 2026-01-05 的电价与负荷关系")
        wait_until(qt_app, lambda: not workspace.is_busy)

        session = workspace.current_session
        assert session.analysis_start_time == "2026-01-02T00:00:00"
        assert session.analysis_end_time == "2026-01-05T23:59:59.999999"
        assert session.data_profile["start_time"].startswith("2026-01-02T00:00:00")
        assert session.data_profile["end_time"].startswith("2026-01-05T23:00:00")
        panel_text = workspace.context.data_panel.ready_card.rows["时间范围"].value_label.text()
        assert panel_text == "2026年1月2日 — 2026年1月5日"
    finally:
        window.close()


def test_confirmation_card_names_actions_rather_than_methods(
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

        card = workspace.conversation.current_plan_widget
        assert isinstance(card, DataPlanMessageWidget)
        assert card.DATA_SECTION == "要用的数据"
        assert card.ACTION_SECTION == "要做的事"

        plan = card.plan
        rendered = [card.step_checks[step.step_id].text() for step in plan.enabled_steps]
        assert len(rendered) > 1
        assert rendered == [plan.step_text(step) for step in plan.enabled_steps]
        # A step reads as the thing it does, and never as the method behind it.
        assert all(text != step.title for text, step in zip(rendered, plan.enabled_steps))
        assert all(not text[:1].isdigit() for text in rendered)
        assert "先看这批数据完不完整、时间点对不对得上" in rendered
    finally:
        window.close()


def test_a_column_without_a_chinese_name_warns_instead_of_stopping(
    qt_app: QApplication,
    desktop_study: Path,
    model_agent: ResearchCoordinator,
    tmp_path: Path,
):
    window = MainWindow(agent=model_agent, session_store=SessionStore(tmp_path / "sessions.json"))
    select_desktop_data(window, desktop_study)
    try:
        workspace = window.workspace
        workspace.submit_question("分析 forecast_generation 与电价的关系")
        wait_until(qt_app, lambda: not workspace.is_busy)

        session = workspace.current_session
        summary = session.data_summary
        assert summary is not None
        unnamed = [item.display for item in summary.variables if not item.resolved]
        assert "forecast_generation" in unnamed
        # The research keeps going; the missing name is an audit note, not a stop.
        assert session.data_state == "ready"
        assert session.current_plan is not None
        warnings = [
            event
            for event in session.trace
            if event.status == "warning" and "没有中文名" in event.name
        ]
        assert warnings
        assert warnings[-1].category == "input"
        assert warnings[-1].name == f"有 {len(unnamed)} 项数据没有中文名，先按原名显示"
        # The stored name travels with the warning so the analyst can ask about it.
        assert "forecast_generation" in warnings[-1].summary
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


def test_approving_fixes_the_data_so_a_later_change_cannot_move_the_result(
    qt_app: QApplication,
    desktop_study: Path,
    model_agent: ResearchCoordinator,
    tmp_path: Path,
):
    """§9.4: what was approved is what runs, whatever happens to the source afterwards."""

    import pandas as pd

    window = MainWindow(agent=model_agent, session_store=SessionStore(tmp_path / "sessions.json"))
    select_desktop_data(window, desktop_study)
    try:
        workspace = window.workspace
        workspace.submit_question("分析电价分布")
        wait_until(qt_app, lambda: not workspace.is_busy)
        session = workspace.current_session
        assert session.data_summary is not None
        fetched_at = session.data_summary.fetched_at_text
        approved = workspace.conversation.current_plan_widget.approved_plan()
        # Proposing wrote the data down; the run below reads that copy, not the file.
        assert snapshot_path(session.dataset_fingerprint).is_dir()

        target = Path(session.inputs["target"].path)
        frame = pd.read_csv(target)
        frame.iloc[0, -1] = float(frame.iloc[0, -1]) + 400.0
        frame.to_csv(target, index=False)

        workspace.run_plan(approved)
        wait_until(qt_app, lambda: not workspace.is_busy)

        assert session.status == "completed"
        assert session.report_path and Path(session.report_path).is_file()
        assert not session.plan_stale
        # The panel keeps naming the moment the data was read, not the moment the
        # file changed underneath it.
        assert session.data_summary.fetched_at_text == fetched_at
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
        assert session.data_summary is not None
        plan_id = session.current_plan["plan_id"]
        fetched_at = session.data_summary.fetched_at_text

        workspace.refetch_dataset()

        assert workspace.current_session.current_plan is not None
        assert workspace.current_session.current_plan["plan_id"] == plan_id
        assert not workspace.current_session.plan_stale
        assert workspace.current_session.data_state == "ready"
        # Identical data is still data from when it was first read.
        assert workspace.current_session.data_summary.fetched_at_text == fetched_at
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


def test_a_plan_card_saved_before_the_merge_renders_in_the_new_card(
    qt_app: QApplication,
    desktop_study: Path,
    model_agent: ResearchCoordinator,
    tmp_path: Path,
):
    """The old approval card is gone, so history renders in the one card that remains."""

    from tests.desktop.test_desktop import _fresh_plan, _session_with_plan

    stored = _session_with_plan(_fresh_plan(desktop_study, model_agent))
    store = SessionStore(tmp_path / "sessions.json")
    store.save([stored])

    window = MainWindow(agent=model_agent, session_store=store)
    try:
        # Opening the app starts a new conversation, so reach for the saved one.
        window.workspace.select_session(stored.session_id)
        card = window.workspace.conversation.current_plan_widget
        assert isinstance(card, DataPlanMessageWidget)
        # That conversation stored no dataset description, and the card says so
        # rather than inventing one from today's data.
        labels = [item.text() for item in card.findChildren(QLabel)]
        assert card.TITLE in labels
        assert "电价　待定" in labels
        assert "影响因素　待定" in labels
        assert "时间范围　待定" in labels
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
        assert restored.schema_version == SESSION_SCHEMA_VERSION
        assert restored.data_state == "empty"
        assert restored.data_summary is None
        assert restored.source_kind == "database"
    finally:
        window.close()
