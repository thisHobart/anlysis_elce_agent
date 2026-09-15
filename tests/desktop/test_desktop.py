"""Headless functional checks for the three-pane PySide6 research workspace."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest
import yaml
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QTextDocument
from PySide6.QtWidgets import QApplication, QFileDialog, QLabel

from app.config import Settings
from app.desktop.input_config import build_runtime_study, parse_chat_time_range
from app.desktop.main_window import MainWindow
from app.desktop.message_widgets import DataPlanMessageWidget, ResultMessageWidget, ThinkingMessageWidget
from app.desktop.panes import ConversationPane, TracePanel
from app.desktop.report_view import ReportBrowser, ReportWindow
from app.desktop.session import (
    SESSION_SCHEMA_VERSION,
    STALE_PLAN_NOTICE,
    ResearchSession,
    SessionMessage,
    SessionRunRecord,
    SessionStore,
    TraceEvent,
)
from app.research.agent.orchestrator import DialogueDecision, MainResearchAgent
from app.research.agent.schemas import EDAResearchScope
from app.research.agent.subagents.eda import EDASubagent
from app.research.application.coordinator import ResearchCoordinator
from app.research.data.inference import infer_study_context
from app.research.graph.narration import STAGE_LABELS
from app.research.graph.process_events import ProcessEvent
from app.research.skills.registry import SkillRegistry
from app.research.tools.catalog import FUNCTION_CATALOG


class DesktopModelPlanner:
    enabled = True
    model_name = "desktop-test-model"

    def propose(self, question, config, quality, history=None, skill=None, feedback=None):
        del quality, history, feedback
        assert skill is not None
        names = [spec.name for spec in config.exogenous]
        relationship_request = any(word in question for word in ("关系", "相关", "滞后", "影响"))
        if not relationship_request:
            functions = [
                function_name
                for word, function_name in (
                    ("季节", "price_calendar_group_profile"),
                    ("尖峰", "price_tukey_outer_fence"),
                )
                if word in question
            ]
            return {
                "objective": "分析用户关注的电价结构",
                "hypotheses": [],
                "selected_variables": [],
                "steps": [
                    {
                        "function": function_name,
                        "enabled": True,
                        "rationale": "问题聚焦电价自身。",
                        "parameters": {},
                    }
                    for function_name in (functions or ["price_descriptive_distribution"])
                ],
                "assumptions": [],
            }
        selected = [name for name in names if name.casefold() in question.casefold()]
        if not selected and names:
            selected = [names[0]]
        lag = 2 if "2 小时" in question or "2小时" in question else 24
        return {
            "objective": "分析所选变量与电价的关系",
            "hypotheses": [],
            "selected_variables": selected,
            "steps": [
                {
                    "function": "price_calendar_group_profile",
                    "enabled": True,
                    "rationale": "检查电价周期结构。",
                    "parameters": {},
                },
                {
                    "function": "price_lag_autocorrelation",
                    "enabled": True,
                    "rationale": "检查电价滞后结构。",
                    "parameters": {"max_lag": lag},
                },
                {
                    "function": "exogenous_descriptive_distribution",
                    "enabled": True,
                    "rationale": "检查变量画像。",
                    "parameters": {"variables": selected},
                },
                {
                    "function": "exogenous_iqr_outliers",
                    "enabled": True,
                    "rationale": "检查变量异常值。",
                    "parameters": {"variables": selected},
                },
                {
                    "function": "relationship_scipy_pearson_pairwise",
                    "enabled": True,
                    "rationale": "检查同期 Pearson 关系。",
                    "parameters": {"variables": selected},
                },
                {
                    "function": "relationship_scipy_spearman_pairwise",
                    "enabled": True,
                    "rationale": "检查同期 Spearman 关系。",
                    "parameters": {"variables": selected},
                },
                {
                    "function": "relationship_pearson_positive_lead_scan",
                    "enabled": True,
                    "rationale": "检查领先滞后关系。",
                    "parameters": {"variables": selected, "max_lag": lag},
                },
            ],
            "assumptions": [],
        }


class DesktopModelDialogue:
    enabled = True
    model_name = "desktop-test-model"

    def decide(self, **kwargs):
        question = kwargs["question"]
        if "按这个执行" in question:
            return DialogueDecision(intent="execute_plan", response="已锁定当前模型方案。")
        if "只保留关系分析" in question:
            return DialogueDecision(
                intent="revise_plan",
                response="已根据用户反馈修订方案。",
                enabled_functions=[
                    "relationship_scipy_pearson_pairwise",
                    "relationship_scipy_spearman_pairwise",
                    "relationship_pearson_positive_lead_scan",
                ],
                selected_variables=["actual_wind"],
                max_lag=4,
            )
        if "结果" in question or "说明什么" in question:
            return DialogueDecision(
                intent="explain_result",
                response="已有确定性证据只支持描述性关系，不构成因果结论。",
            )
        if not kwargs.get("has_executable_data"):
            return DialogueDecision(intent="discussion", response="先检查时间轴和季节性，再选择具体 EDA 方法。")
        return DialogueDecision(intent="new_plan", skill_name="price-exogenous-eda")


SAMPLE_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 300" width="760" height="300">'
    '<rect width="760" height="300" fill="#EEF0FB"/></svg>'
)


def test_dynamic_scope_card_shows_tools_variables_and_budget(qt_app: QApplication):
    scope = EDAResearchScope(
        question="分析电价与负荷",
        objective="识别电价结构及负荷关系",
        study_name="scope-ui-test",
        data_fingerprint="a" * 12,
        skill_name="price-exogenous-eda",
        skill_version="1.0.0",
        authorized_functions=[
            "price_descriptive_distribution",
            "relationship_scipy_pearson_pairwise",
        ],
        authorized_variables=["load", "wind"],
        initial_strategy=["先建立电价画像，再根据证据判断是否分析负荷关系。"],
    )

    widget = DataPlanMessageWidget(scope)
    texts = [label.text() for label in widget.findChildren(QLabel)]

    assert widget.approved_plan() == scope
    assert any("研究目标" in text and scope.objective in text for text in texts)
    assert any("最多 8 轮决策、16 次基础调用" in text for text in texts)
    assert any("每次最多尝试 2 次" in text for text in texts)
    assert any("变量范围" in text and "load" in text and "wind" in text for text in texts)
    widget.deleteLater()
    qt_app.processEvents()


def test_report_reader_renders_markdown_figures_inside_the_app(qt_app: QApplication, tmp_path: Path):
    """Figures rendering badly in Markdown is the reason the report used to be HTML."""

    package = tmp_path / "run-1"
    (package / "figures").mkdir(parents=True)
    (package / "figures" / "price_timeseries.svg").write_text(SAMPLE_SVG, encoding="utf-8")
    report = package / "report.md"
    report.write_text(
        "# 报告标题\n\n## 电价自身规律\n\n![电价时间序列](figures/price_timeseries.svg)\n",
        encoding="utf-8",
    )

    window = ReportWindow(report)
    window.resize(1000, 780)
    window.show()
    qt_app.processEvents()

    text = window.browser.toPlainText()
    assert "报告标题" in text and "电价自身规律" in text

    image = window.browser.loadResource(
        QTextDocument.ResourceType.ImageResource.value, QUrl("figures/price_timeseries.svg")
    )
    assert image is not None and not image.isNull()
    assert image.width() > 0

    # The report must fit its window; a document wider than the viewport scrolls sideways.
    assert window.browser.document().size().width() <= window.browser.viewport().width() + 1
    window.close()


def test_report_reader_renders_svg_in_logical_pixels_on_high_dpi(qt_app: QApplication, tmp_path: Path):
    """The device ratio must not scale and crop SVG content a second time."""

    class HighDpiReportBrowser(ReportBrowser):
        def devicePixelRatioF(self) -> float:
            return 1.5

    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 50">'
        '<rect width="100" height="50" fill="#FFFFFF"/>'
        '<rect x="90" width="10" height="50" fill="#FF0000"/></svg>'
    )
    path = tmp_path / "right-edge.svg"
    path.write_text(svg, encoding="utf-8")
    browser = HighDpiReportBrowser(tmp_path)

    image = browser._render_svg(path)

    assert image is not None
    assert image.devicePixelRatio() == pytest.approx(1.5)
    assert image.pixelColor(image.width() - 2, image.height() // 2).red() > 240
    assert image.pixelColor(image.width() - 2, image.height() // 2).green() < 20


def test_ranked_bar_labels_never_land_on_the_category_name():
    """A long negative bar used to print its value on top of the variable name."""

    import re

    from app.research.reporting.svg_charts import LABEL_CHARACTER_WIDTH, ranked_bar_chart

    rows = [
        ("sys_reserve_type_3", 0.7416),
        ("sys_reserve_type_2", -0.7297),
        ("fcst_new_energy_type_13", -0.6536),
        ("act_new_energy_type_12", -0.5689),
        ("actual_load", 0.4696),
        ("fcst_new_energy_type_1", -0.258),
    ]
    markup = ranked_bar_chart(rows, title="电价—因素同期相关排序", subtitle="同期 Pearson 相关系数")

    # Two variables whose names share a prefix must stay distinguishable.
    names = re.findall(r'font-size="11" fill="#1F2430">([^<]+)</text>', markup)
    assert len(names) == len(set(names)) == len(rows)

    names_end_at = 156.0
    pattern = r'<text x="([\d.]+)" y="[\d.]+" text-anchor="(\w+)" font-size="10.5" fill="(#\w+)">([^<]+)</text>'
    values = [
        (float(x), anchor, fill, label)
        for x, anchor, fill, label in re.findall(pattern, markup)
        if re.fullmatch(r"-?[\d.,]+", label)
    ]
    assert len(values) == len(rows)
    for x, anchor, fill, label in values:
        left_edge = x - len(label) * LABEL_CHARACTER_WIDTH if anchor == "end" else x
        assert left_edge > names_end_at, (label, left_edge)
        # A label pushed inside the bar has to be readable against the fill.
        assert fill in {"#667085", "#FFFFFF"}


def test_report_svg_prefers_a_font_with_chinese_glyphs():
    from app.research.reporting.svg_charts import stem_chart

    markup = stem_chart([1], [0.5], title="电价自相关")

    assert "font-family:'Microsoft YaHei'" in markup
    assert "font-family:'Microsoft YaHei'," not in markup


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture
def model_agent() -> ResearchCoordinator:
    return ResearchCoordinator(
        eda_subagent=EDASubagent(model_planner=DesktopModelPlanner()),
        main_agent=MainResearchAgent(model_dialogue=DesktopModelDialogue()),
    )


def test_result_limitations_composer_offers_natural_language_continue_or_end(
    qt_app: QApplication,
):
    pane = ConversationPane()
    ended: list[bool] = []
    pane.end_research_requested.connect(lambda: ended.append(True))

    pane.set_interaction_context("result_limitations")

    assert "输入下一步研究要求" in pane.input.placeholderText()
    assert "按推荐变量继续" in pane.input.placeholderText()
    assert not pane.end_research_button.isHidden()
    assert pane.end_research_button.text() == "结束本轮研究"

    pane.end_research_button.click()
    qt_app.processEvents()
    assert ended == [True]

    pane.set_interaction_context(None)
    assert pane.end_research_button.isHidden()
    assert "说说你想研究什么" in pane.input.placeholderText()
    pane.close()


def test_end_research_button_sends_typed_stop_and_preserves_result_context(
    qt_app: QApplication,
    model_agent: ResearchCoordinator,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    window = MainWindow(agent=model_agent, session_store=SessionStore(tmp_path / "sessions.json"))
    calls: list[dict] = []
    try:
        workspace = window.workspace
        monkeypatch.setattr(model_agent, "has_thread", lambda _session_id: True)
        monkeypatch.setattr(workspace, "_resume_graph", lambda **kwargs: calls.append(kwargs))
        workspace._active_interrupt_kind = "result_limitations"
        workspace.conversation.set_interaction_context("result_limitations")

        workspace.conversation.end_research_button.click()
        qt_app.processEvents()

        assert calls == [{"action": "stop", "task_kind": "dialogue"}]
        assert workspace.current_session.messages[-1].content == "结束本轮研究"
        assert workspace.current_session.messages[-1].role == "user"
        assert workspace.conversation.end_research_button.isHidden()
        assert workspace.current_session.trace[-1].summary == "保留当前结果，不再执行后续深入分析"
    finally:
        window.close()


@pytest.fixture
def desktop_study(tmp_path: Path) -> Path:
    index = pd.date_range("2026-01-01", periods=24 * 7, freq="1h")
    position = np.arange(len(index))
    load = 100 + np.cos(position / 5) * 10
    price = 240 + pd.Series(load).shift(2).bfill().to_numpy() * 0.8 + np.sin(position / 4) * 20
    pd.DataFrame({"datetime": index, "rt_price": price}).to_csv(tmp_path / "market_prices.csv", index=False)
    pd.DataFrame(
        {
            "datetime": index,
            "actual_load": load,
            "actual_wind": 30 + np.sin(position / 7) * 5,
        }
    ).to_csv(tmp_path / "measurements.csv", index=False)
    pd.DataFrame(
        {
            "datetime": index,
            "forecast_generation": 120 + np.sin(position / 6) * 12,
        }
    ).to_csv(tmp_path / "predictions.csv", index=False)
    config = {
        "study": {
            "name": "desktop_test",
            "market": "test",
            "timezone": "Asia/Shanghai",
            "frequency": "1h",
        },
        "target": {
            "name": "rt_price",
            "path": "market_prices.csv",
            "timestamp_column": "datetime",
            "value_column": "rt_price",
        },
        "exogenous": [
            {
                "name": "actual_load",
                "path": "measurements.csv",
                "timestamp_column": "datetime",
                "value_column": "actual_load",
                "availability_type": "observed_only",
            },
            {
                "name": "actual_wind",
                "path": "measurements.csv",
                "timestamp_column": "datetime",
                "value_column": "actual_wind",
                "availability_type": "observed_only",
            },
            {
                "name": "forecast_generation",
                "path": "predictions.csv",
                "timestamp_column": "datetime",
                "value_column": "forecast_generation",
                "availability_type": "forecast",
            },
        ],
        "analysis": {
            "max_lag": 24,
            "min_relationship_observations": 12,
            "output_directory": "artifacts",
        },
    }
    path = tmp_path / "study_config.yml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def wait_until(qt_app: QApplication, predicate, timeout_seconds: float = 12.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        qt_app.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("Qt operation did not finish before timeout")


def choose_files(window: MainWindow, *paths: str) -> None:
    """Answer one file dialog per role, the way a person works the fallback.

    An empty answer is a cancelled dialog, which leaves that role as it was.
    """

    answers = iter(paths)
    with patch.object(QFileDialog, "getOpenFileName", lambda *args, **kwargs: (next(answers, ""), "")):
        window.workspace.choose_local_files()


def select_desktop_data(window: MainWindow, desktop_study: Path) -> None:
    """Populate the three user-facing data roles without a configuration file."""

    choose_files(
        window,
        *(
            str(desktop_study.parent / name)
            for name in ("market_prices.csv", "measurements.csv", "predictions.csv")
        ),
    )


def test_selecting_local_files_does_not_parse_or_fingerprint_them(
    qt_app: QApplication,
    desktop_study: Path,
    model_agent: ResearchCoordinator,
    tmp_path: Path,
):
    window = MainWindow(agent=model_agent, session_store=SessionStore(tmp_path / "sessions.json"))
    try:
        with patch.object(
            window.workspace,
            "_current_dataset_fingerprint",
            side_effect=AssertionError("file contents must not be read while selecting inputs"),
        ):
            select_desktop_data(window, desktop_study)
        qt_app.processEvents()

        assert window.workspace.current_session.data_state == "selected"
        assert window.workspace.current_session.data_summary is None
        assert all(
            item.status == "selected"
            for item in window.workspace.current_session.inputs.values()
        )
    finally:
        window.close()


def test_trace_keeps_every_loaded_data_file_visible(
    qt_app: QApplication,
    desktop_study: Path,
    model_agent: ResearchCoordinator,
    tmp_path: Path,
):
    window = MainWindow(agent=model_agent, session_store=SessionStore(tmp_path / "sessions.json"))
    try:
        select_desktop_data(window, desktop_study)
        qt_app.processEvents()

        tree = window.workspace.context.trace.tree
        loaded = [
            tree.topLevelItem(index).text(2)
            for index in range(tree.topLevelItemCount())
            if tree.topLevelItem(index).text(2).startswith("改用本地文件")
        ]

        # One choice by the analyst, one row, and every file it took is named in it.
        assert loaded == ["改用本地文件 — market_prices.csv、measurements.csv、predictions.csv"]
    finally:
        window.close()


def test_trace_distinguishes_consecutive_question_processing_stages(qt_app: QApplication):
    panel = TracePanel()
    try:
        panel.set_events(
            [
                TraceEvent(category="session", name="新建研究会话", status="completed"),
                TraceEvent(category="user", name="提交研究问题：问题一", status="completed"),
                TraceEvent(category="agent", name="接收研究问题：问题一", status="completed"),
                TraceEvent(category="agent", name="主 Agent 路由", status="completed"),
                TraceEvent(category="plan", name="生成候选方案：plan-1 · 分析目标", status="completed"),
                TraceEvent(category="plan", name="方案校验通过", status="completed"),
                TraceEvent(category="agent", name="回答用户：回复内容", status="completed"),
            ]
        )

        stages = [
            panel.tree.topLevelItem(index).text(1)
            for index in range(panel.tree.topLevelItemCount())
        ]

        assert stages == [
            "准备",
            "提交问题",
            "解析问题",
            "识别意图",
            "生成方案",
            "生成方案",
            "生成结论",
        ]
    finally:
        panel.close()


def test_conversation_follows_the_bottom_while_a_sent_question_progresses(
    qt_app: QApplication,
    model_agent: ResearchCoordinator,
    tmp_path: Path,
):
    window = MainWindow(agent=model_agent, session_store=SessionStore(tmp_path / "sessions.json"))
    try:
        window.resize(1100, 620)
        window.show()
        workspace = window.workspace
        for index in range(60):
            workspace._append_message(
                SessionMessage(role="assistant", kind="text", content=f"历史消息 {index}\n用于形成可滚动的长对话内容。")
            )
        workspace._start_thinking("read", "解析研究问题", "识别本轮处理方式")

        scrollbar = workspace.conversation.scroll.verticalScrollBar()
        wait_until(qt_app, lambda: scrollbar.maximum() > 0)
        scrollbar.setValue(0)

        workspace._append_thinking_step("design", "生成分析方案", "继续处理新问题")
        wait_until(qt_app, lambda: scrollbar.value() == scrollbar.maximum())
    finally:
        window.close()


def test_grouped_process_timeline_keeps_current_step_open(qt_app: QApplication):
    widget = ThinkingMessageWidget()
    try:
        base = {
            "session_id": "session-1",
            "flow_id": "flow-1",
            "phase": "analysis",
            "source": "model",
        }
        events = [
            ProcessEvent(
                **base,
                round=1,
                step_id="round-1",
                sequence=1,
                event_type="thinking_started",
                content="正在分析已有证据",
            ),
            ProcessEvent(
                **base,
                round=1,
                step_id="round-1",
                sequence=2,
                event_type="thinking_ready",
                title="思考过程",
                content="先检查数据质量，再决定后续方法。",
            ),
            ProcessEvent(
                **base,
                round=1,
                step_id="round-1",
                sequence=3,
                event_type="action_started",
                action_id="call-1",
                tool_name="data_quality",
                title="数据质量检查",
            ),
            ProcessEvent(
                **base,
                round=1,
                step_id="round-1",
                sequence=4,
                event_type="action_completed",
                action_id="call-1",
                tool_name="data_quality",
                title="数据质量检查",
                result="时间轴和缺失率检查通过",
            ),
            ProcessEvent(
                **base,
                round=1,
                step_id="round-1",
                sequence=5,
                event_type="step_completed",
                result="1 个行动已完成",
            ),
            ProcessEvent(
                **base,
                round=2,
                step_id="round-2",
                sequence=6,
                event_type="thinking_started",
                content="根据质量结果选择规律分析方法",
            ),
        ]
        for event in events:
            assert widget.apply_process_event(event)
        assert not widget.apply_process_event(events[-1])
        first, second = widget.process_rows.values()
        assert first.body.isHidden()
        assert not second.body.isHidden()
        assert first.thought_heading.text() == "思考过程"
        assert first.actions["call-1"].result.text() == "时间轴和缺失率检查通过"
        assert second.spinner._timer.isActive()
    finally:
        widget.close()


def test_main_window_uses_one_three_pane_workspace(qt_app: QApplication, tmp_path: Path):
    window = MainWindow(session_store=SessionStore(tmp_path / "sessions.json"))
    try:
        assert window.workspace.count() == 3
        assert not hasattr(window, "tabs")
        assert window.workspace.current_session.title == "新会话"
        assert window.workspace.history.new_button.text() == "＋  新的研究"
        panel = window.workspace.context.data_panel
        assert panel.title_label.text() == "数据"
        assert not hasattr(window.workspace.conversation, "config_label")
        assert window.workspace.current_session.region_id == ""
        assert window.workspace.current_session.region_label == ""
        assert window.workspace.current_session.region_market == ""
        assert panel.region_button.text() == "地区：待选择 ▾"
        assert not any(action.isChecked() for action in panel.region_menu.actions())
        assert panel.state == "empty"
        assert panel.empty_label.text() == "还没取数。请从上方选择地区；也可以直接提问讨论研究方法。"
        assert panel.empty_view.isVisibleTo(panel)
        assert not panel.ready_view.isVisibleTo(panel)
        assert not panel.unavailable_view.isVisibleTo(panel)
        assert list(panel.ready_card.rows) == ["电价", "影响因素", "时间范围"]
        assert not hasattr(window.workspace.context, "inputs")
        assert not hasattr(window.workspace.context, "plan")
        assert window.workspace.context.trace.tree.verticalScrollBar() is not None
        window.workspace.context.trace.maximize_button.click()
        qt_app.processEvents()
        assert window.workspace.history.isHidden()
        assert window.workspace.conversation.isHidden()
        assert window.workspace.context.data_panel.isHidden()
        assert window.workspace.context.trace.maximize_button.text() == "还原"
        window.workspace.context.trace.maximize_button.click()
        qt_app.processEvents()
        assert not window.workspace.history.isHidden()
        assert not window.workspace.conversation.isHidden()
        assert not window.workspace.context.data_panel.isHidden()
        assert not window.workspace.current_session.can_analyze
    finally:
        window.close()


def test_conversation_does_not_require_files_until_analysis(
    qt_app: QApplication,
    model_agent: ResearchCoordinator,
    tmp_path: Path,
):
    window = MainWindow(agent=model_agent, session_store=SessionStore(tmp_path / "sessions.json"))
    try:
        workspace = window.workspace
        workspace.submit_question("先说说电价 EDA 通常应该检查什么？")
        wait_until(qt_app, lambda: not workspace.is_busy)
        assert workspace.current_session.status == "completed"
        assert workspace.current_session.messages[-1].role == "assistant"
        assert "时间轴" in workspace.current_session.messages[-1].content
        assert "季节性" in workspace.current_session.messages[-1].content
        assert not any(
            message.kind in {"thinking", "plan", "data_plan"}
            for message in workspace.current_session.messages
        )
        assert not model_agent.has_thread(workspace.current_session.session_id)
    finally:
        window.close()


def test_target_only_arbitrary_filename_builds_price_only_plan(
    qt_app: QApplication,
    desktop_study: Path,
    model_agent: ResearchCoordinator,
    tmp_path: Path,
):
    window = MainWindow(agent=model_agent, session_store=SessionStore(tmp_path / "sessions.json"))
    try:
        workspace = window.workspace
        choose_files(window, str(desktop_study.parent / "market_prices.csv"))
        config = build_runtime_study(workspace.current_session)
        assert config.target.path.name == "market_prices.csv"
        assert config.exogenous == []

        workspace.submit_question("分析电价季节性和尖峰")
        wait_until(qt_app, lambda: not workspace.is_busy)
        plan = workspace.current_session.current_plan
        assert plan is not None
        assert [step["function"] for step in plan["steps"] if step["enabled"]] == [
            "data_quality",
            "price_tukey_outer_fence",
            "price_calendar_group_profile",
        ]
    finally:
        window.close()


def test_three_data_files_build_the_runtime_study_contract(
    qt_app: QApplication,
    desktop_study: Path,
    model_agent: ResearchCoordinator,
    tmp_path: Path,
):
    window = MainWindow(agent=model_agent, session_store=SessionStore(tmp_path / "sessions.json"))
    try:
        select_desktop_data(window, desktop_study)
        session = window.workspace.current_session
        assert session.can_analyze
        assert {Path(item.path).name for item in session.inputs.values()} == {
            "market_prices.csv",
            "measurements.csv",
            "predictions.csv",
        }
        config = build_runtime_study(session)
        assert len(config.exogenous) == 3
        assert {spec.path.name for spec in config.exogenous} == {
            "measurements.csv",
            "predictions.csv",
        }
    finally:
        window.close()


def test_chat_can_set_and_clear_an_explicit_analysis_time_range():
    selected = parse_chat_time_range("仅分析 2026年1月1日 00:15 到 2026年3月31日")
    assert selected is not None
    assert selected.action == "set"
    assert selected.start_time.isoformat() == "2026-01-01T00:15:00"
    assert selected.end_time.isoformat() == "2026-03-31T23:59:59.999999"

    iso = parse_chat_time_range("分析区间 2026-04-01T00:15 至 2026-04-02T23:45")
    assert iso is not None
    assert iso.start_time.isoformat() == "2026-04-01T00:15:00"
    assert iso.end_time.isoformat() == "2026-04-02T23:45:00"

    cleared = parse_chat_time_range("恢复全部时间范围")
    assert cleared is not None
    assert cleared.action == "clear"
    assert cleared.start_time is None and cleared.end_time is None


def test_chat_time_range_never_guesses_a_missing_boundary():
    with pytest.raises(ValueError, match="开始日期和结束日期"):
        parse_chat_time_range("时间范围选择 2026-01-01")


def test_runtime_study_applies_the_session_chat_time_range(tmp_path: Path):
    index = pd.date_range("2026-01-01", periods=96 * 5, freq="15min")
    target = tmp_path / "prices.parquet"
    pd.DataFrame({"timestamp": index, "rt_price": range(len(index))}).to_parquet(target, index=False)
    session = ResearchSession(source_kind="file")
    session.inputs["target"].path = str(target)
    session.analysis_start_time = "2026-01-02T00:00:00"
    session.analysis_end_time = "2026-01-03T23:59:59.999999"

    config = build_runtime_study(session)

    assert config.study.start_time.isoformat() == "2026-01-02T00:00:00"
    assert config.study.end_time.isoformat() == "2026-01-03T23:59:59.999999"


def test_forecast_availability_column_is_inferred_without_a_config_file(tmp_path: Path):
    index = pd.date_range("2026-01-01", periods=24, freq="1h")
    target = tmp_path / "prices.csv"
    forecasts = tmp_path / "forecasts.csv"
    pd.DataFrame({"datetime": index, "price": range(24)}).to_csv(target, index=False)
    pd.DataFrame(
        {
            "datetime": index,
            "available_at": index - pd.Timedelta(hours=1),
            "load_forecast": range(24),
        }
    ).to_csv(forecasts, index=False)
    session = ResearchSession()
    session.inputs["target"].path = str(target)
    session.inputs["forecasts"].path = str(forecasts)

    context = build_runtime_study(session)

    assert [item.name for item in context.exogenous] == ["load_forecast"]
    assert context.exogenous[0].available_at_column == "available_at"


def test_actual_availability_column_is_metadata_not_a_variable(tmp_path: Path):
    index = pd.date_range("2026-01-01", periods=24, freq="1h")
    target = tmp_path / "prices.csv"
    actuals = tmp_path / "actuals.parquet"
    pd.DataFrame({"datetime": index, "price": range(24)}).to_csv(target, index=False)
    pd.DataFrame(
        {
            "datetime": index,
            "available_at": index + pd.Timedelta(days=1),
            "temperature": range(24),
        }
    ).to_parquet(actuals, index=False)
    session = ResearchSession()
    session.inputs["target"].path = str(target)
    session.inputs["actuals"].path = str(actuals)

    context = build_runtime_study(session)

    assert [item.name for item in context.exogenous] == ["temperature"]
    assert context.exogenous[0].available_at_column == "available_at"


def test_parquet_numeric_series_survives_an_empty_early_sample(tmp_path: Path):
    index = pd.date_range("2026-01-01", periods=3000, freq="15min")
    target = tmp_path / "prices.parquet"
    forecasts = tmp_path / "forecasts.parquet"
    pd.DataFrame({"timestamp": index, "rt_price": range(len(index))}).to_parquet(target, index=False)
    sparse = pd.Series(float("nan"), index=range(len(index)))
    sparse.iloc[2600:] = range(400)
    pd.DataFrame(
        {
            "timestamp": index,
            "available_at": index - pd.Timedelta(hours=1),
            "forecast_solar": sparse,
        }
    ).to_parquet(forecasts, index=False)

    context = infer_study_context(target_path=target, forecasts_path=forecasts)

    assert [item.name for item in context.exogenous] == ["forecast_solar"]


def test_session_history_persists_across_window_restart(
    qt_app: QApplication,
    desktop_study: Path,
    tmp_path: Path,
):
    store = SessionStore(tmp_path / "sessions.json")
    first = MainWindow(session_store=store)
    select_desktop_data(first, desktop_study)
    session_id = first.workspace.current_session.session_id
    first.workspace.rename_session(session_id, "负荷关系研究")
    first.close()

    second = MainWindow(session_store=store)
    try:
        assert second.workspace.current_session.session_id != session_id
        assert second.workspace.current_session.title == "新会话"
        persisted = next(session for session in second.workspace.sessions if session.session_id == session_id)
        assert persisted.title == "负荷关系研究"
        assert persisted.can_analyze
    finally:
        second.close()


def test_selecting_sessions_does_not_reorder_history_when_state_is_reconciled(
    qt_app: QApplication,
    model_agent: ResearchCoordinator,
    tmp_path: Path,
):
    window = MainWindow(agent=model_agent, session_store=SessionStore(tmp_path / "sessions.json"))
    try:
        workspace = window.workspace
        older = workspace.current_session
        older.title = "较早会话"
        older.updated_at = "2026-09-11T08:00:00+00:00"
        workspace.create_session()
        newer = workspace.current_session
        newer.title = "较新会话"
        newer.updated_at = "2026-09-12T08:00:00+00:00"
        workspace.history.set_sessions(workspace.sessions, newer.session_id)

        def history_session_ids() -> list[str]:
            return [
                session_id
                for index in range(workspace.history.list.count())
                if (session_id := str(workspace.history.list.item(index).data(Qt.ItemDataRole.UserRole) or ""))
            ]

        expected_order = [newer.session_id, older.session_id]
        assert history_session_ids() == expected_order

        # Restoring a Graph snapshot currently persists via _persist_and_render,
        # which touches the selected session unless select_session preserves its
        # activity timestamp.
        with (
            patch.object(model_agent, "has_thread", return_value=True),
            patch.object(model_agent, "get_snapshot", return_value=object()),
            patch.object(
                workspace,
                "_loop_completed",
                side_effect=lambda _snapshot: workspace._persist_and_render(keep_timeline=True),
            ),
        ):
            workspace.select_session(older.session_id)

        assert older.updated_at == "2026-09-11T08:00:00+00:00"
        assert history_session_ids() == expected_order
    finally:
        window.close()


def _session_with_plan(plan: dict) -> ResearchSession:
    session = ResearchSession(title="待执行方案", current_plan=plan, status="awaiting_plan_approval")
    session.messages.append(SessionMessage(role="assistant", kind="plan", payload={"plan": plan}))
    return session


def _installed_skill_versions() -> dict[str, str]:
    registry = SkillRegistry.default(Settings(skill_paths=""))
    return {str(item["name"]): str(item["version"]) for item in registry.metadata()}


def _fresh_plan(desktop_study: Path, model_agent: ResearchCoordinator) -> dict:
    return model_agent.propose(question="分析电价分布", config_path=desktop_study).plan.model_dump(mode="json")


def test_a_plan_from_an_older_skill_retires_when_the_session_opens(
    desktop_study: Path, model_agent: ResearchCoordinator, tmp_path: Path
):
    """The execution gate rejects it either way; retiring on open avoids a failed run."""

    plan = _fresh_plan(desktop_study, model_agent)
    plan["skill_version"] = "0.0.1-before-this-build"
    store = SessionStore(tmp_path / "sessions.json")
    store.save([_session_with_plan(plan)])

    restored = store.load(_installed_skill_versions())[0]

    assert restored.current_plan is None
    assert restored.plan_stale
    assert restored.status == "idle"
    assert restored.messages[0].kind == "notice"
    assert restored.messages[0].content == STALE_PLAN_NOTICE


def test_a_plan_from_an_older_function_version_retires_when_the_session_opens(
    desktop_study: Path, model_agent: ResearchCoordinator, tmp_path: Path
):
    plan = _fresh_plan(desktop_study, model_agent)
    plan["steps"] = [{**step, "function_version": "0.0.1"} for step in plan["steps"]]
    store = SessionStore(tmp_path / "sessions.json")
    store.save([_session_with_plan(plan)])

    restored = store.load(_installed_skill_versions())[0]

    assert restored.current_plan is None
    assert restored.plan_stale


def test_a_plan_naming_a_removed_function_retires_when_the_session_opens(
    desktop_study: Path, model_agent: ResearchCoordinator, tmp_path: Path
):
    plan = _fresh_plan(desktop_study, model_agent)
    plan["steps"] = [{**step, "function": "a_function_this_build_no_longer_ships"} for step in plan["steps"]]
    store = SessionStore(tmp_path / "sessions.json")
    store.save([_session_with_plan(plan)])

    assert store.load(_installed_skill_versions())[0].current_plan is None


def test_a_current_plan_survives_reopening(
    desktop_study: Path, model_agent: ResearchCoordinator, tmp_path: Path
):
    plan = _fresh_plan(desktop_study, model_agent)
    store = SessionStore(tmp_path / "sessions.json")
    store.save([_session_with_plan(plan)])

    restored = store.load(_installed_skill_versions())[0]

    assert restored.current_plan is not None
    assert not restored.plan_stale
    assert restored.status == "awaiting_plan_approval"
    assert restored.messages[0].kind == "plan"


def test_result_card_summarises_instead_of_repeating_the_report(qt_app: QApplication):
    """Everything the card drops is written in full to report.md and methods.md."""

    payload = {
        "report_path": "run/report.md",
        "artifact_directory": "run",
        "figure_count": 6,
        "evaluation": {
            "decision": "accept",
            "summary": "本轮登记的 7 条判断全部得到明确结论。",
            "findings": [f"发现 {index}" for index in range(1, 6)],
            "warnings": ["load：缺少预测发布或可获得时间。", "关系系数可能受到共同趋势影响。"],
            "suggested_followups": ["进入预测实验前排除仅事后可得的变量。"],
            "hypothesis_assessments": [
                {"hypothesis": "电价存在日内结构。", "status": "candidate_support", "evidence": "解释 33.9% 方差。"}
            ],
            "checks": [{"name": "数据风险", "status": "warning", "message": "检测到 3 个风险。", "scope": "inherent"}],
        },
    }

    card = ResultMessageWidget.from_payload(payload)
    texts = [widget.text() for widget in card.findChildren(QLabel)]
    body = " | ".join(texts)

    assert "7 条判断全部得到明确结论" in body
    assert sum(1 for text in texts if text.startswith("· ")) == ResultMessageWidget.MAXIMUM_FINDINGS
    assert "发现 4" not in body and "发现 5" not in body
    # Limitations, hypothesis verdicts and validity checks live in the package, not the card.
    assert "缺少预测发布" not in body
    assert "电价存在日内结构" not in body
    assert "检测到 3 个风险" not in body
    assert "报告含 6 张图表，2 条适用边界，1 条下一步建议" in body


def test_result_card_names_open_items_only_when_a_decision_is_needed(qt_app: QApplication):
    payload = {
        "report_path": "run/report.md",
        "artifact_directory": "run",
        "figure_count": 2,
        "evaluation": {
            "decision": "need_user",
            "summary": "需要你决定是否补充数据。",
            "findings": [],
            "warnings": [],
            "suggested_followups": [],
            "hypothesis_assessments": [],
            "checks": [
                {"name": "目标覆盖率", "status": "fail", "message": "覆盖率过低。", "scope": "needs_data"},
                {"name": "数据风险", "status": "warning", "message": "可获得性限制。", "scope": "inherent"},
            ],
        },
    }

    body = " | ".join(widget.text() for widget in ResultMessageWidget.from_payload(payload).findChildren(QLabel))

    assert "需要你决定：目标覆盖率" in body
    # Inherent limitations are not decisions the user has to make.
    assert "数据风险" not in body
    assert "覆盖率过低" not in body


def test_retiring_a_stale_plan_keeps_the_conversation_and_past_reports(
    desktop_study: Path, model_agent: ResearchCoordinator, tmp_path: Path
):
    plan = _fresh_plan(desktop_study, model_agent)
    plan["skill_version"] = "0.0.1-before-this-build"
    session = _session_with_plan(plan)
    session.messages.insert(0, SessionMessage(role="user", kind="text", content="负荷对电价影响有多大？"))
    session.messages.append(
        SessionMessage(role="assistant", kind="result", content="已完成", payload={"report_path": "r/report.md"})
    )
    session.report_path = "r/report.md"
    store = SessionStore(tmp_path / "sessions.json")
    store.save([session])

    restored = store.load(_installed_skill_versions())[0]

    assert [message.kind for message in restored.messages] == ["text", "notice", "result"]
    assert restored.messages[0].content == "负荷对电价影响有多大？"
    assert restored.report_path == "r/report.md"


def test_old_session_plan_without_skill_and_tool_versions_is_invalidated(
    desktop_study: Path,
    model_agent: ResearchCoordinator,
    tmp_path: Path,
):
    stored_plan = model_agent.propose(question="分析电价分布", config_path=desktop_study).plan.model_dump(mode="json")
    stored_plan.pop("skill_name")
    stored_plan.pop("skill_version")
    for step in stored_plan["steps"]:
        step.pop("function_version")
    session = ResearchSession(
        schema_version=3,
        title="旧计划",
        status="awaiting_plan_approval",
        current_plan=stored_plan,
        messages=[
            SessionMessage(
                role="assistant",
                kind="plan",
                content="旧计划",
                payload={"plan": stored_plan},
            )
        ],
    )
    store = SessionStore(tmp_path / "sessions.json")
    store.save([session])

    restored = store.load()[0]

    assert restored.schema_version == SESSION_SCHEMA_VERSION
    assert restored.current_plan is None
    assert restored.plan_stale
    assert restored.status == "idle"
    assert restored.messages[0].kind == "notice"
    assert restored.messages[0].content == STALE_PLAN_NOTICE


def _write_records(store: SessionStore, records: list[dict]) -> None:
    store.path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")


def _stored(store: SessionStore, *titles: str) -> list[dict]:
    store.save([ResearchSession(title=title) for title in titles])
    return json.loads(store.path.read_text(encoding="utf-8"))


def test_one_unreadable_session_never_costs_the_whole_history(tmp_path: Path):
    store = SessionStore(tmp_path / "sessions.json")
    records = _stored(store, "重要研究 A", "重要研究 B")
    records.append({**records[0], "session_id": "broken", "title": "坏记录", "status": "not-a-status"})
    _write_records(store, records)

    loaded = store.load()

    assert [session.title for session in loaded] == ["重要研究 A", "重要研究 B"]
    quarantine = list(tmp_path.glob("sessions.damaged-*.json"))
    assert len(quarantine) == 1
    assert json.loads(quarantine[0].read_text(encoding="utf-8"))[0]["record"]["title"] == "坏记录"
    assert any("已隔离到" in notice for notice in store.recovery_notices)

    store.save(loaded)
    assert len(json.loads(store.path.read_text(encoding="utf-8"))) == 2


def test_a_session_stored_before_the_step_rename_still_opens(tmp_path: Path):
    store = SessionStore(tmp_path / "sessions.json")
    session = ResearchSession(schema_version=7, title="重命名之前的会话")
    session.current_plan = {
        "planner": "llm",
        "skill_name": "price-exogenous-eda",
        "skill_version": "3.1.0",
        "steps": [{"tool": "price_descriptive_distribution", "tool_version": "1.0.0"}],
    }
    session.messages.append(
        SessionMessage(role="assistant", kind="plan", payload={"plan": session.current_plan})
    )
    store.save([session])

    restored = store.load()[0]

    assert restored.schema_version == SESSION_SCHEMA_VERSION
    assert restored.current_plan is not None
    step = restored.current_plan["steps"][0]
    assert step == {"function": "price_descriptive_distribution", "function_version": "1.0.0"}
    assert restored.messages[0].kind == "plan"


def test_retired_research_config_slot_is_removed_from_an_old_session(tmp_path: Path):
    store = SessionStore(tmp_path / "sessions.json")
    record = ResearchSession(title="旧数据入口").model_dump(mode="json")
    record["schema_version"] = 8
    record["inputs"]["config"] = {
        "role": "config",
        "label": "研究配置",
        "path": "old-study.yaml",
        "status": "selected",
    }
    _write_records(store, [record])

    restored = store.load()[0]

    assert restored.schema_version == SESSION_SCHEMA_VERSION
    assert list(restored.inputs) == ["target", "actuals", "forecasts"]


def test_a_session_written_by_a_newer_build_is_kept_not_downgraded(tmp_path: Path):
    store = SessionStore(tmp_path / "sessions.json")
    records = _stored(store, "当前版本会话")
    records.append({**records[0], "session_id": "future", "schema_version": SESSION_SCHEMA_VERSION + 1})
    _write_records(store, records)

    loaded = store.load()

    assert [session.title for session in loaded] == ["当前版本会话"]
    quarantine = list(tmp_path.glob("sessions.damaged-*.json"))
    assert "更新版本的程序" in quarantine[0].read_text(encoding="utf-8")


def test_unknown_fields_from_a_newer_build_do_not_fail_a_session(tmp_path: Path):
    store = SessionStore(tmp_path / "sessions.json")
    records = _stored(store, "会话 A", "会话 B")
    records[1]["a_future_field"] = {"anything": 1}
    _write_records(store, records)

    assert [session.title for session in store.load()] == ["会话 A", "会话 B"]
    assert not list(tmp_path.glob("sessions.damaged-*.json"))


def test_a_corrupt_file_is_preserved_and_never_overwritten(tmp_path: Path):
    store = SessionStore(tmp_path / "sessions.json")
    _stored(store, "唯一的研究")
    original = store.path.read_text(encoding="utf-8")
    store.path.write_text(original[: len(original) // 2], encoding="utf-8")

    assert store.load() == []
    preserved = list(tmp_path.glob("sessions.corrupt-*.json"))
    assert len(preserved) == 1
    assert preserved[0].read_text(encoding="utf-8") == original[: len(original) // 2]

    store.save([ResearchSession(title="新的研究")])
    assert preserved[0].is_file()
    assert [item["title"] for item in json.loads(store.path.read_text(encoding="utf-8"))] == ["新的研究"]


def test_each_save_keeps_one_recoverable_generation(tmp_path: Path):
    store = SessionStore(tmp_path / "sessions.json")
    _stored(store, "第一版")
    _stored(store, "第二版")

    backup = tmp_path / "sessions.json.bak"
    assert [item["title"] for item in json.loads(backup.read_text(encoding="utf-8"))] == ["第一版"]

    store.path.unlink()
    assert [session.title for session in store.load()] == ["第一版"]
    assert any("从备份" in notice for notice in store.recovery_notices)


def test_recovery_notices_reach_the_conversation_once(
    qt_app: QApplication,
    tmp_path: Path,
    model_agent: ResearchCoordinator,
):
    store = SessionStore(tmp_path / "sessions.json")
    records = _stored(store, "保留下来的研究")
    records.append({**records[0], "session_id": "broken", "status": "not-a-status"})
    _write_records(store, records)

    window = MainWindow(agent=model_agent, session_store=store)
    try:
        workspace = window.workspace
        notices = [
            message.content
            for message in workspace.current_session.messages
            if message.kind == "notice"
        ]
        assert any("已隔离到" in notice for notice in notices)

        workspace.create_session()
        assert not [
            message.content
            for message in workspace.current_session.messages
            if message.kind == "notice"
        ]
    finally:
        window.close()


def test_continuous_conversation_runs_plan_and_answers_followup(
    qt_app: QApplication,
    desktop_study: Path,
    model_agent: ResearchCoordinator,
    tmp_path: Path,
):
    store = SessionStore(tmp_path / "sessions.json")
    window = MainWindow(agent=model_agent, session_store=store)
    select_desktop_data(window, desktop_study)
    try:
        workspace = window.workspace
        workspace.submit_question("分析 actual_load 与电价 2 小时的滞后关系")
        wait_until(qt_app, lambda: not workspace.is_busy)
        session = workspace.current_session
        assert session.status == "awaiting_plan_approval"
        assert session.current_plan is not None
        assert workspace.conversation.current_plan_widget is not None
        assert not workspace.conversation.current_plan_widget.reject_button.isHidden()
        assert not hasattr(workspace.conversation.current_plan_widget, "editor_toggle")
        assert all(
            not hasattr(row, "setChecked")
            for row in workspace.conversation.current_plan_widget.step_checks.values()
        )
        assert "等待你确认" in workspace.conversation.current_plan_widget.status_label.text()
        assert not workspace._plan_feedback_timer.isActive()
        assert session.inputs["actuals"].variables
        assert all(event.status != "running" for event in session.trace)

        approved = workspace.conversation.current_plan_widget.approved_plan()
        workspace.run_plan(approved)
        wait_until(qt_app, lambda: not workspace.is_busy)
        assert session.status == "completed"
        assert session.report_path and Path(session.report_path).is_file()
        assert any(message.kind == "result" for message in session.messages)
        assert any(event.category == "tool" and event.duration_ms is not None for event in session.trace)

        before = len(session.messages)
        workspace.submit_question("这个结果说明什么？")
        wait_until(
            qt_app,
            lambda: (
                not workspace.is_busy
                and session.messages
                and session.messages[-1].role == "assistant"
                and session.messages[-1].kind == "text"
            ),
        )
        assert len(session.messages) == before + 2
        assert session.messages[-1].role == "assistant"
        assert "描述性" in session.messages[-1].content
        assert all(message.kind != "thinking" for message in session.messages[before:])
        assert session.latest_eda_summary is not None
        assert session.latest_evaluation is not None
        assert len(session.runs) == 1

        window.close()
        restored = MainWindow(agent=model_agent, session_store=store)
        try:
            restored.workspace.select_session(session.session_id)
            restored_session = restored.workspace.current_session
            assert restored_session.status == "completed"
            assert any(message.kind == "result" for message in restored_session.messages)
            assert restored.workspace.conversation.current_plan_widget is not None
            assert restored_session.latest_eda_summary is not None
            assert restored_session.latest_evaluation is not None
            assert len(restored_session.runs) == 1
        finally:
            restored.close()
    finally:
        if window.isVisible():
            window.close()


def test_plan_reject_button_stops_without_running_research(
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
        widget = workspace.conversation.current_plan_widget
        assert widget is not None

        widget.reject_button.click()
        wait_until(qt_app, lambda: not workspace.is_busy)

        assert workspace.current_session.status == "stopped"
        assert workspace.current_session.runs == []
        assert widget.run_button.isHidden()
    finally:
        window.close()


def test_repeated_assistant_text_projects_once_per_turn_by_message_id(
    qt_app: QApplication,
    model_agent: ResearchCoordinator,
    tmp_path: Path,
):
    window = MainWindow(agent=model_agent, session_store=SessionStore(tmp_path / "sessions.json"))
    try:
        workspace = window.workspace
        expected = "先检查时间轴和季节性，再选择具体 EDA 方法。"
        for question in ("先解释研究方法", "请再解释一次"):
            workspace.submit_question(question)
            wait_until(qt_app, lambda: not workspace.is_busy)

        replies = [
            message
            for message in workspace.current_session.messages
            if message.role == "assistant" and message.kind == "text" and message.content == expected
        ]
        user_turns = [
            message.turn_id
            for message in workspace.current_session.messages
            if message.role == "user" and message.kind == "text"
        ]
        assert len(replies) == 2
        assert len({message.message_id for message in replies}) == 2
        assert [message.turn_id for message in replies] == user_turns
    finally:
        window.close()


def test_thinking_process_is_visible_and_survives_a_restart(
    qt_app: QApplication,
    desktop_study: Path,
    model_agent: ResearchCoordinator,
    tmp_path: Path,
):
    store = SessionStore(tmp_path / "sessions.json")
    window = MainWindow(agent=model_agent, session_store=store)
    select_desktop_data(window, desktop_study)
    try:
        workspace = window.workspace
        workspace.submit_question("分析 actual_load 与电价 2 小时的滞后关系")
        wait_until(qt_app, lambda: not workspace.is_busy)
        session = workspace.current_session

        assert not any(message.kind == "thinking" for message in session.messages)

        workspace.run_plan(workspace.conversation.current_plan_widget.approved_plan())
        wait_until(qt_app, lambda: not workspace.is_busy)

        execution_card = [message for message in session.messages if message.kind == "thinking"][-1]
        events = execution_card.payload["process_events"]
        assert execution_card.payload["state"] == "completed"
        step_ids = list(dict.fromkeys(event["step_id"] for event in events))
        assert len(step_ids) >= len(workspace.current_session.current_plan["steps"])
        for step_id in step_ids[: len(workspace.current_session.current_plan["steps"])]:
            types = [event["event_type"] for event in events if event["step_id"] == step_id]
            assert types[:2] == ["thinking_ready", "action_started"]
            assert "action_completed" in types
            assert "step_completed" in types
        assert all(
            event["source"] == "system"
            for event in events
            if event["event_type"] == "thinking_ready"
        )

        window.close()
        restored = MainWindow(agent=model_agent, session_store=store)
        try:
            restored.workspace.select_session(session.session_id)
            widgets = [
                widget
                for widget in restored.workspace.conversation._message_widgets.values()
                if isinstance(widget, ThinkingMessageWidget)
            ]
            assert len(widgets) == 1
            assert widgets[-1].process_events == events
            rows = list(widgets[-1].process_rows.values())
            assert rows
            assert all(row.body.isHidden() for row in rows[:-1])
            assert not rows[-1].body.isHidden()
            assert widgets[-1].steps_container.isVisibleTo(widgets[-1])
            widgets[-1].toggle_button.click()
            assert not widgets[-1].steps_container.isVisible()
        finally:
            restored.close()
    finally:
        if window.isVisible():
            window.close()


def test_run_trace_reads_as_a_professional_audit_log(
    qt_app: QApplication,
    desktop_study: Path,
    model_agent: ResearchCoordinator,
    tmp_path: Path,
):
    store = SessionStore(tmp_path / "sessions.json")
    window = MainWindow(agent=model_agent, session_store=store)
    select_desktop_data(window, desktop_study)
    try:
        workspace = window.workspace
        workspace.submit_question("分析 actual_load 与电价 2 小时的滞后关系")
        wait_until(qt_app, lambda: not workspace.is_busy)
        workspace.run_plan(workspace.conversation.current_plan_widget.approved_plan())
        wait_until(qt_app, lambda: not workspace.is_busy)

        tree = workspace.context.trace.tree
        rows = [
            (tree.topLevelItem(index).text(1), tree.topLevelItem(index).text(2))
            for index in range(tree.topLevelItemCount())
        ]
        stages = [stage for stage, _text in rows]
        texts = [text for _stage, text in rows]

        # Every visible row is narrated: no internal identifiers, no second person.
        assert rows
        assert not any("_" in text and "[" in text for text in texts)
        assert not any(word in text for text in texts for word in ("你", "plan_id", "run_id", "advance_tool"))
        assert not any(text.startswith("系统事件") for text in texts)
        assert set(stages) <= set(STAGE_LABELS.values())

        # Every loaded input stays visible; only duplicate lifecycle rows for the
        # same completed function are collapsed.
        titles = [text.split(" — ")[0] for text in texts]
        assert titles.count("改用本地文件") == 1
        function_titles = [
            title
            for title in titles
            if title.startswith(("分析中 · ", "已完成 · ", "已复用 · "))
        ]
        assert function_titles == list(dict.fromkeys(function_titles))
        running = {title.removeprefix("分析中 · ") for title in titles if title.startswith("分析中 · ")}
        finished = {title.removeprefix("已完成 · ") for title in titles if title.startswith("已完成 · ")}
        enabled = {step["title"] for step in workspace.current_session.current_plan["steps"] if step["enabled"]}
        assert running == finished == enabled

        # A running row explains the method; the finished row only reports elapsed time.
        started = next(text for text in texts if text.startswith("分析中 · 数据质量与时间对齐"))
        completed = next(text for text in texts if text.startswith("已完成 · 数据质量与时间对齐"))
        assert started.split(" — ")[1] == FUNCTION_CATALOG["data_quality"].description.rstrip("。")
        assert completed.split(" — ")[1].endswith(("毫秒", "秒"))

        # The locked batch is named by the research stages it covers.
        locked = next(text for text in texts if text.startswith("锁定执行计划"))
        assert "数据可用性核验 1 项" in locked
        assert locked.endswith(f"共 {len(enabled)} 项")
    finally:
        if window.isVisible():
            window.close()


def test_plan_can_be_revised_and_executed_through_dialogue(
    qt_app: QApplication,
    desktop_study: Path,
    model_agent: ResearchCoordinator,
    tmp_path: Path,
):
    window = MainWindow(
        agent=model_agent,
        session_store=SessionStore(tmp_path / "sessions.json"),
    )
    select_desktop_data(window, desktop_study)
    try:
        workspace = window.workspace
        workspace.submit_question("分析 actual_load 与电价的滞后关系")
        wait_until(
            qt_app,
            lambda: workspace.current_session.status == "awaiting_plan_approval" and not workspace.is_busy,
        )
        original = workspace.current_session.current_plan
        assert original is not None

        workspace.submit_question("只保留关系分析，把最大滞后改为 4 小时，只分析 actual_wind")
        wait_until(
            qt_app,
            lambda: workspace.current_session.status == "awaiting_plan_approval" and not workspace.is_busy,
        )
        revised = workspace.current_session.current_plan
        assert revised is not None
        assert revised["parent_plan_id"] == original["plan_id"]
        assert revised["revision_source"] == "user_dialogue"
        assert revised["selected_variables"] == ["actual_wind"]
        assert [step["function"] for step in revised["steps"] if step["enabled"]] == [
            "data_quality",
            "relationship_scipy_pearson_pairwise",
            "relationship_scipy_spearman_pairwise",
            "relationship_pearson_positive_lead_scan",
        ]
        relationship = next(
            step for step in revised["steps"] if step["function"] == "relationship_pearson_positive_lead_scan"
        )
        assert relationship["parameters"]["max_lag"] == 4

        workspace.submit_question("按这个执行")
        wait_until(qt_app, lambda: not workspace.is_busy, timeout_seconds=20.0)
        assert workspace.current_session.status == "completed"
        assert workspace.current_session.runs
        assert workspace.current_session.latest_eda_summary is not None
        assert list(workspace.current_session.latest_eda_summary["relationships"]["series"]) == ["actual_wind"]
    finally:
        window.close()


def test_replacing_input_invalidates_existing_plan(
    qt_app: QApplication,
    desktop_study: Path,
    model_agent: ResearchCoordinator,
    tmp_path: Path,
):
    window = MainWindow(
        agent=model_agent,
        session_store=SessionStore(tmp_path / "sessions.json"),
    )
    select_desktop_data(window, desktop_study)
    try:
        workspace = window.workspace
        workspace.submit_question("分析电价与预测发电的关系")
        wait_until(qt_app, lambda: not workspace.is_busy)
        assert workspace.current_session.current_plan is not None
        plan = workspace.current_session.current_plan
        workspace.current_session.runs.append(
            SessionRunRecord(
                run_id="historical-run",
                plan_id=plan["plan_id"],
                question=plan["question"],
                artifact_directory="historical",
                report_path="historical/report.md",
                data_fingerprint=plan["data_fingerprint"],
            )
        )

        replacement = tmp_path / "replacement" / "alternate_predictions.csv"
        replacement.parent.mkdir()
        pd.read_csv(tmp_path / "predictions.csv").to_csv(replacement, index=False)
        # Only the forecast file actually moves; the other two answers repeat themselves.
        choose_files(
            window,
            workspace.current_session.inputs["target"].path,
            workspace.current_session.inputs["actuals"].path,
            str(replacement),
        )
        assert workspace.current_session.current_plan is None
        assert workspace.current_session.plan_stale
        assert workspace.current_session.runs[0].memory_status == "stale"
        imported = workspace._legacy_graph_import(workspace.current_session)
        assert imported["episode_summaries"][0]["memory_status"] == "stale"
        assert workspace.current_session.messages[-1].kind == "notice"
    finally:
        window.close()


def test_model_plan_auto_executes_after_feedback_window(
    qt_app: QApplication,
    desktop_study: Path,
    model_agent: ResearchCoordinator,
    tmp_path: Path,
):
    window = MainWindow(
        agent=model_agent,
        session_store=SessionStore(tmp_path / "sessions.json"),
        plan_feedback_seconds=1,
        auto_execute_plan=True,
    )
    select_desktop_data(window, desktop_study)
    try:
        workspace = window.workspace
        workspace.submit_question("分析 actual_load 与电价的滞后关系")
        wait_until(
            qt_app,
            lambda: workspace.current_session.status == "completed" and not workspace.is_busy,
            timeout_seconds=20.0,
        )

        assert workspace.current_session.runs
        assert any("没有收到修改意见" in message.content for message in workspace.current_session.messages)
        assert any(event.name == "确认超时，自动执行方案" for event in workspace.current_session.trace)
    finally:
        window.close()


def test_invalid_external_skill_is_reported_in_new_session(
    qt_app: QApplication,
    model_agent: ResearchCoordinator,
    tmp_path: Path,
):
    invalid_skill = tmp_path / "skills" / "broken-skill"
    invalid_skill.mkdir(parents=True)
    (invalid_skill / "SKILL.md").write_text("missing frontmatter", encoding="utf-8")
    skills = SkillRegistry.default(Settings(skill_paths=str(tmp_path / "skills")))
    agent = ResearchCoordinator(
        main_agent=model_agent.main_agent,
        eda_subagent=model_agent.eda_subagent,
        skills=skills,
    )
    window = MainWindow(agent=agent, session_store=SessionStore(tmp_path / "sessions.json"))
    try:
        session = window.workspace.current_session
        assert any("外部研究方法包没能加载" in message.content for message in session.messages)
        assert any(event.name == "外部研究方法包加载失败" for event in session.trace)
    finally:
        window.close()
        qt_app.processEvents()


def test_expired_approval_restores_from_sqlite_and_requires_explicit_confirmation(
    qt_app: QApplication,
    desktop_study: Path,
    model_agent: ResearchCoordinator,
    tmp_path: Path,
):
    store = SessionStore(tmp_path / "sessions.json")
    checkpoint = tmp_path / "research_graph.sqlite3"
    first_agent = ResearchCoordinator(
        main_agent=model_agent.main_agent,
        eda_subagent=model_agent.eda_subagent,
        checkpoint_path=checkpoint,
    )
    first = MainWindow(
        agent=first_agent,
        session_store=store,
        plan_feedback_seconds=1,
        auto_execute_plan=True,
    )
    select_desktop_data(first, desktop_study)
    session_id = first.workspace.current_session.session_id
    first.workspace.submit_question("分析 actual_load 与电价关系")
    wait_until(qt_app, lambda: not first.workspace.is_busy)
    assert first.workspace.current_session.status == "awaiting_plan_approval"
    first.close()
    first_agent.close()

    time.sleep(1.05)
    restored_agent = ResearchCoordinator(
        main_agent=model_agent.main_agent,
        eda_subagent=model_agent.eda_subagent,
        checkpoint_path=checkpoint,
    )
    restored = MainWindow(
        agent=restored_agent,
        session_store=store,
        plan_feedback_seconds=1,
        auto_execute_plan=True,
    )
    try:
        restored.workspace.select_session(session_id)
        session = restored.workspace.current_session
        assert session.status == "awaiting_plan_approval"
        assert not restored.workspace._plan_feedback_timer.isActive()
        assert "审批已过期" in restored.workspace.conversation.current_plan_widget.status_label.text()

        restored.workspace.run_plan(restored.workspace.conversation.current_plan_widget.approved_plan())
        wait_until(qt_app, lambda: not restored.workspace.is_busy, timeout_seconds=20.0)
        assert restored.workspace.current_session.status == "completed"
        assert restored.workspace.current_session.runs
    finally:
        restored.close()
        restored_agent.close()
