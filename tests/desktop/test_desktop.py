"""Headless functional checks for the three-pane PySide6 research workspace."""

from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest
import yaml
from PySide6.QtWidgets import QApplication

from app.config import Settings
from app.desktop.input_config import build_session_study_config
from app.desktop.main_window import MainWindow
from app.desktop.message_widgets import ThinkingMessageWidget
from app.desktop.session import ResearchSession, SessionMessage, SessionStore
from app.research.agent.orchestrator import DialogueDecision, MainResearchAgent
from app.research.agent.subagents.eda import EDASubagent
from app.research.application.coordinator import ResearchCoordinator
from app.research.graph.narration import STAGE_LABELS
from app.research.schemas.study import load_study_config
from app.research.skills.registry import SkillRegistry
from app.research.tools.catalog import TOOL_CATALOG


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
                "hypotheses": ["电价可能存在用户关注的结构特征。"],
                "selected_variables": [],
                "steps": [
                    {
                        "tool": function_name,
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
            "hypotheses": ["所选变量可能与电价存在描述性关系。"],
            "selected_variables": selected,
            "steps": [
                {
                    "tool": "price_calendar_group_profile",
                    "enabled": True,
                    "rationale": "检查电价周期结构。",
                    "parameters": {},
                },
                {
                    "tool": "price_lag_autocorrelation",
                    "enabled": True,
                    "rationale": "检查电价滞后结构。",
                    "parameters": {"max_lag": lag},
                },
                {
                    "tool": "exogenous_descriptive_distribution",
                    "enabled": True,
                    "rationale": "检查变量画像。",
                    "parameters": {"variables": selected},
                },
                {
                    "tool": "exogenous_iqr_outliers",
                    "enabled": True,
                    "rationale": "检查变量异常值。",
                    "parameters": {"variables": selected},
                },
                {
                    "tool": "relationship_scipy_pearson_pairwise",
                    "enabled": True,
                    "rationale": "检查同期 Pearson 关系。",
                    "parameters": {"variables": selected},
                },
                {
                    "tool": "relationship_scipy_spearman_pairwise",
                    "enabled": True,
                    "rationale": "检查同期 Spearman 关系。",
                    "parameters": {"variables": selected},
                },
                {
                    "tool": "relationship_pearson_positive_lead_scan",
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
                enabled_tools=[
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
        if kwargs.get("config") is None:
            return DialogueDecision(intent="discussion", response="先检查时间轴和季节性，再选择具体 EDA 方法。")
        return DialogueDecision(intent="new_plan", skill_name="price-exogenous-eda")


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture
def model_agent() -> ResearchCoordinator:
    return ResearchCoordinator(
        eda_subagent=EDASubagent(model_planner=DesktopModelPlanner()),
        main_agent=MainResearchAgent(model_dialogue=DesktopModelDialogue()),
    )


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


def test_main_window_uses_one_three_pane_workspace(qt_app: QApplication, tmp_path: Path):
    window = MainWindow(session_store=SessionStore(tmp_path / "sessions.json"))
    try:
        assert window.workspace.count() == 3
        assert not hasattr(window, "tabs")
        assert window.workspace.current_session.title == "新会话"
        assert window.workspace.history.new_button.text() == "＋  新的研究"
        assert window.workspace.context.inputs.title_label.text() == "数据文件"
        assert window.workspace.conversation.config_label.text() == "未使用研究配置文件（可选）"
        assert list(window.workspace.context.inputs.rows) == ["config", "target", "actuals", "forecasts"]
        assert all(row.name_label.text() == "尚未选择" for row in window.workspace.context.inputs.rows.values())
        assert all(not hasattr(row, "status_label") for row in window.workspace.context.inputs.rows.values())
        assert all(not hasattr(row, "variables") for row in window.workspace.context.inputs.rows.values())
        assert not hasattr(window.workspace.context, "plan")
        assert window.workspace.context.trace.tree.verticalScrollBar() is not None
        window.workspace.context.trace.maximize_button.click()
        qt_app.processEvents()
        assert window.workspace.history.isHidden()
        assert window.workspace.conversation.isHidden()
        assert window.workspace.context.inputs.isHidden()
        assert window.workspace.context.trace.maximize_button.text() == "还原"
        window.workspace.context.trace.maximize_button.click()
        qt_app.processEvents()
        assert not window.workspace.history.isHidden()
        assert not window.workspace.conversation.isHidden()
        assert not window.workspace.context.inputs.isHidden()
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
        workspace.set_input_file("target", str(desktop_study.parent / "market_prices.csv"))
        config = build_session_study_config(workspace.current_session)
        assert config.target.path.name == "market_prices.csv"
        assert config.exogenous == []

        workspace.submit_question("分析电价季节性和尖峰")
        wait_until(qt_app, lambda: not workspace.is_busy)
        plan = workspace.current_session.current_plan
        assert plan is not None
        assert [step["tool"] for step in plan["steps"] if step["enabled"]] == [
            "data_quality",
            "price_tukey_outer_fence",
            "price_calendar_group_profile",
        ]
    finally:
        window.close()


def test_user_config_selection_populates_exactly_four_inputs(
    qt_app: QApplication,
    desktop_study: Path,
    model_agent: ResearchCoordinator,
    tmp_path: Path,
):
    window = MainWindow(
        config_path=desktop_study,
        agent=model_agent,
        session_store=SessionStore(tmp_path / "sessions.json"),
    )
    try:
        session = window.workspace.current_session
        assert session.can_analyze
        assert {Path(item.path).name for item in session.inputs.values()} == {
            "study_config.yml",
            "market_prices.csv",
            "measurements.csv",
            "predictions.csv",
        }
        config = build_session_study_config(session)
        assert len(config.exogenous) == 3
        assert {spec.path.name for spec in config.exogenous} == {
            "measurements.csv",
            "predictions.csv",
        }
    finally:
        window.close()


def test_yaml_and_exogenous_files_can_be_cleared_after_discovery(
    qt_app: QApplication,
    desktop_study: Path,
    model_agent: ResearchCoordinator,
    tmp_path: Path,
):
    window = MainWindow(
        config_path=desktop_study,
        agent=model_agent,
        session_store=SessionStore(tmp_path / "sessions.json"),
    )
    try:
        workspace = window.workspace
        workspace.clear_input_file("actuals")
        workspace.clear_input_file("forecasts")
        config = build_session_study_config(workspace.current_session)
        assert config.exogenous == []
        assert workspace.current_session.inputs["config"].path
        assert workspace.current_session.inputs["target"].path

        workspace.clear_input_file("config")
        inferred = build_session_study_config(workspace.current_session)
        assert inferred.target.path.name == "market_prices.csv"
        assert inferred.exogenous == []
    finally:
        window.close()


def test_default_phase1_config_excludes_day_ahead_price_file():
    root = Path(__file__).resolve().parents[2]
    config = load_study_config(root / "configs" / "research" / "price_exogenous_eda.yaml")
    assert {spec.path.name for spec in config.exogenous} == {
        "feature_actuals.csv",
        "feature_forecasts.csv",
    }
    assert all(spec.name != "da_price" for spec in config.exogenous)


def test_session_history_persists_across_window_restart(
    qt_app: QApplication,
    desktop_study: Path,
    tmp_path: Path,
):
    store = SessionStore(tmp_path / "sessions.json")
    first = MainWindow(config_path=desktop_study, session_store=store)
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


def test_old_session_plan_without_skill_and_tool_versions_is_invalidated(
    desktop_study: Path,
    model_agent: ResearchCoordinator,
    tmp_path: Path,
):
    stored_plan = model_agent.propose(question="分析电价分布", config_path=desktop_study).plan.model_dump(mode="json")
    stored_plan.pop("skill_name")
    stored_plan.pop("skill_version")
    for step in stored_plan["steps"]:
        step.pop("tool_version")
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

    assert restored.schema_version == 7
    assert restored.current_plan is None
    assert restored.plan_stale
    assert restored.status == "idle"
    assert restored.messages[0].kind == "notice"
    assert "缺少 Skill 或工具版本" in restored.messages[0].content


def test_continuous_conversation_runs_plan_and_answers_followup(
    qt_app: QApplication,
    desktop_study: Path,
    model_agent: ResearchCoordinator,
    tmp_path: Path,
):
    store = SessionStore(tmp_path / "sessions.json")
    window = MainWindow(config_path=desktop_study, agent=model_agent, session_store=store)
    try:
        workspace = window.workspace
        workspace.submit_question("分析 actual_load 与电价 2 小时的滞后关系")
        wait_until(qt_app, lambda: not workspace.is_busy)
        session = workspace.current_session
        assert session.status == "awaiting_plan_approval"
        assert session.current_plan is not None
        assert workspace.conversation.current_plan_widget is not None
        assert not hasattr(workspace.conversation.current_plan_widget, "editor_toggle")
        assert all(
            not hasattr(row, "setChecked")
            for row in workspace.conversation.current_plan_widget.step_checks.values()
        )
        assert "自动执行" in workspace.conversation.current_plan_widget.status_label.text()
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
        assert len(session.messages) == before + 3
        assert session.messages[-1].role == "assistant"
        assert "描述性" in session.messages[-1].content
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


def test_thinking_process_is_visible_and_survives_a_restart(
    qt_app: QApplication,
    desktop_study: Path,
    model_agent: ResearchCoordinator,
    tmp_path: Path,
):
    store = SessionStore(tmp_path / "sessions.json")
    window = MainWindow(config_path=desktop_study, agent=model_agent, session_store=store)
    try:
        workspace = window.workspace
        workspace.submit_question("分析 actual_load 与电价 2 小时的滞后关系")
        wait_until(qt_app, lambda: not workspace.is_busy)
        session = workspace.current_session

        planning_card = next(message for message in session.messages if message.kind == "thinking")
        stages = [step["stage"] for step in planning_card.payload["steps"]]
        titles = [step["title"] for step in planning_card.payload["steps"]]
        assert planning_card.payload["state"] == "completed"
        assert len(titles) >= 4
        assert "解析问题" in stages and "生成方案" in stages
        assert titles[0] == "解析研究问题"
        assert titles == list(dict.fromkeys(titles))
        assert all(step["status"] != "running" for step in planning_card.payload["steps"])

        workspace.run_plan(workspace.conversation.current_plan_widget.approved_plan())
        wait_until(qt_app, lambda: not workspace.is_busy)

        execution_card = [message for message in session.messages if message.kind == "thinking"][-1]
        steps = execution_card.payload["steps"]
        computed = [step for step in steps if step["stage"] == "执行分析"]
        assert execution_card.payload["state"] == "completed"
        assert len(computed) == len(workspace.current_session.current_plan["steps"])
        assert all(step["title"].startswith("已完成 · ") for step in computed)
        assert all(step["function_name"] for step in computed)
        assert any(step["stage"] == "评估结果" for step in steps)

        window.close()
        restored = MainWindow(agent=model_agent, session_store=store)
        try:
            restored.workspace.select_session(session.session_id)
            widgets = [
                widget
                for widget in restored.workspace.conversation._message_widgets.values()
                if isinstance(widget, ThinkingMessageWidget)
            ]
            assert len(widgets) == 2
            assert widgets[-1].steps == steps
            assert not widgets[-1].steps_container.isVisible()
            widgets[-1].toggle_button.click()
            assert widgets[-1].steps_container.isVisibleTo(widgets[-1])
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
    window = MainWindow(config_path=desktop_study, agent=model_agent, session_store=store)
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

        # One function produces one "分析中" row and one "已完成" row, never a repeated title.
        titles = [text.split(" — ")[0] for text in texts]
        assert titles == list(dict.fromkeys(titles))
        running = {title.removeprefix("分析中 · ") for title in titles if title.startswith("分析中 · ")}
        finished = {title.removeprefix("已完成 · ") for title in titles if title.startswith("已完成 · ")}
        enabled = {step["title"] for step in workspace.current_session.current_plan["steps"] if step["enabled"]}
        assert running == finished == enabled

        # A running row explains the method; the finished row only reports elapsed time.
        started = next(text for text in texts if text.startswith("分析中 · 数据质量与时间对齐"))
        completed = next(text for text in texts if text.startswith("已完成 · 数据质量与时间对齐"))
        assert started.split(" — ")[1] == TOOL_CATALOG["data_quality"].description.rstrip("。")
        assert completed.split(" — ")[1].endswith(("毫秒", "秒"))

        # The locked batch is named by the research stages it covers.
        locked = next(text for text in texts if text.startswith("锁定执行计划"))
        assert "数据体检 1 项" in locked
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
        config_path=desktop_study,
        agent=model_agent,
        session_store=SessionStore(tmp_path / "sessions.json"),
    )
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
        assert [step["tool"] for step in revised["steps"] if step["enabled"]] == [
            "data_quality",
            "relationship_scipy_pearson_pairwise",
            "relationship_scipy_spearman_pairwise",
            "relationship_pearson_positive_lead_scan",
        ]
        relationship = next(
            step for step in revised["steps"] if step["tool"] == "relationship_pearson_positive_lead_scan"
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
        config_path=desktop_study,
        agent=model_agent,
        session_store=SessionStore(tmp_path / "sessions.json"),
    )
    try:
        workspace = window.workspace
        workspace.submit_question("分析电价与预测发电的关系")
        wait_until(qt_app, lambda: not workspace.is_busy)
        assert workspace.current_session.current_plan is not None

        replacement = tmp_path / "replacement" / "alternate_predictions.csv"
        replacement.parent.mkdir()
        pd.read_csv(tmp_path / "predictions.csv").to_csv(replacement, index=False)
        workspace.set_input_file("forecasts", str(replacement))
        assert workspace.current_session.current_plan is None
        assert workspace.current_session.plan_stale
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
        config_path=desktop_study,
        agent=model_agent,
        session_store=SessionStore(tmp_path / "sessions.json"),
        plan_feedback_seconds=1,
    )
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
        config_path=desktop_study,
        agent=first_agent,
        session_store=store,
        plan_feedback_seconds=1,
    )
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
    restored = MainWindow(agent=restored_agent, session_store=store, plan_feedback_seconds=1)
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
