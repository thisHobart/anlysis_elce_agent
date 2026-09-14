"""Controller for the unified three-pane research conversation workspace."""

from __future__ import annotations

import json
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from math import ceil
from pathlib import Path
from typing import Any
from uuid import uuid4

from PySide6.QtCore import Qt, QThread, QTimer, Signal, Slot
from PySide6.QtWidgets import QFileDialog, QMessageBox, QSplitter, QWidget

from app.config import get_settings
from app.desktop.input_config import (
    FILE_FILTERS,
    build_runtime_study,
    parse_chat_time_range,
    validate_input_path,
)
from app.desktop.message_widgets import DataDetailsDialog, ThinkingMessageWidget
from app.desktop.p2_review import P2ReviewDialog
from app.desktop.panes import STATUS_LABELS, ContextPane, ConversationPane, HistoryPane
from app.desktop.session import (
    DataPanelState,
    ResearchSession,
    SessionMessage,
    SessionRunRecord,
    SessionStatus,
    SessionStore,
    TraceEvent,
    VariableEvidence,
    default_input_files,
)
from app.desktop.worker import FunctionWorker
from app.llm.factory import build_model_gateway
from app.research.agent.orchestrator import requests_analysis_before_forecast
from app.research.agent.schemas import (
    ConversationMessage,
    EDAPlan,
    ResearchDataProfile,
    ResearchProposal,
)
from app.research.application.coordinator import ResearchCoordinator
from app.research.data.snapshot import input_file_manifest, study_fingerprint
from app.research.data.sources.regions import (
    RegionPriceFetch,
    RegionProfile,
    RegionSourceError,
    fetch_region_price,
    load_region_profiles,
)
from app.research.data.sources.summary import (
    DataSummary,
    build_partial_summary,
    build_summary,
    parse_summary,
    summary_payload,
)
from app.research.forecasting.contracts import ForecastPlan, ForecastRunResult
from app.research.forecasting.data import prepare_forecast_plan
from app.research.forecasting.workflow import execute_forecast_workflow
from app.research.full_flow.contracts import FlowRunReference, FullFlowState, P2ReviewSummary
from app.research.full_flow.service import FullFlowStore, FullResearchFlow
from app.research.graph.contracts import ApprovalState, EpisodeSummary, ResearchLoopSnapshot
from app.research.graph.narration import (
    STAGE_LABELS,
    ThinkingStep,
    narrate_event,
    progress_message,
    split_progress_message,
    trace_category,
)
from app.research.news import AdaptiveNewsEventExtractor
from app.research.schemas.results import DataQualityReport
from app.research.schemas.study import StudyConfig
from app.runtime_paths import default_p2_news_path, default_research_output_directory


def _known_so_far(config: StudyConfig | None) -> DataSummary | None:
    """Describe the dataset from the study alone, before any of it has been read."""

    if config is None:
        return None
    return build_partial_summary(
        market=config.study.market,
        target_name=config.target.name,
        exogenous_names=[spec.name for spec in config.exogenous],
        frequency=config.study.frequency,
        variable_kinds={
            spec.name: (
                "forecast"
                if spec.availability_type == "forecast"
                else ("actual" if spec.availability_type in {"known_at_timestamp", "observed_only"} else "unknown")
            )
            for spec in config.exogenous
        },
    )


DATA_CHANGED_NOTICE = "数据换了，之前的分析方案已经作废。重新问一次，我按新数据给方案。"


def _is_forecast_request(question: str, *, has_price_context: bool = False) -> bool:
    compact = "".join(question.lower().split())
    if any(word in compact for word in ("说明什么", "结果", "回测", "表现", "限制", "解释", "为什么")):
        return False
    if requests_analysis_before_forecast(question):
        return False
    return (
        any(word in compact for word in ("预测", "预报"))
        and (has_price_context or any(word in compact for word in ("电价", "价格")))
        and any(word in compact for word in ("明天", "明日", "次日", "未来24", "未来一天", "预测一天", "预报一天"))
    )


def _is_forecast_explanation(question: str) -> bool:
    compact = "".join(question.lower().split())
    return any(word in compact for word in ("说明什么", "结果", "回测", "表现", "限制", "解释", "为什么"))


def _is_forecast_text_approval(question: str) -> bool:
    compact = "".join(question.lower().split()).translate(str.maketrans("", "", "，,。！？!?"))
    return compact in {"确认", "确认执行", "立即执行", "按这个执行", "执行当前方案", "可以开始", "同意", "接受"}


def _is_news_analysis_request(question: str) -> bool:
    """Recognize a request to run P2 instead of sending it back through EDA planning."""

    compact = "".join(question.lower().split())
    if "新闻" not in compact:
        return False
    explicit_actions = (
        "开始新闻",
        "进行新闻",
        "请分析新闻",
        "分析一下新闻",
        "分析新闻",
        "请研究新闻",
        "研究一下新闻",
        "研究新闻",
        "结合新闻",
        "加入新闻",
    )
    return any(phrase in compact for phrase in explicit_actions)


def _requests_news_forecast(question: str) -> bool:
    compact = "".join(question.lower().split())
    return _is_news_analysis_request(question) and any(word in compact for word in ("预测", "预报"))


def _requests_new_full_flow(question: str) -> bool:
    compact = "".join(question.casefold().split())
    return any(
        phrase in compact
        for phrase in (
            "新一轮新闻",
            "另起一轮新闻",
            "重新开始新闻分析",
            "新建新闻分析",
        )
    )


def _full_flow_command(question: str) -> str | None:
    compact = "".join(question.casefold().split()).translate(str.maketrans("", "", "，,。！？!?"))
    if compact in {
        "开始p1",
        "开始p1阶段",
        "重新开始p1",
        "重新开始p1阶段",
        "新一轮p1",
        "开始新的研究",
    }:
        return "new_goal"
    if compact in {"继续第三阶段", "进入第三阶段", "开始第三阶段", "继续p3", "开始p3"}:
        return "request_p3"
    if compact in {"继续", "继续执行", "下一步"}:
        return "continue"
    if compact in {"重试", "重新尝试", "再试一次"}:
        return "retry"
    if compact in {"停止", "停止研究", "结束", "终止"}:
        return "stop"
    if any(phrase in compact for phrase in ("如何复核", "怎么复核", "复合p2", "复核p2", "打开复核")):
        return "review_help"
    if any(phrase in compact for phrase in ("完成p2复核", "完成p2的复核", "p2复核完成", "已经复核")):
        return "review_done"
    return None


@dataclass(frozen=True)
class _WorkerResult:
    generation: int
    session_id: str
    task_kind: str
    flow_id: str | None
    output_directory: str | None
    value: object


class ResearchWorkspace(QSplitter):
    """Single source of UI truth for history, conversation, inputs, plan, and trace."""

    busy_changed = Signal(bool)
    status_changed = Signal(str)

    LOCAL_FILE_PROMPTS: tuple[tuple[str, str], ...] = (
        ("target", "选择电价数据文件"),
        ("actuals", "选择影响因素数据（实际值，可跳过）"),
        ("forecasts", "选择影响因素数据（预测值，可跳过）"),
    )

    def __init__(
        self,
        *,
        agent: ResearchCoordinator | None = None,
        store: SessionStore | None = None,
        plan_feedback_seconds: int = 30,
        auto_execute_plan: bool = False,
        region_profiles: dict[str, RegionProfile] | None = None,
        region_fetcher: Callable[..., RegionPriceFetch] | None = None,
    ) -> None:
        super().__init__(Qt.Orientation.Horizontal)
        self.setObjectName("researchWorkspace")
        provided_store = store is not None
        self.store = store or SessionStore()
        self.research_output_directory = (
            (self.store.path.parent / "research").resolve() if provided_store else default_research_output_directory()
        )
        self.region_data_directory = self.store.path.parent / "region-data"
        region_notice: str | None = None
        if region_profiles is None:
            try:
                region_profiles = load_region_profiles()
            except RegionSourceError as exc:
                region_profiles = {}
                region_notice = str(exc)
        self.region_profiles = dict(region_profiles)
        self.region_fetcher = region_fetcher or fetch_region_price
        self._owns_agent = agent is None
        self.agent = agent or ResearchCoordinator(checkpoint_path=self.store.path.with_name("research_graph.sqlite3"))
        self.plan_feedback_seconds = max(1, int(plan_feedback_seconds))
        self.auto_execute_plan = bool(auto_execute_plan)
        self.sessions = [
            session
            for session in self.store.load(self._installed_skill_versions())
            if not self._is_blank_session(session)
        ]
        # Surfaced once, in the first session created after startup.
        self._recovery_notices = list(self.store.recovery_notices)
        if region_notice:
            self._recovery_notices.append(region_notice)
        self.current_session_id: str | None = None
        self._thread: QThread | None = None
        self._worker: FunctionWorker | None = None
        self._task_kind: str | None = None
        self._task_generation = 0
        self._active_task_generation: int | None = None
        self._task_outcome: str | None = None
        self._task_success_handler: Callable[[Any], None] | None = None
        self._task_failure_handler: Callable[[str], None] | None = None
        self._pending_full_flow_resume = False
        self._thinking_widget: ThinkingMessageWidget | None = None
        self._thinking_message_id: str | None = None
        self._thinking_placeholder = False
        self._thinking_function: str | None = None
        self._last_progress_message = ""
        self._task_previous_status = "idle"
        self._region_failure_summary: DataSummary | None = None
        self._active_interrupt_id: str | None = None
        self._active_state_revision: int | None = None
        self._active_interrupt_kind: str | None = None
        self._pending_execute_plan: EDAPlan | None = None
        self._pending_forecast_request: str | None = None
        self._pre_trace_sizes: list[int] | None = None
        self._p2_reviewer_name = ""
        self._plan_feedback_timer = QTimer(self)
        self._plan_feedback_timer.setInterval(250)
        self._plan_feedback_timer.timeout.connect(self._plan_feedback_tick)

        self.history = HistoryPane()
        self.conversation = ConversationPane()
        self.context = ContextPane()
        self.history.setMinimumWidth(200)
        self.history.setMaximumWidth(320)
        self.conversation.setMinimumWidth(560)
        self.context.setMinimumWidth(300)
        self.context.setMaximumWidth(440)
        self.addWidget(self.history)
        self.addWidget(self.conversation)
        self.addWidget(self.context)
        self.setSizes([236, 760, 336])
        self.setCollapsible(0, True)
        self.setCollapsible(1, False)
        self.setCollapsible(2, True)
        self.setStretchFactor(0, 0)
        self.setStretchFactor(1, 1)
        self.setStretchFactor(2, 0)

        self.history.new_requested.connect(self.create_session)
        self.history.session_selected.connect(self.select_session)
        self.history.rename_requested.connect(self.rename_session)
        self.history.delete_requested.connect(self.delete_session)
        self.conversation.send_requested.connect(self.submit_question)
        self.conversation.cancel_requested.connect(self.cancel_current_task)
        self.conversation.plan_run_requested.connect(self.run_plan)
        self.conversation.plan_reject_requested.connect(self.reject_plan)
        self.conversation.end_research_requested.connect(self.end_current_research)
        self.conversation.draft_changed.connect(self._composer_draft_changed)
        self.conversation.plan_revise_requested.connect(self.request_plan_revision)
        self.conversation.data_details_requested.connect(self.show_data_details)
        self.conversation.p2_review_open_requested.connect(self.open_p2_review)
        self.conversation.p2_revalidate_requested.connect(self.revalidate_p2)
        self.context.refetch_requested.connect(self.refetch_dataset)
        self.context.reselect_requested.connect(self.reselect_dataset)
        self.context.details_requested.connect(self.show_data_details)
        self.context.retry_requested.connect(self.retry_dataset)
        self.context.local_file_requested.connect(self.choose_local_files)
        self.context.region_selected.connect(self.select_region)
        self.context.trace_maximized.connect(self._set_trace_maximized)

        session = self._new_session()
        self.sessions.insert(0, session)
        self.current_session_id = session.session_id
        self._persist_and_render()

    def _installed_skill_versions(self) -> dict[str, str]:
        """Report the Skill versions this build ships, so stale plans retire when a session opens."""

        registry = getattr(self.agent, "skills", None)
        if registry is None:
            return {}
        try:
            return {str(item["name"]): str(item["version"]) for item in registry.metadata()}
        except Exception:  # noqa: BLE001 - a broken registry must not block opening sessions
            return {}

    @property
    def is_busy(self) -> bool:
        return self._thread is not None

    @property
    def current_session(self) -> ResearchSession:
        session = next((item for item in self.sessions if item.session_id == self.current_session_id), None)
        if session is None:
            raise RuntimeError("current research session is missing")
        return session

    def _new_session(self) -> ResearchSession:
        session = ResearchSession()
        session.messages.append(
            SessionMessage(
                role="assistant",
                kind="text",
                content=(
                    "你好，我可以帮你研究电价和它背后的影响因素。\n"
                    "直接说你想弄清什么就行，例如“负荷对实时电价的影响有多大”“峰谷价差在夏天有什么不同”。\n"
                    "要真正跑分析，请在右侧选择地区并取数；只想讨论方法时不取数据也可以。"
                ),
            )
        )
        session.trace.append(
            TraceEvent(category="session", name="新建研究会话", status="completed", summary="等待研究问题输入")
        )
        notices, self._recovery_notices = getattr(self, "_recovery_notices", []), []
        for notice in notices:
            session.messages.append(SessionMessage(role="assistant", kind="notice", content=notice))
            session.trace.append(TraceEvent(category="session", name="历史会话恢复", status="warning", summary=notice))
        skill_errors = list(getattr(self.agent, "skill_load_errors", []))
        if skill_errors:
            summary = "；".join(skill_errors)
            session.messages.append(
                SessionMessage(
                    role="assistant",
                    kind="error",
                    content=f"有外部研究方法包没能加载，对应能力暂时不可用：{summary}",
                )
            )
            session.trace.append(
                TraceEvent(
                    category="error",
                    name="外部研究方法包加载失败",
                    status="warning",
                    summary=summary,
                )
            )
        return session

    def refresh_agent(self) -> None:
        """Reload the planner after the user changes model configuration."""

        if self.is_busy:
            return
        if self._owns_agent:
            self.agent.close()
        self.agent = ResearchCoordinator(checkpoint_path=self.store.path.with_name("research_graph.sqlite3"))
        self._owns_agent = True
        self._append_message(
            SessionMessage(
                role="system",
                kind="notice",
                content="模型设置已更新，接下来的研究会使用新的设置。",
            )
        )
        self._add_trace("session", "更新模型配置", "completed", "已重新加载模型连接参数")
        self._persist_and_render(keep_timeline=True)

    @staticmethod
    def _is_blank_session(session: ResearchSession) -> bool:
        has_user_message = any(message.role == "user" for message in session.messages)
        has_file = any(item.path for item in session.inputs.values())
        return (
            session.title in {"新研究", "新会话"}
            and not has_user_message
            and not has_file
            and session.current_plan is None
            and session.run_id is None
        )

    def create_session(self) -> None:
        if self.is_busy:
            return
        self._cancel_plan_feedback_window()
        session = self._new_session()
        self.sessions.insert(0, session)
        self.current_session_id = session.session_id
        self._persist_and_render()

    def select_session(self, session_id: str) -> None:
        if self.is_busy or session_id == self.current_session_id:
            return
        selected_session = next((item for item in self.sessions if item.session_id == session_id), None)
        if selected_session is not None:
            # Opening a conversation may reconcile its persisted Graph/full-flow
            # projection.  Those helpers save through ``_persist_and_render`` and
            # therefore touch the session, but viewing an existing conversation is
            # not new activity and must not promote it in the history list.
            previous_updated_at = selected_session.updated_at
            self._cancel_plan_feedback_window()
            self.current_session_id = session_id
            self._render_current()
            resolved_flow = self._full_flow_for_session()
            if resolved_flow is not None and not selected_session.read_only:
                flow, state = resolved_flow
                self.update_full_flow_state(
                    state,
                    state_path=flow.store.path,
                    output_directory=flow.output_directory,
                )
            if resolved_flow is None and self.agent.has_thread(session_id) and not selected_session.read_only:
                self._loop_completed(self.agent.get_snapshot(session_id))
            elif self.current_session.status == "awaiting_plan_approval" and self.current_session.current_plan:
                self.current_session.plan_feedback_deadline = None
                self.current_session.plan_feedback_remaining_seconds = 0
                if self.conversation.current_plan_widget is not None:
                    self.conversation.current_plan_widget.set_feedback_paused("需要你重新确认")
            if selected_session.updated_at != previous_updated_at:
                selected_session.updated_at = previous_updated_at
                self.store.save(self.sessions)
                self.history.set_sessions(self.sessions, self.current_session_id)

    def rename_session(self, session_id: str, title: str) -> None:
        if self.is_busy:
            return
        session = next((item for item in self.sessions if item.session_id == session_id), None)
        if session is None:
            return
        session.title = title
        session.touch()
        self._persist_and_render()

    def delete_session(self, session_id: str) -> None:
        if self.is_busy:
            return
        session = next((item for item in self.sessions if item.session_id == session_id), None)
        if session is None:
            return
        answer = QMessageBox.question(
            self,
            "删除研究记录",
            f"确定删除“{session.title}”吗？已经生成的报告文件不会被删除。",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        if self.current_session_id == session_id:
            self._cancel_plan_feedback_window()
        self.sessions = [item for item in self.sessions if item.session_id != session_id]
        if self.agent.has_thread(session_id):
            self.agent.delete_thread(session_id)
        if not self.sessions:
            self.sessions.append(self._new_session())
        if self.current_session_id == session_id:
            self.current_session_id = self.sessions[0].session_id
        self._persist_and_render()

    def _assign_input_file(self, role: str, path: str) -> bool:
        """Point one role at a file; return False when it already pointed there."""

        resolved = validate_input_path(role, path)  # type: ignore[arg-type]
        item = self.current_session.inputs[role]  # type: ignore[index]
        if item.path and Path(item.path).resolve() == resolved:
            return False
        item.path = str(resolved)
        item.status = "selected"
        item.detail = "待检查"
        item.variables = []
        self.current_session.source_kind = "file"
        return True

    def choose_local_files(self) -> None:
        """Fall back to files the analyst already has when no data can be fetched."""

        if self.is_busy:
            return
        chosen: list[str] = []
        for role, caption in self.LOCAL_FILE_PROMPTS:
            selected, _filter = QFileDialog.getOpenFileName(self, caption, "", FILE_FILTERS[role])
            if not selected:
                if role == "target":
                    return
                continue
            try:
                if self._assign_input_file(role, selected):
                    chosen.append(Path(selected).name)
            except (OSError, ValueError) as exc:
                QMessageBox.warning(self, "这个文件暂时用不了", str(exc))
                return
        if not chosen:
            return
        self.current_session.analysis_start_time = None
        self.current_session.analysis_end_time = None
        self._cancel_plan_feedback_window()
        self._invalidate_plan_for_data_change(self._current_dataset_fingerprint())
        self._add_trace("input", "改用本地文件", "completed", "、".join(chosen))
        self._set_data_state("empty", None)
        self._persist_and_render()

    def refetch_dataset(self) -> None:
        """Take the same data again; an unchanged dataset must not void a live plan."""

        if self.current_session.read_only:
            return
        if self.is_busy:
            return
        session = self.current_session
        if session.source_kind == "database":
            self.select_region(session.region_id, force_refresh=True)
            return
        fingerprint = self._current_dataset_fingerprint()
        changed = self._invalidate_plan_for_data_change(fingerprint)
        if changed:
            self._add_trace("input", "重新取数", "completed", "数据有更新，之前的方案已作废")
            self._set_data_state("empty", None)
        else:
            # Identical data is still data from the moment it was first read, so the
            # panel keeps saying so. The trace is where「我刚看过」belongs.
            self._add_trace("input", "重新取数", "completed", "数据没有变化")
            self._set_data_state("ready", session.data_summary)
        self._persist_and_render(keep_timeline=True)

    def reselect_dataset(self) -> None:
        """Throw the current dataset away and let the analyst point at another one."""

        if self.current_session.read_only:
            return
        if self.is_busy:
            return
        self.context.data_panel.open_region_menu()

    def retry_dataset(self) -> None:
        """Try the data source again without touching what was asked for."""

        if self.current_session.read_only:
            return
        if self.is_busy:
            return
        session = self.current_session
        if session.source_kind == "database":
            self.select_region(session.region_id, force_refresh=True)
            return
        if session.can_analyze:
            self._add_trace("input", "重新连接数据", "completed", "已改用本地数据")
            self._set_data_state("ready" if session.data_summary else "empty", session.data_summary)
        else:
            self._add_trace("input", "重新连接数据", "warning", "仍然连不上数据服务器")
            self._set_data_state("unavailable", session.data_summary)
        self._persist_and_render(keep_timeline=True)

    @staticmethod
    def _has_local_region_data(session: ResearchSession) -> bool:
        """Reuse only a completed fetch whose local input files still exist."""

        if (
            session.data_state != "ready"
            or session.data_summary is None
            or not session.database_fetch_details
            or not session.inputs["target"].path
        ):
            return False
        try:
            return all(
                Path(item.path).is_file() and Path(item.path).stat().st_size > 0
                for item in session.inputs.values()
                if item.path
            )
        except OSError:
            return False

    def select_region(self, region_id: str, *, force_refresh: bool = False) -> None:
        """Reuse this conversation's regional data unless a refresh is requested."""

        if self.current_session.read_only:
            return
        if self.is_busy:
            return
        profile = self.region_profiles.get(region_id)
        if profile is None:
            QMessageBox.warning(self, "这个地区暂时不可用", "地区配置已经变化，请重新打开应用。")
            return
        session = self.current_session
        changed_source = session.source_kind != "database" or session.region_id != profile.region_id
        if not changed_source and not force_refresh and self._has_local_region_data(session):
            self._add_trace(
                "input",
                f"复用{profile.label}数据",
                "completed",
                "继续使用本次会话已保存的数据；需要更新时点击“取最新的”。",
            )
            self._persist_and_render(keep_timeline=True)
            return
        previous_summary = session.data_summary
        failure_summary = None if changed_source else previous_summary
        self._region_failure_summary = failure_summary
        self._task_previous_status = session.status
        if changed_source:
            self._invalidate_plan_for_data_change(None)
            session.inputs = default_input_files()
            session.database_fetch_details = {}
            session.analysis_start_time = None
            session.analysis_end_time = None
        session.source_kind = "database"
        session.region_id = profile.region_id
        session.region_label = profile.label
        session.region_market = profile.market
        session.region_timezone = profile.timezone
        session.status = "inspecting_data"
        actual_candidates = [item.name for item in profile.actual_series]
        actual_candidates.extend(profile.weather_columns)
        forecast_candidates = [item.name for item in profile.forecast_series]
        forecast_candidates.extend(
            f"forecast_{name}" if name in actual_candidates else name for name in profile.weather_columns
        )
        table_names = {profile.target_name: profile.price_table}
        table_names.update({item.name: item.table for item in profile.actual_series})
        table_names.update({item.name: item.table for item in profile.forecast_series})
        partial = build_partial_summary(
            market=profile.market,
            target_name=profile.target_name,
            exogenous_names=[*actual_candidates, *forecast_candidates],
            frequency=profile.frequency,
            table_names=table_names,
            variable_kinds={
                **{name: "actual" for name in actual_candidates},
                **{name: "forecast" for name in forecast_candidates},
            },
        )
        self._set_data_state("exploring", partial)
        self._add_trace("input", f"选择{profile.label}数据", "running", "正在取得实际电价")
        self._start_worker(
            kind="data_fetch",
            operation=lambda progress: self.region_fetcher(
                profile,
                output_directory=self.region_data_directory / session.session_id,
                progress=progress,
            ),
            success_handler=self._region_fetch_completed,
            failure_handler=self._region_fetch_failed,
        )

    def _region_fetch_completed(self, result: RegionPriceFetch) -> None:
        self._region_failure_summary = None
        session = self.current_session
        target = session.inputs["target"]
        target.path = str(result.path)
        target.status = "ready"
        target.detail = f"{result.row_count:,} 个时间点"
        target.variables = []
        for role, path, names in (
            ("actuals", result.actuals_path, result.actual_variable_names),
            ("forecasts", result.forecasts_path, result.forecast_variable_names),
        ):
            item = session.inputs[role]
            item.path = str(path) if path is not None else ""
            item.status = "ready" if path is not None else "empty"
            item.detail = f"{len(names)} 个变量" if names else "尚未选择"
            item.variables = []
        session.source_kind = "database"
        session.database_fetch_details = result.safe_details()
        fingerprint = self._current_dataset_fingerprint()
        changed = self._invalidate_plan_for_data_change(fingerprint)
        session.status = "idle" if changed else self._task_previous_status
        self._set_data_state("ready", result.summary())
        self._close_latest_running_trace("completed")
        self._add_trace(
            "input",
            f"{result.profile.label}电价已就绪",
            "completed",
            f"{result.row_count:,} 个电价时间点，{len(result.actual_variable_names) + len(result.forecast_variable_names)} 个影响因素",
        )
        self._persist_and_render(keep_timeline=True)

    def _region_fetch_failed(self, detail: str) -> None:
        session = self.current_session
        previous_summary = self._region_failure_summary
        self._region_failure_summary = None
        session.status = self._task_previous_status if self._task_previous_status != "inspecting_data" else "idle"
        headline = detail.strip().splitlines()[-1] if detail.strip() else "取数失败"
        self._close_latest_running_trace("failed")
        self._set_data_state("unavailable", previous_summary)
        self._append_message(
            SessionMessage(
                role="assistant",
                kind="error",
                content=f"{session.region_label}的数据现在取不到：{headline}",
            )
        )
        self._add_trace("error", f"{session.region_label}取数未完成", "failed", headline)
        self._persist_and_render(keep_timeline=True)

    def show_data_details(self) -> None:
        """Open the one screen where database wording is allowed."""

        DataDetailsDialog(self._dataset_details(), self).exec()

    def request_plan_revision(self) -> None:
        """Send the analyst to the composer instead of opening an editor on the plan."""

        if self.current_session.read_only:
            return
        if self.conversation.current_plan_widget is not None:
            self.conversation.current_plan_widget.set_feedback_paused()
        self._cancel_plan_feedback_window()
        self.conversation.input.setFocus()

    def _dataset_details(self) -> dict[str, Any]:
        session = self.current_session
        details: dict[str, Any] = {
            "source_kind": session.source_kind,
            "region": session.region_label,
            "dataset_fingerprint": session.dataset_fingerprint or "(none)",
            "plan_data_fingerprint": (session.current_plan or {}).get("data_fingerprint") or "(none)",
        }
        files = [
            f"{role}: {session.inputs[role].path}"
            for role in ("target", "actuals", "forecasts")
            if session.inputs[role].path
        ]
        if files:
            details["input_files"] = files
        if session.quality_report:
            alignment = dict(session.quality_report.get("alignment", {}))
            if alignment:
                details["alignment"] = alignment
            details["series"] = sorted(session.quality_report.get("series", {}))
        if session.data_profile:
            details["data_profile"] = session.data_profile
        if session.database_fetch_details:
            details["database_fetch"] = session.database_fetch_details
        return details

    def _current_dataset_fingerprint(self) -> str | None:
        """Hash what would be read right now, using the planner's own definition."""

        session = self.current_session
        if not session.can_analyze:
            return None
        try:
            config = build_runtime_study(session, output_directory=self.research_output_directory)
            return study_fingerprint(config, input_file_manifest(config))
        except (OSError, ValueError):
            return None

    def _invalidate_plan_for_data_change(self, fingerprint: str | None) -> bool:
        """Void the approved plan only when the data behind it actually moved.

        Taking the data again and getting the same thing back is an ordinary
        action, and it must not cost the analyst a plan they already approved.
        """

        session = self.current_session
        current_plan = session.current_plan or {}
        anchor = (
            current_plan.get("source_data_fingerprint")
            if current_plan.get("plan_kind") == "forecast"
            else current_plan.get("data_fingerprint")
        ) or session.dataset_fingerprint
        session.dataset_fingerprint = fingerprint
        if fingerprint is not None and anchor is not None and fingerprint == anchor:
            return False
        self._invalidate_plan_for_input_change()
        return True

    def _set_data_state(self, state: DataPanelState, summary: DataSummary | None) -> None:
        session = self.current_session
        session.data_state = state
        session.data_summary = summary
        session.touch()
        self.context.data_panel.set_state(state, summary)

    def _summary_from_proposal(self, proposal: ResearchProposal) -> DataSummary:
        """Take the words written when the data was frozen, or derive them if it was not.

        The frozen wording is preferred because it names the moment the data was
        actually read; deriving it here would restamp it as「now」every time the
        analyst asks another question about the same data.
        """

        if proposal.data_summary is not None:
            return proposal.data_summary
        profile = proposal.data_profile
        alignment = proposal.quality_report.alignment
        missing = max(0, alignment.expected_rows - alignment.complete_case_rows)
        return build_summary(
            market=profile.market,
            target_name=profile.target_name,
            exogenous_names=list(profile.exogenous_names),
            start_time=profile.start_time,
            end_time=profile.end_time,
            frequency=profile.frequency,
            missing_points=missing,
        )

    def _invalidate_plan_for_input_change(self) -> None:
        self._cancel_plan_feedback_window()
        session = self.current_session
        session.pending_forecast_question = None
        for run in session.runs:
            run.memory_status = "stale"
        if self.agent.has_thread(session.session_id):
            self.agent.delete_thread(session.session_id)
        if session.current_plan is not None:
            session.plan_stale = True
            session.current_plan = None
            session.status = "idle"
            self._set_plan_message_state("stale")
            session.messages.append(
                SessionMessage(
                    role="system",
                    kind="notice",
                    content=DATA_CHANGED_NOTICE,
                )
            )
        session.run_id = None
        session.artifact_directory = None
        session.report_path = None
        session.data_profile = None
        session.quality_report = None
        session.latest_eda_summary = None
        session.latest_evaluation = None
        session.touch()

    def submit_question(self, question: str) -> None:
        if self.current_session.read_only:
            return
        if self.is_busy:
            return
        session = self.current_session
        analysis_then_forecast = requests_analysis_before_forecast(question)
        current_is_forecast = (session.current_plan or {}).get("plan_kind") == "forecast"
        if current_is_forecast and session.status == "awaiting_plan_approval" and _is_forecast_text_approval(question):
            self._append_message(SessionMessage(role="user", kind="text", content=question, turn_id=uuid4().hex))
            self.run_plan(ForecastPlan.model_validate(session.current_plan))
            return
        if any(run.run_kind == "forecast" for run in session.runs) and _is_forecast_explanation(question):
            self._explain_forecast_result(question)
            return
        if self._handle_full_flow_input(question):
            return
        if _is_news_analysis_request(question):
            self._submit_news_analysis_request(
                question,
                prepare_forecast=_requests_news_forecast(question),
            )
            return
        if _is_forecast_request(question, has_price_context=bool(session.inputs["target"].path)):
            self._submit_forecast_request(question)
            return
        if (session.current_plan or {}).get("plan_kind") == "forecast":
            self._append_message(SessionMessage(role="user", kind="text", content=question, turn_id=uuid4().hex))
            self._append_message(
                SessionMessage(
                    role="assistant",
                    kind="text",
                    content="当前是固定范围的山东次日预测方案。你可以确认执行、拒绝方案，或询问方案内容。",
                )
            )
            self._persist_and_render(keep_timeline=True)
            return
        try:
            time_request = parse_chat_time_range(question)
        except ValueError as exc:
            session.derive_title(question)
            self._append_message(SessionMessage(role="user", kind="text", content=question))
            self._append_message(SessionMessage(role="assistant", kind="error", content=str(exc)))
            self._add_trace("input", "时间范围没有更新", "warning", str(exc))
            self._persist_and_render(keep_timeline=True)
            return
        if time_request is not None:
            new_start = time_request.start_time.isoformat() if time_request.start_time else None
            new_end = time_request.end_time.isoformat() if time_request.end_time else None
            changed_window = session.analysis_start_time != new_start or session.analysis_end_time != new_end
            if changed_window:
                self._invalidate_plan_for_input_change()
                session.analysis_start_time = new_start
                session.analysis_end_time = new_end
        self._cancel_plan_feedback_window()
        session.derive_title(question)
        user_message = SessionMessage(role="user", kind="text", content=question, turn_id=uuid4().hex)
        self._append_message(user_message)
        question_label = " ".join(question.split())
        if len(question_label) > 52:
            question_label = f"{question_label[:51]}…"
        self._add_trace("user", f"提交研究问题：{question_label}", "completed", question)
        if time_request is not None and changed_window:
            if time_request.action == "clear":
                self._add_trace("input", "恢复全部时间范围", "completed", "后续分析使用全部可用数据")
            else:
                self._add_trace(
                    "input",
                    "按聊天指定时间分析",
                    "completed",
                    f"{new_start} — {new_end}",
                )
        study_config = None
        if session.can_analyze:
            try:
                study_config = build_runtime_study(
                    session,
                    output_directory=self.research_output_directory,
                )
            except (OSError, ValueError) as exc:
                self._fail_before_task(str(exc))
                return
        self._task_previous_status = session.status
        if analysis_then_forecast:
            session.pending_forecast_question = question
        session.status = "understanding"
        if session.can_analyze and session.data_state != "ready":
            # Only first-time inspection changes the data card. A question about
            # an existing snapshot keeps its settled source, range and fetch time.
            self._set_data_state("exploring", session.data_summary or _known_so_far(study_config))
        elif session.data_state != "ready":
            # Nothing to read yet: say so plainly and offer the way out of it.
            self._set_data_state("unavailable", session.data_summary)
        self._start_thinking("read", "解析研究问题", "识别本轮的处理方式")
        conversation = self._agent_conversation(session, exclude_message_id=user_message.message_id)
        imported_state = self._legacy_graph_import(session) if not self.agent.has_thread(session.session_id) else None
        self._start_worker(
            kind="dialogue",
            operation=lambda progress: self.agent.submit_user_message(
                session_id=session.session_id,
                message=question,
                message_id=user_message.message_id,
                turn_id=user_message.turn_id,
                study_config=study_config,
                conversation=conversation,
                approval_timeout_seconds=self.plan_feedback_seconds,
                automatic_approval_enabled=self.auto_execute_plan,
                imported_state=imported_state,
                progress=progress,
            ),
            success_handler=self._loop_completed,
        )
        self.conversation.scroll_to_bottom()

    def _submit_news_analysis_request(
        self,
        question: str,
        *,
        prepare_forecast: bool,
        record_user_message: bool = True,
    ) -> None:
        """Reuse completed P1 evidence and advance the owned P2/P3 state machine."""

        session = self.current_session
        self._cancel_plan_feedback_window()
        if record_user_message:
            self._append_message(SessionMessage(role="user", kind="text", content=question, turn_id=uuid4().hex))
        if session.region_id != "shandong" or session.source_kind != "database":
            self._fail_before_task("新闻分析与次日预测流程当前只支持已取数的山东数据库会话。")
            return
        profile = self.region_profiles.get("shandong")
        if profile is None:
            self._fail_before_task("山东数据源配置当前不可用，无法准备新闻分析与预测。")
            return
        p1_record = next((run for run in reversed(session.runs) if run.run_kind == "eda"), None)
        if p1_record is None or not session.latest_eda_summary:
            self._fail_before_task("当前会话还没有可复用的P1分析结果；请先完成一次电价分析。")
            return
        if not Path(p1_record.report_path).is_file() or not Path(p1_record.artifact_directory).is_dir():
            self._fail_before_task("当前P1分析产物已经丢失；请重新执行一次电价分析。")
            return
        settings = get_settings()
        news_path = default_p2_news_path(settings.p2_news_path)
        if news_path is None:
            self._fail_before_task("没有找到可审计的P2新闻资料。请通过 VPP_P2_NEWS_PATH 配置冻结的新闻JSONL文件。")
            return
        if not session.inputs["target"].path:
            self._fail_before_task("当前会话的电价数据不可用；请重新取得山东数据。")
            return

        request_id = uuid4().hex[:12]
        flow_root = self.research_output_directory / "full-flow" / request_id
        flow = FullResearchFlow(
            store=FullFlowStore(flow_root / "state.json"),
            output_directory=flow_root / "runs",
        )
        session.full_flow_state_path = str(flow.store.path)
        session.full_flow_output_directory = str(flow.output_directory)
        session.pending_forecast_question = question if prepare_forecast else None
        session.current_plan = None
        session.plan_stale = False
        session.status = "running"
        session.derive_title(question)
        self._active_interrupt_id = None
        self._active_state_revision = None
        self._active_interrupt_kind = None
        if self.agent.has_thread(session.session_id):
            self.agent.delete_thread(session.session_id)
        self._append_message(
            SessionMessage(
                role="assistant",
                kind="text",
                content=(
                    "将复用当前P1结果，分析已冻结且可追溯的山东新闻，生成新闻事件证据与数值特征，"
                    + ("随后准备P3预测确认卡。" if prepare_forecast else "随后完成P1综合分析。")
                ),
            )
        )
        self._add_trace("plan", "进入P1—P2—P3流程", "completed", "复用当前P1结果；P3仍需单独确认")

        p1_run = FlowRunReference(
            run_kind="eda",
            run_id=p1_record.run_id,
            parent_run_id=p1_record.parent_run_id,
            artifact_directory=Path(p1_record.artifact_directory),
            report_path=Path(p1_record.report_path),
            data_fingerprint=p1_record.data_fingerprint,
        )
        state = flow.start(
            p1_run=p1_run,
            eda_summary=dict(session.latest_eda_summary),
            information_cutoff=datetime.now(UTC),
            request_text=question,
            requested_forecast=prepare_forecast,
        )
        state = state.model_copy(update={"p2_news_path": news_path})
        flow.store.save(state)
        self.update_full_flow_state(state, state_path=flow.store.path, output_directory=flow.output_directory)
        self._start_full_flow_stage(flow, state)
        self.conversation.scroll_to_bottom()

    def _full_flow_for_session(self) -> tuple[FullResearchFlow, FullFlowState] | None:
        session = self.current_session
        if not session.full_flow_state_path:
            return None
        store = FullFlowStore(session.full_flow_state_path)
        try:
            state = store.load()
        except (OSError, ValueError):
            return None
        output = session.full_flow_output_directory or str(store.path.parent / "runs")
        return FullResearchFlow(store=store, output_directory=output), state

    def _release_terminal_full_flow(self, state: FullFlowState) -> None:
        """Return a terminal full-flow conversation to ordinary dialogue safely."""

        session = self.current_session
        self._cancel_plan_feedback_window()
        self._set_plan_message_state("stopped")
        for run in session.runs:
            run.memory_status = "stale"
        if self.agent.has_thread(session.session_id):
            self.agent.delete_thread(session.session_id)
        session.current_plan = None
        session.plan_stale = False
        session.run_id = None
        session.artifact_directory = None
        session.report_path = None
        session.data_profile = None
        session.quality_report = None
        session.latest_eda_summary = None
        session.latest_evaluation = None
        session.graph_event_count = 0
        session.graph_event_sequence = 0
        session.full_flow_state_path = None
        session.full_flow_output_directory = None
        session.news_features_path = None
        session.news_run_id = None
        session.news_p1_run_id = None
        session.pending_forecast_question = None
        session.status = "idle"
        self._active_interrupt_id = None
        self._active_state_revision = None
        self._active_interrupt_kind = None
        self._append_message(
            SessionMessage(
                role="system",
                kind="notice",
                content=(
                    "已退出停止的全流程。旧流程、报告和审计记录均已保留；"
                    "当前消息将作为新的普通对话或P1研究处理。"
                ),
                payload={
                    "released_full_flow_id": state.flow_id,
                    "released_phase": state.phase,
                },
            )
        )

    def _handle_full_flow_input(self, question: str) -> bool:
        """Resolve commands against the durable P1/P2/P3 cursor before model routing."""

        resolved = self._full_flow_for_session()
        if resolved is None:
            return False
        flow, state = resolved
        command = _full_flow_command(question)
        terminal_control_commands = {
            "continue",
            "request_p3",
            "retry",
            "stop",
            "review_help",
            "review_done",
        }
        if (
            state.phase == "stopped"
            and command not in terminal_control_commands
            and not _requests_new_full_flow(question)
        ):
            self._release_terminal_full_flow(state)
            return False
        if _requests_new_full_flow(question):
            self._append_message(SessionMessage(role="user", kind="text", content=question, turn_id=uuid4().hex))
            self._append_message(
                SessionMessage(
                    role="assistant",
                    kind="notice",
                    content="当前会话由正在进行或已完成的全流程接管。请点击“新的研究”创建新会话，旧目标不会被隐式替换。",
                )
            )
            self._persist_and_render(keep_timeline=True)
            return True
        guarded_phases = {
            "p1_initial_complete",
            "p2_needs_review",
            "p2_ready",
            "p1_synthesis_complete",
            "awaiting_forecast_approval",
            "failed",
            "forecast_running",
            "forecast_verified",
            "feedback_complete",
            "stopped",
        }
        if state.phase not in guarded_phases:
            return False

        self._append_message(SessionMessage(role="user", kind="text", content=question, turn_id=uuid4().hex))
        lineage_error = self._full_flow_lineage_error(state)
        if lineage_error:
            stopped = flow.stop(state, reason=lineage_error)
            self.update_full_flow_state(
                stopped,
                state_path=flow.store.path,
                output_directory=flow.output_directory,
            )
            return True
        if command == "stop":
            stopped = flow.stop(state)
            self.update_full_flow_state(
                stopped,
                state_path=flow.store.path,
                output_directory=flow.output_directory,
            )
            return True
        if command == "new_goal":
            self._append_message(
                SessionMessage(
                    role="assistant",
                    kind="notice",
                    content=(
                        "当前全流程仍在进行。可先回复“停止”结束它，"
                        "或点击左侧“新的研究”并行开始另一项研究。"
                    ),
                    payload={
                        "full_flow_id": state.flow_id,
                        "phase": state.phase,
                        "allowed_actions": list(state.allowed_actions),
                    },
                )
            )
            self._persist_and_render(keep_timeline=True)
            return True
        if state.phase == "awaiting_forecast_approval":
            self._append_message(
                SessionMessage(
                    role="assistant",
                    kind="notice",
                    content="P3方案已经冻结，请在预测方案卡上单独确认执行，或回复“停止”。",
                    payload={
                        "full_flow_id": state.flow_id,
                        "phase": state.phase,
                        "allowed_actions": list(state.allowed_actions),
                    },
                )
            )
            self._persist_and_render(keep_timeline=True)
            return True
        if state.phase == "p1_synthesis_complete" and not state.requested_forecast:
            forecast_request = _is_forecast_request(
                question,
                has_price_context=bool(self.current_session.inputs["target"].path),
            )
            if command == "request_p3" or forecast_request:
                state = flow.request_p3(state)
                self.update_full_flow_state(
                    state,
                    state_path=flow.store.path,
                    output_directory=flow.output_directory,
                )
                self._resume_full_flow()
                return True
            self._append_message(
                SessionMessage(
                    role="assistant",
                    kind="notice",
                    content=(
                        "当前目标只包含P1综合分析，因此本轮已经完成。"
                        "如需沿用当前新闻特征进入P3，请发送“预测山东明天实时电价”或“继续第三阶段”；"
                        "程序会先生成单独的P3确认卡，不会重跑P1或P2。"
                    ),
                    payload={
                        "full_flow_id": state.flow_id,
                        "phase": state.phase,
                        "allowed_actions": ["request_p3", "new_goal"],
                    },
                )
            )
            self._persist_and_render(keep_timeline=True)
            return True
        if state.phase in {"forecast_verified", "feedback_complete", "stopped"}:
            content = (
                "当前全流程已经停止，不会重复执行已完成阶段。"
                "如需使用右侧当前数据重新开始，请发送“开始P1阶段”，或直接描述新的分析问题；"
                "旧流程和报告会继续保留。"
                if state.phase == "stopped"
                else "当前全流程已经完成，不会重复执行已完成阶段；如需重做，请点击左侧“新的研究”。"
            )
            self._append_message(
                SessionMessage(
                    role="assistant",
                    kind="notice",
                    content=content,
                    payload={
                        "full_flow_id": state.flow_id,
                        "phase": state.phase,
                        "allowed_actions": list(state.allowed_actions),
                    },
                )
            )
            self._persist_and_render(keep_timeline=True)
            return True
        if state.phase == "p2_needs_review":
            try:
                summary = flow.get_p2_review_summary(state.flow_id)
            except (OSError, ValueError) as exc:
                self._append_message(
                    SessionMessage(role="assistant", kind="error", content=f"P2复核状态无法读取：{exc}")
                )
                self._persist_and_render(keep_timeline=True)
                return True
            if command in {"retry", "continue", "review_done"} and summary.can_revalidate:
                self.revalidate_p2(summary.revision)
            else:
                self._show_p2_review_gate(state, summary=summary)
                if command == "review_help":
                    QTimer.singleShot(0, self.open_p2_review)
            return True
        if state.phase == "failed" and command != "retry":
            self._append_message(
                SessionMessage(
                    role="assistant",
                    kind="error",
                    content="当前全流程停在失败状态。回复“重试”会从最后保存的阶段恢复，或回复“停止”。",
                )
            )
            self._persist_and_render(keep_timeline=True)
            return True
        if state.phase == "forecast_running":
            self._append_message(
                SessionMessage(
                    role="assistant",
                    kind="notice",
                    content="预测执行曾被中断，请在已恢复的预测确认卡上重新确认，以继续同一方案。",
                )
            )
            self.update_full_flow_state(state, state_path=flow.store.path, output_directory=flow.output_directory)
            return True
        self._resume_full_flow()
        return True

    @staticmethod
    def _p2_review_payload(summary: P2ReviewSummary) -> dict[str, Any]:
        return {
            "full_flow_id": summary.flow_id,
            "phase": summary.phase,
            "revision": summary.revision,
            "required_pending": summary.required_pending,
            "required_resolved": summary.required_resolved,
            "optional_unreviewed": summary.optional_unreviewed,
            "quality_failures": list(summary.quality_failures),
            "report_path": str(summary.report_path or ""),
        }

    def _show_p2_review_gate(
        self,
        state: FullFlowState,
        *,
        summary: P2ReviewSummary | None = None,
    ) -> None:
        resolved = self._full_flow_for_session()
        if summary is None and resolved is not None:
            try:
                summary = resolved[0].get_p2_review_summary(state.flow_id)
            except (OSError, ValueError):
                summary = None
        if summary is None:
            self._append_message(
                SessionMessage(role="assistant", kind="error", content="P2复核数据库不可读取，当前流程不会继续。")
            )
            self.current_session.status = "awaiting_user"
            self._persist_and_render(keep_timeline=True)
            return
        payload = self._p2_review_payload(summary)
        existing = next(
            (
                message
                for message in reversed(self.current_session.messages)
                if message.kind == "p2_review" and message.payload.get("full_flow_id") == state.flow_id
            ),
            None,
        )
        if existing is None:
            self.current_session.messages.append(
                SessionMessage(
                    role="assistant",
                    kind="p2_review",
                    content="P2 新闻抽取需要复核",
                    payload=payload,
                )
            )
        else:
            existing.payload = payload
            existing.content = "P2 新闻抽取需要复核"
        self.current_session.status = "awaiting_user"
        self._persist_and_render()

    @staticmethod
    def _p2_review_items(review_queue: Path | None) -> list[str]:
        if review_queue is None or not review_queue.is_file():
            return []
        try:
            payload = json.loads(review_queue.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        items: list[str] = []
        for extraction in payload.get("extractions", []):
            if not isinstance(extraction, dict) or extraction.get("review_status") != "unreviewed":
                continue
            document = extraction.get("document") or {}
            result = extraction.get("result") or {}
            title = str(document.get("title") or extraction.get("document_version_id") or "未命名新闻")
            reasons: list[str] = []
            quarantine = result.get("quarantine")
            if isinstance(quarantine, dict):
                reasons.append(str(quarantine.get("message") or quarantine.get("reason_code") or "抽取结果需复核"))
            for candidate in result.get("candidate_quarantines") or []:
                if isinstance(candidate, dict):
                    reasons.append(str(candidate.get("message") or candidate.get("reason_code") or "候选事件需复核"))
            if any(
                isinstance(event, dict)
                and event.get("relevance") == "short_term"
                and not event.get("effective_start_at")
                for event in result.get("events") or []
            ):
                reasons.append("短期事件缺少生效开始时间")
            if reasons:
                summary = "；".join(dict.fromkeys(reasons))
                items.append(f"《{title[:48]}》：{summary[:180]}")
        return items

    def _full_flow_lineage_error(self, state: FullFlowState) -> str | None:
        expected = state.input_fingerprints.get("p1_data")
        current = self.current_session.dataset_fingerprint
        if expected and current and expected != current:
            return "当前数据已经变更，旧全流程已停止；请基于新数据重新完成P1后发起新一轮新闻分析"
        return None

    def open_p2_review(self) -> None:
        """Open the typed review window for the active flow, never an exported JSON queue."""

        if self.is_busy or self.current_session.read_only:
            return
        resolved = self._full_flow_for_session()
        if resolved is None:
            return
        flow, state = resolved
        if state.phase != "p2_needs_review":
            return
        try:
            summary = flow.get_p2_review_summary(state.flow_id)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "无法打开P2复核", str(exc))
            return
        dialog = P2ReviewDialog(
            summary,
            submit=flow.submit_p2_review,
            reviewer=self._p2_reviewer_name,
            parent=self,
        )
        dialog.reviewer_changed.connect(lambda name: setattr(self, "_p2_reviewer_name", name))
        dialog.summary_changed.connect(self._p2_review_summary_changed)
        dialog.exec()

    @Slot(object)
    def _p2_review_summary_changed(self, value: object) -> None:
        summary = P2ReviewSummary.model_validate(value)
        resolved = self._full_flow_for_session()
        if resolved is None:
            return
        _flow, state = resolved
        self._show_p2_review_gate(state, summary=summary)

    @Slot(str)
    def revalidate_p2(self, expected_revision: str) -> None:
        """Revalidate saved decisions in a worker; saving an individual review never calls this."""

        if self.is_busy or self.current_session.read_only:
            return
        resolved = self._full_flow_for_session()
        if resolved is None:
            return
        flow, state = resolved
        if state.phase != "p2_needs_review":
            return
        try:
            summary = flow.get_p2_review_summary(state.flow_id)
        except (OSError, ValueError) as exc:
            self._fail_before_task(f"P2复核状态无法读取：{exc}")
            return
        if summary.revision != expected_revision:
            self._show_p2_review_gate(state, summary=summary)
            QMessageBox.warning(self, "复核状态已更新", "请在刷新后的复核卡上重新点击“校验并继续”。")
            return
        if summary.required_pending:
            self._show_p2_review_gate(state, summary=summary)
            return
        session = self.current_session
        profile = self.region_profiles.get("shandong")
        target_path = session.inputs["target"].path
        if profile is None or not target_path:
            self._fail_before_task("山东数据来源或目标电价数据不可用，无法重新校验P2。")
            return
        settings = get_settings()

        def operation(progress: Callable[[int, str], None]) -> FullFlowState:
            progress(
                30,
                progress_message(
                    ThinkingStep("review", "重新校验P2", "读取已保存复核决定并重建事件、统计与特征", "running"),
                    "P2复核校验",
                ),
            )
            extractor = AdaptiveNewsEventExtractor(
                build_model_gateway(settings),
                market_timezone=session.region_timezone or profile.timezone,
                extraction_passes=3,
            )
            return flow.revalidate_p2(
                state.flow_id,
                expected_revision,
                target_path=target_path,
                extractor=extractor,
            )

        session.status = "running"
        self._start_thinking("review", "重新校验P2", "应用已保存复核决定；成功后继续P1综合分析")
        self._start_worker(
            kind="full_flow_p2_revalidate",
            operation=operation,
            success_handler=self._full_flow_stage_completed,
            failure_handler=self._p2_revalidation_failed,
        )

    @Slot(str)
    def _p2_revalidation_failed(self, detail: str) -> None:
        headline = detail.strip().splitlines()[-1] if detail.strip() else "P2复核校验失败"
        self._close_latest_running_trace("failed")
        self._set_active_tool_status("failed", headline)
        self._append_message(
            SessionMessage(
                role="assistant",
                kind="error",
                content=f"P2仍停在复核阶段：{headline}",
            )
        )
        resolved = self._full_flow_for_session()
        if resolved is not None:
            flow, state = resolved
            self.update_full_flow_state(state, state_path=flow.store.path, output_directory=flow.output_directory)

    def _resume_full_flow(self) -> None:
        resolved = self._full_flow_for_session()
        if resolved is None:
            self._fail_before_task("全流程状态文件不可用，无法继续。")
            return
        flow, state = resolved
        lineage_error = self._full_flow_lineage_error(state)
        if lineage_error:
            stopped = flow.stop(state, reason=lineage_error)
            self.update_full_flow_state(
                stopped,
                state_path=flow.store.path,
                output_directory=flow.output_directory,
            )
            return
        if state.phase == "failed":
            if state.resume_from_phase is None:
                self._fail_before_task("失败状态没有可恢复阶段；请重新发起新闻分析。")
                return
            state = state.model_copy(
                update={
                    "phase": state.resume_from_phase,
                    "error": None,
                    "stop_reason": None,
                    "resume_from_phase": None,
                    "failed_stage": None,
                }
            )
            flow.store.save(state)
        if state.phase == "p2_needs_review":
            self._show_p2_review_gate(state)
            return
        if state.phase in {"p1_initial_complete", "p2_ready", "p1_synthesis_complete"}:
            self._start_full_flow_stage(flow, state)

    def _start_full_flow_stage(self, flow: FullResearchFlow, state: FullFlowState) -> None:
        """Run exactly one durable stage; the GUI decides the next stage after completion."""

        session = self.current_session
        profile = self.region_profiles.get("shandong")
        target_path = session.inputs["target"].path
        if profile is None or not target_path:
            self._fail_before_task("山东数据来源或目标电价数据不可用，无法恢复全流程。")
            return
        settings = get_settings()
        resume_phase = state.phase

        if state.phase == "p1_initial_complete":
            stage = "p2"
            news_path = state.p2_news_path or default_p2_news_path(settings.p2_news_path)
            if news_path is None or not Path(news_path).is_file():
                self._fail_before_task("没有找到可审计的P2新闻资料，无法继续同一流程。")
                return
            title, detail = "分析山东新闻", "抽取事件并核对来源、时间与价格窗口"

            def operation(progress: Callable[[int, str], None]) -> FullFlowState:
                progress(20, progress_message(ThinkingStep("compute", title, detail, "running"), "启动P2新闻分析"))
                pass_progress = 0

                def report_pass(document: Any, pass_number: int, pass_count: int) -> None:
                    nonlocal pass_progress
                    pass_progress += 1
                    progress(
                        min(50, 20 + pass_progress),
                        progress_message(
                            ThinkingStep(
                                "compute",
                                "逐篇抽取新闻事件",
                                f"《{document.title[:36]}》第 {pass_number}/{pass_count} 趟",
                                "running",
                            ),
                            "P2新闻抽取",
                        ),
                    )

                extractor = AdaptiveNewsEventExtractor(
                    build_model_gateway(settings),
                    market_timezone=session.region_timezone or profile.timezone,
                    extraction_passes=3,
                    progress=report_pass,
                )
                return flow.run_p2(
                    state=state,
                    news_path=news_path,
                    target_path=target_path,
                    extractor=extractor,
                )

        elif state.phase == "p2_ready":
            stage = "p1_synthesis"
            title, detail = "综合新闻特征", "检查覆盖率、变化次数和关系稳定性"

            def operation(progress: Callable[[int, str], None]) -> FullFlowState:
                progress(55, progress_message(ThinkingStep("review", title, detail, "running"), "启动P1综合分析"))
                return flow.synthesize_p1(state=state)

        elif state.phase == "p1_synthesis_complete" and state.requested_forecast:
            stage = "p3_prepare"
            title, detail = "冻结P3输入", "准备三折回测与次日96点预测数据"

            def operation(progress: Callable[[int, str], None]) -> FullFlowState:
                progress(75, progress_message(ThinkingStep("compute", title, detail, "running"), "启动P3方案准备"))
                return flow.prepare_p3(
                    state=state,
                    question="预测山东明天实时电价",
                    profile=profile,
                    target_path=target_path,
                    actuals_path=session.inputs["actuals"].path or None,
                    forecasts_path=session.inputs["forecasts"].path or None,
                    output_directory=self.research_output_directory / "forecasting",
                    source_data_fingerprint=session.dataset_fingerprint,
                    snapshot_fetcher=self.region_fetcher,
                )

        else:
            self.update_full_flow_state(state, state_path=flow.store.path, output_directory=flow.output_directory)
            return

        def guarded(progress: Callable[[int, str], None]) -> FullFlowState:
            try:
                return operation(progress)
            except Exception as exc:
                try:
                    latest = flow.store.load()
                except (OSError, ValueError):
                    latest = state
                failed = latest.model_copy(
                    update={
                        "phase": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "stop_reason": f"{title}失败；已保留此前完成的阶段产物",
                        "resume_from_phase": resume_phase,
                        "failed_stage": stage,
                    }
                )
                flow.store.save(failed)
                raise

        session.status = "running"
        self._start_thinking("compute", title, detail)
        self._start_worker(
            kind=f"full_flow_{stage}",
            operation=guarded,
            success_handler=self._full_flow_stage_completed,
            failure_handler=self._full_flow_analysis_failed,
        )

    @Slot(object)
    def _full_flow_stage_completed(self, value: object) -> None:
        state = FullFlowState.model_validate(value)
        resolved = self._full_flow_for_session()
        if resolved is None:
            self._task_failed("全流程阶段完成，但状态文件无法重新读取。")
            return
        flow, _ = resolved
        self._complete_active_tool("全流程阶段已完成并保存")
        self.update_full_flow_state(state, state_path=flow.store.path, output_directory=flow.output_directory)
        self._pending_full_flow_resume = state.phase == "p2_ready" or (
            state.phase == "p1_synthesis_complete" and state.requested_forecast
        )

    @Slot(str)
    def _full_flow_analysis_failed(self, detail: str) -> None:
        resolved = self._full_flow_for_session()
        headline = detail.strip().splitlines()[-1] if detail.strip() else "全流程阶段失败"
        self._close_latest_running_trace("failed")
        self._set_active_tool_status("failed", headline)
        self._append_message(
            SessionMessage(
                role="assistant",
                kind="error",
                content=f"当前阶段没有完成：{headline}\n已完成的阶段和产物仍保留；可回复“重试”或“停止”。",
            )
        )
        if resolved is not None:
            flow, state = resolved
            self.update_full_flow_state(state, state_path=flow.store.path, output_directory=flow.output_directory)
        else:
            self.current_session.status = "failed"
            self._persist_and_render(keep_timeline=True)

    def _submit_forecast_request(self, question: str, *, record_user_message: bool = True) -> None:
        """Prepare the fixed P3 plan without allowing chat text to alter its parameters."""

        session = self.current_session
        self._cancel_plan_feedback_window()
        if session.current_plan is not None:
            self._set_plan_message_state("stale")
            session.current_plan = None
        if self.agent.has_thread(session.session_id):
            self.agent.delete_thread(session.session_id)
        session.derive_title(question)
        if record_user_message:
            self._append_message(SessionMessage(role="user", kind="text", content=question, turn_id=uuid4().hex))
        self._add_trace("user", "提交山东实时电价预测", "completed", question)
        if session.source_kind != "database":
            self._fail_before_task("P3最小预测不读取本地任意文件；请先在右侧选择山东并完成取数。")
            return
        if not session.region_id:
            self._fail_before_task("还没有选择地区。请先在右侧选择山东并完成取数。")
            return
        if session.region_id != "shandong":
            self._fail_before_task("P3最小预测当前只支持山东省级实时电价；四川暂不进入预测分支。")
            return
        if session.data_state != "ready" or not session.inputs["target"].path:
            self._fail_before_task("山东数据还没有准备好。请先完成取数，再发起次日预测。")
            return
        profile = self.region_profiles.get("shandong")
        if profile is None:
            self._fail_before_task("当前安装中缺少山东数据配置，无法固定预测输入。")
            return
        session.status = "inspecting_data"
        self._start_thinking("setup", "准备预测输入", "选择3个历史锚点并固定4份只读数据")
        self._start_worker(
            kind="forecast_prepare",
            operation=lambda progress: prepare_forecast_plan(
                question=question,
                profile=profile,
                target_path=session.inputs["target"].path,
                actuals_path=session.inputs["actuals"].path or None,
                forecasts_path=session.inputs["forecasts"].path or None,
                output_directory=self.research_output_directory / "forecasting",
                source_data_fingerprint=session.dataset_fingerprint,
                progress=progress,
                snapshot_fetcher=self.region_fetcher,
                news_features_path=session.news_features_path,
                news_source_run_id=session.news_run_id,
                news_source_p1_run_id=session.news_p1_run_id,
            ),
            success_handler=self._forecast_plan_completed,
            failure_handler=self._forecast_prepare_failed,
        )

    def update_full_flow_state(
        self,
        state: FullFlowState,
        *,
        state_path: str | Path,
        output_directory: str | Path | None = None,
    ) -> None:
        """Project the application-level P1/P2/P3 cursor into the desktop session."""

        session = self.current_session
        # This legacy field only bridges an ordinary P1 Graph result to the standalone
        # forecast branch. Once a durable full flow owns the request, its phase cursor
        # is the sole continuation mechanism.
        session.pending_forecast_question = None
        session.full_flow_state_path = str(Path(state_path).resolve())
        if output_directory is not None:
            session.full_flow_output_directory = str(Path(output_directory).resolve())
        for message in session.messages:
            if message.kind == "p2_review" and message.payload.get("full_flow_id") == state.flow_id:
                message.payload["phase"] = state.phase
        session.news_features_path = str(state.p2_feature_path) if state.p2_feature_path else None
        news_run = next((run for run in reversed(state.runs) if run.run_kind == "news"), None)
        synthesis_run = next(
            (
                run
                for run in reversed(state.runs)
                if run.run_kind == "eda" and run.parent_run_id == (news_run.run_id if news_run else None)
            ),
            None,
        )
        session.news_run_id = news_run.run_id if news_run else None
        session.news_p1_run_id = synthesis_run.run_id if synthesis_run else None
        questions = {
            "news": "分析本地新闻并生成事件证据与数值特征",
            "feedback": "预测结果反馈：诊断P3未达标或不可评估的原因",
        }
        for run in state.runs:
            if any(existing.run_id == run.run_id for existing in session.runs):
                continue
            session.runs.append(
                SessionRunRecord(
                    run_id=run.run_id,
                    run_kind=run.run_kind,
                    plan_id=f"{state.flow_id}:{run.run_kind}",
                    parent_run_id=run.parent_run_id,
                    question=questions.get(
                        run.run_kind,
                        "综合分析电价、外生变量与新闻证据" if run.parent_run_id else "初步分析电价与外生变量",
                    ),
                    artifact_directory=str(run.artifact_directory),
                    report_path=str(run.report_path),
                    evaluation={
                        "decision": "need_user" if run.run_kind in {"news", "feedback"} else "accept",
                        "summary": f"全流程阶段：{state.phase}",
                    },
                    data_fingerprint=run.data_fingerprint,
                    study_name="P1—P2—P3一轮研究",
                    target_name="rt_price",
                    study_start_time=state.p1_request.price_start.isoformat(),
                    study_end_time=state.p1_request.price_end.isoformat(),
                )
            )
        if state.runs and state.runs[-1].run_kind != "feedback":
            latest = state.runs[-1]
            session.run_id = latest.run_id
            session.artifact_directory = str(latest.artifact_directory)
            session.report_path = str(latest.report_path)
        if state.phase in {"awaiting_forecast_approval", "forecast_running"} and state.forecast_plan:
            plan = ForecastPlan.model_validate(state.forecast_plan)
            session.current_plan = plan.model_dump(mode="json")
            session.plan_stale = False
            session.status = "awaiting_plan_approval"
            self._active_interrupt_id = None
            self._active_state_revision = None
            self._active_interrupt_kind = None
            plan_message = next(
                (
                    message
                    for message in reversed(session.messages)
                    if message.kind == "forecast_plan"
                    and (message.payload.get("plan") or {}).get("plan_id") == plan.plan_id
                ),
                None,
            )
            if plan_message is None:
                self._append_message(
                    SessionMessage(
                        role="assistant",
                        kind="forecast_plan",
                        content="山东次日实时电价预测方案等待确认",
                        payload={"plan": plan.model_dump(mode="json"), "state": "awaiting"},
                    )
                )
            else:
                plan_message.payload["state"] = "awaiting"
            if self.conversation.current_plan_widget is not None:
                self.conversation.current_plan_widget.set_explicit_approval()
        elif state.phase == "p2_needs_review":
            session.current_plan = None
            session.plan_stale = False
            session.status = "awaiting_user"
        elif state.phase in {"p1_initial_complete", "p2_ready"}:
            session.current_plan = None
            session.plan_stale = False
            session.status = "understanding" if self.is_busy else "awaiting_user"
        elif state.phase == "p1_synthesis_complete":
            session.current_plan = None
            session.plan_stale = False
            session.status = (
                "understanding" if state.requested_forecast and self.is_busy else "awaiting_user"
            )
        elif state.phase in {"forecast_verified", "feedback_complete"}:
            session.status = "completed"
        elif state.phase == "failed":
            session.status = "failed"
        elif state.phase == "stopped":
            session.current_plan = None
            session.plan_stale = False
            session.status = "stopped"
        if state.phase == "p2_needs_review":
            self._show_p2_review_gate(state)
            return
        labels = {
            "p1_initial_complete": "P1初步分析完成",
            "p2_ready": "P2新闻证据完成",
            "p2_needs_review": "P2存在待复核项",
            "p1_synthesis_complete": (
                "P1综合分析完成，正在准备P3"
                if state.requested_forecast
                else "P1综合分析完成；如需预测，请发送“继续第三阶段”"
            ),
            "awaiting_forecast_approval": "P3方案等待单独确认",
            "forecast_running": "P3正在运行",
            "forecast_verified": "P3达到目标，本轮结束",
            "feedback_complete": "预测结果反馈完成，本轮结束",
            "failed": "全流程执行失败",
            "stopped": "全流程已停止",
        }
        previous = next(
            (
                message
                for message in reversed(session.messages)
                if message.kind == "notice" and message.payload.get("full_flow_id") == state.flow_id
            ),
            None,
        )
        if previous is None or previous.payload.get("phase") != state.phase:
            self._append_message(
                SessionMessage(
                    role="system",
                    kind="notice",
                    content=labels[state.phase] + (f"：{state.stop_reason}" if state.stop_reason else ""),
                    payload={
                        "full_flow_id": state.flow_id,
                        "phase": state.phase,
                        "forecast_runs_used": state.forecast_runs_used,
                        "feedback_runs_used": state.feedback_runs_used,
                        "current_stage": state.current_stage,
                        "last_completed_stage": state.last_completed_stage,
                        "blocked_reason": state.blocked_reason,
                        "allowed_actions": list(state.allowed_actions),
                        "stage_attempts": state.stage_attempts,
                        "state_path": session.full_flow_state_path,
                    },
                )
            )
        self._persist_and_render(keep_timeline=True)

    def _forecast_prepare_failed(self, detail: str) -> None:
        headline = detail.strip().splitlines()[-1] if detail.strip() else "预测输入检查失败"
        if "ForecastDataError" in detail:
            headline = headline.removeprefix("app.research.forecasting.data.ForecastDataError: ")
        session = self.current_session
        session.status = "failed"
        self._close_latest_running_trace("failed")
        self._set_active_tool_status("failed", headline)
        self._append_message(SessionMessage(role="assistant", kind="error", content=f"预测方案没有生成：{headline}"))
        self._add_trace("error", "预测输入检查未通过", "failed", headline)
        self._persist_and_render(keep_timeline=True)

    def _forecast_plan_completed(self, plan: ForecastPlan) -> None:
        session = self.current_session
        self._complete_active_tool("4份预测输入已冻结")
        session.current_plan = plan.model_dump(mode="json")
        session.plan_stale = False
        session.status = "awaiting_plan_approval"
        self._active_interrupt_id = None
        self._active_state_revision = None
        self._active_interrupt_kind = None
        self._append_message(
            SessionMessage(
                role="assistant",
                kind="forecast_plan",
                content="山东次日实时电价预测方案等待确认",
                payload={"plan": plan.model_dump(mode="json"), "state": "awaiting"},
            )
        )
        if self.conversation.current_plan_widget is not None:
            self.conversation.current_plan_widget.set_explicit_approval()
        self._add_trace(
            "plan",
            "生成山东实时电价预测方案",
            "completed",
            "3个历史回测锚点、1份次日数据；必须显式确认",
        )
        self._persist_and_render(keep_timeline=True)

    def _explain_forecast_result(self, question: str) -> None:
        """Explain the latest P3 evidence without preparing or executing a new plan."""

        session = self.current_session
        self._append_message(SessionMessage(role="user", kind="text", content=question, turn_id=uuid4().hex))
        message = next((item for item in reversed(session.messages) if item.kind == "forecast_result"), None)
        if message is None:
            self._append_message(
                SessionMessage(role="assistant", kind="error", content="当前会话没有可解释的预测结果。")
            )
            self._persist_and_render(keep_timeline=True)
            return
        aggregate = message.payload.get("aggregate") or {}
        model = aggregate.get("model") or {}
        day = aggregate.get("day_naive") or {}
        week = aggregate.get("week_naive") or {}
        diagnostics = message.payload.get("diagnostics") or {}
        status = diagnostics.get("evaluation_status") or (
            "verified" if diagnostics.get("baseline_verified") else "not_verified"
        )
        label = {
            "verified": "达到了逐折样本门禁，并优于日前和周前朴素基线",
            "not_verified": "样本可评估，但没有同时优于日前和周前朴素基线",
            "not_evaluable": "共同有效点不足，当前不能判断预测增益",
        }.get(str(status), "当前没有足够证据判断预测增益")
        answer = (
            f"本次预测{label}。模型聚合 MAE 为 {float(model.get('mae', float('nan'))):.2f}，"
            f"日前同刻基线为 {float(day.get('mae', float('nan'))):.2f}，"
            f"周前同刻基线为 {float(week.get('mae', float('nan'))):.2f}。"
        )
        warnings = [str(item) for item in message.payload.get("warnings", [])]
        if warnings:
            answer += " 主要限制：" + "；".join(warnings)
        self._append_message(SessionMessage(role="assistant", kind="text", content=answer))
        session.status = "completed"
        self._active_interrupt_kind = None
        self._persist_and_render(keep_timeline=True)

    def _legacy_graph_import(self, session: ResearchSession) -> dict[str, Any]:
        latest_record = next(
            (run for run in reversed(session.runs) if not session.run_id or run.run_id == session.run_id),
            session.runs[-1] if session.runs else None,
        )
        latest_evaluation = latest_record.evaluation if latest_record else session.latest_evaluation
        latest_is_eda = latest_record is None or latest_record.run_kind == "eda"
        latest_run = None
        if session.run_id:
            latest_run = {
                "run_id": session.run_id,
                "plan_id": latest_record.plan_id if latest_record else (session.current_plan or {}).get("plan_id"),
                "artifact_directory": session.artifact_directory,
                "report_path": session.report_path,
                "figure_paths": {},
                "evaluation": latest_evaluation,
            }
        episode_summaries = []
        run_history = []
        for number, run in enumerate(session.runs[-48:], start=max(1, len(session.runs) - 47)):
            evaluation = run.evaluation or {}
            decision = evaluation.get("decision")
            episode_summaries.append(
                EpisodeSummary(
                    episode_id=run.episode_id or f"legacy-{run.run_id}",
                    episode_number=number,
                    goal=run.question,
                    status={"accept": "accepted", "reject": "rejected"}.get(decision, "limited"),
                    run_id=run.run_id,
                    plan_id=run.plan_id,
                    evaluation_decision=decision,
                    summary=str(evaluation.get("summary") or ""),
                    findings=[str(item) for item in evaluation.get("findings", [])[:8]],
                    warnings=[str(item) for item in evaluation.get("warnings", [])[:8]],
                    report_path=run.report_path,
                    data_fingerprint=run.data_fingerprint,
                    study_name=run.study_name,
                    target_name=run.target_name,
                    study_start_time=run.study_start_time,
                    study_end_time=run.study_end_time,
                    skill_name=run.skill_name,
                    skill_version=run.skill_version,
                    memory_status=run.memory_status,
                    updated_at=run.created_at,
                ).model_dump(mode="json")
            )
            run_history.append(
                {
                    "run_id": run.run_id,
                    "plan_id": run.plan_id,
                    "artifact_directory": run.artifact_directory,
                    "report_path": run.report_path,
                    "evaluation_decision": decision,
                }
            )
        return {
            "current_plan": session.current_plan,
            "plan_history": [session.current_plan] if session.current_plan else [],
            "data_profile": session.data_profile,
            "quality_report": session.quality_report,
            "data_fingerprint": (session.current_plan or {}).get("data_fingerprint"),
            "eda_summary": session.latest_eda_summary if latest_is_eda else None,
            "evaluation": latest_evaluation,
            "latest_run": latest_run,
            "run_history": run_history or ([latest_run] if latest_run else []),
            "episode_summaries": episode_summaries,
            "phase": "completed" if session.run_id else "idle",
        }

    def _loop_completed(self, snapshot: ResearchLoopSnapshot) -> None:
        session = self.current_session
        self._complete_active_tool("研究循环已到达用户交互点")
        values = snapshot.values
        self._active_interrupt_id = snapshot.interrupt.interrupt_id or None if snapshot.interrupt else None
        self._active_state_revision = snapshot.interrupt.state_revision if snapshot.interrupt else None
        self._active_interrupt_kind = snapshot.interrupt.kind if snapshot.interrupt else None
        self.conversation.set_interaction_context(self._active_interrupt_kind)
        self._sync_graph_events(session, snapshot.events)
        forecast_control = str(values.get("control") or "")
        if forecast_control == "forecast_plan_request":
            # The Graph recognizes intent; the desktop owns the fixed forecast
            # snapshots and must create the card before any approval is accepted.
            self._pending_forecast_request = str(values.get("latest_turn") or values.get("user_request") or "").strip()
            session.status = "understanding"
            self._persist_and_render(keep_timeline=True)
            return
        if forecast_control == "forecast_execute_request":
            self._append_message(
                SessionMessage(
                    role="assistant",
                    kind="error",
                    content="当前没有可确认的预测方案。请先生成并查看预测方案卡。",
                )
            )
            session.status = "awaiting_user"
            self._persist_and_render(keep_timeline=True)
            return
        self._sync_assistant_text(session, values)
        if (
            snapshot.interrupt is None
            and snapshot.phase in {"planning", "validating_plan", "executing_tools", "validating_result", "evaluating"}
            and not self.is_busy
        ):
            session.status = "running"
            self._start_worker(
                kind="execute",
                operation=lambda _progress: self.agent.continue_thread(session.session_id),
                success_handler=self._loop_completed,
            )
            return
        if values.get("current_plan"):
            session.current_plan = dict(values["current_plan"])
            session.plan_stale = False
        session.data_profile = values.get("data_profile")
        session.quality_report = values.get("quality_report")
        session.latest_eda_summary = values.get("eda_summary")
        session.latest_evaluation = values.get("evaluation")

        interrupt_payload = snapshot.interrupt
        if interrupt_payload and interrupt_payload.kind == "plan_approval":
            plan = EDAPlan.model_validate(interrupt_payload.plan or values["current_plan"])
            existing = next(
                (
                    message
                    for message in reversed(session.messages)
                    if message.kind in {"plan", "data_plan"}
                    and message.payload.get("plan", {}).get("plan_id") == plan.plan_id
                ),
                None,
            )
            if existing is None:
                if any(message.kind in {"plan", "data_plan"} for message in session.messages):
                    self._set_plan_message_state("stale")
                profile = values.get("data_profile") or {}
                quality = DataQualityReport.model_validate(values["quality_report"])
                stored_summary = values.get("data_summary")
                proposal = ResearchProposal(
                    plan=plan,
                    assistant_message="模型方案已通过确定性校验，等待你的修改或确认。",
                    data_profile=ResearchDataProfile.model_validate(profile),
                    quality_report=quality,
                    data_summary=parse_summary(stored_summary) if stored_summary else None,
                )
                self._proposal_completed(proposal)
                return
            session.status = "awaiting_plan_approval"
            self._restore_approval_timer(snapshot)
            self._persist_and_render(keep_timeline=True)
            return

        latest = values.get("latest_run")
        added_run = bool(
            latest and latest.get("run_id") and not any(item.run_id == latest["run_id"] for item in session.runs)
        )
        if added_run:
            evaluation = latest.get("evaluation") or {}
            self._append_message(
                SessionMessage(
                    role="assistant",
                    kind="result",
                    content=evaluation.get("summary", "研究循环已完成一轮确定性评估。"),
                    payload={
                        "run_id": latest["run_id"],
                        "report_path": latest.get("report_path"),
                        "artifact_directory": latest.get("artifact_directory"),
                        "figure_count": len(latest.get("figure_paths", {})),
                        "evaluation": evaluation,
                    },
                )
            )
            session.run_id = latest["run_id"]
            session.artifact_directory = latest.get("artifact_directory")
            session.report_path = latest.get("report_path")
            plan_value = values.get("current_plan") or session.current_plan or {}
            study_value = values.get("study_config") or {}
            study_definition = study_value.get("study") or {}
            target_definition = study_value.get("target") or {}
            session.runs.append(
                SessionRunRecord(
                    run_id=latest["run_id"],
                    episode_id=(values.get("loop_cursor") or {}).get("episode_id"),
                    plan_id=latest.get("plan_id") or (session.current_plan or {}).get("plan_id", "unknown"),
                    parent_run_id=session.runs[-1].run_id if session.runs else None,
                    question=(session.current_plan or {}).get("question", "研究循环"),
                    artifact_directory=latest.get("artifact_directory") or "",
                    report_path=latest.get("report_path") or "",
                    evaluation=evaluation,
                    data_fingerprint=plan_value.get("data_fingerprint"),
                    study_name=study_definition.get("name"),
                    target_name=target_definition.get("name"),
                    study_start_time=(
                        str(study_definition["start_time"]) if study_definition.get("start_time") is not None else None
                    ),
                    study_end_time=(
                        str(study_definition["end_time"]) if study_definition.get("end_time") is not None else None
                    ),
                    skill_name=plan_value.get("skill_name"),
                    skill_version=plan_value.get("skill_version"),
                )
            )
            self._set_plan_message_state("completed")

        if (
            added_run
            and interrupt_payload is not None
            and interrupt_payload.kind == "result"
            and session.pending_forecast_question
        ):
            self._pending_forecast_request = session.pending_forecast_question
            session.pending_forecast_question = None

        if interrupt_payload and interrupt_payload.kind in {
            "plan_error",
            "result_limitations",
            "result_rejected",
            "response_error",
            "finalization_error",
        }:
            feedback_codes = {str(item.get("code") or "") for item in values.get("feedback_packets", [])}
            if interrupt_payload.kind == "plan_error":
                if "session_skill_version_mismatch" in feedback_codes:
                    notice = (
                        f"{interrupt_payload.message}\n请点击左侧“新建研究”重新提交问题；旧对话和此前报告仍可查看。"
                    )
                elif values.get("latest_run"):
                    notice = (
                        f"{interrupt_payload.message}\n"
                        "此前的数据质量或研究结果仍然保留；本次新增分析尚未执行。"
                        "你可以修改要求、回复“重试”或“停止”。"
                    )
                else:
                    notice = (
                        f"{interrupt_payload.message}\n"
                        "本次尚未形成可执行方案，也没有启动计算。"
                        "你可以回复“重试”，换个说法提问，或者回复“停止”。"
                    )
            elif interrupt_payload.kind in {"response_error", "finalization_error"}:
                notice = f"{interrupt_payload.message}\n你可以回复“重试”或“停止”。"
            elif interrupt_payload.kind == "result_rejected":
                notice = f"{interrupt_payload.message}\n你可以提出修改意见或回复“停止”。"
            else:
                notice = (
                    f"{interrupt_payload.message}\n在下方输入下一步研究要求，例如“按推荐变量继续”，或说明要增删的变量。"
                )
            recent_notices = {
                message.content
                for message in session.messages[-8:]
                if message.role == "system" and message.kind == "notice"
            }
            if notice not in recent_notices:
                self._append_message(SessionMessage(role="system", kind="notice", content=notice))
        session.status = self._projected_session_status(snapshot)
        self._persist_and_render(keep_timeline=True)

    @staticmethod
    def _projected_session_status(snapshot: ResearchLoopSnapshot) -> SessionStatus:
        """Derive UI status from the authoritative Graph snapshot."""

        if snapshot.interrupt is not None:
            if snapshot.interrupt.kind == "plan_approval":
                return "awaiting_plan_approval"
            if snapshot.interrupt.kind == "result":
                return "completed"
            return "awaiting_user"
        return {
            "idle": "idle",
            "understanding": "understanding",
            "planning": "understanding",
            "validating_plan": "understanding",
            "awaiting_approval": "awaiting_plan_approval",
            "executing_tools": "running",
            "validating_result": "running",
            "evaluating": "evaluating",
            "awaiting_user": "awaiting_user",
            "completed": "completed",
            "stopped": "stopped",
            "failed": "failed",
        }[snapshot.phase]

    def _sync_assistant_text(self, session: ResearchSession, values: dict[str, Any]) -> None:
        """Project one Graph assistant message exactly once using its durable ID."""

        assistant_message = str(values.get("assistant_message") or "").strip()
        if not assistant_message:
            return
        graph_message = next(
            (
                item
                for item in reversed(values.get("messages", []))
                if item.get("role") == "assistant" and str(item.get("content") or "").strip() == assistant_message
            ),
            None,
        )
        graph_message_id = str((graph_message or {}).get("message_id") or "")
        if graph_message_id:
            if any(item.message_id == graph_message_id for item in session.messages):
                return
        else:
            # Compatibility for Graph messages created before message IDs existed.
            matching_indexes = [
                index
                for index, message in enumerate(session.messages)
                if message.role == "assistant" and message.kind == "text" and message.content == assistant_message
            ]
            if matching_indexes and not any(
                message.role == "user" for message in session.messages[matching_indexes[-1] + 1 :]
            ):
                return
        self._append_message(
            SessionMessage(
                message_id=graph_message_id or uuid4().hex,
                turn_id=(graph_message or {}).get("turn_id"),
                episode_id=(graph_message or {}).get("episode_id"),
                role="assistant",
                kind="text",
                content=assistant_message,
            )
        )

    def _sync_graph_events(self, session: ResearchSession, events: list[dict[str, Any]]) -> None:
        sequenced = [(int(item["sequence"]), item) for item in events if isinstance(item.get("sequence"), int)]
        if sequenced:
            new_events = [item for sequence, item in sequenced if sequence > session.graph_event_sequence]
        else:
            new_events = events[session.graph_event_count :]
        valid_categories = {"session", "user", "agent", "input", "plan", "tool", "evaluation", "artifact", "error"}
        for item in new_events:
            status = str(item.get("status", "info"))
            trace_status = (
                status if status in {"info", "running", "completed", "warning", "failed", "stopped"} else "info"
            )
            details = dict(item.get("details", {}))
            name = str(item.get("name", "研究循环事件"))
            category = str(item.get("category", ""))
            if category not in valid_categories:
                category = (
                    "tool"
                    if any(word in name for word in ("工具", "函数"))
                    else "evaluation"
                    if "评估" in name
                    else "plan"
                    if "方案" in name
                    else "agent"
                )
            duration_ms = float(details["duration_ms"]) if details.get("duration_ms") is not None else None
            summary = narrate_event(
                {"name": name, "status": trace_status, "details": details, "category": category}
            ).detail
            existing = next(
                (event for event in reversed(session.trace) if event.name == name and not event.details),
                None,
            )
            if existing is not None:
                existing.category = category  # type: ignore[assignment]
                existing.status = trace_status  # type: ignore[assignment]
                existing.duration_ms = duration_ms
                existing.summary = summary
                existing.details = details
                continue
            session.trace.append(
                TraceEvent(
                    category=category,  # type: ignore[arg-type]
                    name=name,
                    status=trace_status,  # type: ignore[arg-type]
                    duration_ms=duration_ms,
                    summary=summary,
                    details=details,
                )
            )
        session.graph_event_count = len(events)
        if sequenced:
            session.graph_event_sequence = max(sequence for sequence, _item in sequenced)

    def _restore_approval_timer(self, snapshot: ResearchLoopSnapshot) -> None:
        session = self.current_session
        approval = ApprovalState.model_validate(snapshot.values.get("approval_state", {}))
        session.plan_feedback_deadline = approval.deadline
        session.plan_feedback_remaining_seconds = approval.remaining_seconds
        if not self.auto_execute_plan or not bool(snapshot.values.get("automatic_approval_enabled", False)):
            self._plan_feedback_timer.stop()
            if self.conversation.current_plan_widget is not None:
                self.conversation.current_plan_widget.set_explicit_approval()
            return
        if approval.status == "expired":
            if self.conversation.current_plan_widget is not None:
                self.conversation.current_plan_widget.set_feedback_paused("审批已过期，请明确确认")
            return
        if approval.deadline:
            try:
                remaining = ceil((datetime.fromisoformat(approval.deadline) - datetime.now(UTC)).total_seconds())
            except ValueError:
                remaining = approval.remaining_seconds or self.plan_feedback_seconds
            if remaining <= 0:
                session.plan_feedback_deadline = None
                session.plan_feedback_remaining_seconds = 0
                if self.conversation.current_plan_widget is not None:
                    self.conversation.current_plan_widget.set_feedback_paused("等待时间已过，请重新确认")
                return
            self._start_plan_feedback_window(remaining)

    def _proposal_completed(self, proposal: ResearchProposal) -> None:
        session = self.current_session
        self._complete_active_tool("数据检查完成")
        self._apply_quality_to_inputs(proposal)
        summary = self._summary_from_proposal(proposal)
        self._set_data_state("ready", summary)
        self._warn_about_unnamed_variables(summary)
        self._append_message(SessionMessage(role="assistant", kind="text", content=proposal.assistant_message))
        self._append_plan_message(proposal.plan, proposal.data_profile.exogenous_names)
        session.current_plan = proposal.plan.model_dump(mode="json")
        session.plan_stale = False
        session.data_profile = proposal.data_profile.model_dump(mode="json")
        session.quality_report = proposal.quality_report.model_dump(mode="json")
        session.status = "awaiting_plan_approval"
        if self.auto_execute_plan:
            self._start_plan_feedback_window()
        elif self.conversation.current_plan_widget is not None:
            self.conversation.current_plan_widget.set_explicit_approval()
        self._persist_and_render(keep_timeline=True)

    def _append_plan_message(self, plan: EDAPlan, available_variables: list[str]) -> None:
        """Confirm the data and the analysis in one card, never as two decisions."""

        session = self.current_session
        session.dataset_fingerprint = plan.data_fingerprint or session.dataset_fingerprint
        self._append_message(
            SessionMessage(
                role="assistant",
                kind="data_plan",
                content="开始之前，跟你确认一下",
                payload={
                    "plan": plan.model_dump(mode="json"),
                    "available_variables": available_variables,
                    "data_summary": summary_payload(session.data_summary),
                    "state": "awaiting",
                },
            )
        )

    def _warn_about_unnamed_variables(self, summary: DataSummary) -> None:
        """Flag columns shown under their stored name without stopping the research."""

        unnamed = [item.display for item in summary.variables if not item.resolved]
        if not unnamed:
            return
        self._add_trace(
            "input",
            f"有 {len(unnamed)} 项数据没有中文名，先按原名显示",
            "warning",
            "、".join(unnamed),
        )

    def _restored_dialogue_status(self) -> str:
        if self._task_previous_status in {"awaiting_plan_approval", "completed"}:
            return self._task_previous_status
        return "idle"

    def _start_plan_feedback_window(self, seconds: int | None = None) -> None:
        session = self.current_session
        if not self.auto_execute_plan or session.status != "awaiting_plan_approval" or session.current_plan is None:
            return
        duration = max(1, int(seconds or self.plan_feedback_seconds))
        self._plan_feedback_timer.stop()
        session.plan_feedback_remaining_seconds = duration
        session.plan_feedback_deadline = (datetime.now(UTC) + timedelta(seconds=duration)).isoformat()
        if self.conversation.current_plan_widget is not None:
            self.conversation.current_plan_widget.set_feedback_countdown(duration)
        self._plan_feedback_timer.start()

    def _cancel_plan_feedback_window(self) -> None:
        self._plan_feedback_timer.stop()
        if self.current_session_id is None:
            return
        session = self.current_session
        session.plan_feedback_deadline = None
        session.plan_feedback_remaining_seconds = None

    def _remaining_plan_feedback_seconds(self) -> int:
        deadline = self.current_session.plan_feedback_deadline
        if not deadline:
            return self.current_session.plan_feedback_remaining_seconds or self.plan_feedback_seconds
        try:
            remaining = ceil((datetime.fromisoformat(deadline) - datetime.now(UTC)).total_seconds())
        except ValueError:
            return self.plan_feedback_seconds
        return max(0, remaining)

    def _composer_draft_changed(self, has_text: bool) -> None:
        session = self.current_session
        if (
            not self.auto_execute_plan
            or session.status != "awaiting_plan_approval"
            or session.current_plan is None
            or self.is_busy
        ):
            return
        if has_text and self._plan_feedback_timer.isActive():
            remaining = max(1, self._remaining_plan_feedback_seconds())
            self._plan_feedback_timer.stop()
            session.plan_feedback_deadline = None
            session.plan_feedback_remaining_seconds = remaining
            if self.conversation.current_plan_widget is not None:
                self.conversation.current_plan_widget.set_feedback_paused()
            self.store.save(self.sessions)
        elif not has_text and not self._plan_feedback_timer.isActive():
            self._start_plan_feedback_window(session.plan_feedback_remaining_seconds)

    def pause_plan_feedback_window(self) -> None:
        """Pause auto-execution while another modal UI needs the user's attention."""

        session = self.current_session
        if (
            not self.auto_execute_plan
            or session.status != "awaiting_plan_approval"
            or not self._plan_feedback_timer.isActive()
        ):
            return
        remaining = max(1, self._remaining_plan_feedback_seconds())
        self._plan_feedback_timer.stop()
        session.plan_feedback_deadline = None
        session.plan_feedback_remaining_seconds = remaining
        if self.conversation.current_plan_widget is not None:
            self.conversation.current_plan_widget.set_feedback_paused("倒计时已暂停")

    def resume_plan_feedback_window(self) -> None:
        """Resume a paused plan window without resetting its remaining time."""

        session = self.current_session
        if (
            self.auto_execute_plan
            and session.status == "awaiting_plan_approval"
            and session.current_plan
            and not self.is_busy
        ):
            self._start_plan_feedback_window(session.plan_feedback_remaining_seconds)

    def _plan_feedback_tick(self) -> None:
        session = self.current_session
        if not self.auto_execute_plan or session.status != "awaiting_plan_approval" or session.current_plan is None:
            self._cancel_plan_feedback_window()
            return
        remaining = self._remaining_plan_feedback_seconds()
        session.plan_feedback_remaining_seconds = remaining
        if self.conversation.current_plan_widget is not None:
            self.conversation.current_plan_widget.set_feedback_countdown(remaining)
        if remaining > 0 or self.is_busy:
            return
        self._cancel_plan_feedback_window()
        self._append_message(
            SessionMessage(
                role="system",
                kind="notice",
                content=f"{self.plan_feedback_seconds} 秒内没有收到修改意见，已按上面的方案开始分析。",
            )
        )
        self._add_trace("plan", "确认超时，自动执行方案", "completed", "未收到修改意见")
        self._resume_graph(action="timeout_accept", task_kind="execute", foreground_timeout=True)

    def _apply_quality_to_inputs(self, proposal: ResearchProposal) -> None:
        session = self.current_session
        by_role: dict[str, list[VariableEvidence]] = {"target": [], "actuals": [], "forecasts": []}
        for name, report in proposal.quality_report.series.items():
            report_path = Path(report.path).resolve()
            role = next(
                (
                    candidate
                    for candidate in ("target", "actuals", "forecasts")
                    if session.inputs[candidate].path and Path(session.inputs[candidate].path).resolve() == report_path
                ),
                None,
            )
            if role is None:
                continue
            status = "warning" if report.aligned_coverage_rate < 0.9 else "ready"
            by_role[role].append(
                VariableEvidence(
                    name=name,
                    coverage_rate=report.aligned_coverage_rate,
                    status=status,
                    detail=f"有效值 {report.aligned_non_null_rows:,}；缺失 {report.missing_interval_count:,}",
                )
            )
        for role in ("target", "actuals", "forecasts"):
            if not session.inputs[role].path:
                continue
            evidence = by_role[role]
            session.inputs[role].variables = evidence
            session.inputs[role].status = "warning" if any(item.status == "warning" for item in evidence) else "ready"
            session.inputs[role].detail = f"{len(evidence)} 个变量已校验"

    def run_plan(self, approved_plan: object) -> None:
        if self.current_session.read_only:
            return
        if self.is_busy:
            return
        if isinstance(approved_plan, ForecastPlan):
            self._run_forecast_plan(approved_plan)
            return
        approved_plan = EDAPlan.model_validate(approved_plan)
        self._cancel_plan_feedback_window()
        session = self.current_session
        session.current_plan = approved_plan.model_dump(mode="json")
        session.status = "running"
        self._set_plan_message_state("running", plan=approved_plan)
        plan_widget = self.conversation.current_plan_widget
        if plan_widget is not None:
            plan_widget.set_running()
        self._append_message(
            SessionMessage(
                role="assistant",
                kind="text",
                content="好的，开始分析。每一步的进展会显示在下面，右侧“研究过程”里有完整记录。",
            )
        )
        self._start_thinking("compute", "启动方案执行", "按锁定的执行计划逐项分析")
        self._add_trace(
            "plan",
            "方案已确认",
            "completed",
            f"共 {len(approved_plan.enabled_steps)} 项分析待执行",
        )
        self._start_worker(
            kind="execute",
            operation=(
                lambda progress: (
                    self.agent.resume(
                        session_id=session.session_id,
                        action="approve",
                        interrupt_id=self._active_interrupt_id,
                        state_revision=self._active_state_revision,
                        progress=progress,
                    )
                    if self.agent.has_thread(session.session_id)
                    else self.agent.submit_user_message(
                        session_id=session.session_id,
                        message="按这个执行",
                        study_config=build_runtime_study(
                            session,
                            output_directory=self.research_output_directory,
                        ),
                        conversation=self._agent_conversation(session),
                        approval_timeout_seconds=self.plan_feedback_seconds,
                        automatic_approval_enabled=self.auto_execute_plan,
                        imported_state=self._legacy_graph_import(session),
                        progress=progress,
                    )
                )
            ),
            success_handler=self._loop_completed,
        )
        self.conversation.scroll_to_bottom()

    def _run_forecast_plan(self, approved_plan: ForecastPlan) -> None:
        """Run only the four immutable files accepted in the forecast card."""

        self._cancel_plan_feedback_window()
        session = self.current_session
        full_flow = self._full_flow_for_plan(approved_plan)
        if full_flow is not None:
            lineage_error = self._full_flow_lineage_error(full_flow[1])
            if lineage_error:
                stopped = full_flow[0].stop(full_flow[1], reason=lineage_error)
                self.update_full_flow_state(
                    stopped,
                    state_path=full_flow[0].store.path,
                    output_directory=full_flow[0].output_directory,
                )
                return
        if full_flow is not None and full_flow[1].phase in {"forecast_verified", "feedback_complete", "failed"}:
            state = full_flow[1]
            self._append_message(
                SessionMessage(
                    role="assistant",
                    kind="error",
                    content=(
                        "本轮P3执行预算已经结束，不能再次执行。"
                        + (f"停止原因：{state.stop_reason}" if state.stop_reason else "")
                    ),
                )
            )
            self._persist_and_render(keep_timeline=True)
            return
        session.current_plan = approved_plan.model_dump(mode="json")
        session.status = "running"
        self._set_plan_message_state("running", plan=approved_plan)
        if self.conversation.current_plan_widget is not None:
            self.conversation.current_plan_widget.set_running()
        self._append_message(
            SessionMessage(
                role="assistant",
                kind="text",
                content="好的，开始做3个历史锚点回测，然后生成次日96点预测。训练只使用CPU和已固定数据。",
            )
        )
        self._start_thinking("compute", "启动山东电价预测", "回测 1/3 即将开始")
        self._add_trace("plan", "预测方案已确认", "completed", "只读运行，不写业务数据库")
        if full_flow is None:
            self._start_worker(
                kind="forecast_execute",
                operation=lambda progress: execute_forecast_workflow(
                    approved_plan,
                    progress=progress,
                ),
                success_handler=self._forecast_run_completed,
            )
        else:
            flow, state = full_flow
            self._start_worker(
                kind="full_flow_forecast_execute",
                operation=lambda progress: flow.approve_and_run_p3(
                    state=state,
                    plan_id=approved_plan.plan_id,
                    plan_fingerprint=str(state.forecast_plan_fingerprint),
                    progress=progress,
                ),
                success_handler=self._full_flow_forecast_completed,
                failure_handler=self._full_flow_forecast_failed,
            )
        self.conversation.scroll_to_bottom()

    def _full_flow_for_plan(
        self,
        plan: ForecastPlan,
    ) -> tuple[FullResearchFlow, FullFlowState] | None:
        """Resolve a persisted flow only when it owns the exact desktop plan."""

        session = self.current_session
        if not session.full_flow_state_path:
            return None
        store = FullFlowStore(session.full_flow_state_path)
        try:
            state = store.load()
        except (OSError, ValueError):
            return None
        if not state.forecast_plan or state.forecast_plan.get("plan_id") != plan.plan_id:
            return None
        output_directory = session.full_flow_output_directory or str(store.path.parent / "runs")
        return FullResearchFlow(store=store, output_directory=output_directory), state

    def _full_flow_forecast_completed(self, state: FullFlowState) -> None:
        """Render P3 evidence and the one allowed P1 feedback from persisted flow state."""

        session = self.current_session
        feedback = state.feedback
        if feedback is None:
            self._full_flow_forecast_failed("全流程结束但没有生成P3反馈包")
            return
        self._complete_active_tool("3折回测、次日96点预测和有界反馈分析已完成")
        diagnostics = {
            "evaluation_status": feedback.status,
            "baseline_verified": feedback.status == "verified",
            "fold_common_observations": list(feedback.fold_common_observations),
            "news_feature_columns": list(feedback.used_news_features),
            "excluded_news_features": feedback.excluded_news_features,
            "feedback_diagnosis": list(feedback.diagnosis),
        }
        self._append_message(
            SessionMessage(
                role="assistant",
                kind="forecast_result",
                content="山东次日实时电价预测与本轮反馈分析完成",
                payload={
                    "run_id": feedback.forecast_run_id,
                    "aggregate": feedback.aggregate_metrics,
                    "diagnostics": diagnostics,
                    "warnings": list(feedback.warnings),
                    "prediction_path": str(feedback.prediction_path or ""),
                    "backtest_path": str(feedback.backtest_path or ""),
                    "metrics_path": str(feedback.metrics_path or ""),
                    "report_path": str(feedback.report_path or ""),
                    "artifact_directory": str(feedback.artifact_directory or ""),
                    "figure_paths": {name: str(path) for name, path in feedback.figure_paths.items()},
                    "output_hash": feedback.output_hash or "",
                    "full_flow_id": state.flow_id,
                    "stop_reason": state.stop_reason,
                },
            )
        )
        forecast_run = next(
            (run for run in reversed(state.runs) if run.run_kind == "forecast"),
            None,
        )
        if forecast_run is not None and not any(run.run_id == forecast_run.run_id for run in session.runs):
            session.runs.append(
                SessionRunRecord(
                    run_id=forecast_run.run_id,
                    run_kind="forecast",
                    plan_id=feedback.plan_id,
                    parent_run_id=forecast_run.parent_run_id,
                    question=(state.forecast_plan or {}).get("question", "预测山东明天实时电价"),
                    artifact_directory=str(forecast_run.artifact_directory),
                    report_path=str(forecast_run.report_path),
                    evaluation={
                        "decision": "accept" if feedback.status == "verified" else "need_user",
                        "summary": state.stop_reason or "P3流程已结束",
                        "warnings": list(feedback.warnings),
                        "aggregate": feedback.aggregate_metrics,
                    },
                    data_fingerprint=forecast_run.data_fingerprint,
                    study_name="山东省级实时电价次日预测",
                    target_name="rt_price",
                    study_start_time=str((state.forecast_plan or {}).get("forecast_start", "")),
                    study_end_time=str((state.forecast_plan or {}).get("forecast_end", "")),
                )
            )
        session.run_id = feedback.forecast_run_id
        session.artifact_directory = str(
            feedback.artifact_directory or (forecast_run.artifact_directory if forecast_run else "")
        )
        session.report_path = str(feedback.report_path or (forecast_run.report_path if forecast_run else ""))
        session.status = "completed"
        self._set_plan_message_state("completed")
        self._add_trace(
            "artifact",
            "完成P3与一次反馈分析",
            "completed",
            state.stop_reason or "全流程已按预算停止",
        )
        self.update_full_flow_state(
            state,
            state_path=session.full_flow_state_path or "full-flow.json",
            output_directory=session.full_flow_output_directory,
        )

    def _full_flow_forecast_failed(self, detail: str) -> None:
        """Keep the persisted failure and frozen plan visible after a P3 worker error."""

        session = self.current_session
        state = None
        if session.full_flow_state_path:
            try:
                state = FullFlowStore(session.full_flow_state_path).load()
            except (OSError, ValueError):
                state = None
        headline = detail.strip().splitlines()[-1] if detail.strip() else "全流程P3执行失败"
        session.status = "failed"
        self._close_latest_running_trace("failed")
        self._set_plan_message_state("failed")
        self._set_active_tool_status("failed", headline)
        self._append_message(SessionMessage(role="assistant", kind="error", content=f"P3没有完成：{headline}"))
        if state is not None:
            self.update_full_flow_state(
                state,
                state_path=session.full_flow_state_path or "full-flow.json",
                output_directory=session.full_flow_output_directory,
            )
        else:
            self._persist_and_render(keep_timeline=True)

    def _forecast_run_completed(self, result: ForecastRunResult) -> None:
        session = self.current_session
        self._complete_active_tool("3折回测和次日96点预测已完成")
        aggregate = {name: metrics.model_dump(mode="json") for name, metrics in result.aggregate.items()}
        self._append_message(
            SessionMessage(
                role="assistant",
                kind="forecast_result",
                content="山东次日实时电价预测完成",
                payload={
                    "run_id": result.run_id,
                    "aggregate": aggregate,
                    "diagnostics": result.diagnostics,
                    "warnings": result.warnings,
                    "prediction_path": str(result.prediction_path),
                    "backtest_path": str(result.backtest_path),
                    "metrics_path": str(result.metrics_path),
                    "report_path": str(result.report_path),
                    "artifact_directory": str(result.artifact_directory),
                    "figure_paths": {name: str(path) for name, path in result.figure_paths.items()},
                    "output_hash": result.output_hash,
                },
            )
        )
        session.run_id = result.run_id
        session.artifact_directory = str(result.artifact_directory)
        session.report_path = str(result.report_path)
        session.runs.append(
            SessionRunRecord(
                run_id=result.run_id,
                run_kind="forecast",
                plan_id=result.plan.plan_id,
                parent_run_id=session.runs[-1].run_id if session.runs else None,
                question=result.plan.question,
                artifact_directory=str(result.artifact_directory),
                report_path=str(result.report_path),
                evaluation={
                    "decision": "accept" if result.baseline_verified else "need_user",
                    "summary": ("三折已验证出预测增益" if result.baseline_verified else "未验证出预测增益"),
                    "warnings": result.warnings,
                    "aggregate": aggregate,
                },
                data_fingerprint=result.plan.data_fingerprint,
                study_name="山东省级实时电价次日预测",
                target_name="rt_price",
                study_start_time=result.plan.forecast_start.isoformat(),
                study_end_time=result.plan.forecast_end.isoformat(),
            )
        )
        session.status = "completed"
        self._set_plan_message_state("completed")
        self._add_trace(
            "artifact",
            "生成预测研究包",
            "completed",
            f"CSV、报告、回测逐点结果和2张SVG；哈希 {result.output_hash[:12]}",
        )
        self._persist_and_render(keep_timeline=True)

    def reject_plan(self) -> None:
        """Reject the plan through the typed Graph command instead of chat text."""

        if self.current_session.read_only:
            return
        if self.is_busy:
            return
        if (self.current_session.current_plan or {}).get("plan_kind") == "forecast":
            self._cancel_plan_feedback_window()
            self._set_plan_message_state("stopped")
            if self.conversation.current_plan_widget is not None:
                self.conversation.current_plan_widget.set_finished("已拒绝")
            self.current_session.current_plan = None
            self.current_session.status = "stopped"
            self._append_message(SessionMessage(role="system", kind="notice", content="已取消这次预测，没有启动训练。"))
            self._add_trace("plan", "预测方案被拒绝", "stopped", "未启动训练")
            self._persist_and_render(keep_timeline=True)
            return
        if not self.agent.has_thread(self.current_session.session_id):
            return
        self._cancel_plan_feedback_window()
        self._set_plan_message_state("stopped")
        if self.conversation.current_plan_widget is not None:
            self.conversation.current_plan_widget.set_finished("已拒绝")
        self._resume_graph(action="reject", task_kind="dialogue")

    def end_current_research(self) -> None:
        """End a limited result explicitly while preserving completed artifacts."""

        if self.current_session.read_only:
            return
        if (
            self.is_busy
            or self._active_interrupt_kind != "result_limitations"
            or not self.agent.has_thread(self.current_session.session_id)
        ):
            return
        self._append_message(SessionMessage(role="user", kind="text", content="结束本轮研究"))
        self._add_trace("user", "结束本轮研究", "completed", "保留当前结果，不再执行后续深入分析")
        self.conversation.set_interaction_context(None)
        self._resume_graph(action="stop", task_kind="dialogue")

    def _resume_graph(
        self,
        *,
        action: str,
        message: str = "",
        task_kind: str = "dialogue",
        foreground_timeout: bool = False,
    ) -> None:
        if self.is_busy:
            return
        session = self.current_session
        session.status = "running" if task_kind == "execute" else "understanding"
        self._start_worker(
            kind=task_kind,
            operation=lambda progress: self.agent.resume(
                session_id=session.session_id,
                action=action,
                message=message,
                interrupt_id=self._active_interrupt_id,
                state_revision=self._active_state_revision,
                foreground_timeout=foreground_timeout,
                progress=progress,
            ),
            success_handler=self._loop_completed,
        )

    def cancel_current_task(self) -> None:
        if self._worker is None:
            return
        self._worker.cancel()
        self.conversation.send_button.setText("正在停止…")
        self.conversation.send_button.setEnabled(False)
        self._add_trace("agent", "收到停止请求", "warning", "将在当前步骤结束后中断")

    def _task_cancelled(self) -> None:
        self._pending_execute_plan = None
        self._pending_forecast_request = None
        self._pending_full_flow_resume = False
        session = self.current_session
        session.pending_forecast_question = None
        if (self._task_kind or "").startswith("full_flow_"):
            resolved = self._full_flow_for_session()
            if resolved is not None:
                flow, state = resolved
                stopped = flow.stop(state, reason="用户取消了当前全流程")
                self._close_latest_running_trace("stopped")
                self._set_active_tool_status("stopped", "用户停止了当前任务")
                self.update_full_flow_state(
                    stopped,
                    state_path=flow.store.path,
                    output_directory=flow.output_directory,
                )
                return
        self._record_forecast_terminal("cancelled", "用户取消运行")
        if self.agent.has_thread(session.session_id):
            self.agent.cancel(session.session_id)
        session.status = "stopped"
        self._close_latest_running_trace("stopped")
        self._set_plan_message_state("stopped")
        self._set_active_tool_status("stopped", "用户停止了当前任务")
        self._append_message(SessionMessage(role="system", kind="notice", content="已停止。"))
        self._add_trace("agent", "研究已终止", "stopped", "本轮分析已按请求中断")
        self._persist_and_render(keep_timeline=True)

    def _task_failed(self, detail: str) -> None:
        self._pending_execute_plan = None
        self._pending_forecast_request = None
        session = self.current_session
        session.pending_forecast_question = None
        self._record_forecast_terminal("failed", detail)
        session.status = "failed"
        self._close_latest_running_trace("failed")
        self._set_plan_message_state("failed")
        if detail.startswith("后台任务已完成，但界面接收结果失败："):
            headline = detail.strip().splitlines()[0]
        else:
            headline = detail.strip().splitlines()[-1] if detail.strip() else "未知错误"
        self._set_active_tool_status("failed", headline)
        self._append_message(SessionMessage(role="assistant", kind="error", content=f"这一步没能完成：{headline}"))
        self._add_trace("error", "本轮研究未完成", "failed", headline)
        self._persist_and_render(keep_timeline=True)

    def _record_forecast_terminal(self, status: str, detail: str) -> None:
        """Keep failed and cancelled forecast experiments beside their frozen inputs."""

        if self._task_kind not in {"forecast_prepare", "forecast_execute", "full_flow_forecast_execute"}:
            return
        value = self.current_session.current_plan or {}
        if value.get("plan_kind") != "forecast":
            return
        try:
            plan = ForecastPlan.model_validate(value)
            root = next(item.path.parent.parent for item in plan.snapshots if item.role == "future")
            destination = root / "provenance" / "run_status.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps(
                    {
                        "plan_id": plan.plan_id,
                        "status": status,
                        "recorded_at": datetime.now(UTC).isoformat(),
                        "detail": detail.strip().splitlines()[-1] if detail.strip() else "",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except (OSError, StopIteration, ValueError):
            return

    def _fail_before_task(self, message: str) -> None:
        self.current_session.status = "failed"
        self._append_message(SessionMessage(role="assistant", kind="error", content=message))
        self._add_trace("error", "数据文件校验失败", "failed", message)
        self._persist_and_render(keep_timeline=True)

    def _start_worker(
        self,
        *,
        kind: str,
        operation: Any,
        success_handler: Any,
        failure_handler: Any | None = None,
    ) -> None:
        self._task_generation += 1
        generation = self._task_generation
        session = self.current_session
        flow_output_directory = session.full_flow_output_directory
        flow_id = None
        if kind.startswith("full_flow_"):
            resolved = self._full_flow_for_session()
            flow_id = resolved[1].flow_id if resolved is not None else None

        def contextual_operation(progress: Callable[[int, str], None]) -> _WorkerResult:
            return _WorkerResult(
                generation=generation,
                session_id=session.session_id,
                task_kind=kind,
                flow_id=flow_id,
                output_directory=flow_output_directory,
                value=operation(progress),
            )

        thread = QThread(self)
        worker = FunctionWorker(contextual_operation)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._task_progress)
        self._task_success_handler = success_handler
        self._task_failure_handler = failure_handler or self._task_failed
        # Always cross the QObject boundary through bound slots owned by this
        # workspace. Connecting a worker signal to a lambda executes that lambda
        # in the worker thread and makes any QWidget work undefined behavior.
        worker.completed.connect(self._worker_completed)
        worker.completed.connect(thread.quit)
        worker.failed.connect(self._worker_failed)
        worker.failed.connect(thread.quit)
        worker.cancelled.connect(self._worker_cancelled)
        worker.cancelled.connect(thread.quit)
        worker.completed.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        worker.cancelled.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._thread_finished)
        self._thread = thread
        self._worker = worker
        self._task_kind = kind
        self._active_task_generation = generation
        self._task_outcome = None
        self._last_progress_message = ""
        self._set_busy(True)
        thread.start()

    @Slot(object)
    def _worker_completed(self, result: object) -> None:
        """Deliver a worker result on the GUI thread before the thread is retired."""

        self._assert_gui_thread()
        if not isinstance(result, _WorkerResult):
            self._task_outcome = "failed"
            self._task_failed("工作线程返回了无法识别的结果包。")
            return
        if (
            result.generation != self._active_task_generation
            or result.session_id != self.current_session.session_id
            or result.task_kind != self._task_kind
        ):
            self._task_outcome = "failed"
            self._task_failed("收到已失效任务的结果，未更新当前会话。")
            return
        if isinstance(result.value, FullFlowState) and result.flow_id != result.value.flow_id:
            self._task_outcome = "failed"
            self._task_failed("全流程结果与启动任务的流程标识不一致，未更新界面。")
            return
        handler = self._task_success_handler
        if handler is not None:
            try:
                handler(result.value)
            except Exception as exc:  # noqa: BLE001 - UI projection must fail visibly
                self._task_outcome = "failed"
                detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                self._task_failed(f"后台任务已完成，但界面接收结果失败：{type(exc).__name__}: {exc}\n{detail}")
                return
        if self._task_outcome is None:
            self._task_outcome = "completed"

    @Slot(str)
    def _worker_failed(self, detail: str) -> None:
        """Deliver a worker failure on the GUI thread."""

        self._assert_gui_thread()
        self._task_outcome = "failed"
        handler = self._task_failure_handler
        if handler is not None:
            try:
                handler(detail)
            except Exception as exc:  # noqa: BLE001 - failure projection must also fail visibly
                self._task_failed(f"界面接收后台失败状态时出错：{type(exc).__name__}: {exc}")

    @Slot()
    def _worker_cancelled(self) -> None:
        """Project cancellation on the GUI thread."""

        self._assert_gui_thread()
        self._task_outcome = "cancelled"
        try:
            self._task_cancelled()
        except Exception as exc:  # noqa: BLE001 - cancellation projection must not disappear
            self._task_outcome = "failed"
            self._task_failed(f"界面接收停止状态时出错：{type(exc).__name__}: {exc}")

    def _assert_gui_thread(self) -> None:
        if QThread.currentThread() != self.thread():
            raise RuntimeError("桌面状态投影只能在GUI主线程执行")

    def _task_progress(self, value: int, message: str) -> None:
        session = self.current_session
        source_event, step = split_progress_message(message)
        del value
        session.status = {
            "setup": "inspecting_data",
            "submit": "understanding",
            "read": "understanding",
            "route": "understanding",
            "goal": "understanding",
            "design": "understanding",
            "confirm": "understanding",
            "compute": "running",
            "review": "evaluating",
            "answer": "evaluating",
            "issue": "understanding",
        }.get(
            step.stage,
            "running" if self._task_kind in {"execute", "forecast_execute"} else "understanding",
        )  # type: ignore[assignment]
        if message != self._last_progress_message:
            self._append_thinking_step(step.stage, step.title, step.detail, step.function_name)
            self._close_latest_running_trace("completed")
            # Store the raw loop event so the trace panel narrates it exactly once.
            self._add_trace(trace_category(step.stage), source_event or step.title, "running", step.detail)
            self._last_progress_message = message
        elif self._thinking_widget is not None:
            self._thinking_widget.update_current(detail=step.detail)
            self.conversation.scroll_to_bottom()
        self.conversation.set_status(session.status)
        self.status_changed.emit(step.title)

    def _set_trace_maximized(self, maximized: bool) -> None:
        if maximized:
            self._pre_trace_sizes = self.sizes()
            self.history.hide()
            self.conversation.hide()
            self.setSizes([0, 0, max(self.width(), 1)])
            return
        self.history.show()
        self.conversation.show()
        self.setSizes(self._pre_trace_sizes or [236, 760, 336])
        self._pre_trace_sizes = None

    def _thread_finished(self) -> None:
        pending_plan = self._pending_execute_plan
        self._pending_execute_plan = None
        pending_forecast = self._pending_forecast_request
        self._pending_forecast_request = None
        pending_full_flow = self._pending_full_flow_resume
        self._pending_full_flow_resume = False
        self._thread = None
        self._worker = None
        self._task_kind = None
        self._active_task_generation = None
        self._task_success_handler = None
        self._task_failure_handler = None
        if self._thinking_widget is not None and self._task_outcome == "completed":
            self._set_active_tool_status("completed", "")
        self._task_outcome = None
        self.conversation.send_button.setEnabled(True)
        self._set_busy(False)
        self._persist_and_render(keep_timeline=True)
        if pending_plan is not None:
            QTimer.singleShot(0, lambda plan=pending_plan: self.run_plan(plan))
        elif pending_full_flow:
            QTimer.singleShot(0, self._resume_full_flow)
        elif pending_forecast:
            if _is_news_analysis_request(pending_forecast):
                QTimer.singleShot(
                    0,
                    lambda question=pending_forecast: self._submit_news_analysis_request(
                        question,
                        prepare_forecast=True,
                        record_user_message=False,
                    ),
                )
            else:
                QTimer.singleShot(
                    0,
                    lambda question=pending_forecast: self._submit_forecast_request(
                        question,
                        record_user_message=False,
                    ),
                )

    def _set_busy(self, busy: bool) -> None:
        self.history.set_busy(busy)
        self.context.set_busy(busy)
        self.conversation.set_running(busy)
        self.busy_changed.emit(busy)

    def _append_message(self, message: SessionMessage) -> QWidget:
        session = self.current_session
        session.messages.append(message)
        session.touch()
        widget = self.conversation.append_message(message)
        self.store.save(self.sessions)
        self.history.set_sessions(self.sessions, self.current_session_id)
        return widget

    def _add_trace(self, category: str, name: str, status: str, summary: str) -> None:
        event = TraceEvent(category=category, name=name, status=status, summary=summary)  # type: ignore[arg-type]
        self.current_session.trace.append(event)
        self.current_session.touch()
        self.context.trace.add_event(event)

    def _start_thinking(self, stage: str, title: str, detail: str) -> None:
        """Open a live, collapsible view of what the Agent is doing this turn."""

        message = SessionMessage(
            role="assistant",
            kind="thinking",
            content="研究过程",
            payload={"steps": [], "state": "running"},
        )
        widget = self._append_message(message)
        self._thinking_widget = widget if isinstance(widget, ThinkingMessageWidget) else None
        self._thinking_message_id = message.message_id
        self._thinking_function = None
        self._append_thinking_step(stage, title, detail)
        self._thinking_placeholder = True

    def _append_thinking_step(
        self,
        stage: str,
        title: str,
        detail: str,
        function_name: str | None = None,
    ) -> None:
        if self._thinking_widget is None:
            return
        arguments = {
            "stage": STAGE_LABELS.get(stage, "研究"),
            "title": title,
            "detail": detail,
            "status": "running",
            "function_name": function_name,
        }
        same_function = bool(function_name) and function_name == self._thinking_function
        if self._thinking_placeholder or same_function:
            self._thinking_widget.replace_current(**arguments, keep_detail=same_function)
        else:
            self._thinking_widget.add_step(**arguments)
        self._thinking_placeholder = False
        self._thinking_function = function_name
        self._store_thinking_steps()
        self.conversation.scroll_to_bottom()

    def _store_thinking_steps(self, state: str | None = None) -> None:
        if not self._thinking_message_id or self._thinking_widget is None:
            return
        message = next(
            (item for item in self.current_session.messages if item.message_id == self._thinking_message_id),
            None,
        )
        if message is None:
            return
        message.payload["steps"] = self._thinking_widget.steps
        if state is not None:
            message.payload["state"] = state

    def _complete_active_tool(self, detail: str) -> None:
        self._set_active_tool_status("completed", detail)
        self._close_latest_running_trace("completed")

    def _set_active_tool_status(self, status: str, detail: str) -> None:
        """Close the visible thinking process for this turn."""

        if self._thinking_widget is None:
            return
        state = status if status in {"failed", "stopped"} else "completed"
        if state != "completed":
            self._thinking_widget.update_current(detail=detail, status=state)
        self._thinking_widget.finish(state=state, summary=f"{len(self._thinking_widget.steps)} 步")
        self._store_thinking_steps(state)
        self._thinking_widget = None
        self._thinking_message_id = None

    def _close_latest_running_trace(self, status: str) -> None:
        event = next((item for item in reversed(self.current_session.trace) if item.status == "running"), None)
        if event is not None:
            event.status = status  # type: ignore[assignment]
            self.context.trace.set_events(self.current_session.trace)

    def _set_plan_message_state(
        self,
        state: str,
        *,
        plan: EDAPlan | ForecastPlan | None = None,
    ) -> None:
        message = next(
            (
                item
                for item in reversed(self.current_session.messages)
                if item.kind in {"plan", "data_plan", "forecast_plan"}
            ),
            None,
        )
        if message is None:
            return
        message.payload["state"] = state
        if plan is not None:
            message.payload["plan"] = plan.model_dump(mode="json")

    @staticmethod
    def _agent_conversation(
        session: ResearchSession,
        *,
        exclude_message_id: str | None = None,
    ) -> list[ConversationMessage]:
        """Expose the complete durable text transcript as the recall source."""

        return [
            ConversationMessage(
                message_id=message.message_id,
                turn_id=message.turn_id,
                episode_id=message.episode_id,
                role=message.role,
                content=message.content,
                created_at=message.created_at,
            )
            for message in session.messages
            if message.kind == "text"
            and message.role in {"user", "assistant"}
            and message.content
            and message.message_id != exclude_message_id
        ]

    def _persist_and_render(self, *, keep_timeline: bool = False) -> None:
        self.current_session.touch()
        self.store.save(self.sessions)
        self.history.set_sessions(self.sessions, self.current_session_id)
        self._render_region_selector()
        if keep_timeline:
            self.conversation.title_label.setText(self.current_session.title)
            self.conversation.set_status(self.current_session.status)
            self.context.set_session(self.current_session)
        else:
            self._render_current()

    def _render_current(self) -> None:
        session = self.current_session
        self._render_region_selector()
        self.conversation.set_session(session)
        self.context.set_session(session)
        self.history.set_sessions(self.sessions, self.current_session_id)
        self.status_changed.emit(STATUS_LABELS.get(session.status, session.status))

    def _render_region_selector(self) -> None:
        self.context.data_panel.set_regions(
            [(profile.region_id, profile.label) for profile in self.region_profiles.values()],
            self.current_session.region_id,
        )
