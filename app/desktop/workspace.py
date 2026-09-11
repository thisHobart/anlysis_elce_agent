"""Controller for the unified three-pane research conversation workspace."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from math import ceil
from pathlib import Path
from typing import Any
from uuid import uuid4

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import QFileDialog, QMessageBox, QSplitter, QWidget

from app.desktop.input_config import (
    FILE_FILTERS,
    build_runtime_study,
    parse_chat_time_range,
    validate_input_path,
)
from app.desktop.message_widgets import DataDetailsDialog, ThinkingMessageWidget
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
from app.research.graph.contracts import ApprovalState, EpisodeSummary, ResearchLoopSnapshot
from app.research.graph.narration import (
    STAGE_LABELS,
    narrate_event,
    split_progress_message,
    trace_category,
)
from app.research.schemas.results import DataQualityReport
from app.research.schemas.study import StudyConfig
from app.runtime_paths import default_research_output_directory


def _known_so_far(config: StudyConfig | None) -> DataSummary | None:
    """Describe the dataset from the study alone, before any of it has been read."""

    if config is None:
        return None
    return build_partial_summary(
        market=config.study.market,
        target_name=config.target.name,
        exogenous_names=[spec.name for spec in config.exogenous],
        frequency=config.study.frequency,
    )


DATA_CHANGED_NOTICE = "数据换了，之前的分析方案已经作废。重新问一次，我按新数据给方案。"


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
            (self.store.path.parent / "research").resolve()
            if provided_store
            else default_research_output_directory()
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
        self.agent = agent or ResearchCoordinator(
            checkpoint_path=self.store.path.with_name("research_graph.sqlite3")
        )
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
        self._pre_trace_sizes: list[int] | None = None
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
        profile = self.region_profiles.get(session.region_id)
        if profile is None and self.region_profiles:
            profile = next(iter(self.region_profiles.values()))
        if profile is not None:
            session.region_id = profile.region_id
            session.region_label = profile.label
            session.region_market = profile.market
            session.region_timezone = profile.timezone
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
            session.trace.append(
                TraceEvent(category="session", name="历史会话恢复", status="warning", summary=notice)
            )
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
        self.agent = ResearchCoordinator(
            checkpoint_path=self.store.path.with_name("research_graph.sqlite3")
        )
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
        if any(item.session_id == session_id for item in self.sessions):
            self._cancel_plan_feedback_window()
            self.current_session_id = session_id
            self._render_current()
            if self.agent.has_thread(session_id):
                self._loop_completed(self.agent.get_snapshot(session_id))
            elif self.current_session.status == "awaiting_plan_approval" and self.current_session.current_plan:
                self.current_session.plan_feedback_deadline = None
                self.current_session.plan_feedback_remaining_seconds = 0
                if self.conversation.current_plan_widget is not None:
                    self.conversation.current_plan_widget.set_feedback_paused("需要你重新确认")

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

        if self.is_busy:
            return
        session = self.current_session
        if session.source_kind == "database":
            self.select_region(session.region_id)
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

        if self.is_busy:
            return
        self.context.data_panel.open_region_menu()

    def retry_dataset(self) -> None:
        """Try the data source again without touching what was asked for."""

        if self.is_busy:
            return
        session = self.current_session
        if session.source_kind == "database":
            self.select_region(session.region_id)
            return
        if session.can_analyze:
            self._add_trace("input", "重新连接数据", "completed", "已改用本地数据")
            self._set_data_state("ready" if session.data_summary else "empty", session.data_summary)
        else:
            self._add_trace("input", "重新连接数据", "warning", "仍然连不上数据服务器")
            self._set_data_state("unavailable", session.data_summary)
        self._persist_and_render(keep_timeline=True)

    def select_region(self, region_id: str) -> None:
        """Select and fetch the regional actual-price source for this conversation."""

        if self.is_busy:
            return
        profile = self.region_profiles.get(region_id)
        if profile is None:
            QMessageBox.warning(self, "这个地区暂时不可用", "地区配置已经变化，请重新打开应用。")
            return
        session = self.current_session
        changed_source = session.source_kind != "database" or session.region_id != profile.region_id
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
            f"forecast_{name}" if name in actual_candidates else name
            for name in profile.weather_columns
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
        anchor = (session.current_plan or {}).get("data_fingerprint") or session.dataset_fingerprint
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
        if self.is_busy:
            return
        session = self.current_session
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
            changed_window = (
                session.analysis_start_time != new_start
                or session.analysis_end_time != new_end
            )
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
        session.status = "understanding"
        if session.can_analyze:
            # Light up what is already known so the panel fills in as it learns,
            # rather than sitting blank until the whole answer arrives.
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

    def _legacy_graph_import(self, session: ResearchSession) -> dict[str, Any]:
        latest_run = None
        if session.run_id:
            latest_run = {
                "run_id": session.run_id,
                "plan_id": (session.current_plan or {}).get("plan_id"),
                "artifact_directory": session.artifact_directory,
                "report_path": session.report_path,
                "figure_paths": {},
                "evaluation": session.latest_evaluation,
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
            "eda_summary": session.latest_eda_summary,
            "evaluation": session.latest_evaluation,
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
        if latest and latest.get("run_id") and not any(item.run_id == latest["run_id"] for item in session.runs):
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
                        str(study_definition["start_time"])
                        if study_definition.get("start_time") is not None
                        else None
                    ),
                    study_end_time=(
                        str(study_definition["end_time"])
                        if study_definition.get("end_time") is not None
                        else None
                    ),
                    skill_name=plan_value.get("skill_name"),
                    skill_version=plan_value.get("skill_version"),
                )
            )
            self._set_plan_message_state("completed")

        if interrupt_payload and interrupt_payload.kind in {
            "plan_error",
            "result_limitations",
            "result_rejected",
            "response_error",
            "finalization_error",
        }:
            feedback_codes = {
                str(item.get("code") or "") for item in values.get("feedback_packets", [])
            }
            if interrupt_payload.kind == "plan_error":
                if "session_skill_version_mismatch" in feedback_codes:
                    notice = (
                        f"{interrupt_payload.message}\n"
                        "请点击左侧“新建研究”重新提交问题；旧对话和此前报告仍可查看。"
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
                    f"{interrupt_payload.message}\n"
                    "在下方输入下一步研究要求，例如“按推荐变量继续”，或说明要增删的变量。"
                )
            recent_notices = {
                message.content for message in session.messages[-8:] if message.role == "system" and message.kind == "notice"
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
        sequenced = [
            (int(item["sequence"]), item)
            for item in events
            if isinstance(item.get("sequence"), int)
        ]
        if sequenced:
            new_events = [item for sequence, item in sequenced if sequence > session.graph_event_sequence]
        else:
            new_events = events[session.graph_event_count :]
        valid_categories = {"session", "user", "agent", "input", "plan", "tool", "evaluation", "artifact", "error"}
        for item in new_events:
            status = str(item.get("status", "info"))
            trace_status = status if status in {"info", "running", "completed", "warning", "failed", "stopped"} else "info"
            details = dict(item.get("details", {}))
            name = str(item.get("name", "研究循环事件"))
            category = str(item.get("category", ""))
            if category not in valid_categories:
                category = (
                    "tool"
                    if any(word in name for word in ("工具", "函数"))
                    else "evaluation" if "评估" in name else "plan" if "方案" in name else "agent"
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
        if not self.auto_execute_plan or session.status != "awaiting_plan_approval" or not self._plan_feedback_timer.isActive():
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
        if self.auto_execute_plan and session.status == "awaiting_plan_approval" and session.current_plan and not self.is_busy:
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

    def run_plan(self, approved_plan: EDAPlan) -> None:
        if self.is_busy:
            return
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
                lambda progress: self.agent.resume(
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
            ),
            success_handler=self._loop_completed,
        )
        self.conversation.scroll_to_bottom()

    def reject_plan(self) -> None:
        """Reject the plan through the typed Graph command instead of chat text."""

        if self.is_busy or not self.agent.has_thread(self.current_session.session_id):
            return
        self._cancel_plan_feedback_window()
        self._set_plan_message_state("stopped")
        if self.conversation.current_plan_widget is not None:
            self.conversation.current_plan_widget.set_finished("已拒绝")
        self._resume_graph(action="reject", task_kind="dialogue")

    def end_current_research(self) -> None:
        """End a limited result explicitly while preserving completed artifacts."""

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
        session = self.current_session
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
        session = self.current_session
        session.status = "failed"
        self._close_latest_running_trace("failed")
        self._set_plan_message_state("failed")
        headline = detail.strip().splitlines()[-1] if detail.strip() else "未知错误"
        self._set_active_tool_status("failed", headline)
        self._append_message(SessionMessage(role="assistant", kind="error", content=f"这一步没能完成：{headline}"))
        self._add_trace("error", "本轮研究未完成", "failed", headline)
        self._persist_and_render(keep_timeline=True)

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
        thread = QThread(self)
        worker = FunctionWorker(operation)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._task_progress)
        worker.completed.connect(success_handler)
        worker.completed.connect(thread.quit)
        worker.failed.connect(failure_handler or self._task_failed)
        worker.failed.connect(thread.quit)
        worker.cancelled.connect(self._task_cancelled)
        worker.cancelled.connect(thread.quit)
        worker.completed.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        worker.cancelled.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._thread_finished)
        self._thread = thread
        self._worker = worker
        self._task_kind = kind
        self._last_progress_message = ""
        self._set_busy(True)
        thread.start()

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
        }.get(step.stage, "running" if self._task_kind == "execute" else "understanding")  # type: ignore[assignment]
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
        self._thread = None
        self._worker = None
        self._task_kind = None
        if self._thinking_widget is not None:
            self._set_active_tool_status("completed", "")
        self.conversation.send_button.setEnabled(True)
        self._set_busy(False)
        self._persist_and_render(keep_timeline=True)
        if pending_plan is not None:
            QTimer.singleShot(0, lambda plan=pending_plan: self.run_plan(plan))

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

    def _set_plan_message_state(self, state: str, *, plan: EDAPlan | None = None) -> None:
        message = next(
            (item for item in reversed(self.current_session.messages) if item.kind in {"plan", "data_plan"}),
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
