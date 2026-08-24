"""Controller for the unified three-pane research conversation workspace."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from math import ceil
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import QMessageBox, QSplitter, QWidget

from app.desktop.input_config import build_session_study_config, discover_inputs_from_config, validate_input_path
from app.desktop.message_widgets import ToolMessageWidget
from app.desktop.panes import STATUS_LABELS, ContextPane, ConversationPane, HistoryPane
from app.desktop.session import (
    ResearchSession,
    SessionMessage,
    SessionRunRecord,
    SessionStore,
    TraceEvent,
    VariableEvidence,
)
from app.desktop.worker import FunctionWorker
from app.research.agent.schemas import (
    AgentRunResult,
    ConversationMessage,
    EDAPlan,
    ResearchDataProfile,
    ResearchProposal,
    ResearchTurnResult,
)
from app.research.application.coordinator import ResearchCoordinator
from app.research.graph.contracts import ApprovalState, ResearchLoopSnapshot
from app.research.schemas.results import DataQualityReport


class ResearchWorkspace(QSplitter):
    """Single source of UI truth for history, conversation, inputs, plan, and trace."""

    busy_changed = Signal(bool)
    status_changed = Signal(str)

    def __init__(
        self,
        *,
        config_path: Path | None = None,
        agent: ResearchCoordinator | None = None,
        store: SessionStore | None = None,
        plan_feedback_seconds: int = 30,
    ) -> None:
        super().__init__(Qt.Orientation.Horizontal)
        self.setObjectName("researchWorkspace")
        self.store = store or SessionStore()
        self._owns_agent = agent is None
        self.agent = agent or ResearchCoordinator(
            checkpoint_path=self.store.path.with_name("research_graph.sqlite3")
        )
        self.plan_feedback_seconds = max(1, int(plan_feedback_seconds))
        self.sessions = [session for session in self.store.load() if not self._is_blank_session(session)]
        self.current_session_id: str | None = None
        self._thread: QThread | None = None
        self._worker: FunctionWorker | None = None
        self._task_kind: str | None = None
        self._task_session_id: str | None = None
        self._success_handler: Any | None = None
        self._active_tool_widget: ToolMessageWidget | None = None
        self._active_tool_message_id: str | None = None
        self._last_progress_message = ""
        self._task_previous_status = "idle"
        self._pending_execute_plan: EDAPlan | None = None
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
        self.conversation.draft_changed.connect(self._composer_draft_changed)
        self.context.file_selected.connect(self.set_input_file)
        self.context.file_cleared.connect(self.clear_input_file)
        self.context.plan_activated.connect(self.conversation.scroll_to_plan)

        session = self._new_session()
        self.sessions.insert(0, session)
        self.current_session_id = session.session_id
        if config_path is not None:
            self._apply_config_to_session(self.current_session, config_path, show_errors=False)
        self._persist_and_render()

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
                    "研究讨论和方案规划均由已配置的大模型处理。你可以直接提出问题，也可以在右侧“文件输入”"
                    "中按需选择目标电价、实际变量、预测变量或 YAML 配置；执行统计分析时需要目标电价数据。"
                ),
            )
        )
        session.trace.append(
            TraceEvent(category="session", name="创建研究会话", status="completed", summary="新会话已就绪")
        )
        skill_errors = list(getattr(self.agent, "skill_load_errors", []))
        if skill_errors:
            summary = "；".join(skill_errors)
            session.messages.append(
                SessionMessage(
                    role="assistant",
                    kind="error",
                    content=f"部分外部 Skill 加载失败，相关能力已停用：{summary}",
                )
            )
            session.trace.append(
                TraceEvent(
                    category="error",
                    name="外部 Skill 加载失败",
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
                content="大模型配置已更新；新的研究问题将使用当前配置进行方案规划。",
            )
        )
        self._add_trace("session", "更新大模型配置", "completed", "研究 Agent 已重新加载模型配置")
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
                    self.conversation.current_plan_widget.set_feedback_paused("旧审批需重新确认")

    def rename_session(self, session_id: str, title: str) -> None:
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
            "删除会话",
            f"确定删除“{session.title}”吗？研究产物不会被删除。",
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

    def set_input_file(self, role: str, path: str) -> None:
        if self.is_busy:
            return
        self._cancel_plan_feedback_window()
        try:
            resolved = validate_input_path(role, path)  # type: ignore[arg-type]
            if role == "config":
                self._apply_config_to_session(self.current_session, resolved, show_errors=True)
            else:
                item = self.current_session.inputs[role]  # type: ignore[index]
                if item.path and Path(item.path).resolve() == resolved:
                    return
                item.path = str(resolved)
                item.status = "selected"
                item.detail = "等待 Agent 校验"
                item.variables = []
                self._invalidate_plan_for_input_change()
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "无法使用文件", str(exc))
            return
        self._add_trace("input", f"选择 {Path(path).name}", "completed", "用户更新了文件输入")
        self._persist_and_render()

    def clear_input_file(self, role: str) -> None:
        if self.is_busy:
            return
        self._cancel_plan_feedback_window()
        item = self.current_session.inputs[role]  # type: ignore[index]
        if not item.path:
            return
        filename = Path(item.path).name
        item.path = ""
        item.status = "empty"
        item.detail = "尚未选择"
        item.variables = []
        self._invalidate_plan_for_input_change()
        self._add_trace("input", f"清除 {filename}", "completed", "用户移除了可选文件输入")
        self._persist_and_render()

    def _apply_config_to_session(self, session: ResearchSession, path: Path, *, show_errors: bool) -> bool:
        try:
            discovered = discover_inputs_from_config(path)
        except (OSError, ValueError) as exc:
            if show_errors:
                QMessageBox.warning(self, "研究配置无效", str(exc))
            return False
        for role, resolved in discovered.items():
            item = session.inputs[role]
            item.path = str(resolved)
            item.status = "selected"
            item.detail = "等待 Agent 校验"
            item.variables = []
        self._invalidate_plan_for_input_change()
        return True

    def _invalidate_plan_for_input_change(self) -> None:
        self._cancel_plan_feedback_window()
        session = self.current_session
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
                    content="文件输入已变更，原分析方案已失效，请重新向 Agent 提问或生成方案。",
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
        self._cancel_plan_feedback_window()
        session.derive_title(question)
        self._append_message(SessionMessage(role="user", kind="text", content=question))
        self._add_trace("user", "收到研究问题", "completed", question)
        study_config = None
        if session.can_analyze:
            try:
                study_config = build_session_study_config(session)
            except (OSError, ValueError) as exc:
                self._fail_before_task(str(exc))
                return
        self._task_previous_status = session.status
        session.status = "understanding"
        tool_message = SessionMessage(
            role="assistant",
            kind="tool",
            content="理解研究对话",
            payload={"title": "研究 Agent", "detail": "理解问题与当前研究上下文", "status": "running"},
        )
        widget = self._append_message(tool_message)
        self._active_tool_widget = widget if isinstance(widget, ToolMessageWidget) else None
        self._active_tool_message_id = tool_message.message_id
        self._add_trace("agent", "理解研究意图", "running", "路由讨论、规划、修订、执行或证据解释")
        conversation = self._agent_conversation(session)[:-1]
        imported_state = self._legacy_graph_import(session) if not self.agent.has_thread(session.session_id) else None
        self._start_worker(
            kind="dialogue",
            operation=lambda progress: self.agent.submit_user_message(
                session_id=session.session_id,
                message=question,
                study_config=study_config,
                conversation=conversation,
                approval_timeout_seconds=self.plan_feedback_seconds,
                imported_state=imported_state,
                progress=progress,
            ),
            success_handler=self._loop_completed,
        )

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
                "eda_summary": session.latest_eda_summary,
                "quality_report": session.quality_report,
            }
        return {
            "current_plan": session.current_plan,
            "plan_history": [session.current_plan] if session.current_plan else [],
            "data_profile": session.data_profile,
            "quality_report": session.quality_report,
            "data_fingerprint": (session.current_plan or {}).get("data_fingerprint"),
            "eda_summary": session.latest_eda_summary,
            "evaluation": session.latest_evaluation,
            "latest_run": latest_run,
            "run_history": [latest_run] if latest_run else [],
            "phase": "completed" if session.run_id else "idle",
        }

    def _loop_completed(self, snapshot: ResearchLoopSnapshot) -> None:
        session = self.current_session
        self._complete_active_tool("研究循环已到达用户交互点")
        values = snapshot.values
        self._sync_graph_events(session, snapshot.events)
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
                    if message.kind == "plan" and message.payload.get("plan", {}).get("plan_id") == plan.plan_id
                ),
                None,
            )
            if existing is None:
                if any(message.kind == "plan" for message in session.messages):
                    self._set_plan_message_state("stale")
                profile = values.get("data_profile") or {}
                quality = DataQualityReport.model_validate(values["quality_report"])
                proposal = ResearchProposal(
                    plan=plan,
                    assistant_message="模型方案已通过确定性校验，等待你的修改或确认。",
                    data_profile=ResearchDataProfile.model_validate(profile),
                    quality_report=quality,
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
            session.runs.append(
                SessionRunRecord(
                    run_id=latest["run_id"],
                    plan_id=latest.get("plan_id") or (session.current_plan or {}).get("plan_id", "unknown"),
                    parent_run_id=session.runs[-1].run_id if session.runs else None,
                    question=(session.current_plan or {}).get("question", "研究循环"),
                    artifact_directory=latest.get("artifact_directory") or "",
                    report_path=latest.get("report_path") or "",
                    evaluation=evaluation,
                )
            )
            self._set_plan_message_state("completed")

        assistant_message = values.get("assistant_message", "").strip()
        if assistant_message and not any(
            message.role == "assistant" and message.kind == "text" and message.content == assistant_message
            for message in session.messages[-3:]
        ):
            self._append_message(SessionMessage(role="assistant", kind="text", content=assistant_message))

        if interrupt_payload and interrupt_payload.kind == "need_user":
            session.status = "awaiting_user"
            notice = (
                f"{interrupt_payload.message}\n"
                "你可以回复“接受当前限制”、直接提出修改意见，或回复“停止”。"
            )
            if not session.messages or session.messages[-1].content != notice:
                self._append_message(SessionMessage(role="system", kind="notice", content=notice))
        elif interrupt_payload and interrupt_payload.kind == "result":
            session.status = "completed"
        elif snapshot.phase in {"failed", "stopped", "completed"}:
            session.status = snapshot.phase  # type: ignore[assignment]
        else:
            session.status = "idle"
        self._persist_and_render(keep_timeline=True)

    def _sync_graph_events(self, session: ResearchSession, events: list[dict[str, Any]]) -> None:
        new_events = events[session.graph_event_count :]
        for item in new_events:
            status = str(item.get("status", "info"))
            trace_status = status if status in {"info", "running", "completed", "warning", "failed", "stopped"} else "info"
            details = dict(item.get("details", {}))
            name = str(item.get("name", "研究循环事件"))
            session.trace.append(
                TraceEvent(
                    category="tool" if "工具" in name else "agent",
                    name=name,
                    status=trace_status,  # type: ignore[arg-type]
                    duration_ms=float(details["duration_ms"]) if details.get("duration_ms") is not None else None,
                    summary=str(details),
                    details=details,
                )
            )
        session.graph_event_count = len(events)

    def _restore_approval_timer(self, snapshot: ResearchLoopSnapshot) -> None:
        session = self.current_session
        approval = ApprovalState.model_validate(snapshot.values.get("approval_state", {}))
        session.plan_feedback_deadline = approval.deadline
        session.plan_feedback_remaining_seconds = approval.remaining_seconds
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
                    self.conversation.current_plan_widget.set_feedback_paused("审批已过期，请明确确认")
                return
            self._start_plan_feedback_window(remaining, persist_graph=False)

    def _turn_completed(self, turn: ResearchTurnResult) -> None:
        session = self.current_session
        if turn.action == "proposal":
            if turn.proposal is None:  # pragma: no cover - guarded by ResearchTurnResult
                raise RuntimeError("proposal turn is missing proposal payload")
            if self._task_previous_status == "awaiting_plan_approval" and session.current_plan is not None:
                self._set_plan_message_state("stale")
            self._proposal_completed(turn.proposal)
            return
        self._complete_active_tool("研究意图处理完成")
        if turn.action == "reply":
            self._append_message(SessionMessage(role="assistant", kind="text", content=turn.assistant_message))
            session.status = self._restored_dialogue_status()
            if session.status == "awaiting_plan_approval" and session.current_plan is not None:
                self._start_plan_feedback_window()
            self._add_trace("agent", "完成研究回复", "completed", f"响应方式：{turn.responder}")
            self._persist_and_render(keep_timeline=True)
            return
        if turn.action == "plan_revision":
            if turn.revised_plan is None:  # pragma: no cover - guarded by ResearchTurnResult
                raise RuntimeError("revision turn is missing a plan")
            self._set_plan_message_state("stale")
            self._append_message(SessionMessage(role="assistant", kind="text", content=turn.assistant_message))
            self._append_plan_message(turn.revised_plan, turn.available_variables)
            session.current_plan = turn.revised_plan.model_dump(mode="json")
            session.plan_stale = False
            session.status = "awaiting_plan_approval"
            self._start_plan_feedback_window()
            self._add_trace(
                "plan",
                f"修订 EDA 方案 v{turn.revised_plan.revision}",
                "completed",
                turn.revised_plan.revision_reason or "用户通过对话修订方案",
            )
            self._persist_and_render(keep_timeline=True)
            return
        if turn.revised_plan is None:  # pragma: no cover - guarded by ResearchTurnResult
            raise RuntimeError("execution confirmation is missing a plan")
        plan_to_run = turn.revised_plan
        if self.conversation.current_plan_widget is not None:
            try:
                plan_to_run = self.conversation.current_plan_widget.approved_plan()
            except ValueError as exc:
                self._append_message(SessionMessage(role="assistant", kind="error", content=str(exc)))
                session.status = "awaiting_plan_approval"
                self._persist_and_render(keep_timeline=True)
                return
        self._append_message(SessionMessage(role="assistant", kind="text", content=turn.assistant_message))
        session.status = "awaiting_plan_approval"
        self._pending_execute_plan = plan_to_run
        self._add_trace("plan", "通过对话确认分析方案", "completed", plan_to_run.plan_id)
        self._persist_and_render(keep_timeline=True)

    def _proposal_completed(self, proposal: ResearchProposal) -> None:
        session = self.current_session
        self._complete_active_tool("数据检查完成")
        self._apply_quality_to_inputs(proposal)
        self._append_message(SessionMessage(role="assistant", kind="text", content=proposal.assistant_message))
        self._append_plan_message(proposal.plan, proposal.data_profile.exogenous_names)
        session.current_plan = proposal.plan.model_dump(mode="json")
        session.plan_stale = False
        session.data_profile = proposal.data_profile.model_dump(mode="json")
        session.quality_report = proposal.quality_report.model_dump(mode="json")
        session.status = "awaiting_plan_approval"
        self._start_plan_feedback_window()
        self._add_trace(
            "plan",
            "生成 EDA 推荐方案",
            "completed",
            f"{proposal.plan.objective}；Skill {proposal.plan.skill_name}@{proposal.plan.skill_version}",
        )
        self._persist_and_render(keep_timeline=True)

    def _append_plan_message(self, plan: EDAPlan, available_variables: list[str]) -> None:
        self._append_message(
            SessionMessage(
                role="assistant",
                kind="plan",
                content="Agent 推荐方案",
                payload={
                    "plan": plan.model_dump(mode="json"),
                    "available_variables": available_variables,
                    "state": "awaiting",
                },
            )
        )

    def _restored_dialogue_status(self) -> str:
        if self._task_previous_status in {"awaiting_plan_approval", "completed"}:
            return self._task_previous_status
        return "idle"

    def _start_plan_feedback_window(self, seconds: int | None = None, *, persist_graph: bool = False) -> None:
        session = self.current_session
        if session.status != "awaiting_plan_approval" or session.current_plan is None:
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
        if session.status != "awaiting_plan_approval" or session.current_plan is None or self.is_busy:
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
        if session.status != "awaiting_plan_approval" or not self._plan_feedback_timer.isActive():
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
        if session.status == "awaiting_plan_approval" and session.current_plan and not self.is_busy:
            self._start_plan_feedback_window(session.plan_feedback_remaining_seconds)

    def _plan_feedback_tick(self) -> None:
        session = self.current_session
        if session.status != "awaiting_plan_approval" or session.current_plan is None:
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
                content=f"{self.plan_feedback_seconds} 秒内未收到修改意见，已锁定大模型方案并开始执行。",
            )
        )
        self._add_trace("plan", "方案反馈窗口结束", "completed", "未收到用户修改，自动锁定并执行")
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
        if session.inputs["config"].path:
            session.inputs["config"].status = "ready"
            session.inputs["config"].detail = "配置解析完成"
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
                content="已按你确认的步骤、变量和参数开始分析。运行过程会同步记录在右侧轨迹中。",
            )
        )
        tool_message = SessionMessage(
            role="assistant",
            kind="tool",
            content="执行确认方案",
            payload={"title": "执行确认的 EDA 方案", "detail": "等待工具执行", "status": "running"},
        )
        widget = self._append_message(tool_message)
        self._active_tool_widget = widget if isinstance(widget, ToolMessageWidget) else None
        self._active_tool_message_id = tool_message.message_id
        self._add_trace("plan", "用户确认分析方案", "completed", f"{len(approved_plan.enabled_steps)} 个步骤")
        self._start_worker(
            kind="execute",
            operation=(
                lambda progress: self.agent.resume(
                    session_id=session.session_id,
                    action="approve",
                    progress=progress,
                )
                if self.agent.has_thread(session.session_id)
                else self.agent.submit_user_message(
                    session_id=session.session_id,
                    message="按这个执行",
                    study_config=build_session_study_config(session),
                    conversation=self._agent_conversation(session),
                    approval_timeout_seconds=self.plan_feedback_seconds,
                    imported_state=self._legacy_graph_import(session),
                    progress=progress,
                )
            ),
            success_handler=self._loop_completed,
        )

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
                foreground_timeout=foreground_timeout,
                progress=progress,
            ),
            success_handler=self._loop_completed,
        )

    def _execution_completed(self, result: AgentRunResult) -> None:
        session = self.current_session
        parent_run_id = session.run_id
        self._complete_active_tool("分析与研究包生成完成")
        if self.conversation.current_plan_widget is not None:
            self.conversation.current_plan_widget.set_finished()
        self._set_plan_message_state("completed")
        result_payload = {
            "run_id": result.run_id,
            "report_path": str(result.report_path),
            "artifact_directory": str(result.artifact_directory),
            "figure_count": len(result.figure_paths),
            "evaluation": result.evaluation.model_dump(mode="json"),
        }
        self._append_message(
            SessionMessage(role="assistant", kind="result", content=result.evaluation.summary, payload=result_payload)
        )
        session.run_id = result.run_id
        session.artifact_directory = str(result.artifact_directory)
        session.report_path = str(result.report_path)
        session.quality_report = result.quality_report.model_dump(mode="json")
        session.latest_eda_summary = result.eda_summary
        session.latest_evaluation = result.evaluation.model_dump(mode="json")
        session.runs.append(
            SessionRunRecord(
                run_id=result.run_id,
                plan_id=result.plan.plan_id,
                parent_run_id=parent_run_id,
                question=result.plan.question,
                artifact_directory=str(result.artifact_directory),
                report_path=str(result.report_path),
                evaluation=result.evaluation.model_dump(mode="json"),
            )
        )
        session.status = "completed"
        self._add_trace("evaluation", "评估分析证据", "completed", result.evaluation.summary)
        trace_path = Path(result.artifact_directory) / "execution_trace.json"
        if trace_path.is_file():
            try:
                import json

                execution_rows = json.loads(trace_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                execution_rows = []
            for row in execution_rows:
                self.current_session.trace.append(
                    TraceEvent(
                        category="tool",
                        name=str(row.get("title") or row.get("tool") or "分析工具"),
                        status="completed",
                        duration_ms=float(row["duration_ms"]) if row.get("duration_ms") is not None else None,
                        summary=str(row.get("result_key", "")),
                        details=dict(row.get("parameters", {})),
                    )
                )
        self._add_trace("artifact", "生成可复现研究包", "completed", str(result.artifact_directory))
        self._persist_and_render(keep_timeline=True)

    def cancel_current_task(self) -> None:
        if self._worker is None:
            return
        self._worker.cancel()
        self.conversation.send_button.setText("正在停止…")
        self.conversation.send_button.setEnabled(False)
        self._add_trace("agent", "请求停止当前任务", "warning", "将在下一个安全进度边界停止")

    def _task_cancelled(self) -> None:
        self._pending_execute_plan = None
        session = self.current_session
        if self.agent.has_thread(session.session_id):
            self.agent.cancel(session.session_id)
        session.status = "stopped"
        self._close_latest_running_trace("stopped")
        self._set_plan_message_state("stopped")
        self._set_active_tool_status("stopped", "用户停止了当前任务")
        self._append_message(SessionMessage(role="system", kind="notice", content="当前 Agent 任务已停止。"))
        self._add_trace("agent", "停止当前任务", "stopped", "用户发出停止请求")
        self._persist_and_render(keep_timeline=True)

    def _task_failed(self, detail: str) -> None:
        self._pending_execute_plan = None
        session = self.current_session
        session.status = "failed"
        self._close_latest_running_trace("failed")
        self._set_plan_message_state("failed")
        headline = detail.strip().splitlines()[-1] if detail.strip() else "未知错误"
        self._set_active_tool_status("failed", headline)
        self._append_message(SessionMessage(role="assistant", kind="error", content=f"任务失败：{headline}"))
        self._add_trace("error", "Agent 任务失败", "failed", headline)
        self._persist_and_render(keep_timeline=True)

    def _fail_before_task(self, message: str) -> None:
        self.current_session.status = "failed"
        self._append_message(SessionMessage(role="assistant", kind="error", content=message))
        self._add_trace("error", "文件输入校验失败", "failed", message)
        self._persist_and_render(keep_timeline=True)

    def _start_worker(self, *, kind: str, operation: Any, success_handler: Any) -> None:
        thread = QThread(self)
        worker = FunctionWorker(operation)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._task_progress)
        worker.completed.connect(success_handler)
        worker.completed.connect(thread.quit)
        worker.failed.connect(self._task_failed)
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
        self._task_session_id = self.current_session.session_id
        self._last_progress_message = ""
        self._set_busy(True)
        thread.start()

    def _task_progress(self, value: int, message: str) -> None:
        session = self.current_session
        if self._task_kind == "execute" and value >= 78:
            session.status = "evaluating"
        elif self._task_kind == "execute":
            session.status = "running"
        elif self._task_kind == "dialogue":
            session.status = "inspecting_data" if any(word in message for word in ("数据", "加载", "对齐")) else "understanding"
        else:
            session.status = "inspecting_data" if value < 65 else "understanding"
        if self._active_tool_widget is not None:
            self._active_tool_widget.set_status("running", f"{value}% · {message}")
        self._update_active_tool_payload("running", f"{value}% · {message}")
        if message != self._last_progress_message:
            self._close_latest_running_trace("completed")
            category = "tool" if self._task_kind == "execute" else "agent"
            self._add_trace(category, message, "running", f"进度 {value}%")
            self._last_progress_message = message
        self.conversation.set_status(session.status)
        self.context.plan.set_plan(
            EDAPlan.model_validate(session.current_plan) if session.current_plan else None,
            state="running" if self._task_kind == "execute" else "pending",
        )
        self.status_changed.emit(message)

    def _thread_finished(self) -> None:
        pending_plan = self._pending_execute_plan
        self._pending_execute_plan = None
        self._thread = None
        self._worker = None
        self._task_kind = None
        self._task_session_id = None
        self._active_tool_widget = None
        self._active_tool_message_id = None
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

    def _complete_active_tool(self, detail: str) -> None:
        self._set_active_tool_status("completed", detail)
        self._close_latest_running_trace("completed")

    def _set_active_tool_status(self, status: str, detail: str) -> None:
        if self._active_tool_widget is not None:
            self._active_tool_widget.set_status(status, detail)
        self._update_active_tool_payload(status, detail)

    def _close_latest_running_trace(self, status: str) -> None:
        event = next((item for item in reversed(self.current_session.trace) if item.status == "running"), None)
        if event is not None:
            event.status = status  # type: ignore[assignment]
            self.context.trace.set_events(self.current_session.trace)

    def _update_active_tool_payload(self, status: str, detail: str) -> None:
        if not self._active_tool_message_id:
            return
        message = next(
            (item for item in self.current_session.messages if item.message_id == self._active_tool_message_id),
            None,
        )
        if message is not None:
            message.payload["status"] = status
            message.payload["detail"] = detail

    def _set_plan_message_state(self, state: str, *, plan: EDAPlan | None = None) -> None:
        message = next((item for item in reversed(self.current_session.messages) if item.kind == "plan"), None)
        if message is None:
            return
        message.payload["state"] = state
        if plan is not None:
            message.payload["plan"] = plan.model_dump(mode="json")

    def _agent_conversation(self, session: ResearchSession) -> list[ConversationMessage]:
        return [
            ConversationMessage(role=message.role, content=message.content, created_at=message.created_at)
            for message in session.messages
            if message.kind == "text" and message.role in {"user", "assistant"} and message.content
        ][-16:]

    def _persist_and_render(self, *, keep_timeline: bool = False) -> None:
        self.current_session.touch()
        self.store.save(self.sessions)
        self.history.set_sessions(self.sessions, self.current_session_id)
        if keep_timeline:
            self.conversation.title_label.setText(self.current_session.title)
            self.conversation.set_status(self.current_session.status)
            config = self.current_session.inputs["config"]
            self.conversation.config_label.setText(
                Path(config.path).name if config.path else "未使用 YAML 配置（可选）"
            )
            self.context.set_session(self.current_session)
        else:
            self._render_current()

    def _render_current(self) -> None:
        session = self.current_session
        self.conversation.set_session(session)
        self.context.set_session(session)
        self.history.set_sessions(self.sessions, self.current_session_id)
        self.status_changed.emit(STATUS_LABELS.get(session.status, session.status))
