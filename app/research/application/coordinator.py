"""UI-facing coordinator for one persistent LangGraph research loop."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any

from langgraph.types import Command

from app.llm.factory import build_model_gateway
from app.research.agent.orchestrator import MainResearchAgent, ModelResearchDialogue
from app.research.agent.schemas import (
    AgentRunResult,
    ConversationMessage,
    EDAPlan,
    ResearchProposal,
    ResearchTurnResult,
)
from app.research.agent.subagents.eda import EDASubagent, ModelEDAPlanner
from app.research.application.execution import EDAExecutionService
from app.research.application.planning import EDAPlanningService, ProgressCallback
from app.research.graph.checkpoint import CheckpointerHandle
from app.research.graph.contracts import (
    ApprovalState,
    InterruptPayload,
    LoopBudget,
    LoopCursor,
    ResearchLoopSnapshot,
    ResumePayload,
)
from app.research.graph.migrations import archive_incompatible_state, safe_incompatible_state
from app.research.graph.narration import SILENT_NODES, narrate_event, narrate_node, progress_message
from app.research.graph.workflow import build_research_workflow
from app.research.schemas.study import StudyConfig
from app.research.skills.registry import SkillRegistry
from app.research.tools.eda.functions import build_eda_tool_registry
from app.research.tools.registry import ToolRegistry

GRAPH_SCHEMA_VERSION = 9


class ResearchCoordinator:
    """Own the graph, model roles, checkpointer, Skills, and controlled tools."""

    def __init__(
        self,
        *,
        main_agent: MainResearchAgent | None = None,
        eda_subagent: EDASubagent | None = None,
        execution: EDAExecutionService | None = None,
        skills: SkillRegistry | None = None,
        tools: ToolRegistry | None = None,
        checkpoint_path: str | Path | None = None,
        checkpointer_handle: CheckpointerHandle | None = None,
    ) -> None:
        self.skills = skills or SkillRegistry.default()
        self.skill_load_errors = list(self.skills.load_errors)
        self.tools = tools or build_eda_tool_registry()
        if main_agent is None or eda_subagent is None:
            gateway = build_model_gateway()
            main_agent = main_agent or MainResearchAgent(model_dialogue=ModelResearchDialogue(gateway=gateway))
            eda_subagent = eda_subagent or EDASubagent(
                model_planner=ModelEDAPlanner(gateway=gateway, tools=self.tools)
            )
        self.main_agent = main_agent
        self.eda_subagent = eda_subagent
        self.planning = EDAPlanningService(eda_subagent)
        self.execution = execution or EDAExecutionService(registry=self.tools, skills=self.skills)
        self.checkpointer_handle = checkpointer_handle or (
            CheckpointerHandle.sqlite(checkpoint_path) if checkpoint_path is not None else CheckpointerHandle.memory()
        )
        self.graph = build_research_workflow(
            main_agent=self.main_agent,
            planning=self.planning,
            execution=self.execution,
            skills=self.skills,
            tools=self.tools,
            checkpointer=self.checkpointer_handle.saver,
        )
        self.workflow = self.graph
        self._lock = RLock()

    @staticmethod
    def _config(thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    def _run_graph(
        self,
        graph_input: Any,
        *,
        thread_id: str,
        progress: Callable[[int, str], None] | None = None,
    ) -> None:
        for update in self.graph.stream(
            graph_input,
            config=self._config(thread_id),
            stream_mode="updates",
        ):
            if not progress or not isinstance(update, dict):
                continue
            for node_name in update:
                if str(node_name).startswith("__"):
                    continue
                value, step = narrate_node(str(node_name))
                node_update = update.get(node_name)
                source_event = ""
                if isinstance(node_update, dict):
                    events = node_update.get("events")
                    if isinstance(events, list) and events and isinstance(events[-1], dict):
                        step = narrate_event(events[-1])
                        source_event = str(events[-1].get("name", ""))
                if node_name in SILENT_NODES and not source_event:
                    continue
                progress(value, progress_message(step, source_event))

    @staticmethod
    def _initial_state(
        *,
        thread_id: str,
        message: str,
        study_config: StudyConfig | None,
        conversation: list[ConversationMessage] | None,
        approval_timeout_seconds: int,
        imported: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        base = {
            "graph_schema_version": GRAPH_SCHEMA_VERSION,
            "schema_upgrade_required": False,
            "thread_id": thread_id,
            "session_id": thread_id,
            "phase": "idle",
            "control": "understand",
            "loop_cursor": LoopCursor().model_dump(mode="json"),
            "episode_history": [],
            "pending_user_message": message,
            "user_request": "",
            "latest_turn": "",
            "explanation_request": "",
            "messages": [item.model_dump(mode="json") for item in (conversation or [])],
            "study_config": study_config.model_dump(mode="json") if study_config is not None else None,
            "data_profile": None,
            "quality_report": None,
            "active_skill": None,
            "decision": None,
            "current_plan": None,
            "plan_history": [],
            "plan_fingerprints": [],
            "planning_failure_fingerprints": [],
            "user_interrupt_kind": None,
            "plan_origin": "initial",
            "authorization_envelope": None,
            "approval_state": ApprovalState().model_dump(mode="json"),
            "approval_timeout_seconds": max(1, int(approval_timeout_seconds)),
            "tool_queue": [],
            "tool_cursor": 0,
            "tool_records": {},
            "tool_result_cache": {},
            "tool_results": [],
            "pending_tool_result": None,
            "evaluation": None,
            "eda_summary": None,
            "feedback_packets": [],
            "feedback_history": [],
            "budget": LoopBudget().model_dump(mode="json"),
            "evidence_fingerprints": [],
            "agenda_fingerprints": [],
            "run_history": [],
            "latest_run": None,
            "loop_records": [],
            "assistant_message": "",
            "stop_reason": None,
            "events": [],
        }
        if imported:
            base.update(imported)
            base["graph_schema_version"] = GRAPH_SCHEMA_VERSION
            base["schema_upgrade_required"] = False
            base.setdefault("planning_failure_fingerprints", [])
            base.setdefault("control", "understand")
            base.setdefault("user_interrupt_kind", None)
            base.setdefault("latest_turn", base.get("user_request", ""))
            base.setdefault("explanation_request", "")
            base.setdefault("loop_cursor", LoopCursor().model_dump(mode="json"))
            base.setdefault("episode_history", [])
            base.setdefault("agenda_fingerprints", [])
            base["pending_user_message"] = message
            base["study_config"] = study_config.model_dump(mode="json") if study_config is not None else base.get(
                "study_config"
            )
        return base

    @staticmethod
    def _interrupt_from_state(snapshot: Any) -> InterruptPayload | None:
        for task in getattr(snapshot, "tasks", ()):
            for item in getattr(task, "interrupts", ()):
                try:
                    return InterruptPayload.model_validate(item.value)
                except (TypeError, ValueError):
                    continue
        return None

    def get_snapshot(self, thread_id: str) -> ResearchLoopSnapshot:
        snapshot = self.graph.get_state(self._config(thread_id))
        values = dict(snapshot.values or {})
        if values and values.get("graph_schema_version") != GRAPH_SCHEMA_VERSION:
            safe_values = safe_incompatible_state(values, target_version=GRAPH_SCHEMA_VERSION)
            return ResearchLoopSnapshot(
                thread_id=thread_id,
                phase="stopped",
                values=safe_values,
                interrupt=None,
                events=list(safe_values.get("events", [])),
            )
        interrupt_payload = self._interrupt_from_state(snapshot)
        if interrupt_payload is not None and interrupt_payload.kind == "plan_approval":
            approval = ApprovalState.model_validate(values.get("approval_state", {}))
            if approval.deadline:
                try:
                    expired = datetime.fromisoformat(approval.deadline) <= datetime.now(UTC)
                except ValueError:
                    expired = False
                if expired:
                    approval.status = "expired"
                    approval.remaining_seconds = 0
                    values["approval_state"] = approval.model_dump(mode="json")
                    interrupt_payload = interrupt_payload.model_copy(
                        update={
                            "message": "方案审批已过期，请明确确认、修改或拒绝。",
                            "choices": ["approve", "modify", "reject"],
                            "remaining_seconds": 0,
                        }
                    )
        return ResearchLoopSnapshot(
            thread_id=thread_id,
            phase=values.get("phase", "idle"),
            values=values,
            interrupt=interrupt_payload,
            events=list(values.get("events", [])),
        )

    def has_thread(self, thread_id: str) -> bool:
        return bool(self.graph.get_state(self._config(thread_id)).values)

    def submit_user_message(
        self,
        *,
        session_id: str,
        message: str,
        study_config: StudyConfig | None = None,
        conversation: list[ConversationMessage] | None = None,
        approval_timeout_seconds: int = 30,
        imported_state: dict[str, Any] | None = None,
        progress: Callable[[int, str], None] | None = None,
    ) -> ResearchLoopSnapshot:
        text = message.strip()
        if not text:
            raise ValueError("用户消息不能为空")
        with self._lock:
            has_compatible_thread = self.has_thread(session_id)
            if has_compatible_thread:
                existing = self.get_snapshot(session_id)
                if existing.values.get("graph_schema_version") != GRAPH_SCHEMA_VERSION:
                    normalized = "".join(text.split())
                    legacy_request = str(existing.values.get("user_request") or "").strip()
                    if legacy_request and normalized in {
                        "接受",
                        "同意",
                        "重试",
                        "重新尝试",
                        "直接研究",
                        "直接用你给的方案研究",
                        "按这个执行",
                    }:
                        text = legacy_request
                    raw_values = dict(self.graph.get_state(self._config(session_id)).values or {})
                    try:
                        archive_incompatible_state(
                            thread_id=session_id,
                            values=raw_values,
                            checkpoint_path=self.checkpointer_handle.path,
                        )
                    except OSError:
                        pass
                    self.checkpointer_handle.saver.delete_thread(session_id)
                    has_compatible_thread = False
            if not has_compatible_thread:
                payload = self._initial_state(
                    thread_id=session_id,
                    message=text,
                    study_config=study_config,
                    conversation=conversation,
                    approval_timeout_seconds=approval_timeout_seconds,
                    imported=imported_state,
                )
                self._run_graph(payload, thread_id=session_id, progress=progress)
            else:
                snapshot = self.get_snapshot(session_id)
                if snapshot.interrupt is not None:
                    if snapshot.interrupt.kind == "plan_approval":
                        normalized = "".join(text.split())
                        action = (
                            "approve"
                            if normalized
                            in {
                                "接受",
                                "同意",
                                "按这个执行",
                                "立即执行",
                                "确认执行",
                                "执行当前方案",
                                "直接用你给的方案研究",
                            }
                            else "modify"
                        )
                    elif snapshot.interrupt.kind == "result":
                        action = "stop" if "".join(text.split()) in {"结束", "停止", "结束研究"} else "followup"
                    elif snapshot.interrupt.kind == "plan_error":
                        normalized = "".join(text.split())
                        if normalized in {"结束", "停止", "结束研究"}:
                            action = "stop"
                        elif normalized in {"重试", "重新尝试", "再试一次"}:
                            action = "retry"
                        elif normalized in {
                            "接受",
                            "同意",
                            "直接研究",
                            "直接用你给的方案研究",
                            "按这个执行",
                            "执行当前方案",
                        }:
                            action = "clarify"
                        else:
                            action = "modify"
                    elif snapshot.interrupt.kind in {"response_error", "finalization_error"}:
                        normalized = "".join(text.split())
                        action = "stop" if normalized in {"结束", "停止", "结束研究"} else "retry"
                    elif snapshot.interrupt.kind == "result_rejected":
                        normalized = "".join(text.split())
                        action = "stop" if normalized in {"结束", "停止", "结束研究"} else "modify"
                    else:
                        normalized = "".join(text.split())
                        if normalized in {"接受", "同意", "接受当前限制", "接受限制", "按当前结果完成"}:
                            action = "accept_limitations"
                        elif normalized in {"结束", "停止", "结束研究"}:
                            action = "stop"
                        else:
                            action = "modify"
                    self._run_graph(
                        Command(resume=ResumePayload(action=action, message=text).model_dump(mode="json")),
                        thread_id=session_id,
                        progress=progress,
                    )
                else:
                    self._run_graph(
                        {"pending_user_message": text, "study_config": study_config.model_dump(mode="json") if study_config else None},
                        thread_id=session_id,
                        progress=progress,
                    )
            return self.get_snapshot(session_id)

    def resume(
        self,
        *,
        session_id: str,
        action: str,
        message: str = "",
        foreground_timeout: bool = False,
        progress: Callable[[int, str], None] | None = None,
    ) -> ResearchLoopSnapshot:
        payload = ResumePayload(action=action, message=message, automatic_timeout=foreground_timeout)
        with self._lock:
            snapshot = self.get_snapshot(session_id)
            if snapshot.values.get("graph_schema_version") != GRAPH_SCHEMA_VERSION:
                raise ValueError("旧版本研究循环已失效，请重新提交研究问题。")
            if snapshot.interrupt is None:
                raise ValueError("当前研究任务没有等待用户恢复的 interrupt。")
            approval = ApprovalState.model_validate(snapshot.values.get("approval_state", {}))
            if action == "timeout_accept" and approval.status == "expired" and not foreground_timeout:
                raise ValueError("审批已过期，必须由用户明确确认。")
            foreground_timeout_accept = payload.action == "timeout_accept" and foreground_timeout
            if payload.action not in snapshot.interrupt.choices and not foreground_timeout_accept:
                raise ValueError(
                    f"当前 interrupt 不允许操作 {payload.action}；可选操作："
                    f"{', '.join(snapshot.interrupt.choices)}"
                )
            self._run_graph(
                Command(resume=payload.model_dump(mode="json")),
                thread_id=session_id,
                progress=progress,
            )
            return self.get_snapshot(session_id)

    def expire_approval(self, *, session_id: str) -> ResearchLoopSnapshot:
        with self._lock:
            snapshot = self.get_snapshot(session_id)
            approval = ApprovalState.model_validate(snapshot.values.get("approval_state", {}))
            approval.status = "expired"
            approval.deadline = None
            approval.remaining_seconds = 0
            self.graph.update_state(self._config(session_id), {"approval_state": approval.model_dump(mode="json")})
            return self.get_snapshot(session_id)

    def get_history(self, thread_id: str) -> list[dict[str, Any]]:
        return [dict(item.values or {}) for item in self.graph.get_state_history(self._config(thread_id))]

    def continue_thread(self, thread_id: str) -> ResearchLoopSnapshot:
        """Resume a non-interrupted checkpoint, including one left inside a tool node."""

        with self._lock:
            config = self._config(thread_id)
            raw = self.graph.get_state(config)
            values = dict(raw.values or {})
            if values.get("graph_schema_version") != GRAPH_SCHEMA_VERSION:
                raise ValueError("旧版本研究循环已失效，请重新提交研究问题。")
            if "execute_tool" in getattr(raw, "next", ()):
                cursor = int(values.get("tool_cursor", 0))
                queue = values.get("tool_queue", [])
                if cursor < len(queue):
                    call_id = queue[cursor]["call_id"]
                    records = dict(values.get("tool_records", {}))
                    record = dict(records[call_id])
                    budget = LoopBudget.model_validate(values.get("budget", {}))
                    attempts = int(record.get("attempts", 0))
                    if attempts >= budget.max_function_attempts_per_call:
                        raise RuntimeError("工具崩溃恢复次数已耗尽")
                    record["status"] = "running"
                    record["attempts"] = attempts + 1
                    records[call_id] = record
                    self.graph.update_state(
                        config,
                        {
                            "tool_records": records,
                        },
                        as_node="mark_tool_running",
                    )
            self._run_graph(None, thread_id=thread_id)
            return self.get_snapshot(thread_id)

    def delete_thread(self, thread_id: str) -> None:
        self.execution.evict_prepared()
        self.checkpointer_handle.saver.delete_thread(thread_id)

    def cancel(self, thread_id: str) -> None:
        """Stop an active local loop at a safe node boundary and remove its resumable cursor."""

        with self._lock:
            self.execution.evict_prepared()
            self.checkpointer_handle.saver.delete_thread(thread_id)

    def close(self) -> None:
        self.execution.evict_prepared()
        self.checkpointer_handle.close()

    # Compatibility boundaries retained for CLI and deterministic regression tests.
    def propose(
        self,
        *,
        question: str,
        config_path: str | Path | None = None,
        study_config: StudyConfig | None = None,
        conversation: list[ConversationMessage | dict[str, Any]] | None = None,
        progress: ProgressCallback | None = None,
        skill_name: str = "price-exogenous-eda",
    ) -> ResearchProposal:
        skill = self.skills.get(skill_name)
        return self.planning.propose(
            question=question,
            config_path=config_path,
            study_config=study_config,
            conversation=conversation,
            progress=progress,
            skill=skill,
        )

    def execute(
        self,
        *,
        plan: EDAPlan,
        config_path: str | Path | None = None,
        study_config: StudyConfig | None = None,
        conversation: list[ConversationMessage | dict[str, Any]] | None = None,
        output_directory: str | Path | None = None,
        run_id: str | None = None,
        progress: ProgressCallback | None = None,
    ) -> AgentRunResult:
        return self.execution.execute(
            plan=plan,
            config_path=config_path,
            study_config=study_config,
            conversation=conversation,
            output_directory=output_directory,
            run_id=run_id,
            progress=progress,
        )

    def handle_turn(
        self,
        *,
        question: str,
        status: str,
        study_config: StudyConfig | None = None,
        current_plan: EDAPlan | None = None,
        data_profile: dict[str, Any] | None = None,
        quality_report: dict[str, Any] | None = None,
        eda_summary: dict[str, Any] | None = None,
        evaluation: dict[str, Any] | None = None,
        conversation: list[ConversationMessage] | None = None,
        progress: ProgressCallback | None = None,
    ) -> ResearchTurnResult:
        callback = progress or (lambda _value, _message: None)
        decision, responder = self.main_agent.decide(
            question=question,
            status=status,
            config=study_config,
            plan=current_plan,
            data_profile=data_profile,
            quality_report=quality_report,
            summary=eda_summary,
            evaluation=evaluation,
            history=conversation or [],
            available_skills=self.skills.metadata(),
        )
        if decision.intent == "new_plan":
            if study_config is None:
                return ResearchTurnResult(
                    action="reply",
                    assistant_message="请先加载目标电价数据后再生成研究方案。",
                    responder=responder,
                )
            if not decision.skill_name:
                raise ValueError("模型没有选择 Skill")
            proposal = self.propose(
                question=question,
                study_config=study_config,
                conversation=conversation,
                progress=callback,
                skill_name=decision.skill_name,
            )
            return ResearchTurnResult(
                action="proposal",
                proposal=proposal,
                responder=responder,
                available_variables=proposal.data_profile.exogenous_names,
            )
        if decision.intent == "revise_plan" and current_plan is not None and study_config is not None:
            revised, message = self.main_agent.revise_plan(
                question=question,
                plan=current_plan,
                config=study_config,
                decision=decision,
            )
            return ResearchTurnResult(
                action="plan_revision",
                assistant_message=message,
                responder=responder,
                revised_plan=revised,
                available_variables=[spec.name for spec in study_config.exogenous],
            )
        if decision.intent == "execute_plan" and current_plan is not None:
            return ResearchTurnResult(
                action="execute_plan",
                assistant_message=decision.response or "已确认当前方案。",
                responder=responder,
                revised_plan=current_plan,
            )
        return ResearchTurnResult(
            action="reply",
            assistant_message=decision.response,
            responder=responder,
        )
