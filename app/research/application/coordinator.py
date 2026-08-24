"""UI-facing coordinator for one persistent LangGraph research loop."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any, ClassVar

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
    ResearchLoopSnapshot,
    ResumePayload,
)
from app.research.graph.workflow import build_research_workflow
from app.research.schemas.study import StudyConfig
from app.research.skills.registry import SkillRegistry
from app.research.tools.eda.functions import build_eda_tool_registry
from app.research.tools.registry import ToolRegistry


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

    _NODE_PROGRESS: ClassVar[dict[str, tuple[int, str]]] = {
        "ingest_user": (3, "接收用户消息"),
        "main_agent": (8, "主 Agent 理解与路由"),
        "resolve_skill": (14, "选择并校验 Skill"),
        "eda_subagent": (24, "Subagent 生成或修订方案"),
        "validate_plan": (34, "确定性校验研究方案"),
        "prepare_approval": (40, "准备用户审批"),
        "approval_interrupt": (40, "等待用户审批"),
        "lock_plan": (45, "锁定方案并生成工具队列"),
        "mark_tool_running": (50, "准备执行单个工具"),
        "execute_tool": (62, "执行确定性工具"),
        "validate_tool_result": (72, "校验工具结果与证据"),
        "finalize_iteration": (84, "评估证据并生成研究包"),
        "prepare_evaluation_revision": (88, "评估反馈进入下一轮"),
        "explain_result": (96, "Agent 解释已验证结果"),
        "prepare_need_user": (98, "等待用户处理反馈"),
        "user_interrupt": (98, "等待用户决策"),
        "result_interrupt": (100, "等待结果追问"),
    }

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
                value, message = self._NODE_PROGRESS.get(node_name, (50, node_name))
                progress(value, message)

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
            "graph_schema_version": 1,
            "thread_id": thread_id,
            "session_id": thread_id,
            "phase": "idle",
            "pending_user_message": message,
            "user_request": "",
            "messages": [item.model_dump(mode="json") for item in (conversation or [])],
            "study_config": study_config.model_dump(mode="json") if study_config is not None else None,
            "data_profile": None,
            "quality_report": None,
            "data_fingerprint": None,
            "available_skills": [],
            "active_skill": None,
            "decision": None,
            "current_plan": None,
            "plan_history": [],
            "plan_fingerprints": [],
            "plan_origin": "initial",
            "authorization_envelope": None,
            "approval_state": ApprovalState().model_dump(mode="json"),
            "approval_timeout_seconds": max(1, int(approval_timeout_seconds)),
            "tool_queue": [],
            "tool_cursor": 0,
            "tool_records": {},
            "tool_results": [],
            "pending_tool_result": None,
            "tool_outcome": "",
            "evaluation": None,
            "eda_summary": None,
            "feedback_packets": [],
            "budget": LoopBudget().model_dump(mode="json"),
            "evidence_fingerprints": [],
            "run_history": [],
            "latest_run": None,
            "loop_records": [],
            "assistant_message": "",
            "stop_reason": None,
            "events": [],
        }
        if imported:
            base.update(imported)
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
            if not self.has_thread(session_id):
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
                            if normalized in {"按这个执行", "立即执行", "确认执行", "执行当前方案"}
                            else "modify"
                        )
                    elif snapshot.interrupt.kind == "result":
                        action = "stop" if "".join(text.split()) in {"结束", "停止", "结束研究"} else "followup"
                    else:
                        normalized = "".join(text.split())
                        if normalized in {"接受当前限制", "接受限制", "按当前结果完成"}:
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
        payload = ResumePayload(action=action, message=message)
        with self._lock:
            snapshot = self.get_snapshot(session_id)
            approval = ApprovalState.model_validate(snapshot.values.get("approval_state", {}))
            if action == "timeout_accept" and approval.status == "expired" and not foreground_timeout:
                raise ValueError("审批已过期，必须由用户明确确认。")
            self._run_graph(
                Command(resume=payload.model_dump(mode="json")),
                thread_id=session_id,
                progress=progress,
            )
            return self.get_snapshot(session_id)

    def pause_approval(self, *, session_id: str, remaining_seconds: int) -> ResearchLoopSnapshot:
        with self._lock:
            snapshot = self.get_snapshot(session_id)
            approval = ApprovalState.model_validate(snapshot.values.get("approval_state", {}))
            approval.status = "paused"
            approval.deadline = None
            approval.remaining_seconds = max(1, int(remaining_seconds))
            self.graph.update_state(
                self._config(session_id),
                {"approval_state": approval.model_dump(mode="json")},
            )
            return self.get_snapshot(session_id)

    def resume_approval_timer(
        self,
        *,
        session_id: str,
        deadline: str,
        remaining_seconds: int,
    ) -> ResearchLoopSnapshot:
        with self._lock:
            snapshot = self.get_snapshot(session_id)
            approval = ApprovalState.model_validate(snapshot.values.get("approval_state", {}))
            approval.status = "waiting"
            approval.deadline = deadline
            approval.remaining_seconds = max(1, int(remaining_seconds))
            self.graph.update_state(self._config(session_id), {"approval_state": approval.model_dump(mode="json")})
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
            if "execute_tool" in getattr(raw, "next", ()):
                cursor = int(values.get("tool_cursor", 0))
                queue = values.get("tool_queue", [])
                if cursor < len(queue):
                    call_id = queue[cursor]["call_id"]
                    records = dict(values.get("tool_records", {}))
                    record = dict(records[call_id])
                    budget = LoopBudget.model_validate(values.get("budget", {}))
                    retries = budget.tool_retry_counts.get(call_id, 0)
                    if retries >= budget.max_tool_retries:
                        raise RuntimeError("工具崩溃恢复次数已耗尽")
                    budget.tool_retry_counts[call_id] = retries + 1
                    budget.tool_calls_used += 1
                    record["status"] = "running"
                    record["attempts"] = int(record.get("attempts", 0)) + 1
                    records[call_id] = record
                    self.graph.update_state(
                        config,
                        {
                            "tool_records": records,
                            "budget": budget.model_dump(mode="json"),
                            "tool_outcome": "run",
                        },
                        as_node="mark_tool_running",
                    )
            self._run_graph(None, thread_id=thread_id)
            return self.get_snapshot(thread_id)

    def delete_thread(self, thread_id: str) -> None:
        self.checkpointer_handle.saver.delete_thread(thread_id)

    def cancel(self, thread_id: str) -> None:
        """Stop an active local loop at a safe node boundary and remove its resumable cursor."""

        with self._lock:
            self.checkpointer_handle.saver.delete_thread(thread_id)

    def close(self) -> None:
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
