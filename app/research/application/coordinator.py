"""UI-facing coordinator for one persistent LangGraph research loop."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

from langgraph.types import Command

from app.llm.factory import build_model_gateway
from app.research.agent.context import MAX_PERSISTED_CONVERSATION_MESSAGES
from app.research.agent.dynamic import DynamicAnalysisAgent
from app.research.agent.orchestrator import DialogueDecision, MainResearchAgent, ModelResearchDialogue
from app.research.agent.retrieval import select_persisted_conversation_history
from app.research.agent.schemas import (
    AgentRunResult,
    ConversationMessage,
    EDAPlan,
    EDAResearchScope,
    ResearchProposal,
    ResearchTurnResult,
)
from app.research.agent.subagents.eda import EDASubagent, ModelEDAPlanner
from app.research.application.commands import resolve_text_action
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
from app.research.graph.migrations import safe_incompatible_state
from app.research.graph.narration import (
    HUMAN_GATE_NODES,
    SILENT_NODES,
    narrate_event,
    narrate_node,
    progress_message,
)
from app.research.graph.tool_result_store import FileToolResultStore, InMemoryToolResultStore, ToolResultStore
from app.research.graph.workflow import build_research_workflow, validate_analysis_message_protocol
from app.research.schemas.study import StudyConfig, StudyInputDescriptor
from app.research.skills.registry import SkillRegistry
from app.research.tools.eda.functions import build_eda_tool_registry
from app.research.tools.registry import ToolRegistry

GRAPH_SCHEMA_VERSION = 13
GRAPH_RECURSION_LIMIT = 1000
MAX_RECALLED_CONVERSATION_MESSAGES = max(1, MAX_PERSISTED_CONVERSATION_MESSAGES - 2)


def _conversation_payloads(history: list[Any]) -> list[dict[str, Any]]:
    return [ConversationMessage.model_validate(item).model_dump(mode="json") for item in history]


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
        result_store: ToolResultStore | None = None,
        dynamic_agent: DynamicAnalysisAgent | None = None,
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
        if dynamic_agent is None:
            planner_gateway = getattr(getattr(eda_subagent, "model_planner", None), "gateway", None)
            if planner_gateway is not None and hasattr(planner_gateway, "invoke_tool_turn"):
                dynamic_agent = DynamicAnalysisAgent(gateway=planner_gateway, tools=self.tools)
        self.dynamic_agent = dynamic_agent
        self.planning = EDAPlanningService(eda_subagent)
        self.execution = execution or EDAExecutionService(registry=self.tools, skills=self.skills)
        self.checkpointer_handle = checkpointer_handle or (
            CheckpointerHandle.sqlite(checkpoint_path) if checkpoint_path is not None else CheckpointerHandle.memory()
        )
        self.result_store = result_store or (
            FileToolResultStore.beside_checkpoint(self.checkpointer_handle.path)
            if self.checkpointer_handle.path is not None
            else InMemoryToolResultStore()
        )
        self.graph = build_research_workflow(
            main_agent=self.main_agent,
            planning=self.planning,
            execution=self.execution,
            skills=self.skills,
            tools=self.tools,
            checkpointer=self.checkpointer_handle.saver,
            result_store=self.result_store,
            dynamic_agent=self.dynamic_agent,
        )
        self.workflow = self.graph
        self._lock = RLock()

    @staticmethod
    def _config(thread_id: str) -> dict[str, Any]:
        # Business budgets stop all expected cycles first.  This explicit
        # framework ceiling is the final guard against an accidental route loop.
        return {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": GRAPH_RECURSION_LIMIT,
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
                if node_name in HUMAN_GATE_NODES:
                    # The corresponding event is still persisted in the graph
                    # and synchronized into the audit trace.  The conversation
                    # presents the gate through its plan/result/notice card once
                    # the worker returns, instead of pretending it is Agent work.
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
        message_id: str,
        turn_id: str,
        study_config: StudyConfig | None,
        conversation: list[ConversationMessage | dict[str, Any]] | None,
        approval_timeout_seconds: int,
        automatic_approval_enabled: bool,
        study_input: StudyInputDescriptor | None = None,
        imported: dict[str, Any] | None = None,
        routed_decision: DialogueDecision | None = None,
    ) -> dict[str, Any]:
        base = {
            "graph_schema_version": GRAPH_SCHEMA_VERSION,
            "schema_upgrade_required": False,
            "thread_id": thread_id,
            "session_id": thread_id,
            "phase": "idle",
            "control": "understand",
            "return_to_gate": None,
            "loop_cursor": LoopCursor().model_dump(mode="json"),
            "episode_history": [],
            "episode_summaries": [],
            "pending_user_message": message,
            "pending_message_id": message_id,
            "pending_turn_id": turn_id,
            "active_turn_id": None,
            "user_request": "",
            "latest_turn": "",
            "explanation_request": "",
            "messages": _conversation_payloads(
                select_persisted_conversation_history(
                    conversation or [],
                    question=message,
                    current_turn_id=turn_id,
                    max_messages=MAX_RECALLED_CONVERSATION_MESSAGES,
                )
            ),
            "study_config": study_config.model_dump(mode="json") if study_config is not None else None,
            "study_input": study_input.model_dump(mode="json") if study_input is not None else None,
            "has_executable_data": study_config is not None or study_input is not None,
            "pending_research_action": None,
            "data_profile": None,
            "quality_report": None,
            "active_skill": None,
            "decision": None,
            "routed_decision": routed_decision.model_dump(mode="json") if routed_decision else None,
            "current_plan": None,
            "research_scope": None,
            "plan_history": [],
            "plan_fingerprints": [],
            "planning_failure_fingerprints": [],
            "user_interrupt_kind": None,
            "plan_origin": "initial",
            "revision_cycle_id": uuid4().hex[:12],
            "progress_records": [],
            "authorization_envelope": None,
            "approval_state": ApprovalState().model_dump(mode="json"),
            "approval_timeout_seconds": max(1, int(approval_timeout_seconds)),
            "automatic_approval_enabled": bool(automatic_approval_enabled),
            "tool_queue": [],
            "tool_cursor": 0,
            "tool_records": {},
            "tool_result_cache": {},
            "tool_results": [],
            "pending_tool_result": None,
            "analysis_messages": [],
            "model_round": 0,
            "tool_calls_used": 0,
            "current_tool_batch": [],
            "provider_call_groups": {},
            "call_evidence": [],
            "pending_analysis_text": "",
            "post_analysis_action": None,
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
            base.setdefault("revision_cycle_id", uuid4().hex[:12])
            base.setdefault("progress_records", [])
            base.setdefault("control", "understand")
            base.setdefault("return_to_gate", None)
            base.setdefault("user_interrupt_kind", None)
            base.setdefault("latest_turn", base.get("user_request", ""))
            base.setdefault("pending_message_id", message_id)
            base.setdefault("pending_turn_id", turn_id)
            base.setdefault("active_turn_id", None)
            base.setdefault("explanation_request", "")
            base.setdefault("loop_cursor", LoopCursor().model_dump(mode="json"))
            base.setdefault("episode_history", [])
            base.setdefault("episode_summaries", [])
            base.setdefault("agenda_fingerprints", [])
            base.setdefault("automatic_approval_enabled", False)
            base["pending_user_message"] = message
            base["pending_message_id"] = message_id
            base["pending_turn_id"] = turn_id
            base["study_config"] = study_config.model_dump(mode="json") if study_config is not None else base.get(
                "study_config"
            )
            if study_input is not None:
                base["study_input"] = study_input.model_dump(mode="json")
            base["has_executable_data"] = bool(base.get("study_config") or base.get("study_input"))
            base["routed_decision"] = routed_decision.model_dump(mode="json") if routed_decision else None
            base["messages"] = _conversation_payloads(
                select_persisted_conversation_history(
                    base.get("messages", []),
                    question=message,
                    current_turn_id=turn_id,
                    max_messages=MAX_RECALLED_CONVERSATION_MESSAGES,
                )
            )
        return base

    def _route_initial_turn(
        self,
        *,
        question: str,
        study_config: StudyConfig | None,
        study_input: StudyInputDescriptor | None,
        history: list[ConversationMessage | dict[str, Any]],
        imported: dict[str, Any] | None,
        current_turn_id: str,
    ) -> DialogueDecision:
        """Route a new conversation before allocating a persistent research Graph."""

        source = imported or {}
        config = study_config
        if config is None and source.get("study_config"):
            config = StudyConfig.model_validate(source["study_config"])
        plan = EDAPlan.model_validate(source["current_plan"]) if source.get("current_plan") else None
        scope = (
            EDAResearchScope.model_validate(source["research_scope"])
            if source.get("research_scope")
            else None
        )
        cursor = source.get("loop_cursor") or {}
        decision, _ = self.main_agent.decide(
            question=question,
            status=str(source.get("phase") or "idle"),
            config=config,
            plan=plan,
            scope=scope,
            data_profile=source.get("data_profile"),
            quality_report=source.get("quality_report"),
            summary=source.get("eda_summary"),
            evaluation=source.get("evaluation"),
            history=[ConversationMessage.model_validate(item) for item in history],
            available_skills=self.skills.metadata(),
            episode_summaries=list(source.get("episode_summaries") or []),
            active_gate=source.get("user_interrupt_kind") or source.get("return_to_gate"),
            episode_goal=str(cursor.get("episode_goal") or "") or None,
            latest_run=source.get("latest_run"),
            current_turn_id=current_turn_id,
            has_executable_data=bool(config or study_input or source.get("study_input")),
        )
        return decision

    @staticmethod
    def _direct_dialogue_snapshot(
        *,
        thread_id: str,
        decision: DialogueDecision,
        history: list[ConversationMessage | dict[str, Any]],
        turn_id: str,
        source: dict[str, Any] | None,
        study_config: StudyConfig | None,
        study_input: StudyInputDescriptor | None,
    ) -> ResearchLoopSnapshot:
        """Return a desktop-compatible reply without creating a research checkpoint."""

        answer = decision.response.strip()
        if not answer:
            raise ValueError("直接对话路由没有返回可展示回复")
        assistant = ConversationMessage(role="assistant", content=answer, turn_id=turn_id)
        previous = [ConversationMessage.model_validate(item).model_dump(mode="json") for item in history]
        original = source or {}
        values = {
            **original,
            "phase": "awaiting_user",
            "control": "reply",
            "decision": decision.model_dump(mode="json"),
            "direct_dialogue": True,
            "assistant_message": answer,
            "messages": [*previous, assistant.model_dump(mode="json")],
            "study_config": (
                study_config.model_dump(mode="json")
                if study_config is not None
                else original.get("study_config")
            ),
            "study_input": (
                study_input.model_dump(mode="json")
                if study_input is not None
                else original.get("study_input")
            ),
            "has_executable_data": bool(
                study_config or study_input or original.get("study_config") or original.get("study_input")
            ),
        }
        interrupt = InterruptPayload(
            kind="result",
            interrupt_id=uuid4().hex,
            state_revision=0,
            phase="awaiting_user",
            message=answer,
            choices=["followup", "next_round", "stop"],
        )
        return ResearchLoopSnapshot(
            thread_id=thread_id,
            phase="awaiting_user",
            values=values,
            interrupt=interrupt,
            events=[],
        )

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

    def _snapshot_after_run(self, thread_id: str) -> ResearchLoopSnapshot:
        """Capture the user-visible state, then discard obsolete step history."""

        snapshot = self.get_snapshot(thread_id)
        if snapshot.values.get("graph_schema_version") == GRAPH_SCHEMA_VERSION:
            self.checkpointer_handle.compact_thread(thread_id)
            storage_keys: set[str] = set()

            def collect(value: Any) -> None:
                if isinstance(value, dict):
                    if value.get("kind") == "tool_result_ref_v1" and value.get("storage_key"):
                        storage_keys.add(str(value["storage_key"]))
                    for child in value.values():
                        collect(child)
                elif isinstance(value, list):
                    for child in value:
                        collect(child)

            collect(snapshot.values)
            self.result_store.prune_thread(thread_id, storage_keys)
        return snapshot

    def has_thread(self, thread_id: str) -> bool:
        return bool(self.graph.get_state(self._config(thread_id)).values)

    def submit_user_message(
        self,
        *,
        session_id: str,
        message: str,
        message_id: str | None = None,
        turn_id: str | None = None,
        study_config: StudyConfig | None = None,
        study_input: StudyInputDescriptor | None = None,
        conversation: list[ConversationMessage | dict[str, Any]] | None = None,
        approval_timeout_seconds: int = 30,
        automatic_approval_enabled: bool = False,
        imported_state: dict[str, Any] | None = None,
        route_before_graph: bool = False,
        progress: Callable[[int, str], None] | None = None,
    ) -> ResearchLoopSnapshot:
        text = message.strip()
        if not text:
            raise ValueError("用户消息不能为空")
        resolved_message_id = message_id or uuid4().hex
        resolved_turn_id = turn_id or uuid4().hex
        recalled_conversation = select_persisted_conversation_history(
            conversation or [],
            question=text,
            current_turn_id=resolved_turn_id,
            max_messages=MAX_RECALLED_CONVERSATION_MESSAGES,
        )
        with self._lock:
            has_compatible_thread = self.has_thread(session_id)
            if has_compatible_thread:
                existing = self.get_snapshot(session_id)
                if existing.values.get("graph_schema_version") != GRAPH_SCHEMA_VERSION:
                    return existing
            if not has_compatible_thread and route_before_graph:
                routed_decision = self._route_initial_turn(
                    question=text,
                    study_config=study_config,
                    study_input=study_input,
                    history=recalled_conversation,
                    imported=imported_state,
                    current_turn_id=resolved_turn_id,
                )
                if routed_decision.intent in {"discussion", "explain_result"}:
                    return self._direct_dialogue_snapshot(
                        thread_id=session_id,
                        decision=routed_decision,
                        history=recalled_conversation,
                        turn_id=resolved_turn_id,
                        source=imported_state,
                        study_config=study_config,
                        study_input=study_input,
                    )
            else:
                routed_decision = None
            if not has_compatible_thread:
                payload = self._initial_state(
                    thread_id=session_id,
                    message=text,
                    message_id=resolved_message_id,
                    turn_id=resolved_turn_id,
                    study_config=study_config,
                    study_input=study_input,
                    conversation=recalled_conversation,
                    approval_timeout_seconds=approval_timeout_seconds,
                    automatic_approval_enabled=automatic_approval_enabled,
                    imported=imported_state,
                    routed_decision=routed_decision,
                )
                self._run_graph(payload, thread_id=session_id, progress=progress)
            else:
                snapshot = self.get_snapshot(session_id)
                if snapshot.interrupt is not None:
                    action = resolve_text_action(
                        kind=snapshot.interrupt.kind,
                        choices=snapshot.interrupt.choices,
                        message=text,
                    )
                    self._run_graph(
                        Command(
                            resume=ResumePayload(
                                action=action,
                                message=text,
                                message_id=resolved_message_id,
                                turn_id=resolved_turn_id,
                                conversation=_conversation_payloads(recalled_conversation)
                                if conversation is not None
                                else None,
                                interrupt_id=snapshot.interrupt.interrupt_id or None,
                                state_revision=snapshot.interrupt.state_revision,
                            ).model_dump(mode="json")
                        ),
                        thread_id=session_id,
                        progress=progress,
                    )
                else:
                    update = {
                        "pending_user_message": text,
                        "pending_message_id": resolved_message_id,
                        "pending_turn_id": resolved_turn_id,
                        "study_config": study_config.model_dump(mode="json") if study_config else snapshot.values.get(
                            "study_config"
                        ),
                        "study_input": study_input.model_dump(mode="json") if study_input else snapshot.values.get(
                            "study_input"
                        ),
                        "has_executable_data": bool(
                            study_config or study_input or snapshot.values.get("study_config") or snapshot.values.get("study_input")
                        ),
                    }
                    if conversation is not None:
                        update["messages"] = _conversation_payloads(recalled_conversation)
                    self._run_graph(
                        update,
                        thread_id=session_id,
                        progress=progress,
                    )
            return self._snapshot_after_run(session_id)

    def resume(
        self,
        *,
        session_id: str,
        action: str,
        message: str = "",
        message_id: str | None = None,
        turn_id: str | None = None,
        interrupt_id: str | None = None,
        state_revision: int | None = None,
        foreground_timeout: bool = False,
        progress: Callable[[int, str], None] | None = None,
    ) -> ResearchLoopSnapshot:
        payload = ResumePayload(
            action=action,
            message=message,
            message_id=message_id or (uuid4().hex if message.strip() else None),
            turn_id=turn_id or (uuid4().hex if message.strip() else None),
            interrupt_id=interrupt_id,
            state_revision=state_revision,
            automatic_timeout=foreground_timeout,
        )
        with self._lock:
            snapshot = self.get_snapshot(session_id)
            if snapshot.values.get("graph_schema_version") != GRAPH_SCHEMA_VERSION:
                raise ValueError("旧版本研究循环已失效，请新建对话并重新提交研究问题。")
            if snapshot.interrupt is None:
                raise ValueError("当前研究任务没有等待用户恢复的 interrupt。")
            if interrupt_id and interrupt_id != snapshot.interrupt.interrupt_id:
                raise ValueError("当前操作对应的交互状态已经失效，请使用最新界面重试。")
            if state_revision is not None and state_revision != snapshot.interrupt.state_revision:
                raise ValueError("当前操作对应的状态版本已经失效，请使用最新界面重试。")
            payload = payload.model_copy(
                update={
                    "interrupt_id": snapshot.interrupt.interrupt_id or None,
                    "state_revision": snapshot.interrupt.state_revision,
                }
            )
            approval = ApprovalState.model_validate(snapshot.values.get("approval_state", {}))
            if action == "timeout_accept" and approval.status == "expired" and not foreground_timeout:
                raise ValueError("审批已过期，必须由用户明确确认。")
            foreground_timeout_accept = (
                payload.action == "timeout_accept"
                and foreground_timeout
                and bool(snapshot.values.get("automatic_approval_enabled", False))
            )
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
            return self._snapshot_after_run(session_id)

    def continue_thread(self, thread_id: str) -> ResearchLoopSnapshot:
        """Resume a non-interrupted checkpoint, including one left inside a tool node."""

        with self._lock:
            config = self._config(thread_id)
            raw = self.graph.get_state(config)
            values = dict(raw.values or {})
            if values.get("graph_schema_version") != GRAPH_SCHEMA_VERSION:
                raise ValueError("旧版本研究循环已失效，请新建对话并重新提交研究问题。")
            validate_analysis_message_protocol(
                values.get("analysis_messages", []),
                values.get("provider_call_groups", {}),
                allow_pending_current_batch=True,
            )
            if "execute_tool" in getattr(raw, "next", ()):
                cursor = int(values.get("tool_cursor", 0))
                queue = values.get("tool_queue", [])
                if cursor < len(queue):
                    call_id = queue[cursor]["call_id"]
                    records = dict(values.get("tool_records", {}))
                    record = dict(records[call_id])
                    # ``mark_tool_running`` already persisted and counted the
                    # interrupted attempt before the process entered the tool
                    # node.  Route the abandoned attempt back through the
                    # graph's normal retry edge instead of counting it twice or
                    # raising outside the durable state machine.  The mark node
                    # owns the retry-budget decision and will expose an
                    # interrupt when the budget is exhausted.
                    record["status"] = "failed"
                    records[call_id] = record
                    self.graph.update_state(
                        config,
                        {
                            "control": "retry",
                            "tool_records": records,
                        },
                        as_node="execute_tool",
                    )
            self._run_graph(None, thread_id=thread_id)
            return self._snapshot_after_run(thread_id)

    def delete_thread(self, thread_id: str) -> None:
        self.execution.evict_prepared()
        self.checkpointer_handle.delete_thread(thread_id)
        self.result_store.delete_thread(thread_id)

    def cancel(self, thread_id: str) -> None:
        """Stop an active local loop at a safe node boundary and remove its resumable cursor."""

        with self._lock:
            self.execution.evict_prepared()
            self.checkpointer_handle.delete_thread(thread_id)
            self.result_store.delete_thread(thread_id)

    def close(self) -> None:
        self.execution.evict_prepared()
        self.result_store.close()
        self.checkpointer_handle.close()

    # Direct service boundaries retained for deterministic regression tests.
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
        active_gate: str | None = None,
        episode_goal: str | None = None,
        latest_run: dict[str, Any] | None = None,
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
            active_gate=active_gate,
            episode_goal=episode_goal,
            latest_run=latest_run,
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
