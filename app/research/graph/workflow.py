"""One persistent, interruptible, self-validating LangGraph research loop."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any, Literal
from uuid import uuid4

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import ValidationError

from app.llm.gateway import ModelMessage, ModelResponseError
from app.research.agent.context import MAX_EPISODE_SUMMARIES, MAX_PERSISTED_CONVERSATION_MESSAGES
from app.research.agent.dynamic import DynamicAnalysisAgent
from app.research.agent.errors import (
    DataFingerprintMismatchError,
    InsufficientDataError,
    PlanCompatibilityError,
    RepairablePlanError,
    ResearchModelContextLimitError,
    ResearchModelOutputTruncatedError,
    ResearchModelUnavailableError,
    ResearchPlanValidationError,
    SkillVersionMismatchError,
)
from app.research.agent.orchestrator import DialogueDecision, MainResearchAgent
from app.research.agent.retrieval import bounded_recent_turn_history, select_persisted_conversation_history
from app.research.agent.schemas import ConversationMessage, EDAPlan, EDAResearchScope
from app.research.application.execution import EDAExecutionService
from app.research.application.planning import EDAPlanningService, noop_progress
from app.research.data.inference import resolve_study_input
from app.research.data.loader import ResearchDataError
from app.research.data.sources.summary import summary_payload
from app.research.graph.contracts import (
    ApprovalState,
    AuthorizationEnvelope,
    CallEvidenceRecord,
    EpisodeSummary,
    InterruptKind,
    InterruptPayload,
    LoopBudget,
    LoopCursor,
    ResumePayload,
    ToolCallGroupRecord,
    ToolCallRecord,
)
from app.research.graph.guards import (
    authorization_envelope,
    budget_feedback,
    canonical_hash,
    evidence_fingerprint,
    exception_feedback,
    plan_fingerprint,
    scope_authorization_envelope,
    scope_fingerprint,
    validate_automatic_revision,
    validate_scope_authorization,
)
from app.research.graph.process_events import ProcessEvent, append_process_events
from app.research.graph.state import ResearchLoopState
from app.research.graph.tool_result_store import ToolResultStore
from app.research.reporting.loop_history import write_loop_record
from app.research.schemas.feedback import FeedbackPacket
from app.research.schemas.results import DataQualityReport
from app.research.schemas.study import StudyConfig, StudyInputDescriptor
from app.research.skills.loader import SkillLoadError
from app.research.skills.registry import SkillRegistry
from app.research.tools.catalog import FUNCTION_CATALOG, FUNCTION_CATALOG_VERSION
from app.research.tools.contracts import ToolCall, ToolResult
from app.research.tools.executor import ToolExecutionError
from app.research.tools.policy import ToolPermissionError
from app.research.tools.registry import ToolRegistry, ToolRegistryError

MAX_TOOL_RESULT_CACHE = 64
MAX_STATE_EVENTS = 160
MAX_ACTIVE_FEEDBACK = 24
MAX_FEEDBACK_HISTORY = 128
MAX_GRAPH_MESSAGES = MAX_PERSISTED_CONVERSATION_MESSAGES
MAX_RUN_HISTORY = 48
MAX_EPISODE_HISTORY = MAX_EPISODE_SUMMARIES


def validate_analysis_message_protocol(
    messages: list[dict[str, Any]] | list[ModelMessage],
    provider_call_groups: dict[str, dict[str, Any]],
    *,
    allow_pending_current_batch: bool,
) -> None:
    """Validate unique provider IDs and exact assistant/tool message closure."""

    proposed: dict[str, int] = {}
    answered: dict[str, int] = {}
    for index, raw in enumerate(messages):
        message = raw if isinstance(raw, ModelMessage) else ModelMessage.model_validate(raw)
        if message.role == "assistant":
            for call in message.tool_calls:
                call_id = str(call.call_id or "").strip()
                if not call_id:
                    raise ResearchPlanValidationError("assistant 工具调用缺少 provider call_id")
                if call_id in proposed:
                    raise ResearchPlanValidationError(f"provider call_id 被重复提出：{call_id}")
                proposed[call_id] = index
        elif message.role == "tool":
            call_id = str(message.tool_call_id or "").strip()
            if call_id not in proposed:
                raise ResearchPlanValidationError(f"tool 消息没有对应的 assistant 调用：{call_id}")
            if call_id in answered:
                raise ResearchPlanValidationError(f"provider call_id 收到了重复 tool 结果：{call_id}")
            if index <= proposed[call_id]:
                raise ResearchPlanValidationError(f"tool 消息出现在 assistant 调用之前：{call_id}")
            answered[call_id] = index
    pending = set(proposed).difference(answered)
    if not pending:
        return
    if not allow_pending_current_batch:
        raise ResearchPlanValidationError(
            "存在未闭合的 provider tool calls：" + "、".join(sorted(pending))
        )
    groups = {
        key: ToolCallGroupRecord.model_validate(value)
        for key, value in provider_call_groups.items()
    }
    invalid = sorted(
        call_id
        for call_id in pending
        if call_id not in groups or groups[call_id].status not in {"pending", "running"}
    )
    if invalid:
        raise ResearchPlanValidationError(
            "未闭合 provider call 缺少唯一 pending group：" + "、".join(invalid)
        )


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _short(value: Any, limit: int = 56) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else f"{text[: limit - 1]}…"


def _function_label(function_name: str) -> str:
    spec = FUNCTION_CATALOG.get(function_name)
    return f"{spec.title} [{function_name}]" if spec is not None else function_name


def _compact_event_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Keep trace parameters useful without copying complete plan payloads into every checkpoint."""

    compact: dict[str, Any] = {}
    for key, value in arguments.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            compact[key] = value
        elif isinstance(value, list):
            compact[key] = {"count": len(value), "preview": value[:3]}
        elif isinstance(value, dict):
            compact[key] = {"keys": sorted(str(item) for item in value)[:8]}
        else:
            compact[key] = type(value).__name__
    return compact


def _event(
    state: ResearchLoopState,
    name: str,
    status: str = "completed",
    *,
    trace_category: str = "agent",
    **details: Any,
) -> list[dict[str, Any]]:
    existing = list(state.get("events", []))
    prior_sequences = [item.get("sequence") for item in existing if isinstance(item.get("sequence"), int)]
    sequence = (max(prior_sequences) if prior_sequences else len(existing)) + 1
    return [
        *existing,
        {
            "sequence": sequence,
            "created_at": _now(),
            "category": trace_category,
            "name": name,
            "status": status,
            "details": details,
        },
    ][-MAX_STATE_EVENTS:]


def _process_sequence(state: ResearchLoopState) -> int:
    sequences = [
        int(item.get("sequence", 0))
        for item in state.get("process_events", [])
        if isinstance(item, dict)
    ]
    return max(sequences, default=0) + 1


def _process_event(
    state: ResearchLoopState,
    event_type: str,
    *,
    step_id: str,
    round_number: int,
    source: Literal["model", "system"],
    sequence: int | None = None,
    phase: str = "analysis",
    title: str = "",
    content: str = "",
    action_id: str | None = None,
    tool_name: str | None = None,
    arguments: dict[str, Any] | None = None,
    result: str = "",
    report_path: str | None = None,
) -> ProcessEvent:
    cursor = _cursor(state)
    flow_id = cursor.episode_id or str(state.get("revision_cycle_id") or state["thread_id"])
    return ProcessEvent(
        session_id=str(state.get("session_id") or state["thread_id"]),
        flow_id=flow_id,
        round=round_number,
        step_id=step_id,
        sequence=sequence or _process_sequence(state),
        event_type=event_type,  # type: ignore[arg-type]
        phase=phase,
        source=source,
        title=title,
        content=content,
        action_id=action_id,
        tool_name=tool_name,
        arguments=arguments or {},
        result=result,
        report_path=report_path,
    )


def _emit_process_event(event: ProcessEvent) -> None:
    """Stream a live event when invoked inside LangGraph; remain test-friendly."""

    try:
        get_stream_writer()(event.model_dump(mode="json"))
    except RuntimeError:
        # Some unit tests exercise node functions without a streaming runtime.
        return


def _process_result_summary(result: ToolResult) -> str:
    """Build a short deterministic summary without copying the full tool output."""

    value = result.output.value
    for key in ("summary", "conclusion", "message"):
        candidate = value.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return _short(candidate, 120)
    findings = value.get("findings")
    if isinstance(findings, list) and findings:
        return _short(findings[0], 120)
    return f"{result.output.result_key} 已通过校验，用时 {result.duration_ms:.0f} 毫秒"


def _process_report_path(result: ToolResult) -> str | None:
    for key in ("report_path", "artifact_path"):
        value = result.output.value.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _call_process_context(
    state: ResearchLoopState,
    call: ToolCall,
) -> tuple[str, int, Literal["model", "system"], str]:
    step_id = state.get("process_call_steps", {}).get(call.call_id) or f"execution-{call.call_id}"
    scope = _scope(state)
    if scope is not None and not step_id.endswith("-system-preflight"):
        return step_id, int(state.get("model_round", 0)), "model", ""
    if step_id.endswith("-system-preflight"):
        return step_id, 0, "system", "执行已批准研究前的数据质量门槛检查"
    plan = _plan(state)
    planned = next((item for item in plan.enabled_steps if item.step_id == call.step_id), None) if plan else None
    basis = planned.rationale if planned is not None else "按照已确认的研究方案执行"
    return step_id, int(call.step_id[1:]), "system", basis


def _action_attempt_id(state: ResearchLoopState, call: ToolCall) -> str:
    record_payload = state.get("tool_records", {}).get(call.call_id, {})
    attempt = int(record_payload.get("attempts", 1)) if isinstance(record_payload, dict) else 1
    return f"{call.call_id}:attempt:{max(1, attempt)}"


def _action_started_events(
    state: ResearchLoopState,
    call: ToolCall,
) -> tuple[list[dict[str, Any]], ProcessEvent]:
    step_id, round_number, source, basis = _call_process_context(state, call)
    existing = list(state.get("process_events", []))
    step_exists = any(item.get("step_id") == step_id for item in existing)
    additions: list[ProcessEvent] = []
    sequence = _process_sequence(state)
    if not step_exists:
        ready = _process_event(
            state,
            "thinking_ready",
            step_id=step_id,
            round_number=round_number,
            source=source,
            sequence=sequence,
            title="执行依据" if source == "system" else "思考过程",
            content=basis or "本轮未提供思考过程说明",
        )
        additions.append(ready)
        sequence += 1
    started = _process_event(
        state,
        "action_started",
        step_id=step_id,
        round_number=round_number,
        source=source,
        sequence=sequence,
        title=FUNCTION_CATALOG.get(call.name).title if call.name in FUNCTION_CATALOG else call.name,
        content="正在执行",
        action_id=_action_attempt_id(state, call),
        tool_name=call.name,
        arguments=_compact_event_arguments(call.arguments),
    )
    additions.append(started)
    for event in additions:
        _emit_process_event(event)
    return append_process_events(existing, *additions), started


def _action_finished_events(
    state: ResearchLoopState,
    call: ToolCall,
    *,
    result: str,
    failed: bool = False,
    complete_step: bool = False,
    report_path: str | None = None,
) -> list[dict[str, Any]]:
    step_id, round_number, source, _basis = _call_process_context(state, call)
    sequence = _process_sequence(state)
    event = _process_event(
        state,
        "action_failed" if failed else "action_completed",
        step_id=step_id,
        round_number=round_number,
        source=source,
        sequence=sequence,
        title=FUNCTION_CATALOG[call.name].title if call.name in FUNCTION_CATALOG else call.name,
        action_id=_action_attempt_id(state, call),
        tool_name=call.name,
        result=result,
        report_path=report_path,
    )
    additions = [event]
    if complete_step:
        additions.append(
            _process_event(
                state,
                "step_completed",
                step_id=step_id,
                round_number=round_number,
                source=source,
                sequence=sequence + 1,
                title="步骤完成" if not failed else "步骤中断",
                result=result,
            )
        )
    for addition in additions:
        _emit_process_event(addition)
    return append_process_events(list(state.get("process_events", [])), *additions)


def _messages(state: ResearchLoopState) -> list[ConversationMessage]:
    return [ConversationMessage.model_validate(item) for item in state.get("messages", [])]


def _bounded_graph_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply the checkpoint limit without retaining half a turn."""

    return bounded_recent_turn_history(
        messages,
        max_turns=MAX_GRAPH_MESSAGES,
        max_messages=MAX_GRAPH_MESSAGES,
        max_characters=None,
    )


def _messages_before_user_turn(
    messages: list[dict[str, Any]],
    *,
    question: str,
    turn_id: str | None,
) -> list[dict[str, Any]]:
    """Reserve one user/assistant pair while retaining relevant old turns."""

    selected = select_persisted_conversation_history(
        messages,
        question=question,
        current_turn_id=turn_id,
        max_messages=max(1, MAX_GRAPH_MESSAGES - 2),
    )
    return [ConversationMessage.model_validate(item).model_dump(mode="json") for item in selected]


def _resume_messages(state: ResearchLoopState, response: ResumePayload) -> list[dict[str, Any]]:
    return list(response.conversation) if response.conversation is not None else list(state.get("messages", []))


def _conversation_message(
    *,
    role: Literal["user", "assistant", "system"],
    content: str,
    message_id: str | None = None,
    turn_id: str | None = None,
    episode_id: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": role, "content": content}
    if message_id:
        payload["message_id"] = message_id
    if turn_id:
        payload["turn_id"] = turn_id
    if episode_id:
        payload["episode_id"] = episode_id
    return ConversationMessage.model_validate(payload).model_dump(mode="json")


def _config(state: ResearchLoopState) -> StudyConfig | None:
    value = state.get("study_config")
    return StudyConfig.model_validate(value) if value is not None else None


def _study_input(state: ResearchLoopState) -> StudyInputDescriptor | None:
    value = state.get("study_input")
    return StudyInputDescriptor.model_validate(value) if value is not None else None


def _plan(state: ResearchLoopState) -> EDAPlan | None:
    value = state.get("current_plan")
    return EDAPlan.model_validate(value) if value is not None else None


def _scope(state: ResearchLoopState) -> EDAResearchScope | None:
    value = state.get("research_scope")
    return EDAResearchScope.model_validate(value) if value is not None else None


def _current_data_fingerprint(state: ResearchLoopState) -> str | None:
    active = state.get("research_scope") or state.get("current_plan") or {}
    value = active.get("data_fingerprint")
    return str(value) if value else None


def _scoped_episode_summaries(state: ResearchLoopState) -> list[dict[str, Any]]:
    """Return only evidence proven against the currently active data snapshot."""

    fingerprint = _current_data_fingerprint(state)
    if not fingerprint:
        return []
    return [
        item
        for item in state.get("episode_summaries", [])
        if item.get("memory_status", "active") == "active"
        and item.get("data_fingerprint") == fingerprint
    ]


def _feedback(state: ResearchLoopState) -> list[FeedbackPacket]:
    return [FeedbackPacket.model_validate(item) for item in state.get("feedback_packets", [])]


def _append_feedback(state: ResearchLoopState, packet: FeedbackPacket) -> list[dict[str, Any]]:
    existing = list(state.get("feedback_packets", []))
    fingerprint = canonical_hash(
        {
            "source": packet.source,
            "code": packet.code,
            "message": packet.message,
            "step_id": packet.step_id,
        }
    )
    for index, item in enumerate(existing):
        if canonical_hash(
            {
                "source": item.get("source"),
                "code": item.get("code"),
                "message": item.get("message"),
                "step_id": item.get("step_id"),
            }
        ) == fingerprint:
            if packet.requires_user and not item.get("requires_user"):
                existing[index] = packet.model_dump(mode="json")
            return existing
    return [*existing, packet.model_dump(mode="json")][-MAX_ACTIVE_FEEDBACK:]


def _archive_feedback(state: ResearchLoopState) -> list[dict[str, Any]]:
    history = list(state.get("feedback_history", []))
    known = {
        canonical_hash(
            {
                "source": item.get("source"),
                "code": item.get("code"),
                "message": item.get("message"),
                "step_id": item.get("step_id"),
            }
        )
        for item in history
    }
    for item in state.get("feedback_packets", []):
        fingerprint = canonical_hash(
            {
                "source": item.get("source"),
                "code": item.get("code"),
                "message": item.get("message"),
                "step_id": item.get("step_id"),
            }
        )
        if fingerprint not in known:
            history.append(item)
            known.add(fingerprint)
    return history[-MAX_FEEDBACK_HISTORY:]


def _budget(state: ResearchLoopState) -> LoopBudget:
    return LoopBudget.model_validate(state.get("budget", {}))


def _cursor(state: ResearchLoopState) -> LoopCursor:
    return LoopCursor.model_validate(state.get("loop_cursor", {}))


def _episode_summary(
    state: ResearchLoopState,
    *,
    cursor: LoopCursor | None = None,
    latest_run: dict[str, Any] | None = None,
    evaluation: dict[str, Any] | None = None,
    status: str | None = None,
) -> EpisodeSummary | None:
    current = cursor or _cursor(state)
    if not current.episode_id:
        return None
    latest = latest_run if latest_run is not None else state.get("latest_run") or {}
    assessment = evaluation if evaluation is not None else state.get("evaluation") or {}
    plan = state.get("current_plan") or {}
    config = _config(state)
    decision = assessment.get("decision")
    resolved_status = status or current.episode_status
    if status is None:
        resolved_status = {
            "accept": "accepted",
            "need_user": "limited",
            "reject": "rejected",
        }.get(decision, resolved_status)
    return EpisodeSummary(
        episode_id=current.episode_id,
        episode_number=max(1, current.episode_number),
        goal=current.episode_goal or str(state.get("user_request") or ""),
        status=resolved_status,
        run_id=latest.get("run_id"),
        plan_id=latest.get("plan_id") or (state.get("current_plan") or {}).get("plan_id"),
        evaluation_decision=decision,
        summary=str(assessment.get("summary") or ""),
        findings=[str(item) for item in assessment.get("findings", [])[:8]],
        warnings=[str(item) for item in assessment.get("warnings", [])[:8]],
        report_path=latest.get("report_path"),
        figure_count=len(latest.get("figure_paths") or {}),
        figure_keys=[str(key) for key in (latest.get("figure_paths") or {})],
        data_fingerprint=plan.get("data_fingerprint"),
        study_name=config.study.name if config is not None else None,
        target_name=config.target.name if config is not None else None,
        study_start_time=(
            config.study.start_time.isoformat()
            if config is not None and config.study.start_time is not None
            else None
        ),
        study_end_time=(
            config.study.end_time.isoformat()
            if config is not None and config.study.end_time is not None
            else None
        ),
        skill_name=plan.get("skill_name"),
        skill_version=plan.get("skill_version"),
    )


def _upsert_episode_summary(
    existing: list[dict[str, Any]],
    summary: EpisodeSummary | None,
) -> list[dict[str, Any]]:
    if summary is None:
        return existing[-MAX_EPISODE_SUMMARIES:]
    retained = [item for item in existing if item.get("episode_id") != summary.episode_id]
    return [*retained, summary.model_dump(mode="json")][-MAX_EPISODE_SUMMARIES:]


def _planning_interrupt_kind(state: ResearchLoopState) -> str:
    if state.get("plan_origin") == "automatic_evaluation" and state.get("latest_run"):
        return "result_limitations"
    return "plan_error"


def _dialogue_interrupt_kind(state: ResearchLoopState) -> str:
    """Keep conversational provider failures distinct from plan validation failures."""

    if state.get("return_to_gate") in {
        "plan_approval",
        "result",
        "result_limitations",
        "result_rejected",
    }:
        return "response_error"
    return _planning_interrupt_kind(state)


def _latest_turn(state: ResearchLoopState) -> str:
    return str(state.get("latest_turn") or state.get("user_request") or "").strip()


def _episode_goal(state: ResearchLoopState) -> str:
    return str(_cursor(state).episode_goal or state.get("user_request") or _latest_turn(state)).strip()


def _approval_matches(state: ResearchLoopState, plan: EDAPlan) -> bool:
    approval = ApprovalState.model_validate(state.get("approval_state", {}))
    return bool(
        approval.status == "approved"
        and approval.plan_id == plan.plan_id
        and approval.plan_fingerprint == plan_fingerprint(plan)
        and approval.approved_at
    )


def _scope_approval_matches(state: ResearchLoopState, scope: EDAResearchScope) -> bool:
    approval = ApprovalState.model_validate(state.get("approval_state", {}))
    return bool(
        approval.status == "approved"
        and approval.plan_id == scope.scope_id
        and approval.plan_fingerprint == scope_fingerprint(scope)
        and approval.approved_at
    )


def _invalid_resume_feedback(state: ResearchLoopState, response: ResumePayload, payload: InterruptPayload) -> list[dict[str, Any]]:
    packet = FeedbackPacket(
        source="user",
        code="invalid_resume_action",
        severity="error",
        message=f"当前操作不允许：{response.action}",
        observed=response.action,
        expected=payload.choices,
        recommendation="请选择当前界面提供的操作。",
        requires_user=True,
    )
    return _append_feedback(state, packet)


def _interrupt_payload(
    state: ResearchLoopState,
    *,
    kind: InterruptKind,
    **values: Any,
) -> InterruptPayload:
    events = state.get("events", [])
    revision = max(
        (int(item["sequence"]) for item in events if isinstance(item.get("sequence"), int)),
        default=0,
    )
    cursor = _cursor(state)
    interrupt_id = canonical_hash(
        {
            "thread_id": state.get("thread_id"),
            "kind": kind,
            "episode_id": cursor.episode_id,
            "iteration_id": cursor.iteration_id,
            "stage": cursor.stage,
            "plan_id": (state.get("current_plan") or {}).get("plan_id"),
            "run_id": (state.get("latest_run") or {}).get("run_id"),
            "revision": revision,
        }
    )
    return InterruptPayload(
        kind=kind,
        interrupt_id=interrupt_id,
        state_revision=revision,
        **values,
    )


def _resume_identity_matches(response: ResumePayload, payload: InterruptPayload) -> bool:
    return bool(
        (response.interrupt_id is None or response.interrupt_id == payload.interrupt_id)
        and (response.state_revision is None or response.state_revision == payload.state_revision)
    )


def _stale_resume_feedback(
    state: ResearchLoopState,
    response: ResumePayload,
    payload: InterruptPayload,
) -> list[dict[str, Any]]:
    packet = FeedbackPacket(
        source="user",
        code="stale_interrupt_action",
        severity="error",
        message="当前操作来自已经失效的交互状态。",
        observed={"interrupt_id": response.interrupt_id, "state_revision": response.state_revision},
        expected={"interrupt_id": payload.interrupt_id, "state_revision": payload.state_revision},
        recommendation="请使用当前界面中的最新方案或操作。",
        requires_user=True,
    )
    return _append_feedback(state, packet)


def build_research_workflow(
    *,
    main_agent: MainResearchAgent,
    planning: EDAPlanningService,
    execution: EDAExecutionService,
    skills: SkillRegistry,
    tools: ToolRegistry,
    checkpointer: Any,
    result_store: ToolResultStore,
    dynamic_agent: DynamicAnalysisAgent | None = None,
):
    """Compile the only workflow used by desktop research sessions."""

    def skill_version_feedback(state: ResearchLoopState) -> FeedbackPacket | None:
        plan = _plan(state)
        scope = _scope(state)
        skill_value = state.get("active_skill") or {}
        skill_name = (
            plan.skill_name
            if plan is not None
            else (scope.skill_name if scope is not None else str(skill_value.get("name") or ""))
        )
        locked_version = (
            plan.skill_version
            if plan is not None
            else (scope.skill_version if scope is not None else str(skill_value.get("version") or ""))
        )
        if not skill_name or not locked_version:
            return None
        try:
            installed_version = skills.get(skill_name).version
        except Exception:  # noqa: BLE001 - missing Skills use the existing validator path
            return None
        if installed_version == locked_version:
            return None
        return FeedbackPacket(
            source="skill_validator",
            code="session_skill_version_mismatch",
            severity="error",
            message=(
                f"当前对话使用研究协议 {skill_name}@{locked_version}，"
                f"应用当前版本为 {installed_version}。请新建对话重新分析；此前报告仍可查看。"
            ),
            observed={"skill": skill_name, "session_version": locked_version},
            expected={"installed_version": installed_version},
            recommendation="新建对话并重新提交研究问题；不要在旧 checkpoint 上迁移或继续执行方案。",
            requires_user=True,
        )

    def _blocking_feedback(packets: list[dict[str, Any]]) -> FeedbackPacket | None:
        for item in reversed(packets):
            packet = FeedbackPacket.model_validate(item)
            if packet.severity in {"error", "fatal"}:
                return packet
        return None

    def progress_stall_code(state: ResearchLoopState) -> str | None:
        if (
            state.get("plan_origin") != "automatic_evaluation"
            or (state.get("evaluation") or {}).get("decision") != "revise"
        ):
            return None
        cycle_id = state.get("revision_cycle_id")
        records = [
            item
            for item in state.get("progress_records", [])
            if item.get("revision_cycle_id") == cycle_id and item.get("evaluation_completed")
        ]
        if len(records) < 2:
            return None
        current = records[-1]
        previous = records[:-1]
        current_agenda = current.get("agenda_fingerprint")
        if current_agenda and current_agenda in {
            item.get("agenda_fingerprint") for item in previous
        }:
            return "no_agenda_progress"
        current_evidence = current.get("evidence_fingerprint")
        if current_evidence in {item.get("evidence_fingerprint") for item in previous}:
            return "no_new_evidence"
        return None

    def ingest_user(state: ResearchLoopState) -> dict[str, Any]:
        message = state.get("pending_user_message", "").strip()
        turn_id = state.get("pending_turn_id")
        messages = _messages_before_user_turn(
            list(state.get("messages", [])),
            question=message,
            turn_id=turn_id,
        )
        if message:
            messages.append(
                _conversation_message(
                    role="user",
                    content=message,
                    message_id=state.get("pending_message_id"),
                    turn_id=turn_id,
                )
            )
        return {
            "phase": "understanding",
            "control": "understand",
            "user_interrupt_kind": None,
            "latest_turn": message or _latest_turn(state),
            "active_turn_id": turn_id or state.get("active_turn_id"),
            "pending_user_message": "",
            "pending_message_id": None,
            "pending_turn_id": None,
            "messages": _bounded_graph_messages(messages),
            "events": _event(
                state,
                f"接收研究问题：{_short(message)}",
                trace_category="user",
                message_length=len(message),
                message_hash=canonical_hash(message),
            ),
        }

    def understand(state: ResearchLoopState) -> dict[str, Any]:
        if packet := skill_version_feedback(state):
            return {
                "phase": "awaiting_user",
                "control": "need_user",
                "feedback_packets": _append_feedback(state, packet),
                "stop_reason": packet.message,
                "user_interrupt_kind": "plan_error",
                "events": _event(
                    state,
                    f"旧会话版本不兼容：{_short(packet.message, 44)}",
                    "failed",
                    trace_category="error",
                ),
            }
        try:
            routed = state.get("routed_decision")
            if routed is not None:
                decision = DialogueDecision.model_validate(routed)
            else:
                skill_metadata = skills.metadata()
                decision, _ = main_agent.decide(
                    question=_latest_turn(state),
                    status=state.get("phase", "idle"),
                    config=_config(state),
                    plan=_plan(state),
                    scope=_scope(state),
                    data_profile=state.get("data_profile"),
                    quality_report=state.get("quality_report"),
                    summary=state.get("eda_summary"),
                    evaluation=state.get("evaluation"),
                    history=_messages(state),
                    available_skills=skill_metadata,
                    episode_summaries=_scoped_episode_summaries(state),
                    active_gate=state.get("user_interrupt_kind") or state.get("return_to_gate"),
                    episode_goal=_episode_goal(state),
                    latest_run=state.get("latest_run"),
                    current_turn_id=state.get("active_turn_id"),
                    has_executable_data=bool(state.get("has_executable_data")),
                )
            active_gate = state.get("user_interrupt_kind") or state.get("return_to_gate")
            if (
                decision.intent == "execute_plan"
                and active_gate in {"result_limitations", "result_rejected"}
                and state.get("latest_run")
            ):
                decision = DialogueDecision(
                    intent="discussion",
                    response=(
                        "当前方案已经执行并完成评估；原样重跑不会处理现有限制。"
                        "请修改方案以补充所需证据，或明确接受当前结果及限制。"
                    ),
                )
        except Exception as exc:  # noqa: BLE001 - converted into structured loop feedback
            packet = exception_feedback(
                exc,
                source="plan_validator",
                retryable=isinstance(
                    exc,
                    (
                        ResearchModelUnavailableError,
                        ResearchModelOutputTruncatedError,
                        ResearchModelContextLimitError,
                    ),
                ),
                requires_user=True,
            )
            return {
                "phase": "awaiting_user",
                # Route straight to the pause: without a decision the reply node
                # would only add a second, misleading schema error.
                "control": "need_user",
                "feedback_packets": _append_feedback(state, packet),
                "stop_reason": packet.message,
                "user_interrupt_kind": _dialogue_interrupt_kind(state),
                "events": _event(
                    state,
                    f"主 Agent 调用失败：{_short(packet.message)}",
                    "failed",
                    trace_category="error",
                    error=packet.message,
                ),
            }
        update = {
            "phase": (
                "planning"
                if decision.intent in {"new_plan", "revise_plan", "new_forecast_plan"}
                else "understanding"
            ),
            "control": decision.intent if decision.intent != "discussion" else "reply",
            "decision": decision.model_dump(mode="json"),
            "routed_decision": None,
            "post_analysis_action": decision.post_analysis_action,
            "events": _event(
                state,
                f"主 Agent 路由：{decision.intent}",
                intent=decision.intent,
                response=_short(decision.response),
            ),
        }
        if decision.intent == "revise_plan":
            update["revision_cycle_id"] = state.get("active_turn_id") or state.get("revision_cycle_id")
        return update

    def route_main(
        state: ResearchLoopState,
    ) -> Literal[
        "resolve_data",
        "revise_plan",
        "execute_plan",
        "business_handoff",
        "reply",
        "need_user",
    ]:
        control = state.get("control", "reply")
        if control in {"new_news_analysis", "new_forecast_plan", "execute_forecast_plan"}:
            return "business_handoff"
        if control == "new_plan" or (
            control in {"revise_plan", "execute_plan"} and _config(state) is None
        ):
            return "resolve_data"
        if control == "execute_plan" and _plan(state) is None and _scope(state) is None:
            return "reply"
        return control if control in {"revise_plan", "execute_plan", "need_user"} else "reply"  # type: ignore[return-value]

    def resolve_study_context(state: ResearchLoopState) -> dict[str, Any]:
        """Resolve selected files only after the main Agent chose a data-analysis route."""

        action = str(state.get("pending_research_action") or state.get("control") or "new_plan")
        config = _config(state)
        if config is None:
            descriptor = _study_input(state)
            if descriptor is None:
                packet = FeedbackPacket(
                    source="data_loader",
                    code="missing_study_input",
                    severity="error",
                    message="开始实际数据分析前需要先选择目标电价数据。",
                    recommendation="选择数据后重新提交分析请求。",
                    requires_user=True,
                )
                return {
                    "phase": "awaiting_user",
                    "control": "need_user",
                    "pending_research_action": None,
                    "feedback_packets": _append_feedback(state, packet),
                    "stop_reason": packet.message,
                    "user_interrupt_kind": "data_input_error",
                    "events": _event(
                        state,
                        "实际分析缺少数据输入",
                        "failed",
                        trace_category="error",
                    ),
                }
            try:
                config = resolve_study_input(descriptor)
            except Exception as exc:  # noqa: BLE001 - converted into a data-input interrupt
                packet = exception_feedback(exc, source="data_loader", requires_user=True)
                return {
                    "phase": "awaiting_user",
                    "control": "need_user",
                    "pending_research_action": None,
                    "feedback_packets": _append_feedback(state, packet),
                    "stop_reason": packet.message,
                    "user_interrupt_kind": "data_input_error",
                    "events": _event(
                        state,
                        f"研究数据解析失败：{_short(packet.message)}",
                        "failed",
                        trace_category="error",
                    ),
                }
        return {
            "phase": "planning",
            "control": action,
            "pending_research_action": None,
            "study_config": config.model_dump(mode="json"),
            "has_executable_data": True,
            "events": _event(
                state,
                f"实际分析数据已解析：{config.study.name}",
                trace_category="input",
            ),
        }

    def route_resolved_study(
        state: ResearchLoopState,
    ) -> Literal["new_plan", "revise_plan", "execute_plan", "need_user"]:
        control = state.get("control", "need_user")
        return control if control in {"new_plan", "revise_plan", "execute_plan"} else "need_user"  # type: ignore[return-value]

    def prepare_forecast_handoff(state: ResearchLoopState) -> dict[str, Any]:
        """Hand specialized business intent to the application that owns its inputs."""

        intent = DialogueDecision.model_validate(state["decision"]).intent
        control = {
            "execute_forecast_plan": "forecast_execute_request",
            "new_news_analysis": "news_analysis_request",
        }.get(intent, "forecast_plan_request")
        return {
            "phase": "awaiting_user",
            "control": control,
            "assistant_message": "",
            "events": _event(
                state,
                {
                    "forecast_plan_request": "移交固定预测方案入口",
                    "forecast_execute_request": "校验预测方案确认入口",
                    "news_analysis_request": "移交电价新闻分析入口",
                }[control],
                trace_category="plan",
                forecast_control=control,
            ),
        }

    def confirm_existing_plan(state: ResearchLoopState) -> dict[str, Any]:
        active = _scope(state) or _plan(state)
        return {
            "phase": "validating_plan",
            "control": "validate",
            "plan_origin": "explicit_confirm",
            "events": _event(
                state,
                f"确认已有研究范围或方案：{getattr(active, 'scope_id', None) or getattr(active, 'plan_id', 'unknown')}",
                trace_category="plan",
            ),
        }

    def resolve_skill(state: ResearchLoopState) -> dict[str, Any]:
        decision = DialogueDecision.model_validate(state["decision"])
        try:
            if not decision.skill_name:
                raise SkillLoadError("大模型生成新方案时没有选择 Skill。")
            skill = skills.get(decision.skill_name)
            if skill.domain != "eda":
                raise SkillLoadError(f"当前循环不支持 {skill.domain} Skill：{skill.name}")
            tools.function_schemas(skill.allowed_functions)
            if "data_quality" not in skill.allowed_functions:
                raise SkillLoadError(f"EDA Skill 必须授权 data_quality：{skill.name}")
        except Exception as exc:  # noqa: BLE001 - converted into structured loop feedback
            packet = exception_feedback(exc, source="skill_validator", requires_user=True)
            return {
                "phase": "awaiting_user",
                "control": "need_user",
                "feedback_packets": _append_feedback(state, packet),
                "stop_reason": packet.message,
                "user_interrupt_kind": "plan_error",
                "events": _event(
                    state,
                    f"Skill 校验失败：{_short(packet.message)}",
                    "failed",
                    trace_category="error",
                    error=packet.message,
                ),
            }
        protocol = skill.research_protocol
        activation_name = f"激活 Skill：{skill.name}@{skill.version}"
        if protocol is not None:
            activation_name = (
                f"激活 Skill 与领域协议：{skill.name}@{skill.version} · "
                f"{protocol.protocol_id}@{protocol.version}"
            )
        return {
            # The registry is the authority for Skill content.  Persisting the
            # whole prompt/protocol in every checkpoint needlessly duplicated
            # a large immutable object.
            "active_skill": {"name": skill.name, "version": skill.version},
            "phase": "planning",
            "control": "plan",
            "plan_origin": "initial",
            "events": _event(
                state,
                activation_name,
                trace_category="plan",
                skill=skill.name,
                version=skill.version,
                research_protocol=(
                    {"protocol_id": protocol.protocol_id, "version": protocol.version}
                    if protocol is not None
                    else None
                ),
            ),
        }

    def route_skill(state: ResearchLoopState) -> Literal["plan", "need_user"]:
        return "need_user" if state.get("control") == "need_user" else "plan"

    def revise_user_plan(state: ResearchLoopState) -> dict[str, Any]:
        plan = _plan(state)
        scope = _scope(state)
        config = _config(state)
        decision = DialogueDecision.model_validate(state["decision"])
        if scope is not None and plan is None and config is not None:
            changed_fields = (
                decision.objective,
                decision.enabled_functions,
                decision.selected_variables,
            )
            if all(value is None for value in changed_fields):
                packet = FeedbackPacket(
                    source="plan_validator",
                    code="empty_scope_revision",
                    severity="error",
                    message="没有识别到研究目标、工具范围或变量范围的具体修改。",
                    retryable=True,
                )
                return {
                    "phase": "awaiting_user",
                    "control": "need_user",
                    "feedback_packets": _append_feedback(state, packet),
                    "user_interrupt_kind": "plan_error",
                }
            functions = (
                list(scope.authorized_functions)
                if decision.enabled_functions is None
                else list(dict.fromkeys(decision.enabled_functions))
            )
            functions = [name for name in functions if name != "data_quality"]
            outside_functions = sorted(set(functions).difference(scope.authorized_functions))
            variables = (
                list(scope.authorized_variables)
                if decision.selected_variables is None
                else list(dict.fromkeys(decision.selected_variables))
            )
            outside_variables = sorted(set(variables).difference(scope.authorized_variables))
            if outside_functions or outside_variables:
                packet = FeedbackPacket(
                    source="plan_validator",
                    code="scope_revision_expands_authority",
                    severity="error",
                    message="研究范围修改只能收窄；新增工具或变量需要重新发起研究。",
                    observed={"functions": outside_functions, "variables": outside_variables},
                    requires_user=True,
                )
                return {
                    "phase": "awaiting_user",
                    "control": "need_user",
                    "feedback_packets": _append_feedback(state, packet),
                    "user_interrupt_kind": "plan_error",
                }
            revised_scope = scope.model_copy(
                update={
                    "scope_id": uuid4().hex[:12],
                    "revision": scope.revision + 1,
                    "objective": decision.objective or scope.objective,
                    "authorized_functions": functions,
                    "authorized_variables": variables,
                }
            )
            if scope_fingerprint(revised_scope) == scope_fingerprint(scope):
                packet = FeedbackPacket(
                    source="plan_validator",
                    code="unchanged_scope_revision",
                    severity="error",
                    message="修改内容与当前研究范围相同，请说明要收窄的目标、工具或变量。",
                    retryable=True,
                )
                return {
                    "phase": "awaiting_user",
                    "control": "need_user",
                    "feedback_packets": _append_feedback(state, packet),
                    "user_interrupt_kind": "plan_error",
                }
            assistant_message = decision.response.strip() or "已按你的要求收窄研究范围，请重新确认。"
            return {
                "research_scope": revised_scope.model_dump(mode="json"),
                "plan_history": [*state.get("plan_history", []), revised_scope.model_dump(mode="json")],
                "plan_fingerprints": [
                    *state.get("plan_fingerprints", []),
                    scope_fingerprint(revised_scope),
                ],
                "feedback_history": _archive_feedback(state),
                "feedback_packets": [],
                "plan_origin": "user_revision",
                "return_to_gate": None,
                "phase": "validating_plan",
                "control": "validate",
                "stop_reason": None,
                "user_interrupt_kind": None,
                "assistant_message": assistant_message,
                "messages": _bounded_graph_messages(
                    [
                        *state.get("messages", []),
                        _conversation_message(
                            role="assistant",
                            content=assistant_message,
                            turn_id=state.get("active_turn_id"),
                            episode_id=_cursor(state).episode_id,
                        ),
                    ]
                ),
                "events": _event(
                    state,
                    f"修订研究范围：{revised_scope.scope_id}",
                    trace_category="plan",
                    revision=revised_scope.revision,
                ),
            }
        if plan is None or config is None:
            packet = FeedbackPacket(
                source="plan_validator",
                code="missing_plan_for_revision",
                severity="error",
                message="当前没有可修订方案。",
                requires_user=True,
            )
            return {
                "phase": "awaiting_user",
                "control": "need_user",
                "feedback_packets": _append_feedback(state, packet),
                "user_interrupt_kind": "plan_error",
            }
        try:
            revised, assistant_message = main_agent.revise_plan(
                question=_latest_turn(state),
                plan=plan,
                config=config,
                decision=decision,
                previous_evaluation=state.get("evaluation"),
                quality_report=(
                    DataQualityReport.model_validate(state["quality_report"])
                    if state.get("quality_report")
                    else None
                ),
            )
        except Exception as exc:  # noqa: BLE001 - converted into structured loop feedback
            packet = exception_feedback(exc, source="plan_validator", retryable=True)
            return {
                "phase": "planning",
                "control": "retry",
                "feedback_packets": _append_feedback(state, packet),
                "plan_origin": "user_revision",
                "stop_reason": packet.message,
                "user_interrupt_kind": None,
            }
        return {
            "current_plan": revised.model_dump(mode="json"),
            "plan_history": [*state.get("plan_history", []), revised.model_dump(mode="json")],
            "plan_fingerprints": [*state.get("plan_fingerprints", []), plan_fingerprint(revised)],
            "feedback_history": _archive_feedback(state),
            "feedback_packets": [],
            "evaluation": None,
            "plan_origin": "user_revision",
            "return_to_gate": None,
            "phase": "validating_plan",
            "control": "validate",
            "stop_reason": None,
            "user_interrupt_kind": None,
            "messages": _bounded_graph_messages([
                *state.get("messages", []),
                _conversation_message(
                    role="assistant",
                    content=assistant_message,
                    turn_id=state.get("active_turn_id"),
                    episode_id=_cursor(state).episode_id,
                ),
            ]),
            "events": _event(
                state,
                f"生成修订方案：{revised.plan_id} · v{revised.revision}",
                trace_category="plan",
                revision=revised.revision,
                functions=[step.function for step in revised.enabled_steps],
            ),
        }

    def route_user_revision(state: ResearchLoopState) -> Literal["validate", "retry", "need_user"]:
        control = state.get("control", "validate")
        return control if control in {"retry", "need_user"} else "validate"  # type: ignore[return-value]

    def create_plan(state: ResearchLoopState) -> dict[str, Any]:
        budget = _budget(state)
        budget.plan_attempts_in_iteration += 1
        cursor = _cursor(state).model_copy(
            update={
                "plan_attempt_number": budget.plan_attempts_in_iteration,
                "call_attempt_number": 0,
                "current_call_id": None,
                "stage": "planning",
                "episode_status": "planning",
            }
        )
        config = _config(state)
        skill_value = state.get("active_skill")
        if config is None or not skill_value:
            packet = FeedbackPacket(
                source="plan_validator" if config is None else "skill_validator",
                code="missing_planning_context",
                severity="error",
                message="规划缺少研究数据或已激活 Skill。",
                requires_user=True,
            )
            return {
                "phase": "awaiting_user",
                "control": "need_user",
                "feedback_packets": _append_feedback(state, packet),
                "user_interrupt_kind": "plan_error",
                "loop_cursor": cursor.model_dump(mode="json"),
                "budget": budget.model_dump(mode="json"),
            }
        if packet := skill_version_feedback(state):
            return {
                "phase": "awaiting_user",
                "control": "need_user",
                "feedback_packets": _append_feedback(state, packet),
                "stop_reason": packet.message,
                "user_interrupt_kind": "plan_error",
                "loop_cursor": cursor.model_dump(mode="json"),
                "budget": budget.model_dump(mode="json"),
            }
        try:
            skill = skills.get(str(skill_value.get("name") or ""))
            if skill.version != skill_value.get("version"):
                raise SkillLoadError("当前 Skill 版本与 checkpoint 中锁定的版本不一致，请重新生成方案。")
        except Exception as exc:  # noqa: BLE001 - converted into structured feedback
            packet = exception_feedback(exc, source="skill_validator", requires_user=True)
            return {
                "phase": "awaiting_user",
                "control": "need_user",
                "feedback_packets": _append_feedback(state, packet),
                "user_interrupt_kind": "plan_error",
                "loop_cursor": cursor.model_dump(mode="json"),
                "budget": budget.model_dump(mode="json"),
            }
        feedback = _feedback(state)
        origin = state.get("plan_origin", "initial")
        try:
            if dynamic_agent is not None and origin != "automatic_evaluation":
                scope_proposal = planning.propose_scope(
                    question=_episode_goal(state),
                    study_config=config,
                    conversation=_messages(state),
                    progress=noop_progress,
                    skill=skill,
                )
                proposal = None
            elif origin == "automatic_evaluation" and _plan(state) is not None:
                proposal = planning.revise_from_feedback(
                    current_plan=_plan(state),  # type: ignore[arg-type]
                    study_config=config,
                    skill=skill,
                    feedback=feedback,
                    authorization_envelope=state.get("authorization_envelope"),
                    conversation=_messages(state),
                    episode_summaries=_scoped_episode_summaries(state),
                    source="automatic_evaluation",
                    progress=noop_progress,
                )
            else:
                proposal = planning.propose(
                    question=_episode_goal(state),
                    study_config=config,
                    conversation=_messages(state),
                    # The planning service computes the new study fingerprint
                    # before retrieval, so it can safely select matching memory
                    # even though begin_episode has cleared current_plan.
                    episode_summaries=state.get("episode_summaries", []),
                    progress=noop_progress,
                    skill=skill,
                    feedback=feedback,
                )
                scope_proposal = None
        except Exception as exc:  # noqa: BLE001 - model/validator boundary
            packet = exception_feedback(exc, source="plan_validator", retryable=True)
            failure_fingerprint = canonical_hash(
                {
                    "request": _episode_goal(state),
                    "iteration_id": cursor.iteration_id,
                    "catalog": FUNCTION_CATALOG_VERSION,
                    "code": packet.code,
                    "message": packet.message,
                }
            )
            failures = list(state.get("planning_failure_fingerprints", []))
            repeated_failure = failure_fingerprint in failures
            if not repeated_failure:
                failures.append(failure_fingerprint)
            if (
                not repeated_failure
                and budget.plan_attempts_in_iteration < budget.max_plan_attempts_per_iteration
            ):
                phase = "planning"
            else:
                packet = packet.model_copy(update={"retryable": False, "requires_user": True})
                phase = "awaiting_user"
            return {
                "phase": phase,
                "control": "retry" if phase == "planning" else "need_user",
                "feedback_packets": _append_feedback(state, packet),
                "planning_failure_fingerprints": failures,
                "loop_cursor": cursor.model_dump(mode="json"),
                "budget": budget.model_dump(mode="json"),
                "stop_reason": packet.message if phase == "awaiting_user" else None,
                "user_interrupt_kind": _planning_interrupt_kind(state) if phase == "awaiting_user" else None,
                "events": _event(
                    state,
                    f"方案生成失败：{_short(packet.message)}",
                    "warning",
                    trace_category="error",
                    error=packet.message,
                ),
            }
        if dynamic_agent is not None and origin != "automatic_evaluation":
            scope = scope_proposal.scope
            fingerprint = scope_fingerprint(scope)
            if fingerprint in state.get("plan_fingerprints", []):
                packet = budget_feedback(budget, code="duplicate_scope", message="模型生成了重复研究范围。")
                return {
                    "phase": "awaiting_user",
                    "control": "need_user",
                    "feedback_packets": _append_feedback(state, packet),
                    "stop_reason": packet.message,
                    "user_interrupt_kind": "plan_error",
                    "loop_cursor": cursor.model_dump(mode="json"),
                    "budget": budget.model_dump(mode="json"),
                }
            scope_payload = scope.model_dump(mode="json")
            return {
                "phase": "validating_plan",
                "control": "validate",
                "user_interrupt_kind": None,
                "feedback_history": _archive_feedback(state),
                "feedback_packets": [],
                "loop_cursor": cursor.model_dump(mode="json"),
                "research_scope": scope_payload,
                "current_plan": None,
                "data_profile": scope_proposal.data_profile.model_dump(mode="json"),
                "data_summary": summary_payload(scope_proposal.data_summary),
                "quality_report": scope_proposal.quality_report.model_dump(mode="json"),
                "plan_history": [*state.get("plan_history", []), scope_payload],
                "plan_fingerprints": [*state.get("plan_fingerprints", []), fingerprint],
                "budget": budget.model_copy(
                    update={
                        "max_model_rounds": scope.max_model_rounds,
                        "max_tool_calls": scope.max_tool_calls,
                        "max_function_attempts_per_call": scope.max_attempts_per_call,
                    }
                ).model_dump(mode="json"),
                "assistant_message": scope_proposal.assistant_message,
                "stop_reason": None,
                "events": _event(
                    state,
                    f"生成研究范围：{scope.scope_id} · {_short(scope.objective, 42)}",
                    trace_category="plan",
                    scope_id=scope.scope_id,
                    authorized_functions=scope.authorized_functions,
                ),
            }
        fingerprint = plan_fingerprint(proposal.plan)
        if fingerprint in state.get("plan_fingerprints", []):
            packet = budget_feedback(budget, code="duplicate_plan", message="自动修订生成了重复方案。")
            return {
                "phase": "awaiting_user",
                "control": "need_user",
                "feedback_packets": _append_feedback(state, packet),
                "stop_reason": packet.message,
                "user_interrupt_kind": _planning_interrupt_kind(state),
                "loop_cursor": cursor.model_dump(mode="json"),
                "budget": budget.model_dump(mode="json"),
            }
        return {
            "phase": "validating_plan",
            "control": "validate",
            "user_interrupt_kind": None,
            "feedback_history": _archive_feedback(state),
            "feedback_packets": [],
            "loop_cursor": cursor.model_dump(mode="json"),
            "current_plan": proposal.plan.model_dump(mode="json"),
            "data_profile": proposal.data_profile.model_dump(mode="json"),
            "data_summary": summary_payload(proposal.data_summary),
            "quality_report": proposal.quality_report.model_dump(mode="json"),
            "plan_history": [*state.get("plan_history", []), proposal.plan.model_dump(mode="json")],
            "plan_fingerprints": [*state.get("plan_fingerprints", []), fingerprint],
            "budget": budget.model_dump(mode="json"),
            "stop_reason": None,
            "events": _event(
                state,
                f"生成候选方案：{proposal.plan.plan_id} · {_short(proposal.plan.objective, 42)}",
                trace_category="plan",
                plan_id=proposal.plan.plan_id,
                functions=[step.function for step in proposal.plan.enabled_steps],
            ),
        }

    def route_after_plan(state: ResearchLoopState) -> Literal["retry", "validate", "need_user"]:
        control = state.get("control", "validate")
        return control if control in {"retry", "need_user"} else "validate"  # type: ignore[return-value]

    def validate_plan(state: ResearchLoopState) -> dict[str, Any]:
        plan = _plan(state)
        scope = _scope(state)
        config = _config(state)
        if scope is not None:
            if config is None:
                packet = FeedbackPacket(
                    source="plan_validator",
                    code="missing_scope_config",
                    severity="fatal",
                    message="研究范围缺少运行数据契约。",
                    requires_user=True,
                )
                return {
                    "phase": "awaiting_user",
                    "control": "need_user",
                    "feedback_packets": _append_feedback(state, packet),
                    "user_interrupt_kind": "plan_error",
                }
            try:
                execution.prepare_scope(scope=scope, study_config=config)
            except Exception as exc:  # noqa: BLE001 - scope validation boundary
                packet = exception_feedback(exc, source="plan_validator", requires_user=True)
                return {
                    "phase": "awaiting_user",
                    "control": "need_user",
                    "feedback_packets": _append_feedback(state, packet),
                    "stop_reason": packet.message,
                    "user_interrupt_kind": "plan_error",
                }
            control = "lock" if _scope_approval_matches(state, scope) else "approval"
            return {
                "phase": "validating_plan",
                "control": control,
                "user_interrupt_kind": None,
                "events": _event(
                    state,
                    f"研究范围校验通过：{scope.scope_id}",
                    trace_category="plan",
                    authorized_functions=scope.authorized_functions,
                ),
            }
        if plan is None or config is None:
            packet = FeedbackPacket(
                source="plan_validator",
                code="missing_plan_or_config",
                severity="fatal",
                message="计划或运行数据契约缺失。",
                requires_user=True,
            )
            return {
                "phase": "awaiting_user",
                "control": "need_user",
                "feedback_packets": _append_feedback(state, packet),
                "user_interrupt_kind": _planning_interrupt_kind(state),
            }
        if packet := skill_version_feedback(state):
            return {
                "phase": "awaiting_user",
                "control": "need_user",
                "feedback_packets": _append_feedback(state, packet),
                "stop_reason": packet.message,
                "user_interrupt_kind": "plan_error",
            }
        try:
            execution.prepare(plan=plan, study_config=config)
        except Exception as exc:  # noqa: BLE001 - validator boundary
            retryable = isinstance(exc, RepairablePlanError)
            requires_user = isinstance(
                exc,
                (
                    PlanCompatibilityError,
                    DataFingerprintMismatchError,
                    InsufficientDataError,
                    ResearchDataError,
                ),
            )
            if isinstance(exc, SkillVersionMismatchError):
                packet = skill_version_feedback(state) or exception_feedback(
                    exc,
                    source="skill_validator",
                    requires_user=True,
                )
            else:
                packet = exception_feedback(
                    exc,
                    source="plan_validator",
                    retryable=retryable,
                    requires_user=requires_user,
                )
            budget = _budget(state)
            if (
                retryable
                and budget.plan_attempts_in_iteration < budget.max_plan_attempts_per_iteration
            ):
                phase = "planning"
            else:
                phase = "awaiting_user"
                packet = packet.model_copy(update={"retryable": False, "requires_user": True})
            return {
                "phase": phase,
                "control": "retry" if phase == "planning" else "need_user",
                "feedback_packets": _append_feedback(state, packet),
                "budget": budget.model_dump(mode="json"),
                "stop_reason": packet.message if phase == "awaiting_user" else None,
                "user_interrupt_kind": _planning_interrupt_kind(state) if phase == "awaiting_user" else None,
            }
        if state.get("plan_origin") == "automatic_evaluation" and state.get("authorization_envelope"):
            violation = validate_automatic_revision(
                plan,
                AuthorizationEnvelope.model_validate(state["authorization_envelope"]),
            )
            if violation is not None:
                return {
                    "phase": "awaiting_user",
                    "control": "need_user",
                    "feedback_packets": _append_feedback(state, violation),
                    "stop_reason": violation.message,
                    "user_interrupt_kind": _planning_interrupt_kind(state),
                }
        if state.get("plan_origin") == "automatic_evaluation":
            control = "lock"
        else:
            control = "lock" if _approval_matches(state, plan) else "approval"
        return {
            "phase": "validating_plan",
            "control": control,
            "user_interrupt_kind": None,
            "loop_cursor": _cursor(state).model_copy(
                update={"episode_status": "planning", "stage": "plan_validation"}
            ).model_dump(mode="json"),
            "events": _event(
                state,
                f"方案校验通过：{plan.plan_id}",
                trace_category="plan",
                functions=[step.function for step in plan.enabled_steps],
            ),
        }

    def begin_episode(state: ResearchLoopState) -> dict[str, Any]:
        previous = _cursor(state)
        history = list(state.get("episode_history", []))
        episode_summaries = _upsert_episode_summary(
            list(state.get("episode_summaries", [])),
            _episode_summary(state, cursor=previous),
        )
        if previous.episode_id and not any(item.get("episode_id") == previous.episode_id for item in history):
            history.append(
                {
                    **previous.model_dump(mode="json"),
                    "budget": state.get("budget", {}),
                    "plan_id": (state.get("current_plan") or {}).get("plan_id"),
                    "latest_run_id": (state.get("latest_run") or {}).get("run_id"),
                }
            )
        history = history[-MAX_EPISODE_HISTORY:]
        cursor = LoopCursor.start_episode(
            number=previous.episode_number + 1,
            goal=_latest_turn(state),
        )
        budget = _budget(state).reset_for_episode()
        return {
            "phase": "planning",
            "user_request": cursor.episode_goal,
            "explanation_request": "",
            "loop_cursor": cursor.model_dump(mode="json"),
            "revision_cycle_id": cursor.episode_id,
            "episode_history": history,
            "episode_summaries": episode_summaries,
            "active_skill": None,
            "current_plan": None,
            "research_scope": None,
            "plan_history": [],
            "plan_fingerprints": [],
            "planning_failure_fingerprints": [],
            "plan_origin": "initial",
            "return_to_gate": None,
            "authorization_envelope": None,
            "approval_state": ApprovalState().model_dump(mode="json"),
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
            "evaluation": None,
            "eda_summary": None,
            "feedback_packets": [],
            "feedback_history": _archive_feedback(state),
            "budget": budget.model_dump(mode="json"),
            "evidence_fingerprints": [],
            "agenda_fingerprints": [],
            "latest_run": None,
            "assistant_message": "",
            "stop_reason": None,
            "user_interrupt_kind": None,
            "events": _event(
                state,
                f"开始研究 Episode：{cursor.episode_id} · {_short(cursor.episode_goal, 42)}",
                trace_category="plan",
                episode_id=cursor.episode_id,
                episode_number=cursor.episode_number,
                iteration_id=cursor.iteration_id,
            ),
        }

    def route_after_validation(state: ResearchLoopState) -> Literal["retry", "approval", "lock", "need_user"]:
        control = state.get("control", "approval")
        return control if control in {"retry", "lock", "need_user"} else "approval"  # type: ignore[return-value]

    def prepare_approval(state: ResearchLoopState) -> dict[str, Any]:
        plan = _plan(state)
        scope = _scope(state)
        timeout = int(state.get("approval_timeout_seconds", 30))
        automatic_approval_enabled = bool(state.get("automatic_approval_enabled", False))
        approval = ApprovalState(
            status="waiting",
            plan_id=scope.scope_id if scope else (plan.plan_id if plan else None),
            plan_fingerprint=scope_fingerprint(scope) if scope else (plan_fingerprint(plan) if plan else None),
            deadline=(datetime.now(UTC) + timedelta(seconds=timeout)).isoformat()
            if automatic_approval_enabled
            else None,
            remaining_seconds=timeout if automatic_approval_enabled else None,
            origin="user_revision" if state.get("plan_origin") == "user_revision" else "initial",
        )
        return {
            "phase": "awaiting_approval",
            "control": "interrupt",
            "return_to_gate": None,
            "loop_cursor": _cursor(state).model_copy(
                update={"episode_status": "awaiting_approval", "stage": "approval_gate"}
            ).model_dump(mode="json"),
            "approval_state": approval.model_dump(mode="json"),
            "events": _event(
                state,
                f"等待审批：{approval.plan_id or 'unknown'}",
                trace_category="plan",
                deadline=approval.deadline,
                automatic=automatic_approval_enabled,
            ),
        }

    def approval_interrupt(state: ResearchLoopState) -> dict[str, Any]:
        approval = ApprovalState.model_validate(state["approval_state"])
        payload = _interrupt_payload(
            state,
            kind="plan_approval",
            phase="awaiting_approval",
            message="请确认、拒绝，或继续询问和修改当前研究方案。",
            choices=["approve", "modify", "reject", "followup"],
            plan=state.get("current_plan"),
            scope=state.get("research_scope"),
            deadline=approval.deadline,
            remaining_seconds=approval.remaining_seconds,
        )
        response = ResumePayload.model_validate(interrupt(payload.model_dump(mode="json")))
        if not _resume_identity_matches(response, payload):
            return {
                "phase": "awaiting_approval",
                "control": "reinterrupt",
                "feedback_packets": _stale_resume_feedback(state, response, payload),
            }
        automatic_timeout = (
            response.action == "timeout_accept"
            and response.automatic_timeout
            and bool(state.get("automatic_approval_enabled", False))
        )
        if response.action not in payload.choices and not automatic_timeout:
            return {
                "phase": "awaiting_approval",
                "control": "reinterrupt",
                "feedback_packets": _invalid_resume_feedback(state, response, payload),
                "events": _event(
                    state,
                    f"拒绝越界审批操作：{response.action}",
                    "warning",
                    trace_category="error",
                ),
            }
        if automatic_timeout:
            try:
                deadline_reached = bool(
                    approval.status == "waiting"
                    and approval.deadline
                    and datetime.fromisoformat(approval.deadline) <= datetime.now(UTC)
                )
            except ValueError:
                deadline_reached = False
            if not deadline_reached:
                packet = FeedbackPacket(
                    source="user",
                    code="invalid_automatic_approval_timeout",
                    severity="error",
                    message="自动审批只能由前台倒计时在截止时间到达后触发。",
                    observed={"status": approval.status, "deadline": approval.deadline},
                    recommendation="等待倒计时结束，或由用户明确确认方案。",
                    requires_user=True,
                )
                return {
                    "phase": "awaiting_approval",
                    "control": "reinterrupt",
                    "feedback_packets": _append_feedback(state, packet),
                }
        update: dict[str, Any]
        if response.action in {"approve", "timeout_accept"}:
            approval.status = "approved"
            approval.decision = response.action
            approval.approved_at = _now()
            update = {"phase": "validating_plan", "control": "lock", "return_to_gate": None}
        elif response.action == "followup":
            message = response.message.strip()
            if not message:
                packet = FeedbackPacket(
                    source="user",
                    code="empty_plan_followup",
                    severity="error",
                    message="方案讨论消息不能为空。",
                    recommendation="请输入问题或修改意见，或使用方案卡上的明确操作。",
                    requires_user=True,
                )
                return {
                    "phase": "awaiting_approval",
                    "control": "reinterrupt",
                    "feedback_packets": _append_feedback(state, packet),
                }
            update = {
                "phase": "understanding",
                "control": "discuss",
                "latest_turn": message,
                "active_turn_id": response.turn_id or state.get("active_turn_id"),
                "return_to_gate": "plan_approval",
                "messages": _bounded_graph_messages([
                    *_messages_before_user_turn(
                        _resume_messages(state, response),
                        question=message,
                        turn_id=response.turn_id,
                    ),
                    _conversation_message(
                        role="user",
                        content=message,
                        message_id=response.message_id,
                        turn_id=response.turn_id,
                        episode_id=_cursor(state).episode_id,
                    ),
                ]),
            }
        elif response.action == "modify":
            approval.status = "none"
            approval.decision = "modify"
            approval.plan_id = None
            approval.plan_fingerprint = None
            approval.approved_at = None
            message = response.message.strip()
            update = {
                "phase": "understanding",
                "control": "modify",
                "latest_turn": message,
                "active_turn_id": response.turn_id or state.get("active_turn_id"),
                "plan_origin": "user_revision",
                "return_to_gate": None,
                "messages": _bounded_graph_messages([
                    *_messages_before_user_turn(
                        _resume_messages(state, response),
                        question=message,
                        turn_id=response.turn_id,
                    ),
                    _conversation_message(
                        role="user",
                        content=message,
                        message_id=response.message_id,
                        turn_id=response.turn_id,
                        episode_id=_cursor(state).episode_id,
                    ),
                ]),
            }
        else:
            approval.status = "rejected"
            approval.decision = "reject"
            update = {
                "phase": "stopped",
                "control": "stop",
                "return_to_gate": None,
                "stop_reason": "用户拒绝执行当前方案。",
            }
        update["approval_state"] = approval.model_dump(mode="json")
        update["events"] = _event(
            state,
            f"方案操作：{response.action} · {approval.plan_id or 'unknown'}",
            trace_category="plan",
            action=response.action,
        )
        return update

    def route_approval(state: ResearchLoopState) -> Literal["lock", "modify", "discuss", "stop", "reinterrupt"]:
        control = state.get("control", "reinterrupt")
        return control if control in {"lock", "modify", "discuss", "stop"} else "reinterrupt"  # type: ignore[return-value]

    def lock_plan(state: ResearchLoopState) -> dict[str, Any]:
        try:
            scope = _scope(state)
            if scope is not None:
                config = _config(state)
                if config is None or dynamic_agent is None:
                    raise ResearchPlanValidationError("动态研究范围缺少执行组件")
                if not _scope_approval_matches(state, scope):
                    raise ResearchPlanValidationError("当前研究范围没有匹配的用户审批")
                approval = ApprovalState.model_validate(state["approval_state"])
                envelope = scope_authorization_envelope(scope, approved_at=approval.approved_at or _now())
                validate_scope_authorization(scope, envelope)
                preflight = execution.compile_dynamic_call(
                    scope=scope,
                    study_config=config,
                    name="data_quality",
                    arguments={},
                    sequence=1,
                )
                preflight_payload = preflight.model_dump(mode="json")
                record = ToolCallRecord(call=preflight_payload)
                provider_id = "system-preflight-data-quality"
                group = ToolCallGroupRecord(
                    provider_call_id=provider_id,
                    requested_name="data_quality",
                    requested_version=FUNCTION_CATALOG["data_quality"].version,
                    child_call_ids=[preflight.call_id],
                    origin="system_preflight",
                )
                budget = _budget(state).model_copy(
                    update={
                        "max_model_rounds": scope.max_model_rounds,
                        "max_tool_calls": scope.max_tool_calls,
                        "max_function_attempts_per_call": scope.max_attempts_per_call,
                    }
                )
                initial_messages = dynamic_agent.initial_messages(
                    scope=scope,
                    quality_report=state.get("quality_report", {}),
                    data_profile=state.get("data_profile"),
                )
                return {
                    "phase": "executing_tools",
                    "control": "execute",
                    "authorization_envelope": envelope.model_dump(mode="json"),
                    "tool_queue": [preflight_payload],
                    "current_tool_batch": [preflight_payload],
                    "tool_cursor": 0,
                    "tool_records": {preflight.call_id: record.model_dump(mode="json")},
                    "tool_results": [],
                    "pending_tool_result": None,
                    "analysis_messages": [message.model_dump(mode="json") for message in initial_messages],
                    "provider_call_groups": {provider_id: group.model_dump(mode="json")},
                    "process_call_steps": {
                        preflight.call_id: (
                            f"{_cursor(state).episode_id or state['thread_id']}-system-preflight"
                        )
                    },
                    "call_evidence": [],
                    "model_round": 0,
                    "tool_calls_used": 0,
                    "budget": budget.model_dump(mode="json"),
                    "events": _event(
                        state,
                        f"锁定动态研究范围：{scope.scope_id}",
                        trace_category="plan",
                        calls=0,
                        functions=scope.authorized_functions,
                    ),
                }
            plan = _plan(state)
            if plan is None:
                raise ResearchPlanValidationError("无法锁定空方案")
            budget = _budget(state)
            if budget.evaluated_iterations >= budget.max_evaluated_iterations:
                packet = budget_feedback(
                    budget,
                    code="safety_iteration_limit",
                    message="自动研究触发安全熔断；议程尚未在安全范围内收敛，请用户处理当前限制。",
                )
                return {
                    "phase": "awaiting_user",
                    "control": "need_user",
                    "feedback_packets": _append_feedback(state, packet),
                }
            envelope = state.get("authorization_envelope")
            if state.get("plan_origin") == "automatic_evaluation":
                if envelope is None:
                    raise ResearchPlanValidationError("自动修订缺少原始用户审批边界")
            else:
                if not _approval_matches(state, plan):
                    raise ResearchPlanValidationError("当前方案没有与方案 ID 和指纹匹配的用户审批")
                approval = ApprovalState.model_validate(state["approval_state"])
                envelope = authorization_envelope(plan, approved_at=approval.approved_at or _now()).model_dump(
                    mode="json"
                )
            compiled_queue = execution.compile_tool_queue(plan)
            expected_calls = [
                (step.step_id, step.function, step.function_version)
                for step in plan.enabled_steps
            ]
            actual_calls = [
                (call.step_id, call.name, call.version)
                for call in compiled_queue
            ]
            if not compiled_queue:
                raise ResearchPlanValidationError("锁定方案没有生成任何函数调用")
            if actual_calls != expected_calls:
                raise ResearchPlanValidationError("函数队列与已启用计划步骤不完全一致")
            queue = [call.model_dump(mode="json") for call in compiled_queue]
            result_cache = dict(state.get("tool_result_cache", {}))
            records: dict[str, dict[str, Any]] = {}
            for call_payload in queue:
                call = ToolCall.model_validate(call_payload)
                record = ToolCallRecord(call=call_payload)
                cached_payload = result_cache.get(call.work_id)
                if cached_payload is not None:
                    rebound = result_store.get(state["thread_id"], cached_payload, call=call)
                    execution.validate_tool_result(plan=plan, call=call, result=rebound)
                    record.status = "reused"
                    record.result = result_store.bind(state["thread_id"], cached_payload, call)
                    record.finished_at = _now()
                records[call.call_id] = record.model_dump(mode="json")
        except Exception as exc:  # noqa: BLE001 - authorization/queue boundary
            packet = exception_feedback(exc, source="plan_validator", requires_user=True)
            return {
                "phase": "failed",
                "control": "failed",
                "feedback_packets": _append_feedback(state, packet),
                "stop_reason": packet.message,
                "events": _event(
                    state,
                    f"锁定方案失败：{_short(packet.message)}",
                    "failed",
                    trace_category="error",
                ),
            }
        return {
            "phase": "executing_tools",
            "control": "execute",
            "loop_cursor": _cursor(state).model_copy(
                update={"episode_status": "executing", "stage": "executing_functions"}
            ).model_dump(mode="json"),
            "authorization_envelope": envelope,
            "tool_queue": queue,
            "tool_cursor": 0,
            "tool_records": records,
            "process_call_steps": {
                ToolCall.model_validate(item).call_id: (
                    f"{_cursor(state).episode_id or state['thread_id']}-"
                    f"plan-{ToolCall.model_validate(item).step_id}"
                )
                for item in queue
            },
            "tool_results": [],
            "pending_tool_result": None,
            "events": _event(
                state,
                f"锁定函数队列：{plan.plan_id} · {len(queue)} 个调用",
                trace_category="plan",
                calls=len(queue),
                functions=[call["name"] for call in queue],
            ),
        }

    def route_lock(state: ResearchLoopState) -> Literal["execute", "need_user", "failed"]:
        control = state.get("control", "execute")
        return control if control in {"need_user", "failed"} else "execute"  # type: ignore[return-value]

    def _mark_tool_running(state: ResearchLoopState) -> dict[str, Any]:
        cursor = int(state.get("tool_cursor", 0))
        queue = state.get("tool_queue", [])
        if cursor >= len(queue):
            if _scope(state) is not None:
                return {
                    "control": "select",
                    "phase": "executing_tools",
                    "pending_tool_result": None,
                }
            return {
                "control": "evaluate",
                "phase": "evaluating",
                "pending_tool_result": None,
            }
        call = ToolCall.model_validate(queue[cursor])
        records = dict(state.get("tool_records", {}))
        record = ToolCallRecord.model_validate(records[call.call_id])
        if record.status == "pending" and record.result is None:
            cached_payload = state.get("tool_result_cache", {}).get(call.work_id)
            if cached_payload is not None:
                rebound = result_store.get(state["thread_id"], cached_payload, call=call)
                scope = _scope(state)
                if scope is not None:
                    execution.validate_scope_result(scope=scope, call=call, result=rebound)
                else:
                    plan = _plan(state)
                    if plan is None:
                        raise ResearchPlanValidationError("复用函数结果时缺少计划或研究范围")
                    execution.validate_tool_result(plan=plan, call=call, result=rebound)
                record.status = "reused"
                record.result = result_store.bind(state["thread_id"], cached_payload, call)
                record.finished_at = _now()
                records[call.call_id] = record.model_dump(mode="json")
        if record.status in {"completed", "reused"} and record.result is not None:
            result_reference = next(
                (
                    item
                    for item in state.get("tool_results", [])
                    if item.get("call", {}).get("call_id") == call.call_id
                ),
                None,
            )
            if result_reference is None:
                result_reference = state.get("tool_result_cache", {}).get(call.work_id)
                if result_reference is None:
                    raise ResearchPlanValidationError(f"复用函数 {call.name} 缺少缓存结果")
            rebound_reference = result_store.bind(state["thread_id"], result_reference, call)
            record.status = "reused"
            record.result = rebound_reference
            records[call.call_id] = record.model_dump(mode="json")
            results = list(state.get("tool_results", []))
            if not any(item["call"]["call_id"] == call.call_id for item in results):
                results.append(rebound_reference)
            process_history, started = _action_started_events(
                {**state, "tool_records": records}, call
            )
            completed = _process_event(
                state,
                "action_completed",
                step_id=started.step_id,
                round_number=started.round,
                source=started.source,
                sequence=started.sequence + 1,
                title=started.title,
                action_id=started.action_id,
                tool_name=call.name,
                result="复用已校验的相同数据、函数版本和参数结果",
            )
            _emit_process_event(completed)
            process_history = append_process_events(process_history, completed)
            return {
                "tool_records": records,
                "tool_results": results,
                "control": "advance",
                "pending_tool_result": None,
                "process_events": process_history,
                "events": _event(
                    state,
                    f"复用函数结果：{_function_label(call.name)}",
                    trace_category="tool",
                    call_id=call.call_id,
                    function=call.name,
                ),
            }
        if _scope(state) is not None and record.status == "cancelled":
            return {
                "tool_records": records,
                "control": "advance",
                "pending_tool_result": None,
            }
        budget = _budget(state)
        attempt_count = record.attempts
        if (
            record.status in {"running", "failed"}
            and attempt_count >= budget.max_function_attempts_per_call
        ):
            packet = budget_feedback(budget, code="tool_retry_budget", message=f"工具 {call.name} 重试耗尽。")
            if _scope(state) is not None and call.name != "data_quality":
                return _dynamic_tool_error_update(
                    state,
                    call=call,
                    packet=packet,
                    force_user=True,
                )
            return {
                "phase": "awaiting_user",
                "control": "need_user",
                "feedback_packets": _append_feedback(state, packet),
                "pending_tool_result": None,
                "events": _event(
                    state,
                    f"函数重试耗尽：{_function_label(call.name)}",
                    "failed",
                    trace_category="error",
                    call_id=call.call_id,
                ),
            }
        record.status = "running"
        record.attempts += 1
        record.started_at = _now()
        records[call.call_id] = record.model_dump(mode="json")
        process_history, _started = _action_started_events(
            {**state, "tool_records": records}, call
        )
        return {
            "tool_records": records,
            "budget": budget.model_dump(mode="json"),
            "control": "run",
            "process_events": process_history,
            "loop_cursor": _cursor(state).model_copy(
                update={
                    "episode_status": "executing",
                    "stage": f"function:{call.name}",
                    "call_attempt_number": attempt_count + 1,
                    "current_call_id": call.call_id,
                }
            ).model_dump(mode="json"),
            "events": _event(
                state,
                f"准备执行函数：{_function_label(call.name)}",
                "running",
                trace_category="tool",
                call_id=call.call_id,
                function=call.name,
                attempt=record.attempts,
                arguments=_compact_event_arguments(call.arguments),
                arguments_hash=canonical_hash(call.arguments),
            ),
        }

    def mark_tool_running(state: ResearchLoopState) -> dict[str, Any]:
        try:
            return _mark_tool_running(state)
        except Exception as exc:  # noqa: BLE001 - tool state boundary
            packet = exception_feedback(exc, source="tool_executor", requires_user=True).model_copy(
                update={"severity": "fatal"}
            )
            return {
                "phase": "failed",
                "control": "fail",
                "feedback_packets": _append_feedback(state, packet),
                "stop_reason": packet.message,
                "events": _event(
                    state,
                    f"准备函数调用失败：{_short(packet.message, 44)}",
                    "failed",
                    trace_category="error",
                ),
            }

    def route_mark_tool(state: ResearchLoopState) -> Literal["run", "advance", "evaluate", "select", "need_user", "fail"]:
        control = state.get("control", "run")
        return control if control in {"advance", "evaluate", "select", "need_user", "fail"} else "run"  # type: ignore[return-value]

    def _dynamic_tool_error_update(
        state: ResearchLoopState,
        *,
        call: ToolCall,
        packet: FeedbackPacket,
        force_user: bool = False,
    ) -> dict[str, Any]:
        """Close the failed provider call while preserving independent batch work."""

        records = dict(state.get("tool_records", {}))
        groups = {
            key: ToolCallGroupRecord.model_validate(value)
            for key, value in state.get("provider_call_groups", {}).items()
        }
        batch_ids = {
            ToolCall.model_validate(item).call_id for item in state.get("current_tool_batch", [])
        }
        messages = [ModelMessage.model_validate(item) for item in state.get("analysis_messages", [])]
        evidence = list(state.get("call_evidence", []))
        existing_evidence = {str(item.get("call_id")) for item in evidence}
        target_provider_ids = {
            provider_id
            for provider_id, group in groups.items()
            if call.call_id in group.child_call_ids
        }
        if len(target_provider_ids) != 1:
            raise ResearchPlanValidationError(
                f"本地调用 {call.call_id} 没有唯一的 provider call group"
            )

        provider_messages = 0
        new_evidence = 0
        for provider_id, group in groups.items():
            if group.origin == "system_preflight" or not set(group.child_call_ids).intersection(batch_ids):
                continue
            is_target_group = provider_id in target_provider_ids
            if not is_target_group and not force_user:
                # A repairable failure belongs only to its provider call. Other
                # independent calls in the assistant batch remain executable.
                continue
            child_errors = []
            child_results = []
            for child_id in group.child_call_ids:
                if child_id not in batch_ids:
                    continue
                child_record = ToolCallRecord.model_validate(records[child_id])
                child_call = ToolCall.model_validate(child_record.call)
                if child_record.status in {"completed", "reused"} and child_record.result is not None:
                    result = result_store.get(
                        state["thread_id"], child_record.result, call=child_call
                    )
                    child_results.append(
                        {
                            "call_id": child_id,
                            "function": child_call.name,
                            "status": child_record.status,
                            "result_key": result.output.result_key,
                            "value": _compact_tool_value(result.output.value),
                            "output_hash": result.output_hash,
                            "duplicate_notice": (
                                {
                                    "code": "duplicate_work_id",
                                    "message": "相同数据、函数版本和参数的结果已复用，未重复计算。",
                                    "work_id": child_call.work_id,
                                }
                                if child_record.status == "reused"
                                else None
                            ),
                        }
                    )
                    if child_id not in existing_evidence:
                        evidence.append(
                            CallEvidenceRecord(
                                sequence=int(child_call.step_id[1:]),
                                call_id=child_call.call_id,
                                provider_call_id=provider_id,
                                origin=group.origin,
                                function=child_call.name,
                                function_version=child_call.version,
                                arguments=child_call.arguments,
                                result=child_record.result,
                                data_fingerprint=result.data_fingerprint,
                                output_hash=result.output_hash,
                                status=child_record.status,
                            ).model_dump(mode="json")
                        )
                        existing_evidence.add(child_id)
                        if child_record.status == "completed":
                            new_evidence += 1
                    continue

                is_failed_child = child_id == call.call_id
                child_record.status = "failed" if is_failed_child else "cancelled"
                cancellation_packet = FeedbackPacket(
                    source="tool_executor",
                    code="cancelled_due_to_batch_abort",
                    severity="warning",
                    message="同批次因其他调用失败而取消。",
                    step_id=child_id,
                    retryable=False,
                    requires_user=force_user,
                )
                child_record.error = packet if is_failed_child else cancellation_packet
                child_record.finished_at = _now()
                records[child_id] = child_record.model_dump(mode="json")
                child_errors.append(
                    {
                        "call_id": child_id,
                        "function": child_call.name,
                        "status": child_record.status,
                        "code": packet.code if is_failed_child else "cancelled_due_to_batch_abort",
                        "error": (
                            packet.message
                            if is_failed_child
                            else cancellation_packet.message
                        ),
                    }
                )
            group.status = "failed" if is_target_group else "cancelled"
            groups[provider_id] = group
            messages.append(
                ModelMessage(
                    role="tool",
                    tool_call_id=provider_id,
                    content=json.dumps(
                        {
                            "requested": group.requested_name,
                            "requested_version": group.requested_version,
                            "status": group.status,
                            "retryable": packet.retryable,
                            "results": child_results,
                            "errors": child_errors,
                        },
                        ensure_ascii=False,
                    ),
                )
            )
            provider_messages += 1

        budget = _budget(state)
        if force_user:
            budget.no_progress_rounds = 0 if new_evidence else budget.no_progress_rounds + 1
        needs_user = force_user or provider_messages == 0
        serialized_groups = {
            key: value.model_dump(mode="json") for key, value in groups.items()
        }
        validate_analysis_message_protocol(
            messages,
            serialized_groups,
            allow_pending_current_batch=not needs_user,
        )
        return {
            "phase": "awaiting_user" if needs_user else "executing_tools",
            "control": "need_user" if needs_user else "advance",
            "tool_queue": [] if needs_user else list(state.get("tool_queue", [])),
            "current_tool_batch": [] if needs_user else list(state.get("current_tool_batch", [])),
            "tool_cursor": 0 if needs_user else int(state.get("tool_cursor", 0)),
            "tool_records": records,
            "provider_call_groups": serialized_groups,
            "analysis_messages": [message.model_dump(mode="json") for message in messages],
            "call_evidence": evidence,
            "pending_tool_result": None,
            "feedback_packets": _append_feedback(state, packet),
            "budget": budget.model_dump(mode="json"),
            "user_interrupt_kind": "plan_error" if needs_user else None,
            "stop_reason": packet.message if needs_user else None,
        }

    def execute_tool(state: ResearchLoopState) -> dict[str, Any]:
        call: ToolCall | None = None
        scope: EDAResearchScope | None = None
        try:
            cursor = int(state.get("tool_cursor", 0))
            call = ToolCall.model_validate(state["tool_queue"][cursor])
            plan = _plan(state)
            config = _config(state)
            scope = _scope(state)
            if config is None or (plan is None and scope is None):
                raise ResearchPlanValidationError("工具执行缺少计划或配置")
            if scope is not None:
                result = execution.execute_scope_call(
                    scope=scope,
                    study_config=config,
                    call=call,
                )
            else:
                result = execution.execute_call(
                    plan=plan,
                    study_config=config,
                    call=call,
                    validate_result=False,
                )
        except Exception as exc:  # noqa: BLE001 - tool boundary
            permission_or_data = isinstance(
                exc,
                (
                    ToolPermissionError,
                    ResearchDataError,
                    PlanCompatibilityError,
                    DataFingerprintMismatchError,
                    InsufficientDataError,
                ),
            )
            transient = isinstance(exc, (OSError, TimeoutError)) and not permission_or_data
            plan_error = isinstance(
                exc,
                (ToolExecutionError, ResearchPlanValidationError, ToolRegistryError, RepairablePlanError),
            ) and not permission_or_data
            packet = exception_feedback(
                exc,
                source="tool_executor",
                step_id=call.call_id if call is not None else None,
                retryable=transient or plan_error,
                requires_user=permission_or_data,
            )
            if scope is not None and call is not None and plan_error and call.name != "data_quality":
                update = _dynamic_tool_error_update(state, call=call, packet=packet)
                update["process_events"] = _action_finished_events(
                    state,
                    call,
                    result=_short(packet.message, 120),
                    failed=True,
                )
                return update
            outcome = (
                "retry"
                if transient
                else ("need_user" if permission_or_data else ("revise" if plan_error else "fail"))
            )
            if outcome == "fail":
                packet = packet.model_copy(
                    update={
                        "severity": "fatal",
                        "recommendation": "当前工具异常无法安全恢复，研究循环已停止。",
                    }
                )
            if (
                scope is not None
                and call is not None
                and call.name != "data_quality"
                and outcome in {"need_user", "fail"}
            ):
                update = _dynamic_tool_error_update(
                    state,
                    call=call,
                    packet=packet,
                    force_user=True,
                )
                update["process_events"] = _action_finished_events(
                    state,
                    call,
                    result=_short(packet.message, 120),
                    failed=True,
                    complete_step=True,
                )
                return update
            records = dict(state.get("tool_records", {}))
            if call is not None and call.call_id in records:
                record = ToolCallRecord.model_validate(records[call.call_id])
                record.status = "failed"
                record.error = packet
                record.finished_at = _now()
                records[call.call_id] = record.model_dump(mode="json")
            return {
                "phase": "failed" if outcome == "fail" else state.get("phase", "executing_tools"),
                "control": outcome,
                "tool_records": records,
                "feedback_packets": _append_feedback(state, packet),
                "pending_tool_result": None,
                "stop_reason": packet.message if outcome in {"need_user", "fail"} else None,
                "process_events": (
                    _action_finished_events(
                        state,
                        call,
                        result=_short(packet.message, 120),
                        failed=True,
                        complete_step=outcome in {"need_user", "fail"} or scope is None,
                    )
                    if call is not None
                    else list(state.get("process_events", []))
                ),
                "events": _event(
                    state,
                    f"函数执行失败：{_function_label(call.name)} · {_short(packet.message, 36)}",
                    "failed",
                    trace_category="error",
                    call_id=call.call_id if call is not None else None,
                    function=call.name if call is not None else None,
                    outcome=outcome,
                    error=packet.message,
                ),
            }
        return {
            "phase": "validating_result",
            "control": "validate",
            "pending_tool_result": result.model_dump(mode="json"),
            "events": _event(
                state,
                f"函数执行完成：{_function_label(call.name)}",
                trace_category="tool",
                call_id=call.call_id,
                duration_ms=result.duration_ms,
                function=call.name,
            ),
        }

    def route_tool_execution(
        state: ResearchLoopState,
    ) -> Literal["validate", "retry", "revise", "need_user", "stop", "advance", "select", "fail"]:
        return state.get("control", "stop")  # type: ignore[return-value]

    def _validate_tool_result(state: ResearchLoopState) -> dict[str, Any]:
        cursor = int(state.get("tool_cursor", 0))
        call = ToolCall.model_validate(state["tool_queue"][cursor])
        result = ToolResult.model_validate(state["pending_tool_result"])
        try:
            scope = _scope(state)
            if scope is not None:
                execution.validate_scope_result(scope=scope, call=call, result=result)
            else:
                plan = _plan(state)
                if plan is None:
                    raise ResearchPlanValidationError("结果校验缺少锁定计划")
                execution.validate_tool_result(plan=plan, call=call, result=result)
        except Exception as exc:  # noqa: BLE001 - result-validator boundary
            packet = exception_feedback(exc, source="tool_result_validator", step_id=call.call_id, retryable=True)
            if scope is not None and call.name != "data_quality":
                update = _dynamic_tool_error_update(state, call=call, packet=packet)
                update["process_events"] = _action_finished_events(
                    state,
                    call,
                    result=_short(packet.message, 120),
                    failed=True,
                    complete_step=False,
                )
                return update
            records = dict(state.get("tool_records", {}))
            record = ToolCallRecord.model_validate(records[call.call_id])
            record.status = "failed"
            record.error = packet
            record.finished_at = _now()
            records[call.call_id] = record.model_dump(mode="json")
            return {
                "phase": "planning",
                "control": "revise",
                "tool_records": records,
                "pending_tool_result": None,
                "process_events": _action_finished_events(
                    state,
                    call,
                    result=_short(packet.message, 120),
                    failed=True,
                    complete_step=scope is None,
                ),
                "feedback_packets": _append_feedback(state, packet),
                "events": _event(
                    state,
                    f"函数结果校验失败：{_function_label(call.name)} · {_short(packet.message, 34)}",
                    "failed",
                    trace_category="error",
                    call_id=call.call_id,
                    function=call.name,
                    error=packet.message,
                ),
            }
        records = dict(state.get("tool_records", {}))
        record = ToolCallRecord.model_validate(records[call.call_id])
        record.status = "completed"
        result_reference = result_store.put(state["thread_id"], result)
        record.result = result_reference
        record.finished_at = _now()
        records[call.call_id] = record.model_dump(mode="json")
        result_cache = dict(state.get("tool_result_cache", {}))
        result_cache.pop(call.work_id, None)
        result_cache[call.work_id] = result_reference
        while len(result_cache) > MAX_TOOL_RESULT_CACHE:
            result_cache.pop(next(iter(result_cache)))
        process_history = _action_finished_events(
            state,
            call,
            result=_process_result_summary(result),
            complete_step=scope is None,
            report_path=_process_report_path(result),
        )
        return {
            "phase": "executing_tools",
            "control": "advance",
            "tool_records": records,
            "tool_result_cache": result_cache,
            "tool_results": [*state.get("tool_results", []), result_reference],
            "pending_tool_result": None,
            "process_events": process_history,
            "events": _event(
                state,
                f"函数结果校验通过：{_function_label(call.name)}",
                trace_category="tool",
                call_id=call.call_id,
                function=call.name,
                output_hash=result.output_hash,
            ),
        }

    def validate_tool_result(state: ResearchLoopState) -> dict[str, Any]:
        try:
            return _validate_tool_result(state)
        except Exception as exc:  # noqa: BLE001 - result state boundary
            packet = exception_feedback(exc, source="tool_result_validator", requires_user=True).model_copy(
                update={"severity": "fatal"}
            )
            return {
                "phase": "failed",
                "control": "fail",
                "feedback_packets": _append_feedback(state, packet),
                "pending_tool_result": None,
                "stop_reason": packet.message,
                "events": _event(
                    state,
                    f"函数结果状态损坏：{_short(packet.message, 44)}",
                    "failed",
                    trace_category="error",
                ),
            }

    def advance_tool(state: ResearchLoopState) -> dict[str, Any]:
        return {
            "tool_cursor": int(state.get("tool_cursor", 0)) + 1,
            "pending_tool_result": None,
        }

    def _compact_tool_value(value: Any, *, depth: int = 0) -> Any:
        if depth >= 5:
            return "<truncated>"
        if isinstance(value, dict):
            return {
                str(key): _compact_tool_value(item, depth=depth + 1)
                for key, item in list(value.items())[:40]
            }
        if isinstance(value, list):
            compact = [_compact_tool_value(item, depth=depth + 1) for item in value[:20]]
            if len(value) > 20:
                compact.append({"omitted_items": len(value) - 20})
            return compact
        if isinstance(value, str) and len(value) > 1000:
            return f"{value[:997]}..."
        return value

    def complete_tool_batch(state: ResearchLoopState) -> dict[str, Any]:
        scope = _scope(state)
        if scope is None:
            return {"control": "evaluate", "phase": "evaluating"}
        batch = [ToolCall.model_validate(item) for item in state.get("current_tool_batch", [])]
        records = dict(state.get("tool_records", {}))
        groups = {
            key: ToolCallGroupRecord.model_validate(value)
            for key, value in state.get("provider_call_groups", {}).items()
        }
        messages = [ModelMessage.model_validate(item) for item in state.get("analysis_messages", [])]
        evidence = list(state.get("call_evidence", []))
        existing_call_ids = {str(item.get("call_id")) for item in evidence}
        batch_ids = {call.call_id for call in batch}
        new_evidence = 0

        for provider_id, group in groups.items():
            if (
                not set(group.child_call_ids).intersection(batch_ids)
                or group.status in {"completed", "failed", "cancelled"}
            ):
                continue
            child_payloads = []
            for child_id in group.child_call_ids:
                call = next(item for item in batch if item.call_id == child_id)
                record = ToolCallRecord.model_validate(records[child_id])
                if record.result is None:
                    raise ResearchPlanValidationError(f"动态调用 {child_id} 缺少结果引用")
                result = result_store.get(state["thread_id"], record.result, call=call)
                child_payloads.append(
                    {
                        "function": call.name,
                        "arguments": call.arguments,
                        "status": "completed" if record.status == "completed" else "reused",
                        "result_key": result.output.result_key,
                        "value": _compact_tool_value(result.output.value),
                        "output_hash": result.output_hash,
                        "duplicate_notice": (
                            {
                                "code": "duplicate_work_id",
                                "message": "相同数据、函数版本和参数的结果已复用，未重复计算。",
                                "work_id": call.work_id,
                            }
                            if record.status == "reused"
                            else None
                        ),
                    }
                )
                if child_id not in existing_call_ids:
                    evidence.append(
                        CallEvidenceRecord(
                            sequence=int(call.step_id[1:]),
                            call_id=call.call_id,
                            provider_call_id=provider_id,
                            origin=group.origin,
                            function=call.name,
                            function_version=call.version,
                            arguments=call.arguments,
                            result=record.result,
                            data_fingerprint=result.data_fingerprint,
                            output_hash=result.output_hash,
                            status=record.status,
                        ).model_dump(mode="json")
                    )
                    existing_call_ids.add(child_id)
                    if record.status == "completed":
                        new_evidence += 1
            group.status = "completed"
            groups[provider_id] = group
            if group.origin != "system_preflight":
                messages.append(
                    ModelMessage(
                        role="tool",
                        tool_call_id=provider_id,
                        content=json.dumps(
                            {
                                "requested": group.requested_name,
                                "requested_version": group.requested_version,
                                "results": child_payloads,
                            },
                            ensure_ascii=False,
                            allow_nan=False,
                        ),
                    )
                )

        budget = _budget(state)
        validate_analysis_message_protocol(
            messages,
            {key: value.model_dump(mode="json") for key, value in groups.items()},
            allow_pending_current_batch=False,
        )
        if any(group.origin != "system_preflight" and set(group.child_call_ids).intersection(batch_ids) for group in groups.values()):
            budget.no_progress_rounds = 0 if new_evidence else budget.no_progress_rounds + 1
        process_history = list(state.get("process_events", []))
        call_steps = state.get("process_call_steps", {})
        batch_step_ids = list(
            dict.fromkeys(call_steps.get(call.call_id) for call in batch if call_steps.get(call.call_id))
        )
        for step_id in batch_step_ids:
            if any(
                item.get("step_id") == step_id and item.get("event_type") == "step_completed"
                for item in process_history
            ):
                continue
            related = [call for call in batch if call_steps.get(call.call_id) == step_id]
            sample = related[0]
            _resolved_step, round_number, source, _basis = _call_process_context(state, sample)
            completed = _process_event(
                state,
                "step_completed",
                step_id=step_id,
                round_number=round_number,
                source=source,
                title="步骤完成",
                result=f"{len(related)} 个行动已完成",
                sequence=(
                    max(
                        (int(item.get("sequence", 0)) for item in process_history),
                        default=0,
                    )
                    + 1
                ),
            )
            _emit_process_event(completed)
            process_history = append_process_events(process_history, completed)
        if budget.no_progress_rounds >= 2:
            packet = budget_feedback(budget, code="no_new_evidence", message="连续两轮没有新增分析证据。")
            return {
                "phase": "awaiting_user",
                "control": "need_user",
                "feedback_packets": _append_feedback(state, packet),
                "user_interrupt_kind": "result_limitations",
                "tool_queue": [],
                "current_tool_batch": [],
                "tool_cursor": 0,
                "analysis_messages": [message.model_dump(mode="json") for message in messages],
                "provider_call_groups": {key: value.model_dump(mode="json") for key, value in groups.items()},
                "call_evidence": evidence,
                "budget": budget.model_dump(mode="json"),
                "process_events": process_history,
            }
        return {
            "phase": "executing_tools",
            "control": "select",
            "tool_queue": [],
            "current_tool_batch": [],
            "tool_cursor": 0,
            "analysis_messages": [message.model_dump(mode="json") for message in messages],
            "provider_call_groups": {key: value.model_dump(mode="json") for key, value in groups.items()},
            "call_evidence": evidence,
            "budget": budget.model_dump(mode="json"),
            "process_events": process_history,
        }

    def _selector_protocol_retry(
        state: ResearchLoopState,
        *,
        scope: EDAResearchScope,
        budget: LoopBudget,
        messages: list[ModelMessage],
        packet: FeedbackPacket,
    ) -> dict[str, Any]:
        """Reject an invalid provider turn without persisting its assistant calls."""

        budget.no_progress_rounds += 1
        messages.append(
            ModelMessage(
                role="user",
                content=json.dumps(
                    {
                        "internal_protocol_feedback": {
                            "code": packet.code,
                            "message": packet.message,
                            "instruction": (
                                "重新生成整个工具回合；每个 provider call_id 必须非空、"
                                "在当前及历史回合中唯一。"
                            ),
                        }
                    },
                    ensure_ascii=False,
                ),
            )
        )
        needs_user = (
            budget.no_progress_rounds >= 2
            or budget.model_rounds_used >= scope.max_model_rounds
        )
        process_history = list(state.get("process_events", []))
        current = next(
            (
                ProcessEvent.model_validate(item)
                for item in reversed(process_history)
                if item.get("event_type") == "thinking_started"
            ),
            None,
        )
        if current is not None:
            completed = _process_event(
                state,
                "step_completed",
                step_id=current.step_id,
                round_number=current.round,
                source=current.source,
                title="模型响应无效",
                result=_short(packet.message, 120),
            )
            _emit_process_event(completed)
            process_history = append_process_events(process_history, completed)
        return {
            "phase": "awaiting_user" if needs_user else "executing_tools",
            "control": "need_user" if needs_user else "select",
            "analysis_messages": [message.model_dump(mode="json") for message in messages],
            "feedback_packets": _append_feedback(state, packet),
            "budget": budget.model_dump(mode="json"),
            "model_round": budget.model_rounds_used,
            "user_interrupt_kind": "response_error" if needs_user else None,
            "stop_reason": packet.message if needs_user else None,
            "process_events": process_history,
            "events": _event(
                state,
                f"拒绝非法模型工具回合：{packet.code}",
                "warning",
                trace_category="error",
            ),
        }

    def analysis_selector(state: ResearchLoopState) -> dict[str, Any]:
        scope = _scope(state)
        config = _config(state)
        if scope is None or config is None or dynamic_agent is None:
            raise ResearchPlanValidationError("动态分析选择器缺少研究范围或模型组件")
        envelope = AuthorizationEnvelope.model_validate(state.get("authorization_envelope", {}))
        validate_scope_authorization(scope, envelope)
        budget = _budget(state)
        if not scope.authorized_functions:
            completed_calls = [
                ToolCall.model_validate(record["call"])
                for record in state.get("tool_records", {}).values()
                if record.get("status") in {"completed", "reused"}
            ]
            completed_calls.sort(key=lambda call: int(call.step_id[1:]))
            plan = execution.build_dynamic_plan(scope=scope, calls=completed_calls)
            messages = [
                ModelMessage.model_validate(item)
                for item in state.get("analysis_messages", [])
            ]
            messages.append(
                ModelMessage(
                    role="assistant",
                    content="本次批准范围仅包含系统数据质量核验，无需选择其他分析函数。",
                )
            )
            return {
                "phase": "evaluating",
                "control": "evaluate",
                "current_plan": plan.model_dump(mode="json"),
                "analysis_messages": [message.model_dump(mode="json") for message in messages],
                "pending_analysis_text": messages[-1].content,
                "budget": budget.model_dump(mode="json"),
            }
        if budget.model_rounds_used >= scope.max_model_rounds:
            packet = budget_feedback(budget, code="model_round_budget", message="动态分析已达到 8 轮模型决策上限。")
            return {
                "phase": "awaiting_user",
                "control": "need_user",
                "feedback_packets": _append_feedback(state, packet),
                "user_interrupt_kind": "result_limitations",
            }
        messages = [ModelMessage.model_validate(item) for item in state.get("analysis_messages", [])]
        try:
            validate_analysis_message_protocol(
                messages,
                state.get("provider_call_groups", {}),
                allow_pending_current_batch=False,
            )
        except Exception as exc:  # noqa: BLE001 - corrupted/restored protocol state
            packet = exception_feedback(exc, source="plan_validator", requires_user=True)
            return {
                "phase": "awaiting_user",
                "control": "need_user",
                "feedback_packets": _append_feedback(state, packet),
                "stop_reason": packet.message,
                "user_interrupt_kind": "response_error",
            }
        budget.model_rounds_used += 1
        round_number = budget.model_rounds_used
        flow_token = _cursor(state).episode_id or str(
            state.get("revision_cycle_id") or state["thread_id"]
        )
        process_step_id = f"{flow_token}-model-round-{round_number}"
        thinking_started = _process_event(
            state,
            "thinking_started",
            step_id=process_step_id,
            round_number=round_number,
            source="model",
            title=f"第 {round_number} 步",
            content="正在根据已有证据选择下一步行动",
        )
        _emit_process_event(thinking_started)
        process_history = append_process_events(
            list(state.get("process_events", [])),
            thinking_started,
        )
        state = {**state, "process_events": process_history}
        try:
            skill = skills.get(scope.skill_name)
            turn = dynamic_agent.select(scope=scope, skill=skill, messages=messages)
        except (ModelResponseError, ValidationError) as exc:
            packet = FeedbackPacket(
                source="plan_validator",
                code="invalid_provider_tool_turn",
                severity="error",
                message=f"模型返回的工具回合不符合协议：{exc}",
                retryable=True,
            )
            return _selector_protocol_retry(
                state,
                scope=scope,
                budget=budget,
                messages=messages,
                packet=packet,
            )
        except Exception as exc:  # noqa: BLE001 - model tool-selection boundary
            packet = exception_feedback(exc, source="plan_validator", retryable=True, requires_user=True)
            completed = _process_event(
                state,
                "step_completed",
                step_id=process_step_id,
                round_number=round_number,
                source="model",
                title="模型调用失败",
                result=_short(packet.message, 120),
            )
            _emit_process_event(completed)
            process_history = append_process_events(process_history, completed)
            return {
                "phase": "awaiting_user",
                "control": "need_user",
                "feedback_packets": _append_feedback(state, packet),
                "stop_reason": packet.message,
                "user_interrupt_kind": "response_error",
                "process_events": process_history,
            }
        thinking_ready = _process_event(
            state,
            "thinking_ready",
            step_id=process_step_id,
            round_number=round_number,
            source="model",
            title="思考过程",
            content=turn.content.strip() or "本轮未提供思考过程说明",
        )
        _emit_process_event(thinking_ready)
        process_history = append_process_events(process_history, thinking_ready)
        state = {**state, "process_events": process_history}
        if not turn.tool_calls:
            messages.append(ModelMessage(role="assistant", content=turn.content))
            completed_calls = [
                ToolCall.model_validate(record["call"])
                for record in state.get("tool_records", {}).values()
                if record.get("status") in {"completed", "reused"}
            ]
            completed_calls.sort(key=lambda call: int(call.step_id[1:]))
            plan = execution.build_dynamic_plan(scope=scope, calls=completed_calls)
            completed = _process_event(
                state,
                "step_completed",
                step_id=process_step_id,
                round_number=round_number,
                source="model",
                title="分析决策完成",
                result=_short(turn.content, 120),
            )
            _emit_process_event(completed)
            process_history = append_process_events(process_history, completed)
            return {
                "phase": "evaluating",
                "control": "evaluate",
                "current_plan": plan.model_dump(mode="json"),
                "analysis_messages": [message.model_dump(mode="json") for message in messages],
                "pending_analysis_text": turn.content,
                "model_round": budget.model_rounds_used,
                "budget": budget.model_dump(mode="json"),
                "process_events": process_history,
                "events": _event(
                    state,
                    f"动态分析完成选择：{budget.model_rounds_used} 轮",
                    trace_category="plan",
                ),
            }

        provider_ids = [str(call.call_id or "") for call in turn.tool_calls]
        if any(not provider_id.strip() for provider_id in provider_ids):
            packet = FeedbackPacket(
                source="plan_validator",
                code="missing_provider_call_id",
                severity="error",
                message="模型返回了空 provider call_id。",
                retryable=True,
            )
            return _selector_protocol_retry(
                state,
                scope=scope,
                budget=budget,
                messages=messages,
                packet=packet,
            )
        if len(provider_ids) != len(set(provider_ids)):
            packet = FeedbackPacket(
                source="plan_validator",
                code="duplicate_provider_call_id",
                severity="error",
                message="模型在同一回合返回了重复 provider call_id。",
                retryable=True,
            )
            return _selector_protocol_retry(
                state,
                scope=scope,
                budget=budget,
                messages=messages,
                packet=packet,
            )
        reused_provider_ids = sorted(
            set(provider_ids).intersection(state.get("provider_call_groups", {}))
        )
        if reused_provider_ids:
            packet = FeedbackPacket(
                source="plan_validator",
                code="reused_provider_call_id",
                severity="error",
                message="模型重复使用了此前回合的 provider call_id。",
                observed=reused_provider_ids,
                retryable=True,
            )
            return _selector_protocol_retry(
                state,
                scope=scope,
                budget=budget,
                messages=messages,
                packet=packet,
            )

        assistant = ModelMessage(role="assistant", content=turn.content, tool_calls=turn.tool_calls)
        messages.append(assistant)

        calls: list[ToolCall] = []
        groups = dict(state.get("provider_call_groups", {}))
        records = dict(state.get("tool_records", {}))
        process_call_steps = dict(state.get("process_call_steps", {}))
        compile_packets: list[FeedbackPacket] = []
        next_sequence = max(
            (int(ToolCall.model_validate(record["call"]).step_id[1:]) for record in records.values()),
            default=0,
        ) + 1
        for proposal in turn.tool_calls:
            provider_id = str(proposal.call_id)
            origin: Literal["direct", "recipe"] = "direct"
            requested_version: str | None = None
            try:
                if proposal.name in dynamic_agent.recipes.names:
                    if proposal.arguments:
                        raise ResearchPlanValidationError(f"分析配方 {proposal.name} 不接受参数")
                    recipe = dynamic_agent.recipes.get(proposal.name)
                    names_and_arguments = [(name, {}) for name in recipe.functions]
                    origin = "recipe"
                    requested_version = recipe.version
                else:
                    names_and_arguments = [(proposal.name, proposal.arguments)]
                    origin = "direct"
                    requested_version = tools.get(proposal.name).version
                proposal_calls: list[ToolCall] = []
                proposal_records: dict[str, dict[str, Any]] = {}
                proposal_sequence = next_sequence
                for name, arguments in names_and_arguments:
                    call = execution.compile_dynamic_call(
                        scope=scope,
                        study_config=config,
                        name=name,
                        arguments=arguments,
                        sequence=proposal_sequence,
                    )
                    proposal_sequence += 1
                    proposal_calls.append(call)
                    proposal_records[call.call_id] = ToolCallRecord(
                        call=call.model_dump(mode="json")
                    ).model_dump(mode="json")
                calls.extend(proposal_calls)
                process_call_steps.update(
                    {call.call_id: process_step_id for call in proposal_calls}
                )
                records.update(proposal_records)
                next_sequence = proposal_sequence
                groups[provider_id] = ToolCallGroupRecord(
                    provider_call_id=provider_id,
                    requested_name=proposal.name,
                    requested_version=requested_version,
                    child_call_ids=[call.call_id for call in proposal_calls],
                    origin=origin,
                ).model_dump(mode="json")
            except Exception as exc:  # noqa: BLE001 - reject only this independent proposal
                packet = exception_feedback(exc, source="tool_executor", retryable=True)
                compile_packets.append(packet)
                groups[provider_id] = ToolCallGroupRecord(
                    provider_call_id=provider_id,
                    requested_name=proposal.name,
                    requested_version=requested_version,
                    child_call_ids=[],
                    origin=origin,
                    status="failed",
                ).model_dump(mode="json")
                messages.append(
                    ModelMessage(
                    role="tool",
                    tool_call_id=provider_id,
                    content=json.dumps(
                        {
                            "status": "rejected",
                            "error": packet.message,
                            "retryable": True,
                        },
                        ensure_ascii=False,
                    ),
                )
                )

        if not calls:
            budget.no_progress_rounds += 1
            needs_user = budget.no_progress_rounds >= 2
            packet = compile_packets[-1]
            completed = _process_event(
                state,
                "step_completed",
                step_id=process_step_id,
                round_number=round_number,
                source="model",
                title="行动无法执行",
                result=_short(packet.message, 120),
            )
            _emit_process_event(completed)
            process_history = append_process_events(process_history, completed)
            validate_analysis_message_protocol(
                messages,
                groups,
                allow_pending_current_batch=False,
            )
            return {
                "phase": "awaiting_user" if needs_user else "executing_tools",
                "control": "need_user" if needs_user else "select",
                "analysis_messages": [message.model_dump(mode="json") for message in messages],
                "provider_call_groups": groups,
                "feedback_packets": _append_feedback(state, packet),
                "budget": budget.model_dump(mode="json"),
                "user_interrupt_kind": "plan_error" if needs_user else None,
                "process_events": process_history,
            }

        if budget.tool_calls_used + len(calls) > scope.max_tool_calls:
            packet = budget_feedback(budget, code="tool_call_budget", message="动态分析将超过 16 次工具调用上限。")
            for provider_id in provider_ids:
                group = ToolCallGroupRecord.model_validate(groups[provider_id])
                if group.status != "pending":
                    continue
                group.status = "failed"
                groups[provider_id] = group.model_dump(mode="json")
                for child_id in group.child_call_ids:
                    record = ToolCallRecord.model_validate(records[child_id])
                    record.status = "failed"
                    record.error = packet
                    record.finished_at = _now()
                    records[child_id] = record.model_dump(mode="json")
                messages.append(
                    ModelMessage(
                        role="tool",
                        tool_call_id=provider_id,
                        content=json.dumps(
                            {"status": "rejected", "error": packet.message, "retryable": False},
                            ensure_ascii=False,
                        ),
                    )
                )
            validate_analysis_message_protocol(
                messages,
                groups,
                allow_pending_current_batch=False,
            )
            completed = _process_event(
                state,
                "step_completed",
                step_id=process_step_id,
                round_number=round_number,
                source="model",
                title="行动超过预算",
                result=_short(packet.message, 120),
            )
            _emit_process_event(completed)
            process_history = append_process_events(process_history, completed)
            return {
                "phase": "awaiting_user",
                "control": "need_user",
                "feedback_packets": _append_feedback(state, packet),
                "user_interrupt_kind": "result_limitations",
                "analysis_messages": [message.model_dump(mode="json") for message in messages],
                "provider_call_groups": groups,
                "tool_records": records,
                "budget": budget.model_dump(mode="json"),
                "process_events": process_history,
            }
        budget.tool_calls_used += len(calls)
        payloads = [call.model_dump(mode="json") for call in calls]
        validate_analysis_message_protocol(
            messages,
            groups,
            allow_pending_current_batch=True,
        )
        return {
            "phase": "executing_tools",
            "control": "execute",
            "tool_queue": payloads,
            "current_tool_batch": payloads,
            "tool_cursor": 0,
            "tool_records": records,
            "provider_call_groups": groups,
            "process_call_steps": process_call_steps,
            "process_events": process_history,
            "analysis_messages": [message.model_dump(mode="json") for message in messages],
            "model_round": budget.model_rounds_used,
            "tool_calls_used": budget.tool_calls_used,
            "budget": budget.model_dump(mode="json"),
            "events": _event(
                state,
                f"动态选择函数：{len(calls)} 个调用",
                trace_category="tool",
                functions=[call.name for call in calls],
            ),
        }

    def route_analysis_selector(state: ResearchLoopState) -> Literal["execute", "evaluate", "select", "need_user"]:
        control = state.get("control", "need_user")
        return control if control in {"execute", "evaluate", "select"} else "need_user"  # type: ignore[return-value]

    def prepare_plan_repair(state: ResearchLoopState) -> dict[str, Any]:
        budget = _budget(state)
        if budget.plan_attempts_in_iteration >= budget.max_plan_attempts_per_iteration:
            packet = budget_feedback(budget, code="plan_repair_budget", message="计划修订次数已耗尽。")
            return {
                "phase": "awaiting_user",
                "control": "need_user",
                "feedback_packets": _append_feedback(state, packet),
            }
        return {
            "phase": "planning",
            "control": "repair",
            "plan_origin": "automatic_evaluation",
            "budget": budget.model_dump(mode="json"),
            "loop_cursor": _cursor(state).model_copy(
                update={"episode_status": "planning", "stage": "planning"}
            ).model_dump(mode="json"),
            "tool_queue": [],
            "tool_cursor": 0,
            "tool_results": [],
            "pending_tool_result": None,
        }

    def _finalize_iteration(state: ResearchLoopState) -> dict[str, Any]:
        validate_analysis_message_protocol(
            state.get("analysis_messages", []),
            state.get("provider_call_groups", {}),
            allow_pending_current_batch=False,
        )
        plan = _plan(state)
        config = _config(state)
        if plan is None or config is None:
            raise ResearchPlanValidationError("评估前缺少计划或配置")
        evaluating_cursor = _cursor(state).model_copy(
            update={"episode_status": "evaluating", "stage": "evaluation", "terminal_reason": None}
        )
        run_identity = canonical_hash(
            {
                "thread_id": state["thread_id"],
                "episode_id": evaluating_cursor.episode_id,
                "iteration_id": evaluating_cursor.iteration_id,
                "plan": plan_fingerprint(plan),
            }
        )
        resolved_tool_results = [
            result_store.get(state["thread_id"], item)
            for item in state.get("tool_results", [])
        ]
        run = execution.finalize(
            plan=plan,
            study_config=config,
            tool_results=resolved_tool_results,
            conversation=_messages(state),
            run_id=f"agent-loop-{run_identity[:32]}",
            progress=noop_progress,
            loop_context={
                "thread_id": state["thread_id"],
                "episode": evaluating_cursor.model_dump(mode="json"),
                "research_iteration": _budget(state).evaluated_iterations + 1,
                "plan_history": [item.get("plan_id") for item in state.get("plan_history", [])],
                "feedback_packets": state.get("feedback_packets", []),
                "feedback_history": state.get("feedback_history", []),
                "tool_records": state.get("tool_records", {}),
                "call_evidence": state.get("call_evidence", []),
                "authorization_envelope": state.get("authorization_envelope"),
                "budget_before_evaluation": state.get("budget", {}),
            },
        )
        budget = _budget(state)
        budget.evaluated_iterations += 1
        evaluation_payload = run.evaluation.model_dump(mode="json")
        evidence_hash = evidence_fingerprint(state.get("tool_results", []))
        agenda_fingerprint = str(evaluation_payload.get("agenda_fingerprint") or "")
        latest = {
            "run_id": run.run_id,
            "plan_id": run.plan.plan_id,
            "artifact_directory": str(run.artifact_directory),
            "report_path": str(run.report_path),
            "figure_paths": {key: str(value) for key, value in run.figure_paths.items()},
            "evaluation": evaluation_payload,
        }
        history_entry = {
            "run_id": run.run_id,
            "plan_id": run.plan.plan_id,
            "artifact_directory": str(run.artifact_directory),
            "report_path": str(run.report_path),
            "evaluation_decision": run.evaluation.decision,
        }
        episode_summaries = _upsert_episode_summary(
            list(state.get("episode_summaries", [])),
            _episode_summary(
                state,
                cursor=evaluating_cursor,
                latest_run=latest,
                evaluation=evaluation_payload,
            ),
        )
        progress_record = {
            "episode_id": evaluating_cursor.episode_id,
            "revision_cycle_id": state.get("revision_cycle_id") or evaluating_cursor.episode_id,
            "iteration_id": evaluating_cursor.iteration_id,
            "plan_id": plan.plan_id,
            "plan_revision": plan.revision,
            "plan_fingerprint": plan_fingerprint(plan),
            "agenda_fingerprint": agenda_fingerprint,
            "evidence_fingerprint": evidence_hash,
            "evaluation_decision": run.evaluation.decision,
            "evaluation_completed": True,
        }
        return {
            "phase": "evaluating",
            "control": "evaluation",
            "loop_cursor": _cursor(state).model_copy(
                update={"episode_status": "evaluating", "stage": "evaluation"}
            ).model_dump(mode="json"),
            "evaluation": evaluation_payload,
            "eda_summary": run.eda_summary,
            "latest_run": latest,
            "run_history": [*state.get("run_history", []), history_entry][-MAX_RUN_HISTORY:],
            "episode_summaries": episode_summaries,
            "evidence_fingerprints": [*state.get("evidence_fingerprints", []), evidence_hash],
            "agenda_fingerprints": (
                [*state.get("agenda_fingerprints", []), agenda_fingerprint]
                if agenda_fingerprint
                else list(state.get("agenda_fingerprints", []))
            ),
            "progress_records": [*state.get("progress_records", []), progress_record][-64:],
            "budget": budget.model_dump(mode="json"),
            "events": _event(
                state,
                f"评估运行：{run.run_id} → {run.evaluation.decision}",
                trace_category="evaluation",
                decision=run.evaluation.decision,
                run_id=run.run_id,
            ),
        }

    def finalize_iteration(state: ResearchLoopState) -> dict[str, Any]:
        """Convert final merge, evaluation, and artifact failures into durable state."""

        try:
            return _finalize_iteration(state)
        except Exception as exc:  # noqa: BLE001 - finalization boundary
            packet = exception_feedback(exc, source="evaluator", requires_user=True).model_copy(
                update={
                    "severity": "fatal",
                    "recommendation": "工具结果已保留；请重试最终合并与报告生成，或停止并检查错误。",
                }
            )
            return {
                "phase": "awaiting_user",
                "control": "finalization_error",
                "feedback_packets": _append_feedback(state, packet),
                "stop_reason": packet.message,
                "user_interrupt_kind": "finalization_error",
                "events": _event(
                    state,
                    f"最终合并失败：{_short(packet.message, 44)}",
                    "failed",
                    trace_category="error",
                    preserved_tool_results=len(state.get("tool_results", [])),
                ),
            }

    def route_evaluation(
        state: ResearchLoopState,
    ) -> Literal[
        "accept",
        "revise",
        "dynamic_revise",
        "need_user",
        "reject",
        "failed",
        "finalization_error",
        "invalid",
    ]:
        if state.get("control") == "finalization_error":
            return "finalization_error"
        if state.get("control") == "failed":
            return "failed"
        decision = (state.get("evaluation") or {}).get("decision")
        if decision in {"accept", "reject", "need_user"}:
            return decision  # type: ignore[return-value]
        if decision != "revise":
            return "invalid"
        if progress_stall_code(state) is not None:
            return "need_user"
        budget = _budget(state)
        if budget.evaluated_iterations >= budget.max_evaluated_iterations:
            return "need_user"
        scope = _scope(state)
        if scope is not None:
            if (
                budget.model_rounds_used >= scope.max_model_rounds
                or budget.tool_calls_used >= scope.max_tool_calls
            ):
                return "need_user"
            return "dynamic_revise"
        return "revise"

    def prepare_invalid_evaluation(state: ResearchLoopState) -> dict[str, Any]:
        decision = (state.get("evaluation") or {}).get("decision")
        packet = FeedbackPacket(
            source="evaluator",
            code="invalid_evaluation_decision",
            severity="error",
            message=f"评估器返回了未知决策：{decision!r}",
            observed=decision,
            expected=["accept", "revise", "need_user", "reject"],
            recommendation="请检查评估契约或恢复未损坏的 checkpoint。",
            requires_user=True,
        )
        return {
            "phase": "awaiting_user",
            "control": "need_user",
            "feedback_packets": _append_feedback(state, packet),
            "stop_reason": packet.message,
            "user_interrupt_kind": "result_limitations" if state.get("latest_run") else "plan_error",
            "events": _event(
                state,
                f"评估决策无效：{decision!r}",
                "failed",
                trace_category="error",
            ),
        }

    def prepare_rejected_result(state: ResearchLoopState) -> dict[str, Any]:
        evaluation = state.get("evaluation") or {}
        reason = f"评估器拒绝当前结果：{evaluation.get('summary') or '结果未通过验收。'}"
        return {
            "phase": "awaiting_user",
            "control": "need_user",
            "stop_reason": reason,
            "user_interrupt_kind": "result_rejected",
            "loop_cursor": _cursor(state).model_copy(
                update={"episode_status": "limited", "stage": "human_gate", "terminal_reason": reason}
            ).model_dump(mode="json"),
            "events": _event(
                state,
                f"评估拒绝等待用户决策：{_short(reason, 42)}",
                "warning",
                trace_category="evaluation",
            ),
        }

    def prepare_evaluation_revision(state: ResearchLoopState) -> dict[str, Any]:
        evaluation = state.get("evaluation") or {}
        cursor = _cursor(state).next_iteration()
        budget = _budget(state).reset_for_iteration()
        packets = [FeedbackPacket.model_validate(item) for item in evaluation.get("feedback_packets", [])]
        if not packets:
            packets = [
                FeedbackPacket(
                    source="evaluator",
                    code="evaluation_revision",
                    severity="warning",
                    message=evaluation.get("summary", "评估要求修订。"),
                    recommendation="在原审批权限范围内修订方案。",
                    retryable=True,
                )
            ]
        feedback_history = _archive_feedback(state)
        feedback_packets: list[dict[str, Any]] = []
        for packet in packets:
            feedback_packets = _append_feedback({**state, "feedback_packets": feedback_packets}, packet)
        return {
            "phase": "planning",
            "control": "plan",
            "plan_origin": "automatic_evaluation",
            "loop_cursor": cursor.model_dump(mode="json"),
            "budget": budget.model_dump(mode="json"),
            "feedback_packets": feedback_packets,
            "feedback_history": feedback_history,
            "tool_queue": [],
            "tool_cursor": 0,
            "tool_results": [],
            "pending_tool_result": None,
            "events": _event(
                state,
                f"评估触发自动修订：{_plan(state).plan_id if _plan(state) else 'unknown'}",
                trace_category="evaluation",
                feedback_count=len(packets),
            ),
        }

    def prepare_dynamic_evaluation_revision(state: ResearchLoopState) -> dict[str, Any]:
        """Feed deterministic evaluation gaps back into the same approved tool loop."""

        scope = _scope(state)
        if scope is None:
            raise ResearchPlanValidationError("动态评估修订缺少已批准研究范围")
        evaluation = state.get("evaluation") or {}
        cursor = _cursor(state).next_iteration()
        budget = _budget(state).reset_for_iteration()
        messages = [ModelMessage.model_validate(item) for item in state.get("analysis_messages", [])]
        messages.append(
            ModelMessage(
                role="user",
                content=json.dumps(
                    {
                        "type": "deterministic_evaluation_feedback",
                        "decision": evaluation.get("decision"),
                        "summary": evaluation.get("summary"),
                        "checks": evaluation.get("checks", []),
                        "feedback_packets": evaluation.get("feedback_packets", []),
                        "suggested_followups": evaluation.get("suggested_followups", []),
                        "instruction": "仅在已批准研究范围和剩余预算内补充必要证据；否则停止调用并说明限制。",
                    },
                    ensure_ascii=False,
                ),
            )
        )
        return {
            "phase": "executing_tools",
            "control": "select",
            "loop_cursor": cursor.model_dump(mode="json"),
            "budget": budget.model_dump(mode="json"),
            "analysis_messages": [message.model_dump(mode="json") for message in messages],
            "feedback_history": _archive_feedback(state),
            "feedback_packets": [],
            "tool_queue": [],
            "current_tool_batch": [],
            "tool_cursor": 0,
            "pending_tool_result": None,
            "events": _event(
                state,
                "确定性评估要求动态补充证据",
                trace_category="evaluation",
                model_rounds_remaining=max(0, scope.max_model_rounds - budget.model_rounds_used),
                tool_calls_remaining=max(0, scope.max_tool_calls - budget.tool_calls_used),
            ),
        }

    def prepare_need_user(state: ResearchLoopState) -> dict[str, Any]:
        packets = list(state.get("feedback_packets", []))
        blocking = _blocking_feedback(packets)
        stop_reason = blocking.message if blocking is not None else state.get("stop_reason")
        stall_code = progress_stall_code(state) if blocking is None and not stop_reason else None
        if stall_code == "no_agenda_progress":
            packet = budget_feedback(
                _budget(state),
                code="no_agenda_progress",
                message="本轮议程没有新增解决项，自动研究循环已停止。",
            )
            packets = _append_feedback({**state, "feedback_packets": packets}, packet)
            stop_reason = packet.message
        elif stall_code == "no_new_evidence":
            packet = budget_feedback(
                _budget(state),
                code="no_new_evidence",
                message="本轮没有新增证据，自动研究循环已停止。",
            )
            packets = _append_feedback({**state, "feedback_packets": packets}, packet)
            stop_reason = packet.message
        elif not stop_reason and (
            (state.get("evaluation") or {}).get("decision") == "revise"
            and _budget(state).evaluated_iterations >= _budget(state).max_evaluated_iterations
        ):
            packet = budget_feedback(
                _budget(state),
                code="safety_iteration_limit",
                message="自动研究触发安全熔断；议程尚未在安全范围内收敛，请用户处理当前限制。",
            )
            packets = _append_feedback({**state, "feedback_packets": packets}, packet)
            stop_reason = packet.message
        elif state.get("evaluation") and not stop_reason:
            evaluation_summary = str((state.get("evaluation") or {}).get("summary") or "议程仍有需要用户处理的项目。")
            packet = budget_feedback(
                _budget(state),
                code="agenda_requires_user",
                message=evaluation_summary,
            )
            packets = _append_feedback({**state, "feedback_packets": packets}, packet)
            stop_reason = evaluation_summary
        interrupt_kind = state.get("user_interrupt_kind") or state.get("return_to_gate")
        if interrupt_kind not in {"plan_error", "result_limitations", "response_error", "data_input_error"}:
            interrupt_kind = "result_limitations" if state.get("latest_run") else "plan_error"
        cursor = _cursor(state).model_copy(
            update={
                "episode_status": (
                    "limited"
                    if state.get("latest_run") and interrupt_kind in {"result_limitations", "response_error"}
                    else "plan_error"
                ),
                "stage": "human_gate",
                "terminal_reason": stop_reason or interrupt_kind,
            }
        )
        projected = {
            **state,
            "phase": "awaiting_user",
            "feedback_packets": packets,
            "stop_reason": stop_reason,
            "user_interrupt_kind": interrupt_kind,
            "loop_cursor": cursor.model_dump(mode="json"),
        }
        record: str | None = None
        audit_error: str | None = None
        try:
            record = str(
                write_loop_record(
                    thread_id=state["thread_id"],
                    state=projected,
                    outcome="need_user",
                )
            )
        except Exception as exc:  # noqa: BLE001 - audit must not suppress the human gate
            audit_error = f"{type(exc).__name__}: {exc}"
            packet = FeedbackPacket(
                source="evaluator",
                code="loop_audit_write_failed",
                severity="warning",
                message=f"研究循环审计记录写入失败：{audit_error}",
                recommendation="用户交互状态仍然有效；请检查应用数据目录的写入权限。",
            )
            packets = _append_feedback({**state, "feedback_packets": packets}, packet)
        return {
            "phase": "awaiting_user",
            "control": "interrupt",
            "feedback_packets": packets,
            "stop_reason": stop_reason,
            "user_interrupt_kind": interrupt_kind,
            "loop_cursor": cursor.model_dump(mode="json"),
            "loop_records": (
                [*state.get("loop_records", []), record]
                if record is not None
                else list(state.get("loop_records", []))
            ),
            "events": _event(
                state,
                f"等待用户处理：{interrupt_kind} · {_short(stop_reason or '需要用户决策', 42)}",
                trace_category="evaluation" if interrupt_kind == "result_limitations" else "error",
                feedback_count=len(packets),
                audit_error=audit_error,
            ),
        }

    def persist_stop(state: ResearchLoopState) -> dict[str, Any]:
        stop_reason = state.get("stop_reason")
        evaluation = state.get("evaluation") or {}
        if not stop_reason and evaluation.get("decision") == "reject":
            stop_reason = f"评估器拒绝当前结果：{evaluation.get('summary') or '结果未通过验收。'}"
        if not stop_reason:
            stop_reason = "研究循环已停止。"
        stopped_state = dict(state)
        stopped_state["phase"] = "stopped"
        stopped_state["stop_reason"] = stop_reason
        cursor = _cursor(state).model_copy(
            update={
                "episode_status": "rejected" if evaluation.get("decision") == "reject" else "stopped",
                "stage": "terminal",
                "terminal_reason": stop_reason,
            }
        )
        stopped_state["loop_cursor"] = cursor.model_dump(mode="json")
        records = list(state.get("loop_records", []))
        record: str | None = None
        try:
            record = str(write_loop_record(thread_id=state["thread_id"], state=stopped_state, outcome="stopped"))
            records.append(record)
        except Exception as exc:  # noqa: BLE001 - best-effort terminal audit
            stop_reason = f"{stop_reason}；停止记录写入失败：{type(exc).__name__}: {exc}"
        try:
            execution.evict_prepared(_plan(state))
        except Exception:  # noqa: BLE001 - terminal cleanup is best effort
            execution.evict_prepared()
        episode_summaries = _upsert_episode_summary(
            list(state.get("episode_summaries", [])),
            _episode_summary(state, cursor=cursor, status="stopped"),
        )
        return {
            "phase": "stopped",
            "control": "stop",
            "stop_reason": stop_reason,
            "loop_cursor": cursor.model_dump(mode="json"),
            "loop_records": records,
            "episode_summaries": episode_summaries,
            "events": _event(
                state,
                f"保存停止结果：{_short(stop_reason, 44)}",
                "stopped",
                trace_category="evaluation",
                record=record,
            ),
        }

    def persist_failure(state: ResearchLoopState) -> dict[str, Any]:
        reason = state.get("stop_reason") or "研究循环发生不可恢复错误。"
        cursor = _cursor(state).model_copy(
            update={
                "episode_status": "failed",
                "stage": "terminal",
                "terminal_reason": reason,
            }
        )
        failed_state = {
            **state,
            "phase": "failed",
            "stop_reason": reason,
            "loop_cursor": cursor.model_dump(mode="json"),
        }
        records = list(state.get("loop_records", []))
        try:
            record = write_loop_record(thread_id=state["thread_id"], state=failed_state, outcome="failed")
            records.append(str(record))
        except Exception as exc:  # noqa: BLE001 - best-effort terminal audit
            reason = f"{reason}；失败记录写入失败：{type(exc).__name__}: {exc}"
        try:
            execution.evict_prepared(_plan(state))
        except Exception:  # noqa: BLE001 - terminal cleanup is best effort
            execution.evict_prepared()
        episode_summaries = _upsert_episode_summary(
            list(state.get("episode_summaries", [])),
            _episode_summary(state, cursor=cursor, status="failed"),
        )
        return {
            "phase": "failed",
            "control": "failed",
            "stop_reason": reason,
            "loop_cursor": cursor.model_dump(mode="json"),
            "loop_records": records,
            "episode_summaries": episode_summaries,
            "events": _event(
                state,
                f"保存失败状态：{_short(reason, 44)}",
                "failed",
                trace_category="error",
            ),
        }

    def explain_result(state: ResearchLoopState) -> dict[str, Any]:
        try:
            decision_value = state.get("decision") or {}
            explanation_question = (
                state.get("explanation_request") or _latest_turn(state)
                if decision_value.get("intent") == "explain_result"
                else "请根据已验证证据总结结果并明确限制。"
            )
            decision, _ = main_agent.decide(
                question=explanation_question,
                status="completed",
                config=_config(state),
                plan=_plan(state),
                data_profile=state.get("data_profile"),
                quality_report=state.get("quality_report"),
                summary=state.get("eda_summary"),
                evaluation=state.get("evaluation"),
                history=_messages(state),
                available_skills=skills.metadata(),
                episode_summaries=_scoped_episode_summaries(state),
                active_gate=state.get("user_interrupt_kind") or state.get("return_to_gate"),
                episode_goal=_episode_goal(state),
                latest_run=state.get("latest_run"),
                current_turn_id=state.get("active_turn_id"),
            )
            answer = decision.response.strip()
            if not answer:
                raise ResearchModelUnavailableError("模型没有返回结果解释。")
        except Exception as exc:  # noqa: BLE001 - model explanation boundary
            packet = exception_feedback(exc, source="evaluator", requires_user=True)
            return {
                "phase": "awaiting_user",
                "control": "error",
                "assistant_message": "",
                "feedback_packets": _append_feedback(state, packet),
                "stop_reason": packet.message,
                "user_interrupt_kind": "response_error",
                "events": _event(
                    state,
                    f"结果解释失败：{_short(packet.message, 44)}",
                    "failed",
                    trace_category="error",
                ),
            }
        return_to_approval = state.get("return_to_gate") == "plan_approval"
        return {
            "phase": "awaiting_approval" if return_to_approval else "awaiting_user",
            "control": "result",
            "loop_cursor": _cursor(state).model_copy(
                update={
                    "episode_status": "accepted"
                    if (state.get("evaluation") or {}).get("decision") == "accept"
                    else "limited",
                    "stage": "result_review",
                }
            ).model_dump(mode="json"),
            "assistant_message": answer,
            "messages": _bounded_graph_messages([
                *state.get("messages", []),
                _conversation_message(
                    role="assistant",
                    content=answer,
                    turn_id=state.get("active_turn_id"),
                    episode_id=_cursor(state).episode_id,
                ),
            ]),
            "events": _event(
                state,
                f"解释验证结果：{(state.get('latest_run') or {}).get('run_id', '当前证据')}",
                trace_category="evaluation",
            ),
        }

    def reply(state: ResearchLoopState) -> dict[str, Any]:
        try:
            answer = DialogueDecision.model_validate(state["decision"]).response.strip()
            if not answer:
                raise ResearchPlanValidationError("大模型没有返回可展示回复。")
        except Exception as exc:  # noqa: BLE001 - dialogue response boundary
            packet = exception_feedback(exc, source="plan_validator", requires_user=True)
            return {
                "phase": "awaiting_user",
                "control": "error",
                "assistant_message": "",
                "feedback_packets": _append_feedback(state, packet),
                "stop_reason": packet.message,
                "user_interrupt_kind": "response_error",
                "events": _event(
                    state,
                    f"研究回复失败：{_short(packet.message, 44)}",
                    "failed",
                    trace_category="error",
                ),
            }
        return_to_approval = state.get("return_to_gate") == "plan_approval"
        return {
            "phase": "awaiting_approval" if return_to_approval else "awaiting_user",
            "control": "result",
            "assistant_message": answer,
            "messages": _bounded_graph_messages([
                *state.get("messages", []),
                _conversation_message(
                    role="assistant",
                    content=answer,
                    turn_id=state.get("active_turn_id"),
                    episode_id=_cursor(state).episode_id,
                ),
            ]),
            "events": _event(
                state,
                f"回答用户：{_short(_latest_turn(state), 44)}",
                trace_category="agent",
            ),
        }

    def route_response(state: ResearchLoopState) -> Literal["result", "approval", "error"]:
        if state.get("control") == "error":
            return "error"
        if state.get("return_to_gate") == "plan_approval":
            return "approval"
        if state.get("return_to_gate") in {
            "plan_error",
            "result_limitations",
            "result_rejected",
            "response_error",
            "finalization_error",
            "data_input_error",
        }:
            return "error"
        return "result"

    def user_interrupt(state: ResearchLoopState) -> dict[str, Any]:
        interrupt_kind = state.get("user_interrupt_kind") or state.get("return_to_gate")
        if interrupt_kind not in {
            "plan_error",
            "result_limitations",
            "result_rejected",
            "response_error",
            "finalization_error",
            "data_input_error",
        }:
            interrupt_kind = "result_limitations" if state.get("latest_run") else "plan_error"
        has_result = interrupt_kind in {"result_limitations", "result_rejected"}
        is_response_error = interrupt_kind == "response_error"
        is_finalization_error = interrupt_kind == "finalization_error"
        is_data_input_error = interrupt_kind == "data_input_error"
        is_rejected = interrupt_kind == "result_rejected"
        is_version_mismatch = any(
            item.get("code") == "session_skill_version_mismatch"
            for item in state.get("feedback_packets", [])
        )
        if is_version_mismatch:
            message = state.get("stop_reason") or "当前对话与已安装研究协议版本不兼容，请新建对话。"
            choices = ["stop"]
        elif is_finalization_error:
            message = state.get("stop_reason") or "最终证据合并或报告生成失败；工具结果仍保留。"
            choices = ["retry", "stop"]
        elif is_response_error:
            message = state.get("stop_reason") or "大模型没有生成可展示回复。"
            choices = ["retry", "stop"]
        elif is_data_input_error:
            message = state.get("stop_reason") or "研究数据无法解析，请检查或重新选择数据。"
            choices = ["retry", "stop"]
        elif is_rejected:
            message = state.get("stop_reason") or "评估器拒绝当前结果。"
            choices = ["modify", "stop"]
        else:
            message = state.get("stop_reason") or (
                "当前最佳结果需要用户决策。" if has_result else "尚未生成可执行研究方案。"
            )
            choices = (
                ["modify", "followup", "stop"]
                if has_result
                else ["retry", "modify", "clarify", "stop"]
            )
        payload = _interrupt_payload(
            state,
            kind=interrupt_kind,
            phase="awaiting_user",
            message=message,
            choices=choices,
            plan=state.get("current_plan"),
            evaluation=state.get("evaluation"),
            result=state.get("latest_run") if has_result else None,
        )
        response = ResumePayload.model_validate(interrupt(payload.model_dump(mode="json")))
        if not _resume_identity_matches(response, payload):
            return {
                "phase": "awaiting_user",
                "control": "reinterrupt",
                "feedback_packets": _stale_resume_feedback(state, response, payload),
            }
        if response.action not in payload.choices:
            return {
                "phase": "awaiting_user",
                "control": "reinterrupt",
                "feedback_packets": _invalid_resume_feedback(state, response, payload),
                "events": _event(
                    state,
                    f"拒绝越界用户操作：{response.action}",
                    "warning",
                    trace_category="error",
                ),
            }
        if is_finalization_error and response.action == "retry":
            return {
                "phase": "evaluating",
                "control": "retry_finalize",
                "stop_reason": None,
                "user_interrupt_kind": None,
            }
        if is_response_error and response.action == "retry":
            return {
                "phase": "evaluating" if state.get("latest_run") else "understanding",
                "control": "explain" if state.get("latest_run") else "continue",
                "stop_reason": None,
                "user_interrupt_kind": None,
            }
        if not has_result and response.action == "retry":
            budget = _budget(state).reset_for_iteration()
            cursor = _cursor(state).next_iteration()
            can_plan = bool(state.get("active_skill") and state.get("study_config"))
            return {
                "phase": "planning" if can_plan else "understanding",
                "control": "retry_plan" if can_plan else "continue",
                "plan_origin": "initial",
                "stop_reason": None,
                "user_interrupt_kind": None,
                "loop_cursor": cursor.model_dump(mode="json"),
                "budget": budget.model_dump(mode="json"),
            }
        if not has_result and response.action == "clarify":
            return {
                "phase": "awaiting_user",
                "control": "reinterrupt",
                "stop_reason": "当前没有通过校验的研究方案，因此不能接受或直接执行；请回复“重试”或修改研究要求。",
            }
        if response.action in {"modify", "followup", "next_round"}:
            message = response.message.strip()
            return {
                "phase": "understanding",
                "control": "continue",
                "return_to_gate": interrupt_kind,
                "latest_turn": message,
                "active_turn_id": response.turn_id or state.get("active_turn_id"),
                "explanation_request": "",
                # A conversational follow-up does not resolve the pending
                # evaluation. Keep its reason for the gate shown after reply.
                "stop_reason": state.get("stop_reason"),
                "user_interrupt_kind": None,
                "messages": _bounded_graph_messages([
                    *_messages_before_user_turn(
                        _resume_messages(state, response),
                        question=message,
                        turn_id=response.turn_id,
                    ),
                    _conversation_message(
                        role="user",
                        content=message,
                        message_id=response.message_id,
                        turn_id=response.turn_id,
                        episode_id=_cursor(state).episode_id,
                    ),
                ]),
            }
        # Stopping out of an error interrupt must keep the cause in the terminal record.
        cause = state.get("stop_reason") if interrupt_kind in {"response_error", "finalization_error"} else None
        reason = f"用户停止研究循环；原因：{cause}" if cause else "用户停止研究循环。"
        return {"phase": "stopped", "control": "stop", "stop_reason": reason}

    def result_interrupt(state: ResearchLoopState) -> dict[str, Any]:
        payload = _interrupt_payload(
            state,
            kind="result",
            phase="awaiting_user",
            message=state.get("assistant_message") or "研究结果已就绪。",
            choices=["followup", "next_round", "stop"],
            plan=state.get("current_plan"),
            evaluation=state.get("evaluation"),
            result=state.get("latest_run"),
        )
        response = ResumePayload.model_validate(interrupt(payload.model_dump(mode="json")))
        if not _resume_identity_matches(response, payload):
            return {
                "phase": "awaiting_user",
                "control": "reinterrupt",
                "feedback_packets": _stale_resume_feedback(state, response, payload),
            }
        if response.action not in payload.choices:
            return {
                "phase": "awaiting_user",
                "control": "reinterrupt",
                "feedback_packets": _invalid_resume_feedback(state, response, payload),
                "events": _event(
                    state,
                    f"拒绝越界结果操作：{response.action}",
                    "warning",
                    trace_category="error",
                ),
            }
        if response.action in {"followup", "next_round"}:
            message = response.message.strip()
            return {
                "phase": "understanding",
                "control": "continue",
                "latest_turn": message,
                "active_turn_id": response.turn_id or state.get("active_turn_id"),
                "explanation_request": "",
                "return_to_gate": "result" if response.action == "followup" else None,
                "messages": _bounded_graph_messages([
                    *_messages_before_user_turn(
                        _resume_messages(state, response),
                        question=message,
                        turn_id=response.turn_id,
                    ),
                    _conversation_message(
                        role="user",
                        content=message,
                        message_id=response.message_id,
                        turn_id=response.turn_id,
                        episode_id=_cursor(state).episode_id,
                    ),
                ]),
                "plan_origin": "initial" if response.action == "next_round" else state.get("plan_origin", "initial"),
            }
        return {"phase": "completed", "control": "end", "stop_reason": "用户结束当前研究。"}

    def route_result_interrupt(state: ResearchLoopState) -> Literal["end", "continue", "reinterrupt"]:
        control = state.get("control", "continue")
        return control if control in {"end", "reinterrupt"} else "continue"  # type: ignore[return-value]

    def route_user_interrupt(
        state: ResearchLoopState,
    ) -> Literal["stop", "explain", "retry_finalize", "retry_plan", "reinterrupt", "continue"]:
        control = state.get("control", "continue")
        return control if control in {"stop", "explain", "retry_finalize", "retry_plan", "reinterrupt"} else "continue"  # type: ignore[return-value]

    graph = StateGraph(ResearchLoopState)
    for name, node in {
        "ingest_user": ingest_user,
        "main_agent": understand,
        "resolve_study_context": resolve_study_context,
        "begin_episode": begin_episode,
        "resolve_skill": resolve_skill,
        "eda_subagent": create_plan,
        "revise_user_plan": revise_user_plan,
        "confirm_existing_plan": confirm_existing_plan,
        "validate_plan": validate_plan,
        "prepare_approval": prepare_approval,
        "approval_interrupt": approval_interrupt,
        "lock_plan": lock_plan,
        "mark_tool_running": mark_tool_running,
        "execute_tool": execute_tool,
        "validate_tool_result": validate_tool_result,
        "advance_tool": advance_tool,
        "complete_tool_batch": complete_tool_batch,
        "analysis_selector": analysis_selector,
        "prepare_plan_repair": prepare_plan_repair,
        "finalize_iteration": finalize_iteration,
        "prepare_invalid_evaluation": prepare_invalid_evaluation,
        "prepare_rejected_result": prepare_rejected_result,
        "prepare_evaluation_revision": prepare_evaluation_revision,
        "prepare_dynamic_evaluation_revision": prepare_dynamic_evaluation_revision,
        "prepare_need_user": prepare_need_user,
        "explain_result": explain_result,
        "reply": reply,
        "prepare_forecast_handoff": prepare_forecast_handoff,
        "user_interrupt": user_interrupt,
        "result_interrupt": result_interrupt,
        "persist_stop": persist_stop,
        "persist_failure": persist_failure,
    }.items():
        graph.add_node(name, node)

    graph.add_edge(START, "ingest_user")
    graph.add_edge("ingest_user", "main_agent")
    graph.add_conditional_edges(
        "main_agent",
        route_main,
        {
            "resolve_data": "resolve_study_context",
            "revise_plan": "revise_user_plan",
            "execute_plan": "confirm_existing_plan",
            "business_handoff": "prepare_forecast_handoff",
            "reply": "reply",
            "need_user": "prepare_need_user",
        },
    )
    graph.add_edge("prepare_forecast_handoff", END)
    graph.add_edge("confirm_existing_plan", "validate_plan")
    graph.add_conditional_edges("resolve_skill", route_skill, {"plan": "eda_subagent", "need_user": "prepare_need_user"})
    graph.add_conditional_edges(
        "eda_subagent",
        route_after_plan,
        {"retry": "eda_subagent", "validate": "validate_plan", "need_user": "prepare_need_user"},
    )
    graph.add_conditional_edges(
        "revise_user_plan",
        route_user_revision,
        {"validate": "validate_plan", "retry": "eda_subagent", "need_user": "prepare_need_user"},
    )
    graph.add_conditional_edges(
        "validate_plan",
        route_after_validation,
        {"retry": "eda_subagent", "approval": "prepare_approval", "lock": "lock_plan", "need_user": "prepare_need_user"},
    )
    graph.add_edge("prepare_approval", "approval_interrupt")
    graph.add_conditional_edges(
        "approval_interrupt",
        route_approval,
        {
            "lock": "lock_plan",
            "modify": "main_agent",
            "discuss": "main_agent",
            "stop": "persist_stop",
            "reinterrupt": "approval_interrupt",
        },
    )
    graph.add_conditional_edges(
        "lock_plan",
        route_lock,
        {"execute": "mark_tool_running", "need_user": "prepare_need_user", "failed": "persist_failure"},
    )
    graph.add_conditional_edges(
        "mark_tool_running",
        route_mark_tool,
        {
            "run": "execute_tool",
            "advance": "advance_tool",
            "evaluate": "finalize_iteration",
            "select": "complete_tool_batch",
            "need_user": "prepare_need_user",
            "fail": "persist_failure",
        },
    )
    graph.add_conditional_edges(
        "execute_tool",
        route_tool_execution,
        {
            "validate": "validate_tool_result",
            "retry": "mark_tool_running",
            "advance": "advance_tool",
            "revise": "prepare_plan_repair",
            "select": "analysis_selector",
            "need_user": "prepare_need_user",
            "stop": "persist_stop",
            "fail": "persist_failure",
        },
    )
    graph.add_conditional_edges(
        "validate_tool_result",
        route_tool_execution,
        {
            "advance": "advance_tool",
            "revise": "prepare_plan_repair",
            "select": "analysis_selector",
            "need_user": "prepare_need_user",
            "fail": "persist_failure",
        },
    )
    graph.add_edge("advance_tool", "mark_tool_running")
    graph.add_conditional_edges(
        "complete_tool_batch",
        lambda state: "need_user" if state.get("control") == "need_user" else "select",
        {"select": "analysis_selector", "need_user": "prepare_need_user"},
    )
    graph.add_conditional_edges(
        "resolve_study_context",
        route_resolved_study,
        {
            "new_plan": "begin_episode",
            "revise_plan": "revise_user_plan",
            "execute_plan": "confirm_existing_plan",
            "need_user": "prepare_need_user",
        },
    )
    graph.add_conditional_edges(
        "analysis_selector",
        route_analysis_selector,
        {
            "execute": "mark_tool_running",
            "evaluate": "finalize_iteration",
            "select": "analysis_selector",
            "need_user": "prepare_need_user",
        },
    )
    graph.add_conditional_edges(
        "prepare_plan_repair",
        lambda state: "need_user" if state.get("control") == "need_user" else "repair",
        {"repair": "eda_subagent", "need_user": "prepare_need_user"},
    )
    graph.add_conditional_edges(
        "finalize_iteration",
        route_evaluation,
        {
            "accept": "explain_result",
            "revise": "prepare_evaluation_revision",
            "dynamic_revise": "prepare_dynamic_evaluation_revision",
            "need_user": "prepare_need_user",
            "reject": "prepare_rejected_result",
            "failed": "persist_failure",
            "finalization_error": "user_interrupt",
            "invalid": "prepare_invalid_evaluation",
        },
    )
    graph.add_edge("prepare_invalid_evaluation", "prepare_need_user")
    graph.add_edge("prepare_rejected_result", "user_interrupt")
    graph.add_edge("prepare_evaluation_revision", "eda_subagent")
    graph.add_edge("prepare_dynamic_evaluation_revision", "analysis_selector")
    graph.add_edge("prepare_need_user", "user_interrupt")
    graph.add_conditional_edges(
        "user_interrupt",
        route_user_interrupt,
        {
            "stop": "persist_stop",
            "explain": "explain_result",
            "retry_finalize": "finalize_iteration",
            "retry_plan": "eda_subagent",
            "reinterrupt": "user_interrupt",
            "continue": "main_agent",
        },
    )
    graph.add_edge("begin_episode", "resolve_skill")
    graph.add_conditional_edges(
        "reply",
        route_response,
        {"result": "result_interrupt", "approval": "approval_interrupt", "error": "user_interrupt"},
    )
    graph.add_conditional_edges(
        "explain_result",
        route_response,
        {"result": "result_interrupt", "approval": "approval_interrupt", "error": "user_interrupt"},
    )
    graph.add_conditional_edges(
        "result_interrupt",
        route_result_interrupt,
        {"end": END, "continue": "main_agent", "reinterrupt": "result_interrupt"},
    )
    graph.add_edge("persist_stop", END)
    graph.add_edge("persist_failure", END)
    return graph.compile(checkpointer=checkpointer)
