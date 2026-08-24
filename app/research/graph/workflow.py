"""One persistent, interruptible, self-validating LangGraph research loop."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from time import perf_counter
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from app.research.agent.errors import ResearchModelUnavailableError, ResearchPlanValidationError
from app.research.agent.orchestrator import DialogueDecision, MainResearchAgent
from app.research.agent.schemas import ConversationMessage, EDAPlan
from app.research.application.execution import EDAExecutionService
from app.research.application.planning import EDAPlanningService, noop_progress
from app.research.data.loader import ResearchDataError
from app.research.graph.contracts import (
    ApprovalState,
    AuthorizationEnvelope,
    InterruptPayload,
    LoopBudget,
    ResumePayload,
    ToolCallRecord,
)
from app.research.graph.guards import (
    authorization_envelope,
    budget_feedback,
    evidence_fingerprint,
    exception_feedback,
    plan_fingerprint,
    validate_automatic_revision,
)
from app.research.graph.state import ResearchLoopState
from app.research.reporting.loop_history import write_loop_record
from app.research.schemas.feedback import FeedbackPacket
from app.research.schemas.study import StudyConfig
from app.research.skills.contracts import SkillDefinition
from app.research.skills.loader import SkillLoadError
from app.research.skills.registry import SkillRegistry
from app.research.tools.contracts import ToolCall, ToolResult
from app.research.tools.executor import ToolExecutionError
from app.research.tools.policy import ToolPermissionError
from app.research.tools.registry import ToolRegistry, ToolRegistryError


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _event(state: ResearchLoopState, name: str, status: str = "completed", **details: Any) -> list[dict[str, Any]]:
    return [
        *state.get("events", []),
        {"created_at": _now(), "name": name, "status": status, "details": details},
    ]


def _messages(state: ResearchLoopState) -> list[ConversationMessage]:
    return [ConversationMessage.model_validate(item) for item in state.get("messages", [])]


def _config(state: ResearchLoopState) -> StudyConfig | None:
    value = state.get("study_config")
    return StudyConfig.model_validate(value) if value is not None else None


def _plan(state: ResearchLoopState) -> EDAPlan | None:
    value = state.get("current_plan")
    return EDAPlan.model_validate(value) if value is not None else None


def _feedback(state: ResearchLoopState) -> list[FeedbackPacket]:
    return [FeedbackPacket.model_validate(item) for item in state.get("feedback_packets", [])]


def _append_feedback(state: ResearchLoopState, packet: FeedbackPacket) -> list[dict[str, Any]]:
    return [*state.get("feedback_packets", []), packet.model_dump(mode="json")]


def _budget(state: ResearchLoopState) -> LoopBudget:
    return LoopBudget.model_validate(state.get("budget", {}))


def _add_active_time(state: ResearchLoopState, started: float) -> dict[str, Any]:
    budget = _budget(state)
    budget.active_seconds_used += perf_counter() - started
    return budget.model_dump(mode="json")


def build_research_workflow(
    *,
    main_agent: MainResearchAgent,
    planning: EDAPlanningService,
    execution: EDAExecutionService,
    skills: SkillRegistry,
    tools: ToolRegistry,
    checkpointer: Any,
):
    """Compile the only workflow used by desktop research sessions."""

    def ingest_user(state: ResearchLoopState) -> dict[str, Any]:
        message = state.get("pending_user_message", "").strip()
        messages = list(state.get("messages", []))
        if message:
            messages.append(ConversationMessage(role="user", content=message).model_dump(mode="json"))
        return {
            "phase": "understanding",
            "user_request": message or state.get("user_request", ""),
            "pending_user_message": "",
            "messages": messages,
            "events": _event(state, "接收用户消息", message=message),
        }

    def understand(state: ResearchLoopState) -> dict[str, Any]:
        started = perf_counter()
        try:
            decision, _ = main_agent.decide(
                question=state.get("user_request", ""),
                status=state.get("phase", "idle"),
                config=_config(state),
                plan=_plan(state),
                data_profile=state.get("data_profile"),
                quality_report=state.get("quality_report"),
                summary=state.get("eda_summary"),
                evaluation=state.get("evaluation"),
                history=_messages(state),
                available_skills=skills.metadata(),
            )
        except Exception as exc:  # noqa: BLE001 - converted into structured loop feedback
            packet = exception_feedback(exc, source="plan_validator", requires_user=True)
            return {
                "phase": "awaiting_user",
                "feedback_packets": _append_feedback(state, packet),
                "stop_reason": packet.message,
                "budget": _add_active_time(state, started),
                "events": _event(state, "主 Agent 调用失败", "failed", error=packet.message),
            }
        return {
            "phase": "planning" if decision.intent in {"new_plan", "revise_plan"} else "understanding",
            "decision": decision.model_dump(mode="json"),
            "budget": _add_active_time(state, started),
            "events": _event(state, "主 Agent 完成路由", intent=decision.intent),
        }

    def route_main(state: ResearchLoopState) -> Literal["new_plan", "revise_plan", "execute_plan", "reply", "need_user"]:
        if state.get("phase") == "awaiting_user" and state.get("stop_reason"):
            return "need_user"
        decision = DialogueDecision.model_validate(state["decision"])
        if decision.intent == "new_plan":
            return "new_plan"
        if decision.intent == "revise_plan":
            return "revise_plan"
        if decision.intent == "execute_plan" and _plan(state) is not None:
            return "execute_plan"
        return "reply"

    def confirm_existing_plan(state: ResearchLoopState) -> dict[str, Any]:
        return {
            "phase": "validating_plan",
            "plan_origin": "explicit_confirm",
            "events": _event(state, "用户明确确认已有方案"),
        }

    def resolve_skill(state: ResearchLoopState) -> dict[str, Any]:
        decision = DialogueDecision.model_validate(state["decision"])
        try:
            if not decision.skill_name:
                raise SkillLoadError("大模型生成新方案时没有选择 Skill。")
            skill = skills.get(decision.skill_name)
            if skill.domain != "eda":
                raise SkillLoadError(f"当前循环不支持 {skill.domain} Skill：{skill.name}")
            tools.function_schemas(skill.allowed_tools)
            if "data_quality" not in skill.allowed_tools:
                raise SkillLoadError(f"EDA Skill 必须授权 data_quality：{skill.name}")
        except Exception as exc:  # noqa: BLE001 - converted into structured loop feedback
            packet = exception_feedback(exc, source="skill_validator", requires_user=True)
            return {
                "phase": "awaiting_user",
                "feedback_packets": _append_feedback(state, packet),
                "stop_reason": packet.message,
                "events": _event(state, "Skill 校验失败", "failed", error=packet.message),
            }
        return {
            "active_skill": skill.model_dump(mode="json"),
            "phase": "planning",
            "plan_origin": "initial",
            "events": _event(state, "激活研究 Skill", skill=skill.name, version=skill.version),
        }

    def route_skill(state: ResearchLoopState) -> Literal["plan", "need_user"]:
        return "need_user" if state.get("phase") == "awaiting_user" else "plan"

    def revise_user_plan(state: ResearchLoopState) -> dict[str, Any]:
        plan = _plan(state)
        config = _config(state)
        decision = DialogueDecision.model_validate(state["decision"])
        if plan is None or config is None:
            packet = FeedbackPacket(
                source="plan_validator",
                code="missing_plan_for_revision",
                severity="error",
                message="当前没有可修订方案。",
                requires_user=True,
            )
            return {"phase": "awaiting_user", "feedback_packets": _append_feedback(state, packet)}
        try:
            revised, assistant_message = main_agent.revise_plan(
                question=state.get("user_request", ""),
                plan=plan,
                config=config,
                decision=decision,
            )
        except Exception as exc:  # noqa: BLE001 - converted into structured loop feedback
            packet = exception_feedback(exc, source="plan_validator", retryable=True)
            return {
                "phase": "planning",
                "feedback_packets": _append_feedback(state, packet),
                "plan_origin": "user_revision",
            }
        return {
            "current_plan": revised.model_dump(mode="json"),
            "plan_history": [*state.get("plan_history", []), revised.model_dump(mode="json")],
            "plan_fingerprints": [*state.get("plan_fingerprints", []), plan_fingerprint(revised)],
            "plan_origin": "user_revision",
            "phase": "validating_plan",
            "messages": [
                *state.get("messages", []),
                ConversationMessage(role="assistant", content=assistant_message).model_dump(mode="json"),
            ],
            "events": _event(state, "生成用户修订方案", revision=revised.revision),
        }

    def route_user_revision(state: ResearchLoopState) -> Literal["validate", "retry", "need_user"]:
        if state.get("phase") == "awaiting_user":
            return "need_user"
        if state.get("phase") == "planning":
            return "retry"
        return "validate"

    def create_plan(state: ResearchLoopState) -> dict[str, Any]:
        started = perf_counter()
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
            return {"phase": "awaiting_user", "feedback_packets": _append_feedback(state, packet)}
        skill = SkillDefinition.model_validate(skill_value)
        feedback = _feedback(state)
        origin = state.get("plan_origin", "initial")
        try:
            if origin == "automatic_evaluation" and _plan(state) is not None:
                proposal = planning.revise_from_feedback(
                    current_plan=_plan(state),  # type: ignore[arg-type]
                    study_config=config,
                    skill=skill,
                    feedback=feedback,
                    conversation=_messages(state),
                    source="automatic_evaluation",
                    progress=noop_progress,
                )
            else:
                proposal = planning.propose(
                    question=state.get("user_request", ""),
                    study_config=config,
                    conversation=_messages(state),
                    progress=noop_progress,
                    skill=skill,
                    feedback=feedback,
                )
        except Exception as exc:  # noqa: BLE001 - model/validator boundary
            budget = _budget(state)
            packet = exception_feedback(exc, source="plan_validator", retryable=True)
            if budget.plan_repairs_used < budget.max_plan_repairs:
                budget.plan_repairs_used += 1
                phase = "planning"
            else:
                packet = packet.model_copy(update={"retryable": False, "requires_user": True})
                phase = "awaiting_user"
            budget.active_seconds_used += perf_counter() - started
            return {
                "phase": phase,
                "feedback_packets": _append_feedback(state, packet),
                "budget": budget.model_dump(mode="json"),
                "stop_reason": packet.message if phase == "awaiting_user" else None,
                "events": _event(state, "方案生成校验失败", "warning", error=packet.message),
            }
        fingerprint = plan_fingerprint(proposal.plan)
        if fingerprint in state.get("plan_fingerprints", []):
            packet = budget_feedback(_budget(state), code="duplicate_plan", message="自动修订生成了重复方案。")
            return {
                "phase": "awaiting_user",
                "feedback_packets": _append_feedback(state, packet),
                "stop_reason": packet.message,
            }
        budget = _budget(state)
        budget.active_seconds_used += perf_counter() - started
        return {
            "phase": "validating_plan",
            "current_plan": proposal.plan.model_dump(mode="json"),
            "data_profile": proposal.data_profile.model_dump(mode="json"),
            "quality_report": proposal.quality_report.model_dump(mode="json"),
            "data_fingerprint": proposal.plan.data_fingerprint,
            "plan_history": [*state.get("plan_history", []), proposal.plan.model_dump(mode="json")],
            "plan_fingerprints": [*state.get("plan_fingerprints", []), fingerprint],
            "budget": budget.model_dump(mode="json"),
            "events": _event(state, "生成候选方案", plan_id=proposal.plan.plan_id),
        }

    def route_after_plan(state: ResearchLoopState) -> Literal["retry", "validate", "need_user"]:
        if state.get("phase") == "awaiting_user":
            return "need_user"
        return "retry" if state.get("phase") == "planning" else "validate"

    def validate_plan(state: ResearchLoopState) -> dict[str, Any]:
        plan = _plan(state)
        config = _config(state)
        if plan is None or config is None:
            packet = FeedbackPacket(
                source="plan_validator",
                code="missing_plan_or_config",
                severity="fatal",
                message="计划或研究配置缺失。",
                requires_user=True,
            )
            return {"phase": "awaiting_user", "feedback_packets": _append_feedback(state, packet)}
        try:
            execution.prepare(plan=plan, study_config=config)
        except Exception as exc:  # noqa: BLE001 - validator boundary
            message = str(exc)
            immutable = any(word in message for word in ("版本", "指纹", "发生变化", "未注册", "未授权"))
            packet = exception_feedback(
                exc,
                source="plan_validator",
                retryable=not immutable,
                requires_user=immutable,
            )
            budget = _budget(state)
            if not immutable and budget.plan_repairs_used < budget.max_plan_repairs:
                budget.plan_repairs_used += 1
                phase = "planning"
            else:
                phase = "awaiting_user"
                packet = packet.model_copy(update={"retryable": False, "requires_user": True})
            return {
                "phase": phase,
                "feedback_packets": _append_feedback(state, packet),
                "budget": budget.model_dump(mode="json"),
                "stop_reason": packet.message if phase == "awaiting_user" else None,
            }
        if state.get("plan_origin") == "automatic_evaluation" and state.get("authorization_envelope"):
            violation = validate_automatic_revision(
                plan,
                AuthorizationEnvelope.model_validate(state["authorization_envelope"]),
            )
            if violation is not None:
                return {
                    "phase": "awaiting_user",
                    "feedback_packets": _append_feedback(state, violation),
                    "stop_reason": violation.message,
                }
        return {"phase": "validating_plan", "events": _event(state, "确定性计划校验通过")}

    def route_after_validation(state: ResearchLoopState) -> Literal["retry", "approval", "lock", "need_user"]:
        if state.get("phase") == "awaiting_user":
            return "need_user"
        if state.get("phase") == "planning":
            return "retry"
        return "lock" if state.get("plan_origin") in {"automatic_evaluation", "explicit_confirm"} else "approval"

    def prepare_approval(state: ResearchLoopState) -> dict[str, Any]:
        plan = _plan(state)
        timeout = int(state.get("approval_timeout_seconds", 30))
        approval = ApprovalState(
            status="waiting",
            plan_id=plan.plan_id if plan else None,
            deadline=(datetime.now(UTC) + timedelta(seconds=timeout)).isoformat(),
            remaining_seconds=timeout,
            origin="user_revision" if state.get("plan_origin") == "user_revision" else "initial",
        )
        return {
            "phase": "awaiting_approval",
            "approval_state": approval.model_dump(mode="json"),
            "events": _event(state, "等待用户审批", deadline=approval.deadline),
        }

    def approval_interrupt(state: ResearchLoopState) -> dict[str, Any]:
        approval = ApprovalState.model_validate(state["approval_state"])
        payload = InterruptPayload(
            kind="plan_approval",
            phase="awaiting_approval",
            message="请确认、修改或拒绝当前研究方案。",
            choices=["approve", "modify", "reject", "timeout_accept"],
            plan=state.get("current_plan"),
            deadline=approval.deadline,
            remaining_seconds=approval.remaining_seconds,
        )
        response = ResumePayload.model_validate(interrupt(payload.model_dump(mode="json")))
        update: dict[str, Any]
        if response.action in {"approve", "timeout_accept"}:
            approval.status = "approved"
            approval.decision = response.action
            update = {"phase": "validating_plan"}
        elif response.action == "modify":
            approval.status = "none"
            approval.decision = "modify"
            message = response.message.strip()
            update = {
                "phase": "understanding",
                "user_request": message,
                "plan_origin": "user_revision",
                "messages": [
                    *state.get("messages", []),
                    ConversationMessage(role="user", content=message).model_dump(mode="json"),
                ],
            }
        else:
            approval.status = "rejected"
            approval.decision = "reject"
            update = {"phase": "stopped", "stop_reason": "用户拒绝执行当前方案。"}
        update["approval_state"] = approval.model_dump(mode="json")
        update["events"] = _event(state, "用户处理方案", action=response.action)
        return update

    def route_approval(state: ResearchLoopState) -> Literal["lock", "modify", "stop"]:
        decision = ApprovalState.model_validate(state["approval_state"]).decision
        if decision == "modify":
            return "modify"
        return "stop" if decision == "reject" else "lock"

    def lock_plan(state: ResearchLoopState) -> dict[str, Any]:
        plan = _plan(state)
        if plan is None:
            raise ResearchPlanValidationError("无法锁定空方案")
        budget = _budget(state)
        if budget.active_seconds_used >= budget.max_active_seconds:
            packet = budget_feedback(budget, code="active_time_budget", message="活动处理时间预算已耗尽。")
            return {"phase": "awaiting_user", "feedback_packets": _append_feedback(state, packet)}
        if budget.research_iterations_used >= budget.max_research_iterations:
            packet = budget_feedback(budget, code="research_iteration_budget", message="研究循环次数已耗尽。")
            return {"phase": "awaiting_user", "feedback_packets": _append_feedback(state, packet)}
        envelope = state.get("authorization_envelope")
        if state.get("plan_origin") != "automatic_evaluation" or envelope is None:
            envelope = authorization_envelope(plan).model_dump(mode="json")
        queue = [call.model_dump(mode="json") for call in execution.compile_tool_queue(plan)]
        if budget.tool_calls_used + len(queue) > budget.max_tool_calls:
            packet = budget_feedback(budget, code="tool_call_budget", message="工具调用预算不足。")
            return {"phase": "awaiting_user", "feedback_packets": _append_feedback(state, packet)}
        return {
            "phase": "executing_tools",
            "authorization_envelope": envelope,
            "tool_queue": queue,
            "tool_cursor": 0,
            "tool_records": {
                call["call_id"]: ToolCallRecord(call=call).model_dump(mode="json") for call in queue
            },
            "tool_results": [],
            "tool_outcome": "",
            "events": _event(state, "锁定方案并生成工具队列", calls=len(queue)),
        }

    def route_lock(state: ResearchLoopState) -> Literal["execute", "need_user"]:
        return "need_user" if state.get("phase") == "awaiting_user" else "execute"

    def mark_tool_running(state: ResearchLoopState) -> dict[str, Any]:
        cursor = int(state.get("tool_cursor", 0))
        queue = state.get("tool_queue", [])
        if cursor >= len(queue):
            return {"tool_outcome": "done", "phase": "evaluating"}
        call = ToolCall.model_validate(queue[cursor])
        records = dict(state.get("tool_records", {}))
        record = ToolCallRecord.model_validate(records[call.call_id])
        if record.status in {"completed", "reused"} and record.result is not None:
            record.status = "reused"
            records[call.call_id] = record.model_dump(mode="json")
            results = list(state.get("tool_results", []))
            if not any(item["call"]["call_id"] == call.call_id for item in results):
                results.append(record.result)
            return {"tool_records": records, "tool_results": results, "tool_outcome": "reused"}
        budget = _budget(state)
        retry_count = budget.tool_retry_counts.get(call.call_id, 0)
        if record.status in {"running", "failed"}:
            if retry_count >= budget.max_tool_retries:
                packet = budget_feedback(budget, code="tool_retry_budget", message=f"工具 {call.name} 重试耗尽。")
                return {
                    "phase": "awaiting_user",
                    "tool_outcome": "need_user",
                    "feedback_packets": _append_feedback(state, packet),
                }
            budget.tool_retry_counts[call.call_id] = retry_count + 1
        record.status = "running"
        record.attempts += 1
        record.started_at = _now()
        records[call.call_id] = record.model_dump(mode="json")
        budget.tool_calls_used += 1
        return {
            "tool_records": records,
            "budget": budget.model_dump(mode="json"),
            "tool_outcome": "run",
            "events": _event(state, "标记工具运行", call_id=call.call_id, attempt=record.attempts),
        }

    def route_mark_tool(state: ResearchLoopState) -> Literal["run", "advance", "evaluate", "need_user"]:
        outcome = state.get("tool_outcome")
        if outcome == "done":
            return "evaluate"
        if outcome == "reused":
            return "advance"
        return "need_user" if outcome == "need_user" else "run"

    def execute_tool(state: ResearchLoopState) -> dict[str, Any]:
        started = perf_counter()
        cursor = int(state.get("tool_cursor", 0))
        call = ToolCall.model_validate(state["tool_queue"][cursor])
        plan = _plan(state)
        config = _config(state)
        if plan is None or config is None:
            raise ResearchPlanValidationError("工具执行缺少计划或配置")
        try:
            result = execution.execute_call(plan=plan, study_config=config, call=call)
        except Exception as exc:  # noqa: BLE001 - tool boundary
            transient = isinstance(exc, (OSError, TimeoutError))
            permission_or_data = isinstance(exc, (ToolPermissionError, ResearchDataError))
            plan_error = isinstance(exc, (ToolExecutionError, ResearchPlanValidationError, ToolRegistryError))
            packet = exception_feedback(
                exc,
                source="tool_executor",
                step_id=call.call_id,
                retryable=transient or plan_error,
                requires_user=permission_or_data,
            )
            outcome = "retry" if transient else ("revise" if plan_error else ("need_user" if permission_or_data else "stop"))
            if outcome == "stop":
                packet = packet.model_copy(
                    update={
                        "severity": "fatal",
                        "recommendation": "当前工具异常无法安全恢复，研究循环已停止。",
                    }
                )
            records = dict(state.get("tool_records", {}))
            record = ToolCallRecord.model_validate(records[call.call_id])
            record.status = "failed"
            record.error = packet
            record.finished_at = _now()
            records[call.call_id] = record.model_dump(mode="json")
            return {
                "tool_records": records,
                "tool_outcome": outcome,
                "feedback_packets": _append_feedback(state, packet),
                "stop_reason": packet.message if outcome in {"need_user", "stop"} else None,
                "budget": _add_active_time(state, started),
            }
        return {
            "phase": "validating_result",
            "tool_outcome": "validate",
            "pending_tool_result": result.model_dump(mode="json"),
            "budget": _add_active_time(state, started),
            "events": _event(
                state,
                "工具执行完成",
                call_id=call.call_id,
                duration_ms=result.duration_ms,
                tool=call.name,
            ),
        }

    def route_tool_execution(state: ResearchLoopState) -> Literal["validate", "retry", "revise", "need_user", "stop", "advance"]:
        return state.get("tool_outcome", "stop")  # type: ignore[return-value]

    def validate_tool_result(state: ResearchLoopState) -> dict[str, Any]:
        cursor = int(state.get("tool_cursor", 0))
        call = ToolCall.model_validate(state["tool_queue"][cursor])
        result = ToolResult.model_validate(state["pending_tool_result"])
        try:
            plan = _plan(state)
            if plan is None:
                raise ResearchPlanValidationError("结果校验缺少锁定计划")
            execution.validate_tool_result(plan=plan, call=call, result=result)
        except Exception as exc:  # noqa: BLE001 - result-validator boundary
            packet = exception_feedback(exc, source="tool_result_validator", step_id=call.call_id, retryable=True)
            return {
                "phase": "planning",
                "tool_outcome": "revise",
                "feedback_packets": _append_feedback(state, packet),
            }
        records = dict(state.get("tool_records", {}))
        record = ToolCallRecord.model_validate(records[call.call_id])
        record.status = "completed"
        record.result = result.model_dump(mode="json")
        record.finished_at = _now()
        records[call.call_id] = record.model_dump(mode="json")
        return {
            "phase": "executing_tools",
            "tool_records": records,
            "tool_results": [*state.get("tool_results", []), result.model_dump(mode="json")],
            "tool_outcome": "advance",
            "events": _event(state, "工具结果校验通过", output_hash=result.output_hash),
        }

    def advance_tool(state: ResearchLoopState) -> dict[str, Any]:
        return {"tool_cursor": int(state.get("tool_cursor", 0)) + 1, "tool_outcome": ""}

    def prepare_plan_repair(state: ResearchLoopState) -> dict[str, Any]:
        budget = _budget(state)
        if budget.plan_repairs_used >= budget.max_plan_repairs:
            packet = budget_feedback(budget, code="plan_repair_budget", message="计划修订次数已耗尽。")
            return {"phase": "awaiting_user", "feedback_packets": _append_feedback(state, packet)}
        budget.plan_repairs_used += 1
        return {
            "phase": "planning",
            "plan_origin": "automatic_evaluation",
            "budget": budget.model_dump(mode="json"),
            "tool_queue": [],
            "tool_cursor": 0,
            "tool_results": [],
        }

    def finalize_iteration(state: ResearchLoopState) -> dict[str, Any]:
        started = perf_counter()
        plan = _plan(state)
        config = _config(state)
        if plan is None or config is None:
            raise ResearchPlanValidationError("评估前缺少计划或配置")
        run = execution.finalize(
            plan=plan,
            study_config=config,
            tool_results=[ToolResult.model_validate(item) for item in state.get("tool_results", [])],
            conversation=_messages(state),
            progress=noop_progress,
            loop_context={
                "thread_id": state["thread_id"],
                "research_iteration": _budget(state).research_iterations_used + 1,
                "plan_history": [item.get("plan_id") for item in state.get("plan_history", [])],
                "feedback_packets": state.get("feedback_packets", []),
                "tool_records": state.get("tool_records", {}),
                "authorization_envelope": state.get("authorization_envelope"),
                "budget_before_evaluation": state.get("budget", {}),
            },
        )
        budget = _budget(state)
        budget.research_iterations_used += 1
        budget.active_seconds_used += perf_counter() - started
        evaluation_payload = run.evaluation.model_dump(mode="json")
        evidence_hash = evidence_fingerprint(run.eda_summary, evaluation_payload)
        latest = {
            "run_id": run.run_id,
            "plan_id": run.plan.plan_id,
            "artifact_directory": str(run.artifact_directory),
            "report_path": str(run.report_path),
            "figure_paths": {key: str(value) for key, value in run.figure_paths.items()},
            "evaluation": evaluation_payload,
            "eda_summary": run.eda_summary,
            "quality_report": run.quality_report.model_dump(mode="json"),
        }
        return {
            "phase": "evaluating",
            "evaluation": evaluation_payload,
            "eda_summary": run.eda_summary,
            "latest_run": latest,
            "run_history": [*state.get("run_history", []), latest],
            "evidence_fingerprints": [*state.get("evidence_fingerprints", []), evidence_hash],
            "budget": budget.model_dump(mode="json"),
            "events": _event(state, "完成确定性评估", decision=run.evaluation.decision, run_id=run.run_id),
        }

    def route_evaluation(state: ResearchLoopState) -> Literal["accept", "revise", "need_user", "reject"]:
        decision = (state.get("evaluation") or {}).get("decision", "need_user")
        if decision in {"accept", "reject", "need_user"}:
            return decision  # type: ignore[return-value]
        budget = _budget(state)
        evidence = state.get("evidence_fingerprints", [])
        if len(evidence) >= 2 and evidence[-1] == evidence[-2]:
            return "need_user"
        if budget.research_iterations_used >= budget.max_research_iterations:
            return "need_user"
        if budget.active_seconds_used >= budget.max_active_seconds:
            return "need_user"
        return "revise"

    def prepare_evaluation_revision(state: ResearchLoopState) -> dict[str, Any]:
        evaluation = state.get("evaluation") or {}
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
        return {
            "phase": "planning",
            "plan_origin": "automatic_evaluation",
            "feedback_packets": [*state.get("feedback_packets", []), *[item.model_dump(mode="json") for item in packets]],
            "tool_queue": [],
            "tool_cursor": 0,
            "tool_results": [],
            "events": _event(state, "评估反馈进入自动修订", feedback_count=len(packets)),
        }

    def prepare_need_user(state: ResearchLoopState) -> dict[str, Any]:
        packets = list(state.get("feedback_packets", []))
        evidence = state.get("evidence_fingerprints", [])
        if len(evidence) >= 2 and evidence[-1] == evidence[-2]:
            packet = budget_feedback(
                _budget(state),
                code="no_new_evidence",
                message="本轮没有新增证据，自动研究循环已停止。",
            )
            packets.append(packet.model_dump(mode="json"))
        elif state.get("evaluation"):
            packet = budget_feedback(
                _budget(state),
                code="automatic_loop_requires_user",
                message="自动研究循环无法在当前预算和审批范围内继续。",
            )
            packets.append(packet.model_dump(mode="json"))
        projected = {**state, "phase": "awaiting_user", "feedback_packets": packets}
        record = write_loop_record(
            thread_id=state["thread_id"],
            state=projected,
            outcome="need_user",
        )
        return {
            "phase": "awaiting_user",
            "feedback_packets": packets,
            "loop_records": [*state.get("loop_records", []), str(record)],
            "events": _event(state, "等待用户处理研究反馈", feedback_count=len(packets)),
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
        record = write_loop_record(thread_id=state["thread_id"], state=stopped_state, outcome="stopped")
        return {
            "phase": "stopped",
            "stop_reason": stop_reason,
            "loop_records": [*state.get("loop_records", []), str(record)],
            "events": _event(state, "保存停止或负结果", record=str(record)),
        }

    def explain_result(state: ResearchLoopState) -> dict[str, Any]:
        try:
            decision, _ = main_agent.decide(
                question=state.get("user_request") or "请根据已验证证据总结结果并明确限制。",
                status="completed",
                config=_config(state),
                plan=_plan(state),
                data_profile=state.get("data_profile"),
                quality_report=state.get("quality_report"),
                summary=state.get("eda_summary"),
                evaluation=state.get("evaluation"),
                history=_messages(state),
                available_skills=skills.metadata(),
            )
            answer = decision.response.strip()
            if not answer:
                raise ResearchModelUnavailableError("模型没有返回结果解释。")
        except Exception as exc:  # noqa: BLE001 - model explanation boundary
            packet = exception_feedback(exc, source="evaluator", requires_user=True)
            return {"phase": "awaiting_user", "feedback_packets": _append_feedback(state, packet)}
        return {
            "phase": "awaiting_user",
            "assistant_message": answer,
            "messages": [
                *state.get("messages", []),
                ConversationMessage(role="assistant", content=answer).model_dump(mode="json"),
            ],
            "events": _event(state, "Agent 解释验证结果"),
        }

    def reply(state: ResearchLoopState) -> dict[str, Any]:
        answer = DialogueDecision.model_validate(state["decision"]).response.strip()
        if not answer:
            raise ResearchPlanValidationError("大模型没有返回可展示回复。")
        return {
            "phase": "awaiting_user",
            "assistant_message": answer,
            "messages": [
                *state.get("messages", []),
                ConversationMessage(role="assistant", content=answer).model_dump(mode="json"),
            ],
            "events": _event(state, "Agent 完成对话回复"),
        }

    def user_interrupt(state: ResearchLoopState) -> dict[str, Any]:
        payload = InterruptPayload(
            kind="need_user",
            phase="awaiting_user",
            message=state.get("stop_reason") or "自动循环需要用户决策。",
            choices=["accept_limitations", "modify", "stop"],
            plan=state.get("current_plan"),
            evaluation=state.get("evaluation"),
            result=state.get("latest_run"),
        )
        response = ResumePayload.model_validate(interrupt(payload.model_dump(mode="json")))
        if response.action == "accept_limitations":
            return {"phase": "evaluating", "user_request": "请总结当前最佳结果和限制。", "stop_reason": None}
        if response.action in {"modify", "followup", "next_round"}:
            message = response.message.strip()
            return {
                "phase": "understanding",
                "user_request": message,
                "stop_reason": None,
                "messages": [
                    *state.get("messages", []),
                    ConversationMessage(role="user", content=message).model_dump(mode="json"),
                ],
            }
        return {"phase": "stopped", "stop_reason": "用户停止研究循环。"}

    def result_interrupt(state: ResearchLoopState) -> dict[str, Any]:
        payload = InterruptPayload(
            kind="result",
            phase="awaiting_user",
            message=state.get("assistant_message") or "研究结果已就绪。",
            choices=["followup", "next_round", "stop"],
            plan=state.get("current_plan"),
            evaluation=state.get("evaluation"),
            result=state.get("latest_run"),
        )
        response = ResumePayload.model_validate(interrupt(payload.model_dump(mode="json")))
        if response.action in {"followup", "next_round"}:
            message = response.message.strip()
            return {
                "phase": "understanding",
                "user_request": message,
                "messages": [
                    *state.get("messages", []),
                    ConversationMessage(role="user", content=message).model_dump(mode="json"),
                ],
                "plan_origin": "initial" if response.action == "next_round" else state.get("plan_origin", "initial"),
            }
        return {"phase": "completed", "stop_reason": "用户结束当前研究。"}

    graph = StateGraph(ResearchLoopState)
    for name, node in {
        "ingest_user": ingest_user,
        "main_agent": understand,
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
        "prepare_plan_repair": prepare_plan_repair,
        "finalize_iteration": finalize_iteration,
        "prepare_evaluation_revision": prepare_evaluation_revision,
        "prepare_need_user": prepare_need_user,
        "explain_result": explain_result,
        "reply": reply,
        "user_interrupt": user_interrupt,
        "result_interrupt": result_interrupt,
        "persist_stop": persist_stop,
    }.items():
        graph.add_node(name, node)

    graph.add_edge(START, "ingest_user")
    graph.add_edge("ingest_user", "main_agent")
    graph.add_conditional_edges(
        "main_agent",
        route_main,
        {
            "new_plan": "resolve_skill",
            "revise_plan": "revise_user_plan",
            "execute_plan": "confirm_existing_plan",
            "reply": "reply",
            "need_user": "prepare_need_user",
        },
    )
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
        "approval_interrupt", route_approval, {"lock": "lock_plan", "modify": "main_agent", "stop": "persist_stop"}
    )
    graph.add_conditional_edges("lock_plan", route_lock, {"execute": "mark_tool_running", "need_user": "prepare_need_user"})
    graph.add_conditional_edges(
        "mark_tool_running",
        route_mark_tool,
        {"run": "execute_tool", "advance": "advance_tool", "evaluate": "finalize_iteration", "need_user": "prepare_need_user"},
    )
    graph.add_conditional_edges(
        "execute_tool",
        route_tool_execution,
        {
            "validate": "validate_tool_result",
            "retry": "mark_tool_running",
            "revise": "prepare_plan_repair",
            "need_user": "prepare_need_user",
            "stop": "persist_stop",
        },
    )
    graph.add_conditional_edges(
        "validate_tool_result", route_tool_execution, {"advance": "advance_tool", "revise": "prepare_plan_repair"}
    )
    graph.add_edge("advance_tool", "mark_tool_running")
    graph.add_conditional_edges(
        "prepare_plan_repair",
        lambda state: "need_user" if state.get("phase") == "awaiting_user" else "repair",
        {"repair": "eda_subagent", "need_user": "prepare_need_user"},
    )
    graph.add_conditional_edges(
        "finalize_iteration",
        route_evaluation,
        {"accept": "explain_result", "revise": "prepare_evaluation_revision", "need_user": "prepare_need_user", "reject": "persist_stop"},
    )
    graph.add_edge("prepare_evaluation_revision", "eda_subagent")
    graph.add_edge("prepare_need_user", "user_interrupt")
    graph.add_conditional_edges(
        "user_interrupt",
        lambda state: "stop" if state.get("phase") == "stopped" else ("explain" if state.get("phase") == "evaluating" else "continue"),
        {"stop": "persist_stop", "explain": "explain_result", "continue": "main_agent"},
    )
    graph.add_edge("reply", "result_interrupt")
    graph.add_edge("explain_result", "result_interrupt")
    graph.add_conditional_edges(
        "result_interrupt",
        lambda state: "end" if state.get("phase") == "completed" else "continue",
        {"end": END, "continue": "main_agent"},
    )
    graph.add_edge("persist_stop", END)
    return graph.compile(checkpointer=checkpointer)
