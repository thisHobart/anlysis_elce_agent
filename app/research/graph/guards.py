"""Deterministic authorization, repetition, budget, and feedback guards."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from app.research.agent.schemas import EDAPlan
from app.research.graph.contracts import AuthorizationEnvelope, LoopBudget
from app.research.schemas.feedback import FeedbackPacket, FeedbackSource


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def plan_fingerprint(plan: EDAPlan) -> str:
    payload = plan.model_dump(
        mode="json",
        exclude={
            "plan_id",
            "parent_plan_id",
            "revision",
            "revision_reason",
            "revision_source",
            "created_at",
            "planning_notes",
        },
    )
    return canonical_hash(payload)


def evidence_fingerprint(summary: dict[str, Any], evaluation: dict[str, Any]) -> str:
    return canonical_hash({"summary": summary, "evaluation": evaluation})


def authorization_envelope(plan: EDAPlan) -> AuthorizationEnvelope:
    return AuthorizationEnvelope(
        skill_name=plan.skill_name,
        skill_version=plan.skill_version,
        data_fingerprint=plan.data_fingerprint or "",
        tools=[step.tool for step in plan.enabled_steps],
        variables=list(plan.selected_variables),
        methods={
            step.tool: list(step.parameters.get("methods", []))
            for step in plan.enabled_steps
            if step.parameters.get("methods") is not None
        },
        max_lag_by_tool={
            step.tool: int(step.parameters["max_lag"])
            for step in plan.enabled_steps
            if step.parameters.get("max_lag") is not None
        },
    )


def validate_automatic_revision(plan: EDAPlan, envelope: AuthorizationEnvelope) -> FeedbackPacket | None:
    violations: list[str] = []
    if plan.skill_name != envelope.skill_name or plan.skill_version != envelope.skill_version:
        violations.append("Skill 或 Skill 版本发生变化")
    if plan.data_fingerprint != envelope.data_fingerprint:
        violations.append("数据指纹发生变化")
    enabled = {step.tool for step in plan.enabled_steps}
    if not enabled.issubset(envelope.tools):
        violations.append("新增了未审批工具")
    if not set(plan.selected_variables).issubset(envelope.variables):
        violations.append("新增了未审批变量")
    for step in plan.enabled_steps:
        methods = set(step.parameters.get("methods", []))
        if not methods.issubset(envelope.methods.get(step.tool, [])):
            violations.append(f"{step.tool} 新增了未审批方法")
        if step.parameters.get("max_lag") is not None and int(step.parameters["max_lag"]) > envelope.max_lag_by_tool.get(
            step.tool, -1
        ):
            violations.append(f"{step.tool} 扩大了最大滞后范围")
    if not violations:
        return None
    return FeedbackPacket(
        source="budget_guard",
        code="automatic_revision_exceeds_approval",
        severity="error",
        message="；".join(dict.fromkeys(violations)),
        observed=plan.model_dump(mode="json"),
        expected=envelope.model_dump(mode="json"),
        recommendation="请用户明确审批扩大后的研究范围。",
        retryable=False,
        requires_user=True,
    )


def budget_feedback(budget: LoopBudget, *, code: str, message: str) -> FeedbackPacket:
    return FeedbackPacket(
        source="budget_guard",
        code=code,
        severity="warning",
        message=message,
        observed=budget.model_dump(mode="json"),
        recommendation="请用户接受当前限制、修改输入或停止。",
        retryable=False,
        requires_user=True,
    )


def exception_feedback(
    exc: Exception,
    *,
    source: FeedbackSource,
    step_id: str | None = None,
    retryable: bool = False,
    requires_user: bool = False,
) -> FeedbackPacket:
    return FeedbackPacket(
        source=source,
        code=f"{source}_{type(exc).__name__}",
        severity="error",
        message=str(exc) or type(exc).__name__,
        step_id=step_id,
        observed=type(exc).__name__,
        recommendation="根据结构化错误修订方案或输入。",
        retryable=retryable,
        requires_user=requires_user,
    )
