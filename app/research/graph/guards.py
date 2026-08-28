"""Deterministic authorization, repetition, budget, and feedback guards."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from app.research.agent.errors import (
    DataFingerprintMismatchError,
    DuplicateResearchFunctionError,
    InsufficientDataError,
    PlanCompatibilityError,
    RepairablePlanError,
    SkillVersionMismatchError,
)
from app.research.agent.schemas import EDAPlan
from app.research.data.loader import ResearchDataError
from app.research.graph.contracts import AuthorizationEnvelope, LoopBudget
from app.research.schemas.feedback import FeedbackPacket, FeedbackSource
from app.research.tools.catalog import FUNCTION_CATALOG


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
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


def evidence_fingerprint(tool_results: list[dict[str, Any]]) -> str:
    """Hash deterministic tool evidence without plan, wording, or feedback metadata."""

    normalized = sorted(
        (
            {
                "work_id": item.get("call", {}).get("work_id"),
                "data_fingerprint": item.get("data_fingerprint"),
                "output_hash": item.get("output_hash"),
            }
            for item in tool_results
        ),
        key=lambda item: (str(item["work_id"]), str(item["output_hash"])),
    )
    return canonical_hash(normalized)


def authorization_envelope(
    plan: EDAPlan,
    *,
    approved_at: str,
) -> AuthorizationEnvelope:
    if plan.data_fingerprint is None:
        raise PlanCompatibilityError("生成审批边界前必须存在数据指纹")
    return AuthorizationEnvelope(
        approved_plan_id=plan.plan_id,
        approved_plan_fingerprint=plan_fingerprint(plan),
        approved_at=approved_at,
        question_hash=canonical_hash(plan.question.strip()),
        max_steps=len(plan.enabled_steps),
        skill_name=plan.skill_name,
        skill_version=plan.skill_version,
        data_fingerprint=plan.data_fingerprint,
        functions=[step.function for step in plan.enabled_steps],
        variables=list(plan.selected_variables),
        max_lag_by_function={
            step.function: int(step.parameters["max_lag"])
            for step in plan.enabled_steps
            if step.parameters.get("max_lag") is not None
        },
        segment_parameters_by_function={
            step.function: {
                "comparison_id": step.parameters.get("comparison_id"),
                "segments": step.parameters.get("segments"),
            }
            for step in plan.enabled_steps
            if FUNCTION_CATALOG[step.function].uses_segments
        },
        approved_parameters_by_function={
            step.function: dict(step.parameters)
            for step in plan.enabled_steps
        },
    )


def validate_automatic_revision(plan: EDAPlan, envelope: AuthorizationEnvelope) -> FeedbackPacket | None:
    violations: list[str] = []
    if not envelope.approved_plan_id or not envelope.approved_plan_fingerprint or not envelope.approved_at:
        violations.append("原始审批身份不完整")
    if plan.skill_name != envelope.skill_name or plan.skill_version != envelope.skill_version:
        violations.append("Skill 或 Skill 版本发生变化")
    if canonical_hash(plan.question.strip()) != envelope.question_hash:
        violations.append("研究问题发生变化")
    if plan.data_fingerprint != envelope.data_fingerprint:
        violations.append("数据指纹发生变化")
    enabled = {step.function for step in plan.enabled_steps}
    if not enabled.issubset(envelope.functions):
        violations.append("新增了未审批研究函数")
    if len(plan.enabled_steps) > envelope.max_steps:
        violations.append("启用步骤数量超过原审批范围")
    if not set(plan.selected_variables).issubset(envelope.variables):
        violations.append("新增了未审批变量")
    for step in plan.enabled_steps:
        approved_parameters = envelope.approved_parameters_by_function.get(step.function)
        if approved_parameters is None:
            violations.append(f"{step.function} 缺少原审批参数")
            continue
        if step.parameters.get("max_lag") is not None and int(step.parameters["max_lag"]) > envelope.max_lag_by_function.get(
            step.function, -1
        ):
            violations.append(f"{step.function} 扩大了最大滞后范围")
        if FUNCTION_CATALOG[step.function].uses_segments:
            current = {
                "comparison_id": step.parameters.get("comparison_id"),
                "segments": step.parameters.get("segments"),
            }
            if current != envelope.segment_parameters_by_function.get(step.function):
                violations.append(f"{step.function} 改变了未重新审批的分段定义")
        for key, value in step.parameters.items():
            if key in {"variables", "max_lag", "segments", "comparison_id"}:
                continue
            if value != approved_parameters.get(key):
                violations.append(f"{step.function} 改变了未重新审批的参数 {key}")
    if not violations:
        return None
    unique_violations = list(dict.fromkeys(violations))
    return FeedbackPacket(
        source="budget_guard",
        code="automatic_revision_exceeds_approval",
        severity="error",
        message="；".join(unique_violations),
        observed={
            "plan_id": plan.plan_id,
            "revision": plan.revision,
            "violations": unique_violations,
            "functions": [step.function for step in plan.enabled_steps],
            "variables": list(plan.selected_variables),
        },
        expected={
            "approved_plan_id": envelope.approved_plan_id,
            "max_steps": envelope.max_steps,
            "functions": envelope.functions,
            "variables": envelope.variables,
        },
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
        recommendation="请用户说明下一步研究要求，或结束本轮研究。",
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
    recommendation = "根据结构化错误修订方案或输入。"
    if isinstance(exc, DuplicateResearchFunctionError):
        recommendation = exc.recommendation
    elif isinstance(exc, DataFingerprintMismatchError):
        recommendation = "输入内容已变化，请基于新数据重新生成并审批方案。"
    elif isinstance(exc, InsufficientDataError):
        recommendation = "请补充目标数据、扩大有效研究窗口或降低配置中的最低覆盖门槛后重新开始。"
    elif isinstance(exc, SkillVersionMismatchError):
        recommendation = "当前对话基于旧版研究协议；请新建对话重新分析，旧报告仍可查看。"
    elif isinstance(exc, PlanCompatibilityError):
        recommendation = "请使用当前 Skill 与函数版本重新生成并审批方案。"
    elif isinstance(exc, RepairablePlanError):
        recommendation = "请在原 Skill 权限内修正函数参数或变量选择。"
    elif isinstance(exc, ResearchDataError):
        recommendation = "请检查研究文件、字段、时间轴和数据可读性后重新开始。"
    return FeedbackPacket(
        source=source,
        code=f"{source}_{type(exc).__name__}",
        severity="error",
        message=str(exc) or type(exc).__name__,
        step_id=step_id,
        observed=type(exc).__name__,
        recommendation=recommendation,
        retryable=retryable,
        requires_user=requires_user,
    )
