"""Compile model function-call drafts into deterministic versioned plans."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pandas as pd

from app.research.agent.errors import ResearchPlanValidationError
from app.research.agent.schemas import EDAPlan, EDAPlanStep
from app.research.planning.contracts import DraftStep, EDAPlanDraft
from app.research.schemas.study import StudyConfig
from app.research.skills.contracts import SkillDefinition
from app.research.tools.catalog import TOOL_CATALOG, method_keys, selected_implementation_versions

OPTIONAL_TOOLS = ("price_profile", "exogenous_profile", "relationship_analysis")


def _intervals_per_hour(frequency: str) -> float:
    offset = pd.tseries.frequencies.to_offset(frequency)
    seconds = offset.nanos / 1_000_000_000
    return 3600 / seconds


def max_lag_limit(frequency: str, *, days: int = 31) -> int:
    """Convert a duration safety limit to canonical intervals."""

    return max(1, round(days * 24 * _intervals_per_hour(frequency)))


def _parameter_int(parameters: dict[str, Any], key: str, default: int) -> int:
    value = parameters.get(key, default)
    if isinstance(value, bool):
        raise ResearchPlanValidationError(f"{key} 必须是整数")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ResearchPlanValidationError(f"{key} 必须是整数") from exc


def _compile_step(
    number: int,
    tool: str,
    draft: DraftStep | None,
    *,
    selected_variables: list[str],
    config: StudyConfig,
) -> EDAPlanStep:
    catalog = TOOL_CATALOG[tool]
    enabled = bool(draft and draft.enabled)
    rationale = draft.rationale if draft is not None else "大模型未选择本步骤。"
    parameters: dict[str, Any] = {}
    versions: dict[str, str] = {}
    if catalog.methods:
        requested = draft.parameters.get("methods") if draft is not None else []
        if not isinstance(requested, list) or any(not isinstance(method, str) for method in requested):
            raise ResearchPlanValidationError(f"{tool}.methods 必须是方法名称列表")
        requested = list(dict.fromkeys(requested))
        unknown = sorted(set(requested).difference(method_keys(tool)))
        if unknown:
            raise ResearchPlanValidationError(f"{tool} 包含未注册方法：{', '.join(unknown)}")
        if enabled and not requested:
            raise ResearchPlanValidationError(f"大模型启用了 {tool}，但没有选择具体分析方法")
        parameters["methods"] = requested
        versions = selected_implementation_versions(tool, requested)

    if tool in {"exogenous_profile", "relationship_analysis"}:
        parameters["variables"] = selected_variables
        if enabled and not selected_variables:
            raise ResearchPlanValidationError(f"大模型启用了 {tool}，但没有选择可用外生变量")
    if tool in {"price_profile", "relationship_analysis"}:
        maximum = max_lag_limit(config.study.frequency)
        max_lag = _parameter_int(draft.parameters if draft else {}, "max_lag", config.analysis.max_lag)
        if not 0 <= max_lag <= maximum:
            raise ResearchPlanValidationError(f"{tool}.max_lag 必须在 0 到 {maximum} 之间")
        parameters.update({"max_lag": max_lag, "max_lag_limit": maximum})
    if tool == "price_profile":
        parameters["spike_iqr_multiplier"] = config.analysis.spike_iqr_multiplier
    elif tool == "exogenous_profile":
        parameters["outlier_iqr_multiplier"] = config.analysis.outlier_iqr_multiplier
    elif tool == "relationship_analysis":
        parameters["min_observations"] = config.analysis.min_relationship_observations

    return EDAPlanStep(
        step_id=f"S{number}",
        tool=tool,  # type: ignore[arg-type]
        title=catalog.title,
        description=catalog.description,
        rationale=rationale,
        enabled=enabled,
        parameters=parameters,
        tool_version=catalog.version,
        method_versions=versions,
    )


class EDAPlanCompiler:
    """Enforce Skill, tool, method, variable, parameter, and version constraints."""

    def compile(
        self,
        draft: EDAPlanDraft,
        *,
        question: str,
        config: StudyConfig,
        skill: SkillDefinition,
        model_name: str | None,
        prompt_version: str,
    ) -> EDAPlan:
        valid_names = {spec.name for spec in config.exogenous}
        unknown_variables = sorted(set(draft.selected_variables).difference(valid_names))
        if unknown_variables:
            raise ResearchPlanValidationError(f"大模型选择了未知变量：{', '.join(unknown_variables)}")
        selected = list(dict.fromkeys(draft.selected_variables))
        tools = [step.tool for step in draft.steps]
        disallowed_tools = sorted(set(tools).difference(skill.allowed_tools))
        if disallowed_tools:
            raise ResearchPlanValidationError(f"Skill {skill.name} 未授权工具：{', '.join(disallowed_tools)}")
        if len(tools) != len(set(tools)):
            raise ResearchPlanValidationError("大模型方案包含重复工具步骤")
        by_tool = {step.tool: step for step in draft.steps}

        quality_catalog = TOOL_CATALOG["data_quality"]
        steps = [
            EDAPlanStep(
                step_id="S1",
                tool="data_quality",
                title=quality_catalog.title,
                description=quality_catalog.description,
                rationale="任何研究结论都必须先通过确定性数据质量和时间对齐门禁。",
                enabled=True,
                required=True,
                tool_version=quality_catalog.version,
            )
        ]
        steps.extend(
            _compile_step(number, tool, by_tool.get(tool), selected_variables=selected, config=config)
            for number, tool in enumerate(OPTIONAL_TOOLS, start=2)
        )

        assumptions = list(
            dict.fromkeys(
                [
                    *draft.assumptions,
                    "所有统计结论均为描述性证据，不解释为因果关系。",
                    "关系分析使用成对有效样本，不进行隐式插补。",
                ]
            )
        )
        notes = [
            (
                f"方案由大模型 {model_name or 'configured-model'} 生成，"
                "并由确定性计划编译器完成 Skill、工具、方法版本、变量和参数校验。"
            )
        ]
        if config.target.unit.casefold() in {"", "unknown", "unspecified"}:
            notes.append("目标电价单位未知，绝对数值和阈值解释前需要用户确认单位。")
        if "unspecified" in config.study.market.casefold():
            notes.append("市场范围尚未明确，当前不生成依赖具体市场规则的解释。")
        return EDAPlan(
            plan_id=uuid4().hex[:12],
            question=question,
            objective=draft.objective,
            study_name=config.study.name,
            planner="llm",
            planning_model=model_name,
            planning_prompt_version=prompt_version,
            skill_name=skill.name,
            skill_version=skill.version,
            hypotheses=draft.hypotheses,
            selected_variables=selected,
            steps=steps,
            assumptions=assumptions,
            planning_notes=notes,
        )
