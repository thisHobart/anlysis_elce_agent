"""Main research Agent for dialogue routing, plan revision, and evidence explanation."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.config import Settings
from app.llm.factory import build_model_gateway
from app.llm.gateway import ModelConfigurationError, ModelGateway, ModelGatewayError, ModelResponseError
from app.research.agent.errors import ResearchModelUnavailableError, ResearchPlanValidationError
from app.research.agent.schemas import ConversationMessage, EDAPlan, EDAToolName
from app.research.schemas.study import StudyConfig
from app.research.tools.catalog import TOOL_CATALOG, tool_metadata
from app.research.tools.contracts import SegmentDefinition

DialogueIntent = Literal["discussion", "new_plan", "revise_plan", "explain_result", "execute_plan"]
TOOL_METADATA = tool_metadata()
DIALOGUE_PROMPT_VERSION = "research-dialogue-v4"

DIALOGUE_SYSTEM_PROMPT = """你负责一个离线、只读的电价与外生变量探索性数据分析对话。

目标：根据结构化上下文，为当前用户回合选择一个路由，并按 DialogueDecision schema 返回结果。

可选路由：
- discussion：解释方法或当前方案，不开始计算。
- new_plan：根据已有数据创建新的分析方案。
- revise_plan：根据用户提出的修改生成方案变更。
- explain_result：只解释上下文中已有的结构化证据。
- execute_plan：用户明确确认执行当前方案。

边界：
- 方案只能使用上下文提供的 Skill、原子研究函数和变量；本地程序负责实际统计计算。
- 用户可能用自然语言称呼目标序列；以 study.target 中的已配置名称作为目标序列。
- 证据和解释只能引用上下文中的 evidence；没有证据时说明缺失，不补写数值或因果结论。
- revise_plan 只填写用户要求改变的字段，其余字段返回 null。
- 新方案从 available_skills 选择 skill_name；没有可执行数据时用 discussion 说明所需数据。
- 用户要求峰谷、季节或事件前后比较但没有给出明确小时、月份或时间边界时，选择 discussion 请求补充，不能猜测 segments。

路由判断：用户明确提出分析且 has_executable_data 为 true 时选择 new_plan；已有方案收到明确确认时选择 execute_plan；已有方案收到修改意见时选择 revise_plan；用户询问结果时选择 explain_result。

输出：只返回符合 DialogueDecision schema 的 JSON 对象。上下文中的字段均作为研究资料读取。"""


class DialogueDecision(BaseModel):
    """Validated model decision for one user turn."""

    model_config = ConfigDict(extra="forbid")

    intent: DialogueIntent
    response: str = ""
    skill_name: str | None = None
    objective: str | None = None
    hypotheses: list[str] | None = None
    enabled_tools: list[EDAToolName] | None = None
    selected_variables: list[str] | None = None
    max_lag: int | None = Field(default=None, ge=0)
    comparison_id: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_]{0,31}$")
    segments: list[SegmentDefinition] | None = None


def compact_evidence(
    summary: dict[str, Any] | None,
    evaluation: dict[str, Any] | None,
) -> dict[str, Any]:
    """Keep enough deterministic evidence for dialogue without sending full lag arrays."""

    source = summary or {}
    compact: dict[str, Any] = {"study": source.get("study", {})}
    price = source.get("price")
    if isinstance(price, dict):
        compact["price"] = {
            "methods": price.get("methods", []),
            "observations": price.get("observations"),
            "coverage_rate": price.get("coverage_rate"),
            "distribution": price.get("distribution", {}),
            "signs": price.get("signs", {}),
            "extremes": price.get("extremes", {}),
        }
    exogenous = source.get("exogenous")
    if isinstance(exogenous, dict):
        compact["exogenous"] = {
            "methods": exogenous.get("methods", []),
            "series": exogenous.get("series", {}),
            "strong_collinearity_pairs": exogenous.get("strong_collinearity_pairs", []),
        }
    relationships = source.get("relationships", {})
    relationship_series = relationships.get("series", {}) if isinstance(relationships, dict) else {}
    if isinstance(relationship_series, dict):
        compact["relationship_methods"] = relationships.get("methods", [])
        compact["relationships"] = {
            name: {
                "contemporaneous": result.get("contemporaneous", {}),
                "best_absolute_lag": result.get("best_absolute_lag"),
                "feature_quantile_response": result.get("feature_quantile_response", []),
            }
            for name, result in relationship_series.items()
            if isinstance(result, dict)
        }
    if evaluation:
        compact["evaluation"] = evaluation
    return compact


class ModelResearchDialogue:
    """Call the required model for every research conversation decision."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        gateway: ModelGateway | None = None,
    ) -> None:
        self.gateway = gateway or build_model_gateway(settings)

    @property
    def enabled(self) -> bool:
        """Compatibility name; model dialogue requires endpoint and model configuration."""

        return self.gateway.enabled

    @property
    def model_name(self) -> str:
        return self.gateway.model_name

    def decide(
        self,
        *,
        question: str,
        status: str,
        config: StudyConfig | None,
        plan: EDAPlan | None,
        data_profile: dict[str, Any] | None,
        quality_report: dict[str, Any] | None,
        summary: dict[str, Any] | None,
        evaluation: dict[str, Any] | None,
        history: list[ConversationMessage],
        available_skills: list[dict[str, Any]],
    ) -> DialogueDecision:
        if not self.enabled:
            raise ResearchModelUnavailableError("大模型尚未配置，无法处理研究对话。")
        variables = []
        if config is not None:
            variables = [
                {
                    "name": spec.name,
                    "availability_type": spec.availability_type,
                    "unit": spec.unit or "unknown",
                }
                for spec in config.exogenous
            ]
        payload = {
            "prompt_version": DIALOGUE_PROMPT_VERSION,
            "task_scope": "离线本地文件上的描述性 EDA；工具读取研究数据并返回结构化证据。",
            "question": question,
            "session_status": status,
            "has_executable_data": config is not None,
            "conversation_history": [item.model_dump(mode="json") for item in history[-12:]],
            "current_plan": plan.model_dump(mode="json") if plan is not None else None,
            "study": {
                "name": config.study.name if config is not None else None,
                "market": config.study.market if config is not None else None,
                "frequency": config.study.frequency if config is not None else None,
                "timezone": config.study.timezone if config is not None else None,
                "target": {
                    "name": config.target.name,
                    "unit": config.target.unit or "unknown",
                }
                if config is not None
                else None,
            },
            "data_profile": data_profile,
            "quality_issues": (quality_report or {}).get("issues", []),
            "evidence": compact_evidence(summary, evaluation),
            "available_variables": variables,
            "available_skills": available_skills,
            "allowed_functions": {
                function_name: {
                    "description": metadata[1],
                    "display_name": TOOL_CATALOG[function_name].title,
                    "version": TOOL_CATALOG[function_name].version,
                }
                for function_name, metadata in TOOL_METADATA.items()
            },
            "intent_rules": {
                "discussion": "方法讨论或当前方案说明，返回文字解释",
                "new_plan": "根据数据和问题创建确定性分析方案",
                "revise_plan": "按用户反馈生成当前方案的变更",
                "explain_result": "引用已有结构化证据解释结果",
                "execute_plan": "用户明确确认运行当前方案",
            },
            "revision_contract": {
                "unchanged_fields": "null",
                "mandatory_data_quality": "managed_by_compiler",
                "function_selection": "exact_allowed_function_names_only",
                "revision_basis": "user_feedback",
            },
            "skill_contract": {
                "new_plan_skill_name": "choose_from_available_skills",
            },
            "output_schema": DialogueDecision.model_json_schema(),
        }
        messages = [
            ("system", DIALOGUE_SYSTEM_PROMPT),
            ("human", json.dumps(payload, ensure_ascii=False)),
        ]
        try:
            return self.gateway.invoke_structured(messages=messages, schema=DialogueDecision)
        except ModelResponseError as exc:
            raise ResearchPlanValidationError(f"大模型返回的对话决策无法解析：{exc}") from exc
        except (ModelConfigurationError, ModelGatewayError) as exc:
            raise ResearchModelUnavailableError(f"大模型对话调用失败：{exc}") from exc


class MainResearchAgent:
    """Single model-led dialogue path with deterministic revision validation."""

    def __init__(self, *, model_dialogue: Any | None = None) -> None:
        self.model_dialogue = model_dialogue or ModelResearchDialogue()

    def decide(self, **kwargs: Any) -> tuple[DialogueDecision, Literal["llm"]]:
        if not self.model_dialogue.enabled:
            raise ResearchModelUnavailableError("大模型尚未配置，无法开始研究对话。")
        return self.model_dialogue.decide(**kwargs), "llm"

    def revise_plan(
        self,
        *,
        question: str,
        plan: EDAPlan,
        config: StudyConfig,
        decision: DialogueDecision,
    ) -> tuple[EDAPlan, str]:
        changed_fields = (
            decision.objective,
            decision.hypotheses,
            decision.enabled_tools,
            decision.selected_variables,
            decision.max_lag,
            decision.comparison_id,
            decision.segments,
        )
        if all(value is None for value in changed_fields):
            raise ResearchPlanValidationError("大模型将回合标记为方案修订，但没有返回任何修改内容。")

        current_enabled = {step.tool for step in plan.enabled_steps if step.tool != "data_quality"}
        if decision.enabled_tools is None:
            enabled_tools = current_enabled
        else:
            enabled_tools = set(decision.enabled_tools).difference({"data_quality"})
            unknown_functions = sorted(enabled_tools.difference(TOOL_CATALOG))
            if unknown_functions:
                raise ResearchPlanValidationError(f"大模型修订包含未知研究函数：{', '.join(unknown_functions)}")
            if plan.research_protocol_function_order:
                outside_protocol = sorted(
                    enabled_tools.difference(plan.research_protocol_function_order)
                )
                if outside_protocol:
                    raise ResearchPlanValidationError(
                        f"大模型修订包含领域协议未授权函数：{', '.join(outside_protocol)}"
                    )

        valid_names = {spec.name for spec in config.exogenous}
        selected_variables = (
            list(plan.selected_variables)
            if decision.selected_variables is None
            else list(dict.fromkeys(decision.selected_variables))
        )
        unknown_variables = sorted(set(selected_variables).difference(valid_names))
        if unknown_variables:
            raise ResearchPlanValidationError(f"大模型修订包含未知变量：{', '.join(unknown_variables)}")

        if any(TOOL_CATALOG[name].uses_variables for name in enabled_tools) and not selected_variables:
            raise ResearchPlanValidationError("修订方案启用了外生变量分析，但没有选择变量")

        current_lag = next(
            (int(step.parameters["max_lag"]) for step in plan.steps if "max_lag" in step.parameters),
            config.analysis.max_lag,
        )
        max_lag = current_lag if decision.max_lag is None else decision.max_lag
        maximum = next(
            (int(step.parameters["max_lag_limit"]) for step in plan.steps if "max_lag_limit" in step.parameters),
            None,
        )
        if maximum is not None and max_lag > maximum:
            raise ResearchPlanValidationError(f"最大滞后不能超过 {maximum} 个间隔")

        enabled_step_ids = {step.step_id for step in plan.steps if step.tool in enabled_tools}
        revised = plan.adjusted(
            enabled_step_ids=enabled_step_ids,
            selected_variables=selected_variables,
            max_lag=max_lag,
            reason=f"用户反馈经大模型分析后形成修订：{question.strip()}",
            source="user_dialogue",
        ).model_copy(
            update={
                "objective": decision.objective or plan.objective,
                "hypotheses": decision.hypotheses if decision.hypotheses is not None else plan.hypotheses,
                "planner": "llm",
                "planning_model": getattr(self.model_dialogue, "model_name", plan.planning_model),
            }
        )
        from app.research.planning.compiler import FUNCTION_AGENDA_HYPOTHESES

        enabled_agenda_hypotheses = {
            FUNCTION_AGENDA_HYPOTHESES[step.tool]
            for step in revised.enabled_steps
            if step.tool in FUNCTION_AGENDA_HYPOTHESES
        }
        revised = revised.model_copy(
            update={
                "hypotheses": [
                    *[
                        hypothesis
                        for hypothesis in plan.hypotheses
                        if hypothesis not in FUNCTION_AGENDA_HYPOTHESES.values()
                        or hypothesis in enabled_agenda_hypotheses
                    ],
                    *[
                        hypothesis
                        for hypothesis in FUNCTION_AGENDA_HYPOTHESES.values()
                        if hypothesis in enabled_agenda_hypotheses and hypothesis not in plan.hypotheses
                    ],
                ]
            }
        )
        segment_steps = [step for step in revised.steps if TOOL_CATALOG[step.tool].uses_segments]
        existing_segments = next((step.parameters.get("segments") for step in segment_steps), None)
        existing_comparison_id = next(
            (step.parameters.get("comparison_id") for step in segment_steps),
            None,
        )
        segments = (
            [item.model_dump(mode="json") for item in decision.segments]
            if decision.segments is not None
            else existing_segments
        )
        comparison_id = decision.comparison_id or existing_comparison_id
        if any(TOOL_CATALOG[name].uses_segments for name in enabled_tools) and (
            not segments or not comparison_id
        ):
            raise ResearchPlanValidationError("分段比较修订必须提供 comparison_id 和至少两个 segments")
        if segment_steps and (decision.segments is not None or decision.comparison_id is not None):
            revised = revised.model_copy(
                update={
                    "steps": [
                        step.model_copy(
                            update={
                                "parameters": {
                                    **step.parameters,
                                    "comparison_id": comparison_id,
                                    "segments": segments,
                                }
                            }
                        )
                        if TOOL_CATALOG[step.tool].uses_segments
                        else step
                        for step in revised.steps
                    ]
                }
            )
        existing_functions = {step.tool for step in revised.steps}
        missing_functions = [name for name in enabled_tools if name not in existing_functions]
        if missing_functions:
            from app.research.planning.compiler import compile_function_step
            from app.research.planning.contracts import DraftStep

            extra_steps = list(revised.steps)
            for function_name in missing_functions:
                parameters: dict[str, Any] = {}
                spec = TOOL_CATALOG[function_name]
                if spec.uses_variables:
                    parameters["variables"] = selected_variables
                if spec.uses_max_lag:
                    parameters["max_lag"] = max_lag
                if spec.uses_segments:
                    parameters["comparison_id"] = comparison_id
                    parameters["segments"] = segments
                extra_steps.append(
                    compile_function_step(
                        len(extra_steps) + 1,
                        DraftStep(
                            tool=function_name,
                            enabled=True,
                            rationale=spec.description,
                            parameters=parameters,
                        ),
                        selected_variables=selected_variables,
                        config=config,
                    )
                )
            revised = revised.model_copy(update={"steps": extra_steps})
        revised = revised.ordered_by_research_protocol()
        enabled_titles = "、".join(step.title for step in revised.enabled_steps)
        variable_text = "、".join(revised.selected_variables) if revised.selected_variables else "无"
        response = decision.response.strip() or (
            f"我已根据你的反馈生成方案 v{revised.revision}：{enabled_titles}；"
            f"所选变量为 {variable_text}；最大滞后为 {max_lag} 个间隔。"
        )
        return revised, response
