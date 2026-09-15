"""Main research Agent for dialogue routing, plan revision, and evidence explanation."""

from __future__ import annotations

import inspect
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.config import Settings, get_settings
from app.llm.budget import ModelRequestPurpose
from app.llm.factory import build_model_gateway
from app.llm.gateway import (
    ModelConfigurationError,
    ModelContextLimitError,
    ModelGateway,
    ModelGatewayError,
    ModelMessage,
    ModelOutputTruncatedError,
    ModelResponseError,
    ModelTransientError,
)
from app.research.agent.context import compact_episode_context
from app.research.agent.errors import (
    ResearchModelContextLimitError,
    ResearchModelOutputTruncatedError,
    ResearchModelSchemaError,
    ResearchModelTransientError,
    ResearchModelUnavailableError,
    ResearchPlanValidationError,
)
from app.research.agent.prompts import DIALOGUE_PROMPT_VERSION, DIALOGUE_SYSTEM_PROMPT
from app.research.agent.retrieval import select_conversation_context
from app.research.agent.schemas import ConversationMessage, EDAPlan, EDAResearchScope, EDAToolName
from app.research.planning.variables import (
    VariableSelectionMode,
    eligible_exogenous_variables,
    screening_function_set,
)
from app.research.reporting.capabilities import current_artifacts_context, report_capabilities_context
from app.research.schemas.results import DataQualityReport
from app.research.schemas.study import StudyConfig
from app.research.tools.catalog import FUNCTION_CATALOG, function_metadata
from app.research.tools.contracts import SegmentDefinition

DialogueIntent = Literal[
    "discussion",
    "new_plan",
    "revise_plan",
    "explain_result",
    "execute_plan",
    "new_news_analysis",
    "new_forecast_plan",
    "execute_forecast_plan",
]
FUNCTION_METADATA = function_metadata()


def requests_analysis_before_forecast(question: str) -> bool:
    """Return whether one turn explicitly asks for analysis and a forecast."""

    compact = "".join(question.lower().split())
    return any(word in compact for word in ("分析", "研究", "诊断")) and any(
        word in compact for word in ("预测", "预报")
    )


def requests_explicit_data_analysis(question: str) -> bool:
    """Conservatively recognize a request to calculate against selected data."""

    compact = "".join(question.casefold().split())
    discussion_cues = (
        "为什么",
        "是什么",
        "怎么做",
        "如何做",
        "怎么分析",
        "如何分析",
        "哪些方法",
        "什么方法",
        "分析思路",
        "研究思路",
        "给些建议",
        "通常",
        "介绍一下",
        "解释一下",
        "想研究",
    )
    explicit_cues = (
        "请分析",
        "帮我分析",
        "开始分析",
        "进行分析",
        "执行分析",
        "直接分析",
        "请计算",
        "帮我计算",
        "开始计算",
        "执行计算",
        "跑一下",
        "运行分析",
        "基于当前数据",
        "分析这批数据",
        "分析当前数据",
        "分析实际数据",
    )
    starts_with_action = compact.startswith(("分析", "计算", "统计", "检验", "诊断"))
    if any(cue in compact for cue in discussion_cues) and not any(
        cue in compact for cue in ("开始执行", "立即执行", "直接计算", "实际计算")
    ):
        return False
    return (
        starts_with_action
        or any(cue in compact for cue in explicit_cues)
        or ("继续按" in compact and "分析" in compact)
    )


def guard_dialogue_route(decision: DialogueDecision, question: str) -> DialogueDecision:
    """Fail closed when a model tries to start EDA from an ambiguous discussion turn."""

    if decision.intent != "new_plan" or requests_explicit_data_analysis(question):
        return decision
    return DialogueDecision(
        intent="discussion",
        response=(
            "我会先把这条消息当作电价问题讨论，不会仅因为已经选择了数据就启动计算。"
            "如果你希望运行当前数据，请明确说“请分析当前数据”，并补充想回答的问题。"
        ),
    )


class DialogueDecision(BaseModel):
    """Validated model decision for one user turn."""

    model_config = ConfigDict(extra="forbid")

    intent: DialogueIntent
    response: str = ""
    skill_name: str | None = None
    objective: str | None = None
    hypotheses: list[str] | None = None
    enabled_functions: list[EDAToolName] | None = None
    selected_variables: list[str] | None = None
    variable_selection_mode: VariableSelectionMode | None = None
    variable_recommendation_limit: int | None = Field(default=None, ge=1, le=32)
    max_lag: int | None = Field(default=None, ge=0)
    comparison_id: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_]{0,31}$")
    segments: list[SegmentDefinition] | None = None
    post_analysis_action: Literal["forecast"] | None = None

    @model_validator(mode="after")
    def validate_post_analysis_action(self) -> DialogueDecision:
        if self.post_analysis_action is not None and self.intent not in {
            "new_plan",
            "new_news_analysis",
        }:
            raise ValueError("post_analysis_action 只适用于分析后继续预测的路由")
        return self


def _compact_comparisons(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    compact: dict[str, Any] = {"price": {}, "relationships": {}}
    for comparison_id, comparison in (value.get("price") or {}).items():
        compact["price"][comparison_id] = {
            "segments": {
                segment_id: {
                    key: row.get(key)
                    for key in (
                        "label",
                        "selector",
                        "observations",
                        "coverage_rate",
                        "mean",
                        "median",
                        "std",
                        "q25",
                        "q75",
                    )
                }
                for segment_id, row in (comparison.get("segments") or {}).items()
            },
            "contrasts": comparison.get("contrasts", []),
        }
    for comparison_id, comparison in (value.get("relationships") or {}).items():
        compact["relationships"][comparison_id] = {
            "series": {
                variable: {
                    "segments": {
                        segment_id: {
                            key: row.get(key) for key in ("label", "selector", "observations", "correlation", "p_value")
                        }
                        for segment_id, row in (result.get("segments") or {}).items()
                    },
                    "contrasts": result.get("contrasts", []),
                }
                for variable, result in (comparison.get("series") or {}).items()
            }
        }
    return {key: item for key, item in compact.items() if item}


def _compact_autocorrelation(rows: Any) -> dict[str, Any]:
    if not isinstance(rows, list):
        return {}
    usable = [row for row in rows if isinstance(row, dict)]
    strongest = sorted(
        (row for row in usable if isinstance(row.get("correlation"), (int, float))),
        key=lambda row: abs(float(row["correlation"])),
        reverse=True,
    )[:8]
    return {
        "evaluated_lags": len(usable),
        "first_lags": usable[:8],
        "strongest_lags": strongest,
    }


def _compact_price_evidence(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    fields = (
        "methods",
        "observations",
        "missing_observations",
        "coverage_rate",
        "start_time",
        "end_time",
        "unit",
        "distribution",
        "signs",
        "extremes",
        "volatility",
        "rolling_statistics",
        "seasonality",
        "stationarity",
        "decomposition",
        "spike_regime",
        "naive_baselines",
        "variance_stabilization",
    )
    compact = {field: value[field] for field in fields if field in value}
    autocorrelation = _compact_autocorrelation(value.get("autocorrelation"))
    if autocorrelation:
        compact["autocorrelation"] = autocorrelation
    partial = value.get("partial_autocorrelation")
    if isinstance(partial, dict):
        compact["partial_autocorrelation"] = {key: item for key, item in partial.items() if key != "series"}
    duration = value.get("duration_curve")
    if isinstance(duration, dict):
        compact["duration_curve"] = {
            **{key: item for key, item in duration.items() if key != "points"},
            "plotted_point_count": len(duration.get("points") or []),
        }
    return compact


def _compact_relationship_series(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    fields = (
        "lag_semantics",
        "contemporaneous",
        "best_absolute_lag",
        "by_hour",
        "by_month",
        "feature_quantile_response",
    )
    return {field: value[field] for field in fields if field in value}


def _compact_section_series(value: Any, *, drop: frozenset[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    compact = {key: item for key, item in value.items() if key != "series"}
    series = value.get("series")
    if isinstance(series, dict):
        compact["series"] = {
            name: (
                {key: item for key, item in result.items() if key not in drop} if isinstance(result, dict) else result
            )
            for name, result in series.items()
        }
    return compact


def compact_evidence(
    summary: dict[str, Any] | None,
    evaluation: dict[str, Any] | None,
) -> dict[str, Any]:
    """Keep enough deterministic evidence for dialogue without sending full lag arrays."""

    source = summary or {}
    compact: dict[str, Any] = {"study": source.get("study", {})}
    price = source.get("price")
    if isinstance(price, dict):
        compact["price"] = _compact_price_evidence(price)
    exogenous = source.get("exogenous")
    if isinstance(exogenous, dict):
        compact["exogenous"] = {
            "methods": exogenous.get("methods", []),
            "series": exogenous.get("series", {}),
            "strong_collinearity_threshold": exogenous.get("strong_collinearity_threshold"),
            "strong_collinearity_pairs": exogenous.get("strong_collinearity_pairs", []),
            "multicollinearity": exogenous.get("multicollinearity", {}),
            "driver_stationarity": exogenous.get("driver_stationarity", {}),
        }
    relationships = source.get("relationships", {})
    relationship_series = relationships.get("series", {}) if isinstance(relationships, dict) else {}
    if isinstance(relationship_series, dict):
        relationship_evidence: dict[str, Any] = {
            field: relationships[field]
            for field in (
                "methods",
                "target",
                "pairwise_complete_analysis",
                "minimum_observations",
                "correlation_is_not_causation",
            )
            if field in relationships
        }
        relationship_evidence["series"] = {
            name: _compact_relationship_series(result)
            for name, result in relationship_series.items()
            if isinstance(result, dict)
        }
        for field, dropped in (
            ("mutual_information", frozenset({"profile"})),
            ("granger_precedence", frozenset({"tested_lags"})),
            ("rolling_stability", frozenset({"timeline"})),
        ):
            section = _compact_section_series(relationships.get(field), drop=dropped)
            if section:
                relationship_evidence[field] = section
        compact["relationships"] = relationship_evidence
    comparisons = _compact_comparisons(source.get("comparisons"))
    if comparisons:
        compact["comparisons"] = comparisons
    if evaluation:
        compact["evaluation"] = evaluation
    return compact


def _compact_dialogue_retry_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep the facts needed to route a turn after a truncated dialogue response."""

    current_plan = payload.get("current_plan")
    if isinstance(current_plan, dict):
        current_plan = {
            key: current_plan.get(key)
            for key in (
                "plan_id",
                "plan_kind",
                "question",
                "objective",
                "skill_name",
                "skill_version",
                "data_fingerprint",
                "selected_variables",
                "variable_selection_stage",
                "steps",
            )
            if key in current_plan
        }

    def short_messages(items: Any, limit: int) -> list[dict[str, Any]]:
        if not isinstance(items, list):
            return []
        compacted = []
        for item in items[-limit:]:
            if not isinstance(item, dict):
                continue
            compacted.append(
                {
                    key: (str(value)[-1200:] if key == "content" else value)
                    for key, value in item.items()
                    if key in {"role", "content", "turn_id", "episode_id"}
                }
            )
        return compacted

    def short_turns(items: Any, limit: int) -> list[dict[str, Any]]:
        if not isinstance(items, list):
            return []
        turns = []
        for item in items[-limit:]:
            if not isinstance(item, dict):
                continue
            turns.append(
                {
                    "turn_id": item.get("turn_id"),
                    "messages": short_messages(item.get("messages"), 4),
                    "relevance": item.get("relevance"),
                }
            )
        return turns

    skills = payload.get("available_skills")
    compact_skills = [
        {
            key: item.get(key)
            for key in (
                "name",
                "version",
                "domain",
                "allowed_functions",
                "research_protocol",
            )
            if key in item
        }
        for item in skills or []
        if isinstance(item, dict)
    ]
    capabilities = payload.get("output_capabilities")
    return {
        "prompt_version": payload.get("prompt_version"),
        "retry_reason": "上一响应达到输出长度限制；只返回一个简短且完整的结构化决策，response不超过600字。",
        "question": payload.get("question"),
        "session_status": payload.get("session_status"),
        "has_executable_data": payload.get("has_executable_data"),
        "interaction_context": payload.get("interaction_context"),
        "conversation_history": short_messages(payload.get("conversation_history"), 4),
        "earlier_related_turns": short_turns(payload.get("earlier_related_turns"), 2),
        "episode_memory": list(payload.get("episode_memory") or [])[-4:],
        "current_plan": current_plan,
        "study": payload.get("study"),
        "data_profile": payload.get("data_profile"),
        "quality_issues": list(payload.get("quality_issues") or [])[:8],
        "evidence": payload.get("evidence"),
        "available_variables": payload.get("available_variables"),
        "available_skills": compact_skills,
        "allowed_function_names": sorted((payload.get("allowed_functions") or {}).keys()),
        "intent_rules": payload.get("intent_rules"),
        "revision_contract": payload.get("revision_contract"),
        "skill_contract": payload.get("skill_contract"),
        "supported_workflows": (capabilities.get("supported_workflows") if isinstance(capabilities, dict) else None),
    }


class ModelResearchDialogue:
    """Call the required model for every research conversation decision."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        gateway: ModelGateway | None = None,
    ) -> None:
        self.gateway = gateway or build_model_gateway(settings)
        resolved = settings or get_settings()
        self.recent_turns = max(1, resolved.llm_history_messages // 2)
        self.retrieved_turns = resolved.llm_retrieved_turns

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
        scope: EDAResearchScope | None = None,
        data_profile: dict[str, Any] | None,
        quality_report: dict[str, Any] | None,
        summary: dict[str, Any] | None,
        evaluation: dict[str, Any] | None,
        history: list[ConversationMessage],
        available_skills: list[dict[str, Any]],
        episode_summaries: list[dict[str, Any]] | None = None,
        active_gate: str | None = None,
        episode_goal: str | None = None,
        latest_run: dict[str, Any] | None = None,
        current_turn_id: str | None = None,
        has_executable_data: bool | None = None,
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
        # The window answers "what was just said"; retrieval reaches the rest of
        # this session's state for anything else about the same question.
        recent_history, earlier_related_turns = select_conversation_context(
            history,
            question=question,
            current_turn_id=current_turn_id,
            recent_turns=self.recent_turns,
            retrieved_turns=self.retrieved_turns,
        )
        payload = {
            "prompt_version": DIALOGUE_PROMPT_VERSION,
            "output_capabilities": report_capabilities_context(),
            "interaction_context": {
                "active_gate": active_gate,
                "episode_goal": episode_goal,
                "current_run": current_artifacts_context(latest_run),
            },
            "question": question,
            "session_status": status,
            "has_executable_data": config is not None if has_executable_data is None else has_executable_data,
            "conversation_history": [item.model_dump(mode="json") for item in recent_history],
            "earlier_related_turns": [item.as_payload() for item in earlier_related_turns],
            "episode_memory": compact_episode_context(episode_summaries or []),
            "current_plan": plan.model_dump(mode="json") if plan is not None else None,
            "research_scope": scope.model_dump(mode="json") if scope is not None else None,
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
                    "display_name": FUNCTION_CATALOG[function_name].title,
                    "version": FUNCTION_CATALOG[function_name].version,
                }
                for function_name, metadata in FUNCTION_METADATA.items()
            },
            "intent_rules": {
                "discussion": "电价领域问答、问候、方法讨论、必要澄清、当前方案说明或无关问题引导",
                "new_plan": "用户明确要求计算实际数据时，根据数据和问题创建分析任务",
                "revise_plan": "按用户反馈生成当前方案的变更",
                "explain_result": "引用已有结构化证据解释结果",
                "execute_plan": "用户明确确认运行当前方案",
                "new_news_analysis": "用户明确要求分析与电价相关的新闻或政策事件",
                "new_forecast_plan": "用户只要求山东次日省级实时电价预测；参数由本地固定",
                "execute_forecast_plan": "用户明确确认运行已冻结的预测方案",
            },
            "revision_contract": {
                "unchanged_fields": "null",
                "mandatory_data_quality": "managed_by_compiler",
                "function_selection": "non_null_enabled_functions_is_complete_replacement",
                "agenda_selection": "a_replaced_function_set_rebuilds_the_agenda_from_retained_functions",
                "variable_selection": "explicit_all_eligible_or_auto_recommend",
                "revision_basis": "user_feedback",
            },
            "skill_contract": {
                "new_plan_skill_name": "choose_from_available_skills",
                "uploaded_data_without_execution_request": "discussion",
                "domain": "electricity_price_only",
            },
        }

        def invoke(value: dict[str, Any]) -> DialogueDecision:
            messages = [
                ModelMessage(role="system", content=DIALOGUE_SYSTEM_PROMPT),
                ModelMessage(role="user", content=json.dumps(value, ensure_ascii=False)),
            ]
            kwargs: dict[str, Any] = {"messages": messages, "schema": DialogueDecision}
            if "purpose" in inspect.signature(self.gateway.invoke_structured).parameters:
                kwargs["purpose"] = (
                    ModelRequestPurpose.RESULT_EXPLANATION
                    if summary is not None or evaluation is not None
                    else ModelRequestPurpose.DIALOGUE
                )
            return self.gateway.invoke_structured(**kwargs)

        try:
            try:
                return guard_dialogue_route(invoke(payload), question)
            except (ModelOutputTruncatedError, ModelContextLimitError):
                return guard_dialogue_route(invoke(_compact_dialogue_retry_payload(payload)), question)
        except ModelOutputTruncatedError as exc:
            raise ResearchModelOutputTruncatedError(f"大模型对话输出达到长度限制：{exc}") from exc
        except ModelContextLimitError as exc:
            raise ResearchModelContextLimitError(f"大模型对话请求超过上下文限制：{exc}") from exc
        except ModelTransientError as exc:
            raise ResearchModelTransientError(f"大模型对话暂时不可用：{exc}") from exc
        except ModelResponseError as exc:
            raise ResearchModelSchemaError(f"大模型返回的对话决策无法解析：{exc}") from exc
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

    @staticmethod
    def _revised_agenda(
        plan: EDAPlan,
        revised: EDAPlan,
        decision: DialogueDecision,
        previous_evaluation: dict[str, Any] | None = None,
    ) -> list[str]:
        """Keep only agenda items that still belong to the revised executable function set."""

        from app.research.planning.compiler import FUNCTION_AGENDA_HYPOTHESES, FUNCTION_AGENDA_ITEM_IDS

        enabled_functions = {step.function for step in revised.enabled_steps}
        enabled_item_ids = {
            FUNCTION_AGENDA_ITEM_IDS[name] for name in enabled_functions if name in FUNCTION_AGENDA_ITEM_IDS
        }
        canonical_by_item_id = {
            FUNCTION_AGENDA_ITEM_IDS[name]: hypothesis
            for name, hypothesis in FUNCTION_AGENDA_HYPOTHESES.items()
            if name in enabled_functions and name in FUNCTION_AGENDA_ITEM_IDS
        }

        represented_item_ids: set[str] = set()
        if decision.hypotheses is not None:
            kept = list(decision.hypotheses)
        elif decision.enabled_functions is None:
            kept = list(plan.hypotheses)
        else:
            kept = []
            assessments = (previous_evaluation or {}).get("hypothesis_assessments", [])
            for assessment in assessments:
                item_id = str(assessment.get("item_id") or "")
                hypothesis = str(assessment.get("hypothesis") or "").strip()
                if hypothesis and item_id in enabled_item_ids and item_id not in represented_item_ids:
                    kept.append(hypothesis)
                    represented_item_ids.add(item_id)

        generated = set(FUNCTION_AGENDA_HYPOTHESES.values())
        kept = [item for item in kept if item not in generated or item in canonical_by_item_id.values()]
        for item_id, hypothesis in canonical_by_item_id.items():
            if item_id not in represented_item_ids and hypothesis not in kept:
                kept.append(hypothesis)
        return list(dict.fromkeys(kept))

    def revise_plan(
        self,
        *,
        question: str,
        plan: EDAPlan,
        config: StudyConfig,
        decision: DialogueDecision,
        previous_evaluation: dict[str, Any] | None = None,
        quality_report: DataQualityReport | None = None,
    ) -> tuple[EDAPlan, str]:
        changed_fields = (
            decision.objective,
            decision.hypotheses,
            decision.enabled_functions,
            decision.selected_variables,
            decision.variable_selection_mode,
            decision.variable_recommendation_limit,
            decision.max_lag,
            decision.comparison_id,
            decision.segments,
        )
        if all(value is None for value in changed_fields):
            raise ResearchPlanValidationError("大模型将回合标记为方案修订，但没有返回任何修改内容。")

        current_enabled = {step.function for step in plan.enabled_steps if step.function != "data_quality"}
        if decision.enabled_functions is None:
            enabled_functions = current_enabled
        else:
            enabled_functions = set(decision.enabled_functions).difference({"data_quality"})
            unknown_functions = sorted(enabled_functions.difference(FUNCTION_CATALOG))
            if unknown_functions:
                raise ResearchPlanValidationError(f"大模型修订包含未知研究函数：{', '.join(unknown_functions)}")
            if plan.research_protocol_function_order:
                outside_protocol = sorted(enabled_functions.difference(plan.research_protocol_function_order))
                if outside_protocol:
                    raise ResearchPlanValidationError(
                        f"大模型修订包含领域协议未授权函数：{', '.join(outside_protocol)}"
                    )

        valid_names = {spec.name for spec in config.exogenous}
        selection_mode = decision.variable_selection_mode or plan.variable_selection_mode
        if decision.selected_variables is not None and decision.variable_selection_mode is None:
            selection_mode = "explicit"
        selected_variables = (
            list(plan.selected_variables)
            if decision.selected_variables is None
            else list(dict.fromkeys(decision.selected_variables))
        )
        unknown_variables = sorted(set(selected_variables).difference(valid_names))
        if unknown_variables:
            raise ResearchPlanValidationError(f"大模型修订包含未知变量：{', '.join(unknown_variables)}")

        requires_variables = any(FUNCTION_CATALOG[name].uses_variables for name in enabled_functions)
        if requires_variables and not selected_variables:
            if selection_mode == "explicit":
                # When the user asks the Agent to decide, an omitted variable
                # list is an explicit request for evidence-backed screening,
                # not a malformed executable plan.
                selection_mode = "auto_recommend"
            selected_variables = eligible_exogenous_variables(config, quality_report)
            if not selected_variables:
                raise ResearchPlanValidationError("没有外生变量通过自动筛查所需的最低覆盖率与样本门槛")

        selection_stage = plan.variable_selection_stage
        deferred_functions = list(plan.deferred_functions)
        if plan.variable_selection_stage == "screening" and decision.selected_variables:
            selection_mode = "explicit"
            selection_stage = "recommended"
            if decision.enabled_functions is None:
                enabled_functions = enabled_functions.union(plan.deferred_functions)
            deferred_functions = []
        elif selection_mode == "all_eligible" and requires_variables:
            selected_variables = eligible_exogenous_variables(config, quality_report)
            if not selected_variables:
                raise ResearchPlanValidationError("没有外生变量满足全部合格变量模式的最低数据门槛")
            selection_stage = "direct"
            deferred_functions = []
        elif selection_mode == "auto_recommend" and requires_variables:
            selected_variables = eligible_exogenous_variables(config, quality_report)
            if not selected_variables:
                raise ResearchPlanValidationError("没有外生变量通过自动推荐所需的最低数据门槛")
            allowed_functions = set(plan.research_protocol_function_order or FUNCTION_CATALOG)
            enabled_functions, deferred_functions = screening_function_set(
                enabled_functions,
                allowed_functions=allowed_functions,
            )
            selection_stage = "screening"

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

        enabled_step_ids = {step.step_id for step in plan.steps if step.function in enabled_functions}
        revised = plan.adjusted(
            enabled_step_ids=enabled_step_ids,
            selected_variables=selected_variables,
            max_lag=max_lag,
            reason=f"用户反馈经大模型分析后形成修订：{question.strip()}",
            source="user_dialogue",
        ).model_copy(
            update={
                "objective": decision.objective or plan.objective,
                "planner": "llm",
                "planning_model": getattr(self.model_dialogue, "model_name", plan.planning_model),
                "variable_selection_mode": selection_mode,
                "variable_selection_stage": selection_stage,
                "deferred_functions": deferred_functions,
                "variable_recommendation_limit": (
                    decision.variable_recommendation_limit or plan.variable_recommendation_limit
                ),
            }
        )
        segment_steps = [step for step in revised.steps if FUNCTION_CATALOG[step.function].uses_segments]
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
        if any(FUNCTION_CATALOG[name].uses_segments for name in enabled_functions) and (
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
                        if FUNCTION_CATALOG[step.function].uses_segments
                        else step
                        for step in revised.steps
                    ]
                }
            )
        existing_functions = {step.function for step in revised.steps}
        missing_functions = [name for name in enabled_functions if name not in existing_functions]
        if missing_functions:
            from app.research.planning.compiler import compile_function_step
            from app.research.planning.contracts import DraftStep

            extra_steps = list(revised.steps)
            for function_name in missing_functions:
                parameters: dict[str, Any] = {}
                spec = FUNCTION_CATALOG[function_name]
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
                            function=function_name,
                            enabled=True,
                            rationale=spec.description,
                            parameters=parameters,
                        ),
                        selected_variables=selected_variables,
                        config=config,
                    )
                )
            revised = revised.model_copy(update={"steps": extra_steps})
        agenda_decision = decision.model_copy(update={"hypotheses": []}) if selection_stage == "screening" else decision
        revised = revised.model_copy(
            update={
                "hypotheses": self._revised_agenda(
                    plan,
                    revised,
                    agenda_decision,
                    previous_evaluation,
                ),
                # Replacing either the function set or the user's agenda starts
                # a new agenda.  Parked statements from the superseded agenda
                # must not keep the revised run at the old human gate.
                "unverifiable_hypotheses": (
                    []
                    if decision.enabled_functions is not None or decision.hypotheses is not None
                    else plan.unverifiable_hypotheses
                ),
            }
        )
        revised = revised.ordered_by_research_protocol()
        enabled_titles = "、".join(step.title for step in revised.enabled_steps)
        variable_text = "、".join(revised.selected_variables) if revised.selected_variables else "无"
        response = decision.response.strip() or (
            f"我已根据你的反馈生成方案 v{revised.revision}：{enabled_titles}；"
            f"所选变量为 {variable_text}；最大滞后为 {max_lag} 个间隔。"
        )
        return revised, response
