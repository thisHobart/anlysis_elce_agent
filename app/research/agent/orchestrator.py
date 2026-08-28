"""Main research Agent for dialogue routing, plan revision, and evidence explanation."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.config import Settings, get_settings
from app.llm.factory import build_model_gateway
from app.llm.gateway import (
    ModelConfigurationError,
    ModelGateway,
    ModelGatewayError,
    ModelMessage,
    ModelResponseError,
)
from app.research.agent.context import bounded_recent_history, compact_episode_context
from app.research.agent.errors import ResearchModelUnavailableError, ResearchPlanValidationError
from app.research.agent.prompts import DIALOGUE_PROMPT_VERSION, DIALOGUE_SYSTEM_PROMPT
from app.research.agent.schemas import ConversationMessage, EDAPlan, EDAToolName
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

DialogueIntent = Literal["discussion", "new_plan", "revise_plan", "explain_result", "execute_plan"]
FUNCTION_METADATA = function_metadata()


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
                            key: row.get(key)
                            for key in ("label", "selector", "observations", "correlation", "p_value")
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
        compact["partial_autocorrelation"] = {
            key: item for key, item in partial.items() if key != "series"
        }
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
            name: ({key: item for key, item in result.items() if key not in drop} if isinstance(result, dict) else result)
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


class ModelResearchDialogue:
    """Call the required model for every research conversation decision."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        gateway: ModelGateway | None = None,
    ) -> None:
        self.gateway = gateway or build_model_gateway(settings)
        self.history_messages = (settings or get_settings()).llm_history_messages

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
        episode_summaries: list[dict[str, Any]] | None = None,
        active_gate: str | None = None,
        episode_goal: str | None = None,
        latest_run: dict[str, Any] | None = None,
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
            "output_capabilities": report_capabilities_context(),
            "interaction_context": {
                "active_gate": active_gate,
                "episode_goal": episode_goal,
                "current_run": current_artifacts_context(latest_run),
            },
            "question": question,
            "session_status": status,
            "has_executable_data": config is not None,
            "conversation_history": [
                item.model_dump(mode="json")
                for item in bounded_recent_history(history, max_messages=self.history_messages)
            ],
            "episode_memory": compact_episode_context(episode_summaries or []),
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
                    "display_name": FUNCTION_CATALOG[function_name].title,
                    "version": FUNCTION_CATALOG[function_name].version,
                }
                for function_name, metadata in FUNCTION_METADATA.items()
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
                "function_selection": "non_null_enabled_functions_is_complete_replacement",
                "agenda_selection": "a_replaced_function_set_rebuilds_the_agenda_from_retained_functions",
                "variable_selection": "explicit_all_eligible_or_auto_recommend",
                "revision_basis": "user_feedback",
            },
            "skill_contract": {
                "new_plan_skill_name": "choose_from_available_skills",
            },
        }
        messages = [
            ModelMessage(role="system", content=DIALOGUE_SYSTEM_PROMPT),
            ModelMessage(role="user", content=json.dumps(payload, ensure_ascii=False)),
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
            FUNCTION_AGENDA_ITEM_IDS[name]
            for name in enabled_functions
            if name in FUNCTION_AGENDA_ITEM_IDS
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
        kept = [
            item
            for item in kept
            if item not in generated or item in canonical_by_item_id.values()
        ]
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
                outside_protocol = sorted(
                    enabled_functions.difference(plan.research_protocol_function_order)
                )
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
        agenda_decision = (
            decision.model_copy(update={"hypotheses": []})
            if selection_stage == "screening"
            else decision
        )
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
