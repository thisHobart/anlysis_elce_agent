"""EDA Subagent: model-led planning compiled into deterministic executable steps."""

from __future__ import annotations

import inspect
import json
from typing import Any

from pydantic import ValidationError

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
from app.research.agent.errors import (
    ResearchModelContextLimitError,
    ResearchModelOutputTruncatedError,
    ResearchModelSchemaError,
    ResearchModelTransientError,
    ResearchModelUnavailableError,
    ResearchPlanValidationError,
)
from app.research.agent.prompts import (
    PLANNING_MINIMAL_RECOVERY_SYSTEM_PROMPT,
    PLANNING_PROMPT_VERSION,
    PLANNING_SYSTEM_PROMPT,
)
from app.research.agent.retrieval import select_conversation_context
from app.research.agent.schemas import EDAPlan
from app.research.planning.compiler import EDAPlanCompiler, max_lag_limit
from app.research.planning.contracts import (
    EDAPlanDraft,
    EDAPlanIntent,
    MinimalEDAPlanIntent,
)
from app.research.planning.variables import eligible_exogenous_variables, screening_function_set
from app.research.reporting.capabilities import report_capabilities_context
from app.research.schemas.feedback import FeedbackPacket
from app.research.schemas.results import DataQualityReport
from app.research.schemas.study import StudyConfig
from app.research.skills.contracts import SkillDefinition
from app.research.tools.catalog import FUNCTION_CATALOG, function_metadata, optional_function_names
from app.research.tools.eda.functions import build_eda_tool_registry
from app.research.tools.registry import ToolRegistry

FUNCTION_METADATA = function_metadata()
OPTIONAL_FUNCTIONS = optional_function_names()

_DATA_QUALITY_ONLY_CUES = (
    "数据质量",
    "质量检查",
    "质量核验",
    "数据可用",
    "能不能用",
    "是否可用",
    "缺失率",
    "覆盖率",
    "重复时间戳",
    "时间戳对齐",
)
_ANALYSIS_CUES = (
    "关系",
    "相关",
    "滞后",
    "领先",
    "分布",
    "周期",
    "季节",
    "平稳",
    "趋势",
    "波动",
    "尖峰",
    "极端",
    "异常值",
    "共线",
    "因果",
    "granger",
    "pearson",
    "spearman",
    "互信息",
    "基线",
    "可预测",
)


def _accepts_keyword(callback: Any, name: str) -> bool:
    """Preserve compatibility with deterministic test/custom planners."""

    try:
        parameters = inspect.signature(callback).parameters.values()
    except (TypeError, ValueError):
        return False
    # Require an explicit opt-in.  A wrapper may expose **kwargs and forward
    # them to an older planner that does not understand the new argument.
    return any(parameter.name == name for parameter in parameters)


def _allows_data_quality_only_plan(question: str) -> bool:
    """Recognize the narrow case where the compiler-managed quality step is sufficient.

    The default is deliberately conservative. A mixed request mentioning both data
    readiness and an analytical objective must still select at least one research
    function; otherwise an empty compact intent could silently turn a substantive
    request into a quality-only plan.
    """

    normalized = question.casefold()
    return any(cue in normalized for cue in _DATA_QUALITY_ONLY_CUES) and not any(
        cue in normalized for cue in _ANALYSIS_CUES
    )


def _invoke_structured_for_purpose(
    gateway: ModelGateway,
    *,
    messages: list[ModelMessage],
    schema: type[EDAPlanIntent | MinimalEDAPlanIntent],
    purpose: ModelRequestPurpose,
) -> EDAPlanIntent | MinimalEDAPlanIntent:
    invoke = gateway.invoke_structured
    kwargs: dict[str, Any] = {"messages": messages, "schema": schema}
    if _accepts_keyword(invoke, "purpose"):
        kwargs["purpose"] = purpose
    return invoke(**kwargs)


def _compact_function_cards(allowed_functions: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "description": FUNCTION_METADATA[name][1],
            "uses_variables": FUNCTION_CATALOG[name].uses_variables,
            "minimum_variables": FUNCTION_CATALOG[name].min_variables,
            "accepts_max_lag": FUNCTION_CATALOG[name].uses_max_lag,
            "accepts_segments": FUNCTION_CATALOG[name].uses_segments,
            "planning_guidance": FUNCTION_CATALOG[name].planning_guidance,
        }
        for name in allowed_functions
    ]


def _intent_to_draft(
    intent: EDAPlanIntent | MinimalEDAPlanIntent,
    *,
    question: str,
    config: StudyConfig,
    quality: DataQualityReport,
    allowed_functions: list[str],
    variable_ids: dict[str, str],
) -> EDAPlanDraft:
    mode = intent.variable_selection_mode
    unknown_ids = sorted(set(intent.selected_variable_ids).difference(variable_ids))
    if unknown_ids:
        raise ResearchPlanValidationError(f"大模型选择了未知变量 ID：{', '.join(unknown_ids)}")
    if mode == "explicit":
        selected_variables = [variable_ids[item] for item in intent.selected_variable_ids]
    else:
        if intent.selected_variable_ids:
            raise ResearchPlanValidationError(f"{mode} 模式不得返回显式变量 ID")
        selected_variables = eligible_exogenous_variables(config, quality)
        if not selected_variables:
            raise ResearchPlanValidationError("没有外生变量满足自动筛查的最低数据门槛")

    selected_names = [item.function for item in intent.functions]
    disallowed = sorted(set(selected_names).difference(allowed_functions))
    if disallowed:
        raise ResearchPlanValidationError(f"大模型选择了未授权研究函数：{', '.join(disallowed)}")
    if not selected_names and not _allows_data_quality_only_plan(question):
        raise ResearchPlanValidationError("研究问题至少需要选择一个可执行研究函数")

    deferred_functions: list[str] = []
    selection_stage = "direct"
    retained_names = set(selected_names)
    if mode == "auto_recommend":
        retained_names, deferred_functions = screening_function_set(
            retained_names,
            allowed_functions=set(allowed_functions),
        )
        selection_stage = "screening"
    selections = {item.function: item for item in intent.functions}
    steps = []
    for function_name in allowed_functions:
        if function_name not in retained_names:
            continue
        selection = selections.get(function_name)
        parameters: dict[str, Any] = {}
        spec = FUNCTION_CATALOG[function_name]
        if spec.uses_variables:
            parameters["variables"] = selected_variables
        if selection is not None and selection.max_lag is not None:
            parameters["max_lag"] = selection.max_lag
        if selection is not None and selection.comparison_id is not None:
            parameters["comparison_id"] = selection.comparison_id
        if selection is not None and selection.segments is not None:
            parameters["segments"] = [item.model_dump(mode="json") for item in selection.segments]
        steps.append(
            {
                "function": function_name,
                "enabled": True,
                "rationale": spec.description,
                "parameters": parameters,
            }
        )
    full_intent = intent if isinstance(intent, EDAPlanIntent) else None
    return EDAPlanDraft(
        objective=(full_intent.objective if full_intent is not None else question.strip()),
        hypotheses=(full_intent.hypotheses if full_intent is not None and mode != "auto_recommend" else []),
        selected_variables=selected_variables,
        variable_selection_mode=mode,
        variable_selection_stage=selection_stage,
        deferred_functions=deferred_functions,
        variable_recommendation_limit=intent.variable_recommendation_limit,
        steps=steps,
        assumptions=full_intent.assumptions if full_intent is not None else [],
    )


class ModelEDAPlanner:
    """Call the required OpenAI-compatible model for a structured research proposal."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        gateway: ModelGateway | None = None,
        tools: ToolRegistry | None = None,
    ) -> None:
        self.gateway = gateway or build_model_gateway(settings)
        self.tools = tools or build_eda_tool_registry()
        resolved = settings or get_settings()
        self.recent_turns = max(1, resolved.llm_history_messages // 2)
        self.retrieved_turns = resolved.llm_retrieved_turns

    @property
    def enabled(self) -> bool:
        """Compatibility name; model planning is available only when endpoint and model are configured."""

        return self.gateway.enabled

    @property
    def model_name(self) -> str:
        return self.gateway.model_name

    def propose(
        self,
        question: str,
        config: StudyConfig,
        quality: DataQualityReport,
        history: list[dict[str, Any]] | None = None,
        skill: SkillDefinition | None = None,
        feedback: list[FeedbackPacket] | None = None,
        revision_context: dict[str, Any] | None = None,
        episode_memory: list[dict[str, Any]] | None = None,
        data_fingerprint: str | None = None,
    ) -> EDAPlanDraft:
        if not self.enabled:
            raise ResearchModelUnavailableError("大模型尚未配置，无法生成研究方案。")
        variables = [
            {
                "id": f"v{index}",
                "name": spec.name,
                "availability_type": spec.availability_type,
                "coverage_rate": quality.series[spec.name].aligned_coverage_rate,
                "unit": spec.unit or "unknown",
            }
            for index, spec in enumerate(config.exogenous, start=1)
        ]
        variable_ids = {item["id"]: item["name"] for item in variables}
        allowed_functions = [
            name
            for name in (skill.allowed_functions if skill is not None else OPTIONAL_FUNCTIONS)
            if name != "data_quality"
        ]
        recent_history, earlier_related_turns = select_conversation_context(
            history or [],
            question=question,
            recent_turns=self.recent_turns,
            retrieved_turns=self.retrieved_turns,
        )
        payload = {
            "prompt_version": PLANNING_PROMPT_VERSION,
            "data_fingerprint": data_fingerprint,
            "output_capability_ids": sorted(report_capabilities_context()),
            "question": question,
            "conversation_history": recent_history,
            "earlier_related_turns": [turn.as_payload() for turn in earlier_related_turns],
            # Episode memory has its own budget and scope policy.  Keeping it
            # outside conversation_history prevents a long chat from trimming
            # the durable evidence before the planner sees it.
            "episode_memory": episode_memory or [],
            "active_skill": skill.prompt_context() if skill is not None else None,
            "validation_feedback": [item.model_dump(mode="json") for item in (feedback or [])],
            "revision_context": revision_context,
            "study": {
                "name": config.study.name,
                "market": config.study.market,
                "frequency": config.study.frequency,
                "timezone": config.study.timezone,
                "target": config.target.name,
                "target_unit": config.target.unit or "unknown",
            },
            "variables": variables,
            "quality_issues": [issue.model_dump(mode="json") for issue in quality.issues],
            "allowed_functions": _compact_function_cards(allowed_functions),
            "constraints": {
                "default_max_lag": config.analysis.max_lag,
                "max_lag_limit": max_lag_limit(config.study.frequency),
                "minimum_observations": config.analysis.min_relationship_observations,
                "data_quality_step": "compiler_adds_required_step",
                "executable_functions": "provided_function_names_only",
                "variable_source": "variables_exact_names",
                "target_field": "separate_from_exogenous_variables",
                "trusted_arguments": "compiler_injects_thresholds_and_minimum_observations",
                "function_cardinality": "each_function_at_most_once_per_plan",
                "research_agenda": "returned_inside_the_single_plan_intent",
                "minimum_plan": "empty_function_list_only_for_explicit_data_quality_only_requests",
            },
        }
        messages = [
            ModelMessage(role="system", content=PLANNING_SYSTEM_PROMPT),
            ModelMessage(role="user", content=json.dumps(payload, ensure_ascii=False)),
        ]
        try:
            try:
                intent = _invoke_structured_for_purpose(
                    self.gateway,
                    messages=messages,
                    schema=EDAPlanIntent,
                    purpose=ModelRequestPurpose.EDA_PLANNING,
                )
            except (ModelOutputTruncatedError, ModelContextLimitError):
                recovery_payload = {
                    "prompt_version": PLANNING_PROMPT_VERSION,
                    "data_fingerprint": data_fingerprint,
                    "question": question,
                    "study": payload["study"],
                    "variables": variables,
                    "active_skill": payload["active_skill"],
                    "allowed_functions": [
                        {
                            "name": item["name"],
                            "uses_variables": item["uses_variables"],
                            "accepts_max_lag": item["accepts_max_lag"],
                            "accepts_segments": item["accepts_segments"],
                        }
                        for item in payload["allowed_functions"]
                    ],
                    "constraints": payload["constraints"],
                    "revision_context": revision_context,
                }
                recovery_messages = [
                    ModelMessage(role="system", content=PLANNING_MINIMAL_RECOVERY_SYSTEM_PROMPT),
                    ModelMessage(role="user", content=json.dumps(recovery_payload, ensure_ascii=False)),
                ]
                intent = _invoke_structured_for_purpose(
                    self.gateway,
                    messages=recovery_messages,
                    schema=MinimalEDAPlanIntent,
                    purpose=ModelRequestPurpose.EDA_PLANNING_RECOVERY,
                )
            return _intent_to_draft(
                intent,
                question=question,
                config=config,
                quality=quality,
                allowed_functions=allowed_functions,
                variable_ids=variable_ids,
            )
        except KeyError as exc:
            raise ResearchPlanValidationError(f"大模型调用了未注册研究函数：{exc.args[0]}") from exc
        except ModelOutputTruncatedError as exc:
            raise ResearchModelOutputTruncatedError(
                f"大模型规划输出达到长度限制，紧凑恢复仍未完成：{exc}。请在大模型配置中补全或核对模型画像。"
            ) from exc
        except ModelContextLimitError as exc:
            raise ResearchModelContextLimitError(
                f"大模型规划请求在紧凑恢复后仍超过上下文限制：{exc}。请在大模型配置中补全或核对模型画像。"
            ) from exc
        except ModelTransientError as exc:
            raise ResearchModelTransientError(f"大模型规划调用暂时失败：{exc}") from exc
        except ModelResponseError as exc:
            raise ResearchModelSchemaError(f"大模型返回的研究方案无法解析：{exc}") from exc
        except (ModelConfigurationError, ModelGatewayError) as exc:
            raise ResearchModelUnavailableError(f"大模型规划调用失败：{exc}") from exc

    def propose_with_context(
        self,
        question: str,
        config: StudyConfig,
        quality: DataQualityReport,
        *,
        history: list[dict[str, Any]] | None = None,
        skill: SkillDefinition | None = None,
        feedback: list[FeedbackPacket] | None = None,
        revision_context: dict[str, Any],
        episode_memory: list[dict[str, Any]] | None = None,
        data_fingerprint: str | None = None,
    ) -> EDAPlanDraft:
        """Generate a constrained draft while exposing the approved revision baseline."""

        return self.propose(
            question,
            config,
            quality,
            history=history,
            skill=skill,
            feedback=feedback,
            revision_context=revision_context,
            episode_memory=episode_memory,
            data_fingerprint=data_fingerprint,
        )


class EDASubagent:
    """Compile the required model proposal into a versioned deterministic plan."""

    def __init__(self, *, model_planner: Any | None = None, compiler: EDAPlanCompiler | None = None) -> None:
        self.model_planner = model_planner or ModelEDAPlanner()
        self.compiler = compiler or EDAPlanCompiler()

    def propose(
        self,
        question: str,
        config: StudyConfig,
        quality: DataQualityReport,
        history: list[dict[str, Any]] | None = None,
        skill: SkillDefinition | None = None,
        feedback: list[FeedbackPacket] | None = None,
        revision_context: dict[str, Any] | None = None,
        episode_memory: list[dict[str, Any]] | None = None,
        data_fingerprint: str | None = None,
    ) -> EDAPlan:
        normalized = question.strip()
        if not normalized:
            raise ValueError("research question must not be empty")
        if not self.model_planner.enabled:
            raise ResearchModelUnavailableError("大模型尚未配置，无法生成研究方案。")
        if skill is None:
            raise ResearchPlanValidationError("EDA Subagent 缺少已激活的 Skill。")
        planner_kwargs: dict[str, Any] = {"history": history, "skill": skill}
        if feedback is not None:
            planner_kwargs["feedback"] = feedback
        if revision_context is not None:
            contextual_propose = getattr(self.model_planner, "propose_with_context", None)
            if callable(contextual_propose):
                if data_fingerprint and _accepts_keyword(contextual_propose, "data_fingerprint"):
                    planner_kwargs["data_fingerprint"] = data_fingerprint
                if episode_memory and _accepts_keyword(contextual_propose, "episode_memory"):
                    planner_kwargs["episode_memory"] = episode_memory
                draft_value = contextual_propose(
                    normalized,
                    config,
                    quality,
                    **planner_kwargs,
                    revision_context=revision_context,
                )
            else:
                if data_fingerprint and _accepts_keyword(self.model_planner.propose, "data_fingerprint"):
                    planner_kwargs["data_fingerprint"] = data_fingerprint
                if episode_memory and _accepts_keyword(self.model_planner.propose, "episode_memory"):
                    planner_kwargs["episode_memory"] = episode_memory
                draft_value = self.model_planner.propose(normalized, config, quality, **planner_kwargs)
        else:
            if data_fingerprint and _accepts_keyword(self.model_planner.propose, "data_fingerprint"):
                planner_kwargs["data_fingerprint"] = data_fingerprint
            if episode_memory and _accepts_keyword(self.model_planner.propose, "episode_memory"):
                planner_kwargs["episode_memory"] = episode_memory
            draft_value = self.model_planner.propose(normalized, config, quality, **planner_kwargs)
        try:
            draft = draft_value if isinstance(draft_value, EDAPlanDraft) else EDAPlanDraft.model_validate(draft_value)
        except ValidationError as exc:
            raise ResearchPlanValidationError(f"大模型研究方案不符合结构化契约：{exc}") from exc
        return self.compiler.compile(
            draft,
            question=normalized,
            config=config,
            skill=skill,
            model_name=getattr(self.model_planner, "model_name", None),
            prompt_version=PLANNING_PROMPT_VERSION,
        )
