"""EDA Subagent: model-led planning compiled into deterministic executable steps."""

from __future__ import annotations

import inspect
import json
import re
from typing import Any

from pydantic import ValidationError

from app.config import Settings, get_settings
from app.llm.factory import build_model_gateway
from app.llm.gateway import (
    ModelConfigurationError,
    ModelGateway,
    ModelGatewayError,
    ModelMessage,
    ModelResponseError,
    ModelToolCall,
)
from app.research.agent.context import bounded_recent_history
from app.research.agent.errors import ResearchModelUnavailableError, ResearchPlanValidationError
from app.research.agent.prompts import PLANNING_PROMPT_VERSION, PLANNING_SYSTEM_PROMPT
from app.research.agent.schemas import EDAPlan
from app.research.planning.compiler import EDAPlanCompiler, max_lag_limit
from app.research.planning.contracts import EDAPlanDraft
from app.research.planning.variables import eligible_exogenous_variables, screening_function_set
from app.research.reporting.capabilities import report_capabilities_context
from app.research.schemas.feedback import FeedbackPacket
from app.research.schemas.results import DataQualityReport
from app.research.schemas.study import StudyConfig
from app.research.skills.contracts import SkillDefinition
from app.research.tools.catalog import FUNCTION_CATALOG, function_metadata, optional_function_names
from app.research.tools.eda.functions import build_eda_tool_registry
from app.research.tools.registry import ToolRegistry

AGENDA_FUNCTION_NAME = "declare_research_agenda"
"""Planning-protocol call that carries the agenda; never registered, never executed."""

AGENDA_FUNCTION_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": AGENDA_FUNCTION_NAME,
        "description": (
            "声明本轮研究议程。必须调用一次，与分析函数在同一次回复中一起返回。"
            "这个调用不执行任何计算，只记录本轮要判定什么、要验证哪些假设。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "objective": {
                    "type": "string",
                    "description": "本轮研究要判定的问题，一句话，不要复述用户原话。",
                },
                "hypotheses": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "本轮要验证的假设。每条都必须能被同一次回复中选择的分析函数验证；"
                        "无法验证的猜想不要写进来。"
                    ),
                },
                "assumptions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "本轮依赖的前提，例如时区、市场产品或变量可获得性。",
                },
                "variable_selection_mode": {
                    "type": "string",
                    "enum": ["explicit", "all_eligible", "auto_recommend"],
                    "description": (
                        "用户明确给变量用 explicit；要求全部合格变量用 all_eligible；"
                        "不知道选什么或要求系统推荐用 auto_recommend。"
                    ),
                },
                "variable_recommendation_limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 32,
                    "description": "auto_recommend 最多推荐多少个变量，默认 8。",
                },
            },
            "required": ["objective"],
        },
    },
}
FUNCTION_METADATA = function_metadata()
OPTIONAL_FUNCTIONS = optional_function_names()


def _accepts_keyword(callback: Any, name: str) -> bool:
    """Preserve compatibility with deterministic test/custom planners."""

    try:
        parameters = inspect.signature(callback).parameters.values()
    except (TypeError, ValueError):
        return False
    # Require an explicit opt-in.  A wrapper may expose **kwargs and forward
    # them to an older planner that does not understand the new argument.
    return any(parameter.name == name for parameter in parameters)

def _agenda_text(agenda: ModelToolCall | None, field: str) -> str:
    """Read one string field from the agenda call, tolerating a model that skipped it."""

    if agenda is None:
        return ""
    value = agenda.arguments.get(field)
    return value.strip() if isinstance(value, str) else ""


def _agenda_list(agenda: ModelToolCall | None, field: str) -> list[str]:
    """Read one string list from the agenda call, dropping blanks and duplicates."""

    if agenda is None:
        return []
    values = agenda.arguments.get(field)
    if not isinstance(values, list):
        return []
    return list(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))


def _validate_agenda_function_coverage(
    agenda: ModelToolCall | None,
    analysis_calls: list[ModelToolCall],
) -> None:
    """Reject an agenda that claims coverage from functions it did not call."""

    hypotheses = _agenda_list(agenda, "hypotheses")
    if not hypotheses:
        return
    selected = {call.name for call in analysis_calls}
    text = "\n".join(hypotheses)
    referenced = {
        name
        for name in FUNCTION_CATALOG
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", text)
    }
    missing = sorted(referenced.difference(selected))
    if missing:
        raise ResearchPlanValidationError(f"议程引用了未选择的研究函数：{', '.join(missing)}")
    if not analysis_calls:
        raise ResearchPlanValidationError("研究议程包含待判定假设，但没有选择任何研究函数")


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
        self.history_messages = (settings or get_settings()).llm_history_messages

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
        history: list[dict[str, str]] | None = None,
        skill: SkillDefinition | None = None,
        feedback: list[FeedbackPacket] | None = None,
        revision_context: dict[str, Any] | None = None,
        episode_memory: list[dict[str, Any]] | None = None,
    ) -> EDAPlanDraft:
        if not self.enabled:
            raise ResearchModelUnavailableError("大模型尚未配置，无法生成研究方案。")
        variables = [
            {
                "name": spec.name,
                "availability_type": spec.availability_type,
                "coverage_rate": quality.series[spec.name].aligned_coverage_rate,
                "unit": spec.unit or "unknown",
            }
            for spec in config.exogenous
        ]
        allowed_functions = [
            name
            for name in (skill.allowed_functions if skill is not None else OPTIONAL_FUNCTIONS)
            if name != "data_quality"
        ]
        function_schemas = self.tools.function_schemas(allowed_functions)
        for schema in function_schemas:
            parameters = schema["function"]["parameters"]
            properties = parameters.get("properties", {})
            hidden = {"spike_iqr_multiplier", "outlier_iqr_multiplier", "min_observations"}
            for name in hidden:
                properties.pop(name, None)
            if isinstance(parameters.get("required"), list):
                parameters["required"] = [name for name in parameters["required"] if name not in hidden]
        payload = {
            "prompt_version": PLANNING_PROMPT_VERSION,
            "output_capabilities": report_capabilities_context(),
            "question": question,
            "conversation_history": bounded_recent_history(
                history or [],
                max_messages=self.history_messages,
            ),
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
            "allowed_functions": {
                function_name: {
                    "display_name": FUNCTION_CATALOG[function_name].title,
                    "description": FUNCTION_METADATA[function_name][1],
                    "version": FUNCTION_CATALOG[function_name].version,
                    "result_section": FUNCTION_CATALOG[function_name].result_key,
                    "max_calls_per_plan": FUNCTION_CATALOG[function_name].max_calls_per_plan,
                    "batch_parameter": FUNCTION_CATALOG[function_name].batch_parameter,
                    "planning_guidance": FUNCTION_CATALOG[function_name].planning_guidance,
                }
                for function_name in allowed_functions
            },
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
                "research_agenda": "declare_research_agenda_once_alongside_analysis_functions",
                "minimum_plan": "declare_research_agenda_alone_yields_a_data_quality_only_plan",
            },
        }
        messages = [
            ModelMessage(role="system", content=PLANNING_SYSTEM_PROMPT),
            ModelMessage(role="user", content=json.dumps(payload, ensure_ascii=False)),
        ]
        try:
            invoke_calls = getattr(self.gateway, "invoke_tool_calls", None)
            # The agenda call is always offered, even for a skill whose only
            # function is the compiler-managed data_quality step; otherwise the
            # planner would be told to declare an agenda with no way to send it.
            if callable(invoke_calls):
                calls = invoke_calls(messages=messages, tools=[*function_schemas, AGENDA_FUNCTION_SCHEMA])
                agenda = next((call for call in calls if call.name == AGENDA_FUNCTION_NAME), None)
                analysis_calls = [call for call in calls if call.name != AGENDA_FUNCTION_NAME]
                _validate_agenda_function_coverage(agenda, analysis_calls)
                mode_value = _agenda_text(agenda, "variable_selection_mode") or "explicit"
                if mode_value not in {"explicit", "all_eligible", "auto_recommend"}:
                    raise ResearchPlanValidationError(f"未知外生变量选择模式：{mode_value}")
                selected_set = {
                    str(name)
                    for call in analysis_calls
                    for name in call.arguments.get("variables", [])
                    if isinstance(name, str)
                }
                selected_variables = [spec.name for spec in config.exogenous if spec.name in selected_set]
                deferred_functions: list[str] = []
                selection_stage = "direct"
                if mode_value in {"all_eligible", "auto_recommend"}:
                    selected_variables = eligible_exogenous_variables(config, quality)
                    if not selected_variables:
                        raise ResearchPlanValidationError("没有外生变量满足自动筛查的最低数据门槛")
                    rewritten: list[ModelToolCall] = []
                    for call in analysis_calls:
                        if FUNCTION_CATALOG[call.name].uses_variables:
                            rewritten.append(
                                ModelToolCall(
                                    name=call.name,
                                    arguments={**call.arguments, "variables": selected_variables},
                                    call_id=call.call_id,
                                )
                            )
                        else:
                            rewritten.append(call)
                    analysis_calls = rewritten
                if mode_value == "auto_recommend":
                    requested = {call.name for call in analysis_calls}
                    screening, deferred_functions = screening_function_set(
                        requested,
                        allowed_functions=set(allowed_functions),
                    )
                    by_name = {call.name: call for call in analysis_calls}
                    analysis_calls = []
                    for function_name in allowed_functions:
                        if function_name not in screening:
                            continue
                        previous = by_name.get(function_name)
                        arguments = dict(previous.arguments) if previous is not None else {}
                        if FUNCTION_CATALOG[function_name].uses_variables:
                            arguments["variables"] = selected_variables
                        analysis_calls.append(
                            ModelToolCall(
                                name=function_name,
                                arguments=arguments,
                                call_id=previous.call_id if previous is not None else None,
                            )
                        )
                    selection_stage = "screening"
                recommendation_limit = 8
                if agenda is not None:
                    raw_limit = agenda.arguments.get("variable_recommendation_limit", 8)
                    if isinstance(raw_limit, int) and not isinstance(raw_limit, bool):
                        recommendation_limit = min(32, max(1, raw_limit))
                return EDAPlanDraft(
                    objective=_agenda_text(agenda, "objective") or question.strip(),
                    hypotheses=(
                        []
                        if mode_value == "auto_recommend"
                        else _agenda_list(agenda, "hypotheses")
                    ),
                    selected_variables=selected_variables,
                    variable_selection_mode=mode_value,
                    variable_selection_stage=selection_stage,
                    deferred_functions=deferred_functions,
                    variable_recommendation_limit=recommendation_limit,
                    steps=[
                        {
                            "function": call.name,
                            "enabled": True,
                            "rationale": FUNCTION_CATALOG[call.name].description,
                            "parameters": call.arguments,
                        }
                        for call in analysis_calls
                    ],
                    assumptions=_agenda_list(agenda, "assumptions"),
                )
            return self.gateway.invoke_structured(messages=messages, schema=EDAPlanDraft)
        except KeyError as exc:
            raise ResearchPlanValidationError(f"大模型调用了未注册研究函数：{exc.args[0]}") from exc
        except ModelResponseError as exc:
            raise ResearchPlanValidationError(f"大模型返回的研究方案无法解析：{exc}") from exc
        except (ModelConfigurationError, ModelGatewayError) as exc:
            raise ResearchModelUnavailableError(f"大模型规划调用失败：{exc}") from exc

    def propose_with_context(
        self,
        question: str,
        config: StudyConfig,
        quality: DataQualityReport,
        *,
        history: list[dict[str, str]] | None = None,
        skill: SkillDefinition | None = None,
        feedback: list[FeedbackPacket] | None = None,
        revision_context: dict[str, Any],
        episode_memory: list[dict[str, Any]] | None = None,
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
        history: list[dict[str, str]] | None = None,
        skill: SkillDefinition | None = None,
        feedback: list[FeedbackPacket] | None = None,
        revision_context: dict[str, Any] | None = None,
        episode_memory: list[dict[str, Any]] | None = None,
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
                if episode_memory and _accepts_keyword(self.model_planner.propose, "episode_memory"):
                    planner_kwargs["episode_memory"] = episode_memory
                draft_value = self.model_planner.propose(normalized, config, quality, **planner_kwargs)
        else:
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
