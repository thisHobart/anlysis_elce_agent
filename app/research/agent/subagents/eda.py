"""EDA Subagent: model-led planning compiled into deterministic executable steps."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from app.config import Settings
from app.llm.factory import build_model_gateway
from app.llm.gateway import ModelConfigurationError, ModelGateway, ModelGatewayError, ModelResponseError
from app.research.agent.errors import ResearchModelUnavailableError, ResearchPlanValidationError
from app.research.agent.schemas import EDAPlan
from app.research.planning.compiler import EDAPlanCompiler, max_lag_limit
from app.research.planning.contracts import EDAPlanDraft
from app.research.schemas.feedback import FeedbackPacket
from app.research.schemas.results import DataQualityReport
from app.research.schemas.study import StudyConfig
from app.research.skills.contracts import SkillDefinition
from app.research.tools.catalog import TOOL_CATALOG, optional_function_names, tool_metadata
from app.research.tools.eda.functions import build_eda_tool_registry
from app.research.tools.registry import ToolRegistry

PLANNING_PROMPT_VERSION = "eda-plan-v9"
TOOL_METADATA = tool_metadata()
OPTIONAL_FUNCTIONS = optional_function_names()

PLANNING_SYSTEM_PROMPT = """你负责一个离线、只读的电价与外生变量探索性数据分析规划任务。

目标：根据研究问题、对话历史、数据质量和已激活 Skill，选择最小但充分的原子研究函数。

规划边界：
- 你只是受限函数选择器，不自行设计通用推理步骤；严格服从 active_skill.research_protocol 的阶段、函数规则和停止条件。
- 每个函数名只对应一种确定性统计过程；只调用当前提供的函数，不生成 methods 参数。
- 每个研究函数在一个计划中最多调用一次。多变量合并到 variables，多滞后用一个最大 max_lag，峰谷、季节或事件前后对比合并到一个 segments 集合。
- segments 中的小时、月份和时间边界必须来自用户或研究配置的明确值；不得自行猜测市场峰谷时段、季节定义或政策事件日期。
- 外生变量参数使用 variables 中的精确名称；目标序列是单独的 target，不放入 variables。
- data_quality 由方案编译器作为必需步骤加入，不由模型调用。
- 方案阶段只描述将要运行的本地确定性分析；统计量由工具生成，当前回复不做计算或因果判断。
- 只记录上下文能够支持的假设、限制和可获得性说明。
- 优先选择最小函数集合；不要因为函数可用就全部调用。
- 如果 revision_context 存在，必须把 current_plan 作为修订基线；只调整 allowed_changes 中允许的字段，其他字段保持不变。
- 自动修订不得改变研究问题、Skill、数据指纹、分段定义或审批范围；无法在边界内修复时返回原方案边界内的最小变更。

输出：只通过 Function Calling 返回一个或多个具体研究函数调用，不输出推理文本；程序只把调用编译为候选计划，不会立即执行。"""


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
            for name in (skill.allowed_tools if skill is not None else OPTIONAL_FUNCTIONS)
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
            "task_scope": "离线本地文件上的描述性 EDA；工具读取研究数据并返回结构化证据。",
            "question": question,
            "conversation_history": (history or [])[-12:],
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
                    "display_name": TOOL_CATALOG[function_name].title,
                    "description": TOOL_METADATA[function_name][1],
                    "version": TOOL_CATALOG[function_name].version,
                    "result_section": TOOL_CATALOG[function_name].result_key,
                    "max_calls_per_plan": TOOL_CATALOG[function_name].max_calls_per_plan,
                    "batch_parameter": TOOL_CATALOG[function_name].batch_parameter,
                    "planning_guidance": TOOL_CATALOG[function_name].planning_guidance,
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
            },
            "output_schema": EDAPlanDraft.model_json_schema(),
        }
        messages = [
            ("system", PLANNING_SYSTEM_PROMPT),
            ("human", json.dumps(payload, ensure_ascii=False)),
        ]
        try:
            invoke_calls = getattr(self.gateway, "invoke_tool_calls", None)
            if callable(invoke_calls) and function_schemas:
                calls = invoke_calls(messages=messages, tools=function_schemas)
                selected_set = {
                    str(name)
                    for call in calls
                    for name in call.arguments.get("variables", [])
                    if isinstance(name, str)
                }
                selected_variables = [spec.name for spec in config.exogenous if spec.name in selected_set]
                return EDAPlanDraft(
                    objective=question.strip(),
                    hypotheses=[],
                    selected_variables=selected_variables,
                    steps=[
                        {
                            "tool": call.name,
                            "enabled": True,
                            "rationale": TOOL_CATALOG[call.name].description,
                            "parameters": call.arguments,
                        }
                        for call in calls
                    ],
                    assumptions=[],
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
                draft_value = contextual_propose(
                    normalized,
                    config,
                    quality,
                    **planner_kwargs,
                    revision_context=revision_context,
                )
            else:
                draft_value = self.model_planner.propose(normalized, config, quality, **planner_kwargs)
        else:
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
