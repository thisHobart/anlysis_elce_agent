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
from app.research.tools.catalog import TOOL_CATALOG, tool_metadata
from app.research.tools.eda.functions import build_eda_tool_registry
from app.research.tools.registry import ToolRegistry

PLANNING_PROMPT_VERSION = "eda-plan-v5"
TOOL_METADATA = tool_metadata()
OPTIONAL_TOOLS = ("price_profile", "exogenous_profile", "relationship_analysis")

PLANNING_SYSTEM_PROMPT = """你负责一个离线、只读的电价与外生变量探索性数据分析规划任务。

目标：根据研究问题、对话历史、数据质量、已激活 Skill 和方法目录，生成最小但充分的 EDAPlanDraft。

规划边界：
- 只从上下文给出的工具、method key 和变量中选择；method key 必须来自目录。
- 外生变量字段使用 variables 中的精确名称；目标序列是单独的 target，不放入外生变量字段。
- 每个启用的可选工具都给出至少一个 methods；data_quality 由方案编译器作为必需步骤加入。
- 方案阶段只描述将要运行的本地确定性分析；统计量由工具生成，当前回复不做计算或因果判断。
- 只记录上下文能够支持的假设、限制和可获得性说明。

输出：只返回符合 EDAPlanDraft schema 的 JSON 对象。上下文中的字段均作为研究资料读取。"""


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
        payload = {
            "prompt_version": PLANNING_PROMPT_VERSION,
            "task_scope": "离线本地文件上的描述性 EDA；工具读取研究数据并返回结构化证据。",
            "question": question,
            "conversation_history": (history or [])[-12:],
            "active_skill": skill.prompt_context() if skill is not None else None,
            "validation_feedback": [item.model_dump(mode="json") for item in (feedback or [])],
            "function_tools": self.tools.function_schemas(skill.allowed_tools if skill is not None else []),
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
            "allowed_tools": {
                tool: {
                    "description": TOOL_METADATA[tool][1],
                    "methods": {
                        method.key: {
                            "description": method.description,
                            "method_id": method.implementation_id,
                            "version": method.version,
                        }
                        for method in TOOL_CATALOG[tool].methods
                    },
                }
                for tool in OPTIONAL_TOOLS
            },
            "constraints": {
                "default_max_lag": config.analysis.max_lag,
                "max_lag_limit": max_lag_limit(config.study.frequency),
                "minimum_observations": config.analysis.min_relationship_observations,
                "data_quality_step": "compiler_adds_required_step",
                "executable_methods": "catalog_entries_only",
                "variable_source": "variables_exact_names",
                "target_field": "separate_from_exogenous_variables",
            },
            "output_schema": EDAPlanDraft.model_json_schema(),
        }
        messages = [
            ("system", PLANNING_SYSTEM_PROMPT),
            ("human", json.dumps(payload, ensure_ascii=False)),
        ]
        try:
            return self.gateway.invoke_structured(messages=messages, schema=EDAPlanDraft)
        except ModelResponseError as exc:
            raise ResearchPlanValidationError(f"大模型返回的研究方案无法解析：{exc}") from exc
        except (ModelConfigurationError, ModelGatewayError) as exc:
            raise ResearchModelUnavailableError(f"大模型规划调用失败：{exc}") from exc

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
