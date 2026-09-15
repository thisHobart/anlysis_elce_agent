"""Model-led dynamic selection of approved EDA functions."""

from __future__ import annotations

import inspect
import json
from copy import deepcopy
from typing import Any

from app.llm.budget import ModelRequestPurpose
from app.llm.gateway import ModelGateway, ModelMessage, ModelToolTurn
from app.research.agent.prompts import ANALYSIS_TOOL_PROMPT_VERSION, ANALYSIS_TOOL_SYSTEM_PROMPT
from app.research.agent.schemas import EDAResearchScope
from app.research.skills.contracts import SkillDefinition
from app.research.tools.catalog import FUNCTION_CATALOG
from app.research.tools.recipes import RecipeRegistry, build_recipe_registry
from app.research.tools.registry import ToolRegistry


def _function_guidance(skill: SkillDefinition) -> dict[str, str]:
    if skill.research_protocol is None:
        return {}
    return {
        name: rule
        for stage in skill.research_protocol.stages
        for name, rule in stage.function_rules.items()
    }


def _tool_schemas(
    registry: ToolRegistry,
    recipes: RecipeRegistry,
    scope: EDAResearchScope,
    skill: SkillDefinition,
) -> list[dict[str, Any]]:
    guidance = _function_guidance(skill)
    schemas = deepcopy(registry.function_schemas(scope.authorized_functions))
    for schema in schemas:
        function = schema["function"]
        catalog = FUNCTION_CATALOG[str(function["name"])]
        parameters = function["parameters"]
        properties = parameters.get("properties", {})
        # These values are safety policy, not model choices.  The compiler
        # injects them from trusted local configuration after a call arrives.
        locally_managed = {
            "spike_iqr_multiplier",
            "outlier_iqr_multiplier",
            "min_observations",
            "max_lag_limit",
        }
        for name in locally_managed:
            properties.pop(name, None)
        if isinstance(parameters.get("required"), list):
            parameters["required"] = [
                name for name in parameters["required"] if name not in locally_managed
            ]
        if "variables" in properties:
            properties["variables"]["description"] = (
                "只可选择本次研究范围授权的外生变量："
                + "、".join(scope.authorized_variables)
            )
        rule = guidance.get(str(function["name"]))
        function["description"] = (
            f"{function.get('description', '')} 回答：{catalog.answers}。"
            f"输出证据归入 {catalog.result_key}；只能作描述性判读，不能据此声称因果或样本外预测增益。"
        ).strip()
        if rule:
            function["description"] = f"{function.get('description', '')} 调用时机与判读：{rule}".strip()
    return [*schemas, *recipes.schemas_for(set(scope.authorized_functions))]


class DynamicAnalysisAgent:
    """Choose the next approved tool batch or finish from accumulated evidence."""

    def __init__(
        self,
        *,
        gateway: ModelGateway,
        tools: ToolRegistry,
        recipes: RecipeRegistry | None = None,
    ) -> None:
        self.gateway = gateway
        self.tools = tools
        self.recipes = recipes or build_recipe_registry()

    @property
    def enabled(self) -> bool:
        return self.gateway.enabled

    def initial_messages(
        self,
        *,
        scope: EDAResearchScope,
        quality_report: dict[str, Any],
        data_profile: dict[str, Any] | None,
    ) -> list[ModelMessage]:
        payload = {
            "prompt_version": ANALYSIS_TOOL_PROMPT_VERSION,
            "approved_scope": scope.model_dump(mode="json"),
            "data_profile": data_profile or {},
            "data_quality": {
                "usable_for_eda": quality_report.get("usable_for_eda"),
                "issues": quality_report.get("issues", []),
                "series": quality_report.get("series", {}),
            },
            "instruction": "根据现有证据选择下一批函数；证据足够时返回正文并停止调用。",
        }
        return [
            ModelMessage(role="system", content=ANALYSIS_TOOL_SYSTEM_PROMPT),
            ModelMessage(role="user", content=json.dumps(payload, ensure_ascii=False)),
        ]

    def select(
        self,
        *,
        scope: EDAResearchScope,
        skill: SkillDefinition,
        messages: list[ModelMessage],
    ) -> ModelToolTurn:
        schemas = _tool_schemas(self.tools, self.recipes, scope, skill)
        kwargs: dict[str, Any] = {
            "messages": messages,
            "tools": schemas,
        }
        if "purpose" in inspect.signature(self.gateway.invoke_tool_turn).parameters:
            kwargs["purpose"] = ModelRequestPurpose.EDA_ANALYSIS
        return self.gateway.invoke_tool_turn(**kwargs)
