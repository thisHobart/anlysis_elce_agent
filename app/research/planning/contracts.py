"""Structured model draft compiled into an executable EDA plan."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.research.agent.schemas import rename_legacy_step_keys
from app.research.tools.catalog import FUNCTION_CATALOG, LEGACY_METHOD_TO_FUNCTION, ResearchFunctionName


class DraftStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    function: ResearchFunctionName
    enabled: bool = True
    rationale: str = Field(min_length=1)
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Only execution arguments supported by this exact research function.",
    )


class EDAPlanDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    objective: str = Field(min_length=1)
    hypotheses: list[str] = Field(default_factory=list)
    selected_variables: list[str] = Field(
        default_factory=list,
        description=(
            "Exact names of exogenous variables selected from the provided variables list. "
            "Never include the target price name. Use an empty list for price-only analysis."
        ),
    )
    steps: list[DraftStep] = Field(
        default_factory=list,
        description="Atomic EDA function calls only; data_quality is mandatory and added by the compiler.",
    )
    assumptions: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_aggregate_steps(cls, value: Any) -> Any:
        """Accept persisted/test v1 drafts while keeping the public schema atomic."""

        if not isinstance(value, dict) or not isinstance(value.get("steps"), list):
            return value
        selected_variables = list(dict.fromkeys(value.get("selected_variables") or []))
        migrated: list[dict[str, Any]] = []
        for raw_step in value["steps"]:
            step = rename_legacy_step_keys(raw_step)
            if not isinstance(step, dict) or step.get("function") not in {
                "price_profile",
                "exogenous_profile",
                "relationship_analysis",
            }:
                migrated.append(step)
                continue
            legacy_tool = str(step["function"])
            parameters = dict(step.get("parameters") or {})
            methods = parameters.pop("methods", [])
            for method in methods:
                function_name = LEGACY_METHOD_TO_FUNCTION.get((legacy_tool, str(method)))
                if function_name is None:
                    raise ValueError(f"旧方案包含未注册方法：{legacy_tool}.{method}")
                spec = FUNCTION_CATALOG[function_name]
                function_parameters: dict[str, Any] = {}
                if spec.uses_variables:
                    function_parameters["variables"] = parameters.get("variables", selected_variables)
                if spec.uses_max_lag and "max_lag" in parameters:
                    function_parameters["max_lag"] = parameters["max_lag"]
                migrated.append(
                    {
                        "function": function_name,
                        "enabled": bool(step.get("enabled", True)),
                        "rationale": step.get("rationale") or spec.description,
                        "parameters": function_parameters,
                    }
                )
        return {**value, "steps": migrated}
