"""Structured model draft compiled into an executable EDA plan."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.research.agent.schemas import rename_legacy_step_keys
from app.research.planning.variables import VariableSelectionMode, VariableSelectionStage
from app.research.tools.catalog import FUNCTION_CATALOG, LEGACY_METHOD_TO_FUNCTION, ResearchFunctionName
from app.research.tools.contracts import SegmentDefinition


class IntentFunctionSelection(BaseModel):
    """One compact function choice with only user-controlled argument overrides."""

    model_config = ConfigDict(extra="forbid")

    function: ResearchFunctionName
    max_lag: int | None = Field(default=None, ge=0)
    comparison_id: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_]{0,31}$")
    segments: list[SegmentDefinition] | None = Field(default=None, min_length=2, max_length=12)


class MinimalEDAPlanIntent(BaseModel):
    """Bounded recovery contract used after a truncated normal planning response."""

    model_config = ConfigDict(extra="forbid")

    variable_selection_mode: VariableSelectionMode = "explicit"
    selected_variable_ids: list[str] = Field(default_factory=list, max_length=64)
    functions: list[IntentFunctionSelection] = Field(default_factory=list, max_length=32)
    variable_recommendation_limit: int = Field(default=8, ge=1, le=32)

    @model_validator(mode="after")
    def validate_unique_selections(self) -> MinimalEDAPlanIntent:
        variables = list(dict.fromkeys(self.selected_variable_ids))
        if len(variables) != len(self.selected_variable_ids):
            raise ValueError("selected_variable_ids must be unique")
        functions = [item.function for item in self.functions]
        if len(functions) != len(set(functions)):
            raise ValueError("functions must be unique")
        return self


class EDAPlanIntent(MinimalEDAPlanIntent):
    """Compact model-owned planning intent expanded by the deterministic compiler."""

    objective: str = Field(min_length=1, max_length=160)
    hypotheses: list[str] = Field(default_factory=list, max_length=8)
    assumptions: list[str] = Field(default_factory=list, max_length=6)


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
    variable_selection_mode: VariableSelectionMode = "explicit"
    variable_selection_stage: VariableSelectionStage = "direct"
    deferred_functions: list[ResearchFunctionName] = Field(default_factory=list)
    variable_recommendation_limit: int = Field(default=8, ge=1, le=32)
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
