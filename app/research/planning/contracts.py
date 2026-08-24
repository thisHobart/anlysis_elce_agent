"""Structured model draft compiled into an executable EDA plan."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class DraftStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: Literal["price_profile", "exogenous_profile", "relationship_analysis"]
    enabled: bool = True
    rationale: str = Field(min_length=1)
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Only parameters supported by the selected function schema.",
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
        description="Optional EDA function calls only; data_quality is mandatory and added by the compiler.",
    )
    assumptions: list[str] = Field(default_factory=list)
