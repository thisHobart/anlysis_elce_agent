"""Structured contracts exchanged by the EDA research agent and desktop UI."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.research.schemas.feedback import FeedbackPacket
from app.research.schemas.results import DataQualityReport
from app.research.tools.catalog import method_keys, selected_implementation_versions

EDAToolName = Literal[
    "data_quality",
    "price_profile",
    "exogenous_profile",
    "relationship_analysis",
]


class ConversationMessage(BaseModel):
    """One persisted message from the research conversation."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant", "system"]
    content: str = Field(min_length=1)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class EDAPlanStep(BaseModel):
    """One allow-listed analytical tool call proposed by the agent."""

    model_config = ConfigDict(extra="forbid")

    step_id: str = Field(pattern=r"^S[1-9][0-9]*$")
    tool: EDAToolName
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    enabled: bool = True
    required: bool = False
    parameters: dict[str, Any] = Field(default_factory=dict)
    tool_version: str = Field(min_length=1)
    method_versions: dict[str, str] = Field(default_factory=dict)


class EDAPlan(BaseModel):
    """An editable and reproducible EDA plan proposed for one question."""

    model_config = ConfigDict(extra="forbid")

    plan_id: str = Field(min_length=1)
    data_fingerprint: str | None = Field(default=None, pattern=r"^[a-f0-9]{12}$")
    parent_plan_id: str | None = None
    revision: int = Field(default=1, ge=1)
    revision_reason: str | None = None
    revision_source: Literal[
        "initial",
        "user_ui",
        "user_dialogue",
        "automatic_validation",
        "automatic_evaluation",
    ] = "initial"
    question: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    study_name: str = Field(min_length=1)
    planner: Literal["llm"]
    planning_model: str | None = None
    planning_prompt_version: str = "eda-plan-v5"
    skill_name: str = Field(min_length=1)
    skill_version: str = Field(min_length=1)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    hypotheses: list[str] = Field(default_factory=list)
    selected_variables: list[str] = Field(default_factory=list)
    steps: list[EDAPlanStep] = Field(min_length=1)
    assumptions: list[str] = Field(default_factory=list)
    planning_notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_steps(self) -> EDAPlan:
        ids = [step.step_id for step in self.steps]
        tools = [step.tool for step in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("plan step ids must be unique")
        if len(tools) != len(set(tools)):
            raise ValueError("each EDA tool may appear at most once")
        quality = next((step for step in self.steps if step.tool == "data_quality"), None)
        if quality is None or not quality.required or not quality.enabled:
            raise ValueError("data_quality must be an enabled, required plan step")
        for step in self.steps:
            methods = step.parameters.get("methods")
            if methods is None:
                continue
            if not isinstance(methods, list) or any(not isinstance(method, str) for method in methods):
                raise ValueError(f"{step.tool} methods must be a list of strings")
            unknown_methods = sorted(set(methods).difference(method_keys(step.tool)))
            if unknown_methods:
                raise ValueError(f"{step.tool} contains unknown methods: {', '.join(unknown_methods)}")
            if step.enabled and method_keys(step.tool) and not methods:
                raise ValueError(f"enabled tool {step.tool} must select at least one method")
        return self

    @property
    def enabled_steps(self) -> list[EDAPlanStep]:
        return [step for step in self.steps if step.enabled]

    def adjusted(
        self,
        *,
        enabled_step_ids: set[str],
        selected_variables: list[str],
        max_lag: int,
        selected_methods: dict[str, list[str]] | None = None,
        reason: str = "用户在计划卡中确认或调整了分析方案。",
        source: Literal["user_ui", "user_dialogue"] = "user_ui",
    ) -> EDAPlan:
        """Return a user-edited copy while preserving mandatory safety checks."""

        if max_lag < 0:
            raise ValueError("max_lag must not be negative")
        lag_limits = [
            int(step.parameters["max_lag_limit"])
            for step in self.steps
            if step.parameters.get("max_lag_limit") is not None
        ]
        if lag_limits:
            max_lag = min(max_lag, min(lag_limits))
        variables = list(dict.fromkeys(selected_variables))
        updated_steps: list[EDAPlanStep] = []
        for step in self.steps:
            parameters = dict(step.parameters)
            if selected_methods is not None and step.tool in selected_methods:
                allowed = set(method_keys(step.tool))
                parameters["methods"] = [
                    method for method in dict.fromkeys(selected_methods[step.tool]) if method in allowed
                ]
            if step.tool in {"exogenous_profile", "relationship_analysis"}:
                parameters["variables"] = variables
            if step.tool in {"price_profile", "relationship_analysis"}:
                parameters["max_lag"] = max_lag
            enabled = step.required or step.step_id in enabled_step_ids
            if enabled and method_keys(step.tool) and not parameters.get("methods"):
                raise ValueError(f"enabled tool {step.tool} must select at least one method")
            updated_steps.append(
                step.model_copy(
                    update={
                        "enabled": enabled,
                        "parameters": parameters,
                        "method_versions": selected_implementation_versions(
                            step.tool,
                            parameters.get("methods", []),
                        ),
                    }
                )
            )
        return self.model_copy(
            update={
                "plan_id": uuid4().hex[:12],
                "parent_plan_id": self.plan_id,
                "revision": self.revision + 1,
                "revision_reason": reason,
                "revision_source": source,
                "created_at": datetime.now(UTC).isoformat(),
                "selected_variables": variables,
                "steps": updated_steps,
                "planning_notes": [*self.planning_notes, reason],
            }
        )


class ResearchDataProfile(BaseModel):
    """Compact data evidence shown before an analytical plan is accepted."""

    model_config = ConfigDict(extra="forbid")

    target_name: str
    target_unit: str = "unknown"
    market: str = "unspecified"
    exogenous_names: list[str]
    aligned_rows: int
    start_time: str
    end_time: str
    frequency: str
    timezone: str
    target_coverage_rate: float
    issue_counts: dict[str, int]


class ResearchProposal(BaseModel):
    """Agent response containing both evidence and an editable recommendation."""

    model_config = ConfigDict(extra="forbid")

    plan: EDAPlan
    assistant_message: str
    data_profile: ResearchDataProfile
    quality_report: DataQualityReport


class ResearchTurnResult(BaseModel):
    """One routed conversational turn returned to the desktop controller."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["reply", "proposal", "plan_revision", "execute_plan"]
    assistant_message: str = ""
    responder: Literal["llm"]
    proposal: ResearchProposal | None = None
    revised_plan: EDAPlan | None = None
    available_variables: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_payload(self) -> ResearchTurnResult:
        if self.action == "proposal" and self.proposal is None:
            raise ValueError("proposal action requires a proposal")
        if self.action in {"plan_revision", "execute_plan"} and self.revised_plan is None:
            raise ValueError(f"{self.action} action requires a plan")
        if self.action == "reply" and not self.assistant_message.strip():
            raise ValueError("reply action requires an assistant message")
        return self


class EvaluationCheck(BaseModel):
    """One deterministic check applied to executed Agent evidence."""

    model_config = ConfigDict(extra="forbid")

    name: str
    status: Literal["pass", "warning", "fail"]
    message: str


class HypothesisAssessment(BaseModel):
    """Evaluator outcome for one Agent-proposed exploratory hypothesis."""

    model_config = ConfigDict(extra="forbid")

    hypothesis: str
    status: Literal["candidate_support", "not_supported", "inconclusive", "not_tested"]
    evidence: str


class AgentEvaluation(BaseModel):
    """Evidence-based assessment produced after tool execution."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["accept", "revise", "need_user", "reject"]
    summary: str
    checks: list[EvaluationCheck]
    hypothesis_assessments: list[HypothesisAssessment] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    suggested_followups: list[str] = Field(default_factory=list)
    feedback_packets: list[FeedbackPacket] = Field(default_factory=list)


class AgentRunResult(BaseModel):
    """Desktop-facing result from one approved Agent plan."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    run_id: str
    status: Literal["completed"] = "completed"
    artifact_directory: Path
    report_path: Path
    figure_paths: dict[str, Path]
    aligned_rows: int
    plan: EDAPlan
    quality_report: DataQualityReport
    eda_summary: dict[str, Any]
    evaluation: AgentEvaluation
