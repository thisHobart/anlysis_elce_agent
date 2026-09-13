"""Structured contracts exchanged by the EDA research agent and desktop UI."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.research.agent.prompts import PLANNING_PROMPT_VERSION
from app.research.data.sources.summary import DataSummary
from app.research.planning.variables import VariableSelectionMode, VariableSelectionStage
from app.research.schemas.feedback import FeedbackPacket
from app.research.schemas.results import DataQualityReport
from app.research.tools.catalog import FUNCTION_CATALOG, LEGACY_METHOD_TO_FUNCTION, ResearchFunctionName

EDAToolName = ResearchFunctionName
AgendaScope = Literal[
    "within_envelope",
    "needs_approval",
    "needs_data",
    "needs_restatement",
    "inherent",
]


class ConversationMessage(BaseModel):
    """One persisted message from the research conversation."""

    model_config = ConfigDict(extra="forbid")

    message_id: str = Field(default_factory=lambda: uuid4().hex)
    turn_id: str | None = None
    episode_id: str | None = None
    role: Literal["user", "assistant", "system"]
    content: str = Field(min_length=1)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


def rename_legacy_step_keys(step: Any) -> Any:
    """Accept plans written before a research function stopped being called a tool."""

    if not isinstance(step, dict):
        return step
    renamed = dict(step)
    for old_key, new_key in (("tool", "function"), ("tool_version", "function_version")):
        if old_key in renamed and new_key not in renamed:
            renamed[new_key] = renamed.pop(old_key)
        else:
            renamed.pop(old_key, None)
    return renamed


class EDAPlanStep(BaseModel):
    """One allow-listed atomic research function proposed by the agent."""

    model_config = ConfigDict(extra="forbid")

    step_id: str = Field(pattern=r"^S[1-9][0-9]*$")
    function: EDAToolName
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    enabled: bool = True
    required: bool = False
    parameters: dict[str, Any] = Field(default_factory=dict)
    function_version: str = Field(min_length=1)


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
    planning_prompt_version: str = PLANNING_PROMPT_VERSION
    skill_name: str = Field(min_length=1)
    skill_version: str = Field(min_length=1)
    research_protocol_id: str | None = None
    research_protocol_version: str | None = None
    research_protocol_function_order: list[EDAToolName] = Field(default_factory=list)
    research_protocol_step_texts: dict[EDAToolName, str] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    hypotheses: list[str] = Field(default_factory=list)
    unverifiable_hypotheses: list[str] = Field(default_factory=list)
    analysis_kind: Literal["research", "descriptive"] = "research"
    requested_statistics: tuple[Literal["mean", "min", "max"], ...] = ()
    selected_variables: list[str] = Field(default_factory=list)
    variable_selection_mode: VariableSelectionMode = "explicit"
    variable_selection_stage: VariableSelectionStage = "direct"
    deferred_functions: list[EDAToolName] = Field(default_factory=list)
    variable_recommendation_limit: int = Field(default=8, ge=1, le=32)
    steps: list[EDAPlanStep] = Field(min_length=1)
    assumptions: list[str] = Field(default_factory=list)
    planning_notes: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_steps(cls, value: Any) -> Any:
        """Translate v1 aggregate tool/method plans into atomic function calls."""

        if not isinstance(value, dict) or not isinstance(value.get("steps"), list):
            return value
        value = {**value, "steps": [rename_legacy_step_keys(step) for step in value["steps"]]}
        legacy_tools = {"price_profile", "exogenous_profile", "relationship_analysis"}
        if not any(isinstance(step, dict) and step.get("function") in legacy_tools for step in value["steps"]):
            return value
        migrated: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw_step in value["steps"]:
            if not isinstance(raw_step, dict):
                continue
            name = str(raw_step.get("function", ""))
            if name == "data_quality":
                quality = {key: item for key, item in raw_step.items() if key != "method_versions"}
                migrated.append(quality)
                seen.add(name)
                continue
            if name not in legacy_tools:
                migrated.append({key: item for key, item in raw_step.items() if key != "method_versions"})
                seen.add(name)
                continue
            parameters = dict(raw_step.get("parameters") or {})
            methods = parameters.pop("methods", [])
            for method in methods:
                function_name = LEGACY_METHOD_TO_FUNCTION.get((name, str(method)))
                if function_name is None:
                    raise ValueError(f"旧计划包含无法迁移的方法：{name}.{method}")
                if function_name in seen:
                    continue
                spec = FUNCTION_CATALOG[function_name]
                function_parameters: dict[str, Any] = {}
                if spec.uses_variables and "variables" in parameters:
                    function_parameters["variables"] = parameters["variables"]
                if spec.uses_max_lag:
                    if "max_lag" in parameters:
                        function_parameters["max_lag"] = parameters["max_lag"]
                    if "max_lag_limit" in parameters:
                        function_parameters["max_lag_limit"] = parameters["max_lag_limit"]
                if function_name == "price_tukey_outer_fence" and "spike_iqr_multiplier" in parameters:
                    function_parameters["spike_iqr_multiplier"] = parameters["spike_iqr_multiplier"]
                if function_name == "exogenous_iqr_outliers" and "outlier_iqr_multiplier" in parameters:
                    function_parameters["outlier_iqr_multiplier"] = parameters["outlier_iqr_multiplier"]
                if spec.category == "relationship" and "min_observations" in parameters:
                    function_parameters["min_observations"] = parameters["min_observations"]
                migrated.append(
                    {
                        "step_id": "S1",
                        "function": function_name,
                        "title": spec.title,
                        "description": spec.description,
                        "rationale": raw_step.get("rationale") or spec.description,
                        "enabled": bool(raw_step.get("enabled", True)),
                        "required": False,
                        "parameters": function_parameters,
                        "function_version": spec.version,
                    }
                )
                seen.add(function_name)
        for index, step in enumerate(migrated, start=1):
            step["step_id"] = f"S{index}"
        return {**value, "steps": migrated}

    @model_validator(mode="after")
    def validate_steps(self) -> EDAPlan:
        ids = [step.step_id for step in self.steps]
        tools = [step.function for step in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("plan step ids must be unique")
        if len(tools) != len(set(tools)):
            raise ValueError("each research function may appear at most once")
        if len(self.deferred_functions) != len(set(self.deferred_functions)):
            raise ValueError("deferred research functions must not contain duplicates")
        if self.variable_selection_stage == "screening" and self.variable_selection_mode != "auto_recommend":
            raise ValueError("variable screening stage requires auto_recommend mode")
        quality = next((step for step in self.steps if step.function == "data_quality"), None)
        if quality is None or not quality.required or not quality.enabled:
            raise ValueError("data_quality must be an enabled, required plan step")
        if bool(self.research_protocol_id) != bool(self.research_protocol_version):
            raise ValueError("research protocol id and version must be recorded together")
        if self.research_protocol_function_order:
            if not self.research_protocol_id:
                raise ValueError("research protocol function order requires a protocol id")
            if len(self.research_protocol_function_order) != len(
                set(self.research_protocol_function_order)
            ):
                raise ValueError("research protocol function order must not contain duplicates")
            uncovered = sorted(set(tools).difference(self.research_protocol_function_order))
            if uncovered:
                raise ValueError(f"research protocol does not cover plan functions: {', '.join(uncovered)}")
        selected = list(dict.fromkeys(self.selected_variables))
        for step in self.steps:
            spec = FUNCTION_CATALOG[step.function]
            if "methods" in step.parameters:
                raise ValueError(f"atomic function {step.function} must not contain methods")
            if spec.uses_variables and step.enabled and step.parameters.get("variables") != selected:
                raise ValueError(f"{step.function} variables must match plan.selected_variables")
            if spec.uses_max_lag and step.enabled and step.parameters.get("max_lag") is None:
                raise ValueError(f"{step.function} requires max_lag")
        return self

    @property
    def enabled_steps(self) -> list[EDAPlanStep]:
        return [step for step in self.steps if step.enabled]

    def step_text(self, step: EDAPlanStep) -> str:
        """Say what one step does in the words the research protocol chose.

        The catalog title names a statistical method, which is the right label
        for a report and the wrong one for the analyst approving the run, so the
        protocol line wins wherever it exists.
        """

        return self.research_protocol_step_texts.get(step.function) or step.title

    def ordered_by_research_protocol(self) -> EDAPlan:
        """Return a copy whose step order follows the protocol captured in this plan."""

        if not self.research_protocol_function_order:
            return self
        rank = {
            name: index for index, name in enumerate(self.research_protocol_function_order)
        }
        ordered = sorted(self.steps, key=lambda step: rank[step.function])
        renumbered = [
            step.model_copy(update={"step_id": f"S{index}"})
            for index, step in enumerate(ordered, start=1)
        ]
        return self.model_copy(update={"steps": renumbered})

    def adjusted(
        self,
        *,
        enabled_step_ids: set[str],
        selected_variables: list[str],
        max_lag: int,
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
            spec = FUNCTION_CATALOG[step.function]
            if spec.uses_variables:
                parameters["variables"] = variables
            if spec.uses_max_lag:
                parameters["max_lag"] = max_lag
            enabled = step.required or step.step_id in enabled_step_ids
            updated_steps.append(
                step.model_copy(
                    update={
                        "enabled": enabled,
                        "parameters": parameters,
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
    # Written once when the dataset was frozen, so the panel and the confirmation
    # card keep naming the moment the data was actually read.
    data_summary: DataSummary | None = None


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
    scope: AgendaScope = "inherent"
    remediable: bool = False
    remediation: str | None = None

    @model_validator(mode="after")
    def migrate_remediable_scope(self) -> EvaluationCheck:
        """Keep persisted pre-scope checks compatible with the agenda contract."""

        if self.remediable and self.scope == "inherent":
            self.scope = "within_envelope"
        return self


class HypothesisAssessment(BaseModel):
    """Evaluator outcome for one Agent-proposed exploratory hypothesis."""

    model_config = ConfigDict(extra="forbid")

    item_id: str = ""
    hypothesis: str
    status: Literal["candidate_support", "not_supported", "inconclusive", "not_tested"]
    evidence: str
    scope: AgendaScope = "inherent"
    remediation: str | None = None


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
    variable_recommendations: dict[str, Any] | None = None
    feedback_packets: list[FeedbackPacket] = Field(default_factory=list)
    agenda_fingerprint: str = ""


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
