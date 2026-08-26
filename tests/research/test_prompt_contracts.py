"""Regression tests for the bounded model prompts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.research.agent.orchestrator import DIALOGUE_PROMPT_VERSION, ModelResearchDialogue
from app.research.agent.subagents.eda import PLANNING_PROMPT_VERSION, ModelEDAPlanner
from app.research.application.planning import prepare_research_data
from app.research.schemas.study import load_study_config
from app.research.skills.loader import load_skill


class CaptureGateway:
    enabled = True
    model_name = "capture"

    def __init__(self) -> None:
        self.calls: list[list[tuple[str, str]]] = []

    def invoke_structured(self, *, messages, schema):
        self.calls.append(messages)
        raise RuntimeError(f"captured {schema.__name__}")

    def invoke_text(self, *, messages):
        self.calls.append(messages)
        raise RuntimeError("captured text")


def _payload(messages: list[tuple[str, str]]) -> dict:
    return json.loads(next(content for role, content in messages if role == "human"))


def test_planning_prompt_states_local_eda_scope_and_version(synthetic_study: Path):
    config = load_study_config(synthetic_study)
    prepared = prepare_research_data(config)
    skill = load_skill(Path("app/research/skills/price-exogenous-eda/SKILL.md"), source="builtin")
    gateway = CaptureGateway()

    with pytest.raises(RuntimeError, match="captured EDAPlanDraft"):
        ModelEDAPlanner(gateway=gateway).propose(
            "分析电价分布",
            config,
            prepared.quality,
            skill=skill,
        )

    messages = gateway.calls[0]
    assert "离线、只读" in messages[0][1]
    assert "Function Calling" in messages[0][1]
    payload = _payload(messages)
    assert payload["prompt_version"] == PLANNING_PROMPT_VERSION
    assert payload["task_scope"].startswith("离线本地文件上的描述性 EDA")
    assert payload["constraints"]["executable_functions"] == "provided_function_names_only"
    assert payload["constraints"]["function_cardinality"] == "each_function_at_most_once_per_plan"
    assert "price_calendar_group_profile" in payload["allowed_functions"]
    assert payload["allowed_functions"]["price_segment_distribution_comparison"]["batch_parameter"] == "segments"
    assert payload["allowed_functions"]["price_lag_autocorrelation"]["batch_parameter"] == "max_lag"
    assert "reasoning_control" not in payload
    assert (
        payload["active_skill"]["research_protocol"]["protocol_id"]
        == "electricity-price-evidence-ladder"
    )
    assert "method_id" not in json.dumps(payload, ensure_ascii=False)
    assert "target_must_never_appear_in_selected_variables" not in payload["constraints"]


def test_dialogue_prompt_states_route_contract_and_version(synthetic_study: Path):
    config = load_study_config(synthetic_study)
    prepared = prepare_research_data(config)
    skill = load_skill(Path("app/research/skills/price-exogenous-eda/SKILL.md"), source="builtin")
    gateway = CaptureGateway()

    with pytest.raises(RuntimeError, match="captured DialogueDecision"):
        ModelResearchDialogue(gateway=gateway).decide(
            question="分析电价结构",
            status="idle",
            config=config,
            plan=None,
            data_profile=None,
            quality_report=prepared.quality.model_dump(mode="json"),
            summary=None,
            evaluation=None,
            history=[],
            available_skills=[skill.prompt_context()],
        )

    messages = gateway.calls[0]
    assert "离线、只读" in messages[0][1]
    assert "DialogueDecision schema" in messages[0][1]
    payload = _payload(messages)
    assert payload["prompt_version"] == DIALOGUE_PROMPT_VERSION
    assert payload["task_scope"].startswith("离线本地文件上的描述性 EDA")
    assert payload["revision_contract"]["function_selection"] == "exact_allowed_function_names_only"
    assert "reasoning_control" not in payload
    assert payload["study"]["target"]["name"] == config.target.name


def test_automatic_revision_prompt_contains_current_plan_and_approval_boundary(synthetic_study: Path):
    config = load_study_config(synthetic_study)
    prepared = prepare_research_data(config)
    skill = load_skill(Path("app/research/skills/price-exogenous-eda/SKILL.md"), source="builtin")
    gateway = CaptureGateway()
    revision_context = {
        "current_plan": {"plan_id": "approved-plan", "selected_variables": ["load"]},
        "authorization_envelope": {"functions": ["price_descriptive_distribution"]},
        "allowed_changes": {"max_lag": "decrease only"},
    }

    with pytest.raises(RuntimeError, match="captured EDAPlanDraft"):
        ModelEDAPlanner(gateway=gateway).propose_with_context(
            "分析电价",
            config,
            prepared.quality,
            skill=skill,
            revision_context=revision_context,
        )

    messages = gateway.calls[0]
    assert "revision_context" in messages[0][1]
    payload = _payload(messages)
    assert payload["revision_context"] == revision_context
