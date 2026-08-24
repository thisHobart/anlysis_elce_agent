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
    assert "EDAPlanDraft schema" in messages[0][1]
    payload = _payload(messages)
    assert payload["prompt_version"] == PLANNING_PROMPT_VERSION
    assert payload["task_scope"].startswith("离线本地文件上的描述性 EDA")
    assert payload["constraints"]["executable_methods"] == "catalog_entries_only"
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
    assert payload["revision_contract"]["method_selection"] == "catalog_keys_only"
    assert payload["study"]["target"]["name"] == config.target.name
