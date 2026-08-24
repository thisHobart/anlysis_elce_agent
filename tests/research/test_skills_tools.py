"""Tests for Skill discovery and controlled local function calling."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.research.agent.errors import ResearchPlanValidationError
from app.research.agent.orchestrator import DialogueDecision, MainResearchAgent
from app.research.agent.subagents.eda import EDASubagent
from app.research.application.coordinator import ResearchCoordinator
from app.research.application.planning import prepare_research_data
from app.research.schemas.study import load_study_config
from app.research.skills.registry import SkillRegistry
from app.research.tools.contracts import ToolCall, ToolContext
from app.research.tools.eda.functions import build_eda_tool_registry
from app.research.tools.executor import ToolExecutor
from app.research.tools.policy import ToolPermissionError, ToolPolicy


def test_builtin_eda_skill_is_discoverable_and_versioned():
    registry = SkillRegistry.default(Settings(skill_paths=""))
    skill = registry.get("price-exogenous-eda")

    assert skill.source == "builtin"
    assert skill.version == "1.0.0"
    assert skill.domain == "eda"
    assert set(skill.allowed_tools) == {
        "data_quality",
        "price_profile",
        "exogenous_profile",
        "relationship_analysis",
    }


def test_external_professional_skill_is_loaded_without_executing_scripts(tmp_path: Path):
    skill_root = tmp_path / "professional-eda"
    skill_root.mkdir()
    (skill_root / "SKILL.md").write_text(
        "---\n"
        "name: professional-eda\n"
        "description: Professional external EDA guidance.\n"
        "metadata:\n"
        "  version: '2.1.0'\n"
        "  domain: eda\n"
        "allowed-tools: data_quality price_profile\n"
        "---\n"
        "Use robust price diagnostics and never invent evidence.\n",
        encoding="utf-8",
    )
    scripts = skill_root / "scripts"
    scripts.mkdir()
    marker = tmp_path / "must-not-exist.txt"
    (scripts / "unsafe.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n",
        encoding="utf-8",
    )

    registry = SkillRegistry.default(Settings(skill_paths=str(tmp_path)))
    skill = registry.get("professional-eda")

    assert skill.source == "external"
    assert skill.version == "2.1.0"
    assert skill.allowed_tools == ["data_quality", "price_profile"]
    assert not marker.exists()


def test_external_professional_skill_can_drive_the_eda_subagent(tmp_path: Path, synthetic_study: Path):
    skill_root = tmp_path / "professional-eda"
    skill_root.mkdir()
    (skill_root / "SKILL.md").write_text(
        "---\n"
        "name: professional-eda\n"
        "description: Professional external EDA guidance.\n"
        "metadata: {version: '2.1.0', domain: eda}\n"
        "allowed-tools: data_quality price_profile\n"
        "---\n"
        "Use a minimal price-distribution workflow.\n",
        encoding="utf-8",
    )
    skills = SkillRegistry.default(Settings(skill_paths=str(tmp_path)))

    class ExternalDialogue:
        enabled = True
        model_name = "test-model"

        def decide(self, **_kwargs):
            return DialogueDecision(intent="new_plan", skill_name="professional-eda")

    class ExternalPlanner:
        enabled = True
        model_name = "test-model"

        def propose(self, _question, _config, _quality, history=None, skill=None, feedback=None):
            del history, feedback
            assert skill.name == "professional-eda"
            return {
                "objective": "检查电价分布",
                "hypotheses": [],
                "selected_variables": [],
                "steps": [
                    {
                        "tool": "price_profile",
                        "enabled": True,
                        "rationale": "外部专业 Skill 要求最小分布分析。",
                        "parameters": {"methods": ["distribution"], "max_lag": 24},
                    }
                ],
                "assumptions": [],
            }

    coordinator = ResearchCoordinator(
        skills=skills,
        main_agent=MainResearchAgent(model_dialogue=ExternalDialogue()),
        eda_subagent=EDASubagent(model_planner=ExternalPlanner()),
    )
    turn = coordinator.handle_turn(
        question="按专业流程分析电价分布",
        status="idle",
        study_config=load_study_config(synthetic_study),
    )

    assert turn.action == "proposal"
    assert turn.proposal.plan.skill_name == "professional-eda"
    assert turn.proposal.plan.skill_version == "2.1.0"


def test_external_skill_cannot_plan_an_unapproved_function(tmp_path: Path, synthetic_study: Path):
    skill_root = tmp_path / "price-only-skill"
    skill_root.mkdir()
    (skill_root / "SKILL.md").write_text(
        "---\n"
        "name: price-only-skill\n"
        "description: Only permits price profiling.\n"
        "metadata: {version: '1.0.0', domain: eda}\n"
        "allowed-tools: data_quality price_profile\n"
        "---\n"
        "Only profile the target price.\n",
        encoding="utf-8",
    )
    skill = SkillRegistry.default(Settings(skill_paths=str(tmp_path))).get("price-only-skill")

    class DisallowedPlanner:
        enabled = True
        model_name = "test-model"

        def propose(self, _question, config, _quality, history=None, skill=None, feedback=None):
            del history, feedback
            return {
                "objective": "越权关系分析",
                "selected_variables": [config.exogenous[0].name],
                "steps": [
                    {
                        "tool": "relationship_analysis",
                        "enabled": True,
                        "rationale": "应被编译器拒绝。",
                        "parameters": {"methods": ["pearson"], "max_lag": 1},
                    }
                ],
            }

    subagent = EDASubagent(model_planner=DisallowedPlanner())
    config = load_study_config(synthetic_study)
    prepared = prepare_research_data(config)

    with pytest.raises(ResearchPlanValidationError, match="未授权工具"):
        subagent.propose("分析关系", config, prepared.quality, skill=skill)


def test_tool_registry_exposes_openai_compatible_function_schemas():
    registry = build_eda_tool_registry()
    schemas = registry.function_schemas(["price_profile", "data_quality"])

    assert [item["function"]["name"] for item in schemas] == ["data_quality", "price_profile"]
    price_schema = next(item for item in schemas if item["function"]["name"] == "price_profile")
    assert {"methods", "max_lag", "spike_iqr_multiplier"}.issubset(
        price_schema["function"]["parameters"]["properties"]
    )


def test_tool_executor_enforces_skill_policy_and_runs_registered_function(synthetic_study: Path):
    from app.research.schemas.study import load_study_config

    config = load_study_config(synthetic_study)
    prepared = prepare_research_data(config)
    registry = build_eda_tool_registry()
    executor = ToolExecutor(registry)
    context = ToolContext(config=config, frame=prepared.aligned.frame, quality=prepared.quality)
    call = ToolCall(
        call_id="test:price",
        name="price_profile",
        version="1.0.0",
        arguments={"methods": ["distribution"], "max_lag": 24, "spike_iqr_multiplier": 3.0},
    )

    with pytest.raises(ToolPermissionError):
        executor.execute(call, context=context, policy=ToolPolicy(frozenset({"data_quality"})))

    result = executor.execute(
        call,
        context=context,
        policy=ToolPolicy(frozenset({"data_quality", "price_profile"})),
    )
    assert result.status == "completed"
    assert result.output.result_key == "price"
    assert result.output.value["methods"] == ["distribution"]
