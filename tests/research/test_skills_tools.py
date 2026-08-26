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
from app.research.skills.loader import SkillLoadError, load_skill
from app.research.skills.registry import SkillRegistry
from app.research.tools.catalog import TOOL_CATALOG
from app.research.tools.contracts import ToolCall, ToolContext
from app.research.tools.eda.functions import build_eda_tool_registry
from app.research.tools.executor import ToolExecutor
from app.research.tools.policy import ToolPermissionError, ToolPolicy


def test_builtin_eda_skill_is_discoverable_and_versioned():
    registry = SkillRegistry.default(Settings(skill_paths=""))
    skill = registry.get("price-exogenous-eda")

    assert skill.source == "builtin"
    assert skill.version == "3.0.0"
    assert skill.domain == "eda"
    assert set(skill.allowed_tools) == set(TOOL_CATALOG)
    assert skill.research_protocol is not None
    assert skill.research_protocol.protocol_id == "electricity-price-evidence-ladder"
    assert skill.research_protocol.version == "2.0.0"
    assert set(skill.research_protocol.function_order) == set(TOOL_CATALOG)


def test_builtin_domain_protocol_orders_model_calls_locally(synthetic_study: Path):
    skill = SkillRegistry.default(Settings(skill_paths="")).get("price-exogenous-eda")
    config = load_study_config(synthetic_study)
    prepared = prepare_research_data(config)

    class OutOfOrderPlanner:
        enabled = True
        model_name = "test-model"

        def propose(self, _question, _config, _quality, history=None, skill=None, feedback=None):
            del history, feedback
            return {
                "objective": "研究 wind 与电价的领先关系",
                "selected_variables": ["wind"],
                "steps": [
                    {
                        "tool": "relationship_pearson_positive_lead_scan",
                        "rationale": "检查正领先关系。",
                        "parameters": {"variables": ["wind"], "max_lag": 24},
                    },
                    {
                        "tool": "exogenous_descriptive_distribution",
                        "rationale": "先检查变量画像。",
                        "parameters": {"variables": ["wind"]},
                    },
                    {
                        "tool": "price_lag_autocorrelation",
                        "rationale": "建立电价自身滞后基线。",
                        "parameters": {"max_lag": 24},
                    },
                ],
            }

    plan = EDASubagent(model_planner=OutOfOrderPlanner()).propose(
        "研究 wind 与电价的领先关系",
        config,
        prepared.quality,
        skill=skill,
    )

    assert [step.tool for step in plan.enabled_steps] == [
        "data_quality",
        "price_lag_autocorrelation",
        "exogenous_descriptive_distribution",
        "relationship_pearson_positive_lead_scan",
    ]
    assert plan.research_protocol_id == "electricity-price-evidence-ladder"
    assert plan.research_protocol_version == "2.0.0"
    assert plan.research_protocol_function_order == list(skill.research_protocol.function_order)
    assert any("请求参数禁用 thinking" in note for note in plan.planning_notes)

    class RevisionDialogue:
        enabled = True
        model_name = "test-model"

    revised, _ = MainResearchAgent(model_dialogue=RevisionDialogue()).revise_plan(
        question="增加极端值识别",
        plan=plan,
        config=config,
        decision=DialogueDecision(
            intent="revise_plan",
            enabled_tools=[
                "price_tukey_outer_fence",
                "price_lag_autocorrelation",
                "exogenous_descriptive_distribution",
                "relationship_pearson_positive_lead_scan",
            ],
            selected_variables=["wind"],
            max_lag=24,
        ),
    )
    assert [step.tool for step in revised.enabled_steps] == [
        "data_quality",
        "price_tukey_outer_fence",
        "price_lag_autocorrelation",
        "exogenous_descriptive_distribution",
        "relationship_pearson_positive_lead_scan",
    ]


def test_skill_protocol_reference_cannot_escape_bundle(tmp_path: Path):
    skill_root = tmp_path / "unsafe-protocol-skill"
    skill_root.mkdir()
    outside = tmp_path / "outside.yaml"
    outside.write_text("protocol_id: outside\n", encoding="utf-8")
    (skill_root / "SKILL.md").write_text(
        "---\n"
        "name: unsafe-protocol-skill\n"
        "description: Must not load protocol files outside the bundle.\n"
        "metadata: {version: '1.0.0', domain: eda, protocol-file: '../outside.yaml'}\n"
        "allowed-tools: data_quality\n"
        "---\n"
        "Use the declared local protocol.\n",
        encoding="utf-8",
    )

    with pytest.raises(SkillLoadError, match="不能离开 Skill 目录"):
        load_skill(skill_root, source="external")


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
    assert skill.allowed_tools == [
        "data_quality",
        *[name for name, spec in TOOL_CATALOG.items() if spec.category == "price"],
    ]
    assert not marker.exists()


def test_external_professional_skill_can_drive_the_eda_subagent(tmp_path: Path, synthetic_study: Path):
    skill_root = tmp_path / "professional-eda"
    skill_root.mkdir()
    (skill_root / "SKILL.md").write_text(
        "---\n"
        "name: professional-eda\n"
        "description: Professional external EDA guidance.\n"
        "metadata: {version: '2.1.0', domain: eda}\n"
        "allowed-tools: data_quality price_descriptive_distribution\n"
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
                        "tool": "price_descriptive_distribution",
                        "enabled": True,
                        "rationale": "外部专业 Skill 要求最小分布分析。",
                        "parameters": {},
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
        "allowed-tools: data_quality price_descriptive_distribution\n"
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
                        "tool": "relationship_scipy_pearson_pairwise",
                        "enabled": True,
                        "rationale": "应被编译器拒绝。",
                        "parameters": {"variables": [config.exogenous[0].name]},
                    }
                ],
            }

    subagent = EDASubagent(model_planner=DisallowedPlanner())
    config = load_study_config(synthetic_study)
    prepared = prepare_research_data(config)

    with pytest.raises(ResearchPlanValidationError, match="未授权函数"):
        subagent.propose("分析关系", config, prepared.quality, skill=skill)


def test_tool_registry_exposes_openai_compatible_function_schemas():
    registry = build_eda_tool_registry()
    schemas = registry.function_schemas(["price_lag_autocorrelation", "data_quality"])

    assert [item["function"]["name"] for item in schemas] == ["data_quality", "price_lag_autocorrelation"]
    price_schema = next(item for item in schemas if item["function"]["name"] == "price_lag_autocorrelation")
    assert set(price_schema["function"]["parameters"]["properties"]) == {"max_lag"}


def test_tool_executor_enforces_skill_policy_and_runs_registered_function(synthetic_study: Path):
    from app.research.schemas.study import load_study_config

    config = load_study_config(synthetic_study)
    prepared = prepare_research_data(config)
    registry = build_eda_tool_registry()
    executor = ToolExecutor(registry)
    context = ToolContext(config=config, frame=prepared.aligned.frame, quality=prepared.quality)
    call = ToolCall(
        call_id="test:price",
        work_id="0" * 24,
        step_id="S1",
        name="price_descriptive_distribution",
        version="1.0.0",
        arguments={},
    )

    with pytest.raises(ToolPermissionError):
        executor.execute(call, context=context, policy=ToolPolicy(frozenset({"data_quality"})))

    result = executor.execute(
        call,
        context=context,
        policy=ToolPolicy(frozenset({"data_quality", "price_descriptive_distribution"})),
    )
    assert result.status == "completed"
    assert result.output.result_key == "price"
    assert result.output.value["methods"] == ["distribution"]
