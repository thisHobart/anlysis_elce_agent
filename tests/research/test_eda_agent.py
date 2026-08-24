"""Tests for model-led planning, deterministic execution, and Agent artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from app.config import Settings
from app.llm.gateway import ModelGatewayError
from app.research.agent.errors import ResearchModelUnavailableError, ResearchPlanValidationError
from app.research.agent.orchestrator import DialogueDecision, MainResearchAgent, ModelResearchDialogue
from app.research.agent.schemas import ConversationMessage
from app.research.agent.subagents.eda import EDASubagent, ModelEDAPlanner, max_lag_limit
from app.research.application.coordinator import ResearchCoordinator
from app.research.data.loader import ResearchDataError
from app.research.evaluation.eda import evaluate_agent_run
from app.research.schemas.results import QualityIssue
from app.research.schemas.study import load_study_config


class ScriptedModelPlanner:
    """Deterministic model double; production code still compiles its structured output."""

    enabled = True
    model_name = "scripted-test-model"

    def propose(self, question, config, quality, history=None, skill=None, feedback=None):
        del quality, history, feedback
        assert skill is not None
        names = [spec.name for spec in config.exogenous]
        if "质量" in question or "缺失" in question:
            return {
                "objective": "确认数据是否满足后续 EDA 要求",
                "hypotheses": ["目标序列可能存在需要处理的数据缺口。"],
                "selected_variables": [],
                "steps": [],
                "assumptions": [],
            }
        if not any(word in question for word in ("关系", "相关", "滞后", "影响", "驱动")):
            methods: list[str] = []
            if "季节" in question:
                methods.append("seasonality")
            if any(word in question for word in ("尖峰", "极端", "负价")):
                methods.append("extremes")
            if not methods:
                methods.append("distribution")
            return {
                "objective": "验证用户指定的电价结构特征",
                "hypotheses": ["电价可能存在用户关注的结构特征。"],
                "selected_variables": [],
                "steps": [
                    {
                        "tool": "price_profile",
                        "enabled": True,
                        "rationale": "问题聚焦电价自身结构。",
                        "parameters": {"methods": list(dict.fromkeys(methods)), "max_lag": 24},
                    }
                ],
                "assumptions": [],
            }

        selected = [name for name in names if name.casefold() in question.casefold()]
        if not selected and names:
            selected = [names[0]]
        lag = 2 if "2 小时" in question or "2小时" in question else 24
        return {
            "objective": "验证所选外生变量与电价的关系",
            "hypotheses": ["所选变量与电价可能存在同期或领先滞后关系。"],
            "selected_variables": selected,
            "steps": [
                {
                    "tool": "price_profile",
                    "enabled": True,
                    "rationale": "先识别目标序列的周期结构。",
                    "parameters": {"methods": ["seasonality", "autocorrelation"], "max_lag": lag},
                },
                {
                    "tool": "exogenous_profile",
                    "enabled": True,
                    "rationale": "检查所选变量的分布和异常观测。",
                    "parameters": {"methods": ["distribution", "outliers"]},
                },
                {
                    "tool": "relationship_analysis",
                    "enabled": True,
                    "rationale": "检查同期与领先滞后关系。",
                    "parameters": {"methods": ["pearson", "spearman", "lag_scan"], "max_lag": lag},
                },
            ],
            "assumptions": [],
        }


class ScriptedModelDialogue:
    enabled = True
    model_name = "scripted-test-model"

    def decide(self, **kwargs):
        question = kwargs["question"]
        if "按这个执行" in question or "立即执行" in question:
            return DialogueDecision(intent="execute_plan", response="已锁定当前模型方案。")
        if "只保留关系分析" in question:
            return DialogueDecision(
                intent="revise_plan",
                response="已根据你的意见生成修订方案。",
                objective="只验证 actual_wind 与电价的滞后关系",
                hypotheses=["actual_wind 可能领先电价变化。"],
                enabled_tools=["relationship_analysis"],
                selected_variables=["actual_wind"],
                selected_methods={"relationship_analysis": ["pearson", "spearman", "lag_scan"]},
                max_lag=4,
            )
        if "结果" in question or "说明什么" in question:
            return DialogueDecision(
                intent="explain_result",
                response="已有确定性证据只支持描述性关系，不构成因果结论。",
            )
        if kwargs.get("config") is None:
            return DialogueDecision(
                intent="discussion",
                response="先明确时间轴、季节性和变量可获得时点，再设计 EDA。",
            )
        return DialogueDecision(intent="new_plan", skill_name="price-exogenous-eda")


class FakeModelGateway:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    enabled = True
    model_name = "fake-model"

    def invoke_structured(self, *, messages, schema):
        del messages
        return schema.model_validate(self.payload)

    def invoke_text(self, *, messages):
        del messages
        return json.dumps(self.payload, ensure_ascii=False)


def _model_agent() -> ResearchCoordinator:
    return ResearchCoordinator(
        eda_subagent=EDASubagent(model_planner=ScriptedModelPlanner()),
        main_agent=MainResearchAgent(model_dialogue=ScriptedModelDialogue()),
    )


def test_research_coordinator_uses_the_active_langgraph_workflow():
    agent = _model_agent()

    assert {
        "main_agent",
        "resolve_skill",
        "eda_subagent",
        "approval_interrupt",
        "execute_tool",
        "validate_tool_result",
        "finalize_iteration",
        "user_interrupt",
        "reply",
    }.issubset(
        agent.workflow.get_graph().nodes
    )


def test_default_agent_roles_share_one_model_gateway():
    agent = ResearchCoordinator()

    assert agent.main_agent.model_dialogue.gateway is agent.eda_subagent.model_planner.gateway


def test_model_discusses_eda_without_data_and_does_not_invent_results():
    turn = _model_agent().handle_turn(
        question="电价与外生变量 EDA 通常应该检查什么？",
        status="idle",
    )

    assert turn.action == "reply"
    assert turn.responder == "llm"
    assert "时间轴" in turn.assistant_message
    assert "本轮电价样本" not in turn.assistant_message


def test_missing_model_stops_planning_instead_of_falling_back(synthetic_study: Path):
    settings = Settings(llm_enabled=False, llm_base_url="", llm_model="")
    planner = EDASubagent(model_planner=ModelEDAPlanner(settings))
    agent = ResearchCoordinator(
        eda_subagent=planner,
        main_agent=MainResearchAgent(model_dialogue=ModelResearchDialogue(settings)),
    )

    with pytest.raises(ResearchModelUnavailableError, match="无法生成研究方案"):
        agent.propose(question="分析电价分布", config_path=synthetic_study)
    with pytest.raises(ResearchModelUnavailableError, match="无法开始研究对话"):
        agent.handle_turn(question="先讨论 EDA", status="idle")


def test_model_call_failure_stops_without_local_dialogue_fallback():
    class FailingGateway:
        enabled = True
        model_name = "failing-model"

        def invoke_structured(self, *, messages, schema):
            del messages, schema
            raise ModelGatewayError("simulated provider outage")

        def invoke_text(self, *, messages):
            del messages
            raise ModelGatewayError("simulated provider outage")

    agent = ResearchCoordinator(
        main_agent=MainResearchAgent(
            model_dialogue=ModelResearchDialogue(gateway=FailingGateway())
        ),
        eda_subagent=EDASubagent(model_planner=ScriptedModelPlanner()),
    )

    with pytest.raises(ResearchModelUnavailableError, match="simulated provider outage"):
        agent.handle_turn(question="讨论电价EDA", status="idle")


def test_model_recommends_different_processes_for_different_questions(synthetic_study: Path):
    agent = _model_agent()
    quality_plan = agent.propose(question="先检查数据质量和缺失情况", config_path=synthetic_study).plan
    relationship_plan = agent.propose(
        question="分析 load 与电价 2 小时的滞后关系",
        config_path=synthetic_study,
    ).plan

    assert [step.tool for step in quality_plan.enabled_steps] == ["data_quality"]
    assert [step.tool for step in relationship_plan.enabled_steps] == [
        "data_quality",
        "price_profile",
        "exogenous_profile",
        "relationship_analysis",
    ]
    relationship_step = next(step for step in relationship_plan.steps if step.tool == "relationship_analysis")
    assert relationship_plan.planner == "llm"
    assert relationship_plan.planning_model == "scripted-test-model"
    assert relationship_plan.selected_variables == ["load"]
    assert relationship_step.parameters["max_lag"] == 2
    assert relationship_step.parameters["methods"] == ["pearson", "spearman", "lag_scan"]
    assert relationship_step.method_versions == {
        "relationship.scipy_pearson_pairwise": "1.0.0",
        "relationship.scipy_spearman_pairwise": "1.0.0",
        "relationship.pearson_positive_lead_scan": "1.0.0",
    }
    assert max_lag_limit("15min") == 31 * 24 * 4


def test_model_selects_only_requested_price_methods(synthetic_study: Path):
    plan = _model_agent().propose(
        question="只分析电价季节性和尖峰",
        config_path=synthetic_study,
    ).plan
    price_step = next(step for step in plan.steps if step.tool == "price_profile")

    assert price_step.parameters["methods"] == ["seasonality", "extremes"]
    assert price_step.method_versions == {
        "price.calendar_group_profile": "1.0.0",
        "price.tukey_outer_fence": "1.0.0",
    }
    assert plan.data_fingerprint is not None


def test_invalid_model_plan_is_rejected_without_local_repair(synthetic_study: Path):
    model_planner = ModelEDAPlanner(
        gateway=FakeModelGateway(
        {
            "objective": "验证负荷与电价的同期关系",
            "hypotheses": [],
            "selected_variables": ["not_a_real_variable"],
            "steps": [
                {
                    "tool": "relationship_analysis",
                    "enabled": True,
                    "rationale": "问题聚焦同期关系。",
                    "parameters": {"methods": ["pearson", "not_allowed"], "max_lag": 99999},
                }
            ],
            "assumptions": [],
        }
        )
    )
    agent = ResearchCoordinator(
        eda_subagent=EDASubagent(model_planner=model_planner),
        main_agent=MainResearchAgent(model_dialogue=ScriptedModelDialogue()),
    )

    with pytest.raises(ResearchPlanValidationError, match="未知变量"):
        agent.propose(question="分析负荷与电价同期关系", config_path=synthetic_study)


def test_model_revision_controls_the_executable_plan(synthetic_study: Path, tmp_path: Path):
    agent = _model_agent()
    proposal = agent.propose(question="分析 load 与电价的滞后关系", config_path=synthetic_study)
    config = load_study_config(synthetic_study)
    revised, _message = agent.main_agent.revise_plan(
        question="只保留关系分析，把最大滞后改为 4 小时，只分析 wind",
        plan=proposal.plan,
        config=config,
        decision=DialogueDecision(
            intent="revise_plan",
            enabled_tools=["relationship_analysis"],
            selected_variables=["wind"],
            selected_methods={"relationship_analysis": ["pearson", "lag_scan"]},
            max_lag=4,
        ),
    )
    result = agent.execute(
        plan=revised,
        config_path=synthetic_study,
        conversation=[ConversationMessage(role="user", content=revised.question)],
        output_directory=tmp_path / "agent-artifacts",
        run_id="agent-model-revision",
    )

    assert revised.parent_plan_id == proposal.plan.plan_id
    assert revised.revision_source == "user_dialogue"
    assert [step.tool for step in revised.enabled_steps] == ["data_quality", "relationship_analysis"]
    assert list(result.eda_summary["relationships"]["series"]) == ["wind"]
    assert set(result.figure_paths) == {"correlation_matrix", "lag_relationships"}

    package_files = {path.name for path in result.artifact_directory.iterdir()}
    assert {"conversation.json", "research_plan.json", "execution_trace.json", "manifest.json"}.issubset(
        package_files
    )
    manifest = json.loads((result.artifact_directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["research_agent"]["planning_model"] == "scripted-test-model"
    assert manifest["research_agent"]["skill"] == {"name": "price-exogenous-eda", "version": "1.0.0"}
    assert manifest["research_agent"]["tool_versions"]["S4"] == "1.0.0"
    assert manifest["research_agent"]["method_versions"]["S4"] == {
        "relationship.scipy_pearson_pairwise": "1.0.0",
        "relationship.pearson_positive_lead_scan": "1.0.0",
    }
    for output in manifest["outputs"]:
        path = result.artifact_directory / output["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == output["sha256"]


def test_evaluator_scopes_unselected_variable_risks(synthetic_study: Path, tmp_path: Path):
    agent = _model_agent()
    plan = agent.propose(question="分析电价季节性和尖峰", config_path=synthetic_study).plan
    result = agent.execute(
        plan=plan,
        config_path=synthetic_study,
        output_directory=tmp_path / "price-artifacts",
        run_id="agent-price-only",
    )
    quality_with_unselected_risk = result.quality_report.model_copy(
        update={
            "issues": [
                QualityIssue(
                    severity="high",
                    code="unselected_variable_risk",
                    series="load",
                    message="This variable was not used by the model plan.",
                )
            ]
        }
    )
    scoped = evaluate_agent_run(plan=plan, quality=quality_with_unselected_risk, summary=result.eda_summary)

    assert scoped.decision == "accept"
    assert not scoped.warnings


def test_execution_rejects_files_changed_after_plan_generation(synthetic_study: Path, tmp_path: Path):
    agent = _model_agent()
    proposal = agent.propose(question="分析电价分布", config_path=synthetic_study)
    config = load_study_config(synthetic_study)
    frame = pd.read_csv(config.target.path)
    frame.loc[0, config.target.value_column] = float(frame.loc[0, config.target.value_column]) + 1
    frame.to_csv(config.target.path, index=False)

    with pytest.raises(ResearchDataError, match="方案生成后发生变化"):
        agent.execute(
            plan=proposal.plan,
            config_path=synthetic_study,
            output_directory=tmp_path / "changed-input-artifacts",
        )


def test_execution_rejects_missing_method_version(synthetic_study: Path, tmp_path: Path):
    agent = _model_agent()
    plan = agent.propose(question="分析电价分布", config_path=synthetic_study).plan
    price = next(step for step in plan.steps if step.tool == "price_profile")
    invalid_steps = [
        step.model_copy(update={"method_versions": {}}) if step.step_id == price.step_id else step
        for step in plan.steps
    ]
    invalid = plan.model_copy(update={"steps": invalid_steps})

    with pytest.raises(ResearchPlanValidationError, match="method version mismatch"):
        agent.execute(
            plan=invalid,
            config_path=synthetic_study,
            output_directory=tmp_path / "invalid-version-artifacts",
        )


def test_execution_rejects_skill_and_tool_version_changes(synthetic_study: Path, tmp_path: Path):
    agent = _model_agent()
    plan = agent.propose(question="分析电价季节性", config_path=synthetic_study).plan

    with pytest.raises(ResearchPlanValidationError, match="Skill 版本不匹配"):
        agent.execute(
            plan=plan.model_copy(update={"skill_version": "0.0.0"}),
            config_path=synthetic_study,
            output_directory=tmp_path / "invalid-skill-version",
        )

    price = next(step for step in plan.steps if step.tool == "price_profile")
    changed_steps = [
        step.model_copy(update={"tool_version": "0.0.0"}) if step.step_id == price.step_id else step
        for step in plan.steps
    ]
    with pytest.raises(ResearchPlanValidationError, match="工具版本不匹配"):
        agent.execute(
            plan=plan.model_copy(update={"steps": changed_steps}),
            config_path=synthetic_study,
            output_directory=tmp_path / "invalid-tool-version",
        )


def test_same_data_and_locked_plan_produce_identical_structured_results(
    synthetic_study: Path,
    tmp_path: Path,
):
    agent = _model_agent()
    plan = agent.propose(
        question="分析 load 与电价的滞后关系",
        config_path=synthetic_study,
    ).plan

    first = agent.execute(
        plan=plan,
        config_path=synthetic_study,
        output_directory=tmp_path / "repeat-a",
        run_id="repeat-a",
    )
    second = agent.execute(
        plan=plan,
        config_path=synthetic_study,
        output_directory=tmp_path / "repeat-b",
        run_id="repeat-b",
    )

    assert first.eda_summary == second.eda_summary
    assert first.quality_report == second.quality_report
    assert first.evaluation == second.evaluation
