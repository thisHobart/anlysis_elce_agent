"""Tests for model-led planning, deterministic execution, and Agent artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from app.config import Settings
from app.llm.gateway import ModelGatewayError, ModelOutputTruncatedError
from app.research.agent.errors import ResearchModelUnavailableError, ResearchPlanValidationError
from app.research.agent.orchestrator import DialogueDecision, MainResearchAgent, ModelResearchDialogue
from app.research.agent.schemas import ConversationMessage
from app.research.agent.subagents.eda import EDASubagent, ModelEDAPlanner, max_lag_limit
from app.research.application.coordinator import ResearchCoordinator
from app.research.application.execution import EDAExecutionService
from app.research.application.planning import prepare_research_data
from app.research.data.loader import ResearchDataError
from app.research.data.snapshot import input_file_manifest, study_fingerprint
from app.research.data.sources.materialize import snapshot_root
from app.research.evaluation.eda import evaluate_agent_run
from app.research.planning.contracts import EDAPlanDraft
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
            functions: list[str] = []
            if "季节" in question:
                functions.append("price_calendar_group_profile")
            if any(word in question for word in ("尖峰", "极端", "负价")):
                functions.append("price_tukey_outer_fence")
            if not functions:
                functions.append("price_descriptive_distribution")
            return {
                "objective": "验证用户指定的电价结构特征",
                "hypotheses": [],
                "selected_variables": [],
                "steps": [
                    {
                        "function": function_name,
                        "enabled": True,
                        "rationale": "问题聚焦电价自身结构。",
                        "parameters": {},
                    }
                    for function_name in dict.fromkeys(functions)
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
                    "function": "price_calendar_group_profile",
                    "enabled": True,
                    "rationale": "先识别目标序列的周期结构。",
                    "parameters": {},
                },
                {
                    "function": "price_lag_autocorrelation",
                    "enabled": True,
                    "rationale": "检查目标序列的滞后结构。",
                    "parameters": {"max_lag": lag},
                },
                {
                    "function": "exogenous_descriptive_distribution",
                    "enabled": True,
                    "rationale": "检查所选变量的分布。",
                    "parameters": {"variables": selected},
                },
                {
                    "function": "exogenous_iqr_outliers",
                    "enabled": True,
                    "rationale": "检查所选变量的异常观测。",
                    "parameters": {"variables": selected},
                },
                {
                    "function": "relationship_scipy_pearson_pairwise",
                    "enabled": True,
                    "rationale": "检查同期线性关系。",
                    "parameters": {"variables": selected},
                },
                {
                    "function": "relationship_scipy_spearman_pairwise",
                    "enabled": True,
                    "rationale": "检查同期秩关系。",
                    "parameters": {"variables": selected},
                },
                {
                    "function": "relationship_pearson_positive_lead_scan",
                    "enabled": True,
                    "rationale": "检查领先滞后关系。",
                    "parameters": {"variables": selected, "max_lag": lag},
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
                enabled_functions=[
                    "relationship_scipy_pearson_pairwise",
                    "relationship_scipy_spearman_pairwise",
                    "relationship_pearson_positive_lead_scan",
                ],
                selected_variables=["actual_wind"],
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
    }.issubset(agent.workflow.get_graph().nodes)


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
    settings = Settings(llm_base_url="", llm_model="")
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
        main_agent=MainResearchAgent(model_dialogue=ModelResearchDialogue(gateway=FailingGateway())),
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

    assert [step.function for step in quality_plan.enabled_steps] == ["data_quality"]
    assert [step.function for step in relationship_plan.enabled_steps] == [
        "data_quality",
        "price_calendar_group_profile",
        "price_lag_autocorrelation",
        "exogenous_descriptive_distribution",
        "exogenous_iqr_outliers",
        "relationship_scipy_pearson_pairwise",
        "relationship_scipy_spearman_pairwise",
        "relationship_pearson_positive_lead_scan",
    ]
    relationship_step = next(
        step for step in relationship_plan.steps if step.function == "relationship_pearson_positive_lead_scan"
    )
    assert relationship_plan.planner == "llm"
    assert relationship_plan.planning_model == "scripted-test-model"
    assert relationship_plan.selected_variables == ["load"]
    assert relationship_step.parameters["max_lag"] == 2
    assert all(step.function_version == "1.0.0" for step in relationship_plan.enabled_steps)
    assert max_lag_limit("15min") == 31 * 24 * 4


def test_model_selects_only_requested_price_methods(synthetic_study: Path):
    plan = (
        _model_agent()
        .propose(
            question="只分析电价季节性和尖峰",
            config_path=synthetic_study,
        )
        .plan
    )
    assert [step.function for step in plan.enabled_steps] == [
        "data_quality",
        "price_tukey_outer_fence",
        "price_calendar_group_profile",
    ]
    assert plan.data_fingerprint is not None


def test_legacy_implementation_ids_migrate_to_atomic_functions(synthetic_study: Path):
    class LegacyDraftPlanner:
        enabled = True
        model_name = "legacy-draft-model"

        def propose(self, *args, **kwargs):
            del args, kwargs
            return EDAPlanDraft.model_validate(
                {
                    "objective": "兼容旧模型输出",
                    "selected_variables": [],
                    "steps": [
                        {
                            "tool": "price_profile",
                            "enabled": True,
                            "rationale": "旧模型错误地返回实现 ID。",
                            "parameters": {
                                "methods": [
                                    "price.descriptive_distribution",
                                    "price.calendar_group_profile",
                                ]
                            },
                        }
                    ],
                }
            )

    planner = LegacyDraftPlanner()
    coordinator = ResearchCoordinator(
        eda_subagent=EDASubagent(model_planner=planner),
        main_agent=MainResearchAgent(model_dialogue=ScriptedModelDialogue()),
    )

    plan = coordinator.propose(question="分析电价分布和季节性", config_path=synthetic_study).plan

    assert [step.function for step in plan.enabled_steps] == [
        "data_quality",
        "price_descriptive_distribution",
        "price_calendar_group_profile",
    ]
    assert all("methods" not in step.parameters for step in plan.steps)


def test_compact_intent_compiles_into_an_approval_plan(synthetic_study: Path):
    class CompactIntentGateway:
        enabled = True
        model_name = "compact-intent-model"

        def __init__(self):
            self.schemas = []
            self.messages = []

        def invoke_structured(self, *, messages, schema, purpose=None):
            self.schemas.append(schema)
            self.messages.append(messages)
            return schema.model_validate(
                {
                    "objective": "分析电价分布和自相关",
                    "variable_selection_mode": "explicit",
                    "selected_variable_ids": [],
                    "functions": [
                        {"function": "price_descriptive_distribution"},
                        {"function": "price_lag_autocorrelation", "max_lag": 2},
                    ],
                }
            )

    gateway = CompactIntentGateway()
    coordinator = ResearchCoordinator(
        eda_subagent=EDASubagent(model_planner=ModelEDAPlanner(gateway=gateway)),
        main_agent=MainResearchAgent(model_dialogue=ScriptedModelDialogue()),
    )

    plan = coordinator.propose(question="分析电价分布和自相关", config_path=synthetic_study).plan

    assert [step.function for step in plan.enabled_steps] == [
        "data_quality",
        "price_descriptive_distribution",
        "price_lag_autocorrelation",
    ]
    assert "电价可能存在自相关或持续性。" in plan.hypotheses
    assert next(step for step in plan.steps if step.function == "price_lag_autocorrelation").parameters["max_lag"] == 2
    assert gateway.schemas[0].__name__ == "EDAPlanIntent"
    payload = json.loads(gateway.messages[0][-1].content)
    assert len(payload["allowed_functions"]) == 29
    assert payload["variables"][0]["id"] == "v1"
    assert payload["data_fingerprint"] == plan.data_fingerprint


class _CompactIntentGateway:
    """Return one bounded intent and retain every attempted schema."""

    enabled = True
    model_name = "compact-intent-model"

    def __init__(self, payload, *, truncate_first: bool = False):
        self.payload = payload
        self.truncate_first = truncate_first
        self.schemas: list = []
        self.messages: list = []

    def invoke_structured(self, *, messages, schema, purpose=None):
        del purpose
        self.schemas.append(schema)
        self.messages.append(messages)
        if self.truncate_first and len(self.schemas) == 1:
            raise ModelOutputTruncatedError("length")
        return schema.model_validate(self.payload)


def _intent_coordinator(gateway) -> ResearchCoordinator:
    return ResearchCoordinator(
        eda_subagent=EDASubagent(model_planner=ModelEDAPlanner(gateway=gateway)),
        main_agent=MainResearchAgent(model_dialogue=ScriptedModelDialogue()),
    )


def test_compact_intent_carries_the_agenda_without_pseudo_tool(synthetic_study: Path):
    gateway = _CompactIntentGateway(
        {
            "objective": "判定电价分布是否需要方差稳定预处理",
            "hypotheses": ["电价分布可能明显偏斜或存在厚尾。"],
            "assumptions": ["目标序列时区已按自动识别的数据上下文对齐。"],
            "variable_selection_mode": "explicit",
            "functions": [{"function": "price_descriptive_distribution"}],
        },
    )

    plan = _intent_coordinator(gateway).propose(question="电价分布长什么样", config_path=synthetic_study).plan

    assert plan.objective == "判定电价分布是否需要方差稳定预处理"
    assert "电价分布可能明显偏斜或存在厚尾。" in plan.hypotheses
    assert "目标序列时区已按自动识别的数据上下文对齐。" in plan.assumptions
    assert [step.function for step in plan.enabled_steps] == [
        "data_quality",
        "price_descriptive_distribution",
    ]

    assert gateway.schemas[0].__name__ == "EDAPlanIntent"
    assert "declare_research_agenda" not in gateway.messages[0][0].content


def test_empty_compact_function_list_produces_data_quality_only_plan(synthetic_study: Path):
    gateway = _CompactIntentGateway(
        {"objective": "先确认这批数据能不能用", "variable_selection_mode": "explicit", "functions": []}
    )

    plan = _intent_coordinator(gateway).propose(question="先看看数据质量", config_path=synthetic_study).plan

    assert [step.function for step in plan.enabled_steps] == ["data_quality"]
    assert plan.objective == "先确认这批数据能不能用"
    assert plan.hypotheses == []


def test_truncated_plan_retries_once_with_minimal_schema(synthetic_study: Path):
    gateway = _CompactIntentGateway(
        {
            "variable_selection_mode": "explicit",
            "selected_variable_ids": ["v1", "v2"],
            "functions": [
                {"function": "relationship_scipy_pearson_pairwise"},
                {"function": "relationship_pearson_positive_lead_scan", "max_lag": 48},
            ],
        },
        truncate_first=True,
    )

    plan = (
        _intent_coordinator(gateway)
        .propose(
            question="分析 load、wind 与实时电价的同期和领先滞后关系",
            config_path=synthetic_study,
        )
        .plan
    )

    assert [step.function for step in plan.enabled_steps] == [
        "data_quality",
        "relationship_scipy_pearson_pairwise",
        "relationship_pearson_positive_lead_scan",
    ]
    assert [schema.__name__ for schema in gateway.schemas] == [
        "EDAPlanIntent",
        "MinimalEDAPlanIntent",
    ]
    assert plan.objective == "分析 load、wind 与实时电价的同期和领先滞后关系"
    recovery_payload = json.loads(gateway.messages[1][-1].content)
    assert "conversation_history" not in recovery_payload
    assert "quality_issues" not in recovery_payload


def test_non_quality_analysis_requires_a_function(synthetic_study: Path):
    gateway = _CompactIntentGateway(
        {
            "objective": "分析电价周期结构",
            "variable_selection_mode": "explicit",
            "functions": [],
        },
    )

    with pytest.raises(ResearchPlanValidationError, match="至少需要选择一个"):
        _intent_coordinator(gateway).propose(question="分析电价周期结构", config_path=synthetic_study)


def test_compact_auto_recommend_compiles_all_eligible_variables_into_a_screening_plan(
    synthetic_study: Path,
):
    gateway = _CompactIntentGateway(
        {
            "objective": "筛查值得深入研究的外生变量",
            "hypotheses": [],
            "variable_selection_mode": "auto_recommend",
            "functions": [{"function": "relationship_pearson_by_hour"}],
        },
    )

    plan = (
        _intent_coordinator(gateway)
        .propose(
            question="我不知道选什么外生变量，请先推荐",
            config_path=synthetic_study,
        )
        .plan
    )

    assert plan.variable_selection_stage == "screening"
    assert plan.selected_variables == ["load", "wind", "temperature"]
    assert plan.deferred_functions == ["relationship_pearson_by_hour"]
    assert "relationship_scipy_pearson_pairwise" in {step.function for step in plan.enabled_steps}


def test_invalid_model_plan_is_rejected_without_local_repair(synthetic_study: Path):
    model_planner = ModelEDAPlanner(
        gateway=FakeModelGateway(
            {
                "objective": "验证负荷与电价的同期关系",
                "variable_selection_mode": "explicit",
                "selected_variable_ids": ["v99"],
                "functions": [{"function": "relationship_scipy_pearson_pairwise"}],
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
            enabled_functions=[
                "relationship_scipy_pearson_pairwise",
                "relationship_pearson_positive_lead_scan",
            ],
            selected_variables=["wind"],
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
    assert [step.function for step in revised.enabled_steps] == [
        "data_quality",
        "relationship_scipy_pearson_pairwise",
        "relationship_pearson_positive_lead_scan",
    ]
    assert list(result.eda_summary["relationships"]["series"]) == ["wind"]
    assert set(result.figure_paths) == {"correlation_matrix", "correlation_ranking", "lag_relationships"}
    assert all(path.suffix == ".svg" and path.is_file() for path in result.figure_paths.values())
    assert result.report_path.name == "report.md"
    assert result.evaluation.decision == "accept"
    assert any(check.name == "时间序列混杂控制" and check.status == "warning" for check in result.evaluation.checks)
    assert not result.evaluation.feedback_packets

    package_files = {path.name for path in result.artifact_directory.iterdir()}
    assert {"report.md", "methods.md", "manifest.json"}.issubset(package_files)
    assert {"conversation.json", "research_plan.json", "execution_trace.json"}.issubset(
        {path.name for path in (result.artifact_directory / "provenance").iterdir()}
    )
    manifest = json.loads((result.artifact_directory / "manifest.json").read_text(encoding="utf-8"))
    trace = json.loads((result.artifact_directory / "provenance" / "execution_trace.json").read_text(encoding="utf-8"))
    assert all(item["started_at"] and item["finished_at"] for item in trace)
    assert all(
        datetime.fromisoformat(item["started_at"]) <= datetime.fromisoformat(item["finished_at"]) for item in trace
    )
    assert manifest["research_agent"]["planning_model"] == "scripted-test-model"
    assert manifest["research_agent"]["skill"] == {"name": "price-exogenous-eda", "version": "3.2.1"}
    assert manifest["research_agent"]["research_protocol"] == {
        "protocol_id": "electricity-price-evidence-ladder",
        "version": "2.2.1",
        "function_order": revised.research_protocol_function_order,
    }
    assert manifest["research_agent"]["function_versions"] == {step.function: "1.0.0" for step in revised.enabled_steps}
    for output in manifest["outputs"]:
        path = result.artifact_directory / output["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == output["sha256"]

    incomplete_summary = dict(result.eda_summary)
    incomplete_summary.pop("relationships")
    incomplete = evaluate_agent_run(
        plan=result.plan,
        quality=result.quality_report,
        summary=incomplete_summary,
    )
    packet = next(item for item in incomplete.feedback_packets if item.code.endswith("_fail"))
    assert incomplete.decision == "revise"
    assert next(check for check in incomplete.checks if check.name == "计划执行完整性").scope == "within_envelope"
    assert packet.recommendation == "补齐缺少的确定性结果步骤：relationships，然后重新执行。"


def test_auto_recommend_screens_all_eligible_variables_before_deep_analysis(
    synthetic_study: Path,
    tmp_path: Path,
):
    agent = _model_agent()
    proposal = agent.propose(question="先分析电价分布", config_path=synthetic_study)
    config = load_study_config(synthetic_study)
    quality = prepare_research_data(config).quality

    revised, _ = agent.main_agent.revise_plan(
        question="我不知道该选哪些外生变量，请先筛查后推荐",
        plan=proposal.plan,
        config=config,
        quality_report=quality,
        decision=DialogueDecision(
            intent="revise_plan",
            variable_selection_mode="auto_recommend",
            enabled_functions=[
                "relationship_scipy_pearson_pairwise",
                "relationship_pearson_by_hour",
                "relationship_pearson_by_month",
            ],
        ),
    )

    assert revised.variable_selection_mode == "auto_recommend"
    assert revised.variable_selection_stage == "screening"
    assert revised.selected_variables == ["load", "wind", "temperature"]
    assert set(revised.deferred_functions) == {
        "relationship_pearson_by_hour",
        "relationship_pearson_by_month",
    }
    assert {step.function for step in revised.enabled_steps} == {
        "data_quality",
        "exogenous_descriptive_distribution",
        "exogenous_pearson_collinearity",
        "exogenous_stationarity_tests",
        "relationship_scipy_pearson_pairwise",
        "relationship_scipy_spearman_pairwise",
    }

    result = agent.execute(
        plan=revised,
        config_path=synthetic_study,
        output_directory=tmp_path / "auto-variable-screening",
        run_id="auto-variable-screening",
    )

    recommendations = result.evaluation.variable_recommendations
    assert result.evaluation.decision == "need_user"
    assert recommendations is not None
    assert recommendations["screened_variable_count"] == 3
    assert "load" in recommendations["recommended_variables"]
    assert "外生变量自动筛查已完成" in result.evaluation.summary
    assert "深入方法尚未执行" in result.evaluation.summary
    assert "请确认、增删变量" not in result.evaluation.summary

    deep_plan, _ = agent.main_agent.revise_plan(
        question="接受推荐变量并继续深入分析",
        plan=revised,
        config=config,
        quality_report=quality,
        previous_evaluation=result.evaluation.model_dump(mode="json"),
        decision=DialogueDecision(
            intent="revise_plan",
            selected_variables=recommendations["recommended_variables"],
        ),
    )
    assert deep_plan.variable_selection_stage == "recommended"
    assert deep_plan.deferred_functions == []
    assert {
        "relationship_pearson_by_hour",
        "relationship_pearson_by_month",
    }.issubset({step.function for step in deep_plan.enabled_steps})


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


def _price_only_plan(study: Path, hypotheses: list[str]):
    agent = _model_agent()
    plan = agent.propose(question="分析电价分布", config_path=study).plan
    return agent, plan.model_copy(update={"hypotheses": hypotheses})


def test_the_real_evaluator_can_ask_for_an_in_envelope_revision(synthetic_study: Path, tmp_path: Path):
    """A lag scan that outruns its pairs is repairable inside the approved envelope."""

    from app.research.graph.guards import authorization_envelope, validate_automatic_revision
    from app.research.planning.compiler import EDAPlanCompiler
    from app.research.planning.contracts import EDAPlanDraft
    from app.research.skills.registry import SkillRegistry

    config = load_study_config(synthetic_study)
    skill = SkillRegistry.default(Settings(skill_paths="")).get("price-exogenous-eda")
    starving_lag = 715
    plan = EDAPlanCompiler().compile(
        EDAPlanDraft(
            objective="扫描 load 对电价的领先关系",
            selected_variables=["load"],
            steps=[
                {
                    "function": "relationship_pearson_positive_lead_scan",
                    "enabled": True,
                    "rationale": "定位领先滞后。",
                    "parameters": {"variables": ["load"], "max_lag": starving_lag},
                }
            ],
        ),
        question="load 对电价的领先滞后关系",
        config=config,
        skill=skill,
        model_name="regression",
        prompt_version="regression",
    )
    plan = plan.model_copy(update={"data_fingerprint": study_fingerprint(config, input_file_manifest(config))})
    result = EDAExecutionService().execute(
        plan=plan,
        study_config=config,
        output_directory=tmp_path / "revise-artifacts",
        run_id="within-envelope-revision",
    )

    check = next(item for item in result.evaluation.checks if item.name == "滞后扫描样本")
    assert check.scope == "within_envelope"
    assert result.evaluation.decision == "revise"
    packet = next(item for item in result.evaluation.feedback_packets if "滞后扫描" in item.message)
    assert packet.retryable and not packet.requires_user

    # The remediation it asks for is genuinely inside the approved envelope.
    envelope = authorization_envelope(plan, approved_at="2026-01-01T00:00:00+00:00")
    shrunk = plan.model_copy(
        update={
            "steps": [
                step.model_copy(update={"parameters": {**step.parameters, "max_lag": 24}})
                if "max_lag" in step.parameters
                else step
                for step in plan.steps
            ]
        }
    )
    assert validate_automatic_revision(shrunk, envelope) is None


def test_a_variable_without_any_usable_correlation_is_flagged_for_removal():
    from app.research.evaluation.eda import _variables_without_relationship_evidence

    relationships = {
        "load": {
            "contemporaneous": {"pearson": {"correlation": 0.7, "observations": 700}},
            "lag_profile": [{"lag": 0, "correlation": 0.7, "observations": 700}],
        },
        "flat": {
            "contemporaneous": {"pearson": {"correlation": None, "observations": 4}},
            "lag_profile": [{"lag": 0, "correlation": None, "observations": 4}],
        },
    }

    assert _variables_without_relationship_evidence(relationships) == ["flat"]


def test_a_hypothesis_no_rule_can_settle_blocks_instead_of_vanishing(synthetic_study: Path, tmp_path: Path):
    agent, plan = _price_only_plan(synthetic_study, ["碳配额成本可能传导到批发市场。"])
    result = agent.execute(
        plan=plan,
        config_path=synthetic_study,
        output_directory=tmp_path / "restatement-artifacts",
        run_id="agenda-restatement",
    )

    assessment = result.evaluation.hypothesis_assessments[0]
    assert assessment.status == "not_tested"
    assert assessment.scope == "needs_restatement"
    assert assessment.remediation
    assert result.evaluation.decision == "need_user"
    assert "改写" in result.evaluation.summary
    packet = next(item for item in result.evaluation.feedback_packets if "碳配额" in item.message)
    assert packet.requires_user and not packet.retryable


def test_inconclusive_stationarity_is_not_filed_as_a_method_limitation(synthetic_study: Path, tmp_path: Path):
    agent, plan = _price_only_plan(synthetic_study, ["电价可能存在单位根，建模前需要差分。"])
    result = agent.execute(
        plan=plan,
        config_path=synthetic_study,
        output_directory=tmp_path / "stationarity-artifacts",
        run_id="agenda-stationarity",
    )
    summary = {
        **result.eda_summary,
        "price": {
            **result.eda_summary["price"],
            "stationarity": {"verdict": "inconclusive", "recommended_transform": "none"},
        },
    }

    evaluation = evaluate_agent_run(plan=plan, quality=result.quality_report, summary=summary)
    assessment = evaluation.hypothesis_assessments[0]

    assert assessment.status == "inconclusive"
    assert assessment.scope == "needs_data"
    assert evaluation.decision == "need_user"


def test_calendar_evidence_settles_its_own_agenda_item(synthetic_study: Path, tmp_path: Path):
    agent = _model_agent()
    plan = agent.propose(question="分析电价季节性", config_path=synthetic_study).plan
    result = agent.execute(
        plan=plan,
        config_path=synthetic_study,
        output_directory=tmp_path / "calendar-artifacts",
        run_id="agenda-calendar",
    )

    assessment = next(item for item in result.evaluation.hypothesis_assessments if "季节" in item.hypothesis)
    assert assessment.status in {"candidate_support", "not_supported"}
    # Wording is reviewed separately; what matters here is that the calendar item is
    # settled by its own grouped evidence rather than by another item's numbers.
    assert assessment.item_id == "price.seasonality"
    assert "分组能解释" in assessment.evidence
    assert result.evaluation.decision == "accept"
    assert "全部得到明确结论" in result.evaluation.summary


def test_user_revision_keeps_new_hypotheses_and_the_new_function_agenda(synthetic_study: Path):
    from app.research.planning.compiler import FUNCTION_AGENDA_HYPOTHESES

    agent = _model_agent()
    config = load_study_config(synthetic_study)
    plan = agent.propose(question="分析电价季节性和尖峰", config_path=synthetic_study).plan
    added = FUNCTION_AGENDA_HYPOTHESES["price_stationarity_tests"]
    dropped = FUNCTION_AGENDA_HYPOTHESES["price_tukey_outer_fence"]
    assert dropped in plan.hypotheses

    revised, _ = agent.main_agent.revise_plan(
        question="加一个平稳性检验，并记下我关心的新问题",
        plan=plan,
        config=config,
        decision=DialogueDecision(
            intent="revise_plan",
            response="好的",
            hypotheses=["用户新提出：夏季检修可能推高日前价格。"],
            enabled_functions=["price_calendar_group_profile", "price_stationarity_tests"],
        ),
    )

    assert "price_stationarity_tests" in {step.function for step in revised.enabled_steps}
    assert "用户新提出：夏季检修可能推高日前价格。" in revised.hypotheses
    assert added in revised.hypotheses
    assert dropped not in revised.hypotheses


def test_unplanned_hypothesis_requires_approval_instead_of_auto_repair(synthetic_study: Path, tmp_path: Path):
    agent = _model_agent()
    plan = agent.propose(question="分析电价分布", config_path=synthetic_study).plan
    plan = plan.model_copy(update={"hypotheses": ["电价可能存在季节性结构。"]})

    result = agent.execute(
        plan=plan,
        config_path=synthetic_study,
        output_directory=tmp_path / "agenda-approval-artifacts",
        run_id="agent-agenda-approval",
    )

    assessment = result.evaluation.hypothesis_assessments[0]
    assert result.evaluation.decision == "need_user"
    assert assessment.status == "not_tested"
    assert assessment.scope == "needs_approval"
    assert any(packet.requires_user for packet in result.evaluation.feedback_packets)


def _move_the_target_data(synthetic_study: Path, *, by: float) -> None:
    config = load_study_config(synthetic_study)
    frame = pd.read_csv(config.target.path)
    frame.loc[0, config.target.value_column] = float(frame.loc[0, config.target.value_column]) + by
    frame.to_csv(config.target.path, index=False)


def test_an_approved_plan_runs_on_the_data_it_was_approved_for(synthetic_study: Path, tmp_path: Path):
    """A source that moves after approval must not move the conclusions."""

    proposal = _model_agent().propose(question="分析电价分布", config_path=synthetic_study)
    baseline = _model_agent().execute(
        plan=proposal.plan,
        config_path=synthetic_study,
        output_directory=tmp_path / "before-artifacts",
        run_id="frozen-before",
    )

    _move_the_target_data(synthetic_study, by=250.0)

    # A separate coordinator, so the answer comes from the frozen data rather than
    # from an in-process cache of the previous run.
    after = _model_agent().execute(
        plan=proposal.plan,
        config_path=synthetic_study,
        output_directory=tmp_path / "after-artifacts",
        run_id="frozen-after",
    )

    assert after.eda_summary == baseline.eda_summary
    assert after.quality_report == baseline.quality_report
    assert after.aligned_rows == baseline.aligned_rows


def test_execution_rejects_files_changed_after_plan_generation(synthetic_study: Path, tmp_path: Path):
    """Without the frozen copy there is nothing to fall back on, so the run stops."""

    agent = _model_agent()
    proposal = agent.propose(question="分析电价分布", config_path=synthetic_study)
    shutil.rmtree(snapshot_root(), ignore_errors=True)
    _move_the_target_data(synthetic_study, by=1.0)

    with pytest.raises(ResearchDataError, match="方案生成后发生变化"):
        agent.execute(
            plan=proposal.plan,
            config_path=synthetic_study,
            output_directory=tmp_path / "changed-input-artifacts",
        )


def test_execution_rejects_unregistered_atomic_function(synthetic_study: Path, tmp_path: Path):
    agent = _model_agent()
    plan = agent.propose(question="分析电价分布", config_path=synthetic_study).plan
    price = next(step for step in plan.steps if step.function == "price_descriptive_distribution")
    invalid_steps = [
        step.model_copy(update={"function": "price_profile"}) if step.step_id == price.step_id else step
        for step in plan.steps
    ]
    invalid = plan.model_copy(update={"steps": invalid_steps})

    with pytest.raises(ResearchPlanValidationError, match="工具未注册"):
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

    price = next(step for step in plan.steps if step.function == "price_calendar_group_profile")
    changed_steps = [
        step.model_copy(update={"function_version": "0.0.0"}) if step.step_id == price.step_id else step
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
