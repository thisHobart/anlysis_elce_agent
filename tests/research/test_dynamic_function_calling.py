"""Regression coverage for scope approval and the dynamic tool-selection loop."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.llm.gateway import ModelMessage, ModelToolCall, ModelToolTurn
from app.research.agent.dynamic import DynamicAnalysisAgent
from app.research.agent.errors import PlanCompatibilityError
from app.research.agent.orchestrator import DialogueDecision, MainResearchAgent
from app.research.agent.schemas import AgentEvaluation, EDAResearchScope, EvaluationCheck
from app.research.agent.subagents.eda import EDASubagent
from app.research.application.coordinator import ResearchCoordinator
from app.research.application.execution import EDAExecutionService
from app.research.data.inference import resolve_study_input
from app.research.graph.guards import scope_authorization_envelope, validate_scope_authorization
from app.research.graph.workflow import validate_analysis_message_protocol
from app.research.schemas.feedback import FeedbackPacket
from app.research.schemas.study import StudyInputDescriptor, load_study_config
from app.research.tools.eda.functions import build_eda_tool_registry
from app.research.tools.executor import ToolExecutionError
from app.research.tools.policy import ToolPermissionError


class ScopeDialogue:
    enabled = True
    model_name = "dynamic-test-model"

    def decide(self, **kwargs):
        if kwargs.get("summary"):
            return DialogueDecision(intent="explain_result", response="动态证据已完成。")
        if kwargs.get("config") is None:
            return DialogueDecision(intent="discussion", response="请提供电价数据。")
        return DialogueDecision(intent="new_plan", skill_name="price-exogenous-eda")


class RevisingScopeDialogue(ScopeDialogue):
    def decide(self, **kwargs):
        if "只保留电价分布" in kwargs.get("question", ""):
            assert kwargs.get("scope") is not None
            return DialogueDecision(
                intent="revise_plan",
                response="已收窄研究范围。",
                objective="只分析电价分布",
                enabled_functions=["price_descriptive_distribution"],
                selected_variables=[],
            )
        return super().decide(**kwargs)


class DeferredScopeDialogue(ScopeDialogue):
    def decide(self, **kwargs):
        if "概念" in kwargs.get("question", ""):
            return DialogueDecision(intent="discussion", response="这是一个电价领域概念。")
        if kwargs.get("has_executable_data"):
            return DialogueDecision(intent="new_plan", skill_name="price-exogenous-eda")
        return DialogueDecision(intent="discussion", response="请提供电价数据。")


class ScopePlanner:
    enabled = True
    model_name = "dynamic-test-model"

    def propose(self, _question, _config, _quality, **_kwargs):
        return {
            "objective": "动态分析电价结构",
            "hypotheses": [],
            "selected_variables": [],
            "steps": [
                {
                    "function": "price_descriptive_distribution",
                    "enabled": True,
                    "rationale": "先建立电价画像。",
                    "parameters": {},
                }
            ],
            "assumptions": [],
        }


class LimitedScopePlanner(ScopePlanner):
    def propose_scope(
        self,
        question,
        config,
        _quality,
        *,
        skill,
        data_fingerprint,
        **_kwargs,
    ):
        return EDAResearchScope(
            question=question,
            objective="验证动态调用预算",
            study_name=config.study.name,
            data_fingerprint=data_fingerprint,
            skill_name=skill.name,
            skill_version=skill.version,
            authorized_functions=[
                "price_descriptive_distribution",
                "price_calendar_group_profile",
                "price_rolling_mean_std",
            ],
            authorized_variables=[],
            initial_strategy=["先尝试快速画像。"],
            max_tool_calls=2,
        )


class SequenceToolGateway:
    enabled = True
    model_name = "dynamic-test-model"

    def __init__(self, turns: list[ModelToolTurn]) -> None:
        self.turns = list(turns)
        self.requests: list[tuple[list[ModelMessage], list[dict]]] = []

    def invoke_tool_turn(self, *, messages, tools, **_kwargs):
        self.requests.append((list(messages), list(tools)))
        return self.turns.pop(0)


def accepting_evaluation(**_kwargs):
    return AgentEvaluation(
        decision="accept",
        summary="动态证据通过确定性评估。",
        checks=[EvaluationCheck(name="完整性", status="pass", message="通过")],
    )


def _coordinator(gateway: SequenceToolGateway) -> ResearchCoordinator:
    tools = build_eda_tool_registry()
    return ResearchCoordinator(
        main_agent=MainResearchAgent(model_dialogue=ScopeDialogue()),
        eda_subagent=EDASubagent(model_planner=ScopePlanner()),
        tools=tools,
        dynamic_agent=DynamicAnalysisAgent(gateway=gateway, tools=tools),
    )


def _limited_coordinator(gateway: SequenceToolGateway) -> ResearchCoordinator:
    tools = build_eda_tool_registry()
    return ResearchCoordinator(
        main_agent=MainResearchAgent(model_dialogue=ScopeDialogue()),
        eda_subagent=EDASubagent(model_planner=LimitedScopePlanner()),
        tools=tools,
        dynamic_agent=DynamicAnalysisAgent(gateway=gateway, tools=tools),
    )


def test_scope_approval_recipe_expands_and_round_trips_one_provider_call(
    synthetic_study: Path,
    monkeypatch,
):
    monkeypatch.setattr(
        "app.research.application.execution.evaluate_agent_run",
        accepting_evaluation,
    )
    gateway = SequenceToolGateway(
        [
            ModelToolTurn(
                tool_calls=[
                    ModelToolCall(
                        name="price_quick_profile",
                        arguments={},
                        call_id="provider-recipe-1",
                    )
                ]
            ),
            ModelToolTurn(content="现有证据足够，可以完成评估。"),
        ]
    )
    coordinator = _coordinator(gateway)
    approval = coordinator.submit_user_message(
        session_id="dynamic-recipe",
        message="请整体分析这批电价数据",
        study_config=load_study_config(synthetic_study),
    )

    assert approval.interrupt is not None
    assert approval.interrupt.kind == "plan_approval"
    assert approval.interrupt.scope is not None
    assert approval.interrupt.plan is None
    assert approval.interrupt.scope["max_model_rounds"] == 8
    assert approval.interrupt.scope["max_tool_calls"] == 16
    assert "data_quality" not in approval.interrupt.scope["authorized_functions"]

    result = coordinator.resume(session_id="dynamic-recipe", action="approve")

    assert result.interrupt is not None
    assert result.interrupt.kind == "result", result.values.get("stop_reason")
    assert result.values["budget"]["model_rounds_used"] == 2
    assert result.values["budget"]["tool_calls_used"] == 3
    assert [step["function"] for step in result.values["current_plan"]["steps"]] == [
        "data_quality",
        "price_descriptive_distribution",
        "price_calendar_group_profile",
        "price_rolling_mean_std",
    ]
    group = result.values["provider_call_groups"]["provider-recipe-1"]
    assert group["requested_version"] == "1.0.0"
    assert group["origin"] == "recipe"
    assert len(group["child_call_ids"]) == 3
    assert len(result.values["call_evidence"]) == 4
    assert [item["origin"] for item in result.values["call_evidence"]] == [
        "system_preflight",
        "recipe",
        "recipe",
        "recipe",
    ]
    tool_messages = [
        message
        for message in result.values["analysis_messages"]
        if message["role"] == "tool"
    ]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "provider-recipe-1"
    assert len(gateway.requests) == 2
    assert any(
        tool["function"]["name"] == "price_quick_profile"
        for tool in gateway.requests[0][1]
    )
    relationship_schema = next(
        tool["function"]
        for tool in gateway.requests[0][1]
        if tool["function"]["name"] == "relationship_scipy_pearson_pairwise"
    )
    assert "min_observations" not in relationship_schema["parameters"]["properties"]
    assert "只可选择本次研究范围授权" in relationship_schema["parameters"]["properties"][
        "variables"
    ]["description"]


def test_identical_work_is_reused_and_reported_by_call_id(
    synthetic_study: Path,
    monkeypatch,
):
    monkeypatch.setattr(
        "app.research.application.execution.evaluate_agent_run",
        accepting_evaluation,
    )
    gateway = SequenceToolGateway(
        [
            ModelToolTurn(
                tool_calls=[
                    ModelToolCall(
                        name="price_descriptive_distribution",
                        arguments={},
                        call_id="provider-direct-1",
                    )
                ]
            ),
            ModelToolTurn(
                tool_calls=[
                    ModelToolCall(
                        name="price_descriptive_distribution",
                        arguments={},
                        call_id="provider-direct-2",
                    )
                ]
            ),
            ModelToolTurn(content="重复结果已复用，分析结束。"),
        ]
    )
    coordinator = _coordinator(gateway)
    approval = coordinator.submit_user_message(
        session_id="dynamic-reuse",
        message="分析电价分布，并核对重复调用能否复用",
        study_config=load_study_config(synthetic_study),
    )
    assert approval.interrupt is not None

    result = coordinator.resume(session_id="dynamic-reuse", action="approve")
    assert result.interrupt is not None, result.values
    assert result.interrupt.kind == "result", result.values.get("stop_reason")

    statuses = [
        item["status"]
        for item in result.values["call_evidence"]
        if item["function"] == "price_descriptive_distribution"
    ]
    assert statuses == ["completed", "reused"]
    runs = result.values["eda_summary"]["runs_by_function"][
        "price_descriptive_distribution"
    ]
    assert len(runs) == 2
    assert runs[0]["call_id"] != runs[1]["call_id"]
    figure_names = set(result.values["latest_run"]["figure_paths"])
    assert len([name for name in figure_names if name.startswith("price_distribution__")]) == 2
    methods = (
        Path(result.values["latest_run"]["artifact_directory"]) / "methods.md"
    ).read_text(encoding="utf-8")
    assert runs[0]["call_id"] in methods
    assert runs[1]["call_id"] in methods


def test_invalid_model_call_is_returned_as_a_matched_tool_error_then_repaired(
    synthetic_study: Path,
    monkeypatch,
):
    monkeypatch.setattr(
        "app.research.application.execution.evaluate_agent_run",
        accepting_evaluation,
    )
    gateway = SequenceToolGateway(
        [
            ModelToolTurn(
                tool_calls=[
                    ModelToolCall(
                        name="unknown_analysis_function",
                        arguments={},
                        call_id="provider-invalid-1",
                    )
                ]
            ),
            ModelToolTurn(
                tool_calls=[
                    ModelToolCall(
                        name="price_descriptive_distribution",
                        arguments={},
                        call_id="provider-repaired-1",
                    )
                ]
            ),
            ModelToolTurn(content="修复后的证据足够。"),
        ]
    )
    coordinator = _coordinator(gateway)
    approval = coordinator.submit_user_message(
        session_id="dynamic-repair",
        message="分析电价分布",
        study_config=load_study_config(synthetic_study),
    )
    assert approval.interrupt is not None

    result = coordinator.resume(session_id="dynamic-repair", action="approve")

    assert result.interrupt is not None
    assert result.interrupt.kind == "result", result.values.get("stop_reason")
    error_message = next(
        message
        for message in result.values["analysis_messages"]
        if message["role"] == "tool"
        and message["tool_call_id"] == "provider-invalid-1"
    )
    assert '"status": "rejected"' in error_message["content"]
    assert result.values["budget"]["tool_calls_used"] == 1


def test_invalid_proposal_does_not_cancel_an_independent_valid_call(
    synthetic_study: Path,
    monkeypatch,
):
    monkeypatch.setattr(
        "app.research.application.execution.evaluate_agent_run",
        accepting_evaluation,
    )
    gateway = SequenceToolGateway(
        [
            ModelToolTurn(
                tool_calls=[
                    ModelToolCall(name="unknown_analysis_function", arguments={}, call_id="mixed-bad"),
                    ModelToolCall(
                        name="price_descriptive_distribution",
                        arguments={},
                        call_id="mixed-good",
                    ),
                ]
            ),
            ModelToolTurn(content="独立的有效调用已经完成。"),
        ]
    )
    coordinator = _coordinator(gateway)
    approval = coordinator.submit_user_message(
        session_id="dynamic-mixed-compile",
        message="分析电价分布",
        study_config=load_study_config(synthetic_study),
    )

    result = coordinator.resume(session_id="dynamic-mixed-compile", action="approve")

    assert approval.interrupt is not None
    assert result.interrupt is not None and result.interrupt.kind == "result"
    tool_messages = {
        message["tool_call_id"]: json.loads(message["content"])
        for message in result.values["analysis_messages"]
        if message["role"] == "tool"
    }
    assert tool_messages["mixed-bad"]["status"] == "rejected"
    assert len(tool_messages["mixed-good"]["results"]) == 1
    assert result.values["budget"]["tool_calls_used"] == 1
    validate_analysis_message_protocol(
        result.values["analysis_messages"],
        result.values["provider_call_groups"],
        allow_pending_current_batch=False,
    )


def test_recipe_child_failure_is_closed_once_and_written_to_canonical_ledger(
    synthetic_study: Path,
    monkeypatch,
):
    monkeypatch.setattr(
        "app.research.application.execution.evaluate_agent_run",
        accepting_evaluation,
    )

    class FailingRecipeExecution(EDAExecutionService):
        def execute_scope_call(self, **kwargs):
            if kwargs["call"].name == "price_calendar_group_profile":
                raise ToolExecutionError("simulated recipe child failure")
            return super().execute_scope_call(**kwargs)

    tools = build_eda_tool_registry()
    gateway = SequenceToolGateway(
        [
            ModelToolTurn(
                tool_calls=[
                    ModelToolCall(name="price_quick_profile", arguments={}, call_id="failed-recipe")
                ]
            ),
            ModelToolTurn(content="保留成功证据并结束。"),
        ]
    )
    coordinator = ResearchCoordinator(
        main_agent=MainResearchAgent(model_dialogue=ScopeDialogue()),
        eda_subagent=EDASubagent(model_planner=ScopePlanner()),
        execution=FailingRecipeExecution(registry=tools),
        tools=tools,
        dynamic_agent=DynamicAnalysisAgent(gateway=gateway, tools=tools),
    )
    approval = coordinator.submit_user_message(
        session_id="dynamic-recipe-child-error",
        message="整体分析电价",
        study_config=load_study_config(synthetic_study),
    )

    result = coordinator.resume(session_id="dynamic-recipe-child-error", action="approve")

    assert approval.interrupt is not None
    assert result.interrupt is not None and result.interrupt.kind == "result"
    recipe_messages = [
        message
        for message in result.values["analysis_messages"]
        if message["role"] == "tool" and message["tool_call_id"] == "failed-recipe"
    ]
    assert len(recipe_messages) == 1
    recipe_payload = json.loads(recipe_messages[0]["content"])
    assert [item["status"] for item in recipe_payload["errors"]] == ["failed", "cancelled"]
    assert recipe_payload["errors"][1]["code"] == "cancelled_due_to_batch_abort"
    ledger_path = (
        Path(result.values["latest_run"]["artifact_directory"])
        / "evidence"
        / "call_evidence.json"
    )
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    statuses = [ledger["calls"][call_id]["status"] for call_id in ledger["call_order"]]
    assert statuses == ["completed", "completed", "failed", "cancelled"]
    cancelled_id = ledger["call_order"][-1]
    assert ledger["calls"][cancelled_id]["error"]["code"] == "cancelled_due_to_batch_abort"
    report = Path(result.values["latest_run"]["report_path"]).read_text(encoding="utf-8")
    assert "失败或取消的调用" in report


def test_fatal_error_closes_and_cancels_every_provider_call_in_the_batch(
    synthetic_study: Path,
):
    class FatalExecution(EDAExecutionService):
        def execute_scope_call(self, **kwargs):
            if kwargs["call"].name != "data_quality":
                raise ToolPermissionError("simulated authorization failure")
            return super().execute_scope_call(**kwargs)

    tools = build_eda_tool_registry()
    gateway = SequenceToolGateway(
        [
            ModelToolTurn(
                tool_calls=[
                    ModelToolCall(
                        name="price_descriptive_distribution",
                        arguments={},
                        call_id="fatal-first",
                    ),
                    ModelToolCall(
                        name="price_calendar_group_profile",
                        arguments={},
                        call_id="fatal-second",
                    ),
                ]
            )
        ]
    )
    coordinator = ResearchCoordinator(
        main_agent=MainResearchAgent(model_dialogue=ScopeDialogue()),
        eda_subagent=EDASubagent(model_planner=ScopePlanner()),
        execution=FatalExecution(registry=tools),
        tools=tools,
        dynamic_agent=DynamicAnalysisAgent(gateway=gateway, tools=tools),
    )
    approval = coordinator.submit_user_message(
        session_id="dynamic-fatal-batch",
        message="分析电价",
        study_config=load_study_config(synthetic_study),
    )

    interrupted = coordinator.resume(session_id="dynamic-fatal-batch", action="approve")

    assert approval.interrupt is not None
    assert interrupted.interrupt is not None and interrupted.interrupt.kind == "plan_error"
    dynamic_records = [
        record
        for record in interrupted.values["tool_records"].values()
        if record["call"]["name"] != "data_quality"
    ]
    assert [record["status"] for record in dynamic_records] == ["failed", "cancelled"]
    tool_payloads = {
        message["tool_call_id"]: json.loads(message["content"])
        for message in interrupted.values["analysis_messages"]
        if message["role"] == "tool"
    }
    assert tool_payloads["fatal-first"]["errors"][0]["status"] == "failed"
    assert tool_payloads["fatal-second"]["errors"][0]["code"] == "cancelled_due_to_batch_abort"
    validate_analysis_message_protocol(
        interrupted.values["analysis_messages"],
        interrupted.values["provider_call_groups"],
        allow_pending_current_batch=False,
    )


def test_recipe_is_rejected_before_execution_when_atomic_budget_is_too_small(
    synthetic_study: Path,
):
    gateway = SequenceToolGateway(
        [
            ModelToolTurn(
                tool_calls=[
                    ModelToolCall(
                        name="price_quick_profile",
                        arguments={},
                        call_id="provider-over-budget-1",
                    )
                ]
            )
        ]
    )
    coordinator = _limited_coordinator(gateway)
    approval = coordinator.submit_user_message(
        session_id="dynamic-budget",
        message="整体分析电价",
        study_config=load_study_config(synthetic_study),
    )
    assert approval.interrupt is not None

    limited = coordinator.resume(session_id="dynamic-budget", action="approve")

    assert limited.interrupt is not None
    assert limited.interrupt.kind == "result_limitations"
    assert limited.values["budget"]["tool_calls_used"] == 0
    assert len(limited.values["call_evidence"]) == 1
    matched = next(
        message
        for message in limited.values["analysis_messages"]
        if message["role"] == "tool"
    )
    assert matched["tool_call_id"] == "provider-over-budget-1"
    assert '"status": "rejected"' in matched["content"]


def test_user_can_narrow_scope_before_reapproving(synthetic_study: Path):
    tools = build_eda_tool_registry()
    gateway = SequenceToolGateway([])
    coordinator = ResearchCoordinator(
        main_agent=MainResearchAgent(model_dialogue=RevisingScopeDialogue()),
        eda_subagent=EDASubagent(model_planner=ScopePlanner()),
        tools=tools,
        dynamic_agent=DynamicAnalysisAgent(gateway=gateway, tools=tools),
    )
    first = coordinator.submit_user_message(
        session_id="dynamic-revise-scope",
        message="请分析这批电价数据",
        study_config=load_study_config(synthetic_study),
    )
    assert first.interrupt is not None
    first_scope = first.interrupt.scope

    revised = coordinator.resume(
        session_id="dynamic-revise-scope",
        action="modify",
        message="只保留电价分布，不分析外生变量",
    )

    assert revised.interrupt is not None
    assert revised.interrupt.kind == "plan_approval"
    assert revised.interrupt.scope is not None
    assert revised.interrupt.scope["scope_id"] != first_scope["scope_id"]
    assert revised.interrupt.scope["revision"] == first_scope["revision"] + 1
    assert revised.interrupt.scope["objective"] == "只分析电价分布"
    assert revised.interrupt.scope["authorized_functions"] == [
        "price_descriptive_distribution"
    ]
    assert revised.interrupt.scope["authorized_variables"] == []


def test_same_function_with_different_parameters_keeps_distinct_evidence(
    synthetic_study: Path,
    monkeypatch,
):
    monkeypatch.setattr(
        "app.research.application.execution.evaluate_agent_run",
        accepting_evaluation,
    )
    gateway = SequenceToolGateway(
        [
            ModelToolTurn(
                tool_calls=[
                    ModelToolCall(
                        name="price_lag_autocorrelation",
                        arguments={"max_lag": 4},
                        call_id="provider-lag-4",
                    )
                ]
            ),
            ModelToolTurn(
                tool_calls=[
                    ModelToolCall(
                        name="price_lag_autocorrelation",
                        arguments={"max_lag": 8},
                        call_id="provider-lag-8",
                    )
                ]
            ),
            ModelToolTurn(content="两个滞后范围都已完成。"),
        ]
    )
    coordinator = _coordinator(gateway)
    approval = coordinator.submit_user_message(
        session_id="dynamic-distinct-parameters",
        message="分别观察4阶和8阶电价自相关",
        study_config=load_study_config(synthetic_study),
    )
    assert approval.interrupt is not None

    result = coordinator.resume(
        session_id="dynamic-distinct-parameters",
        action="approve",
    )

    assert result.interrupt is not None
    assert result.interrupt.kind == "result", result.values.get("stop_reason")
    runs = result.values["eda_summary"]["runs_by_function"][
        "price_lag_autocorrelation"
    ]
    assert [run["arguments"]["max_lag"] for run in runs] == [4, 8]
    assert len({run["work_id"] for run in runs}) == 2
    assert all(run["status"] == "completed" for run in runs)
    figure_names = set(result.values["latest_run"]["figure_paths"])
    assert len([name for name in figure_names if name.startswith("price_autocorrelation__")]) == 2


def test_deterministic_evaluation_can_send_the_model_back_for_more_evidence(
    synthetic_study: Path,
    monkeypatch,
):
    evaluations = 0

    def evaluate_twice(**_kwargs):
        nonlocal evaluations
        evaluations += 1
        if evaluations == 1:
            return AgentEvaluation(
                decision="revise",
                summary="还需要补充日历结构证据。",
                checks=[
                    EvaluationCheck(
                        name="日历结构",
                        status="warning",
                        message="日历结构尚未核验。",
                        scope="within_envelope",
                        remediable=True,
                    )
                ],
                feedback_packets=[
                    FeedbackPacket(
                        source="evaluator",
                        code="need_calendar_profile",
                        severity="warning",
                        message="补充日历结构证据。",
                        retryable=True,
                    )
                ],
            )
        return accepting_evaluation()

    monkeypatch.setattr(
        "app.research.application.execution.evaluate_agent_run",
        evaluate_twice,
    )
    gateway = SequenceToolGateway(
        [
            ModelToolTurn(
                tool_calls=[
                    ModelToolCall(
                        name="price_descriptive_distribution",
                        arguments={},
                        call_id="provider-first-evidence",
                    )
                ]
            ),
            ModelToolTurn(content="先提交现有证据评估。"),
            ModelToolTurn(
                tool_calls=[
                    ModelToolCall(
                        name="price_calendar_group_profile",
                        arguments={},
                        call_id="provider-followup-evidence",
                    )
                ]
            ),
            ModelToolTurn(content="补充证据已完成。"),
        ]
    )
    coordinator = _coordinator(gateway)
    approval = coordinator.submit_user_message(
        session_id="dynamic-evaluation-revision",
        message="分析电价分布与日历结构",
        study_config=load_study_config(synthetic_study),
    )
    assert approval.interrupt is not None

    result = coordinator.resume(
        session_id="dynamic-evaluation-revision",
        action="approve",
    )

    assert result.interrupt is not None
    assert result.interrupt.kind == "result", result.values.get("stop_reason")
    assert evaluations == 2
    assert len(result.values["run_history"]) == 2
    assert result.values["budget"]["model_rounds_used"] == 4
    assert result.values["budget"]["tool_calls_used"] == 2
    assert any(
        message["role"] == "user"
        and "deterministic_evaluation_feedback" in message["content"]
        for message in result.values["analysis_messages"]
    )


def test_two_consecutive_duplicate_rounds_pause_for_the_user(synthetic_study: Path):
    gateway = SequenceToolGateway(
        [
            ModelToolTurn(
                tool_calls=[
                    ModelToolCall(
                        name="price_descriptive_distribution",
                        arguments={},
                        call_id="provider-original",
                    )
                ]
            ),
            ModelToolTurn(
                tool_calls=[
                    ModelToolCall(
                        name="price_descriptive_distribution",
                        arguments={},
                        call_id="provider-duplicate-1",
                    )
                ]
            ),
            ModelToolTurn(
                tool_calls=[
                    ModelToolCall(
                        name="price_descriptive_distribution",
                        arguments={},
                        call_id="provider-duplicate-2",
                    )
                ]
            ),
        ]
    )
    coordinator = _coordinator(gateway)
    approval = coordinator.submit_user_message(
        session_id="dynamic-duplicate-pause",
        message="分析电价分布",
        study_config=load_study_config(synthetic_study),
    )
    assert approval.interrupt is not None

    limited = coordinator.resume(
        session_id="dynamic-duplicate-pause",
        action="approve",
    )

    assert limited.interrupt is not None
    assert limited.interrupt.kind == "result_limitations"
    assert limited.values["budget"]["no_progress_rounds"] == 2
    assert limited.values["budget"]["tool_calls_used"] == 3
    duplicate_messages = [
        message["content"]
        for message in limited.values["analysis_messages"]
        if message["role"] == "tool" and "duplicate_work_id" in message["content"]
    ]
    assert len(duplicate_messages) == 2


def test_selected_data_is_not_parsed_until_the_main_agent_chooses_analysis(
    synthetic_study: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config = load_study_config(synthetic_study)
    descriptor = StudyInputDescriptor(
        target_path=config.target.path,
        output_directory=config.analysis.output_directory,
    )

    def forbidden_parse(_descriptor):
        raise AssertionError("discussion must not parse the selected data")

    monkeypatch.setattr("app.research.graph.workflow.resolve_study_input", forbidden_parse)
    tools = build_eda_tool_registry()
    coordinator = ResearchCoordinator(
        main_agent=MainResearchAgent(model_dialogue=DeferredScopeDialogue()),
        eda_subagent=EDASubagent(model_planner=ScopePlanner()),
        tools=tools,
        dynamic_agent=DynamicAnalysisAgent(gateway=SequenceToolGateway([]), tools=tools),
    )

    discussed = coordinator.submit_user_message(
        session_id="deferred-data-discussion",
        message="解释一个电价概念",
        study_input=descriptor,
        route_before_graph=True,
    )

    assert discussed.interrupt is not None
    assert discussed.interrupt.kind == "result"
    assert discussed.values["study_config"] is None
    assert discussed.values["has_executable_data"] is True
    assert discussed.values["direct_dialogue"] is True
    assert not coordinator.has_thread("deferred-data-discussion")


def test_analysis_route_resolves_selected_data_exactly_once(
    synthetic_study: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config = load_study_config(synthetic_study)
    descriptor = StudyInputDescriptor(
        target_path=config.target.path,
        output_directory=config.analysis.output_directory,
    )
    calls = 0

    def counted_parse(value):
        nonlocal calls
        calls += 1
        return resolve_study_input(value)

    monkeypatch.setattr("app.research.graph.workflow.resolve_study_input", counted_parse)
    tools = build_eda_tool_registry()
    coordinator = ResearchCoordinator(
        main_agent=MainResearchAgent(model_dialogue=DeferredScopeDialogue()),
        eda_subagent=EDASubagent(model_planner=ScopePlanner()),
        tools=tools,
        dynamic_agent=DynamicAnalysisAgent(gateway=SequenceToolGateway([]), tools=tools),
    )

    approval = coordinator.submit_user_message(
        session_id="deferred-data-analysis",
        message="分析这批实际电价数据",
        study_input=descriptor,
    )

    assert approval.interrupt is not None
    assert approval.interrupt.kind == "plan_approval"
    assert calls == 1
    assert approval.values["study_config"] is not None


def test_invalid_selected_data_is_reported_only_after_analysis_routing(tmp_path: Path):
    broken = tmp_path / "broken.csv"
    broken.write_text("not_a_timestamp,not_a_number\nhello,world\n", encoding="utf-8")
    descriptor = StudyInputDescriptor(target_path=broken, output_directory=tmp_path / "artifacts")
    tools = build_eda_tool_registry()
    coordinator = ResearchCoordinator(
        main_agent=MainResearchAgent(model_dialogue=DeferredScopeDialogue()),
        eda_subagent=EDASubagent(model_planner=ScopePlanner()),
        tools=tools,
        dynamic_agent=DynamicAnalysisAgent(gateway=SequenceToolGateway([]), tools=tools),
    )

    failed = coordinator.submit_user_message(
        session_id="deferred-invalid-data",
        message="分析这批实际电价数据",
        study_input=descriptor,
    )

    assert failed.interrupt is not None
    assert failed.interrupt.kind == "data_input_error"
    assert failed.values["research_scope"] is None
    assert failed.values["tool_results"] == []


def test_duplicate_provider_ids_are_rejected_before_assistant_history_is_committed(
    synthetic_study: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("app.research.application.execution.evaluate_agent_run", accepting_evaluation)
    gateway = SequenceToolGateway(
        [
            ModelToolTurn(
                tool_calls=[
                    ModelToolCall(name="price_descriptive_distribution", arguments={}, call_id="duplicate-id"),
                    ModelToolCall(name="price_calendar_group_profile", arguments={}, call_id="duplicate-id"),
                ]
            ),
            ModelToolTurn(
                tool_calls=[
                    ModelToolCall(name="price_descriptive_distribution", arguments={}, call_id="repaired-id")
                ]
            ),
            ModelToolTurn(content="协议修复后结束。"),
        ]
    )
    coordinator = _coordinator(gateway)
    coordinator.submit_user_message(
        session_id="duplicate-provider-id",
        message="分析电价分布",
        study_config=load_study_config(synthetic_study),
    )

    result = coordinator.resume(session_id="duplicate-provider-id", action="approve")

    assert result.interrupt is not None and result.interrupt.kind == "result"
    assistant_ids = [
        str(call["call_id"])
        for message in result.values["analysis_messages"]
        if message["role"] == "assistant"
        for call in message.get("tool_calls", [])
    ]
    assert assistant_ids == ["repaired-id"]
    assert result.values["budget"]["model_rounds_used"] == 3
    assert result.values["budget"]["tool_calls_used"] == 1
    validate_analysis_message_protocol(
        result.values["analysis_messages"],
        result.values["provider_call_groups"],
        allow_pending_current_batch=False,
    )


def test_reused_provider_id_is_rejected_without_creating_an_orphan_call(
    synthetic_study: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("app.research.application.execution.evaluate_agent_run", accepting_evaluation)
    gateway = SequenceToolGateway(
        [
            ModelToolTurn(
                tool_calls=[
                    ModelToolCall(name="price_descriptive_distribution", arguments={}, call_id="provider-stable")
                ]
            ),
            ModelToolTurn(
                tool_calls=[
                    ModelToolCall(name="price_calendar_group_profile", arguments={}, call_id="provider-stable")
                ]
            ),
            ModelToolTurn(
                tool_calls=[
                    ModelToolCall(name="price_calendar_group_profile", arguments={}, call_id="provider-new")
                ]
            ),
            ModelToolTurn(content="完成。"),
        ]
    )
    coordinator = _coordinator(gateway)
    coordinator.submit_user_message(
        session_id="reused-provider-id",
        message="分析电价分布和日历结构",
        study_config=load_study_config(synthetic_study),
    )

    result = coordinator.resume(session_id="reused-provider-id", action="approve")

    assert result.interrupt is not None and result.interrupt.kind == "result"
    tool_ids = [
        message["tool_call_id"]
        for message in result.values["analysis_messages"]
        if message["role"] == "tool"
    ]
    assert tool_ids == ["provider-stable", "provider-new"]
    validate_analysis_message_protocol(
        result.values["analysis_messages"],
        result.values["provider_call_groups"],
        allow_pending_current_batch=False,
    )


def test_two_invalid_provider_id_rounds_pause_without_spending_tool_budget(synthetic_study: Path):
    invalid_turn = ModelToolTurn(
        tool_calls=[
            ModelToolCall(name="price_descriptive_distribution", arguments={}, call_id="same-id"),
            ModelToolCall(name="price_calendar_group_profile", arguments={}, call_id="same-id"),
        ]
    )
    coordinator = _coordinator(SequenceToolGateway([invalid_turn, invalid_turn]))
    coordinator.submit_user_message(
        session_id="invalid-provider-id-limit",
        message="分析电价",
        study_config=load_study_config(synthetic_study),
    )

    stopped = coordinator.resume(session_id="invalid-provider-id-limit", action="approve")

    assert stopped.interrupt is not None and stopped.interrupt.kind == "response_error"
    assert stopped.values["budget"]["model_rounds_used"] == 2
    assert stopped.values["budget"]["tool_calls_used"] == 0
    assert not any(
        message["role"] == "assistant" and message.get("tool_calls")
        for message in stopped.values["analysis_messages"]
    )


@pytest.mark.parametrize(
    "update",
    [
        {"data_fingerprint": "f" * 12},
        {"skill_version": "999.0.0"},
        {"authorized_functions": ["price_descriptive_distribution"]},
        {"authorized_variables": ["outside"]},
        {"max_model_rounds": 9},
        {"max_tool_calls": 17},
        {"max_attempts_per_call": 3},
    ],
)
def test_dynamic_authorization_rejects_every_approved_boundary_mutation(update: dict):
    scope = EDAResearchScope(
        question="分析电价",
        objective="验证授权边界",
        study_name="authorization-study",
        data_fingerprint="a" * 12,
        skill_name="price-exogenous-eda",
        skill_version="2.0.0",
        authorized_functions=["price_descriptive_distribution", "price_calendar_group_profile"],
        authorized_variables=[],
        initial_strategy=["先看分布。"],
    )
    envelope = scope_authorization_envelope(scope, approved_at="2026-01-01T00:00:00+00:00")

    with pytest.raises(PlanCompatibilityError):
        validate_scope_authorization(scope.model_copy(update=update), envelope)


def test_dynamic_compiler_rejects_out_of_scope_function_and_variable(synthetic_study: Path):
    config = load_study_config(synthetic_study)
    scope = EDAResearchScope(
        question="分析电价",
        objective="验证编译权限",
        study_name=config.study.name,
        data_fingerprint="a" * 12,
        skill_name="price-exogenous-eda",
        skill_version="2.0.0",
        authorized_functions=["price_descriptive_distribution", "exogenous_descriptive_distribution"],
        authorized_variables=[],
        initial_strategy=["只在范围内调用。"],
    )
    execution = EDAExecutionService()

    with pytest.raises(ToolPermissionError):
        execution.compile_dynamic_call(
            scope=scope,
            study_config=config,
            name="price_calendar_group_profile",
            arguments={},
            sequence=2,
        )
    with pytest.raises(ToolPermissionError):
        execution.compile_dynamic_call(
            scope=scope,
            study_config=config,
            name="exogenous_descriptive_distribution",
            arguments={"variables": [config.exogenous[0].name]},
            sequence=2,
        )


@pytest.mark.parametrize("crash_on", [1, 2])
def test_dynamic_batch_recovers_from_sqlite_without_reexecuting_completed_children(
    synthetic_study: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crash_on: int,
):
    monkeypatch.setattr("app.research.application.execution.evaluate_agent_run", accepting_evaluation)

    class CrashingDynamicExecution(EDAExecutionService):
        calls = 0

        def execute_scope_call(self, **kwargs):
            if kwargs["call"].name != "data_quality":
                self.calls += 1
                if self.calls == crash_on:
                    raise KeyboardInterrupt("simulated dynamic process crash")
            return super().execute_scope_call(**kwargs)

    database = tmp_path / "dynamic-checkpoint.sqlite3"
    tools = build_eda_tool_registry()
    first_gateway = SequenceToolGateway(
        [
            ModelToolTurn(
                tool_calls=[
                    ModelToolCall(name="price_quick_profile", arguments={}, call_id="checkpoint-recipe")
                ]
            )
        ]
    )
    first = ResearchCoordinator(
        main_agent=MainResearchAgent(model_dialogue=ScopeDialogue()),
        eda_subagent=EDASubagent(model_planner=ScopePlanner()),
        execution=CrashingDynamicExecution(registry=tools),
        tools=tools,
        dynamic_agent=DynamicAnalysisAgent(gateway=first_gateway, tools=tools),
        checkpoint_path=database,
    )
    first.submit_user_message(
        session_id=f"dynamic-checkpoint-{crash_on}",
        message="整体分析电价",
        study_config=load_study_config(synthetic_study),
    )
    with pytest.raises(KeyboardInterrupt, match="simulated dynamic process crash"):
        first.resume(session_id=f"dynamic-checkpoint-{crash_on}", action="approve")
    crashed = first.get_snapshot(f"dynamic-checkpoint-{crash_on}")
    validate_analysis_message_protocol(
        crashed.values["analysis_messages"],
        crashed.values["provider_call_groups"],
        allow_pending_current_batch=True,
    )
    first.close()

    restored_tools = build_eda_tool_registry()
    restored = ResearchCoordinator(
        main_agent=MainResearchAgent(model_dialogue=ScopeDialogue()),
        eda_subagent=EDASubagent(model_planner=ScopePlanner()),
        tools=restored_tools,
        dynamic_agent=DynamicAnalysisAgent(
            gateway=SequenceToolGateway([ModelToolTurn(content="恢复后完成。")]),
            tools=restored_tools,
        ),
        checkpoint_path=database,
    )
    result = restored.continue_thread(f"dynamic-checkpoint-{crash_on}")

    assert result.interrupt is not None and result.interrupt.kind == "result"
    assert len(result.values["call_evidence"]) == 4
    recipe_messages = [
        message
        for message in result.values["analysis_messages"]
        if message["role"] == "tool" and message["tool_call_id"] == "checkpoint-recipe"
    ]
    assert len(recipe_messages) == 1
    assert len(json.loads(recipe_messages[0]["content"])["results"]) == 3
    attempts = sorted(
        record["attempts"]
        for record in result.values["tool_records"].values()
        if record["call"]["name"] != "data_quality"
    )
    assert attempts == [1, 1, 2]
    validate_analysis_message_protocol(
        result.values["analysis_messages"],
        result.values["provider_call_groups"],
        allow_pending_current_batch=False,
    )
    restored.close()


def test_dynamic_checkpoint_resumes_after_a_closed_batch_before_next_selection(
    synthetic_study: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("app.research.application.execution.evaluate_agent_run", accepting_evaluation)

    class CrashBeforeSecondTurnGateway(SequenceToolGateway):
        def invoke_tool_turn(self, *, messages, tools, **kwargs):
            if self.requests:
                raise KeyboardInterrupt("simulated selector crash")
            return super().invoke_tool_turn(messages=messages, tools=tools, **kwargs)

    database = tmp_path / "dynamic-selector-checkpoint.sqlite3"
    tools = build_eda_tool_registry()
    first = ResearchCoordinator(
        main_agent=MainResearchAgent(model_dialogue=ScopeDialogue()),
        eda_subagent=EDASubagent(model_planner=ScopePlanner()),
        tools=tools,
        dynamic_agent=DynamicAnalysisAgent(
            gateway=CrashBeforeSecondTurnGateway(
                [
                    ModelToolTurn(
                        tool_calls=[
                            ModelToolCall(
                                name="price_descriptive_distribution",
                                arguments={},
                                call_id="selector-checkpoint-call",
                            )
                        ]
                    )
                ]
            ),
            tools=tools,
        ),
        checkpoint_path=database,
    )
    first.submit_user_message(
        session_id="dynamic-selector-checkpoint",
        message="分析电价分布",
        study_config=load_study_config(synthetic_study),
    )
    with pytest.raises(KeyboardInterrupt, match="simulated selector crash"):
        first.resume(session_id="dynamic-selector-checkpoint", action="approve")
    closed = first.get_snapshot("dynamic-selector-checkpoint")
    validate_analysis_message_protocol(
        closed.values["analysis_messages"],
        closed.values["provider_call_groups"],
        allow_pending_current_batch=False,
    )
    first.close()

    restored_tools = build_eda_tool_registry()
    restored = ResearchCoordinator(
        main_agent=MainResearchAgent(model_dialogue=ScopeDialogue()),
        eda_subagent=EDASubagent(model_planner=ScopePlanner()),
        tools=restored_tools,
        dynamic_agent=DynamicAnalysisAgent(
            gateway=SequenceToolGateway([ModelToolTurn(content="恢复后完成。")]),
            tools=restored_tools,
        ),
        checkpoint_path=database,
    )
    result = restored.continue_thread("dynamic-selector-checkpoint")

    assert result.interrupt is not None and result.interrupt.kind == "result"
    assert result.values["budget"]["tool_calls_used"] == 1
    assert len(
        [
            item
            for item in result.values["call_evidence"]
            if item["function"] == "price_descriptive_distribution"
        ]
    ) == 1
    restored.close()
