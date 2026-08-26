"""Behavior tests for the persistent, bounded LangGraph research loop."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from app.research.agent.orchestrator import DialogueDecision, MainResearchAgent
from app.research.agent.schemas import AgentEvaluation, EvaluationCheck
from app.research.agent.subagents.eda import EDASubagent
from app.research.application.coordinator import GRAPH_SCHEMA_VERSION, ResearchCoordinator
from app.research.application.execution import EDAExecutionService
from app.research.graph.contracts import ApprovalState, LoopBudget
from app.research.schemas.feedback import FeedbackPacket
from app.research.schemas.study import load_study_config


class LoopDialogue:
    enabled = True
    model_name = "loop-test-model"

    def decide(self, **kwargs):
        if kwargs.get("summary"):
            return DialogueDecision(intent="explain_result", response="只根据已校验证据解释结果和限制。")
        if kwargs.get("config") is None:
            return DialogueDecision(intent="discussion", response="请先明确数据和时间范围。")
        return DialogueDecision(intent="new_plan", skill_name="price-exogenous-eda")


class LoopPlanner:
    enabled = True
    model_name = "loop-test-model"

    def propose(self, _question, _config, _quality, history=None, skill=None, feedback=None):
        del history
        assert skill is not None
        functions = (
            ["price_descriptive_distribution"]
            if feedback
            else ["price_descriptive_distribution", "price_calendar_group_profile"]
        )
        return {
            "objective": "验证电价结构",
            "hypotheses": ["电价可能存在稳定结构。"],
            "selected_variables": [],
            "steps": [
                {
                    "function": function_name,
                    "enabled": True,
                    "rationale": "执行受控电价画像。",
                    "parameters": {},
                }
                for function_name in functions
            ],
            "assumptions": [],
        }


class FourToolPlanner:
    """Produce a four-call EDA queue for multi-iteration loop tests."""

    enabled = True
    model_name = "loop-test-model"

    def __init__(self) -> None:
        self.calls = 0

    def propose(self, _question, config, _quality, history=None, skill=None, feedback=None):
        del history, feedback
        assert skill is not None
        self.calls += 1
        variables = [spec.name for spec in config.exogenous[:2]]
        return {
            "objective": f"验证电价结构第 {self.calls} 轮",
            "hypotheses": ["电价与所选变量可能存在描述性关系。"],
            "selected_variables": variables,
            "steps": [
                {
                    "function": "price_descriptive_distribution",
                    "enabled": True,
                    "rationale": "检查电价分布。",
                    "parameters": {},
                },
                {
                    "function": "exogenous_descriptive_distribution",
                    "enabled": True,
                    "rationale": "检查外生变量画像。",
                    "parameters": {"variables": variables},
                },
                {
                    "function": "relationship_scipy_pearson_pairwise",
                    "enabled": True,
                    "rationale": "检查同期关系。",
                    "parameters": {"variables": variables},
                },
            ],
            "assumptions": [],
        }


class NeedUserDialogue(LoopDialogue):
    def decide(self, **kwargs):
        question = kwargs.get("question", "")
        if "修改" in question:
            return DialogueDecision(
                intent="revise_plan",
                response="已生成用户要求的收缩方案。",
                objective="仅保留电价画像",
                enabled_functions=["price_descriptive_distribution"],
                selected_variables=[],
                max_lag=4,
            )
        return super().decide(**kwargs)


def loop_coordinator(**kwargs) -> ResearchCoordinator:
    return ResearchCoordinator(
        main_agent=MainResearchAgent(model_dialogue=LoopDialogue()),
        eda_subagent=EDASubagent(model_planner=LoopPlanner()),
        **kwargs,
    )


def accepting_evaluation(*, plan, quality, summary):
    del plan, quality, summary
    return AgentEvaluation(
        decision="accept",
        summary="证据通过确定性评估。",
        checks=[EvaluationCheck(name="完整性", status="pass", message="通过")],
    )


def test_one_graph_runs_approval_tools_evaluation_and_followup(
    synthetic_study: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("app.research.application.execution.evaluate_agent_run", accepting_evaluation)
    coordinator = loop_coordinator()
    config = load_study_config(synthetic_study)
    thread_id = "loop-happy"

    approval = coordinator.submit_user_message(
        session_id=thread_id,
        message="分析电价分布和季节性",
        study_config=config,
    )
    assert approval.interrupt is not None
    assert approval.interrupt.kind == "plan_approval"
    assert approval.phase == "awaiting_approval"

    result = coordinator.resume(session_id=thread_id, action="approve")
    assert result.interrupt is not None
    assert result.interrupt.kind == "result"
    assert len(result.values["run_history"]) == 1
    assert len(result.values["tool_results"]) == 3
    assert result.values["budget"]["evaluated_iterations"] == 1
    assert result.values["loop_cursor"]["episode_number"] == 1
    event_names = [event["name"] for event in result.events]
    assert any(
        name.startswith("函数执行完成：") and "price_descriptive_distribution" in name
        for name in event_names
    )
    assert any(name.startswith("评估运行：") and "accept" in name for name in event_names)
    artifact_directory = Path(result.values["latest_run"]["artifact_directory"])
    assert (artifact_directory / "research_loop.json").is_file()

    followup = coordinator.submit_user_message(
        session_id=thread_id,
        message="这个结果说明什么？",
        study_config=config,
    )
    assert followup.interrupt is not None
    assert followup.interrupt.kind == "result"
    assert "已校验证据" in followup.values["assistant_message"]


def test_evaluator_revision_auto_executes_once_without_second_approval(
    synthetic_study: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    calls = 0

    def evaluating(*, plan, quality, summary):
        nonlocal calls
        del quality, summary
        calls += 1
        if calls == 1:
            return AgentEvaluation(
                decision="revise",
                summary="需要在原权限内收缩方案。",
                checks=[EvaluationCheck(name="风险", status="warning", message="收缩方法")],
                feedback_packets=[
                    FeedbackPacket(
                        source="evaluator",
                        code="reduce_methods",
                        severity="warning",
                        message="只保留分布分析。",
                        retryable=True,
                        created_at=plan.created_at,
                    )
                ],
            )
        return accepting_evaluation(plan=plan, quality=None, summary=None)

    monkeypatch.setattr("app.research.application.execution.evaluate_agent_run", evaluating)
    coordinator = loop_coordinator()
    config = load_study_config(synthetic_study)
    approval = coordinator.submit_user_message(
        session_id="loop-auto-revise",
        message="分析电价结构",
        study_config=config,
    )
    assert approval.interrupt and approval.interrupt.kind == "plan_approval"

    result = coordinator.resume(session_id="loop-auto-revise", action="approve")

    assert result.interrupt and result.interrupt.kind == "result"
    assert calls == 2
    assert len(result.values["run_history"]) == 2
    assert len(result.values["plan_history"]) == 2
    assert sum(event["name"].startswith("等待审批：") for event in result.events) == 1
    assert result.values["budget"]["evaluated_iterations"] == 2
    assert result.values["loop_cursor"]["iteration_number"] == 2
    assert sum(event["name"].startswith("复用函数结果：") for event in result.events) == 2


def test_new_research_episode_resets_completed_iteration_budget(
    synthetic_study: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    class EpisodeDialogue(LoopDialogue):
        def decide(self, **kwargs):
            if "新一轮" in kwargs.get("question", ""):
                return DialogueDecision(intent="new_plan", skill_name="price-exogenous-eda")
            return super().decide(**kwargs)

    monkeypatch.setattr("app.research.application.execution.evaluate_agent_run", accepting_evaluation)
    coordinator = ResearchCoordinator(
        main_agent=MainResearchAgent(model_dialogue=EpisodeDialogue()),
        eda_subagent=EDASubagent(model_planner=LoopPlanner()),
    )
    config = load_study_config(synthetic_study)
    session_id = "loop-episode-reset"
    first_approval = coordinator.submit_user_message(
        session_id=session_id,
        message="分析电价结构",
        study_config=config,
        imported_state={"budget": LoopBudget(max_evaluated_iterations=1).model_dump(mode="json")},
    )
    first_episode_id = first_approval.values["loop_cursor"]["episode_id"]
    first_result = coordinator.resume(session_id=session_id, action="approve")
    assert first_result.interrupt and first_result.interrupt.kind == "result"
    assert first_result.values["budget"]["evaluated_iterations"] == 1

    second_approval = coordinator.submit_user_message(
        session_id=session_id,
        message="开始新一轮电价研究",
        study_config=config,
    )
    assert second_approval.interrupt and second_approval.interrupt.kind == "plan_approval"
    assert second_approval.values["loop_cursor"]["episode_number"] == 2
    assert second_approval.values["loop_cursor"]["episode_id"] != first_episode_id
    assert second_approval.values["budget"]["evaluated_iterations"] == 0
    assert second_approval.values["episode_history"][-1]["episode_id"] == first_episode_id

    second_result = coordinator.resume(session_id=session_id, action="approve")
    assert second_result.interrupt and second_result.interrupt.kind == "result"
    assert second_result.values["budget"]["evaluated_iterations"] == 1
    assert len(second_result.values["run_history"]) == 2


def test_sqlite_restores_approval_and_expired_timeout_requires_confirmation(
    synthetic_study: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("app.research.application.execution.evaluate_agent_run", accepting_evaluation)
    config = load_study_config(synthetic_study)
    database = tmp_path / "loop.sqlite3"
    first = loop_coordinator(checkpoint_path=database)
    approval = first.submit_user_message(
        session_id="sqlite-approval",
        message="分析电价结构",
        study_config=config,
        approval_timeout_seconds=1,
    )
    assert approval.interrupt and approval.interrupt.kind == "plan_approval"
    first.close()

    time.sleep(1.05)
    restored = loop_coordinator(checkpoint_path=database)
    snapshot = restored.get_snapshot("sqlite-approval")
    assert snapshot.interrupt and snapshot.interrupt.kind == "plan_approval"
    assert ApprovalState.model_validate(snapshot.values["approval_state"]).status == "expired"
    with pytest.raises(ValueError, match="必须由用户明确确认"):
        restored.resume(session_id="sqlite-approval", action="timeout_accept")

    result = restored.resume(session_id="sqlite-approval", action="approve")
    assert result.interrupt and result.interrupt.kind == "result"
    restored.close()


def test_old_graph_schema_restarts_on_the_next_user_message(synthetic_study: Path):
    coordinator = loop_coordinator()
    session_id = "loop-old-schema"
    config = load_study_config(synthetic_study)
    coordinator.submit_user_message(
        session_id=session_id,
        message="分析电价结构",
        study_config=config,
    )
    coordinator.graph.update_state(coordinator._config(session_id), {"graph_schema_version": 1})

    restarted = coordinator.submit_user_message(
        session_id=session_id,
        message="接受",
        study_config=config,
    )

    assert restarted.values["graph_schema_version"] == GRAPH_SCHEMA_VERSION
    assert restarted.values["user_request"] == "分析电价结构"
    assert restarted.interrupt and restarted.interrupt.kind == "plan_approval"
    assert restarted.values["budget"]["plan_attempts_in_iteration"] == 1


def test_transient_tool_failure_retries_once(
    synthetic_study: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("app.research.application.execution.evaluate_agent_run", accepting_evaluation)

    class FlakyExecution(EDAExecutionService):
        attempts = 0

        def execute_call(self, **kwargs):
            self.attempts += 1
            if self.attempts == 1:
                raise OSError("temporary read failure")
            return super().execute_call(**kwargs)

    execution = FlakyExecution()
    coordinator = loop_coordinator(execution=execution)
    config = load_study_config(synthetic_study)
    coordinator.submit_user_message(
        session_id="loop-retry",
        message="分析电价结构",
        study_config=config,
    )
    result = coordinator.resume(session_id="loop-retry", action="approve")

    assert result.interrupt and result.interrupt.kind == "result"
    assert execution.attempts >= 2
    assert any(record["attempts"] == 2 for record in result.values["tool_records"].values())


def test_invalid_model_plan_receives_feedback_and_repairs_before_approval(
    synthetic_study: Path,
):
    class RepairPlanner(LoopPlanner):
        calls = 0

        def propose(self, question, config, quality, history=None, skill=None, feedback=None):
            self.calls += 1
            if self.calls == 1:
                return {
                    "objective": "无效方案",
                    "selected_variables": ["unknown_variable"],
                    "steps": [],
                }
            assert feedback and feedback[-1].source == "plan_validator"
            return super().propose(question, config, quality, history=history, skill=skill, feedback=feedback)

    planner = RepairPlanner()
    coordinator = ResearchCoordinator(
        main_agent=MainResearchAgent(model_dialogue=LoopDialogue()),
        eda_subagent=EDASubagent(model_planner=planner),
    )
    snapshot = coordinator.submit_user_message(
        session_id="loop-plan-repair",
        message="分析电价结构",
        study_config=load_study_config(synthetic_study),
    )

    assert snapshot.interrupt and snapshot.interrupt.kind == "plan_approval"
    assert planner.calls == 2
    assert snapshot.values["budget"]["plan_attempts_in_iteration"] == 2
    assert not snapshot.values["feedback_packets"]
    assert any(item["source"] == "plan_validator" for item in snapshot.values["feedback_history"])


def test_repeated_invalid_plan_stops_at_repair_budget(
    synthetic_study: Path,
):
    class InvalidPlanner(LoopPlanner):
        calls = 0

        def propose(self, *_args, **_kwargs):
            self.calls += 1
            return {"objective": "始终无效", "selected_variables": ["unknown_variable"], "steps": []}

    planner = InvalidPlanner()
    coordinator = ResearchCoordinator(
        main_agent=MainResearchAgent(model_dialogue=LoopDialogue()),
        eda_subagent=EDASubagent(model_planner=planner),
    )
    snapshot = coordinator.submit_user_message(
        session_id="loop-invalid-budget",
        message="分析电价结构",
        study_config=load_study_config(synthetic_study),
    )

    assert snapshot.interrupt and snapshot.interrupt.kind == "plan_error"
    assert planner.calls == 2
    assert snapshot.values["budget"]["plan_attempts_in_iteration"] == 2
    assert snapshot.values["loop_records"]
    assert Path(snapshot.values["loop_records"][-1]).is_file()


def test_plan_error_does_not_treat_acceptance_as_a_new_planning_request(
    synthetic_study: Path,
):
    class InvalidPlanner(LoopPlanner):
        calls = 0

        def propose(self, *_args, **_kwargs):
            self.calls += 1
            return {"objective": "始终无效", "selected_variables": ["unknown_variable"], "steps": []}

    planner = InvalidPlanner()
    coordinator = ResearchCoordinator(
        main_agent=MainResearchAgent(model_dialogue=LoopDialogue()),
        eda_subagent=EDASubagent(model_planner=planner),
    )
    session_id = "loop-plan-error-acceptance"
    config = load_study_config(synthetic_study)
    failed = coordinator.submit_user_message(
        session_id=session_id,
        message="分析电价结构",
        study_config=config,
    )
    assert failed.interrupt and failed.interrupt.kind == "plan_error"
    assert planner.calls == 2
    assert len(failed.values["feedback_packets"]) == 1

    clarified = coordinator.submit_user_message(
        session_id=session_id,
        message="直接用你给的方案研究",
        study_config=config,
    )
    assert clarified.interrupt and clarified.interrupt.kind == "plan_error"
    assert "没有通过校验的研究方案" in clarified.interrupt.message
    assert planner.calls == 2
    assert len(clarified.values["feedback_packets"]) == 1

    retried = coordinator.submit_user_message(session_id=session_id, message="重试", study_config=config)
    assert retried.interrupt and retried.interrupt.kind == "plan_error"
    assert planner.calls == 4
    assert len(retried.values["feedback_packets"]) == 1
    assert retried.values["budget"]["evaluated_iterations"] == 0
    assert retried.values["loop_cursor"]["iteration_number"] == 2


def test_second_evaluation_revision_exhausts_loop_to_need_user(
    synthetic_study: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    def always_revise(*, plan, quality, summary):
        del quality, summary
        return AgentEvaluation(
            decision="revise",
            summary="仍需修订。",
            checks=[EvaluationCheck(name="风险", status="warning", message="仍有风险")],
            feedback_packets=[
                FeedbackPacket(
                    source="evaluator",
                    code="keep_revising",
                    severity="warning",
                    message="继续收缩方案。",
                    retryable=True,
                    created_at=plan.created_at,
                )
            ],
        )

    monkeypatch.setattr("app.research.application.execution.evaluate_agent_run", always_revise)
    coordinator = loop_coordinator()
    coordinator.submit_user_message(
        session_id="loop-budget-exhaustion",
        message="分析电价结构",
        study_config=load_study_config(synthetic_study),
    )
    result = coordinator.resume(session_id="loop-budget-exhaustion", action="approve")

    assert result.interrupt and result.interrupt.kind == "result_limitations"
    assert len(result.values["run_history"]) == 2
    assert result.values["budget"]["evaluated_iterations"] == 2
    assert any(item["source"] == "budget_guard" for item in result.values["feedback_packets"])


def test_evaluation_revision_cannot_expand_original_approval(
    synthetic_study: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "app.research.application.execution.evaluate_agent_run",
        lambda **kwargs: AgentEvaluation(
            decision="revise",
            summary="尝试扩大方法。",
            checks=[EvaluationCheck(name="风险", status="warning", message="增加方法")],
            feedback_packets=[
                FeedbackPacket(
                    source="evaluator",
                    code="expand_method",
                    severity="warning",
                    message="增加季节性方法。",
                    retryable=True,
                    created_at=kwargs["plan"].created_at,
                )
            ],
        ),
    )

    class ExpansionPlanner(LoopPlanner):
        def propose(self, _question, _config, _quality, history=None, skill=None, feedback=None):
            del history
            return {
                "objective": "测试审批边界",
                "selected_variables": [],
                "steps": [
                    {
                        "function": "price_descriptive_distribution",
                        "enabled": True,
                        "rationale": "测试已审批函数。",
                        "parameters": {},
                    },
                    *(
                        [
                            {
                                "function": "price_calendar_group_profile",
                                "enabled": True,
                                "rationale": "测试新增未审批函数。",
                                "parameters": {},
                            }
                        ]
                        if feedback
                        else []
                    ),
                ],
            }

    coordinator = ResearchCoordinator(
        main_agent=MainResearchAgent(model_dialogue=LoopDialogue()),
        eda_subagent=EDASubagent(model_planner=ExpansionPlanner()),
    )
    coordinator.submit_user_message(
        session_id="loop-authorization-envelope",
        message="分析电价分布",
        study_config=load_study_config(synthetic_study),
    )
    result = coordinator.resume(session_id="loop-authorization-envelope", action="approve")

    assert result.interrupt and result.interrupt.kind == "result_limitations"
    assert len(result.values["run_history"]) == 1
    assert any(
        item["code"] == "automatic_revision_exceeds_approval"
        for item in result.values["feedback_packets"]
    )


def test_duplicate_automatic_plan_stops_before_reexecuting_tools(
    synthetic_study: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "app.research.application.execution.evaluate_agent_run",
        lambda **kwargs: AgentEvaluation(
            decision="revise",
            summary="请求修订。",
            checks=[EvaluationCheck(name="风险", status="warning", message="风险")],
            feedback_packets=[
                FeedbackPacket(
                    source="evaluator",
                    code="retry_same",
                    severity="warning",
                    message="再次检查。",
                    retryable=True,
                    created_at=kwargs["plan"].created_at,
                )
            ],
        ),
    )

    class DuplicatePlanner(LoopPlanner):
        def propose(self, _question, _config, _quality, history=None, skill=None, feedback=None):
            del history, feedback
            return {
                "objective": "重复方案",
                "selected_variables": [],
                "steps": [
                    {
                        "function": "price_descriptive_distribution",
                        "enabled": True,
                        "rationale": "始终相同。",
                        "parameters": {},
                    }
                ],
            }

    coordinator = ResearchCoordinator(
        main_agent=MainResearchAgent(model_dialogue=LoopDialogue()),
        eda_subagent=EDASubagent(model_planner=DuplicatePlanner()),
    )
    coordinator.submit_user_message(
        session_id="loop-duplicate-plan",
        message="分析电价分布",
        study_config=load_study_config(synthetic_study),
    )
    result = coordinator.resume(session_id="loop-duplicate-plan", action="approve")

    assert result.interrupt and result.interrupt.kind == "result_limitations"
    assert len(result.values["run_history"]) == 1
    assert any(item["code"] == "duplicate_plan" for item in result.values["feedback_packets"])


def test_crash_inside_tool_node_recovers_current_call_once_from_sqlite(
    synthetic_study: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("app.research.application.execution.evaluate_agent_run", accepting_evaluation)

    class CrashingExecution(EDAExecutionService):
        def execute_call(self, **_kwargs):
            raise KeyboardInterrupt("simulated process crash")

    config = load_study_config(synthetic_study)
    database = tmp_path / "crash.sqlite3"
    first = loop_coordinator(execution=CrashingExecution(), checkpoint_path=database)
    first.submit_user_message(
        session_id="loop-crash-recovery",
        message="分析电价结构",
        study_config=config,
    )
    with pytest.raises(KeyboardInterrupt, match="simulated process crash"):
        first.resume(session_id="loop-crash-recovery", action="approve")
    crashed = first.get_snapshot("loop-crash-recovery")
    running = next(iter(crashed.values["tool_records"].values()))
    assert running["status"] == "running"
    first.close()

    restored = loop_coordinator(checkpoint_path=database)
    result = restored.continue_thread("loop-crash-recovery")

    assert result.interrupt and result.interrupt.kind == "result"
    recovered = next(iter(result.values["tool_records"].values()))
    assert recovered["attempts"] == 2
    restored.close()


def test_tool_argument_error_returns_feedback_to_planning_layer(
    synthetic_study: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("app.research.application.execution.evaluate_agent_run", accepting_evaluation)

    class InvalidOnceExecution(EDAExecutionService):
        invalidated = False

        def compile_tool_queue(self, plan):
            calls = super().compile_tool_queue(plan)
            if not self.invalidated:
                self.invalidated = True
                price = next(call for call in calls if call.name == "price_descriptive_distribution")
                price.arguments["unexpected"] = True
            return calls

    coordinator = loop_coordinator(execution=InvalidOnceExecution())
    coordinator.submit_user_message(
        session_id="loop-tool-argument-feedback",
        message="分析电价结构",
        study_config=load_study_config(synthetic_study),
    )
    result = coordinator.resume(session_id="loop-tool-argument-feedback", action="approve")

    assert result.interrupt and result.interrupt.kind == "result"
    assert len(result.values["plan_history"]) == 2
    assert not result.values["feedback_packets"]
    assert any(item["source"] == "tool_executor" for item in result.values["feedback_history"])


def test_unregistered_skill_stops_at_need_user_interrupt(synthetic_study: Path):
    class UnknownSkillDialogue(LoopDialogue):
        def decide(self, **_kwargs):
            return DialogueDecision(intent="new_plan", skill_name="not-installed")

    coordinator = ResearchCoordinator(
        main_agent=MainResearchAgent(model_dialogue=UnknownSkillDialogue()),
        eda_subagent=EDASubagent(model_planner=LoopPlanner()),
    )
    result = coordinator.submit_user_message(
        session_id="loop-unknown-skill",
        message="使用不存在的Skill",
        study_config=load_study_config(synthetic_study),
    )

    assert result.interrupt and result.interrupt.kind == "plan_error"
    assert any(item["source"] == "skill_validator" for item in result.values["feedback_packets"])
    assert result.values["loop_records"]


def test_unknown_tool_internal_error_persists_negative_stop_without_user_revision(
    synthetic_study: Path,
):
    class BrokenExecution(EDAExecutionService):
        def execute_call(self, **_kwargs):
            raise RuntimeError("unexpected internal tool failure")

    coordinator = loop_coordinator(execution=BrokenExecution())
    coordinator.submit_user_message(
        session_id="loop-unknown-tool-error",
        message="分析电价结构",
        study_config=load_study_config(synthetic_study),
    )
    stopped = coordinator.resume(session_id="loop-unknown-tool-error", action="approve")

    assert stopped.phase == "failed"
    assert stopped.interrupt is None
    assert "unexpected internal tool failure" in stopped.values["stop_reason"]
    record_path = Path(stopped.values["loop_records"][-1])
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["outcome"] == "failed"
    assert record["stop_reason"] == stopped.values["stop_reason"]
    assert record["feedback_packets"][-1]["severity"] == "fatal"


def test_evaluator_reject_persists_negative_result_and_stops(
    synthetic_study: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "app.research.application.execution.evaluate_agent_run",
        lambda **_kwargs: AgentEvaluation(
            decision="reject",
            summary="结果未通过验收。",
            checks=[EvaluationCheck(name="证据", status="fail", message="证据不足")],
        ),
    )
    coordinator = loop_coordinator()
    coordinator.submit_user_message(
        session_id="loop-evaluator-reject",
        message="分析电价结构",
        study_config=load_study_config(synthetic_study),
    )
    rejected = coordinator.resume(session_id="loop-evaluator-reject", action="approve")

    assert rejected.interrupt and rejected.interrupt.kind == "result_rejected"
    assert rejected.interrupt.choices == ["modify", "stop"]
    assert rejected.values["latest_run"]["evaluation"]["decision"] == "reject"
    assert "评估器拒绝当前结果" in rejected.values["stop_reason"]

    stopped = coordinator.resume(session_id="loop-evaluator-reject", action="stop")
    assert stopped.phase == "stopped"
    assert stopped.interrupt is None
    record = json.loads(Path(stopped.values["loop_records"][-1]).read_text(encoding="utf-8"))
    assert record["outcome"] == "stopped"
    assert record["evaluation"]["decision"] == "reject"


def test_evaluator_need_user_accept_limitations_reaches_result(
    synthetic_study: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "app.research.application.execution.evaluate_agent_run",
        lambda **_kwargs: AgentEvaluation(
            decision="need_user",
            summary="需要用户确认当前限制。",
            checks=[EvaluationCheck(name="限制", status="warning", message="请确认")],
        ),
    )
    coordinator = ResearchCoordinator(
        main_agent=MainResearchAgent(model_dialogue=NeedUserDialogue()),
        eda_subagent=EDASubagent(model_planner=LoopPlanner()),
    )
    session_id = "loop-need-user-accept"
    coordinator.submit_user_message(
        session_id=session_id,
        message="分析电价结构",
        study_config=load_study_config(synthetic_study),
    )
    waiting = coordinator.resume(session_id=session_id, action="approve")
    assert waiting.interrupt and waiting.interrupt.kind == "result_limitations"
    result = coordinator.resume(session_id=session_id, action="accept_limitations")
    assert result.interrupt and result.interrupt.kind == "result"
    assert "只根据已校验证据" in result.values["assistant_message"]


def test_evaluator_need_user_modify_returns_to_plan_approval(
    synthetic_study: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "app.research.application.execution.evaluate_agent_run",
        lambda **_kwargs: AgentEvaluation(
            decision="need_user",
            summary="需要用户确认当前限制。",
            checks=[EvaluationCheck(name="限制", status="warning", message="请确认")],
        ),
    )
    coordinator = ResearchCoordinator(
        main_agent=MainResearchAgent(model_dialogue=NeedUserDialogue()),
        eda_subagent=EDASubagent(model_planner=LoopPlanner()),
    )
    session_id = "loop-need-user-modify"
    coordinator.submit_user_message(
        session_id=session_id,
        message="分析电价结构",
        study_config=load_study_config(synthetic_study),
    )
    waiting = coordinator.resume(session_id=session_id, action="approve")
    assert waiting.interrupt and waiting.interrupt.kind == "result_limitations"
    revised = coordinator.resume(session_id=session_id, action="modify", message="修改方案，只保留电价画像")
    assert revised.interrupt and revised.interrupt.kind == "plan_approval"
    assert [step["function"] for step in revised.values["current_plan"]["steps"] if step["enabled"]] == [
        "data_quality",
        "price_descriptive_distribution",
    ]


def test_evaluator_need_user_stop_persists_stop_record(
    synthetic_study: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "app.research.application.execution.evaluate_agent_run",
        lambda **_kwargs: AgentEvaluation(
            decision="need_user",
            summary="需要用户确认当前限制。",
            checks=[EvaluationCheck(name="限制", status="warning", message="请确认")],
        ),
    )
    coordinator = loop_coordinator()
    session_id = "loop-need-user-stop"
    coordinator.submit_user_message(
        session_id=session_id,
        message="分析电价结构",
        study_config=load_study_config(synthetic_study),
    )
    waiting = coordinator.resume(session_id=session_id, action="approve")
    assert waiting.interrupt and waiting.interrupt.kind == "result_limitations"
    stopped = coordinator.resume(session_id=session_id, action="stop")
    assert stopped.phase == "stopped"
    assert stopped.interrupt is None
    assert stopped.values["loop_records"]


def test_plan_approval_reject_persists_record_without_running_tools(synthetic_study: Path):
    class CountingExecution(EDAExecutionService):
        calls = 0

        def execute_call(self, **kwargs):
            self.calls += 1
            return super().execute_call(**kwargs)

    execution = CountingExecution()
    coordinator = loop_coordinator(execution=execution)
    session_id = "loop-approval-reject"
    approval = coordinator.submit_user_message(
        session_id=session_id,
        message="分析电价结构",
        study_config=load_study_config(synthetic_study),
    )
    assert approval.interrupt and approval.interrupt.kind == "plan_approval"
    stopped = coordinator.resume(session_id=session_id, action="reject")
    assert stopped.phase == "stopped"
    assert stopped.interrupt is None
    assert execution.calls == 0
    assert stopped.values["loop_records"]


def test_function_queue_size_does_not_reduce_iteration_limit(
    synthetic_study: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    evaluations = 0

    def revise_each_round(*, plan, quality, summary):
        nonlocal evaluations
        del quality, summary
        evaluations += 1
        return AgentEvaluation(
            decision="revise",
            summary=f"第 {evaluations} 轮需要修订。",
            checks=[EvaluationCheck(name="风险", status="warning", message=f"轮次 {evaluations}")],
            feedback_packets=[
                FeedbackPacket(
                    source="evaluator",
                    code=f"revise_{evaluations}",
                    severity="warning",
                    message=f"继续第 {evaluations} 轮。",
                    retryable=True,
                    created_at=plan.created_at,
                )
            ],
        )

    monkeypatch.setattr("app.research.application.execution.evaluate_agent_run", revise_each_round)
    class VaryingLagPlanner(FourToolPlanner):
        def propose(self, *args, **kwargs):
            draft = super().propose(*args, **kwargs)
            draft["steps"].append(
                {
                    "function": "price_lag_autocorrelation",
                    "enabled": True,
                    "rationale": "每轮收缩滞后范围以产生不同证据。",
                    "parameters": {"max_lag": max(1, 5 - self.calls)},
                }
            )
            return draft

    planner = VaryingLagPlanner()
    coordinator = ResearchCoordinator(
        main_agent=MainResearchAgent(model_dialogue=LoopDialogue()),
        eda_subagent=EDASubagent(model_planner=planner),
    )
    budget = LoopBudget(max_evaluated_iterations=4)
    session_id = "loop-iteration-limit-only"
    coordinator.submit_user_message(
        session_id=session_id,
        message="分析电价与外生变量关系",
        study_config=load_study_config(synthetic_study),
        imported_state={"budget": budget.model_dump(mode="json")},
    )
    result = coordinator.resume(session_id=session_id, action="approve")
    assert result.interrupt and result.interrupt.kind == "result_limitations"
    assert result.values["budget"]["evaluated_iterations"] == 4
    assert evaluations == 4
    assert len(result.values["run_history"]) == 4
    assert not any(
        item["code"] in {"tool_call_budget", "active_time_budget"}
        for item in result.values["feedback_packets"]
    )


def test_no_new_evidence_stops_automatic_loop(synthetic_study: Path, monkeypatch: pytest.MonkeyPatch):
    fixed_packet = FeedbackPacket(
        feedback_id="fixed-evidence-feedback",
        source="evaluator",
        code="same_evidence",
        severity="warning",
        message="证据没有变化。",
        retryable=True,
        created_at="2026-01-01T00:00:00+00:00",
    )

    def same_evidence(*, plan, quality, summary):
        del plan, quality, summary
        return AgentEvaluation(
            decision="revise",
            summary="相同证据。",
            checks=[EvaluationCheck(name="证据", status="warning", message="相同")],
            feedback_packets=[fixed_packet],
        )

    monkeypatch.setattr("app.research.application.execution.evaluate_agent_run", same_evidence)
    planner = FourToolPlanner()
    coordinator = ResearchCoordinator(
        main_agent=MainResearchAgent(model_dialogue=LoopDialogue()),
        eda_subagent=EDASubagent(model_planner=planner),
    )
    session_id = "loop-no-new-evidence"
    coordinator.submit_user_message(
        session_id=session_id,
        message="分析电价与外生变量关系",
        study_config=load_study_config(synthetic_study),
        imported_state={"budget": LoopBudget(max_evaluated_iterations=3).model_dump(mode="json")},
    )
    result = coordinator.resume(session_id=session_id, action="approve")
    assert result.interrupt and result.interrupt.kind == "result_limitations"
    assert any(item["code"] == "no_new_evidence" for item in result.values["feedback_packets"])
    assert result.values["budget"]["evaluated_iterations"] == 2
