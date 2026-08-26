"""Regression coverage for P1 convergence, reuse, and authorization design."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.research.agent.errors import PlanCompatibilityError
from app.research.agent.orchestrator import DialogueDecision, MainResearchAgent
from app.research.agent.schemas import AgentEvaluation, EvaluationCheck
from app.research.agent.subagents.eda import EDASubagent
from app.research.application.coordinator import ResearchCoordinator
from app.research.application.execution import EDAExecutionService
from app.research.graph.guards import authorization_envelope, validate_automatic_revision
from app.research.schemas.feedback import FeedbackPacket
from app.research.schemas.study import load_study_config
from app.research.tools.contracts import ToolOutput


class Dialogue:
    enabled = True
    model_name = "p1-test-model"

    def decide(self, **kwargs):
        if kwargs.get("summary"):
            return DialogueDecision(intent="explain_result", response="解释确定性结果。")
        if kwargs.get("config") is None:
            return DialogueDecision(intent="discussion", response="请提供数据。")
        return DialogueDecision(intent="new_plan", skill_name="price-exogenous-eda")


class Planner:
    enabled = True
    model_name = "p1-test-model"

    def __init__(self) -> None:
        self.calls = 0

    def propose(self, _question, _config, _quality, history=None, skill=None, feedback=None):
        del history, feedback
        assert skill is not None
        self.calls += 1
        return {
            "objective": f"电价分析 {self.calls}",
            "selected_variables": [],
            "steps": [
                {
                    "tool": "price_descriptive_distribution",
                    "rationale": "检查分布。",
                    "parameters": {},
                }
            ],
        }


def _coordinator(*, planner=None, execution=None) -> ResearchCoordinator:
    return ResearchCoordinator(
        main_agent=MainResearchAgent(model_dialogue=Dialogue()),
        eda_subagent=EDASubagent(model_planner=planner or Planner()),
        execution=execution,
    )


def test_execution_rejects_plan_without_data_fingerprint(synthetic_study: Path):
    coordinator = _coordinator()
    proposal = coordinator.propose(question="分析电价", config_path=synthetic_study)

    with pytest.raises(PlanCompatibilityError, match="必须存在数据指纹"):
        EDAExecutionService().prepare(
            plan=proposal.plan.model_copy(update={"data_fingerprint": None}),
            study_config=load_study_config(synthetic_study),
        )


def test_work_id_is_stable_across_plan_revision_while_call_id_preserves_provenance(synthetic_study: Path):
    coordinator = _coordinator()
    plan = coordinator.propose(question="分析电价", config_path=synthetic_study).plan
    revised = plan.model_copy(update={"plan_id": "revisedplan01", "revision": plan.revision + 1})
    service = EDAExecutionService()

    initial_calls = service.compile_tool_queue(plan)
    revised_calls = service.compile_tool_queue(revised)

    assert [call.work_id for call in initial_calls] == [call.work_id for call in revised_calls]
    assert [call.call_id for call in initial_calls] != [call.call_id for call in revised_calls]
    assert [call.step_id for call in initial_calls] == [call.step_id for call in revised_calls]


def test_authorization_envelope_blocks_question_and_unapproved_parameter_changes(synthetic_study: Path):
    coordinator = _coordinator()
    plan = coordinator.propose(question="分析电价", config_path=synthetic_study).plan
    envelope = authorization_envelope(plan, approved_at="2026-08-25T00:00:00+00:00")

    question_violation = validate_automatic_revision(
        plan.model_copy(update={"question": "另一个研究问题"}),
        envelope,
    )
    assert question_violation is not None
    assert "研究问题发生变化" in question_violation.message

    changed_steps = [
        step.model_copy(update={"parameters": {"unexpected_threshold": 99}})
        if step.tool == "price_descriptive_distribution"
        else step
        for step in plan.steps
    ]
    parameter_violation = validate_automatic_revision(
        plan.model_copy(update={"steps": changed_steps}),
        envelope,
    )
    assert parameter_violation is not None
    assert "未重新审批的参数 unexpected_threshold" in parameter_violation.message


def test_graph_result_validation_owns_invalid_output_feedback(synthetic_study: Path):
    class InvalidOutputExecution(EDAExecutionService):
        def execute_call(self, **kwargs):
            result = super().execute_call(**kwargs)
            return result.model_copy(update={"output": ToolOutput(result_key="wrong", value={})})

    coordinator = _coordinator(execution=InvalidOutputExecution())
    waiting = coordinator.submit_user_message(
        session_id="p1-result-validation",
        message="分析电价",
        study_config=load_study_config(synthetic_study),
    )
    assert waiting.interrupt and waiting.interrupt.kind == "plan_approval"
    result = coordinator.resume(session_id="p1-result-validation", action="approve")

    assert result.interrupt and result.interrupt.kind in {"plan_error", "result_limitations"}
    assert any(item["source"] == "tool_result_validator" for item in result.values["feedback_packets"])
    assert not any(
        item["source"] == "tool_executor" and "错误结果键" in item["message"]
        for item in result.values["feedback_packets"]
    )
    assert result.values["pending_tool_result"] is None
    assert any(record["status"] == "failed" for record in result.values["tool_records"].values())


def test_plan_compatibility_failure_waits_for_user_instead_of_auto_repair(synthetic_study: Path):
    class IncompatibleExecution(EDAExecutionService):
        def execute_call(self, **_kwargs):
            raise PlanCompatibilityError("工具版本不匹配")

    planner = Planner()
    coordinator = _coordinator(planner=planner, execution=IncompatibleExecution())
    waiting = coordinator.submit_user_message(
        session_id="p1-compatibility-route",
        message="分析电价",
        study_config=load_study_config(synthetic_study),
    )
    assert waiting.interrupt and waiting.interrupt.kind == "plan_approval"

    blocked = coordinator.resume(session_id="p1-compatibility-route", action="approve")

    assert blocked.interrupt and blocked.interrupt.kind == "plan_error"
    assert planner.calls == 1
    assert blocked.values["feedback_packets"][-1]["requires_user"] is True


def test_a_b_a_evidence_cycle_stops_on_membership_not_only_adjacent(
    synthetic_study: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    class OscillatingPlanner(Planner):
        def propose(self, _question, _config, _quality, history=None, skill=None, feedback=None):
            del history, feedback
            assert skill is not None
            self.calls += 1
            max_lag = [2, 1, 2][min(self.calls - 1, 2)]
            return {
                "objective": f"振荡方案 {self.calls}",
                "selected_variables": [],
                "steps": [
                    {
                        "tool": "price_lag_autocorrelation",
                        "rationale": "制造 A-B-A 工具证据。",
                        "parameters": {"max_lag": max_lag},
                    }
                ],
            }

    evaluations = 0

    def always_revise(*, plan, quality, summary):
        nonlocal evaluations
        del quality, summary
        evaluations += 1
        return AgentEvaluation(
            decision="revise",
            summary="继续修订。",
            checks=[EvaluationCheck(name="证据", status="warning", message="继续")],
            feedback_packets=[
                FeedbackPacket(
                    source="evaluator",
                    code=f"cycle_{evaluations}",
                    severity="warning",
                    message="继续。",
                    retryable=True,
                    created_at=plan.created_at,
                )
            ],
        )

    monkeypatch.setattr("app.research.application.execution.evaluate_agent_run", always_revise)
    coordinator = _coordinator(planner=OscillatingPlanner())
    coordinator.submit_user_message(
        session_id="p1-evidence-cycle",
        message="检查证据振荡",
        study_config=load_study_config(synthetic_study),
    )
    stopped = coordinator.resume(session_id="p1-evidence-cycle", action="approve")

    assert stopped.interrupt and stopped.interrupt.kind == "result_limitations"
    assert evaluations == 3
    assert stopped.values["budget"]["evaluated_iterations"] == 3
    assert any(item["code"] == "no_new_evidence" for item in stopped.values["feedback_packets"])


def test_unchanged_agenda_stops_before_iteration_budget(
    synthetic_study: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    evaluations = 0

    def same_agenda(*, plan, quality, summary):
        nonlocal evaluations
        del quality, summary
        evaluations += 1
        return AgentEvaluation(
            decision="revise",
            summary="仍有同一个参数级议程项。",
            checks=[
                EvaluationCheck(
                    name="参数范围",
                    status="fail",
                    message="仍需收缩参数。",
                    scope="within_envelope",
                    remediation="收缩 max_lag。",
                )
            ],
            feedback_packets=[
                FeedbackPacket(
                    source="evaluator",
                    code="same_agenda",
                    severity="error",
                    message="仍需收缩参数。",
                    retryable=True,
                    created_at=plan.created_at,
                )
            ],
            agenda_fingerprint="stable-agenda",
        )

    monkeypatch.setattr("app.research.application.execution.evaluate_agent_run", same_agenda)

    class VaryingPlanner(Planner):
        def propose(self, question, config, quality, history=None, skill=None, feedback=None):
            draft = super().propose(question, config, quality, history=history, skill=skill, feedback=feedback)
            draft["objective"] = f"不同参数方案 {self.calls}"
            return draft

    planner = VaryingPlanner()
    coordinator = _coordinator(planner=planner)
    coordinator.submit_user_message(
        session_id="p1-unchanged-agenda",
        message="分析电价",
        study_config=load_study_config(synthetic_study),
        imported_state={"budget": {"max_evaluated_iterations": 4}},
    )
    stopped = coordinator.resume(session_id="p1-unchanged-agenda", action="approve")

    assert stopped.interrupt and stopped.interrupt.kind == "result_limitations"
    assert evaluations == 2
    assert stopped.values["budget"]["evaluated_iterations"] == 2
    assert any(item["code"] == "no_agenda_progress" for item in stopped.values["feedback_packets"])


def test_episode_goal_remains_stable_while_latest_turn_changes(synthetic_study: Path):
    class ModifyDialogue(Dialogue):
        def decide(self, **kwargs):
            if kwargs.get("question") == "只保留分布":
                return DialogueDecision(
                    intent="revise_plan",
                    response="已修订。",
                    enabled_tools=["price_descriptive_distribution"],
                    selected_variables=[],
                )
            return super().decide(**kwargs)

    coordinator = ResearchCoordinator(
        main_agent=MainResearchAgent(model_dialogue=ModifyDialogue()),
        eda_subagent=EDASubagent(model_planner=Planner()),
    )
    first = coordinator.submit_user_message(
        session_id="p1-stable-goal",
        message="研究最初目标",
        study_config=load_study_config(synthetic_study),
    )
    original_goal = first.values["loop_cursor"]["episode_goal"]
    revised = coordinator.resume(session_id="p1-stable-goal", action="modify", message="只保留分布")

    assert revised.values["loop_cursor"]["episode_goal"] == original_goal == "研究最初目标"
    assert revised.values["user_request"] == "研究最初目标"
    assert revised.values["latest_turn"] == "只保留分布"


def test_unknown_evaluation_decision_fails_closed_with_feedback(
    synthetic_study: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "app.research.application.execution.evaluate_agent_run",
        lambda **_kwargs: AgentEvaluation(
            decision="accept",
            summary="通过",
            checks=[EvaluationCheck(name="完整性", status="pass", message="通过")],
        ),
    )
    coordinator = _coordinator()
    thread_id = "p1-invalid-evaluation"
    coordinator.submit_user_message(
        session_id=thread_id,
        message="分析电价",
        study_config=load_study_config(synthetic_study),
    )
    result = coordinator.resume(session_id=thread_id, action="approve")
    assert result.interrupt and result.interrupt.kind == "result"

    coordinator.graph.update_state(
        coordinator._config(thread_id),
        {"evaluation": {"decision": "typo"}, "control": "evaluation", "phase": "completed"},
        as_node="finalize_iteration",
    )
    coordinator._run_graph(None, thread_id=thread_id)
    failed_closed = coordinator.get_snapshot(thread_id)

    assert failed_closed.interrupt and failed_closed.interrupt.kind == "result_limitations"
    assert any(
        item["code"] == "invalid_evaluation_decision"
        for item in failed_closed.values["feedback_packets"]
    )
