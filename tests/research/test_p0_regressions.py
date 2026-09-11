"""Regression coverage for the Phase-1 P0 state and execution boundaries."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.research.agent.errors import RepairablePlanError, ResearchPlanValidationError
from app.research.agent.orchestrator import DialogueDecision, MainResearchAgent
from app.research.agent.subagents.eda import EDASubagent
from app.research.application.coordinator import ResearchCoordinator
from app.research.application.execution import EDAExecutionService
from app.research.application.planning import prepare_research_data
from app.research.graph.guards import authorization_envelope, validate_automatic_revision
from app.research.schemas.study import load_study_config
from app.research.tools.contracts import SegmentDefinition, ToolOutput
from app.research.tools.eda.segments import compare_price_segments
from app.research.tools.executor import output_fingerprint


class PricePlanner:
    enabled = True
    model_name = "p0-test-model"

    def __init__(self) -> None:
        self.calls = 0

    def propose(self, _question, _config, _quality, history=None, skill=None, feedback=None):
        del history, feedback
        assert skill is not None
        self.calls += 1
        return {
            "objective": "验证电价结构",
            "selected_variables": [],
            "steps": [
                {
                    "function": "price_descriptive_distribution",
                    "rationale": "检查电价分布。",
                    "parameters": {},
                }
            ],
        }


class ResearchDialogue:
    enabled = True
    model_name = "p0-test-model"

    def decide(self, **kwargs):
        if kwargs.get("summary"):
            return DialogueDecision(intent="explain_result", response="只解释确定性证据。")
        if kwargs.get("config") is None:
            return DialogueDecision(intent="discussion", response="请先提供数据。")
        return DialogueDecision(intent="new_plan", skill_name="price-exogenous-eda")


def _coordinator(*, planner=None, dialogue=None, execution=None) -> ResearchCoordinator:
    return ResearchCoordinator(
        main_agent=MainResearchAgent(model_dialogue=dialogue or ResearchDialogue()),
        eda_subagent=EDASubagent(model_planner=planner or PricePlanner()),
        execution=execution,
    )


def test_segment_comparisons_share_one_summary_and_evaluation(synthetic_study: Path, tmp_path: Path):
    class SegmentPlanner(PricePlanner):
        def propose(self, _question, _config, _quality, history=None, skill=None, feedback=None):
            del history, feedback
            assert skill is not None
            segments = [
                {"segment_id": "peak", "label": "峰段", "kind": "hours", "hours": [8, 9, 10, 18, 19, 20]},
                {"segment_id": "valley", "label": "谷段", "kind": "hours", "hours": [0, 1, 2, 3, 4, 5]},
            ]
            return {
                "objective": "比较峰谷电价及负荷关系",
                "hypotheses": ["峰段与谷段存在分段差异。"],
                "selected_variables": ["load"],
                "steps": [
                    {
                        "function": "price_segment_distribution_comparison",
                        "rationale": "在同一快照比较峰谷电价。",
                        "parameters": {"comparison_id": "peak_vs_valley", "segments": segments},
                    },
                    {
                        "function": "relationship_pearson_segment_comparison",
                        "rationale": "在同一快照比较峰谷负荷关系。",
                        "parameters": {
                            "comparison_id": "peak_vs_valley",
                            "segments": segments,
                            "variables": ["load"],
                        },
                    },
                ],
            }

    coordinator = _coordinator(planner=SegmentPlanner())
    proposal = coordinator.propose(question="比较峰段和谷段", config_path=synthetic_study)
    result = coordinator.execute(
        plan=proposal.plan,
        config_path=synthetic_study,
        output_directory=tmp_path / "segment-artifacts",
        run_id="segment-comparison",
    )

    comparisons = result.eda_summary["comparisons"]
    assert set(comparisons) == {"methods", "price", "relationships"}
    assert set(comparisons["price"]["peak_vs_valley"]["segments"]) == {"peak", "valley"}
    assert "load" in comparisons["relationships"]["peak_vs_valley"]["series"]
    assert any(check.name == "分段比较证据" for check in result.evaluation.checks)
    assert result.evaluation.hypothesis_assessments[0].status == "candidate_support"
    assert "标准差" in result.evaluation.hypothesis_assessments[0].evidence

    envelope = authorization_envelope(proposal.plan, approved_at="2026-08-25T00:00:00+00:00")
    changed_steps = [
        step.model_copy(
            update={
                "parameters": {
                    **step.parameters,
                    "segments": [
                        *step.parameters["segments"],
                        {"segment_id": "extra", "label": "额外", "kind": "hours", "hours": [6]},
                    ],
                }
            }
        )
        if step.function == "price_segment_distribution_comparison"
        else step
        for step in proposal.plan.steps
    ]
    violation = validate_automatic_revision(
        proposal.plan.model_copy(update={"steps": changed_steps}),
        envelope,
    )
    assert violation is not None
    assert "改变了未重新审批的分段定义" in violation.message


def test_event_before_after_time_ranges_are_compared_on_one_snapshot(synthetic_study: Path):
    config = load_study_config(synthetic_study)
    prepared = prepare_research_data(config)
    result = compare_price_segments(
        prepared.aligned.frame[config.target.name],
        comparison_id="policy_event",
        segments=[
            SegmentDefinition(
                segment_id="before",
                label="事件前",
                kind="time_range",
                start_time="2025-01-01T00:00:00+08:00",
                end_time="2025-01-14T23:00:00+08:00",
            ),
            SegmentDefinition(
                segment_id="after",
                label="事件后",
                kind="time_range",
                start_time="2025-01-15T00:00:00+08:00",
                end_time="2025-01-30T23:00:00+08:00",
            ),
        ],
        min_observations=12,
    )

    comparison = result["price"]["policy_event"]
    assert comparison["segments"]["before"]["observations"] > 0
    assert comparison["segments"]["after"]["observations"] > 0
    assert comparison["contrasts"][0]["overlap_rows"] == 0
    assert comparison["contrasts"][0]["mean_difference_left_minus_right"] is not None


def test_mixed_segment_dimensions_are_never_compared_with_each_other(synthetic_study: Path):
    config = load_study_config(synthetic_study)
    prepared = prepare_research_data(config)
    result = compare_price_segments(
        prepared.aligned.frame[config.target.name],
        comparison_id="hour_month",
        segments=[
            SegmentDefinition(segment_id="valley", label="谷段", kind="hours", hours=[0, 1, 2, 3, 4, 5]),
            SegmentDefinition(segment_id="peak", label="峰段", kind="hours", hours=[18, 19, 20, 21, 22, 23]),
            SegmentDefinition(segment_id="january", label="一月", kind="months", months=[1]),
            SegmentDefinition(segment_id="february", label="二月", kind="months", months=[2]),
        ],
        min_observations=12,
    )

    contrasts = result["price"]["hour_month"]["contrasts"]
    assert len(contrasts) == 2
    assert {row["comparison_group"] for row in contrasts} == {"hours", "months"}
    assert not any(
        {row["left_segment_id"], row["right_segment_id"]} & {"valley", "peak"}
        and {row["left_segment_id"], row["right_segment_id"]} & {"january", "february"}
        for row in contrasts
    )
    hour_contrast = next(row for row in contrasts if row["comparison_group"] == "hours")
    assert hour_contrast["std_difference_left_minus_right"] is not None


def test_execution_reuses_one_prepared_snapshot_for_queue_and_finalize(
    synthetic_study: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import app.research.application.execution as execution_module

    coordinator = _coordinator()
    proposal = coordinator.propose(question="分析电价", config_path=synthetic_study)
    config = load_study_config(synthetic_study)
    service = EDAExecutionService(prepared_cache_size=2)
    real_prepare = execution_module.prepare_research_data
    real_restore = execution_module.restore_prepared_data
    calls = 0

    def counted_prepare(study_config):
        nonlocal calls
        calls += 1
        return real_prepare(study_config)

    def counted_restore(fingerprint, study_config):
        nonlocal calls
        restored = real_restore(fingerprint, study_config)
        if restored is not None:
            calls += 1
        return restored

    # Either route reads the data; the point is that it happens once for the whole run.
    monkeypatch.setattr(execution_module, "prepare_research_data", counted_prepare)
    monkeypatch.setattr(execution_module, "restore_prepared_data", counted_restore)
    service.prepare(plan=proposal.plan, study_config=config)
    results = [
        service.execute_call(plan=proposal.plan, study_config=config, call=call)
        for call in service.compile_tool_queue(proposal.plan)
    ]
    service.finalize(
        plan=proposal.plan,
        study_config=config,
        tool_results=results,
        output_directory=tmp_path / "snapshot-artifacts",
        run_id="single-snapshot",
    )

    assert calls == 1
    assert service.prepared_cache_entries == 0


def test_prepared_snapshot_cache_is_bounded(synthetic_study: Path):
    coordinator = _coordinator()
    proposal = coordinator.propose(question="分析电价", config_path=synthetic_study)
    config = load_study_config(synthetic_study)
    service = EDAExecutionService(prepared_cache_size=2)

    for index in range(3):
        service.prepare(
            plan=proposal.plan.model_copy(update={"plan_id": f"cacheplan{index}"}),
            study_config=config,
        )

    assert service.prepared_cache_entries == 2


def test_insufficient_data_does_not_retry_model_planning(synthetic_study: Path):
    planner = PricePlanner()
    coordinator = _coordinator(planner=planner)
    config = load_study_config(synthetic_study)
    config = config.model_copy(
        update={
            "analysis": config.analysis.model_copy(update={"min_target_coverage_rate": 1.0})
        }
    )

    snapshot = coordinator.submit_user_message(
        session_id="p0-insufficient-data",
        message="分析电价",
        study_config=config,
    )

    assert snapshot.interrupt and snapshot.interrupt.kind == "plan_error"
    assert planner.calls == 1
    assert any("minimum observation requirement" in item["message"] for item in snapshot.values["feedback_packets"])


def test_duplicate_function_feedback_explains_batch_alternative(synthetic_study: Path):
    class DuplicateLagPlanner(PricePlanner):
        def propose(self, _question, _config, _quality, history=None, skill=None, feedback=None):
            del history, feedback
            assert skill is not None
            self.calls += 1
            return {
                "objective": "检查两个滞后范围",
                "selected_variables": [],
                "steps": [
                    {
                        "function": "price_lag_autocorrelation",
                        "rationale": "检查短滞后。",
                        "parameters": {"max_lag": 1},
                    },
                    {
                        "function": "price_lag_autocorrelation",
                        "rationale": "检查日滞后。",
                        "parameters": {"max_lag": 24},
                    },
                ],
            }

    planner = DuplicateLagPlanner()
    coordinator = _coordinator(planner=planner)
    failed = coordinator.submit_user_message(
        session_id="p0-duplicate-function",
        message="比较 1 和 24 小时滞后",
        study_config=load_study_config(synthetic_study),
    )

    assert failed.interrupt and failed.interrupt.kind == "plan_error"
    assert planner.calls == 2
    feedback = failed.values["feedback_packets"][-1]
    assert "每个计划最多调用一次" in feedback["message"]
    assert "最大的 max_lag" in feedback["recommendation"]


def test_modify_then_model_execute_intent_cannot_bypass_approval(synthetic_study: Path):
    class MisroutingDialogue(ResearchDialogue):
        def decide(self, **kwargs):
            if kwargs.get("question") == "修改方案":
                return DialogueDecision(intent="execute_plan", response="执行旧方案。")
            return super().decide(**kwargs)

    class CountingExecution(EDAExecutionService):
        calls = 0

        def execute_call(self, **kwargs):
            self.calls += 1
            return super().execute_call(**kwargs)

    execution = CountingExecution()
    coordinator = _coordinator(dialogue=MisroutingDialogue(), execution=execution)
    approval = coordinator.submit_user_message(
        session_id="p0-approval-bypass",
        message="分析电价",
        study_config=load_study_config(synthetic_study),
    )
    assert approval.interrupt and approval.interrupt.kind == "plan_approval"

    second_gate = coordinator.resume(
        session_id="p0-approval-bypass",
        action="modify",
        message="修改方案",
    )

    assert second_gate.interrupt and second_gate.interrupt.kind == "plan_approval"
    assert execution.calls == 0
    assert second_gate.values["approval_state"]["status"] == "waiting"


def test_invalid_resume_action_keeps_current_interrupt(synthetic_study: Path, monkeypatch: pytest.MonkeyPatch):
    from app.research.agent.schemas import AgentEvaluation, EvaluationCheck

    monkeypatch.setattr(
        "app.research.application.execution.evaluate_agent_run",
        lambda **_kwargs: AgentEvaluation(
            decision="accept",
            summary="通过",
            checks=[EvaluationCheck(name="完整性", status="pass", message="通过")],
        ),
    )
    coordinator = _coordinator()
    coordinator.submit_user_message(
        session_id="p0-invalid-resume",
        message="分析电价",
        study_config=load_study_config(synthetic_study),
    )
    result = coordinator.resume(session_id="p0-invalid-resume", action="approve")
    assert result.interrupt and result.interrupt.kind == "result"
    envelope = result.values["authorization_envelope"]
    assert envelope["approved_plan_id"] == result.values["current_plan"]["plan_id"]
    assert envelope["approved_plan_fingerprint"]
    assert envelope["approved_at"]

    with pytest.raises(ValueError, match="不允许操作 approve"):
        coordinator.resume(session_id="p0-invalid-resume", action="approve")

    unchanged = coordinator.get_snapshot("p0-invalid-resume")
    assert unchanged.interrupt and unchanged.interrupt.kind == "result"


def _executed_call(synthetic_study: Path):
    """Run one real function and hand back the pieces its validation needs."""

    coordinator = _coordinator()
    config = load_study_config(synthetic_study)
    proposal = coordinator.propose(question="分析电价分布", config_path=synthetic_study)
    plan = proposal.plan
    service = EDAExecutionService()
    service.prepare(plan=plan, study_config=config)
    call = next(
        item for item in service.compile_tool_queue(plan) if item.name == "price_descriptive_distribution"
    )
    result = service.execute_call(plan=plan, study_config=config, call=call)
    return service, plan, call, result


def test_a_result_missing_its_evidence_field_is_rejected(synthetic_study: Path):
    service, plan, call, result = _executed_call(synthetic_study)
    stripped = {key: value for key, value in result.output.value.items() if key != "distribution"}
    hollow = ToolOutput(result_key=result.output.result_key, value=stripped)
    damaged = result.model_copy(
        update={"output": hollow, "output_hash": output_fingerprint(hollow)}
    )

    with pytest.raises(RepairablePlanError, match="distribution"):
        service.validate_tool_result(plan=plan, call=call, result=damaged)


def test_a_payload_that_no_longer_matches_its_output_hash_is_rejected(synthetic_study: Path):
    service, plan, call, result = _executed_call(synthetic_study)
    tampered_value = {
        **result.output.value,
        "distribution": {**result.output.value["distribution"], "mean": 999999.0},
    }
    tampered = result.model_copy(
        update={"output": ToolOutput(result_key=result.output.result_key, value=tampered_value)}
    )

    with pytest.raises(ResearchPlanValidationError, match="output_hash"):
        service.validate_tool_result(plan=plan, call=call, result=tampered)

    service.validate_tool_result(plan=plan, call=call, result=result)


def test_every_catalog_function_declares_where_its_evidence_lives():
    from app.research.tools.catalog import FUNCTION_CATALOG

    missing = [name for name, spec in FUNCTION_CATALOG.items() if spec.evidence_field is None]
    assert missing == ["data_quality"]


def test_finalize_failure_offers_a_retry_over_the_preserved_tool_results(synthetic_study: Path):
    class FlakyFinalizeExecution(EDAExecutionService):
        attempts = 0

        def finalize(self, **kwargs):
            type(self).attempts += 1
            if type(self).attempts == 1:
                raise RuntimeError("simulated finalization failure")
            return super().finalize(**kwargs)

    coordinator = _coordinator(execution=FlakyFinalizeExecution())
    coordinator.submit_user_message(
        session_id="p0-finalize-failure",
        message="分析电价",
        study_config=load_study_config(synthetic_study),
    )
    interrupted = coordinator.resume(session_id="p0-finalize-failure", action="approve")

    assert interrupted.phase == "awaiting_user"
    assert interrupted.interrupt is not None
    assert interrupted.interrupt.kind == "finalization_error"
    assert interrupted.interrupt.choices == ["retry", "stop"]
    assert interrupted.values["tool_results"]
    assert "simulated finalization failure" in interrupted.values["stop_reason"]

    completed = coordinator.resume(session_id="p0-finalize-failure", action="retry")

    assert FlakyFinalizeExecution.attempts == 2
    assert completed.values["latest_run"]
    assert not completed.values.get("stop_reason")


def test_finalize_retry_reuses_a_package_committed_before_the_checkpoint(
    synthetic_study: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from app.research.agent.schemas import AgentEvaluation, EvaluationCheck

    monkeypatch.setattr(
        "app.research.application.execution.evaluate_agent_run",
        lambda **_kwargs: AgentEvaluation(
            decision="accept",
            summary="通过。",
            checks=[EvaluationCheck(name="完整性", status="pass", message="通过。")],
        ),
    )

    class FailAfterCommitExecution(EDAExecutionService):
        attempts = 0

        def finalize(self, **kwargs):
            result = super().finalize(**kwargs)
            type(self).attempts += 1
            if type(self).attempts == 1:
                raise RuntimeError("simulated crash after artifact commit")
            return result

    config = load_study_config(synthetic_study)
    coordinator = _coordinator(execution=FailAfterCommitExecution())
    coordinator.submit_user_message(
        session_id="p0-finalize-idempotent",
        message="分析电价",
        study_config=config,
    )
    interrupted = coordinator.resume(session_id="p0-finalize-idempotent", action="approve")
    committed_before_retry = sorted(
        path for path in config.analysis.output_directory.glob("agent-*") if path.is_dir()
    )

    completed = coordinator.resume(session_id="p0-finalize-idempotent", action="retry")
    committed_after_retry = sorted(
        path for path in config.analysis.output_directory.glob("agent-*") if path.is_dir()
    )

    assert interrupted.interrupt and interrupted.interrupt.kind == "finalization_error"
    assert len(committed_before_retry) == 1
    assert committed_after_retry == committed_before_retry
    assert completed.values["latest_run"]["run_id"] == committed_before_retry[0].name


def test_a_finalize_failure_the_user_stops_still_leaves_a_terminal_record(synthetic_study: Path):
    class BrokenFinalizeExecution(EDAExecutionService):
        def finalize(self, **_kwargs):
            raise RuntimeError("simulated finalization failure")

    coordinator = _coordinator(execution=BrokenFinalizeExecution())
    coordinator.submit_user_message(
        session_id="p0-finalize-stop",
        message="分析电价",
        study_config=load_study_config(synthetic_study),
    )
    coordinator.resume(session_id="p0-finalize-stop", action="approve")
    stopped = coordinator.resume(session_id="p0-finalize-stop", action="stop")

    assert stopped.phase == "stopped"
    assert stopped.values["tool_results"]
    assert "simulated finalization failure" in stopped.values["stop_reason"]
    record = json.loads(Path(stopped.values["loop_records"][-1]).read_text(encoding="utf-8"))
    assert record["outcome"] == "stopped"
    assert record["tool_records"]


def test_explanation_failure_routes_to_visible_response_error(synthetic_study: Path, monkeypatch: pytest.MonkeyPatch):
    from app.research.agent.schemas import AgentEvaluation, EvaluationCheck

    class EmptyExplanationDialogue(ResearchDialogue):
        def decide(self, **kwargs):
            if kwargs.get("summary"):
                return DialogueDecision(intent="explain_result", response="")
            return super().decide(**kwargs)

    monkeypatch.setattr(
        "app.research.application.execution.evaluate_agent_run",
        lambda **_kwargs: AgentEvaluation(
            decision="accept",
            summary="通过",
            checks=[EvaluationCheck(name="完整性", status="pass", message="通过")],
        ),
    )
    coordinator = _coordinator(dialogue=EmptyExplanationDialogue())
    coordinator.submit_user_message(
        session_id="p0-explanation-error",
        message="分析电价",
        study_config=load_study_config(synthetic_study),
    )
    failed = coordinator.resume(session_id="p0-explanation-error", action="approve")

    assert failed.interrupt and failed.interrupt.kind == "response_error"
    assert failed.interrupt.choices == ["retry", "stop"]
    assert failed.values["assistant_message"] == ""
    assert any("模型没有返回结果解释" in item["message"] for item in failed.values["feedback_packets"])


def test_model_dialogue_failure_stops_without_a_second_schema_error(synthetic_study: Path):
    """A failed routing call pauses on the model error, not on a missing decision."""

    from app.research.agent.errors import ResearchModelUnavailableError

    class UnavailableDialogue(ResearchDialogue):
        def decide(self, **kwargs):
            raise ResearchModelUnavailableError("大模型对话调用失败：大模型调用失败：APITimeoutError")

    coordinator = _coordinator(dialogue=UnavailableDialogue())
    failed = coordinator.submit_user_message(
        session_id="p0-dialogue-unavailable",
        message="分析电价",
        study_config=load_study_config(synthetic_study),
    )

    assert failed.interrupt and failed.interrupt.kind == "plan_error"
    assert "APITimeoutError" in failed.values["stop_reason"]
    assert any(item["retryable"] for item in failed.values["feedback_packets"])
    events = [item["name"] for item in failed.values["events"]]
    assert any(name.startswith("主 Agent 调用失败") for name in events)
    assert not any(name.startswith("研究回复失败") for name in events)


def test_limited_result_followup_failure_retries_without_losing_its_gate_or_evidence(
    synthetic_study: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from app.research.agent.errors import ResearchModelUnavailableError
    from app.research.agent.schemas import AgentEvaluation, EvaluationCheck

    class OneFailedResultFollowup(ResearchDialogue):
        def __init__(self) -> None:
            self.failed = False

        def decide(self, **kwargs):
            if kwargs.get("summary") and not self.failed:
                self.failed = True
                raise ResearchModelUnavailableError("大模型结果追问暂时不可用：ConnectionError")
            return super().decide(**kwargs)

    monkeypatch.setattr(
        "app.research.application.execution.evaluate_agent_run",
        lambda **_kwargs: AgentEvaluation(
            decision="need_user",
            summary="已有结果需要用户确认限制。",
            checks=[EvaluationCheck(name="限制", status="warning", message="需要用户确认")],
        ),
    )
    coordinator = _coordinator(dialogue=OneFailedResultFollowup())
    config = load_study_config(synthetic_study)
    session_id = "p0-limited-followup-retry"
    coordinator.submit_user_message(
        session_id=session_id,
        message="分析电价",
        study_config=config,
    )
    limited = coordinator.resume(session_id=session_id, action="approve")
    assert limited.interrupt and limited.interrupt.kind == "result_limitations"
    run_id = limited.values["latest_run"]["run_id"]

    failed = coordinator.submit_user_message(
        session_id=session_id,
        message="这些结果说明什么？",
        study_config=config,
    )

    assert failed.interrupt and failed.interrupt.kind == "response_error"
    assert failed.interrupt.choices == ["retry", "stop"]
    assert failed.values["latest_run"]["run_id"] == run_id
    assert len(failed.values["run_history"]) == 1

    recovered = coordinator.resume(session_id=session_id, action="retry")

    assert recovered.interrupt and recovered.interrupt.kind == "result_limitations"
    assert recovered.values["latest_run"]["run_id"] == run_id
    assert len(recovered.values["run_history"]) == 1
    assert recovered.values["assistant_message"] == "只解释确定性证据。"
