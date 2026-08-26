"""Regression coverage for P2 persistence, audit, and state-hygiene fixes."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from app.desktop.input_config import build_session_study_config
from app.desktop.session import ResearchSession, SessionInputFile
from app.research.agent.orchestrator import DialogueDecision, MainResearchAgent
from app.research.agent.subagents.eda import EDASubagent
from app.research.application.coordinator import ResearchCoordinator
from app.research.application.execution import EDAExecutionService
from app.research.graph.contracts import LoopCursor
from app.research.graph.guards import canonical_hash
from app.research.graph.workflow import MAX_STATE_EVENTS, _event
from app.research.reporting.loop_history import write_loop_record
from app.research.schemas.study import load_study_config
from app.runtime_paths import application_data_directory, default_research_output_directory


class Dialogue:
    enabled = True
    model_name = "p2-test-model"

    def decide(self, **kwargs):
        if kwargs.get("summary"):
            return DialogueDecision(intent="explain_result", response="解释确定性结果。")
        return DialogueDecision(intent="new_plan", skill_name="price-exogenous-eda")


class Planner:
    enabled = True
    model_name = "p2-test-model"

    def propose(self, _question, _config, _quality, history=None, skill=None, feedback=None):
        del history, feedback
        assert skill is not None
        return {
            "objective": "检查电价分布",
            "selected_variables": [],
            "steps": [
                {
                    "tool": "price_descriptive_distribution",
                    "rationale": "检查电价分布。",
                    "parameters": {},
                }
            ],
        }


def _coordinator(*, execution: EDAExecutionService | None = None, checkpoint_path: Path | None = None):
    return ResearchCoordinator(
        main_agent=MainResearchAgent(model_dialogue=Dialogue()),
        eda_subagent=EDASubagent(model_planner=Planner()),
        execution=execution,
        checkpoint_path=checkpoint_path,
    )


def test_events_are_bounded_and_keep_monotonic_sequence():
    state = {"events": []}
    for index in range(MAX_STATE_EVENTS + 25):
        state["events"] = _event(state, f"event-{index}")

    assert len(state["events"]) == MAX_STATE_EVENTS
    assert state["events"][0]["sequence"] == 26
    assert state["events"][-1]["sequence"] == MAX_STATE_EVENTS + 25


def test_runtime_writes_use_configured_user_directories():
    record = write_loop_record(thread_id="p2-runtime-path", state={}, outcome="stopped")

    assert record.is_relative_to(application_data_directory())
    assert default_research_output_directory().name == "research-output"


def test_desktop_yaml_config_uses_runtime_output_directory(synthetic_study: Path, tmp_path: Path):
    session = ResearchSession(
        inputs={
            "config": SessionInputFile(role="config", label="研究配置", path=str(synthetic_study)),
            "target": SessionInputFile(role="target", label="目标电价"),
            "actuals": SessionInputFile(role="actuals", label="实际外生变量"),
            "forecasts": SessionInputFile(role="forecasts", label="预测外生变量"),
        }
    )
    output_directory = tmp_path / "runtime-research"

    config = build_session_study_config(session, output_directory=output_directory)

    assert config.analysis.output_directory == output_directory.resolve()


def test_canonical_hash_rejects_unknown_objects_and_non_finite_floats():
    class Unknown:
        pass

    with pytest.raises(TypeError):
        canonical_hash(Unknown())
    with pytest.raises(ValueError):
        canonical_hash({"value": float("nan")})


def test_legacy_cursor_attempt_is_migrated_to_stage_specific_field():
    planning = LoopCursor.model_validate({"stage": "planning", "attempt_number": 2})
    function = LoopCursor.model_validate({"stage": "function:data_quality", "attempt_number": 3})

    assert planning.plan_attempt_number == 2
    assert planning.call_attempt_number == 0
    assert function.plan_attempt_number == 0
    assert function.call_attempt_number == 3


def test_tool_result_records_real_wall_clock_times(synthetic_study: Path):
    coordinator = _coordinator()
    config = load_study_config(synthetic_study)
    plan = coordinator.propose(question="分析电价", study_config=config).plan
    call = coordinator.execution.compile_tool_queue(plan)[0]

    result = coordinator.execution.execute_call(plan=plan, study_config=config, call=call)

    assert result.started_at and result.finished_at
    assert datetime.fromisoformat(result.started_at) <= datetime.fromisoformat(result.finished_at)
    coordinator.close()


def test_empty_compiled_queue_fails_closed_before_evaluation(synthetic_study: Path):
    class EmptyQueueExecution(EDAExecutionService):
        def compile_tool_queue(self, plan):
            del plan
            return []

    coordinator = _coordinator(execution=EmptyQueueExecution())
    config = load_study_config(synthetic_study)
    approval = coordinator.submit_user_message(session_id="p2-empty-queue", message="分析电价", study_config=config)
    assert approval.interrupt and approval.interrupt.kind == "plan_approval"

    failed = coordinator.resume(session_id="p2-empty-queue", action="approve")

    assert failed.phase == "failed"
    assert any("没有生成任何函数调用" in item["message"] for item in failed.values["feedback_packets"])


def test_foreground_timeout_cannot_approve_before_deadline(synthetic_study: Path):
    coordinator = _coordinator()
    config = load_study_config(synthetic_study)
    approval = coordinator.submit_user_message(
        session_id="p2-early-timeout",
        message="分析电价",
        study_config=config,
        approval_timeout_seconds=30,
    )
    assert approval.interrupt and approval.interrupt.kind == "plan_approval"

    blocked = coordinator.resume(
        session_id="p2-early-timeout",
        action="timeout_accept",
        foreground_timeout=True,
    )

    assert blocked.interrupt and blocked.interrupt.kind == "plan_approval"
    assert any(
        item["code"] == "invalid_automatic_approval_timeout"
        for item in blocked.values["feedback_packets"]
    )


def test_old_sqlite_schema_is_safe_to_open_and_backed_up_before_restart(
    synthetic_study: Path,
    tmp_path: Path,
):
    checkpoint = tmp_path / "research.sqlite3"
    coordinator = _coordinator(checkpoint_path=checkpoint)
    config = load_study_config(synthetic_study)
    approval = coordinator.submit_user_message(session_id="p2-old-schema", message="分析电价", study_config=config)
    legacy_approval = {**approval.values["approval_state"], "removed_legacy_field": True}
    coordinator.graph.update_state(
        coordinator._config("p2-old-schema"),
        {"graph_schema_version": 1, "approval_state": legacy_approval},
    )

    safe = coordinator.get_snapshot("p2-old-schema")
    assert safe.phase == "stopped"
    assert safe.values["schema_upgrade_required"] is True

    restarted = coordinator.submit_user_message(
        session_id="p2-old-schema",
        message="接受",
        study_config=config,
    )
    assert restarted.values["graph_schema_version"] == 8
    assert list((tmp_path / "checkpoint_backups").rglob("*.json"))
    coordinator.close()
