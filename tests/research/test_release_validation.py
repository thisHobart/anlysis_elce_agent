"""Regression checks for the real Phase 1 acceptance script."""

from app.research.graph.contracts import InterruptPayload, ResearchLoopSnapshot
from scripts.validate_phase1 import (
    _finish_result_flow,
    _retry_recoverable_plan_error,
    enabled_function_names,
)


def test_acceptance_evidence_reads_current_function_field() -> None:
    plan = {
        "steps": [
            {"function": "data_quality", "tool": "legacy-wrong-name", "enabled": True},
            {"function": "price_descriptive_distribution", "enabled": True},
            {"function": "price_lag_autocorrelation", "enabled": False},
        ]
    }

    assert enabled_function_names(plan) == [
        "data_quality",
        "price_descriptive_distribution",
    ]


def test_limited_real_result_uses_the_public_stop_contract_and_keeps_evidence() -> None:
    limited = ResearchLoopSnapshot(
        thread_id="release-limited",
        phase="awaiting_user",
        values={"latest_run": {"run_id": "run-1"}, "assistant_message": "已有受限结果"},
        interrupt=InterruptPayload(
            kind="result_limitations",
            phase="awaiting_user",
            message="需要确认限制",
            choices=["modify", "followup", "stop"],
        ),
    )
    explained = limited.model_copy(
        update={"values": {**limited.values, "assistant_message": "只引用已执行证据的解释"}}
    )
    stopped = ResearchLoopSnapshot(
        thread_id="release-limited",
        phase="stopped",
        values=explained.values,
    )

    class FakeCoordinator:
        def __init__(self) -> None:
            self.actions: list[str] = []

        def submit_user_message(self, **_kwargs):
            return explained

        def resume(self, *, action: str, **_kwargs):
            self.actions.append(action)
            return stopped

    coordinator = FakeCoordinator()
    final, answer, terminal = _finish_result_flow(
        coordinator=coordinator,  # type: ignore[arg-type]
        session_id="release-limited",
        outcome=limited,
        config=object(),
        progress=None,
    )

    assert coordinator.actions == ["stop"]
    assert final.phase == "stopped"
    assert final.values["latest_run"]["run_id"] == "run-1"
    assert answer == "只引用已执行证据的解释"
    assert terminal == "limited"


def test_real_validator_retries_only_a_retryable_public_plan_error() -> None:
    plan_error = ResearchLoopSnapshot(
        thread_id="release-retry",
        phase="awaiting_user",
        values={"feedback_packets": [{"retryable": True}]},
        interrupt=InterruptPayload(
            kind="plan_error",
            phase="awaiting_user",
            message="temporary model outage",
            choices=["retry", "modify", "stop"],
        ),
    )
    recovered = ResearchLoopSnapshot(
        thread_id="release-retry",
        phase="awaiting_approval",
        values={},
        interrupt=InterruptPayload(
            kind="plan_approval",
            phase="awaiting_approval",
            message="ready",
            choices=["approve", "modify", "reject"],
        ),
    )

    class FakeCoordinator:
        def __init__(self) -> None:
            self.actions: list[str] = []

        def resume(self, *, action: str, **_kwargs):
            self.actions.append(action)
            return recovered

    coordinator = FakeCoordinator()
    result = _retry_recoverable_plan_error(
        coordinator=coordinator,  # type: ignore[arg-type]
        session_id="release-retry",
        snapshot=plan_error,
        progress=None,
    )

    assert result is recovered
    assert coordinator.actions == ["retry"]
