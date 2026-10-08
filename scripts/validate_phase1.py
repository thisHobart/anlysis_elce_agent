"""Run the persistent Phase 1 loop with the configured real model and data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.config import Settings, get_settings
from app.research.application.coordinator import ResearchCoordinator
from app.research.data.inference import infer_study_context
from app.research.graph.contracts import ResearchLoopSnapshot

VALIDATION_MINIMUM_TIMEOUT_SECONDS = 120.0


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Validate the real persistent Phase 1 Agent loop.")
    parser.add_argument("--target", type=Path, default=root / "data" / "target_rt_price.csv")
    parser.add_argument("--actuals", type=Path, default=root / "data" / "feature_actuals.csv")
    parser.add_argument("--forecasts", type=Path, default=root / "data" / "feature_forecasts.csv")
    parser.add_argument("--output", type=Path, default=root / "artifacts" / "research")
    parser.add_argument(
        "--question",
        default=("分析 actual_load、forecast_total_generation 与实时电价的同期和领先滞后关系，同时检查电价季节性。"),
    )
    parser.add_argument(
        "--revision",
        default=(
            "把最大滞后改为48个15分钟间隔，只分析actual_load和forecast_total_generation；"
            "关系分析保留Pearson、Spearman和领先滞后，并保留电价季节性。"
        ),
    )
    parser.add_argument(
        "--target-only-question",
        default=("只基于目标电价序列评估可预测性、平稳性、季节结构和朴素基线误差，不要使用任何外生变量。"),
    )
    parser.add_argument(
        "--minimum-request-timeout",
        type=float,
        default=VALIDATION_MINIMUM_TIMEOUT_SECONDS,
        help="Process-local timeout floor for a slow real-model acceptance run.",
    )
    return parser


def _progress(value: int, message: str) -> None:
    print(json.dumps({"progress": value, "message": message}, ensure_ascii=True), flush=True)


def enabled_function_names(plan: dict[str, Any]) -> list[str]:
    """Return the current function identifiers recorded by an executable plan."""

    return [str(step["function"]) for step in plan["steps"] if step["enabled"]]


def validation_settings(minimum_timeout_seconds: float) -> Settings:
    """Load real settings with a process-local timeout floor for acceptance runs.

    Real planning requests are substantially larger than a connectivity probe.
    A short desktop/backend override must not make the release validator fail
    before the configured endpoint has had a realistic chance to answer.  The
    user's ``.env`` file is never modified.
    """

    configured = get_settings()
    minimum = max(1.0, float(minimum_timeout_seconds))
    if configured.llm_timeout_seconds >= minimum:
        return configured
    return configured.model_copy(update={"llm_timeout_seconds": minimum})


def _require_interrupt(
    snapshot: ResearchLoopSnapshot,
    *,
    kinds: set[str],
    label: str,
) -> None:
    actual = snapshot.interrupt.kind if snapshot.interrupt else None
    if actual in kinds:
        return
    detail = str(snapshot.values.get("assistant_message") or snapshot.values.get("stop_reason") or "").strip()
    suffix = f" Detail: {detail}" if detail else ""
    raise RuntimeError(
        f"{label} expected one of {sorted(kinds)}, got phase={snapshot.phase!r}, interrupt={actual!r}.{suffix}"
    )


def _retry_recoverable_plan_error(
    *,
    coordinator: ResearchCoordinator,
    session_id: str,
    snapshot: ResearchLoopSnapshot,
    progress: Any,
) -> ResearchLoopSnapshot:
    if not snapshot.interrupt or snapshot.interrupt.kind != "plan_error":
        return snapshot
    retryable = any(bool(item.get("retryable")) for item in snapshot.values.get("feedback_packets", []))
    if not retryable or "retry" not in snapshot.interrupt.choices:
        return snapshot
    return coordinator.resume(session_id=session_id, action="retry", progress=progress)


def _finish_result_flow(
    *,
    coordinator: ResearchCoordinator,
    session_id: str,
    outcome: ResearchLoopSnapshot,
    config: Any,
    progress: Any,
) -> tuple[ResearchLoopSnapshot, str, str]:
    """Explain preserved evidence and close either accepted or limited results legally."""

    _require_interrupt(
        outcome,
        kinds={"result", "result_limitations"},
        label="Approved execution",
    )
    explanation = coordinator.submit_user_message(
        session_id=session_id,
        message="这些结果说明什么？只引用已有确定性证据，并明确限制。",
        study_config=config,
        progress=progress,
    )
    expected_kind = outcome.interrupt.kind if outcome.interrupt else "result"
    if explanation.interrupt and explanation.interrupt.kind == "response_error":
        explanation = coordinator.resume(
            session_id=session_id,
            action="retry",
            progress=progress,
        )
    _require_interrupt(explanation, kinds={expected_kind}, label="Evidence follow-up")
    answer = str(explanation.values.get("assistant_message") or "").strip()
    if not answer:
        raise RuntimeError("The real loop returned an empty evidence follow-up.")
    if expected_kind == "result":
        return explanation, answer, "accepted"

    stopped = coordinator.resume(session_id=session_id, action="stop", progress=progress)
    if stopped.phase != "stopped" or stopped.interrupt is not None:
        raise RuntimeError("Stopping after a limited result did not preserve a legal stopped terminal state.")
    if not stopped.values.get("latest_run"):
        raise RuntimeError("Stopping after a limited result discarded the completed run.")
    return stopped, answer, "limited"


def _run_real_scenario(
    *,
    coordinator: ResearchCoordinator,
    config: Any,
    question: str,
    expected_skill: str,
    revision: str | None,
) -> dict[str, Any]:
    session_id = f"phase1-real-{expected_skill}-{uuid4().hex[:10]}"
    approval = coordinator.submit_user_message(
        session_id=session_id,
        message=question,
        study_config=config,
        progress=_progress,
    )
    approval = _retry_recoverable_plan_error(
        coordinator=coordinator,
        session_id=session_id,
        snapshot=approval,
        progress=_progress,
    )
    _require_interrupt(approval, kinds={"plan_approval"}, label="Initial planning")

    planned = approval
    if revision:
        planned = coordinator.resume(
            session_id=session_id,
            action="modify",
            message=revision,
            progress=_progress,
        )
        planned = _retry_recoverable_plan_error(
            coordinator=coordinator,
            session_id=session_id,
            snapshot=planned,
            progress=_progress,
        )
        _require_interrupt(planned, kinds={"plan_approval"}, label="User plan revision")

    plan = planned.values["current_plan"]
    if plan["skill_name"] != expected_skill:
        raise RuntimeError(f"Expected Skill {expected_skill!r}, got {plan['skill_name']!r}.")
    if expected_skill == "price-forecastability-audit" and plan["selected_variables"]:
        raise RuntimeError("The target-only Skill selected exogenous variables.")
    if revision and int(plan["revision"]) < 2:
        raise RuntimeError("The real loop did not persist the user's revised plan.")

    outcome = coordinator.resume(session_id=session_id, action="approve", progress=_progress)
    final, explanation, terminal = _finish_result_flow(
        coordinator=coordinator,
        session_id=session_id,
        outcome=outcome,
        config=config,
        progress=_progress,
    )
    values = final.values
    latest = values["latest_run"]
    report_path = Path(latest["report_path"])
    methods_path = report_path.with_name("methods.md")
    if not report_path.is_file() or not methods_path.is_file():
        raise RuntimeError("The real loop did not produce report.md and methods.md.")
    enabled = enabled_function_names(plan)
    if not enabled:
        raise RuntimeError("The approved real-model plan contains no executable functions.")
    return {
        "status": "passed",
        "terminal": terminal,
        "thread_id": session_id,
        "skill": {"name": plan["skill_name"], "version": plan["skill_version"]},
        "plan_id": plan["plan_id"],
        "plan_revision": plan["revision"],
        "enabled_functions": enabled,
        "selected_variables": plan["selected_variables"],
        "run_id": latest["run_id"],
        "episode_id": values["loop_cursor"]["episode_id"],
        "evaluated_iterations": values["budget"]["evaluated_iterations"],
        "evaluation": latest["evaluation"]["decision"],
        "report_path": str(report_path),
        "methods_path": str(methods_path),
        "artifact_directory": latest["artifact_directory"],
        "explanation": explanation,
        "feedback_count": len(values["feedback_packets"]),
        "loop_records": values.get("loop_records", []),
    }


def main() -> int:
    args = build_parser().parse_args()
    settings = validation_settings(args.minimum_request_timeout)
    if not settings.llm_base_url or not settings.llm_model:
        raise SystemExit("Real model Base URL and model name are required.")

    target_only_config = infer_study_context(
        target_path=args.target,
        output_directory=args.output,
    )
    exogenous_config = infer_study_context(
        target_path=args.target,
        actuals_path=args.actuals,
        forecasts_path=args.forecasts,
        output_directory=args.output,
    )
    coordinator = ResearchCoordinator()
    try:
        scenarios = [
            _run_real_scenario(
                coordinator=coordinator,
                config=target_only_config,
                question=args.target_only_question,
                expected_skill="price-forecastability-audit",
                revision=None,
            ),
            _run_real_scenario(
                coordinator=coordinator,
                config=exogenous_config,
                question=args.question,
                expected_skill="price-exogenous-eda",
                revision=args.revision,
            ),
        ]
        evidence = {
            "status": "passed",
            "model": settings.llm_model,
            "structured_mode": "native",
            "request_timeout_seconds": settings.llm_timeout_seconds,
            "scenarios": scenarios,
        }
        acceptance_directory = args.output.resolve().parent / "acceptance"
        acceptance_directory.mkdir(parents=True, exist_ok=True)
        evidence_path = acceptance_directory / f"phase1-loop-{uuid4().hex[:12]}.json"
        evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
        printable = {
            **evidence,
            "scenarios": [{**scenario, "explanation": scenario["explanation"][:200]} for scenario in scenarios],
        }
        print(json.dumps(printable, ensure_ascii=True, indent=2))
        return 0
    finally:
        coordinator.close()


if __name__ == "__main__":
    raise SystemExit(main())
