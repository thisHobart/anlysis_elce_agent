"""Run the persistent Phase 1 loop with the configured real model and data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.config import get_settings
from app.research.application.coordinator import ResearchCoordinator
from app.research.data.inference import infer_study_context


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Validate the real persistent Phase 1 Agent loop.")
    parser.add_argument("--target", type=Path, default=root / "data" / "target_rt_price.csv")
    parser.add_argument("--actuals", type=Path, default=root / "data" / "feature_actuals.csv")
    parser.add_argument("--forecasts", type=Path, default=root / "data" / "feature_forecasts.csv")
    parser.add_argument("--output", type=Path, default=root / "artifacts" / "research")
    parser.add_argument(
        "--question",
        default=(
            "分析 actual_load、forecast_total_generation 与实时电价的同期和领先滞后关系，"
            "同时检查电价季节性。"
        ),
    )
    parser.add_argument(
        "--revision",
        default=(
            "把最大滞后改为48个15分钟间隔，只分析actual_load和forecast_total_generation；"
            "关系分析保留Pearson、Spearman和领先滞后，并保留电价季节性。"
        ),
    )
    return parser


def _progress(value: int, message: str) -> None:
    print(json.dumps({"progress": value, "message": message}, ensure_ascii=True), flush=True)


def enabled_function_names(plan: dict[str, Any]) -> list[str]:
    """Return the current function identifiers recorded by an executable plan."""

    return [str(step["function"]) for step in plan["steps"] if step["enabled"]]


def main() -> int:
    args = build_parser().parse_args()
    settings = get_settings()
    if not settings.llm_base_url or not settings.llm_model:
        raise SystemExit("Real model Base URL and model name are required.")

    config = infer_study_context(
        target_path=args.target,
        actuals_path=args.actuals,
        forecasts_path=args.forecasts,
        output_directory=args.output,
    )
    coordinator = ResearchCoordinator()
    thread_id = f"phase1-real-{uuid4().hex[:12]}"
    approval = coordinator.submit_user_message(
        session_id=thread_id,
        message=args.question,
        study_config=config,
        progress=_progress,
    )
    if not approval.interrupt or approval.interrupt.kind != "plan_approval":
        raise RuntimeError("The real loop did not reach plan approval.")

    revised = coordinator.resume(
        session_id=thread_id,
        action="modify",
        message=args.revision,
        progress=_progress,
    )
    if not revised.interrupt or revised.interrupt.kind != "plan_approval":
        raise RuntimeError("The real loop did not produce an approved user revision.")

    outcome = coordinator.resume(session_id=thread_id, action="approve", progress=_progress)
    if outcome.interrupt and outcome.interrupt.kind == "result_limitations":
        outcome = coordinator.resume(
            session_id=thread_id,
            action="accept_limitations",
            progress=_progress,
        )
    if not outcome.interrupt or outcome.interrupt.kind != "result":
        raise RuntimeError("The real loop did not reach a result interrupt.")

    followup = coordinator.submit_user_message(
        session_id=thread_id,
        message="这些结果说明什么？只引用已有确定性证据，并明确限制。",
        study_config=config,
        progress=_progress,
    )
    if not followup.interrupt or followup.interrupt.kind != "result":
        raise RuntimeError("The real loop did not answer the result follow-up.")

    values = followup.values
    latest = values["latest_run"]
    plan = values["current_plan"]
    evidence = {
        "status": "passed",
        "model": settings.llm_model,
        "structured_mode": "native",
        "thread_id": thread_id,
        "skill": {"name": plan["skill_name"], "version": plan["skill_version"]},
        "plan_id": plan["plan_id"],
        "plan_revision": plan["revision"],
        "enabled_functions": enabled_function_names(plan),
        "selected_variables": plan["selected_variables"],
        "run_id": latest["run_id"],
        "episode_id": values["loop_cursor"]["episode_id"],
        "evaluated_iterations": values["budget"]["evaluated_iterations"],
        "evaluation": latest["evaluation"]["decision"],
        "report_path": latest["report_path"],
        "artifact_directory": latest["artifact_directory"],
        "explanation": values["assistant_message"],
        "feedback_count": len(values["feedback_packets"]),
        "loop_records": values.get("loop_records", []),
    }
    evidence_path = Path(latest["artifact_directory"]) / "phase1_loop_acceptance.json"
    evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**evidence, "explanation": values["assistant_message"][:200]}, ensure_ascii=True, indent=2))
    coordinator.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
