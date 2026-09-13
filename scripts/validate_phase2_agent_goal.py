"""Run the configured model against the fixed P2 goal and validate its answer."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from app.config import Settings, get_settings
from app.llm.factory import build_model_gateway
from app.research.news import (
    P2_AGENT_GOAL_USER_PROMPT,
    JsonlCollectedNewsAdapter,
    MarketClock,
    evaluate_phase2_goal_agent,
    load_price_csv,
    run_news_price_study,
    run_phase2_goal_agent,
)

VALIDATION_MINIMUM_TIMEOUT_SECONDS = 120.0


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    fixtures = root / "tests" / "fixtures" / "news_price"
    parser = argparse.ArgumentParser(
        description="Qualify one real-model P2 goal answer against deterministic evidence."
    )
    parser.add_argument("--news", type=Path, default=fixtures / "synthetic_news.jsonl")
    parser.add_argument("--prices", type=Path, default=fixtures / "synthetic_prices.csv")
    parser.add_argument("--manifest", type=Path, default=fixtures / "fixture_manifest.yaml")
    parser.add_argument("--output", type=Path, default=root / "artifacts" / "acceptance")
    parser.add_argument("--as-of", default="2026-03-01T23:30:00+00:00")
    parser.add_argument("--goal-prompt", default=P2_AGENT_GOAL_USER_PROMPT)
    parser.add_argument(
        "--minimum-request-timeout",
        type=float,
        default=VALIDATION_MINIMUM_TIMEOUT_SECONDS,
    )
    return parser


def validation_settings(minimum_timeout_seconds: float) -> Settings:
    """Apply a process-local model timeout floor without editing the user's .env."""

    configured = get_settings()
    minimum = max(1.0, float(minimum_timeout_seconds))
    if configured.llm_timeout_seconds >= minimum:
        return configured
    return configured.model_copy(update={"llm_timeout_seconds": minimum})


def load_gold_effects(path: Path) -> dict[str, float]:
    """Read the answer key only for the evaluator, after Agent input is built."""

    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {str(item["fixture_id"]): float(item["delta"]) for item in manifest.get("effects", [])}


def _write_artifact(directory: Path, payload: dict[str, Any]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"phase2-agent-goal-{uuid4().hex[:12]}.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def main() -> int:
    args = build_parser().parse_args()
    settings = validation_settings(args.minimum_request_timeout)
    if not settings.llm_base_url or not settings.llm_model:
        raise SystemExit("Real model Base URL and model name are required.")

    clock = MarketClock(market="TEST_MARKET", timezone="UTC", interval_minutes=30)
    prices = load_price_csv(args.prices, clock)
    study = run_news_price_study(
        adapter=JsonlCollectedNewsAdapter(args.news),
        prices=prices,
        as_of=datetime.fromisoformat(args.as_of),
        axis="effective",
    )
    started_at = datetime.now(UTC)
    try:
        run = run_phase2_goal_agent(
            build_model_gateway(settings),
            study,
            goal_prompt=args.goal_prompt,
        )
        # Deliberately load the fixture answer key after the model call.  It is not part of
        # run.input_payload and therefore cannot leak the injected deltas to the Agent.
        validation = evaluate_phase2_goal_agent(
            study,
            run,
            gold_effects=load_gold_effects(args.manifest),
        )
        payload: dict[str, Any] = {
            "status": "passed" if validation.passed else "failed",
            "started_at": started_at.isoformat(),
            "completed_at": datetime.now(UTC).isoformat(),
            "model": settings.llm_model,
            "request_timeout_seconds": settings.llm_timeout_seconds,
            "run": run.model_dump(mode="json"),
            "validation": {
                **validation.model_dump(mode="json"),
                "passed": validation.passed,
                "overall_assessment": validation.overall_assessment,
                "failed_check_codes": [check.code for check in validation.failed_checks],
            },
        }
    except Exception as exc:  # noqa: BLE001 - qualification failures must leave evidence
        payload = {
            "status": "failed",
            "started_at": started_at.isoformat(),
            "completed_at": datetime.now(UTC).isoformat(),
            "model": settings.llm_model,
            "request_timeout_seconds": settings.llm_timeout_seconds,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    artifact_path = _write_artifact(args.output, payload)
    printable = {
        "status": payload["status"],
        "model": payload["model"],
        "artifact_path": str(artifact_path.resolve()),
        "validation": payload.get("validation"),
        "error_type": payload.get("error_type"),
        "error": payload.get("error"),
    }
    print(json.dumps(printable, ensure_ascii=True, indent=2))
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
