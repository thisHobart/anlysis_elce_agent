"""Run the explicit P1 tool-selection cases against the configured real model."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.llm.factory import build_model_gateway
from app.research.agent.dynamic import DynamicAnalysisAgent, build_tool_request
from app.research.agent.schemas import EDAResearchScope
from app.research.evaluation.tool_calling import ToolSelectionCase, score_tool_turn
from app.research.skills.registry import SkillRegistry
from app.research.tools.eda.functions import build_eda_tool_registry
from app.research.tools.recipes import build_recipe_registry

DEFAULT_CASES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "tool_calling" / "cases.json"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, required=True, help="JSON with base_scope, quality_report and data_profile")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _json(path: Path) -> Any:
    return json.loads(path.resolve().read_text(encoding="utf-8"))


def main() -> int:
    args = _arguments()
    context = _json(args.context)
    cases = [ToolSelectionCase.model_validate(item) for item in _json(args.cases)]
    registry = build_eda_tool_registry()
    recipes = build_recipe_registry()
    skills = SkillRegistry.default()
    gateway = build_model_gateway()
    dynamic = DynamicAnalysisAgent(gateway=gateway, tools=registry, recipes=recipes)
    results: list[dict[str, Any]] = []

    for case in cases:
        try:
            scope = EDAResearchScope.model_validate(
                {
                    **context["base_scope"],
                    "question": case.question,
                    "objective": case.question,
                    "authorized_functions": case.authorized_functions,
                    "authorized_variables": case.authorized_variables,
                }
            )
            messages = dynamic.initial_messages(
                scope=scope,
                quality_report=context["quality_report"],
                data_profile=context.get("data_profile"),
            )
            request = build_tool_request(
                scope=scope,
                skill=skills.get(scope.skill_name),
                messages=messages,
                registry=registry,
                recipes=recipes,
            )
            turn = gateway.invoke_tool_turn(**request.as_gateway_kwargs())
            score = score_tool_turn(case, turn)
            results.append(
                {
                    "case": case.model_dump(mode="json"),
                    "turn": turn.model_dump(mode="json"),
                    "score": score.model_dump(mode="json"),
                    "infrastructure_error": None,
                }
            )
        except Exception as exc:  # noqa: BLE001 - isolate each real provider case
            results.append(
                {
                    "case": case.model_dump(mode="json"),
                    "turn": None,
                    "score": None,
                    "infrastructure_error": f"{type(exc).__name__}: {exc}",
                }
            )

    scored = [item for item in results if item["score"] is not None]
    passed = sum(bool(item["score"]["passed"]) for item in scored)
    report = {
        "schema_version": 1,
        "executed_at": datetime.now(UTC).isoformat(),
        "model": getattr(gateway, "model_name", ""),
        "attempted": len(results),
        "scored": len(scored),
        "passed": passed,
        "pass_rate": passed / len(scored) if scored else None,
        "infrastructure_failures": len(results) - len(scored),
        "results": results,
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    if args.output is None:
        print(encoded)
    else:
        destination = args.output.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(encoded + "\n", encoding="utf-8")
        print(f"工具选择评测报告已写入：{destination}")
    return 0 if len(scored) == len(results) and passed == len(scored) else 1


if __name__ == "__main__":
    raise SystemExit(main())
