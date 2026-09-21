"""Debug one P1 tool-selection turn without starting ResearchCoordinator or LangGraph.

Input is a JSON object containing ``scope``, ``study_config_path`` and, depending on
the command, ``messages``, ``turn`` or ``call``.  Every command uses the same request,
compiler, executor and validator as the production research graph.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.llm.factory import build_model_gateway
from app.llm.gateway import ModelMessage, ModelToolTurn
from app.research.agent.dynamic import build_tool_request
from app.research.agent.schemas import EDAResearchScope
from app.research.application.execution import EDAExecutionService
from app.research.schemas.study import load_study_config
from app.research.skills.registry import SkillRegistry
from app.research.tools.calling import (
    build_tool_feedback,
    build_tool_rejection_feedback,
    compile_tool_turn,
    tool_result_payload,
)
from app.research.tools.contracts import ToolCall
from app.research.tools.eda.functions import build_eda_tool_registry
from app.research.tools.recipes import build_recipe_registry


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("request", "select", "compile", "execute", "round"))
    parser.add_argument("--input", type=Path, required=True, help="JSON debug case")
    parser.add_argument("--output", type=Path, help="write JSON result instead of stdout")
    return parser.parse_args()


def _read_case(path: Path) -> dict[str, Any]:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("debug input must contain one JSON object")
    return payload


def _write(payload: dict[str, Any], path: Path | None) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    if path is None:
        print(encoded)
        return
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(encoded + "\n", encoding="utf-8")
    print(f"调试结果已写入：{destination}")


def _components(case: dict[str, Any]):
    scope = EDAResearchScope.model_validate(case["scope"])
    config = load_study_config(case["study_config_path"])
    registry = build_eda_tool_registry()
    recipes = build_recipe_registry()
    skills = SkillRegistry.default()
    skill = skills.get(scope.skill_name)
    messages = [ModelMessage.model_validate(item) for item in case.get("messages", [])]
    return scope, config, registry, recipes, skills, skill, messages


def _request_payload(request) -> dict[str, Any]:
    return {
        "messages": [message.model_dump(mode="json") for message in request.messages],
        "tools": list(request.tools),
        "purpose": request.purpose.value,
    }


def _compile(case: dict[str, Any], turn: ModelToolTurn, *, scope, config, registry, recipes):
    return compile_tool_turn(
        turn,
        scope=scope,
        study_config=config,
        registry=registry,
        recipes=recipes,
        existing_provider_call_ids=set(case.get("existing_provider_call_ids", [])),
        next_sequence=int(case.get("next_sequence", 1)),
        remaining_tool_calls=case.get("remaining_tool_calls"),
    )


def _batch_payload(batch) -> dict[str, Any]:
    return {
        "calls": [call.model_dump(mode="json") for call in batch.calls],
        "groups": {key: value.model_dump(mode="json") for key, value in batch.groups.items()},
        "proposals": [
            {
                "provider_call_id": proposal.provider_call_id,
                "requested_name": proposal.requested_name,
                "requested_arguments": proposal.requested_arguments,
                "requested_version": proposal.requested_version,
                "origin": proposal.origin,
                "compiled_arguments": list(proposal.compiled_arguments),
                "error": str(proposal.error) if proposal.error is not None else None,
            }
            for proposal in batch.proposals
        ],
        "exceeds_budget": batch.exceeds_budget,
    }


def main() -> int:
    args = _arguments()
    case = _read_case(args.input)
    scope, config, registry, recipes, skills, skill, messages = _components(case)
    request = build_tool_request(
        scope=scope,
        skill=skill,
        messages=messages,
        registry=registry,
        recipes=recipes,
    )

    if args.command == "request":
        _write({"request": _request_payload(request)}, args.output)
        return 0

    if args.command in {"select", "round"}:
        turn = build_model_gateway().invoke_tool_turn(**request.as_gateway_kwargs())
    else:
        turn = ModelToolTurn.model_validate(case["turn"]) if "turn" in case else None

    if args.command == "select":
        assert turn is not None
        _write({"request": _request_payload(request), "turn": turn.model_dump(mode="json")}, args.output)
        return 0

    if args.command == "execute":
        call = ToolCall.model_validate(case["call"])
        execution = EDAExecutionService(registry=registry, skills=skills)
        prepared = execution.prepare_scope(scope=scope, study_config=config)
        result = execution.executor.execute(
            call,
            context=prepared.context,
            policy=prepared.policy,
            data_fingerprint=prepared.data_fingerprint,
        )
        execution.validate_scope_result(scope=scope, call=call, result=result)
        _write({"call": call.model_dump(mode="json"), "result": result.model_dump(mode="json")}, args.output)
        return 0

    assert turn is not None
    batch = _compile(case, turn, scope=scope, config=config, registry=registry, recipes=recipes)
    if args.command == "compile":
        _write({"turn": turn.model_dump(mode="json"), "compiled": _batch_payload(batch)}, args.output)
        return 0

    execution = EDAExecutionService(registry=registry, skills=skills)
    prepared = execution.prepare_scope(scope=scope, study_config=config)
    results_by_call: dict[str, Any] = {}
    for call in batch.calls:
        result = execution.executor.execute(
            call,
            context=prepared.context,
            policy=prepared.policy,
            data_fingerprint=prepared.data_fingerprint,
        )
        execution.validate_scope_result(scope=scope, call=call, result=result)
        results_by_call[call.call_id] = result

    feedback = []
    for proposal in batch.proposals:
        if proposal.error is not None:
            message = build_tool_rejection_feedback(
                proposal.provider_call_id,
                error=str(proposal.error),
                retryable=True,
            )
        else:
            group = batch.groups[proposal.provider_call_id]
            payloads = [
                tool_result_payload(call, results_by_call[call.call_id], status="completed")
                for call in proposal.calls
            ]
            message = build_tool_feedback(group, payloads)
        feedback.append(message.model_dump(mode="json"))
    _write(
        {
            "request": _request_payload(request),
            "turn": turn.model_dump(mode="json"),
            "compiled": _batch_payload(batch),
            "results": [result.model_dump(mode="json") for result in results_by_call.values()],
            "tool_messages": feedback,
        },
        args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
