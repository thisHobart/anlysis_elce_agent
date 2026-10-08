"""Deterministic scoring for raw model tool-selection turns."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.llm.gateway import ModelToolTurn


class ToolSelectionCase(BaseModel):
    """One reviewable expectation for a raw model turn."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    question: str
    authorized_functions: list[str]
    authorized_variables: list[str] = Field(default_factory=list)
    expected_outcome: Literal["tools", "no_tools"] = "tools"
    required_tools: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    expected_arguments: dict[str, dict[str, Any]] = Field(default_factory=dict)
    forbidden_argument_keys: dict[str, list[str]] = Field(default_factory=dict)


class ToolSelectionScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    passed: bool
    expected_outcome: Literal["tools", "no_tools"]
    actual_tools: list[str]
    missing_tools: list[str] = Field(default_factory=list)
    disallowed_tools: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    argument_errors: list[str] = Field(default_factory=list)


def score_tool_turn(case: ToolSelectionCase, turn: ModelToolTurn) -> ToolSelectionScore:
    """Score names and model-owned arguments without invoking research evaluation."""

    actual_tools = [call.name for call in turn.tool_calls]
    missing = sorted(set(case.required_tools).difference(actual_tools))
    allowed = set(case.allowed_tools or case.required_tools)
    disallowed = sorted({name for name in actual_tools if allowed and name not in allowed})
    forbidden = sorted(set(actual_tools).intersection(case.forbidden_tools))
    argument_errors: list[str] = []
    calls_by_name = {call.name: call for call in turn.tool_calls}
    for name, expected in case.expected_arguments.items():
        call = calls_by_name.get(name)
        if call is None:
            continue
        for key, value in expected.items():
            if call.arguments.get(key) != value:
                argument_errors.append(
                    f"{name}.{key}: expected {value!r}, observed {call.arguments.get(key)!r}"
                )
    for name, keys in case.forbidden_argument_keys.items():
        call = calls_by_name.get(name)
        if call is None:
            continue
        for key in keys:
            if key in call.arguments:
                argument_errors.append(f"{name}.{key}: model must not provide this locally managed argument")
    outcome_error = (case.expected_outcome == "tools" and not turn.tool_calls) or (
        case.expected_outcome == "no_tools" and bool(turn.tool_calls)
    )
    return ToolSelectionScore(
        case_id=case.case_id,
        passed=not (outcome_error or missing or disallowed or forbidden or argument_errors),
        expected_outcome=case.expected_outcome,
        actual_tools=actual_tools,
        missing_tools=missing,
        disallowed_tools=disallowed,
        forbidden_tools=forbidden,
        argument_errors=argument_errors,
    )
