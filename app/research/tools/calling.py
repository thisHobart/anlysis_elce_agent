"""Pure compilation and feedback helpers for model-proposed tool turns."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

from app.llm.gateway import ModelMessage, ModelToolTurn
from app.research.agent.errors import ResearchPlanValidationError
from app.research.agent.schemas import EDAResearchScope
from app.research.schemas.study import StudyConfig
from app.research.tools.contracts import ToolCall, ToolCallGroupRecord, ToolResult
from app.research.tools.parameters import compile_function_parameters
from app.research.tools.policy import ToolPermissionError
from app.research.tools.recipes import RecipeRegistry
from app.research.tools.registry import ToolRegistry

LOCALLY_MANAGED_ARGUMENTS = frozenset(
    {"spike_iqr_multiplier", "outlier_iqr_multiplier", "min_observations", "max_lag_limit"}
)


class ToolTurnProtocolError(ValueError):
    """The provider call identities cannot be safely added to the conversation."""

    def __init__(self, code: str, message: str, *, observed: list[str] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.observed = observed or []


@dataclass(frozen=True)
class CompiledProposal:
    """One provider proposal and its atomic compiled calls or isolated failure."""

    provider_call_id: str
    requested_name: str
    requested_arguments: dict[str, Any]
    requested_version: str | None
    origin: Literal["direct", "recipe"]
    calls: tuple[ToolCall, ...]
    compiled_arguments: tuple[dict[str, Any], ...]
    error: Exception | None = None

    @property
    def group(self) -> ToolCallGroupRecord:
        return ToolCallGroupRecord(
            provider_call_id=self.provider_call_id,
            requested_name=self.requested_name,
            requested_version=self.requested_version,
            child_call_ids=[call.call_id for call in self.calls],
            origin=self.origin,
            status="failed" if self.error is not None else "pending",
        )


@dataclass(frozen=True)
class CompiledToolBatch:
    """Deterministic result of compiling a provider tool turn."""

    proposals: tuple[CompiledProposal, ...]
    calls: tuple[ToolCall, ...]
    groups: dict[str, ToolCallGroupRecord]
    exceeds_budget: bool

    @property
    def failures(self) -> tuple[CompiledProposal, ...]:
        return tuple(proposal for proposal in self.proposals if proposal.error is not None)


def compile_dynamic_call(
    *,
    scope: EDAResearchScope,
    study_config: StudyConfig,
    registry: ToolRegistry,
    name: str,
    arguments: dict[str, Any],
    sequence: int,
) -> ToolCall:
    """Compile one proposal without graph, coordinator, persistence, or model I/O."""

    if name != "data_quality" and name not in scope.authorized_functions:
        raise ToolPermissionError(f"研究范围未授权工具：{name}")
    spec = registry.get(name)
    supplied_policy_values = sorted(set(arguments).intersection(LOCALLY_MANAGED_ARGUMENTS))
    if supplied_policy_values:
        raise ResearchPlanValidationError(
            f"{name} 的安全参数由本地配置注入，模型不得提供："
            + ", ".join(supplied_policy_values)
        )
    supplied_variables = list(dict.fromkeys(arguments.get("variables") or []))
    unknown_variables = sorted(set(supplied_variables).difference(scope.authorized_variables))
    if unknown_variables:
        raise ToolPermissionError(f"工具参数包含未授权变量：{', '.join(unknown_variables)}")
    compiled = compile_function_parameters(
        name,
        dict(arguments),
        selected_variables=supplied_variables,
        config=study_config,
        enabled=True,
    )
    compiled.pop("max_lag_limit", None)
    payload = {"name": name, "version": spec.version, "arguments": compiled}
    work_id = hashlib.sha256(
        json.dumps(
            {"data_fingerprint": scope.data_fingerprint, "payload": payload},
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:24]
    call_id = hashlib.sha256(
        json.dumps(
            {"scope_id": scope.scope_id, "sequence": sequence, "work_id": work_id},
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:24]
    return ToolCall(
        call_id=call_id,
        work_id=work_id,
        step_id=f"S{sequence}",
        name=name,
        version=spec.version,
        arguments=compiled,
    )


def compile_tool_turn(
    turn: ModelToolTurn,
    *,
    scope: EDAResearchScope,
    study_config: StudyConfig,
    registry: ToolRegistry,
    recipes: RecipeRegistry,
    existing_provider_call_ids: set[str] | frozenset[str] = frozenset(),
    next_sequence: int = 1,
    remaining_tool_calls: int | None = None,
) -> CompiledToolBatch:
    """Validate provider identities, expand recipes, and compile independent proposals."""

    provider_ids = [str(call.call_id or "") for call in turn.tool_calls]
    if any(not provider_id.strip() for provider_id in provider_ids):
        raise ToolTurnProtocolError("missing_provider_call_id", "模型返回了空 provider call_id。")
    if len(provider_ids) != len(set(provider_ids)):
        raise ToolTurnProtocolError("duplicate_provider_call_id", "模型在同一回合返回了重复 provider call_id。")
    reused = sorted(set(provider_ids).intersection(existing_provider_call_ids))
    if reused:
        raise ToolTurnProtocolError(
            "reused_provider_call_id",
            "模型重复使用了此前回合的 provider call_id。",
            observed=reused,
        )

    sequence = next_sequence
    proposals: list[CompiledProposal] = []
    calls: list[ToolCall] = []
    groups: dict[str, ToolCallGroupRecord] = {}
    for proposal in turn.tool_calls:
        provider_id = str(proposal.call_id)
        origin: Literal["direct", "recipe"] = "direct"
        requested_version: str | None = None
        proposal_calls: list[ToolCall] = []
        try:
            if proposal.name in recipes.names:
                origin = "recipe"
                if proposal.arguments:
                    raise ResearchPlanValidationError(f"分析配方 {proposal.name} 不接受参数")
                recipe = recipes.get(proposal.name)
                requested_version = recipe.version
                names_and_arguments = [(name, {}) for name in recipe.functions]
            else:
                requested_version = registry.get(proposal.name).version
                names_and_arguments = [(proposal.name, proposal.arguments)]
            candidate_sequence = sequence
            for name, arguments in names_and_arguments:
                proposal_calls.append(
                    compile_dynamic_call(
                        scope=scope,
                        study_config=study_config,
                        registry=registry,
                        name=name,
                        arguments=arguments,
                        sequence=candidate_sequence,
                    )
                )
                candidate_sequence += 1
            sequence = candidate_sequence
            calls.extend(proposal_calls)
            compiled = CompiledProposal(
                provider_call_id=provider_id,
                requested_name=proposal.name,
                requested_arguments=dict(proposal.arguments),
                requested_version=requested_version,
                origin=origin,
                calls=tuple(proposal_calls),
                compiled_arguments=tuple(dict(call.arguments) for call in proposal_calls),
            )
        except (ResearchPlanValidationError, ToolPermissionError, TypeError, ValueError) as exc:
            compiled = CompiledProposal(
                provider_call_id=provider_id,
                requested_name=proposal.name,
                requested_arguments=dict(proposal.arguments),
                requested_version=requested_version,
                origin=origin,
                calls=(),
                compiled_arguments=(),
                error=exc,
            )
        proposals.append(compiled)
        groups[provider_id] = compiled.group

    exceeds_budget = remaining_tool_calls is not None and len(calls) > remaining_tool_calls
    return CompiledToolBatch(
        proposals=tuple(proposals),
        calls=tuple(calls),
        groups=groups,
        exceeds_budget=exceeds_budget,
    )


def compact_tool_value(value: Any, *, depth: int = 0) -> Any:
    """Bound tool evidence before it is returned to the model."""

    if depth >= 5:
        return "<truncated>"
    if isinstance(value, dict):
        return {str(key): compact_tool_value(item, depth=depth + 1) for key, item in list(value.items())[:40]}
    if isinstance(value, list):
        compact = [compact_tool_value(item, depth=depth + 1) for item in value[:20]]
        if len(value) > 20:
            compact.append({"omitted_items": len(value) - 20})
        return compact
    if isinstance(value, str) and len(value) > 1000:
        return f"{value[:997]}..."
    return value


def tool_result_payload(call: ToolCall, result: ToolResult, *, status: Literal["completed", "reused"]) -> dict[str, Any]:
    """Create the stable child payload returned for one atomic call."""

    return {
        "function": call.name,
        "arguments": call.arguments,
        "status": status,
        "result_key": result.output.result_key,
        "value": compact_tool_value(result.output.value),
        "output_hash": result.output_hash,
        "duplicate_notice": (
            {
                "code": "duplicate_work_id",
                "message": "相同数据、函数版本和参数的结果已复用，未重复计算。",
                "work_id": call.work_id,
            }
            if status == "reused"
            else None
        ),
    }


def build_tool_feedback(group: ToolCallGroupRecord, results: list[dict[str, Any]]) -> ModelMessage:
    """Close one provider call with all of its atomic child results."""

    return ModelMessage(
        role="tool",
        tool_call_id=group.provider_call_id,
        content=json.dumps(
            {
                "requested": group.requested_name,
                "requested_version": group.requested_version,
                "results": results,
            },
            ensure_ascii=False,
            allow_nan=False,
        ),
    )


def build_tool_rejection_feedback(
    provider_call_id: str,
    *,
    error: str,
    retryable: bool,
) -> ModelMessage:
    """Close one rejected provider call with a matched protocol message."""

    return ModelMessage(
        role="tool",
        tool_call_id=provider_call_id,
        content=json.dumps(
            {"status": "rejected", "error": error, "retryable": retryable},
            ensure_ascii=False,
        ),
    )


def build_tool_error_feedback(
    group: ToolCallGroupRecord,
    *,
    retryable: bool,
    results: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> ModelMessage:
    """Close an executed provider group that ended in failure or cancellation."""

    return ModelMessage(
        role="tool",
        tool_call_id=group.provider_call_id,
        content=json.dumps(
            {
                "requested": group.requested_name,
                "requested_version": group.requested_version,
                "status": group.status,
                "retryable": retryable,
                "results": results,
                "errors": errors,
            },
            ensure_ascii=False,
        ),
    )
