"""Validate and execute approved function calls through the registry."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from time import perf_counter

from app.research.tools.contracts import ToolCall, ToolContext, ToolOutput, ToolResult
from app.research.tools.policy import ToolPolicy
from app.research.tools.registry import ToolRegistry, ToolRegistryError


class ToolExecutionError(RuntimeError):
    """A controlled tool call failed validation or execution."""


def output_fingerprint(output: ToolOutput) -> str:
    """Hash one tool output; the single rule used to write and to re-check output_hash."""

    payload = json.dumps(
        output.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class ToolExecutor:
    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def execute(
        self,
        call: ToolCall,
        *,
        context: ToolContext,
        policy: ToolPolicy,
        data_fingerprint: str = "000000000000",
    ) -> ToolResult:
        policy.authorize(call)
        spec = self.registry.get(call.name)
        if call.version != spec.version:
            raise ToolRegistryError(
                f"工具版本不匹配 {call.name}: plan={call.version}, installed={spec.version}"
            )
        try:
            arguments = spec.arguments_model.model_validate(call.arguments)
            wall_started = datetime.now(UTC).isoformat()
            started = perf_counter()
            output = spec.handler(context, arguments)
            wall_finished = datetime.now(UTC).isoformat()
        except (TypeError, ValueError) as exc:
            raise ToolExecutionError(f"工具 {call.name} 调用失败：{exc}") from exc
        return ToolResult(
            call=call,
            status="completed",
            output=output,
            started_at=wall_started,
            finished_at=wall_finished,
            duration_ms=round((perf_counter() - started) * 1000, 3),
            provider=spec.provider,
            tool_version=spec.version,
            data_fingerprint=data_fingerprint,
            output_hash=output_fingerprint(output),
        )
