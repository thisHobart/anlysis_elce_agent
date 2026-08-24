"""Validate and execute approved function calls through the registry."""

from __future__ import annotations

import hashlib
import json
from time import perf_counter

from app.research.tools.contracts import ToolCall, ToolContext, ToolResult
from app.research.tools.policy import ToolPolicy
from app.research.tools.registry import ToolRegistry, ToolRegistryError


class ToolExecutionError(RuntimeError):
    """A controlled tool call failed validation or execution."""


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
            started = perf_counter()
            output = spec.handler(context, arguments)
        except (TypeError, ValueError) as exc:
            raise ToolExecutionError(f"工具 {call.name} 调用失败：{exc}") from exc
        output_payload = json.dumps(
            output.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ).encode("utf-8")
        return ToolResult(
            call=call,
            status="completed",
            output=output,
            duration_ms=round((perf_counter() - started) * 1000, 3),
            provider=spec.provider,
            tool_version=spec.version,
            data_fingerprint=data_fingerprint,
            output_hash=hashlib.sha256(output_payload).hexdigest(),
        )
