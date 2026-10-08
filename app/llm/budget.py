"""Purpose-specific request accounting and per-route output limits."""

from __future__ import annotations

import json
import math
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict

from app.llm.context_safety import conservative_token_count
from app.llm.gateway import ModelConfigurationError, ModelContextLimitError, ModelMessage
from app.llm.model_profiles import ResolvedModelProfile
from app.llm.runtime_settings import LLMRuntimeSettings, OperationTokenPolicy


class ModelRequestPurpose(StrEnum):
    DIALOGUE = "dialogue"
    EDA_PLANNING = "eda_planning"
    EDA_PLANNING_RECOVERY = "eda_planning_recovery"
    EDA_ANALYSIS = "eda_analysis"
    RESULT_EXPLANATION = "result_explanation"
    NEWS_EXTRACTION = "news_extraction"
    GENERIC = "generic"


class RequestBudget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    purpose: ModelRequestPurpose
    route_key: str
    profile_source: str
    profile_verified: bool
    message_tokens: int
    schema_tokens: int
    tool_tokens: int
    framing_tokens: int
    input_tokens: int
    target_output_tokens: int
    minimum_output_tokens: int
    safety_tokens: int
    context_window_tokens: int | None
    model_max_output_tokens: int | None
    effective_output_tokens: int
    compaction_level: int = 0


def request_input_tokens(
    messages: list[ModelMessage],
    *,
    schema: type[BaseModel] | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> tuple[int, int, int, int]:
    if any(not isinstance(message, ModelMessage) for message in messages):
        raise ModelConfigurationError("模型边界只接受类型化 ModelMessage 消息。")
    message_tokens = sum(conservative_token_count(message.content) + 8 for message in messages)
    schema_tokens = 0
    if schema is not None:
        schema_text = json.dumps(
            schema.model_json_schema(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        schema_tokens = conservative_token_count(schema_text)
    tool_tokens = 0
    if tools:
        tool_text = json.dumps(tools, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        tool_tokens = conservative_token_count(tool_text)
    framing_tokens = 16 + 8 * len(messages) + (16 if schema is not None else 0) + 12 * len(tools or [])
    return message_tokens, schema_tokens, tool_tokens, framing_tokens


class ModelBudgetManager:
    def __init__(
        self,
        *,
        resolved_profile: ResolvedModelProfile,
        runtime: LLMRuntimeSettings | None = None,
    ) -> None:
        self.resolved_profile = resolved_profile
        self.runtime = runtime or LLMRuntimeSettings()

    def policy(self, purpose: ModelRequestPurpose) -> OperationTokenPolicy:
        policies = self.runtime.operation_policies
        return getattr(policies, purpose.value)

    def budget(
        self,
        messages: list[ModelMessage],
        *,
        purpose: ModelRequestPurpose = ModelRequestPurpose.GENERIC,
        schema: type[BaseModel] | None = None,
        tools: list[dict[str, Any]] | None = None,
        compaction_level: int = 0,
    ) -> RequestBudget:
        if purpose == ModelRequestPurpose.EDA_PLANNING_RECOVERY:
            compaction_level = max(1, compaction_level)
        message_tokens, schema_tokens, tool_tokens, framing_tokens = request_input_tokens(
            messages,
            schema=schema,
            tools=tools,
        )
        input_tokens = message_tokens + schema_tokens + tool_tokens + framing_tokens
        policy = self.policy(purpose)
        profile = self.resolved_profile.profile
        if profile is None:
            if input_tokens > self.runtime.unverified_input_tokens:
                raise ModelContextLimitError(
                    "未验证模型的短上下文预算不足："
                    f"input={input_tokens}, safe_input_limit={self.runtime.unverified_input_tokens}"
                )
            return RequestBudget(
                purpose=purpose,
                route_key=self.resolved_profile.route_key,
                profile_source=self.resolved_profile.source,
                profile_verified=False,
                message_tokens=message_tokens,
                schema_tokens=schema_tokens,
                tool_tokens=tool_tokens,
                framing_tokens=framing_tokens,
                input_tokens=input_tokens,
                target_output_tokens=policy.target_output_tokens,
                minimum_output_tokens=policy.minimum_output_tokens,
                safety_tokens=0,
                context_window_tokens=None,
                model_max_output_tokens=None,
                effective_output_tokens=policy.target_output_tokens,
                compaction_level=compaction_level,
            )
        safety = min(4096, max(512, math.ceil(profile.context_window_tokens * 0.05)))
        available = profile.context_window_tokens - input_tokens - safety
        effective = min(profile.max_output_tokens, policy.target_output_tokens, max(0, available))
        if effective < policy.minimum_output_tokens:
            raise ModelContextLimitError(
                "完整模型请求超过任务预算："
                f"input={input_tokens}, output_available={max(0, available)}, "
                f"minimum_output={policy.minimum_output_tokens}, safe_buffer={safety}, "
                f"context_window={profile.context_window_tokens}"
            )
        return RequestBudget(
            purpose=purpose,
            route_key=self.resolved_profile.route_key,
            profile_source=self.resolved_profile.source,
            profile_verified=True,
            message_tokens=message_tokens,
            schema_tokens=schema_tokens,
            tool_tokens=tool_tokens,
            framing_tokens=framing_tokens,
            input_tokens=input_tokens,
            target_output_tokens=policy.target_output_tokens,
            minimum_output_tokens=policy.minimum_output_tokens,
            safety_tokens=safety,
            context_window_tokens=profile.context_window_tokens,
            model_max_output_tokens=profile.max_output_tokens,
            effective_output_tokens=effective,
            compaction_level=compaction_level,
        )
