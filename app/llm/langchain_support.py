"""Provider-neutral helpers shared by LangChain-backed model adapters."""

from __future__ import annotations

from typing import Any

from app.llm import compat
from app.llm.gateway import (
    ModelConfigurationError,
    ModelMessage,
    ModelThinkingError,
    ModelToolCall,
)


def response_text(response: Any) -> str:
    """Return visible text blocks without treating reasoning blocks as answers."""

    content = getattr(response, "content", response)
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            block_type = str(item.get("type", "")).casefold()
            if block_type in {"reasoning", "thinking"}:
                continue
            value = item.get("text") or item.get("content")
            if value:
                parts.append(str(value))
        return "".join(parts).strip()
    return str(content).strip()


def reasoning_text(response: Any) -> str:
    """Read common provider reasoning fields without retaining the raw response."""

    direct = getattr(response, "reasoning_content", None)
    if direct:
        return str(direct).strip()
    additional = getattr(response, "additional_kwargs", None)
    if isinstance(additional, dict):
        for key in ("reasoning_content", "reasoning"):
            value = additional.get(key)
            if value:
                return str(value).strip()
    content = getattr(response, "content", None)
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            block_type = str(item.get("type", "")).casefold()
            if block_type in {"reasoning", "thinking"}:
                value = item.get("text") or item.get("content")
                if value:
                    return str(value).strip()
    return ""


def reject_nonempty_thinking(response: Any) -> None:
    """Fail closed when a provider still emits a reasoning trace."""

    if reasoning_text(response):
        raise ModelThinkingError("模型仍返回 reasoning_content，思考模式禁用失败。")
    if compat.thinking_fragments(response_text(response)):
        raise ModelThinkingError("模型仍返回非空 <think> 内容，思考模式禁用失败。")


def visible_text(response: Any) -> str:
    """Return the answer with any inline reasoning trace removed."""

    return compat.strip_thinking(response_text(response))


def normalized_calls(raw_calls: Any) -> list[ModelToolCall]:
    """Normalize native function calls while preserving their provider call id."""

    return [
        ModelToolCall(
            name=str(item.get("name", "")),
            arguments=item.get("args") or item.get("arguments") or {},
            call_id=str(item.get("id")) if item.get("id") else None,
        )
        for item in (raw_calls or [])
        if isinstance(item, dict)
    ]


def tool_names(tools: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for tool in tools:
        function = tool.get("function", tool)
        if isinstance(function, dict) and function.get("name"):
            names.add(str(function["name"]))
    return names


def transport_messages(messages: list[ModelMessage]) -> list[Any]:
    """Convert canonical Agent messages to LangChain message objects."""

    try:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
    except ImportError as exc:  # pragma: no cover - installed with provider integrations
        raise ModelConfigurationError("langchain-core 尚未安装。") from exc

    converted: list[Any] = []
    for message in messages:
        if not isinstance(message, ModelMessage):
            raise ModelConfigurationError("研究 Agent 消息必须使用 ModelMessage 协议。")
        if message.role == "system":
            converted.append(SystemMessage(content=message.content))
        elif message.role == "user":
            converted.append(HumanMessage(content=message.content))
        elif message.role == "assistant":
            converted.append(
                AIMessage(
                    content=message.content,
                    tool_calls=[
                        {
                            "name": call.name,
                            "args": call.arguments,
                            "id": call.call_id,
                            "type": "tool_call",
                        }
                        for call in message.tool_calls
                    ],
                )
            )
        else:
            converted.append(
                ToolMessage(content=message.content, tool_call_id=message.tool_call_id or "")
            )
    return converted


def response_metadata(response: Any) -> dict[str, Any]:
    """Keep compact, JSON-safe response metadata for explicit diagnostics."""

    metadata = getattr(response, "response_metadata", None)
    if not isinstance(metadata, dict):
        return {}
    allowed = {
        "finish_reason",
        "finish_message",
        "model_name",
        "model_provider",
        "prompt_feedback",
    }
    return {key: metadata[key] for key in allowed if key in metadata}
