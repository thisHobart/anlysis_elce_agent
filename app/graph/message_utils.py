from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.graph.state import AgentState


def history_for_llm(state: AgentState, limit: int) -> list[dict[str, str]]:
    """Serialize the latest graph messages without leaking internal metadata."""
    history: list[dict[str, str]] = []
    for message in state.get("messages", [])[-limit:]:
        if not isinstance(message, BaseMessage):
            continue
        role = "user" if isinstance(message, HumanMessage) else "assistant" if isinstance(message, AIMessage) else "system"
        content: Any = message.content
        history.append({"role": role, "content": content if isinstance(content, str) else str(content)})
    return history
