from langchain_core.messages import HumanMessage

from app.graph.state import AgentState


def prepare_turn(state: AgentState) -> dict:
    """Start a new turn while preserving checkpointed messages and context."""
    question = state["question"].strip()
    return {
        "question": question,
        "role": state.get("role", "viewer"),
        "params": dict(state.get("params", {})),
        "request_action": bool(state.get("request_action", False)),
        "messages": [HumanMessage(content=question)],
        "intent": None,
        "route": None,
        "faq_id": None,
        "entities": {},
        "classification_source": "rules",
        "compose_source": "fixed",
        "fallback": False,
        "fallback_reason": None,
        "tool_name": None,
        "validation_errors": [],
        "data": {},
        "answer": "",
        "screen_action": None,
        "trace": ["turn:prepared"],
    }
