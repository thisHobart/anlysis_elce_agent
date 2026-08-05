from typing import Any

from langchain_core.messages import AIMessage

from app.config import get_settings
from app.graph.message_utils import history_for_llm
from app.graph.state import AgentState
from app.services.faq_service import FAQService
from app.services.llm import AgentLLM, LLMServiceError, get_llm_service
from app.services.template_service import render_data_answer, render_faq_template

FAQ = FAQService()


FIXED_FALLBACKS = {
    "knowledge": "知识库服务暂未接入，请稍后再试。",
    "data": "实时数据查询暂未接入，目前无法提供准确数值。",
    "direct": "当前请求暂时无法处理。",
    "screen_action": "大屏联动功能暂未启用。",
    "chitchat": "您好，我是虚拟电厂数字人助手。",
    "out_of_scope": "抱歉，我只能协助处理虚拟电厂相关问题。",
}

# Fallback reasons that indicate data sources are not connected
NOT_CONNECTED_REASONS = {
    "knowledge_not_connected",
    "data_query_not_connected",
    "faq_data_not_connected",
}

# Routes that should always use fixed messages (safety-critical boundaries)
FIXED_MESSAGE_ROUTES = {"out_of_scope"}


def _fallback_answer(state: AgentState, entry: dict[str, Any] | None) -> tuple[str, str]:
    """Generate a safe fallback answer when LLM is unavailable or must not be used."""
    if state.get("validation_errors"):
        return "请求未通过安全校验：" + " ".join(state["validation_errors"]), "fixed"

    # Check if this is a not-connected scenario
    fallback_reason = state.get("fallback_reason")
    if fallback_reason == "faq_data_not_connected" and entry:
        faq_id = state.get("faq_id", "")
        return f"FAQ {faq_id} 需要实时数据，但数据源暂未接入。请稍后再试。", "fixed"

    if entry:
        return render_faq_template(entry["template"], state.get("data", {})), "template"

    route = state.get("route")
    if route in FIXED_FALLBACKS:
        return FIXED_FALLBACKS[route], "fixed"
    return render_data_answer(state.get("data", {})), "fixed"


def compose(state: AgentState, llm_service: AgentLLM | None = None) -> dict:
    """P3 composer: LLM for normal paths, but FORCED fallback when data sources are not connected.

    Security: When knowledge/data/faq_data are marked as not connected via fallback_reason,
    we MUST NOT invoke the LLM. The LLM could hallucinate plausible-sounding business facts
    from its general knowledge, which is unacceptable for a VPP production system.
    """
    entry = FAQ.get(state.get("faq_id")) if state.get("faq_id") else None
    service = llm_service or get_llm_service()
    source = "fixed"
    trace_marker = "composed:fixed"

    # FORCED fallback: data sources not connected OR safety-critical routes → never call LLM
    fallback_reason = state.get("fallback_reason")
    route = state.get("route")
    must_use_fixed = fallback_reason in NOT_CONNECTED_REASONS or route in FIXED_MESSAGE_ROUTES

    if must_use_fixed or not service.enabled:
        answer, source = _fallback_answer(state, entry)
        trace_marker = f"composed:{source}:forced" if must_use_fixed else f"composed:{source}"
    else:
        payload = {
            "question": state["question"],
            "intent": state.get("intent"),
            "route": state.get("route"),
            "faq_id": state.get("faq_id"),
            "faq_template": entry.get("template") if entry else None,
            "data_requirements": entry.get("data_requirements", []) if entry else [],
            "data": state.get("data", {}),
            "entities": state.get("entities", {}),
            "context": state.get("context", {}),
            "history": history_for_llm(state, get_settings().llm_history_messages),
            "validation_errors": state.get("validation_errors", []),
            "fallback": state.get("fallback", False),
            "fallback_reason": fallback_reason,
        }
        try:
            answer = service.compose(payload=payload)
            source = "llm"
            trace_marker = "composed:llm"
        except LLMServiceError:
            answer, source = _fallback_answer(state, entry)
            trace_marker = f"composed:{source}:fallback"

    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)],
        "compose_source": source,
        "trace": [*state.get("trace", []), trace_marker],
    }
