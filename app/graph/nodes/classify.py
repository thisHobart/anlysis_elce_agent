from typing import Any

from app.config import get_settings
from app.graph.message_utils import history_for_llm
from app.graph.state import AgentState
from app.schemas.llm import ClassificationResult, ExtractedEntities, Intent, Route
from app.services.faq_service import FAQService
from app.services.llm import AgentLLM, LLMServiceError, get_llm_service

FAQ = FAQService()

# 知识问答路由的关键词（未命中 FAQ 时，判断是否走知识库检索）
KNOWLEDGE_HINTS = ("知识", "说明", "规则", "文档", "政策", "概念", "术语", "原理")
SCREEN_HINTS = ("大屏", "切换视图", "高亮", "打开面板", "展示页面")
CHITCHAT_HINTS = ("你好", "您好", "谢谢", "感谢", "再见", "拜拜")
OUT_OF_SCOPE_HINTS = ("天气", "笑话", "写代码", "股票", "电影", "菜谱")

INTENT_ROUTES: dict[Intent, Route] = {
    "faq": "faq",
    "knowledge_query": "knowledge",
    "data_query": "data",
    "screen_action": "screen_action",
    "chitchat": "chitchat",
    "out_of_scope": "out_of_scope",
}


def _entities_from_params(params: dict[str, Any]) -> ExtractedEntities:
    supported = {name: params[name] for name in ExtractedEntities.model_fields if name in params}
    return ExtractedEntities.model_validate(supported)


def _classify_by_rules(question: str, params: dict[str, Any]) -> ClassificationResult:
    match = FAQ.match(question)
    if match:
        return ClassificationResult(intent="faq", faq_id=match.faq_id, entities=_entities_from_params(params))
    lower = question.casefold()
    if any(word in lower for word in SCREEN_HINTS):
        return ClassificationResult(intent="screen_action", entities=_entities_from_params(params))
    if any(word in lower for word in KNOWLEDGE_HINTS):
        return ClassificationResult(intent="knowledge_query", entities=_entities_from_params(params))
    if any(word in lower for word in CHITCHAT_HINTS):
        return ClassificationResult(intent="chitchat", entities=_entities_from_params(params))
    if any(word in lower for word in OUT_OF_SCOPE_HINTS):
        return ClassificationResult(intent="out_of_scope", entities=_entities_from_params(params))
    return ClassificationResult(intent="data_query", entities=_entities_from_params(params))


def _validated_llm_result(result: ClassificationResult) -> ClassificationResult:
    if result.intent == "faq":
        if not result.faq_id or FAQ.get(result.faq_id) is None:
            raise LLMServiceError("The model selected an unknown FAQ.")
    elif result.faq_id is not None:
        result = result.model_copy(update={"faq_id": None})
    return result


def classify(state: AgentState, llm_service: AgentLLM | None = None) -> dict:
    """P3 hybrid classifier: validated LLM result with deterministic P2 fallback."""
    question = state["question"]
    params = state.get("params", {})
    service = llm_service or get_llm_service()
    source = "rules"
    fallback = state.get("fallback", False)
    fallback_reason = state.get("fallback_reason")

    if service.enabled:
        try:
            result = _validated_llm_result(
                service.classify(
                    question=question,
                    faq_catalog=FAQ.classification_catalog(),
                    history=history_for_llm(state, get_settings().llm_history_messages),
                    context=state.get("context", {}),
                )
            )
            source = "llm"
        except LLMServiceError:
            result = _classify_by_rules(question, params)
            fallback = True
            fallback_reason = "classification_llm_error"
    else:
        result = _classify_by_rules(question, params)

    entities = result.entities.model_dump(exclude_none=True)
    context = {**state.get("context", {}), **entities}
    route = INTENT_ROUTES[result.intent]
    suffix = result.faq_id or result.intent
    marker = f"classified:{source}:{route}:{suffix}"
    return {
        "intent": result.intent,
        "route": route,
        "faq_id": result.faq_id,
        "entities": entities,
        "context": context,
        "classification_source": source,
        "fallback": fallback,
        "fallback_reason": fallback_reason,
        "trace": [*state.get("trace", []), marker],
    }
