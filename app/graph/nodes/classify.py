from app.graph.state import AgentState
from app.services.faq_service import FAQService

FAQ = FAQService()


def classify(state: AgentState) -> dict:
    question = state["question"]
    if FAQ.match(question):
        return {"route": "faq", "trace": [*state.get("trace", []), "classified:faq"]}
    lower = question.lower()
    if any(word in lower for word in ("知识", "说明", "规则", "文档")):
        return {"route": "knowledge", "trace": [*state.get("trace", []), "classified:knowledge"]}
    return {"route": "data", "trace": [*state.get("trace", []), "classified:data"]}
