from app.graph.state import AgentState
from app.services.template_service import render_data_answer


def compose(state: AgentState) -> dict:
    if state.get("validation_errors"):
        answer = "请求未通过安全校验：" + " ".join(state["validation_errors"])
    elif state.get("route") == "faq":
        answer = state.get("data", {}).get("faq_answer", "未找到相关 FAQ。")
    elif state.get("route") == "knowledge":
        answer = state.get("data", {}).get("knowledge", "未找到相关知识。")
    else:
        answer = render_data_answer(state.get("data", {}))
    return {"answer": answer, "trace": [*state.get("trace", []), "composed"]}
