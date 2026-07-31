from app.graph.state import AgentState
from app.services.faq_service import FAQService
from app.services.vpp_tools import VPPTools

FAQ = FAQService()
TOOLS = VPPTools()


def query_data(state: AgentState) -> dict:
    if state.get("route") == "faq":
        match = FAQ.match(state["question"])
        return {"data": {"faq_answer": match.answer if match else ""}, "trace": [*state.get("trace", []), "faq:matched"]}
    if state.get("route") == "knowledge":
        return {"data": {"knowledge": "知识库连接点已预留；请接入现有 RAG 检索器。"}, "trace": [*state.get("trace", []), "knowledge:placeholder"]}
    if state.get("validation_errors"):
        return {"data": {}, "trace": [*state.get("trace", []), "query:blocked"]}
    return {
        "data": TOOLS.execute(state["tool_name"], state.get("params", {})),
        "trace": [*state.get("trace", []), f"tool:{state['tool_name']}"],
    }
