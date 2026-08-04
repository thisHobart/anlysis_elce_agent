from app.graph.state import AgentState
from app.services.faq_service import FAQService

FAQ = FAQService()
def query_data(state: AgentState) -> dict:
    """P3 placeholder routing; real API, DB and RAG execution are intentionally deferred."""
    route = state.get("route")
    trace = state.get("trace", [])
    if state.get("validation_errors"):
        return {
            "data": {},
            "fallback": True,
            "fallback_reason": "validation_failed",
            "trace": [*trace, "query:blocked"],
        }
    if route == "faq":
        entry = FAQ.get(state.get("faq_id")) if state.get("faq_id") else None
        needs_data = bool(
            entry
            and any(
                requirement.get("source") != "static" and requirement.get("variable") not in {None, "none"}
                for requirement in entry.get("data_requirements", [])
            )
        )
        update = {"data": {}, "trace": [*trace, f"faq:{state.get('faq_id')}"]}
        if needs_data:
            update.update({"fallback": True, "fallback_reason": "faq_data_not_connected"})
        return update
    if route == "knowledge":
        return {
            "data": {},
            "fallback": True,
            "fallback_reason": "knowledge_not_connected",
            "trace": [*trace, "knowledge:placeholder"],
        }
    if route == "data":
        return {
            "data": {},
            "fallback": True,
            "fallback_reason": "data_query_not_connected",
            "trace": [*trace, "data:placeholder"],
        }
    return {"data": {}, "trace": [*trace, f"query:skipped:{route}"]}
