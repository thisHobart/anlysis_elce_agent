from app.graph.state import AgentState
from app.services.faq_service import FAQService
from app.services.knowledge import KnowledgeSearchError, get_knowledge_service, is_available
from app.services.vpp_db import VPPDatabase, VPPDBError

FAQ = FAQService()
DB = VPPDatabase()
KNOWLEDGE = get_knowledge_service()


def query_data(state: AgentState) -> dict:
    """P6 data query: real database + knowledge search execution.

    Routes:
    - faq with db source → execute real database queries
    - faq with api source → placeholder (P1 pending)
    - knowledge → execute RAG search (P6)
    - data → placeholder
    """
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
        if not entry:
            return {"data": {}, "trace": [*trace, "faq:not_found"]}

        # Check if this FAQ needs data
        needs_data = any(
            req.get("source") != "static" and req.get("variable") not in {None, "none"}
            for req in entry.get("data_requirements", [])
        )

        if not needs_data:
            # Static FAQ, no data needed
            return {"data": {}, "trace": [*trace, f"faq:{state.get('faq_id')}:static"]}

        # Determine data source
        data_sources = {req.get("source") for req in entry.get("data_requirements", [])}

        if "db" in data_sources:
            # Execute real database queries
            try:
                data = {}
                db_reqs = [req for req in entry.get("data_requirements", []) if req.get("source") == "db"]

                for req in db_reqs:
                    # Check if this is a custom SQL query
                    if req.get("db_query"):
                        # Custom SQL execution
                        result = DB.execute("exec", {"query": req["db_query"], "params": None})
                        if result.get("code") == 0 and result.get("data"):
                            # Custom query returns a list of rows
                            # For single-row results (like FAQ-03), extract the first row
                            if len(result["data"]) > 0:
                                # Merge all fields from the first row into data
                                data.update(result["data"][0])
                        else:
                            # Query failed
                            raise VPPDBError(result.get("message", "Custom SQL query failed"))
                    elif req.get("db_command"):
                        # Predefined command execution
                        result = DB.execute(req["db_command"])
                        # Map result to variables expected by the FAQ template
                        data.update(result)

                return {"data": data, "trace": [*trace, f"faq:{state.get('faq_id')}:db:success"]}

            except VPPDBError as e:
                return {
                    "data": {},
                    "fallback": True,
                    "fallback_reason": "db_query_failed",
                    "trace": [*trace, f"faq:{state.get('faq_id')}:db:error:{e!s}"],
                }

        if "api" in data_sources:
            # API data source not yet connected (P1 pending)
            return {
                "data": {},
                "fallback": True,
                "fallback_reason": "faq_data_not_connected",
                "trace": [*trace, f"faq:{state.get('faq_id')}:api:not_connected"],
            }

        # Other sources (knowledge, static)
        return {
            "data": {},
            "fallback": True,
            "fallback_reason": "faq_data_not_connected",
            "trace": [*trace, f"faq:{state.get('faq_id')}:unknown_source"],
        }

    if route == "knowledge":
        # P6: Execute RAG knowledge search
        question = state.get("question", "")
        if not question:
            return {
                "data": {},
                "fallback": True,
                "fallback_reason": "knowledge_no_question",
                "trace": [*trace, "knowledge:no_question"],
            }

        # Check if knowledge service is available
        if not is_available():
            return {
                "data": {},
                "fallback": True,
                "fallback_reason": "knowledge_not_available",
                "trace": [*trace, "knowledge:not_available"],
            }

        try:
            result = KNOWLEDGE.execute("search", {"query": question})
            if result.get("code") == 0:
                # Store formatted XML in data for compose node
                return {
                    "data": {
                        "knowledge_results": result["data"]["results"],
                        "knowledge_formatted": result["data"]["formatted"],
                        "knowledge_count": result["data"]["count"],
                    },
                    "trace": [*trace, f"knowledge:success:{result['data']['count']}_results"],
                }
            else:
                return {
                    "data": {},
                    "fallback": True,
                    "fallback_reason": "knowledge_search_failed",
                    "trace": [*trace, f"knowledge:error:{result.get('error')}"],
                }

        except KnowledgeSearchError as e:
            return {
                "data": {},
                "fallback": True,
                "fallback_reason": "knowledge_search_error",
                "trace": [*trace, f"knowledge:error:{e!s}"],
            }

    if route == "data":
        return {
            "data": {},
            "fallback": True,
            "fallback_reason": "data_query_not_connected",
            "trace": [*trace, "data:placeholder"],
        }

    return {"data": {}, "trace": [*trace, f"query:skipped:{route}"]}
