from app.graph.state import AgentState
from app.security.policy import validate_tool_request


def infer_tool(question: str, params: dict) -> str:
    if params.get("device_id") or "设备" in question:
        return "get_device_detail"
    if params.get("station_id") or "电站" in question or "站点" in question:
        return "get_station_status"
    return "get_aggregate_metrics"


def validate(state: AgentState) -> dict:
    params = state.get("params", {})
    route = state.get("route")
    tool_name = infer_tool(state["question"], params) if route == "data" else None
    errors = validate_tool_request(tool_name, state.get("role", "viewer"), params) if tool_name else []
    return {
        "tool_name": tool_name,
        "validation_errors": errors,
        "trace": [*state.get("trace", []), "validated"],
    }
