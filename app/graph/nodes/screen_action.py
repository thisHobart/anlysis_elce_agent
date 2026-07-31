from app.config import get_settings
from app.graph.state import AgentState
from app.security.policy import is_action_allowed


def screen_action(state: AgentState) -> dict:
    if not state.get("request_action") or not get_settings().enable_screen_actions:
        return {"screen_action": None}
    role, params = state.get("role", "viewer"), state.get("params", {})
    action_type = "highlight_station" if params.get("station_id") else "show_metric"
    if not is_action_allowed(action_type, role):
        return {"screen_action": None, "trace": [*state.get("trace", []), "action:blocked"]}
    payload = {"type": action_type, "target": params.get("station_id", "vpp-overview")}
    return {"screen_action": payload, "trace": [*state.get("trace", []), "action:created"]}
