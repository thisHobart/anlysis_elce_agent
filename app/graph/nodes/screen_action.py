from app.config import get_settings
from app.graph.state import AgentState
from app.security.policy import is_action_allowed
from app.services.faq_service import FAQService

FAQ = FAQService()


def screen_action(state: AgentState) -> dict:
    """大屏联动。

    FAQ 路由：直接携带命中条目自带的 ``screen_action`` 指令（仅 operator/admin）。
    非 FAQ 路由：沿用原 station/metric 启发式 payload。
    """
    if not state.get("request_action") or not get_settings().enable_screen_actions:
        return {"screen_action": None}
    role = state.get("role", "viewer")
    params = state.get("params", {})
    route = state.get("route")

    if route == "screen_action":
        return {"screen_action": None, "trace": [*state.get("trace", []), "action:not_connected"]}

    if route not in {"faq", "data"}:
        return {"screen_action": None, "trace": [*state.get("trace", []), "action:skipped"]}

    if route == "faq" and state.get("faq_id"):
        if role not in {"operator", "admin"}:
            return {"screen_action": None, "trace": [*state.get("trace", []), "action:blocked"]}
        entry = FAQ.get(state["faq_id"])
        payload = {
            "type": "faq_screen_action",
            "faq_id": state["faq_id"],
            "instruction": entry.get("screen_action", "") if entry else "",
        }
        return {"screen_action": payload, "trace": [*state.get("trace", []), "action:created"]}

    action_type = "highlight_station" if params.get("station_id") else "show_metric"
    if not is_action_allowed(action_type, role):
        return {"screen_action": None, "trace": [*state.get("trace", []), "action:blocked"]}
    payload = {"type": action_type, "target": params.get("station_id", "vpp-overview")}
    return {"screen_action": payload, "trace": [*state.get("trace", []), "action:created"]}
