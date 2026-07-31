import re
from typing import Any

ALLOWED_TOOLS = {"get_station_status", "get_aggregate_metrics", "get_device_detail"}
ALLOWED_ACTIONS = {"highlight_station", "open_station_panel", "show_metric"}
ROLE_TOOL_ALLOWLIST = {
    "viewer": {"get_station_status", "get_aggregate_metrics"},
    "operator": ALLOWED_TOOLS,
    "admin": ALLOWED_TOOLS,
}
SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
FORBIDDEN_SQL = re.compile(
    r"\b(drop|delete|insert|update|alter|truncate|grant|revoke|exec(?:ute)?)\b",
    re.IGNORECASE,
)


def is_safe_identifier(value: object) -> bool:
    return isinstance(value, str) and bool(SAFE_IDENTIFIER.fullmatch(value))


def validate_sql(sql: str) -> bool:
    """Allow only read-only SQL as a defense-in-depth guard."""
    statement = sql.strip()
    return statement.lower().startswith("select") and ";" not in statement and not FORBIDDEN_SQL.search(statement)


def validate_tool_request(tool_name: str, role: str, params: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if tool_name not in ALLOWED_TOOLS:
        errors.append("Requested tool is not allowlisted.")
    elif tool_name not in ROLE_TOOL_ALLOWLIST.get(role, set()):
        errors.append("Your role is not permitted to use this tool.")

    for field in ("station_id", "device_id"):
        if field in params and not is_safe_identifier(params[field]):
            errors.append(f"{field} must contain only letters, numbers, '_' or '-'.")
    if "sql" in params and (not isinstance(params["sql"], str) or not validate_sql(params["sql"])):
        errors.append("Only a single read-only SELECT SQL statement is allowed.")
    return errors


def is_action_allowed(action_type: str, role: str) -> bool:
    return action_type in ALLOWED_ACTIONS and role in {"operator", "admin"}
