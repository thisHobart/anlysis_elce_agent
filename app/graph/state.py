from typing import Any, Literal, TypedDict


class AgentState(TypedDict, total=False):
    """Data carried between workflow nodes."""

    question: str
    user_id: str
    role: str
    params: dict[str, Any]
    request_action: bool
    route: Literal["faq", "knowledge", "data"]
    tool_name: str
    validation_errors: list[str]
    data: dict[str, Any]
    answer: str
    screen_action: dict[str, Any] | None
    trace: list[str]
