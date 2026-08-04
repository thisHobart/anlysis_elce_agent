from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from app.schemas.llm import ClassificationSource, ComposeSource, Intent, Route


class AgentState(TypedDict, total=False):
    """Data carried between workflow nodes."""

    question: str
    user_id: str
    role: str
    params: dict[str, Any]
    request_action: bool
    messages: Annotated[list[AnyMessage], add_messages]
    context: dict[str, Any]
    intent: Intent | None
    route: Route | None
    faq_id: str | None
    entities: dict[str, Any]
    classification_source: ClassificationSource
    compose_source: ComposeSource
    fallback: bool
    fallback_reason: str | None
    tool_name: str | None
    validation_errors: list[str]
    data: dict[str, Any]
    answer: str
    screen_action: dict[str, Any] | None
    trace: list[str]
