from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from app.schemas.llm import ClassificationSource, ComposeSource, Intent, Route


class QueryRequest(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1, max_length=128)
    question: str = Field(min_length=1, max_length=2000)
    user_id: str = Field(default="anonymous", max_length=128)
    role: Literal["viewer", "operator", "admin"] = "viewer"
    params: dict[str, Any] = Field(default_factory=dict)
    request_action: bool = False


class QueryResponse(BaseModel):
    session_id: str
    answer: str
    intent: Intent | None = None
    route: Route
    faq_id: str | None = None
    classification_source: ClassificationSource = "rules"
    compose_source: ComposeSource = "fixed"
    data: dict[str, Any] = Field(default_factory=dict)
    screen_action: dict[str, Any] | None = None
    validation_errors: list[str] = Field(default_factory=list)
    trace: list[str] = Field(default_factory=list)
