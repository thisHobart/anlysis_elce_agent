from typing import Any, Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    user_id: str = Field(default="anonymous", max_length=128)
    role: Literal["viewer", "operator", "admin"] = "viewer"
    params: dict[str, Any] = Field(default_factory=dict)
    request_action: bool = False


class QueryResponse(BaseModel):
    answer: str
    route: Literal["faq", "knowledge", "data"]
    data: dict[str, Any] = Field(default_factory=dict)
    screen_action: dict[str, Any] | None = None
    validation_errors: list[str] = Field(default_factory=list)
    trace: list[str] = Field(default_factory=list)
