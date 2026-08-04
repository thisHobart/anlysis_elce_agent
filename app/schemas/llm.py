from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

Intent: TypeAlias = Literal[
    "faq",
    "knowledge_query",
    "data_query",
    "screen_action",
    "chitchat",
    "out_of_scope",
]

Route: TypeAlias = Literal[
    "faq",
    "knowledge",
    "data",
    "direct",
    "screen_action",
    "chitchat",
    "out_of_scope",
]

ClassificationSource: TypeAlias = Literal["llm", "rules"]
ComposeSource: TypeAlias = Literal["llm", "template", "fixed"]


class ExtractedEntities(BaseModel):
    """Business entities extracted from the current user turn."""

    model_config = ConfigDict(extra="forbid")

    enterprise_name: str | None = None
    station_id: str | None = None
    device_id: str | None = None
    time_range: str | None = None
    metric: str | None = None
    screen_target: str | None = None


class ClassificationResult(BaseModel):
    """Strict structured output produced by the classification model."""

    model_config = ConfigDict(extra="forbid")

    intent: Intent
    faq_id: str | None = None
    entities: ExtractedEntities = Field(default_factory=ExtractedEntities)
