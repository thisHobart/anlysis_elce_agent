"""P2 contracts for collected news and extracted electricity-market events."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

NewsRelevance = Literal["short_term", "long_horizon", "irrelevant"]
NewsEventType = Literal[
    "generation_outage",
    "generation_restore",
    "transmission_constraint",
    "demand_shock",
    "renewable_supply_change",
    "fuel_supply_change",
    "policy_long_horizon",
    "irrelevant",
    "unknown",
]
EventDirection = Literal["up", "down", "mixed", "unknown"]
QuarantineReason = Literal[
    "ambiguous_multi_event",
    "ambiguous_event_time",
    "missing_effective_start",
    "event_contract_violation",
]
TimeBasis = Literal["stated_absolute", "derived_from_publication"]


def _require_aware(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone offset")
    return value


class CollectedNewsRecord(BaseModel):
    """Provider-neutral record emitted by a collection adapter before normalization."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    source_name: str = Field(min_length=1, max_length=128)
    source_document_id: str = Field(min_length=1, max_length=256)
    source_ref: str = Field(min_length=1, max_length=2048)
    version: int = Field(default=1, ge=1)
    title: str = Field(min_length=1, max_length=1024)
    body: str = Field(min_length=1)
    published_at: datetime
    collected_at: datetime
    updated_at: datetime | None = None
    language: str = Field(default="und", min_length=2, max_length=32)
    market_tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("published_at", "collected_at", "updated_at")
    @classmethod
    def validate_aware_datetime(cls, value: datetime | None, info) -> datetime | None:
        if value is None:
            return None
        return _require_aware(value, field_name=info.field_name)

    @field_validator("market_tags")
    @classmethod
    def validate_market_tags(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(item.strip() for item in value)
        if any(not item for item in cleaned):
            raise ValueError("market_tags must not contain empty values")
        return cleaned

    @model_validator(mode="after")
    def validate_source_times(self) -> CollectedNewsRecord:
        if self.collected_at < self.published_at:
            raise ValueError("collected_at must not be before published_at")
        if self.updated_at is not None and self.updated_at < self.published_at:
            raise ValueError("updated_at must not be before published_at")
        return self


class NewsDocument(BaseModel):
    """Canonical immutable news version consumed by the P2 domain."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0.0"] = "1.0.0"
    normalizer_version: Literal["1.0.0"] = "1.0.0"
    document_id: str = Field(pattern=r"^news_[a-f0-9]{24}$")
    document_version_id: str = Field(pattern=r"^newsv_[a-f0-9]{24}$")
    source_name: str = Field(min_length=1, max_length=128)
    source_document_id: str = Field(min_length=1, max_length=256)
    source_ref: str = Field(min_length=1, max_length=2048)
    version: int = Field(ge=1)
    title: str = Field(min_length=1, max_length=1024)
    body: str = Field(min_length=1)
    published_at: datetime
    first_seen_at: datetime
    updated_at: datetime | None = None
    available_at: datetime
    language: str = Field(min_length=2, max_length=32)
    market_tags: tuple[str, ...] = ()
    content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    raw_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("published_at", "first_seen_at", "updated_at", "available_at")
    @classmethod
    def validate_utc_datetime(cls, value: datetime | None, info) -> datetime | None:
        if value is None:
            return None
        aware = _require_aware(value, field_name=info.field_name)
        if aware.utcoffset().total_seconds() != 0:
            raise ValueError(f"{info.field_name} must be normalized to UTC")
        return aware

    @model_validator(mode="after")
    def validate_normalized_times(self) -> NewsDocument:
        if self.first_seen_at < self.published_at:
            raise ValueError("first_seen_at must not be before published_at")
        expected_available_at = max(self.published_at, self.first_seen_at)
        if self.available_at != expected_available_at:
            raise ValueError("available_at must equal max(published_at, first_seen_at)")
        if self.updated_at is not None and self.updated_at < self.published_at:
            raise ValueError("updated_at must not be before published_at")
        return self


class EvidenceSpan(BaseModel):
    """Exact text span supporting one extracted event field."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    field_name: str = Field(min_length=1, max_length=64)
    document_version_id: str = Field(pattern=r"^newsv_[a-f0-9]{24}$")
    text_field: Literal["title", "body"]
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    quote: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_span(self) -> EvidenceSpan:
        if self.end_char <= self.start_char:
            raise ValueError("end_char must be greater than start_char")
        if len(self.quote) != self.end_char - self.start_char:
            raise ValueError("quote length must match the declared character span")
        return self


class TimeResolution(BaseModel):
    """How an event time was obtained, so a derived time is never read as a stated one."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    basis: TimeBasis
    anchor_at: datetime | None = None
    market_timezone: str | None = None

    @field_validator("anchor_at")
    @classmethod
    def validate_anchor(cls, value: datetime | None, info) -> datetime | None:
        if value is None:
            return None
        aware = _require_aware(value, field_name=info.field_name)
        if aware.utcoffset().total_seconds() != 0:
            raise ValueError(f"{info.field_name} must be normalized to UTC")
        return aware

    @model_validator(mode="after")
    def validate_derived_times_declare_their_assumptions(self) -> TimeResolution:
        if self.basis == "derived_from_publication":
            if self.anchor_at is None or self.market_timezone is None:
                raise ValueError("a derived time must record its publication anchor and market timezone")
        elif self.anchor_at is not None or self.market_timezone is not None:
            raise ValueError("a stated absolute time must not claim a derivation anchor")
        return self


class EventRecord(BaseModel):
    """One structured event candidate extracted from one normalized news version."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0.0"] = "1.0.0"
    event_id: str = Field(pattern=r"^evt_[a-f0-9]{24}$")
    document_version_id: str = Field(pattern=r"^newsv_[a-f0-9]{24}$")
    relevance: NewsRelevance
    event_type: NewsEventType
    affected_regions: tuple[str, ...] = ()
    affected_assets: tuple[str, ...] = ()
    capacity_mw: float | None = Field(default=None, gt=0)
    effective_start_at: datetime | None = None
    effective_end_at: datetime | None = None
    direction: EventDirection = "unknown"
    time_resolution: TimeResolution | None = None
    extractor_id: str = Field(min_length=1, max_length=128)
    extractor_version: str = Field(min_length=1, max_length=32)
    evidence: tuple[EvidenceSpan, ...] = ()

    @field_validator("effective_start_at", "effective_end_at")
    @classmethod
    def validate_event_time(cls, value: datetime | None, info) -> datetime | None:
        if value is None:
            return None
        aware = _require_aware(value, field_name=info.field_name)
        if aware.utcoffset().total_seconds() != 0:
            raise ValueError(f"{info.field_name} must be normalized to UTC")
        return aware

    @model_validator(mode="after")
    def validate_event_semantics(self) -> EventRecord:
        if self.effective_end_at is not None and self.effective_start_at is None:
            raise ValueError("effective_end_at requires effective_start_at")
        if (
            self.effective_start_at is not None
            and self.effective_end_at is not None
            and self.effective_end_at < self.effective_start_at
        ):
            raise ValueError("effective_end_at must not be before effective_start_at")
        if self.relevance == "short_term" and self.effective_start_at is None:
            raise ValueError("short_term events require effective_start_at")
        if self.event_type == "irrelevant" and self.relevance != "irrelevant":
            raise ValueError("irrelevant event_type requires irrelevant relevance")
        if self.effective_start_at is not None and self.time_resolution is None:
            raise ValueError("an event time requires a declared time resolution")
        return self


class ExtractionQuarantine(BaseModel):
    """Why one document produced no event; it must not silently reach the event table."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_version_id: str = Field(pattern=r"^newsv_[a-f0-9]{24}$")
    reason_code: QuarantineReason
    message: str = Field(min_length=1, max_length=2048)
    extractor_id: str = Field(min_length=1, max_length=128)
    extractor_version: str = Field(min_length=1, max_length=32)


class EventExtractionResult(BaseModel):
    """Versioned extraction response; one document may later yield multiple events."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_version_id: str = Field(pattern=r"^newsv_[a-f0-9]{24}$")
    events: tuple[EventRecord, ...] = ()
    quarantine: ExtractionQuarantine | None = None

    @model_validator(mode="after")
    def validate_outcome_is_exclusive(self) -> EventExtractionResult:
        if bool(self.events) == (self.quarantine is not None):
            raise ValueError("an extraction result must carry either events or one quarantine reason")
        if self.quarantine is not None and self.quarantine.document_version_id != self.document_version_id:
            raise ValueError("quarantine must reference the same document version")
        if any(event.document_version_id != self.document_version_id for event in self.events):
            raise ValueError("events must reference the same document version")
        return self


class EventExtractionBatch(BaseModel):
    """Batch outcome; one unusable document must never discard the usable ones."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    results: tuple[EventExtractionResult, ...] = ()

    @property
    def events(self) -> tuple[EventRecord, ...]:
        return tuple(event for result in self.results for event in result.events)

    @property
    def quarantined(self) -> tuple[ExtractionQuarantine, ...]:
        return tuple(result.quarantine for result in self.results if result.quarantine is not None)

