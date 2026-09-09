"""P2 contracts for collected news and extracted electricity-market events."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.research.news.entities import canonical_assets, canonical_regions, split_entity_keys
from app.research.news.entity_resolution import EntityResolution, RawEntityMention

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
EventStatus = Literal[
    "occurred",
    "restored",
    "planned",
    "forecast",
    "corrected",
    "cancelled",
    "unknown",
]
PhysicalEffect = Literal[
    "supply_up",
    "supply_down",
    "demand_up",
    "demand_down",
    "transfer_up",
    "transfer_down",
    "mixed",
    "unknown",
]
EventTimePrecision = Literal["instant", "hour", "day", "month", "range", "vague", "unknown"]
QuantityUnit = Literal["kW", "MW", "GW", "万千瓦", "亿千瓦"]
QuantitySemantic = Literal[
    "capacity_level",
    "capacity_change",
    "generation_loss",
    "generation_restore",
    "demand_level",
    "demand_change",
    "output_level",
    "supply_change",
    "transfer_change",
    "unknown",
]
QuantityDirection = Literal["increase", "decrease", "mixed", "unknown"]
QuarantineReason = Literal[
    "ambiguous_multi_event",
    "disputed_irrelevance",
    "ambiguous_event_time",
    "ambiguous_quantity",
    "inconsistent_extraction",
    "invalid_evidence",
    "market_mismatch",
    "missing_effective_start",
    "model_response_invalid",
    "context_limit_exceeded",
    "model_output_truncated",
    "model_unavailable",
    "uncertain_extraction",
    "event_contract_violation",
]
TimeBasis = Literal["stated_absolute", "stated_components", "derived_from_publication"]
ReviewStatus = Literal["unreviewed", "accepted", "corrected", "rejected"]
ContextStrategy = Literal["full_context", "chunk_merge", "incremental_state"]
RevisionFieldName = Literal[
    "affected_regions",
    "affected_assets",
    "asset_groups",
    "capacity_mw",
    "magnitude",
    "effective_start_at",
    "effective_end_at",
]


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
    normalizer_version: Literal["1.1.0"] = "1.1.0"
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
    event_id: str | None = Field(default=None, pattern=r"^evt_[a-f0-9]{24}$")
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
    precision: EventTimePrecision = "instant"
    anchor_at: datetime | None = None
    market_timezone: str | None = None
    stated_text: str | None = Field(default=None, min_length=1, max_length=512)

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
        elif self.basis == "stated_components":
            if self.market_timezone is None or self.stated_text is None:
                raise ValueError("stated time components must record their text and market timezone")
            if self.anchor_at is not None:
                raise ValueError("stated time components must not claim a publication anchor")
        elif self.anchor_at is not None or self.market_timezone is not None:
            raise ValueError("a stated absolute time must not claim a derivation anchor")
        return self


class EventMagnitude(BaseModel):
    """A source quantity whose raw wording and analytical meaning remain distinguishable."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    raw_value: float = Field(gt=0)
    raw_unit: QuantityUnit
    semantic: QuantitySemantic
    direction: QuantityDirection = "unknown"
    normalized_mw: float = Field(gt=0)
    raw_text: str = Field(min_length=1, max_length=256)


class ExtractionTrace(BaseModel):
    """Version and hashes needed to reproduce one model-assisted extraction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_name: str = Field(min_length=1, max_length=256)
    prompt_version: str = Field(min_length=1, max_length=32)
    output_schema_version: str = Field(min_length=1, max_length=32)
    input_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    output_hash: str = Field(pattern=r"^[a-f0-9]{64}$")


class CharacterRange(BaseModel):
    """Half-open source range processed by one long-context strategy step."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_range(self) -> CharacterRange:
        if self.end_char <= self.start_char:
            raise ValueError("end_char must be greater than start_char")
        return self


class ExtractionCoverage(BaseModel):
    """Auditable source coverage and cost for one long-context extraction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy: ContextStrategy
    total_characters: int = Field(ge=1)
    processed_ranges: tuple[CharacterRange, ...] = ()
    failed_ranges: tuple[CharacterRange, ...] = ()
    model_calls: int = Field(default=0, ge=0)
    estimated_input_tokens: int = Field(default=0, ge=0)
    elapsed_seconds: float = Field(default=0.0, ge=0)

    @property
    def complete(self) -> bool:
        if self.failed_ranges:
            return False
        covered_until = 0
        for item in sorted(self.processed_ranges, key=lambda value: (value.start_char, value.end_char)):
            if item.start_char > covered_until:
                return False
            covered_until = max(covered_until, item.end_char)
        return covered_until >= self.total_characters


class NewsInputIssue(BaseModel):
    """One rejected input line, kept independently from document extraction failures."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    line_number: int = Field(ge=1)
    source_path: str
    reason: str
    raw_line: str


class EntityFields(BaseModel):
    """Optional versioned resolution; absent on legacy records, whose keys stay unchanged."""

    entity_resolution: EntityResolution | None = None

    def entity_fields_are_consistent(self) -> bool:
        return self.entity_resolution is None or (
            canonical_regions(self.affected_regions) == self.entity_resolution.region_keys
            and canonical_assets(self.affected_assets) == self.entity_resolution.asset_keys
        )

    @model_validator(mode="after")
    def validate_entity_fields(self):
        if not self.entity_fields_are_consistent():
            raise ValueError("entity fields disagree with their persisted resolution")
        return self


    @property
    def market_scope(self):
        return self.entity_resolution.market_scope if self.entity_resolution else ()

    @property
    def mentioned_regions(self):
        return self.entity_resolution.mentioned_regions if self.entity_resolution else ()

    @property
    def asset_groups(self):
        return self.entity_resolution.asset_groups if self.entity_resolution else ()

    @property
    def raw_entity_mentions(self):
        return self.entity_resolution.raw_entity_mentions if self.entity_resolution else ()

    @property
    def market_keys(self):
        return self.entity_resolution.market_keys if self.entity_resolution else ()

    @property
    def group_keys(self):
        return self.entity_resolution.group_keys if self.entity_resolution else ()


class EventRecord(EntityFields):
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
    magnitude: EventMagnitude | None = None
    announcement_available_at: datetime
    effective_start_at: datetime | None = None
    effective_end_at: datetime | None = None
    direction: EventDirection = "unknown"
    status: EventStatus = "unknown"
    physical_effect: PhysicalEffect = "unknown"
    time_resolution: TimeResolution | None = None
    extractor_id: str = Field(min_length=1, max_length=128)
    extractor_version: str = Field(min_length=1, max_length=32)
    extraction_trace: ExtractionTrace | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    analysis_eligibility: Literal[
        "eligible", "needs_time_review", "needs_coverage_review"
    ] = "eligible"
    review_status: ReviewStatus = "unreviewed"
    evidence: tuple[EvidenceSpan, ...] = ()
    # A missing field means "this version did not restate it".  A cleared field means the
    # source explicitly withdrew the previous value, so revision merging must not resurrect it.
    cleared_fields: tuple[RevisionFieldName, ...] = ()

    @field_validator("announcement_available_at", "effective_start_at", "effective_end_at")
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
        if len(self.cleared_fields) != len(set(self.cleared_fields)):
            raise ValueError("cleared_fields must not contain duplicates")
        for field_name in self.cleared_fields:
            value = getattr(self, field_name)
            if value not in (None, ()):
                raise ValueError(f"a cleared field must be empty: {field_name}")
        if self.effective_end_at is not None and self.effective_start_at is None:
            raise ValueError("effective_end_at requires effective_start_at")
        if (
            self.effective_start_at is not None
            and self.effective_end_at is not None
            and self.effective_end_at < self.effective_start_at
        ):
            raise ValueError("effective_end_at must not be before effective_start_at")
        if (self.relevance == "short_term" and self.effective_start_at is None
                and self.analysis_eligibility == "eligible"):
            raise ValueError("short_term events require effective_start_at")
        if self.event_type == "irrelevant" and self.relevance != "irrelevant":
            raise ValueError("irrelevant event_type requires irrelevant relevance")
        if self.effective_start_at is not None and self.time_resolution is None:
            raise ValueError("an event time requires a declared time resolution")
        if self.magnitude is not None:
            change_semantics = {
                "capacity_change",
                "generation_loss",
                "generation_restore",
                "demand_change",
                "supply_change",
                "transfer_change",
            }
            if self.magnitude.semantic in change_semantics:
                if self.capacity_mw != self.magnitude.normalized_mw:
                    raise ValueError("change magnitude must equal the backward-compatible capacity_mw")
            elif self.capacity_mw is not None:
                raise ValueError("level or unknown quantities must not populate capacity_mw")
        return self

    # The stored fields keep the source wording so evidence stays verbatim; these keys are
    # what identity, merging and scoring compare, so one grid written two ways — or filed
    # under regions on one run and assets on the next — stays one event.
    @property
    def region_keys(self) -> tuple[str, ...]:
        if self.entity_resolution is not None:
            return self.entity_resolution.region_keys
        return split_entity_keys(self.affected_regions, self.affected_assets)[0]

    @property
    def asset_keys(self) -> tuple[str, ...]:
        if self.entity_resolution is not None:
            return self.entity_resolution.asset_keys
        return split_entity_keys(self.affected_regions, self.affected_assets)[1]


class ExtractionQuarantine(BaseModel):
    """Why one document produced no event; it must not silently reach the event table."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    raw_entity_mentions: tuple[RawEntityMention, ...] = ()
    candidate_payload: dict[str, Any] | None = None
    extraction_trace: ExtractionTrace | None = None
    review_status: ReviewStatus = "unreviewed"

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
    # Candidate-level failures let a multi-event article keep its usable events.  `quarantine`
    # remains the document-level outcome used when no event can safely be retained.
    candidate_quarantines: tuple[ExtractionQuarantine, ...] = ()
    pass_results: tuple[dict[str, Any], ...] = ()
    coverage: ExtractionCoverage | None = None
    review_status: ReviewStatus = "unreviewed"
    supersedes_document_version_id: str | None = Field(default=None, pattern=r"^newsv_[a-f0-9]{24}$")

    @model_validator(mode="after")
    def validate_outcome_is_exclusive(self) -> EventExtractionResult:
        if not self.events and self.quarantine is None:
            raise ValueError("an extraction result must carry either events or one quarantine reason")
        if self.events and self.quarantine is not None:
            raise ValueError("an extraction result must carry either events or one quarantine reason")
        if self.quarantine is not None and self.quarantine.document_version_id != self.document_version_id:
            raise ValueError("quarantine must reference the same document version")
        if any(
            item.document_version_id != self.document_version_id
            for item in self.candidate_quarantines
        ):
            raise ValueError("candidate quarantines must reference the same document version")
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
        return tuple(
            item
            for result in self.results
            for item in (
                *((result.quarantine,) if result.quarantine is not None else ()),
                *result.candidate_quarantines,
            )
        )


class DocumentRef(BaseModel):
    """One news version that contributed to a merged event, kept for provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str = Field(pattern=r"^news_[a-f0-9]{24}$")
    document_version_id: str = Field(pattern=r"^newsv_[a-f0-9]{24}$")
    version: int = Field(ge=1)
    source_name: str = Field(min_length=1, max_length=128)
    source_tier: str = Field(default="unknown", min_length=1, max_length=128)
    content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    available_at: datetime

    @field_validator("available_at")
    @classmethod
    def validate_available_at(cls, value: datetime, info) -> datetime:
        aware = _require_aware(value, field_name=info.field_name)
        if aware.utcoffset().total_seconds() != 0:
            raise ValueError(f"{info.field_name} must be normalized to UTC")
        return aware


class EventStateRevision(EntityFields):
    """One event state as it became knowable, used to build point-in-time features."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    available_at: datetime
    relevance: NewsRelevance
    event_type: NewsEventType
    affected_regions: tuple[str, ...] = ()
    affected_assets: tuple[str, ...] = ()
    capacity_mw: float | None = Field(default=None, gt=0)
    effective_start_at: datetime | None = None
    effective_end_at: datetime | None = None
    direction: EventDirection = "unknown"
    status: EventStatus = "unknown"
    physical_effect: PhysicalEffect = "unknown"
    review_status: ReviewStatus = "unreviewed"
    analysis_eligibility: Literal[
        "eligible", "needs_time_review", "needs_coverage_review"
    ] = "eligible"

    @property
    def region_keys(self) -> tuple[str, ...]:
        if self.entity_resolution is not None:
            return self.entity_resolution.region_keys
        return split_entity_keys(self.affected_regions, self.affected_assets)[0]

    @property
    def asset_keys(self) -> tuple[str, ...]:
        if self.entity_resolution is not None:
            return self.entity_resolution.asset_keys
        return split_entity_keys(self.affected_regions, self.affected_assets)[1]

    @field_validator("available_at", "effective_start_at", "effective_end_at")
    @classmethod
    def validate_revision_time(cls, value: datetime | None, info) -> datetime | None:
        if value is None:
            return None
        aware = _require_aware(value, field_name=info.field_name)
        if aware.utcoffset().total_seconds() != 0:
            raise ValueError(f"{info.field_name} must be normalized to UTC")
        return aware

    @model_validator(mode="after")
    def validate_revision_range(self) -> EventStateRevision:
        if self.effective_end_at is not None and self.effective_start_at is None:
            raise ValueError("state effective_end_at requires effective_start_at")
        if (
            self.effective_start_at is not None
            and self.effective_end_at is not None
            and self.effective_end_at < self.effective_start_at
        ):
            raise ValueError("state effective_end_at must not be before effective_start_at")
        return self


class MergedEvent(EntityFields):
    """One real-world event assembled from every version and repost visible at `as_of`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0.0"] = "1.0.0"
    merger_version: str = Field(min_length=1, max_length=32)
    event_id: str = Field(pattern=r"^evt_[a-f0-9]{24}$")
    as_of: datetime
    relevance: NewsRelevance
    event_type: NewsEventType
    affected_regions: tuple[str, ...] = ()
    affected_assets: tuple[str, ...] = ()
    capacity_mw: float | None = Field(default=None, gt=0)
    magnitude: EventMagnitude | None = None
    announcement_available_at: datetime
    effective_start_at: datetime | None = None
    effective_end_at: datetime | None = None
    direction: EventDirection = "unknown"
    status: EventStatus = "unknown"
    physical_effect: PhysicalEffect = "unknown"
    extraction_traces: tuple[ExtractionTrace, ...] = ()
    source_event_ids: tuple[str, ...] = ()
    document_refs: tuple[DocumentRef, ...] = Field(min_length=1)
    revision_count: int = Field(ge=1)
    review_status: ReviewStatus = "unreviewed"
    analysis_eligibility: Literal[
        "eligible", "needs_time_review", "needs_coverage_review"
    ] = "eligible"
    confidence: float | None = Field(default=None, ge=0, le=1)
    time_resolution: TimeResolution | None = None
    state_history: tuple[EventStateRevision, ...] = ()

    @field_validator("as_of", "announcement_available_at", "effective_start_at", "effective_end_at")
    @classmethod
    def validate_merged_time(cls, value: datetime | None, info) -> datetime | None:
        if value is None:
            return None
        aware = _require_aware(value, field_name=info.field_name)
        if aware.utcoffset().total_seconds() != 0:
            raise ValueError(f"{info.field_name} must be normalized to UTC")
        return aware

    @property
    def region_keys(self) -> tuple[str, ...]:
        if self.entity_resolution is not None:
            return self.entity_resolution.region_keys
        return split_entity_keys(self.affected_regions, self.affected_assets)[0]

    @property
    def asset_keys(self) -> tuple[str, ...]:
        if self.entity_resolution is not None:
            return self.entity_resolution.asset_keys
        return split_entity_keys(self.affected_regions, self.affected_assets)[1]

    @model_validator(mode="after")
    def validate_merge_is_leak_free(self) -> MergedEvent:
        if self.effective_end_at is not None and self.effective_start_at is None:
            raise ValueError("merged effective_end_at requires effective_start_at")
        if (
            self.effective_start_at is not None
            and self.effective_end_at is not None
            and self.effective_end_at < self.effective_start_at
        ):
            raise ValueError("merged effective_end_at must not be before effective_start_at")
        if self.announcement_available_at > self.as_of:
            raise ValueError("a merged event cannot be announced after the as_of it belongs to")
        if any(ref.available_at > self.as_of for ref in self.document_refs):
            raise ValueError("a merged event cannot cite a document version that was not yet available")
        earliest = min(ref.available_at for ref in self.document_refs)
        if self.announcement_available_at != earliest:
            raise ValueError("announcement_available_at must equal the earliest contributing document")
        if self.revision_count != len(self.document_refs):
            raise ValueError("revision_count must match the number of contributing document versions")
        if self.state_history:
            instants = [revision.available_at for revision in self.state_history]
            if instants != sorted(instants) or len(instants) != len(set(instants)):
                raise ValueError("state_history must contain unique revisions in availability order")
            if instants[0] != self.announcement_available_at or instants[-1] > self.as_of:
                raise ValueError("state_history must start at announcement and end no later than as_of")
        return self

    @property
    def lead_time_hours(self) -> float | None:
        """How long the market could act before the event took effect; negative means late news."""

        if self.effective_start_at is None:
            return None
        return (self.effective_start_at - self.announcement_available_at).total_seconds() / 3600


class EventFeatureRow(BaseModel):
    """Event-derived numeric features for one settlement interval."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    interval_start: datetime
    active_event_count: int = Field(ge=0)
    active_capacity_mw: float | None = Field(default=None, ge=0)
    direction_up_count: int = Field(ge=0)
    direction_down_count: int = Field(ge=0)
    event_type_counts: dict[str, int] = Field(default_factory=dict)
    source_event_ids: tuple[str, ...] = ()
    new_announcement_count: int = Field(default=0, ge=0)
    new_announcement_event_ids: tuple[str, ...] = ()
    upcoming_event_count: int = Field(default=0, ge=0)
    upcoming_event_ids: tuple[str, ...] = ()
    next_effective_in_hours: float | None = Field(default=None, ge=0)
    capacity_by_effect_mw: dict[str, float | None] = Field(default_factory=dict)

    @field_validator("interval_start")
    @classmethod
    def validate_interval_start(cls, value: datetime, info) -> datetime:
        aware = _require_aware(value, field_name=info.field_name)
        if aware.utcoffset().total_seconds() != 0:
            raise ValueError(f"{info.field_name} must be normalized to UTC")
        return aware

    @model_validator(mode="after")
    def validate_counts(self) -> EventFeatureRow:
        if self.new_announcement_count != len(self.new_announcement_event_ids):
            raise ValueError("new announcement count must match its source events")
        if self.upcoming_event_count != len(self.upcoming_event_ids):
            raise ValueError("upcoming count must match its source events")
        if any(value is not None and (value < 0 or not math.isfinite(value))
               for value in self.capacity_by_effect_mw.values()):
            raise ValueError("capacity by effect must be finite and non-negative or unknown")
        if self.active_event_count != len(self.source_event_ids):
            raise ValueError("active_event_count must match the listed source events")
        if self.active_event_count == 0 and self.active_capacity_mw != 0:
            raise ValueError("an interval with no active event has a known zero capacity impact")
        return self


class EventFeatureSnapshot(BaseModel):
    """A numeric event-feature table valid as of one instant; this is the P1 hand-off shape."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0.0"] = "1.0.0"
    feature_version: str = Field(min_length=1, max_length=32)
    market: str = Field(min_length=1, max_length=128)
    market_timezone: str = Field(min_length=1, max_length=64)
    interval_minutes: int = Field(gt=0)
    as_of: datetime
    rows: tuple[EventFeatureRow, ...] = Field(min_length=1)
    source_event_ids: tuple[str, ...] = ()
    content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")

    @field_validator("as_of")
    @classmethod
    def validate_as_of(cls, value: datetime, info) -> datetime:
        aware = _require_aware(value, field_name=info.field_name)
        if aware.utcoffset().total_seconds() != 0:
            raise ValueError(f"{info.field_name} must be normalized to UTC")
        return aware

    @model_validator(mode="after")
    def validate_rows_are_ordered_and_gated(self) -> EventFeatureSnapshot:
        starts = [row.interval_start for row in self.rows]
        if starts != sorted(starts):
            raise ValueError("feature rows must be ordered by interval_start")
        if len(set(starts)) != len(starts):
            raise ValueError("feature rows must not repeat an interval_start")
        if any(start > self.as_of for start in starts):
            raise ValueError("feature rows must not extend beyond as_of")
        return self

