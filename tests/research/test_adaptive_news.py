"""Capacity-aware news extraction keeps coverage and source offsets auditable."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from app.llm.budget import ModelBudgetManager
from app.llm.model_profiles import ModelProfile, ResolvedModelProfile
from app.research.news.adaptive import AdaptiveNewsEventExtractor
from app.research.news.contracts import (
    EventExtractionResult,
    EventRecord,
    EvidenceSpan,
    ExtractionQuarantine,
    NewsDocument,
)


def _document(body: str) -> NewsDocument:
    instant = datetime(2025, 1, 1, tzinfo=UTC)
    return NewsDocument(
        document_id="news_" + "a" * 24,
        document_version_id="newsv_" + "b" * 24,
        source_name="test",
        source_document_id="test-1",
        source_ref="test://1",
        version=1,
        title="普通市场消息",
        body=body,
        published_at=instant,
        first_seen_at=instant,
        available_at=instant,
        language="zh",
        content_hash=hashlib.sha256(body.encode()).hexdigest(),
    )


class _Gateway:
    model_name = "adaptive-test-model"

    def __init__(self, *, verified: bool = False) -> None:
        profile = (
            ModelProfile(
                provider="custom",
                base_url="https://adaptive.invalid/v1",
                model=self.model_name,
                context_window_tokens=128_000,
                max_output_tokens=16_384,
                api_style="chat",
                structured_output_method="json_schema",
            )
            if verified
            else None
        )
        self.budget_manager = ModelBudgetManager(
            resolved_profile=ResolvedModelProfile(
                route_key="custom|test://adaptive|adaptive-test-model",
                source="user" if verified else "unverified",
                profile=profile,
            )
        )


class _Delegate:
    def __init__(self, *, truncate_above: int) -> None:
        self.truncate_above = truncate_above
        self.model_calls = 1

    def extract(self, document: NewsDocument) -> EventExtractionResult:
        if len(document.body) > self.truncate_above:
            return EventExtractionResult(
                document_version_id=document.document_version_id,
                quarantine=ExtractionQuarantine(
                    document_version_id=document.document_version_id,
                    reason_code="model_output_truncated",
                    message="length",
                    extractor_id="test",
                    extractor_version="1.0.0",
                ),
            )
        return EventExtractionResult(
            document_version_id=document.document_version_id,
            events=(
                EventRecord(
                    event_id="evt_" + "c" * 24,
                    document_version_id=document.document_version_id,
                    relevance="irrelevant",
                    event_type="irrelevant",
                    announcement_available_at=document.available_at,
                    extractor_id="test",
                    extractor_version="1.0.0",
                ),
            ),
        )


class _Adaptive(AdaptiveNewsEventExtractor):
    def __init__(
        self,
        *,
        truncate_above: int,
        max_split_depth: int = 3,
        verified: bool = False,
    ) -> None:
        super().__init__(
            _Gateway(verified=verified),
            market_timezone="Asia/Shanghai",
            extraction_passes=1,
            max_split_depth=max_split_depth,
        )
        self.truncate_above = truncate_above
        self.seen_lengths: list[int] = []

    def _delegate(self):
        delegate = _Delegate(truncate_above=self.truncate_above)
        original_extract = delegate.extract

        def extract(document: NewsDocument) -> EventExtractionResult:
            self.seen_lengths.append(len(document.body))
            return original_extract(document)

        delegate.extract = extract
        return delegate


def test_truncated_chunk_is_bisected_and_merged_with_complete_coverage() -> None:
    document = _document("甲" * 80)

    result = _Adaptive(truncate_above=40).extract(document)

    assert result.coverage is not None
    assert result.coverage.strategy == "chunk_merge"
    assert result.coverage.complete
    assert len(result.coverage.processed_ranges) == 2
    assert result.coverage.failed_ranges == ()
    assert len(result.events) == 1


def test_verified_full_context_truncation_splits_without_resending_the_full_body() -> None:
    document = _document("甲" * 80)
    extractor = _Adaptive(truncate_above=40, verified=True)

    result = extractor.extract(document)

    assert result.coverage is not None
    assert result.coverage.strategy == "chunk_merge"
    assert result.coverage.complete
    assert extractor.seen_lengths == [80, 40, 40]


def test_chunk_split_stops_after_three_levels_and_preserves_failed_ranges() -> None:
    document = _document("甲" * 80)

    result = _Adaptive(truncate_above=0, max_split_depth=3).extract(document)

    assert result.quarantine is not None
    assert result.coverage is not None
    assert not result.coverage.complete
    assert len(result.coverage.failed_ranges) == 8


def test_body_evidence_offsets_are_rebased_to_the_original_document() -> None:
    document = _document("前文目标证据后文")
    event = EventRecord(
        event_id="evt_" + "d" * 24,
        document_version_id=document.document_version_id,
        relevance="irrelevant",
        event_type="irrelevant",
        announcement_available_at=document.available_at,
        extractor_id="test",
        extractor_version="1.0.0",
        evidence=(
            EvidenceSpan(
                field_name="relevance",
                document_version_id=document.document_version_id,
                text_field="body",
                start_char=0,
                end_char=4,
                quote="目标证据",
            ),
        ),
    )

    rebased = AdaptiveNewsEventExtractor._rebase_event(event, 2)

    assert rebased.evidence[0].start_char == 2
    assert rebased.evidence[0].end_char == 6
    assert document.body[2:6] == rebased.evidence[0].quote
