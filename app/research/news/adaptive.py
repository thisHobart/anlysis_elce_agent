"""Capacity-aware full-context and bounded chunk-merge news extraction."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any

from app.llm.budget import ModelBudgetManager, ModelRequestPurpose
from app.llm.gateway import ModelContextLimitError, ModelGateway
from app.research.news.contracts import (
    CharacterRange,
    EventExtractionBatch,
    EventExtractionResult,
    EventRecord,
    ExtractionCoverage,
    ExtractionQuarantine,
    NewsDocument,
)
from app.research.news.model_extraction import (
    ModelNewsExtraction,
    StructuredNewsEventExtractor,
    build_model_news_extraction_messages,
)


@dataclass
class _RangeResult:
    source_range: CharacterRange
    result: EventExtractionResult
    model_calls: int
    estimated_input_tokens: int


class AdaptiveNewsEventExtractor:
    """Use full context only for verified routes; otherwise split without losing offsets."""

    extractor_id = "adaptive-news-extractor"
    extractor_version = "1.0.0"

    def __init__(
        self,
        gateway: ModelGateway,
        *,
        market_timezone: str,
        minimum_confidence: float = 0.7,
        max_repair_attempts: int = 1,
        extraction_passes: int = 3,
        progress: Any | None = None,
        max_split_depth: int = 3,
    ) -> None:
        self.gateway = gateway
        self.market_timezone = market_timezone
        self.minimum_confidence = minimum_confidence
        self.max_repair_attempts = max_repair_attempts
        self.extraction_passes = extraction_passes
        self.progress = progress
        self.max_split_depth = max_split_depth
        self.model_calls = 0
        self.model_seconds = 0.0
        self.estimated_input_tokens = 0

    @property
    def _budget_manager(self) -> ModelBudgetManager | None:
        value = getattr(self.gateway, "budget_manager", None)
        return value if isinstance(value, ModelBudgetManager) else None

    def _budget(self, document: NewsDocument) -> int:
        manager = self._budget_manager
        if manager is None:
            return 0
        budget = manager.budget(
            build_model_news_extraction_messages(document, market_timezone=self.market_timezone),
            purpose=ModelRequestPurpose.NEWS_EXTRACTION,
            schema=ModelNewsExtraction,
        )
        return budget.input_tokens

    def _delegate(self) -> StructuredNewsEventExtractor:
        return StructuredNewsEventExtractor(
            self.gateway,
            market_timezone=self.market_timezone,
            minimum_confidence=self.minimum_confidence,
            max_repair_attempts=self.max_repair_attempts,
            extraction_passes=self.extraction_passes,
            progress=self.progress,
        )

    @staticmethod
    def _chunk_document(document: NewsDocument, source_range: CharacterRange) -> NewsDocument:
        body = document.body[source_range.start_char : source_range.end_char]
        return document.model_copy(
            update={
                "body": body,
                "content_hash": hashlib.sha256(body.encode("utf-8")).hexdigest(),
            }
        )

    @staticmethod
    def _preferred_boundary(body: str, start: int, end: int) -> int:
        floor = start + max(1, (end - start) // 2)
        for marker in ("\n\n", "\n", "。", "；", ". "):
            position = body.rfind(marker, floor, end)
            if position >= floor:
                return min(end, position + len(marker))
        return end

    def _initial_ranges(
        self,
        document: NewsDocument,
        *,
        force_split: bool = False,
    ) -> list[CharacterRange]:
        manager = self._budget_manager
        if manager is None:
            return [CharacterRange(start_char=0, end_char=len(document.body))]
        ranges: list[CharacterRange] = []
        start = 0
        while start < len(document.body):
            low, high = start + 1, len(document.body)
            best: int | None = None
            while low <= high:
                middle = (low + high) // 2
                candidate = CharacterRange(start_char=start, end_char=middle)
                try:
                    self._budget(self._chunk_document(document, candidate))
                except ModelContextLimitError:
                    high = middle - 1
                else:
                    best = middle
                    low = middle + 1
            if best is None:
                raise ModelContextLimitError("新闻抽取固定提示本身超过当前模型输入预算")
            end = self._preferred_boundary(document.body, start, best)
            if end <= start:
                end = best
            ranges.append(CharacterRange(start_char=start, end_char=end))
            start = end
        if force_split and len(ranges) == 1 and len(document.body) > 1:
            whole = ranges[0]
            middle = self._preferred_boundary(
                document.body,
                whole.start_char,
                whole.start_char + (whole.end_char - whole.start_char) // 2,
            )
            if middle <= whole.start_char or middle >= whole.end_char:
                middle = whole.start_char + (whole.end_char - whole.start_char) // 2
            return [
                CharacterRange(start_char=whole.start_char, end_char=middle),
                CharacterRange(start_char=middle, end_char=whole.end_char),
            ]
        return ranges

    @staticmethod
    def _rebase_event(event: EventRecord, offset: int) -> EventRecord:
        evidence = tuple(
            span.model_copy(
                update={
                    "start_char": span.start_char + offset,
                    "end_char": span.end_char + offset,
                }
            )
            if span.text_field == "body"
            else span
            for span in event.evidence
        )
        return event.model_copy(update={"evidence": evidence})

    def _extract_range(
        self,
        document: NewsDocument,
        source_range: CharacterRange,
        *,
        depth: int,
    ) -> list[_RangeResult]:
        chunk = self._chunk_document(document, source_range)
        estimated = self._budget(chunk)
        delegate = self._delegate()
        started = time.perf_counter()
        result = delegate.extract(chunk)
        self.model_seconds += time.perf_counter() - started
        self.model_calls += delegate.model_calls
        self.estimated_input_tokens += estimated * max(1, delegate.model_calls)
        retryable = bool(
            result.quarantine
            and result.quarantine.reason_code in {"context_limit_exceeded", "model_output_truncated"}
        )
        if retryable and depth < self.max_split_depth and source_range.end_char - source_range.start_char > 1:
            middle = self._preferred_boundary(
                document.body,
                source_range.start_char,
                source_range.start_char + (source_range.end_char - source_range.start_char) // 2,
            )
            if middle <= source_range.start_char or middle >= source_range.end_char:
                middle = source_range.start_char + (source_range.end_char - source_range.start_char) // 2
            return [
                *self._extract_range(
                    document,
                    CharacterRange(start_char=source_range.start_char, end_char=middle),
                    depth=depth + 1,
                ),
                *self._extract_range(
                    document,
                    CharacterRange(start_char=middle, end_char=source_range.end_char),
                    depth=depth + 1,
                ),
            ]
        rebased = result.model_copy(
            update={
                "events": tuple(self._rebase_event(event, source_range.start_char) for event in result.events)
            }
        )
        return [
            _RangeResult(
                source_range=source_range,
                result=rebased,
                model_calls=delegate.model_calls,
                estimated_input_tokens=estimated * max(1, delegate.model_calls),
            )
        ]

    def extract(self, document: NewsDocument) -> EventExtractionResult:
        started = time.perf_counter()
        calls_before = self.model_calls
        estimated_before = self.estimated_input_tokens
        manager = self._budget_manager
        verified = bool(manager and manager.resolved_profile.verified)
        force_split = False
        try:
            full_input = self._budget(document)
        except ModelContextLimitError:
            full_input = 0
            full_fits = False
        else:
            full_fits = verified
        if full_fits:
            delegate = self._delegate()
            model_started = time.perf_counter()
            result = delegate.extract(document)
            self.model_seconds += time.perf_counter() - model_started
            self.model_calls += delegate.model_calls
            self.estimated_input_tokens += full_input * max(1, delegate.model_calls)
            coverage = ExtractionCoverage(
                strategy="full_context",
                total_characters=len(document.body),
                processed_ranges=(CharacterRange(start_char=0, end_char=len(document.body)),),
                model_calls=self.model_calls - calls_before,
                estimated_input_tokens=self.estimated_input_tokens - estimated_before,
                elapsed_seconds=time.perf_counter() - started,
            )
            if result.quarantine and result.quarantine.reason_code in {
                "context_limit_exceeded",
                "model_output_truncated",
            }:
                force_split = True
            else:
                return result.model_copy(update={"coverage": coverage})

        try:
            ranges = self._initial_ranges(document, force_split=force_split)
            results = [
                item
                for source_range in ranges
                for item in self._extract_range(document, source_range, depth=0)
            ]
        except ModelContextLimitError as exc:
            full = CharacterRange(start_char=0, end_char=len(document.body))
            quarantine = ExtractionQuarantine(
                document_version_id=document.document_version_id,
                reason_code="context_limit_exceeded",
                message=str(exc),
                extractor_id=self.extractor_id,
                extractor_version=self.extractor_version,
            )
            return EventExtractionResult(
                document_version_id=document.document_version_id,
                quarantine=quarantine,
                coverage=ExtractionCoverage(
                    strategy="chunk_merge",
                    total_characters=len(document.body),
                    failed_ranges=(full,),
                    elapsed_seconds=time.perf_counter() - started,
                ),
            )

        events: dict[str, EventRecord] = {}
        irrelevant: EventRecord | None = None
        failures: list[tuple[CharacterRange, ExtractionQuarantine]] = []
        candidate_failures: list[ExtractionQuarantine] = []
        processed: list[CharacterRange] = []
        for item in results:
            if item.result.quarantine is not None:
                failures.append((item.source_range, item.result.quarantine))
                continue
            processed.append(item.source_range)
            candidate_failures.extend(item.result.candidate_quarantines)
            for event in item.result.events:
                if event.event_type == "irrelevant":
                    irrelevant = irrelevant or event
                else:
                    events.setdefault(event.event_id, event)
        coverage = ExtractionCoverage(
            strategy="chunk_merge",
            total_characters=len(document.body),
            processed_ranges=tuple(processed),
            failed_ranges=tuple(item[0] for item in failures),
            model_calls=self.model_calls - calls_before,
            estimated_input_tokens=self.estimated_input_tokens - estimated_before,
            elapsed_seconds=time.perf_counter() - started,
        )
        if events:
            return EventExtractionResult(
                document_version_id=document.document_version_id,
                events=tuple(events.values()),
                candidate_quarantines=(*candidate_failures, *(item[1] for item in failures)),
                coverage=coverage,
            )
        if failures:
            return EventExtractionResult(
                document_version_id=document.document_version_id,
                quarantine=failures[0][1],
                candidate_quarantines=tuple(item[1] for item in failures[1:]),
                coverage=coverage,
            )
        assert irrelevant is not None
        return EventExtractionResult(
            document_version_id=document.document_version_id,
            events=(irrelevant,),
            candidate_quarantines=tuple(candidate_failures),
            coverage=coverage,
        )

    def extract_many(self, documents: tuple[NewsDocument, ...]) -> EventExtractionBatch:
        return EventExtractionBatch(results=tuple(self.extract(document) for document in documents))
