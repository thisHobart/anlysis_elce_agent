"""Weakly coupled P2 news normalization and explicit-fixture event extraction."""

from app.research.news.adapters import (
    CollectedNewsAdapter,
    CollectedNewsAdapterError,
    ExternalNewsAdapterNotConfigured,
    ExternalNewsApiPlaceholder,
    JsonlCollectedNewsAdapter,
)
from app.research.news.contracts import (
    CollectedNewsRecord,
    EventExtractionBatch,
    EventExtractionResult,
    EventRecord,
    EvidenceSpan,
    ExtractionQuarantine,
    NewsDocument,
    QuarantineReason,
    TimeResolution,
)
from app.research.news.extraction import ObviousNewsEventExtractor
from app.research.news.normalization import NewsNormalizer, normalize_news_text

__all__ = [
    "CollectedNewsAdapter",
    "CollectedNewsAdapterError",
    "CollectedNewsRecord",
    "EventExtractionBatch",
    "EventExtractionResult",
    "EventRecord",
    "EvidenceSpan",
    "ExternalNewsAdapterNotConfigured",
    "ExternalNewsApiPlaceholder",
    "ExtractionQuarantine",
    "JsonlCollectedNewsAdapter",
    "NewsDocument",
    "NewsNormalizer",
    "ObviousNewsEventExtractor",
    "QuarantineReason",
    "TimeResolution",
    "normalize_news_text",
]
