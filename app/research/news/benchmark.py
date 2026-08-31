"""Accuracy benchmark for frozen, traceable excerpts from real electricity news."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.research.news.contracts import (
    CollectedNewsRecord,
    EventExtractionResult,
    NewsDocument,
    NewsEventType,
    NewsRelevance,
)
from app.research.news.extraction import NewsEventExtractor, ObviousNewsEventExtractor
from app.research.news.normalization import NewsNormalizer

REAL_NEWS_BENCHMARK_VERSION = "1.0.0"

ExpectedDisposition = Literal["event", "quarantine", "irrelevant"]
ActualDisposition = Literal["event", "quarantine", "irrelevant", "invalid"]


class RealNewsGoldRecord(BaseModel):
    """Human-reviewed expected handling for one source excerpt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    corpus_id: str = Field(pattern=r"^R\d{2}$")
    market_timezone: str = Field(min_length=1)
    expected_disposition: ExpectedDisposition
    expected_event_types: tuple[NewsEventType, ...] = Field(min_length=1)
    expected_relevance: NewsRelevance
    expected_regions: tuple[str, ...] = ()
    expected_assets: tuple[str, ...] = ()
    expected_capacity_mw: float | None = Field(default=None, gt=0)
    expected_start_at: datetime | None = None
    gold_reason: str = Field(min_length=1)

    @field_validator("expected_start_at")
    @classmethod
    def normalize_start(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("expected_start_at must carry a timezone")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_irrelevant_case(self) -> RealNewsGoldRecord:
        if self.expected_disposition == "irrelevant" and self.expected_relevance != "irrelevant":
            raise ValueError("irrelevant disposition requires irrelevant relevance")
        return self


class BenchmarkThresholds(BaseModel):
    """Minimum quality required before an extractor may be called realistic-news ready."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    extractable_event_recall: float = Field(ge=0, le=1)
    quarantine_recall: float = Field(ge=0, le=1)
    irrelevant_accuracy: float = Field(ge=0, le=1)
    maximum_unknown_rate: float = Field(ge=0, le=1)


class RealNewsGoldManifest(BaseModel):
    """Versioned answer key kept separate from the extractor input."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    corpus_version: str = Field(min_length=1)
    retrieved_at: datetime
    purpose: str = Field(min_length=1)
    qualification_thresholds: BenchmarkThresholds
    records: tuple[RealNewsGoldRecord, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_ids(self) -> RealNewsGoldManifest:
        ids = [record.corpus_id for record in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("real-news gold corpus IDs must be unique")
        return self


class RealNewsCaseResult(BaseModel):
    """Observed result for one real-source excerpt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    corpus_id: str
    source_ref: str
    expected_disposition: ExpectedDisposition
    actual_disposition: ActualDisposition
    expected_event_types: tuple[str, ...]
    actual_event_type: str | None = None
    actual_relevance: str | None = None
    actual_regions: tuple[str, ...] = ()
    actual_assets: tuple[str, ...] = ()
    actual_capacity_mw: float | None = None
    actual_start_at: datetime | None = None
    quarantine_reason: str | None = None
    event_type_match: bool = False
    relevance_match: bool = False
    region_match: bool = False
    asset_match: bool = False
    capacity_match: bool = False
    start_match: bool = False
    evidence_span_count: int = Field(ge=0)
    evidence_integrity: bool | None = None
    passed: bool


class RealNewsBenchmarkCheck(BaseModel):
    """One qualification threshold and its measured value."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(min_length=1)
    severity: Literal["blocker", "warning"]
    passed: bool
    measured: float | None = None
    threshold: float | None = None
    detail: str = Field(min_length=1)


class RealNewsBenchmarkReport(BaseModel):
    """Accuracy, abstention and provenance verdict for an extractor."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark_version: str
    corpus_version: str
    extractor_id: str
    extractor_version: str
    document_count: int = Field(ge=1)
    source_profile: dict[str, object]
    extractable_event_recall: float
    quarantine_recall: float
    irrelevant_accuracy: float
    unknown_rate: float
    full_case_accuracy: float
    cases: tuple[RealNewsCaseResult, ...]
    checks: tuple[RealNewsBenchmarkCheck, ...]

    @property
    def qualified(self) -> bool:
        return all(check.passed for check in self.checks if check.severity == "blocker")

    @property
    def failed_checks(self) -> tuple[RealNewsBenchmarkCheck, ...]:
        return tuple(check for check in self.checks if not check.passed)


def load_real_news_gold(path: str | Path) -> RealNewsGoldManifest:
    """Load the human-reviewed answer key; analysis inputs never read this file."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return RealNewsGoldManifest.model_validate(payload)


def benchmark_real_news_extractor(
    records: Sequence[CollectedNewsRecord],
    manifest: RealNewsGoldManifest,
    *,
    extractor_factory: Callable[[str], NewsEventExtractor] | None = None,
) -> RealNewsBenchmarkReport:
    """Run one extractor without letting it see labels, then score against gold."""

    factory = extractor_factory or (
        lambda timezone: ObviousNewsEventExtractor(market_timezone=timezone)
    )
    documents = NewsNormalizer().normalize_many(tuple(records))
    documents_by_id = {
        str(document.raw_metadata.get("corpus_id")): document for document in documents
    }
    gold_by_id = {record.corpus_id: record for record in manifest.records}
    if set(documents_by_id) != set(gold_by_id):
        missing = sorted(set(gold_by_id) - set(documents_by_id))
        extra = sorted(set(documents_by_id) - set(gold_by_id))
        raise ValueError(f"real-news corpus/gold mismatch: missing={missing}, extra={extra}")

    cases: list[RealNewsCaseResult] = []
    extractor_identity: tuple[str, str] | None = None
    for corpus_id in sorted(gold_by_id):
        document = documents_by_id[corpus_id]
        gold = gold_by_id[corpus_id]
        extractor = factory(gold.market_timezone)
        identity = (extractor.extractor_id, extractor.extractor_version)
        if extractor_identity is None:
            extractor_identity = identity
        elif identity != extractor_identity:
            raise ValueError("one benchmark run must use one extractor identity/version")
        cases.append(_score_case(document, gold, extractor.extract(document)))

    assert extractor_identity is not None
    event_cases = [case for case in cases if case.expected_disposition == "event"]
    quarantine_cases = [
        case for case in cases if case.expected_disposition == "quarantine"
    ]
    irrelevant_cases = [
        case for case in cases if case.expected_disposition == "irrelevant"
    ]
    event_recall = _rate(
        sum(
            case.actual_disposition == "event" and case.event_type_match
            for case in event_cases
        ),
        len(event_cases),
    )
    quarantine_recall = _rate(
        sum(case.actual_disposition == "quarantine" for case in quarantine_cases),
        len(quarantine_cases),
    )
    irrelevant_accuracy = _rate(
        sum(case.passed for case in irrelevant_cases),
        len(irrelevant_cases),
    )
    unknown_rate = _rate(
        sum(case.actual_event_type == "unknown" for case in cases),
        len(cases),
    )
    full_accuracy = _rate(sum(case.passed for case in cases), len(cases))

    thresholds = manifest.qualification_thresholds
    source_traceability = all(
        document.source_ref.startswith(("https://", "http://"))
        and bool(document.raw_metadata.get("retrieved_at"))
        and bool(document.raw_metadata.get("source_tier"))
        for document in documents
    )
    evidence_cases = [case for case in cases if case.evidence_integrity is not None]
    evidence_integrity = (
        _rate(sum(bool(case.evidence_integrity) for case in evidence_cases), len(evidence_cases))
        if evidence_cases
        else None
    )
    checks = (
        RealNewsBenchmarkCheck(
            code="source_traceability",
            severity="blocker",
            passed=source_traceability,
            detail="每条摘录必须保留 URL、抓取时点和来源层级",
        ),
        _minimum_check(
            "extractable_event_recall",
            event_recall,
            thresholds.extractable_event_recall,
            f"可按现有契约抽取的真实事件命中 {sum(case.actual_disposition == 'event' and case.event_type_match for case in event_cases)}/{len(event_cases)}",
        ),
        _minimum_check(
            "quarantine_recall",
            quarantine_recall,
            thresholds.quarantine_recall,
            f"应隔离的含糊或缺时刻新闻命中 {sum(case.actual_disposition == 'quarantine' for case in quarantine_cases)}/{len(quarantine_cases)}",
        ),
        _minimum_check(
            "irrelevant_accuracy",
            irrelevant_accuracy,
            thresholds.irrelevant_accuracy,
            f"无关新闻正确处理 {sum(case.passed for case in irrelevant_cases)}/{len(irrelevant_cases)}",
        ),
        RealNewsBenchmarkCheck(
            code="unknown_rate",
            severity="blocker",
            passed=unknown_rate <= thresholds.maximum_unknown_rate,
            measured=unknown_rate,
            threshold=thresholds.maximum_unknown_rate,
            detail="真实语料不能大面积静默退化为 unknown/irrelevant",
        ),
        RealNewsBenchmarkCheck(
            code="evidence_integrity",
            severity="warning",
            passed=evidence_integrity == 1.0,
            measured=evidence_integrity,
            threshold=1.0,
            detail=(
                "所有已抽取字段证据均精确回链原文"
                if evidence_cases
                else "没有抽取出带证据的真实事件，证据完整性无法评估"
            ),
        ),
    )
    source_profile: dict[str, object] = {
        "languages": dict(Counter(document.language for document in documents)),
        "markets": dict(
            Counter(tag for document in documents for tag in document.market_tags)
        ),
        "source_tiers": dict(
            Counter(str(document.raw_metadata["source_tier"]) for document in documents)
        ),
        "date_only_publication_count": sum(
            document.raw_metadata.get("published_time_precision") == "day"
            for document in documents
        ),
        "unique_source_url_count": len({document.source_ref for document in documents}),
    }
    return RealNewsBenchmarkReport(
        benchmark_version=REAL_NEWS_BENCHMARK_VERSION,
        corpus_version=manifest.corpus_version,
        extractor_id=extractor_identity[0],
        extractor_version=extractor_identity[1],
        document_count=len(documents),
        source_profile=source_profile,
        extractable_event_recall=event_recall,
        quarantine_recall=quarantine_recall,
        irrelevant_accuracy=irrelevant_accuracy,
        unknown_rate=unknown_rate,
        full_case_accuracy=full_accuracy,
        cases=tuple(cases),
        checks=checks,
    )


def _score_case(
    document: NewsDocument,
    gold: RealNewsGoldRecord,
    extraction: EventExtractionResult,
) -> RealNewsCaseResult:
    if extraction.quarantine is not None:
        actual_disposition: ActualDisposition = "quarantine"
        event = None
    elif len(extraction.events) != 1:
        actual_disposition = "invalid"
        event = None
    else:
        event = extraction.events[0]
        actual_disposition = (
            "irrelevant"
            if event.event_type in {"irrelevant", "unknown"}
            else "event"
        )

    event_type_match = bool(event and event.event_type in gold.expected_event_types)
    relevance_match = bool(event and event.relevance == gold.expected_relevance)
    region_match = bool(event and event.affected_regions == gold.expected_regions)
    asset_match = bool(event and event.affected_assets == gold.expected_assets)
    capacity_match = bool(
        event and _optional_float_equal(event.capacity_mw, gold.expected_capacity_mw)
    )
    start_match = bool(
        event and event.effective_start_at == gold.expected_start_at
    )
    if gold.expected_disposition == "event":
        passed = bool(
            actual_disposition == "event"
            and event_type_match
            and relevance_match
            and region_match
            and asset_match
            and capacity_match
            and start_match
        )
    elif gold.expected_disposition == "quarantine":
        passed = actual_disposition == "quarantine"
    else:
        passed = bool(
            actual_disposition == "irrelevant"
            and event_type_match
            and relevance_match
        )

    evidence_integrity: bool | None = None
    evidence_count = 0
    if event is not None and event.evidence:
        evidence_count = len(event.evidence)
        evidence_integrity = all(
            getattr(document, span.text_field)[span.start_char : span.end_char]
            == span.quote
            for span in event.evidence
        )

    return RealNewsCaseResult(
        corpus_id=gold.corpus_id,
        source_ref=document.source_ref,
        expected_disposition=gold.expected_disposition,
        actual_disposition=actual_disposition,
        expected_event_types=tuple(gold.expected_event_types),
        actual_event_type=event.event_type if event is not None else None,
        actual_relevance=event.relevance if event is not None else None,
        actual_regions=event.affected_regions if event is not None else (),
        actual_assets=event.affected_assets if event is not None else (),
        actual_capacity_mw=event.capacity_mw if event is not None else None,
        actual_start_at=event.effective_start_at if event is not None else None,
        quarantine_reason=(
            extraction.quarantine.reason_code if extraction.quarantine is not None else None
        ),
        event_type_match=event_type_match,
        relevance_match=relevance_match,
        region_match=region_match,
        asset_match=asset_match,
        capacity_match=capacity_match,
        start_match=start_match,
        evidence_span_count=evidence_count,
        evidence_integrity=evidence_integrity,
        passed=passed,
    )


def _minimum_check(
    code: str,
    measured: float,
    threshold: float,
    detail: str,
) -> RealNewsBenchmarkCheck:
    return RealNewsBenchmarkCheck(
        code=code,
        severity="blocker",
        passed=measured >= threshold,
        measured=measured,
        threshold=threshold,
        detail=detail,
    )


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _optional_float_equal(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(left, right, rel_tol=0, abs_tol=1e-9)
