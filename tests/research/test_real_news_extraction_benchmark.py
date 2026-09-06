"""Real-source news corpus quality and extractor qualification tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from app.research.news import (
    EventExtractionResult,
    EventRecord,
    EvidenceSpan,
    ExtractionQuarantine,
    JsonlCollectedNewsAdapter,
    TimeResolution,
    benchmark_real_news_extractor,
    load_real_news_gold,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "news_realistic"
NEWS = FIXTURES / "source_news.jsonl"
GOLD = FIXTURES / "gold_manifest.yaml"


@pytest.fixture(scope="module")
def records():
    return JsonlCollectedNewsAdapter(NEWS).load()


@pytest.fixture(scope="module")
def manifest():
    return load_real_news_gold(GOLD)


def test_real_news_corpus_is_traceable_diverse_and_free_of_template_markers(
    records,
    manifest,
) -> None:
    corpus_ids = [record.metadata["corpus_id"] for record in records]

    assert len(records) == 10
    assert len(set(corpus_ids)) == len(records)
    assert set(corpus_ids) == {item.corpus_id for item in manifest.records}
    assert len({record.source_ref for record in records}) == len(records)
    assert all(record.source_ref.startswith("https://") for record in records)
    assert all(record.metadata.get("retrieved_at") for record in records)
    assert all(str(record.metadata.get("source_tier", "")).startswith("primary_") for record in records)
    assert len({record.language for record in records}) >= 4
    assert len({tag for record in records for tag in record.market_tags}) >= 9
    assert sum(record.metadata["published_time_precision"] == "day" for record in records) == 8
    assert all("TEST_" not in f"{record.title}\n{record.body}" for record in records)
    assert all("事件开始：" not in record.body for record in records)
    assert all(len(record.body) < 220 for record in records)


def test_current_rule_baseline_is_explicitly_not_qualified_on_real_news(
    records,
    manifest,
) -> None:
    report = benchmark_real_news_extractor(records, manifest)

    assert not report.qualified
    assert report.extractable_event_recall == 0.0
    assert report.quarantine_recall == 0.0
    assert report.irrelevant_accuracy == 1.0
    assert report.unknown_rate == 1.0
    assert report.full_case_accuracy == 0.1
    assert {check.code for check in report.failed_checks} == {
        "extractable_event_recall",
        "quarantine_recall",
            "unknown_rate",
            "evidence_integrity",
            "full_case_accuracy",
        }
    assert [case.corpus_id for case in report.cases if case.passed] == ["R10"]
    assert all(case.evidence_span_count == 0 for case in report.cases)


def test_benchmark_accepts_a_gold_faithful_extractor(records, manifest) -> None:
    gold_by_id = {record.corpus_id: record for record in manifest.records}
    report = benchmark_real_news_extractor(
        records,
        manifest,
        extractor_factory=lambda _timezone: _GoldExtractor(gold_by_id),
    )

    assert report.qualified
    assert report.extractable_event_recall == 1.0
    assert report.quarantine_recall == 1.0
    assert report.irrelevant_accuracy == 1.0
    assert report.unknown_rate == 0.0
    assert report.full_case_accuracy == 1.0
    assert report.failed_checks == ()
    assert all(
        case.quarantine_message
        for case in report.cases
        if case.actual_disposition == "quarantine"
    )


def test_full_field_errors_are_blocking_even_when_event_recall_is_perfect(records, manifest) -> None:
    class WrongFields(_GoldExtractor):
        def extract(self, document):
            result = super().extract(document)
            if not result.events or result.events[0].event_type in {"irrelevant", "unknown"}:
                return result
            event = result.events[0].model_copy(
                update={"affected_regions": ("WRONG",), "affected_assets": ("INVENTED",)}
            )
            return result.model_copy(update={"events": (event,)})

    gold_by_id = {record.corpus_id: record for record in manifest.records}
    report = benchmark_real_news_extractor(
        records,
        manifest,
        extractor_factory=lambda _timezone: WrongFields(gold_by_id),
    )

    assert report.extractable_event_recall == 1.0
    assert report.full_case_accuracy < 1.0
    assert not report.qualified
    assert "full_case_accuracy" in {check.code for check in report.failed_checks}


def test_missing_field_level_evidence_is_a_blocking_failure(records, manifest) -> None:
    class MissingEvidence(_GoldExtractor):
        def extract(self, document):
            result = super().extract(document)
            if not result.events or result.events[0].event_type in {"irrelevant", "unknown"}:
                return result
            event = result.events[0].model_copy(update={"evidence": result.events[0].evidence[:1]})
            return result.model_copy(update={"events": (event,)})

    gold_by_id = {record.corpus_id: record for record in manifest.records}
    report = benchmark_real_news_extractor(
        records,
        manifest,
        extractor_factory=lambda _timezone: MissingEvidence(gold_by_id),
    )

    assert report.full_case_accuracy == 1.0
    assert not report.qualified
    assert "evidence_integrity" in {check.code for check in report.failed_checks}


class _GoldExtractor:
    extractor_id = "gold-faithful-test-double"
    extractor_version = "1.0.0"

    def __init__(self, gold_by_id) -> None:
        self.gold_by_id = gold_by_id

    def extract(self, document):
        gold = self.gold_by_id[document.raw_metadata["corpus_id"]]
        if gold.expected_disposition == "quarantine":
            reason = (
                "ambiguous_multi_event"
                if len(gold.expected_event_types) > 1
                else "missing_effective_start"
            )
            return EventExtractionResult(
                document_version_id=document.document_version_id,
                quarantine=ExtractionQuarantine(
                    document_version_id=document.document_version_id,
                    reason_code=reason,
                    message=gold.gold_reason,
                    extractor_id=self.extractor_id,
                    extractor_version=self.extractor_version,
                ),
            )

        event_type = gold.expected_event_types[0]
        irrelevant = gold.expected_disposition == "irrelevant"
        event_id = f"evt_{hashlib.sha256(gold.corpus_id.encode()).hexdigest()[:24]}"
        evidence = ()
        if not irrelevant:
            fields = ["relevance", "event_type"]
            if gold.expected_regions:
                fields.append("affected_regions")
            if gold.expected_assets:
                fields.append("affected_assets")
            if gold.expected_capacity_mw is not None:
                fields.append("capacity_mw")
            if gold.expected_start_at is not None:
                fields.append("effective_start_at")
            if event_type == "generation_outage":
                fields.append("direction")
            evidence = tuple(
                EvidenceSpan(
                    field_name=field_name,
                    event_id=event_id,
                    document_version_id=document.document_version_id,
                    text_field="body",
                    start_char=0,
                    end_char=len(document.body),
                    quote=document.body,
                )
                for field_name in fields
            )
        event = EventRecord(
            event_id=event_id,
            document_version_id=document.document_version_id,
            relevance=gold.expected_relevance,
            event_type=event_type,
            affected_regions=gold.expected_regions,
            affected_assets=gold.expected_assets,
            capacity_mw=gold.expected_capacity_mw,
            announcement_available_at=document.available_at,
            effective_start_at=gold.expected_start_at,
            direction=(
                "up" if event_type == "generation_outage" else "unknown"
            ),
            time_resolution=(
                TimeResolution(basis="stated_absolute")
                if gold.expected_start_at is not None
                else None
            ),
            extractor_id=self.extractor_id,
            extractor_version=self.extractor_version,
            evidence=evidence,
        )
        return EventExtractionResult(
            document_version_id=document.document_version_id,
            events=(event,),
        )
