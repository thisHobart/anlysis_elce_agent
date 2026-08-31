"""Regression tests for defects found by the 2026-08-31 P2 integration audit."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.research.news import (
    AnalysisMethod,
    AsOfEventAssembler,
    CollectedNewsRecord,
    DocumentRef,
    EventExtractionResult,
    EventPriceAnalyzer,
    JsonlCollectedNewsAdapter,
    MarketClock,
    MergedEvent,
    NewsNormalizer,
    NewsVersionStore,
    ObviousNewsEventExtractor,
    PriceObservations,
    PriceSeriesError,
    load_price_csv,
    run_news_price_study,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "news_price"
NEWS = FIXTURES / "synthetic_news.jsonl"
PRICES = FIXTURES / "synthetic_prices.csv"
BASE_CLOCK = MarketClock(market="TEST_MARKET", timezone="UTC", interval_minutes=30)
FINAL_AS_OF = datetime.fromisoformat("2026-03-01T23:30:00+00:00")


@pytest.fixture(scope="module")
def prices() -> PriceObservations:
    return load_price_csv(PRICES, BASE_CLOCK)


@pytest.fixture(scope="module")
def study(prices: PriceObservations):
    return run_news_price_study(
        adapter=JsonlCollectedNewsAdapter(NEWS),
        prices=prices,
        as_of=FINAL_AS_OF,
        axis="effective",
    )


def test_pipeline_rejects_a_clock_that_disagrees_with_the_price_clock(prices: PriceObservations) -> None:
    conflicting = MarketClock(market="OTHER_MARKET", timezone="Asia/Shanghai", interval_minutes=60)

    with pytest.raises(ValueError, match="clock|时钟|market|市场"):
        run_news_price_study(
            adapter=JsonlCollectedNewsAdapter(NEWS),
            prices=prices,
            as_of=FINAL_AS_OF,
            clock=conflicting,
        )


def test_pipeline_rejects_news_from_a_different_market(prices: PriceObservations) -> None:
    other_market_prices = PriceObservations(
        clock=MarketClock(market="OTHER_MARKET", timezone="UTC", interval_minutes=30),
        timestamps=prices.timestamps,
        prices=prices.prices,
    )

    with pytest.raises(ValueError, match="market|市场"):
        run_news_price_study(
            adapter=JsonlCollectedNewsAdapter(NEWS),
            prices=other_market_prices,
            as_of=FINAL_AS_OF,
        )


def test_restoration_without_an_end_does_not_remain_active_to_the_end_of_the_dataset(study) -> None:
    restoration = next(event for event in study.view.events if event.event_type == "generation_restore")

    assert restoration.event_id not in study.features.rows[-1].source_event_ids


@pytest.mark.parametrize("case", ["gap", "off_grid", "non_finite"])
def test_price_contract_rejects_invalid_settlement_series(case: str) -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    timestamps = (start, start + timedelta(minutes=30), start + timedelta(minutes=60))
    values = (1.0, 2.0, 3.0)
    if case == "gap":
        timestamps = (start, start + timedelta(minutes=30), start + timedelta(minutes=90))
    elif case == "off_grid":
        timestamps = (start, start + timedelta(minutes=31), start + timedelta(minutes=60))
    else:
        values = (1.0, float("nan"), 3.0)

    with pytest.raises(PriceSeriesError):
        PriceObservations(clock=BASE_CLOCK, timestamps=timestamps, prices=values)


def test_a_time_correction_updates_one_event_instead_of_creating_a_second_event() -> None:
    records = (
        _correction_record(version=1, event_time="2026-01-12T10:00:00Z", seen_at="2026-01-12T10:05:00Z"),
        _correction_record(version=2, event_time="2026-01-12T10:30:00Z", seen_at="2026-01-12T11:00:00Z"),
    )
    documents = NewsNormalizer().normalize_many(records)
    view = AsOfEventAssembler(NewsVersionStore(documents)).view_at(
        datetime.fromisoformat("2026-01-12T12:00:00+00:00")
    )

    assert len(view.events) == 1
    assert view.events[0].effective_start_at == datetime.fromisoformat("2026-01-12T10:30:00+00:00")
    assert view.events[0].revision_count == 2


def test_evidence_coverage_is_not_full_when_the_extractor_returns_no_spans(prices: PriceObservations) -> None:
    no_evidence = run_news_price_study(
        adapter=JsonlCollectedNewsAdapter(NEWS),
        prices=prices,
        as_of=FINAL_AS_OF,
        extractor=_NoEvidenceExtractor(),
    )

    assert no_evidence.package.untraceable_links()
    assert no_evidence.package.quality.evidence_coverage == 0.0
    assert not no_evidence.result_quality.passed
    assert {check.code for check in no_evidence.result_quality.failed_checks} == {
        "complete_evidence_chain"
    }


def test_market_clock_floors_against_local_market_midnight() -> None:
    clock = MarketClock(market="TEST_MARKET", timezone="Asia/Kathmandu", interval_minutes=60)
    instant = datetime.fromisoformat("2026-01-01T00:10:00+05:45")
    expected = datetime.fromisoformat("2026-01-01T00:00:00+05:45").astimezone(UTC)

    assert clock.floor(instant) == expected


def test_placebo_windows_exclude_intervals_occupied_by_other_events() -> None:
    clock = MarketClock(market="TEST_MARKET", timezone="UTC", interval_minutes=30)
    start = datetime(2026, 1, 1, tzinfo=UTC)
    timestamps = tuple(start + index * clock.interval for index in range(48 * 28))
    first_at = datetime(2026, 1, 8, 10, tzinfo=UTC)
    second_at = first_at + timedelta(days=7)
    values = tuple(150.0 if second_at <= moment < second_at + timedelta(hours=1) else 50.0 for moment in timestamps)
    first, second = _merged_event("a", first_at), _merged_event("b", second_at)
    method = AnalysisMethod(window_hours=(1.0,), placebo_offset_days=(7,), permutation_samples=20)

    analysis = EventPriceAnalyzer(method).analyze(
        PriceObservations(clock=clock, timestamps=timestamps, prices=values),
        (first, second),
        axis="effective",
    )
    first_result = next(result for result in analysis.results if result.event_id == first.event_id)

    assert first_result.placebo_deviations == ()


class _NoEvidenceExtractor(ObviousNewsEventExtractor):
    def extract(self, document):
        result = super().extract(document)
        if not result.events:
            return result
        return EventExtractionResult(
            document_version_id=result.document_version_id,
            events=tuple(event.model_copy(update={"evidence": ()}) for event in result.events),
        )


def _correction_record(*, version: int, event_time: str, seen_at: str) -> CollectedNewsRecord:
    return CollectedNewsRecord(
        source_name="测试运营方",
        source_document_id="CORRECTED-TIME",
        source_ref="fixture://news/CORRECTED-TIME",
        version=version,
        title="GEN_X 突发停运",
        body=(
            "区域 TEST_NORTH，资产 GEN_X 突发停运，不可用容量 500 MW。"
            f"事件开始：{event_time}。"
        ),
        published_at="2026-01-12T10:00:00+00:00",
        collected_at=seen_at,
        updated_at=seen_at if version > 1 else None,
        language="zh-CN",
        market_tags=("TEST_MARKET",),
    )


def _merged_event(suffix: str, effective_at: datetime) -> MergedEvent:
    reference = DocumentRef(
        document_id=f"news_{suffix * 24}",
        document_version_id=f"newsv_{suffix * 24}",
        version=1,
        source_name="测试来源",
        content_hash=suffix * 64,
        available_at=effective_at - timedelta(hours=1),
    )
    return MergedEvent(
        merger_version="1.0.0",
        event_id=f"evt_{suffix * 24}",
        as_of=effective_at + timedelta(days=30),
        relevance="short_term",
        event_type="generation_outage",
        affected_regions=("TEST_NORTH",),
        affected_assets=(f"GEN_{suffix.upper()}",),
        capacity_mw=100.0,
        announcement_available_at=reference.available_at,
        effective_start_at=effective_at,
        effective_end_at=effective_at + timedelta(hours=1),
        direction="up",
        document_refs=(reference,),
        revision_count=1,
    )
