"""Integration coverage from long news through the price-analysis eligibility gate."""

from datetime import UTC, datetime, timedelta

from app.research.news import (
    CollectedNewsRecord,
    MarketClock,
    PriceObservations,
    run_news_price_study,
)
from app.research.news.long_context import build_long_context_extractor
from scripts.benchmark_long_context_strategy import (
    FunctionalLongContextGateway,
    load_cases,
)


class _Adapter:
    def __init__(self, record: CollectedNewsRecord) -> None:
        self.record = record

    def load(self):
        return (self.record,)


def _record(case):
    return CollectedNewsRecord(
        source_name="long-context-integration",
        source_document_id=case["case_id"],
        source_ref=f"fixture://long-context-integration/{case['case_id']}",
        title=case["description"],
        body=case["body"],
        published_at=datetime(2026, 5, 24, 7, tzinfo=UTC),
        collected_at=datetime(2026, 5, 24, 7, 1, tzinfo=UTC),
        language="zh-CN",
        market_tags=("TEST",),
    )


def _prices():
    clock = MarketClock(market="TEST", timezone="Asia/Shanghai", interval_minutes=30)
    start = datetime(2026, 4, 1, tzinfo=UTC)
    timestamps = tuple(start + timedelta(minutes=30 * index) for index in range(62 * 48))
    return PriceObservations(
        clock=clock,
        timestamps=timestamps,
        prices=tuple(50.0 + (index % 48) / 10 for index in range(len(timestamps))),
        provenance={"source": "long-context-integration"},
    )


def _study(case_id: str):
    case = next(item for item in load_cases() if item["case_id"] == case_id)
    prices = _prices()
    extractor = build_long_context_extractor(
        FunctionalLongContextGateway(),
        market_timezone=prices.clock.timezone,
        context_window_tokens=30_000,
        unit_tokens=40,
    )
    return run_news_price_study(
        adapter=_Adapter(_record(case)),
        prices=prices,
        as_of=datetime(2026, 6, 2, tzinfo=UTC),
        extractor=extractor,
    )


def test_complete_long_context_event_reaches_event_ledger_and_price_analysis():
    study = _study("LC01")

    assert len(study.view.events) == 1
    event = study.view.events[0]
    assert event.analysis_eligibility == "eligible"
    assert study.analysis.for_event(event.event_id)
    assert any(link.event_id == event.event_id for link in study.package.evidence_links)


def test_incomplete_long_context_processing_never_reaches_price_windows():
    study = _study("LC05")

    assert not study.analysis.results
    assert not any(event.analysis_eligibility == "eligible" for event in study.view.events)
    assert study.view.quarantined or study.package.events_not_analyzed

