"""Post-execution quality acceptance for P2 analytical results, not just pipeline wiring."""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from app.research.news import (
    EventExtractionResult,
    JsonlCollectedNewsAdapter,
    MarketClock,
    NewsPriceStudy,
    ObviousNewsEventExtractor,
    PriceObservations,
    load_price_csv,
    render_report,
    run_news_price_study,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "news_price"
NEWS = FIXTURES / "synthetic_news.jsonl"
PRICES = FIXTURES / "synthetic_prices.csv"
MANIFEST = FIXTURES / "fixture_manifest.yaml"
CLOCK = MarketClock(market="TEST_MARKET", timezone="UTC", interval_minutes=30)
FINAL_AS_OF = datetime.fromisoformat("2026-03-01T23:30:00+00:00")


@pytest.fixture(scope="module")
def prices() -> PriceObservations:
    return load_price_csv(PRICES, CLOCK)


@pytest.fixture(scope="module")
def effective_study(prices: PriceObservations) -> NewsPriceStudy:
    return run_news_price_study(
        adapter=JsonlCollectedNewsAdapter(NEWS),
        prices=prices,
        as_of=FINAL_AS_OF,
        axis="effective",
    )


@pytest.fixture(scope="module")
def announcement_study(prices: PriceObservations) -> NewsPriceStudy:
    return run_news_price_study(
        adapter=JsonlCollectedNewsAdapter(NEWS),
        prices=prices,
        as_of=FINAL_AS_OF,
        axis="announcement",
    )


def test_completed_study_passes_every_internal_quality_gate(effective_study: NewsPriceStudy) -> None:
    assessment = effective_study.result_quality

    assert assessment.scope == "synthetic_p2_internal_validity"
    assert assessment.passed
    assert assessment.failed_checks == ()
    assert {check.code for check in assessment.checks} == {
        "single_market_clock",
        "document_market_coverage",
        "price_grid_and_values",
        "unique_event_identity",
        "valid_statistics",
        "complete_evidence_chain",
        "feature_availability",
        "complete_event_accounting",
        "reproducible_fingerprints",
    }


def test_manifest_signals_and_negative_controls_produce_the_expected_conclusions(
    effective_study: NewsPriceStudy,
    announcement_study: NewsPriceStudy,
) -> None:
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    studies = {"effective_axis": effective_study, "announcement_axis": announcement_study}

    for axis_name, expected_by_fixture in manifest["expected_outcomes"].items():
        study = studies[axis_name]
        for fixture_id, expected_conclusion in expected_by_fixture.items():
            event = _event_for_fixture(study, fixture_id)
            short_window = next(
                result
                for result in study.analysis.for_event(event.event_id)
                if result.metrics.window_label == "[0,1h]"
            )
            assert short_window.conclusion == expected_conclusion

    injected = {item["fixture_id"]: item["delta"] for item in manifest["effects"]}
    for fixture_id, delta in injected.items():
        event = _event_for_fixture(effective_study, fixture_id)
        result = next(
            item
            for item in effective_study.analysis.for_event(event.event_id)
            if item.metrics.window_label == "[0,1h]"
        )
        assert result.metrics.deviation == pytest.approx(delta, rel=0.35)
        assert result.metrics.deviation * delta > 0


def test_result_statistics_are_finite_bounded_and_multiplicity_adjusted(
    effective_study: NewsPriceStudy,
) -> None:
    assert effective_study.analysis.results
    for result in effective_study.analysis.results:
        assert math.isfinite(result.metrics.deviation)
        assert math.isfinite(result.metrics.mean_price)
        assert 0 <= result.permutation_p_value <= result.corrected_p_value <= 1
        assert result.control_sample_size >= 10
        assert result.metrics.interval_count >= 2


def test_point_event_lifecycle_and_negative_controls_do_not_pollute_features(
    effective_study: NewsPriceStudy,
) -> None:
    restoration = _event_for_fixture(effective_study, "N04")
    restoration_rows = [
        row for row in effective_study.features.rows if restoration.event_id in row.source_event_ids
    ]
    assert len(restoration_rows) == 1

    irrelevant = _event_for_fixture(effective_study, "N09")
    assert irrelevant.event_id not in effective_study.features.source_event_ids
    assert effective_study.analysis.for_event(irrelevant.event_id) == ()

    long_horizon = _event_for_fixture(effective_study, "N08")
    assert any(
        event_id == long_horizon.event_id
        for event_id, _ in effective_study.package.events_not_analyzed
    )
    assert _event_for_fixture(effective_study, "N10").event_id == _event_for_fixture(
        effective_study, "N01"
    ).event_id


def test_evidence_and_report_support_every_result_claim(effective_study: NewsPriceStudy) -> None:
    package = effective_study.package
    report = render_report(package)

    assert package.quality.evidence_coverage == 1.0
    assert package.untraceable_links() == ()
    assert package.unexplained_events == ()
    for link in package.evidence_links:
        assert link.document_version_ids
        assert link.quotes
        assert link.window_start_at < link.window_end_at
        assert link.event_id in report
    assert "事件关联也不等于因果效应" in report
    assert "不等于证明没有影响" in report


def test_quality_assessment_detects_an_extractor_that_drops_all_text_evidence(
    prices: PriceObservations,
) -> None:
    study = run_news_price_study(
        adapter=JsonlCollectedNewsAdapter(NEWS),
        prices=prices,
        as_of=FINAL_AS_OF,
        extractor=_NoEvidenceExtractor(),
    )

    assert not study.result_quality.passed
    assert {check.code for check in study.result_quality.failed_checks} == {"complete_evidence_chain"}
    assert study.package.quality.evidence_coverage == 0.0
    assert study.package.untraceable_links()


class _NoEvidenceExtractor(ObviousNewsEventExtractor):
    def extract(self, document):
        result = super().extract(document)
        if not result.events:
            return result
        return EventExtractionResult(
            document_version_id=result.document_version_id,
            events=tuple(event.model_copy(update={"evidence": ()}) for event in result.events),
        )


def _event_for_fixture(study: NewsPriceStudy, fixture_id: str):
    document = next(
        item for item in study.documents if item.raw_metadata.get("fixture_id") == fixture_id
    )
    return next(
        event
        for event in study.view.events
        if any(ref.document_version_id == document.document_version_id for ref in event.document_refs)
    )
