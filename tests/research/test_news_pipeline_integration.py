"""Integration tests for the whole P2 path, including the hand-off P1 would consume.

These run the real modules against the committed fixtures with nothing stubbed: manifest →
prices → news → versions → events → features → analysis → evidence package → P1-readable
export. A unit test can pass while the seams are wrong; this is what checks the seams.
"""

from __future__ import annotations

import itertools
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from app.research.data import load_series
from app.research.news import (
    JsonlCollectedNewsAdapter,
    MarketClock,
    load_price_csv,
    render_report,
    run_news_price_study,
    snapshot_to_csv_rows,
)
from app.research.news.synthetic import config_from_manifest, generate_price_series
from app.research.schemas.study import SeriesSpec

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "news_price"
NEWS = FIXTURES / "synthetic_news.jsonl"
PRICES = FIXTURES / "synthetic_prices.csv"
MANIFEST = FIXTURES / "fixture_manifest.yaml"

CLOCK = MarketClock(market="TEST_MARKET", timezone="UTC", interval_minutes=30)
FINAL_AS_OF = datetime.fromisoformat("2026-03-01T23:30:00+00:00")


@pytest.fixture(scope="module")
def prices():
    return load_price_csv(PRICES, CLOCK)


@pytest.fixture(scope="module")
def study(prices):
    return run_news_price_study(
        adapter=JsonlCollectedNewsAdapter(NEWS), prices=prices, as_of=FINAL_AS_OF, axis="effective"
    )


def test_committed_price_fixture_still_matches_its_manifest(prices) -> None:
    """The answer key and the data must never drift apart without someone noticing."""

    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    regenerated = generate_price_series(config_from_manifest(manifest))

    assert regenerated.content_hash() == prices.content_hash()
    assert regenerated.timestamps == prices.timestamps
    assert regenerated.prices == prices.prices
    assert len(prices.timestamps) == manifest["baseline"]["weeks"] * 7 * 48


def test_full_pipeline_produces_a_complete_package_from_a_raw_snapshot(study) -> None:
    package = study.package

    assert study.documents and study.store.version_count == 10
    assert study.view.events, "端到端流程必须产出事件"
    assert package.analysis.results, "端到端流程必须产出分析结果"
    assert package.evidence_links and package.untraceable_links() == ()
    assert package.features is not None and package.features.rows

    report = render_report(package)
    assert report.startswith("# P2 事件—电价证据包")
    assert all(value in report for value in package.fingerprint().values() if value)

    # Every stage agrees on one clock.
    assert package.features.market_timezone == CLOCK.timezone
    assert package.features.interval_minutes == CLOCK.interval_minutes
    assert package.analysis.clock == CLOCK


def test_the_pipeline_is_reproducible_end_to_end(prices) -> None:
    first = run_news_price_study(
        adapter=JsonlCollectedNewsAdapter(NEWS), prices=prices, as_of=FINAL_AS_OF, axis="effective"
    )
    second = run_news_price_study(
        adapter=JsonlCollectedNewsAdapter(NEWS), prices=prices, as_of=FINAL_AS_OF, axis="effective"
    )

    assert first.package.fingerprint() == second.package.fingerprint()
    assert render_report(first.package) == render_report(second.package)


def test_an_earlier_as_of_never_sees_more_than_a_later_one(prices) -> None:
    """History only accumulates; replaying an earlier instant must not reveal later news."""

    instants = [
        datetime.fromisoformat(value)
        for value in (
            "2026-01-12T10:10:00+00:00",
            "2026-01-12T13:00:00+00:00",
            "2026-01-21T10:00:00+00:00",
            "2026-03-01T23:30:00+00:00",
        )
    ]
    seen: list[set[str]] = []
    for as_of in instants:
        view = run_news_price_study(
            adapter=JsonlCollectedNewsAdapter(NEWS), prices=prices, as_of=as_of, axis="effective"
        ).view
        assert all(event.announcement_available_at <= as_of for event in view.events)
        assert all(
            ref.available_at <= as_of for event in view.events for ref in event.document_refs
        )
        seen.append({ref.document_version_id for event in view.events for ref in event.document_refs})

    for earlier, later in itertools.pairwise(seen):
        assert earlier.issubset(later), "较早的视图不得包含较晚视图没有的文档版本"


def test_exported_features_are_readable_by_the_p1_loader(study, tmp_path) -> None:
    """The P2→P1 hand-off: a plain numeric table P1 can load without knowing about news."""

    rows = snapshot_to_csv_rows(study.package.features)
    export = tmp_path / "event_features.csv"
    export.write_text("\n".join(",".join(row) for row in rows) + "\n", encoding="utf-8")

    spec = SeriesSpec(
        name="active_capacity_mw",
        path=export,
        timestamp_column="timestamp",
        value_column="active_capacity_mw",
        unit="MW",
        timezone="UTC",
        file_format="csv",
        available_at_column="available_at",
        availability_type="known_at_timestamp",
    )
    loaded = load_series(spec, study_timezone="UTC")

    assert loaded.raw_rows == len(study.package.features.rows)
    assert loaded.invalid_timestamp_rows == 0
    assert "available_at" in loaded.frame.columns
    # P2 pre-aligned the table, so availability never lags the timestamp it is stated for.
    assert (loaded.frame["available_at"] == loaded.frame["timestamp"]).all()


def test_the_exported_table_contains_no_future_information(study) -> None:
    """Read the export back and re-derive the gate, rather than trusting the builder."""

    events = {event.event_id: event for event in study.view.events}
    for row in study.package.features.rows:
        for event_id in row.source_event_ids:
            event = events[event_id]
            assert event.announcement_available_at <= row.interval_start, (
                f"{event_id} 在 {row.interval_start.isoformat()} 尚不可用却被计入特征"
            )
            assert event.effective_start_at is not None
            assert event.effective_start_at < row.interval_start + study.clock.interval


def test_both_time_axes_run_on_the_same_inputs_without_interfering(prices) -> None:
    effective = run_news_price_study(
        adapter=JsonlCollectedNewsAdapter(NEWS), prices=prices, as_of=FINAL_AS_OF, axis="effective"
    )
    announcement = run_news_price_study(
        adapter=JsonlCollectedNewsAdapter(NEWS), prices=prices, as_of=FINAL_AS_OF, axis="announcement"
    )

    # Same events, same evidence, different zero points and therefore different findings.
    assert {event.event_id for event in effective.view.events} == {
        event.event_id for event in announcement.view.events
    }
    assert effective.analysis.event_hash == announcement.analysis.event_hash
    assert effective.analysis.price_hash == announcement.analysis.price_hash
    assert effective.analysis.content_hash() != announcement.analysis.content_hash()
    assert effective.features.content_hash == announcement.features.content_hash

    for result in effective.analysis.results:
        assert result.axis == "effective"
    for result in announcement.analysis.results:
        assert result.axis == "announcement"
