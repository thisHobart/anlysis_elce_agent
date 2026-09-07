"""Product-level regression boundaries, including restart and review history."""

import itertools
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.research.news import (
    AsOfEventAssembler,
    JsonlCollectedNewsAdapter,
    MarketClock,
    NewsNormalizer,
    NewsVersionStore,
    ObviousNewsEventExtractor,
    build_event_features,
    load_price_csv,
    merge_event_records,
    run_news_price_study,
)
from app.research.news.analysis import (
    AnalysisMethod,
    EventPriceAnalyzer,
    WindowSpec,
    _build_baseline,
    _control_windows,
    _window_metrics,
)
from app.research.news.workspace import CachedNewsExtractor, NewsWorkspace, WorkspaceNewsAdapter, export_study
from scripts.run_news_workbench import main

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "news_price"
CLOCK = MarketClock(market="TEST_MARKET", timezone="UTC", interval_minutes=30)
CUTOFF = datetime(2026, 3, 1, 23, 30, tzinfo=UTC)


@pytest.fixture(scope="module")
def study():
    return run_news_price_study(
        adapter=JsonlCollectedNewsAdapter(FIXTURES / "synthetic_news.jsonl"),
        prices=load_price_csv(FIXTURES / "synthetic_prices.csv", CLOCK),
        as_of=CUTOFF,
    )


def test_withdrawal_survives_later_omission_and_allows_explicit_restatement():
    record = JsonlCollectedNewsAdapter(FIXTURES / "synthetic_news.jsonl").load()[0]
    first_doc = NewsNormalizer().normalize(record)
    first = ObviousNewsEventExtractor().extract(first_doc).events[0]
    pairs = [(first, first_doc)]
    for version, capacity, cleared in ((2, None, ("capacity_mw", "magnitude")), (3, None, ()), (4, 350.0, ())):
        now = first_doc.available_at + timedelta(hours=version)
        document = NewsNormalizer().normalize(
            record.model_copy(
                update={
                    "version": version,
                    "collected_at": now,
                    "updated_at": now,
                }
            )
        )
        event = first.model_copy(
            update={
                "document_version_id": document.document_version_id,
                "announcement_available_at": now,
                "capacity_mw": capacity,
                "cleared_fields": cleared,
            }
        )
        pairs.append((event, document))
        merged = merge_event_records(pairs, as_of=now)
        assert len(merged) == 1
        assert merged[0].capacity_mw == capacity
    assert [r.capacity_mw for r in merged[0].state_history] == [500, None, None, 350]


def test_partial_window_never_gets_a_full_duration_label():
    prices = load_price_csv(FIXTURES / "synthetic_prices.csv", CLOCK)
    baseline = _build_baseline(prices, set(), AnalysisMethod())
    assert _window_metrics(prices, baseline, WindowSpec(24), prices.end_at - CLOCK.interval) is None
    assert _window_metrics(prices, baseline, WindowSpec(1), prices.end_at - CLOCK.interval).interval_count == 2


def test_controls_match_calendar_and_never_overlap():
    prices = load_price_csv(FIXTURES / "synthetic_prices.csv", CLOCK)
    baseline = _build_baseline(prices, set(), AnalysisMethod())
    anchor = datetime(2026, 1, 12, 10, 30, tzinfo=UTC)
    occupied = {anchor}
    controls = _control_windows(prices, baseline, WindowSpec(24), occupied, anchor=anchor)
    assert controls
    for control in controls:
        assert (control.start_at.hour, control.start_at.minute) == (10, 30)
        assert control.start_at.weekday() < 5
        assert control.interval_count == 48
        assert not control.start_at <= anchor < control.end_at
    assert all(a.end_at <= b.start_at for a, b in itertools.pairwise(controls))


def test_concurrent_events_cannot_produce_single_event_attribution(study):
    event = next(e for e in study.view.events if e.event_type == "generation_outage")
    other = event.model_copy(update={"event_id": "evt_" + "b" * 24})
    prices = load_price_csv(FIXTURES / "synthetic_prices.csv", CLOCK)
    result = EventPriceAnalyzer().analyze(prices, [event, other], axis="effective")
    assert result.results
    assert all(r.confounding_event_ids and r.conclusion == "not_supported_by_current_data" for r in result.results)


def test_advance_notices_exist_before_effect_but_never_before_knowledge(study):
    event = next(e for e in study.view.events if e.event_type == "demand_shock")
    before, upcoming = [], []
    for row in study.features.rows:
        if row.interval_start < event.announcement_available_at:
            before.append(row)
        elif row.interval_start < event.effective_start_at:
            upcoming.append(row)
    assert all(event.event_id not in r.upcoming_event_ids for r in before)
    assert len(upcoming) == 57
    assert all(event.event_id in r.upcoming_event_ids for r in upcoming)
    assert sum(event.event_id in r.new_announcement_event_ids for r in study.features.rows) == 1
    assert all(event.event_id not in r.source_event_ids for r in upcoming[:-1])


def test_bad_input_line_is_accounted_for_without_discarding_good_records(tmp_path):
    source = tmp_path / "news.jsonl"
    source.write_text((FIXTURES / "synthetic_news.jsonl").read_text(encoding="utf-8") + "\n{bad\n", encoding="utf-8")
    result = run_news_price_study(
        adapter=JsonlCollectedNewsAdapter(source),
        prices=load_price_csv(FIXTURES / "synthetic_prices.csv", CLOCK),
        as_of=CUTOFF,
    )
    assert result.view.events
    assert len(result.package.input_issues) == 1
    assert not result.result_quality.passed
    output = export_study(result, NewsWorkspace(tmp_path / "workspace"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "needs_review"
    assert "input-issues.json" in manifest["files"]


class CountingExtractor(ObviousNewsEventExtractor):
    calls = 0

    def extract(self, document):
        self.calls += 1
        return super().extract(document)


def test_restart_reuses_cache_but_configuration_change_does_not(tmp_path):
    document = NewsNormalizer().normalize(JsonlCollectedNewsAdapter(FIXTURES / "synthetic_news.jsonl").load()[0])
    first = CountingExtractor()
    CachedNewsExtractor(first, NewsWorkspace(tmp_path), {"model": "one"}).extract(document)
    second = CountingExtractor()
    CachedNewsExtractor(second, NewsWorkspace(tmp_path), {"model": "one"}).extract(document)
    assert first.calls == 1 and second.calls == 0
    CachedNewsExtractor(second, NewsWorkspace(tmp_path), {"model": "two"}).extract(document)
    assert second.calls == 1


def test_review_is_a_new_visible_revision_and_does_not_rewrite_old_features(tmp_path):
    workspace = NewsWorkspace(tmp_path)
    document = NewsNormalizer().normalize(JsonlCollectedNewsAdapter(FIXTURES / "synthetic_news.jsonl").load()[0])
    extractor = CachedNewsExtractor(CountingExtractor(), workspace, {})
    extractor.extract(document)
    with workspace.connect() as db:
        key = db.execute("SELECT cache_key FROM extractions").fetchone()[0]
    workspace.review(key, decision="rejected", reviewer="tester", reason="人工判断不适用", market_timezone="UTC")
    review_doc = NewsNormalizer().normalize(workspace.review_records()[0])
    store = NewsVersionStore((document, review_doc))
    assembler = AsOfEventAssembler(store, extractor=extractor)
    old = assembler.view_at(CUTOFF)
    new = assembler.view_at(review_doc.available_at)
    assert len(old.events) == len(new.events) == 1
    assert old.events[0].review_status == "unreviewed"
    assert new.events[0].review_status == "rejected"
    old_features = build_event_features(
        old.events,
        clock=CLOCK,
        start_at=document.available_at,
        end_at=document.available_at + timedelta(hours=3),
        as_of=CUTOFF,
    )
    new_features = build_event_features(
        new.events,
        clock=CLOCK,
        start_at=document.available_at,
        end_at=document.available_at + timedelta(hours=3),
        as_of=new.as_of,
    )
    assert old_features.rows == new_features.rows


def test_import_cannot_impersonate_local_review(tmp_path):
    record = JsonlCollectedNewsAdapter(FIXTURES / "synthetic_news.jsonl").load()[0]
    record.metadata["local_review_id"] = 1
    source = tmp_path / "spoof.jsonl"
    source.write_text(record.model_dump_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="local_review_id"):
        WorkspaceNewsAdapter(JsonlCollectedNewsAdapter(source), NewsWorkspace(tmp_path / "db")).load_batch()


def test_cli_exports_a_complete_run_with_explicit_market_and_extractor(tmp_path):
    assert (
        main(
            [
                "run",
                "--workspace",
                str(tmp_path),
                "--news",
                str(FIXTURES / "synthetic_news.jsonl"),
                "--prices",
                str(FIXTURES / "synthetic_prices.csv"),
                "--market",
                "TEST_MARKET",
                "--timezone",
                "UTC",
                "--interval-minutes",
                "30",
                "--as-of",
                CUTOFF.isoformat(),
                "--extractor",
                "rule",
            ]
        )
        == 0
    )
    output = next((tmp_path / "runs").iterdir())
    assert {"report.md", "manifest.json", "features.csv", "package.json", "events.json", "review-queue.json"}.issubset(
        p.name for p in output.iterdir()
    )


def test_repeat_qualification_never_uses_only_the_last_successful_run():
    from test_benchmark_script_diagnostics import _case, _Run

    from scripts.benchmark_real_news_extraction import repeated_runs_qualified

    failed = _Run([_case("R01", passed=False, disposition="quarantine")], qualified=False)
    passed = _Run([_case("R01", passed=True, disposition="event")])
    assert not repeated_runs_qualified([failed, passed])
    assert repeated_runs_qualified([passed, passed])


def test_date_only_facts_and_confidence_survive_while_time_gate_stays_closed():
    from test_news_realism_regressions import _candidate, _extract

    result = _extract(_candidate(time_precision="day", time_text="2026-09-01", event_instant=None))
    event = result.events[0]
    assert event.capacity_mw == 870
    assert event.confidence == 0.99
    assert event.analysis_eligibility == "needs_time_review"
    assert event.effective_start_at is None
    assert event.time_resolution.precision == "day"


def test_an_isolated_signal_can_be_supported_when_matched_controls_are_sufficient(study):
    from app.research.news.prices import PriceObservations

    event = next(e for e in study.view.events if e.event_type == "generation_outage")
    start = event.effective_start_at - timedelta(days=100)
    timestamps = tuple(start + i * CLOCK.interval for i in range(200 * 48))
    values = tuple(
        150.0 if event.effective_start_at <= t < event.effective_start_at + timedelta(hours=1) else 50.0
        for t in timestamps
    )
    result = (
        EventPriceAnalyzer(AnalysisMethod(window_hours=(1.0,)))
        .analyze(PriceObservations(CLOCK, timestamps, values), [event], axis="effective")
        .results[0]
    )
    assert result.control_sample_size > 100
    assert result.conclusion == "association_consistent_with_expected_direction"
    assert result.metrics.deviation == 100


def test_technical_failure_is_retried_after_restart_and_attempts_are_preserved(tmp_path):
    from app.research.news.contracts import EventExtractionResult, ExtractionQuarantine

    document = NewsNormalizer().normalize(JsonlCollectedNewsAdapter(FIXTURES / "synthetic_news.jsonl").load()[0])

    class OnceFailing(CountingExtractor):
        def extract(self, doc):
            self.calls += 1
            if self.calls == 1:
                return EventExtractionResult(
                    document_version_id=doc.document_version_id,
                    quarantine=ExtractionQuarantine(
                        document_version_id=doc.document_version_id,
                        reason_code="model_unavailable",
                        message="temporary",
                        extractor_id=self.extractor_id,
                        extractor_version=self.extractor_version,
                    ),
                )
            return ObviousNewsEventExtractor.extract(self, doc)

    delegate = OnceFailing()
    first = CachedNewsExtractor(delegate, NewsWorkspace(tmp_path), {})
    assert first.extract(document).quarantine
    second = CachedNewsExtractor(delegate, NewsWorkspace(tmp_path), {})
    assert second.extract(document).events
    assert delegate.calls == 2
    with NewsWorkspace(tmp_path).connect() as db:
        assert db.execute("SELECT COUNT(*) FROM extraction_attempts").fetchone()[0] == 2


def test_reviewed_quarantine_leaves_pending_queue_without_erasing_its_audit(tmp_path):
    from app.research.news.contracts import EventExtractionResult, ExtractionQuarantine

    workspace = NewsWorkspace(tmp_path)
    document = NewsNormalizer().normalize(JsonlCollectedNewsAdapter(FIXTURES / "synthetic_news.jsonl").load()[0])
    result = EventExtractionResult(
        document_version_id=document.document_version_id,
        quarantine=ExtractionQuarantine(
            document_version_id=document.document_version_id,
            reason_code="invalid_evidence",
            message="missing quote",
            extractor_id="test",
            extractor_version="1",
        ),
    )
    workspace.cache_result("key", document, result)
    workspace.review("key", decision="rejected", reviewer="tester", reason="not applicable", market_timezone="UTC")
    review_doc = NewsNormalizer().normalize(workspace.review_records()[0])

    class Delegate(CountingExtractor):
        def extract(self, doc):
            return result

    assembler = AsOfEventAssembler(
        NewsVersionStore((document, review_doc)), extractor=CachedNewsExtractor(Delegate(), workspace, {})
    )
    assert assembler.view_at(CUTOFF).quarantined[0].review_status == "unreviewed"
    now = assembler.view_at(review_doc.available_at)
    assert now.quarantined
    assert all(q.review_status == "rejected" for q in now.quarantined)


def test_corrected_review_revalidates_source_before_creating_a_new_event(tmp_path):
    from test_news_realism_regressions import BODY, _candidate, _document, _extract

    workspace = NewsWorkspace(tmp_path)
    document = _document(body=BODY)
    wrong = _extract(_candidate(affected_assets=["Invented Plant"]))
    assert wrong.quarantine
    workspace.cache_result("wrong", document, wrong)
    with pytest.raises(ValueError, match="证据门禁"):
        workspace.review("wrong", decision="corrected", reviewer="tester", reason="校正资产",
                         corrected={"disposition": "event", "events": [_candidate(affected_assets=["Invented Plant"])]},
                         market_timezone="UTC")
    assert workspace.review_records() == ()
    workspace.review("wrong", decision="corrected", reviewer="tester", reason="按原文校正资产",
                     corrected={"disposition": "event", "events": [_candidate()]}, market_timezone="UTC")
    revision = NewsNormalizer().normalize(workspace.review_records()[0])
    checked = workspace.reviewed_result(revision)
    assert checked.events[0].review_status == "corrected"
    assert checked.events[0].affected_assets == ("Plant A",)
    assert checked.events[0].announcement_available_at == revision.available_at
    assert all(span.document_version_id == revision.document_version_id for span in checked.events[0].evidence)
