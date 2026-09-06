"""Verify that benchmark diagnostics remain attached to the correct document."""

from types import SimpleNamespace

from scripts.benchmark_real_news_extraction import (
    _DiagnosticExtractor,
    _model_output_collector,
    build_parser,
)


def test_the_benchmark_measures_what_the_extractor_actually_does_by_default():
    """A benchmark run must reflect production behaviour, not a cheaper configuration.

    The extra passes cost three times the model calls, which is a real trade — but a run
    that quietly measured a single pass would report stability the deployed default does
    not have.
    """

    assert build_parser().parse_args([]).extraction_passes == 3
    assert build_parser().parse_args(["--extraction-passes", "1"]).extraction_passes == 1


def test_repeat_runs_are_available_for_measuring_stability():
    """Instability has to be measured; a single run cannot report whether a case flips."""

    assert build_parser().parse_args([]).repeat == 1
    assert build_parser().parse_args(["--repeat", "5"]).repeat == 5


def test_collector_keeps_repair_attempts_under_the_same_corpus_id():
    observe, select, captured = _model_output_collector()

    select("R01")
    observe({"parsed": None})
    observe({"parsed": {"ok": True}})
    select("R02")
    observe({"parsed": {"ok": True}})

    assert captured == {
        "R01": [{"parsed": None}, {"parsed": {"ok": True}}],
        "R02": [{"parsed": {"ok": True}}],
    }


def test_diagnostic_extractor_selects_the_document_before_extraction():
    selected: list[str] = []

    class Delegate:
        extractor_id = "test"
        extractor_version = "1.0.0"

        def extract(self, document):
            assert selected == ["R07"]
            return "result"

    extractor = _DiagnosticExtractor(Delegate(), selected.append)
    document = SimpleNamespace(raw_metadata={"corpus_id": "R07"})

    assert extractor.extract(document) == "result"


def _case(corpus_id: str, *, passed: bool, disposition: str):
    from app.research.news.benchmark import RealNewsCaseResult

    return RealNewsCaseResult(
        corpus_id=corpus_id,
        source_ref="fixture://case",
        expected_disposition="event",
        actual_disposition=disposition,
        expected_event_types=("demand_shock",),
        evidence_span_count=0,
        passed=passed,
    )


class _Run:
    def __init__(self, cases, qualified=True):
        self.cases = cases
        self.qualified = qualified


def test_stability_reports_a_case_that_flips_even_when_the_totals_hold_still():
    """Two cases trading places leave every metric unchanged; that is not stability."""

    from scripts.benchmark_real_news_extraction import _stability_summary

    first = _Run([_case("R05", passed=True, disposition="quarantine"), _case("R08", passed=False, disposition="irrelevant")])
    second = _Run([_case("R05", passed=False, disposition="event"), _case("R08", passed=True, disposition="quarantine")])

    summary = _stability_summary([first, second])

    assert summary["run_count"] == 2
    assert summary["unstable_verdicts"] == ["R05", "R08"]
    assert summary["metric_stability_is_incidental"] is True
    assert summary["cases"]["R05"]["pass_count"] == 1
    assert "actual_disposition" in summary["cases"]["R05"]["varying_fields"]


def test_stability_says_so_when_every_run_agreed():
    from scripts.benchmark_real_news_extraction import _stability_summary

    run = _Run([_case("R01", passed=True, disposition="event")])
    summary = _stability_summary([run, run])

    assert summary["unstable_cases"] == []
    assert summary["metric_stability_is_incidental"] is False


def test_cost_summary_reports_calls_per_document_not_just_the_pass_count():
    from scripts.benchmark_real_news_extraction import _cost_summary

    summary = _cost_summary(
        [{"model_calls": 30, "model_seconds": 60.0, "wall_seconds": 65.0}],
        document_count=10,
    )

    assert summary["model_calls_per_document"] == 3.0
    assert summary["seconds_per_document"] == 6.5
