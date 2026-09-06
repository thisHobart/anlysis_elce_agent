"""Regression suite for the six P2 research questions.

Each test answers one question end to end against the committed fixtures, the way the
proposal states it. These are acceptance checks, not unit tests: they exercise the whole
path from a collected-news snapshot to a conclusion, and they are what should fail first
if a later change quietly breaks a research claim.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from app.research.news import (
    CollectedNewsRecord,
    JsonlCollectedNewsAdapter,
    MarketClock,
    NewsNormalizer,
    ObviousNewsEventExtractor,
    StructuredNewsEventExtractor,
    is_usable_at,
    lead_time_table,
    load_price_csv,
    render_report,
    run_news_price_study,
    zero_point,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "news_price"
NEWS = FIXTURES / "synthetic_news.jsonl"
GOLDEN = FIXTURES / "expected_events.json"
PRICES = FIXTURES / "synthetic_prices.csv"
MANIFEST = FIXTURES / "fixture_manifest.yaml"

CLOCK = MarketClock(market="TEST_MARKET", timezone="UTC", interval_minutes=30)
FINAL_AS_OF = datetime.fromisoformat("2026-03-01T23:30:00+00:00")

OUTAGE_ANNOUNCED_AT = datetime.fromisoformat("2026-01-12T10:07:00+00:00")
CORRECTION_AVAILABLE_AT = datetime.fromisoformat("2026-01-12T12:02:00+00:00")
RETROSPECTIVE_AVAILABLE_AT = datetime.fromisoformat("2026-01-13T08:05:00+00:00")


def _instant(value: str) -> datetime:
    return datetime.fromisoformat(value)


@pytest.fixture(scope="module")
def documents():
    return NewsNormalizer().normalize_many(JsonlCollectedNewsAdapter(NEWS).load())


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def manifest() -> dict:
    return yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def prices():
    return load_price_csv(PRICES, CLOCK)


@pytest.fixture(scope="module")
def effective_study(prices):
    return run_news_price_study(
        adapter=JsonlCollectedNewsAdapter(NEWS), prices=prices, as_of=FINAL_AS_OF, axis="effective"
    )


@pytest.fixture(scope="module")
def announcement_study(prices):
    return run_news_price_study(
        adapter=JsonlCollectedNewsAdapter(NEWS), prices=prices, as_of=FINAL_AS_OF, axis="announcement"
    )


def _fixture_id(document) -> str:
    return document.raw_metadata["fixture_id"]


def _event_of(study, fixture_id: str):
    """Find the merged event a given fixture contributed to."""

    extractor = ObviousNewsEventExtractor(market_timezone=CLOCK.timezone)
    document = next(item for item in study.documents if _fixture_id(item) == fixture_id)
    result = extractor.extract(document)
    assert result.events, f"{fixture_id} 未产出事件"
    return study.view.by_id(result.events[0].event_id)


# ---------------------------------------------------------------------------
# 研究问题 1：已收集新闻能否被稳定规范化，并保留来源、版本、内容哈希和三类时间语义？
# ---------------------------------------------------------------------------


def test_question_1_news_normalizes_stably_with_source_version_hash_and_three_times(documents) -> None:
    repeated = NewsNormalizer().normalize_many(JsonlCollectedNewsAdapter(NEWS).load())
    assert repeated == documents, "同一输入两次规范化必须完全一致"

    assert len(documents) == 10
    assert len({document.document_version_id for document in documents}) == 10
    # N01 and its correction are two versions of ONE document, so identities are 9, not 10.
    assert len({document.document_id for document in documents}) == 9

    for document in documents:
        assert document.source_name and document.source_document_id
        assert document.source_ref.startswith("fixture://news/")
        assert document.version >= 1
        assert len(document.content_hash) == 64
        # The three time semantics stay separate and never collapse into one stamp.
        assert document.published_at.utcoffset().total_seconds() == 0
        assert document.first_seen_at.utcoffset().total_seconds() == 0
        assert document.first_seen_at >= document.published_at
        assert document.available_at == max(document.published_at, document.first_seen_at)

    converted = next(item for item in documents if _fixture_id(item) == "N06")
    assert converted.published_at == _instant("2026-01-21T09:00:00+00:00"), "非 UTC 来源必须换算"

    original = next(item for item in documents if _fixture_id(item) == "N01")
    normalizer = NewsNormalizer()
    record = next(
        item for item in JsonlCollectedNewsAdapter(NEWS).load() if item.metadata["fixture_id"] == "N01"
    )
    spaced = normalizer.normalize(record.model_copy(update={"title": f"  {record.title}  "}))
    changed = normalizer.normalize(
        record.model_copy(update={"body": record.body.replace("500 MW", "450 MW")})
    )
    assert spaced.content_hash == original.content_hash, "空白差异不得改变内容身份"
    assert changed.content_hash != original.content_hash, "实质内容变化必须改变内容身份"


# ---------------------------------------------------------------------------
# 研究问题 2：对含义明确的测试新闻，抽取结果能否正确识别相关性、事件类型、区域、
#             资产、容量和生效区间？
# ---------------------------------------------------------------------------


def test_question_2_obvious_news_yields_the_expected_fields_with_verbatim_evidence(documents, golden) -> None:
    extractor = ObviousNewsEventExtractor(market_timezone=CLOCK.timezone)

    for document in documents:
        fixture_id = _fixture_id(document)
        expected = golden[fixture_id]
        result = extractor.extract(document)
        assert result.quarantine is None, f"{fixture_id} 不应进入隔离区"
        assert result == extractor.extract(document), f"{fixture_id} 两次抽取必须一致"

        event = result.events[0]
        assert event.relevance == expected["relevance"], fixture_id
        assert event.event_type == expected["event_type"], fixture_id
        assert list(event.affected_regions) == expected["affected_regions"], fixture_id
        assert list(event.affected_assets) == expected["affected_assets"], fixture_id
        assert event.capacity_mw == expected["capacity_mw"], fixture_id
        assert event.direction == expected["direction"], fixture_id
        assert _iso_z(event.effective_start_at) == expected["effective_start_at"], fixture_id
        assert _iso_z(event.effective_end_at) == expected["effective_end_at"], fixture_id

        basis = event.time_resolution.basis if event.time_resolution else None
        assert basis == expected["time_basis"], fixture_id

        # Every field that carries a value must point at the exact characters it came from.
        fields = {span.field_name for span in event.evidence}
        for span in event.evidence:
            source = getattr(document, span.text_field)
            assert source[span.start_char : span.end_char] == span.quote, fixture_id
        if event.event_type != "unknown":
            assert {"relevance", "event_type", "direction"}.issubset(fields), fixture_id
        for field_name, value in (
            ("affected_regions", event.affected_regions),
            ("affected_assets", event.affected_assets),
            ("capacity_mw", event.capacity_mw),
            ("effective_start_at", event.effective_start_at),
            ("effective_end_at", event.effective_end_at),
        ):
            if value:
                assert field_name in fields, f"{fixture_id} 的 {field_name} 缺少证据"


def test_question_2_unrecognized_news_fabricates_nothing(documents) -> None:
    record = next(item for item in JsonlCollectedNewsAdapter(NEWS).load() if item.metadata["fixture_id"] == "N09")
    bland = NewsNormalizer().normalize(
        record.model_copy(
            update={
                "source_document_id": "UNKNOWN-1",
                "source_ref": "fixture://news/UNKNOWN-1",
                "title": "季度例行信息",
                "body": "本季度办公室完成了常规设备盘点。",
            }
        )
    )

    event = ObviousNewsEventExtractor().extract(bland).events[0]

    assert event.event_type == "unknown"
    assert event.relevance == "irrelevant"
    assert event.affected_regions == () and event.affected_assets == ()
    assert event.capacity_mw is None
    assert event.effective_start_at is None
    assert event.evidence == ()


# ---------------------------------------------------------------------------
# 研究问题 3：同一事件的转载、重复抓取和更正新闻能否避免重复计数，并在任意 as_of
#             时点重建当时可见版本？
# ---------------------------------------------------------------------------


def test_question_3_reposts_and_revisions_never_double_count_and_replay_correctly(prices) -> None:
    def view_at(as_of: datetime):
        return run_news_price_study(
            adapter=JsonlCollectedNewsAdapter(NEWS), prices=prices, as_of=as_of, axis="effective"
        ).view

    just_announced = view_at(_instant("2026-01-12T10:10:00+00:00"))
    after_repost = view_at(_instant("2026-01-12T10:30:00+00:00"))
    after_correction = view_at(_instant("2026-01-12T13:00:00+00:00"))
    after_retrospective = view_at(RETROSPECTIVE_AVAILABLE_AT)

    outages = [
        [event for event in view.events if event.event_type == "generation_outage"]
        for view in (just_announced, after_repost, after_correction, after_retrospective)
    ]
    assert all(len(group) == 1 for group in outages), "转载与更正不得产生第二个停运事件"
    assert len({group[0].event_id for group in outages}) == 1, "同一事件的身份必须跨视图保持稳定"

    first, repost, corrected, retrospective = (group[0] for group in outages)

    # A repost adds provenance and nothing else.
    assert first.capacity_mw == 500.0
    assert repost.capacity_mw == 500.0
    assert repost.revision_count == first.revision_count + 1
    assert len(set(repost.document_refs)) == len(repost.document_refs)

    # The correction supersedes the figure; the announcement clock does not move.
    assert corrected.capacity_mw == 350.0
    assert corrected.announcement_available_at == OUTAGE_ANNOUNCED_AT
    assert first.announcement_available_at == OUTAGE_ANNOUNCED_AT

    # A later retrospective must not resurrect the superseded figure.
    assert retrospective.capacity_mw == 350.0
    assert retrospective.announcement_available_at == OUTAGE_ANNOUNCED_AT

    # Point-in-time replay: the correction is invisible before it was published.
    assert view_at(_instant("2026-01-12T12:01:00+00:00")).events[0].capacity_mw == 500.0
    assert view_at(CORRECTION_AVAILABLE_AT).events[0].capacity_mw == 350.0

    # The retrospective simply does not exist for anyone standing before it was collected.
    before = view_at(_instant("2026-01-13T08:04:00+00:00"))
    assert all(
        ref.available_at < RETROSPECTIVE_AVAILABLE_AT
        for event in before.events
        for ref in event.document_refs
    )


def test_question_3_identical_recollection_is_dropped_rather_than_counted_twice(prices) -> None:
    from app.research.news import NewsVersionStore

    documents = NewsNormalizer().normalize_many(JsonlCollectedNewsAdapter(NEWS).load())
    store = NewsVersionStore(documents)
    before = store.version_count

    for document in documents:
        assert store.add(document) is False, "完全相同的重复采集必须被丢弃"

    assert store.version_count == before
    assert store.duplicates.duplicate_count == len(documents)


# ---------------------------------------------------------------------------
# 研究问题 4：新闻可获得时间、事件生效时间和价格时间轴能否在同一市场时钟下
#             无泄漏地对齐？
# ---------------------------------------------------------------------------


def test_question_4_the_three_time_axes_align_on_one_clock_without_leakage(effective_study) -> None:
    view = effective_study.view

    # No event may ever be announced after the as_of it belongs to; the contract enforces it.
    assert all(event.announcement_available_at <= view.as_of for event in view.events)

    advance = _event_of(effective_study, "N06")
    assert zero_point(advance, "announcement") == _instant("2026-01-21T09:03:00+00:00")
    assert zero_point(advance, "effective") == _instant("2026-01-22T14:00:00+00:00")
    assert zero_point(advance, "announcement") != zero_point(advance, "effective")

    readings = {reading.event_id: reading for reading in lead_time_table(view.events)}
    advance_reading = readings[advance.event_id]
    assert advance_reading.is_advance_notice
    assert advance_reading.lead_time_hours == pytest.approx(28.95, abs=0.01)

    outage = _event_of(effective_study, "N01")
    assert readings[outage.event_id].is_after_the_fact, "10:00 发生、10:07 才可用，属于事后可用"

    # An event cannot inform a decision taken before it was available, ever.
    assert not is_usable_at(advance, _instant("2026-01-21T09:02:00+00:00"))
    assert is_usable_at(advance, _instant("2026-01-21T09:03:00+00:00"))

    # Every price timestamp and every feature row sit on the same settlement grid.
    clock = effective_study.clock
    for row in effective_study.features.rows[:48]:
        assert clock.floor(row.interval_start) == row.interval_start


def test_question_4_features_never_count_an_event_before_it_was_announced(prices) -> None:
    early = _instant("2026-01-12T11:00:00+00:00")
    study = run_news_price_study(
        adapter=JsonlCollectedNewsAdapter(NEWS), prices=prices, as_of=early, axis="effective"
    )
    rows = {row.interval_start: row for row in study.features.rows}

    before_announcement = rows[_instant("2026-01-12T09:30:00+00:00")]
    # 10:00 is the sharpest case in the whole fixture set: the outage is already in effect,
    # but the bulletin only became available at 10:07. Counting it here would be leakage.
    in_effect_but_not_yet_known = rows[_instant("2026-01-12T10:00:00+00:00")]
    after_announcement = rows[_instant("2026-01-12T10:30:00+00:00")]

    assert before_announcement.active_event_count == 0
    assert before_announcement.active_capacity_mw is None, "无活跃事件是未知，不是数值零"
    assert in_effect_but_not_yet_known.active_event_count == 0, "事件已生效但尚未可用，不得计入"
    assert in_effect_but_not_yet_known.active_capacity_mw is None
    assert after_announcement.active_event_count == 1
    assert after_announcement.active_capacity_mw == 500.0
    assert after_announcement.direction_up_count == 1
    assert after_announcement.event_type_counts["generation_outage"] == 1


# ---------------------------------------------------------------------------
# 研究问题 5：给定金标准事件，确定性分析能否恢复合成价格中注入的方向和时间窗口，
#             同时不为无关新闻和长期新闻制造短期效应？
# ---------------------------------------------------------------------------


def test_question_5_analysis_recovers_the_injected_direction_and_window(effective_study, manifest) -> None:
    injected = {item["fixture_id"]: item for item in manifest["effects"]}
    analysis = effective_study.analysis

    for fixture_id, expected_delta in (("N01", 80.0), ("N05", 45.0), ("N06", 35.0), ("N07", -70.0)):
        event = _event_of(effective_study, fixture_id)
        short_window = next(
            result
            for result in analysis.for_event(event.event_id)
            if result.metrics.window_label == "[0,1h]"
        )
        assert short_window.conclusion == "association_consistent_with_expected_direction", fixture_id
        # Sign must match, and the magnitude must land near the number the manifest injected.
        assert short_window.metrics.deviation == pytest.approx(expected_delta, rel=0.35), fixture_id
        assert short_window.metrics.deviation * injected[fixture_id]["delta"] > 0, fixture_id
        assert short_window.control_sample_size > 100, fixture_id
        assert short_window.placebo_deviations, fixture_id


def test_question_5_negative_controls_never_become_an_effect(effective_study, announcement_study, manifest) -> None:
    controls = {item["fixture_id"] for item in manifest["negative_controls"]}
    assert controls == {"N04", "N08", "N09", "N10"}

    for study in (effective_study, announcement_study):
        for fixture_id in ("N04", "N08"):
            event = _event_of(study, fixture_id)
            if event is None:
                continue
            for result in study.analysis.for_event(event.event_id):
                assert result.conclusion != "association_consistent_with_expected_direction", (
                    f"{fixture_id} 在 {study.analysis.axis} 轴上被误报为稳定效应"
                )

    # The long-horizon plan has no zero point inside the price window at all.
    long_horizon = _event_of(effective_study, "N08")
    assert long_horizon.effective_start_at.year == 2030
    assert any(
        event_id == long_horizon.event_id for event_id, _ in effective_study.analysis.excluded_events
    )

    # The retrospective contributes provenance to the outage; it never becomes its own event.
    assert _event_of(effective_study, "N10").event_id == _event_of(effective_study, "N01").event_id

    # Irrelevant news is kept out of the analysis and out of the numeric features entirely.
    irrelevant = _event_of(effective_study, "N09")
    assert irrelevant.event_type == "irrelevant"
    assert effective_study.analysis.for_event(irrelevant.event_id) == ()
    assert irrelevant.event_id not in effective_study.features.source_event_ids


def test_question_5_advance_notice_separates_the_two_timelines(effective_study, announcement_study) -> None:
    event_id = _event_of(effective_study, "N06").event_id

    on_effect = {result.metrics.window_label: result for result in effective_study.analysis.for_event(event_id)}
    on_announcement = {
        result.metrics.window_label: result for result in announcement_study.analysis.for_event(event_id)
    }

    assert on_effect["[0,1h]"].conclusion == "association_consistent_with_expected_direction"
    assert on_announcement["[0,1h]"].conclusion == "not_supported_by_current_data", (
        "公告发出的第二天才生效，公告窗内不应出现被设计的效应"
    )


def test_question_5_every_outcome_the_manifest_predicts_actually_happens(
    effective_study, announcement_study, manifest
) -> None:
    """The manifest is the single answer key; the tests must not keep their own copy."""

    studies = {
        "announcement_axis": announcement_study,
        "effective_axis": effective_study,
    }
    checked = 0
    for axis_key, expectations in manifest["expected_outcomes"].items():
        study = studies[axis_key]
        for fixture_id, expected_conclusion in expectations.items():
            event = _event_of(study, fixture_id)
            short_window = next(
                result
                for result in study.analysis.for_event(event.event_id)
                if result.metrics.window_label == "[0,1h]"
            )
            assert short_window.conclusion == expected_conclusion, (
                f"{axis_key}/{fixture_id}：预期 {expected_conclusion}，实际 {short_window.conclusion}"
            )
            checked += 1

    assert checked == 7, "manifest 中的预期结论条数发生变化时必须同步复核"


def test_question_5_analysis_tolerates_zero_and_negative_prices(prices, effective_study) -> None:
    assert min(prices.prices) < 0, "夹具必须包含负价，用来证明分析不依赖对数"
    assert effective_study.analysis.results, "存在负价时分析仍须产出结果"
    assert any("对数" in note for note in effective_study.analysis.notes)


def test_question_5_the_same_input_produces_the_same_result_twice(prices) -> None:
    first = run_news_price_study(
        adapter=JsonlCollectedNewsAdapter(NEWS), prices=prices, as_of=FINAL_AS_OF, axis="effective"
    )
    second = run_news_price_study(
        adapter=JsonlCollectedNewsAdapter(NEWS), prices=prices, as_of=FINAL_AS_OF, axis="effective"
    )

    assert first.analysis.content_hash() == second.analysis.content_hash()
    assert first.features.content_hash == second.features.content_hash
    assert first.package.fingerprint() == second.package.fingerprint()


# ---------------------------------------------------------------------------
# 研究问题 6：每个结论能否回链到新闻证据片段、事件记录、价格窗口、方法版本和
#             输入输出哈希？
# ---------------------------------------------------------------------------


def test_question_6_every_conclusion_traces_back_to_the_text_it_rests_on(effective_study) -> None:
    package = effective_study.package
    documents = {document.document_version_id: document for document in effective_study.documents}

    assert package.evidence_links, "证据链不能为空"
    assert package.untraceable_links() == (), "每个结论都必须能回链到原文"

    for link in package.evidence_links:
        assert link.conclusion in {
            "association_consistent_with_expected_direction",
            "not_supported_by_current_data",
            "insufficient_sample",
            "contradicts_expected_direction",
        }
        assert link.conclusion_reason
        assert link.window_end_at > link.window_start_at
        assert link.document_version_ids and link.content_hashes and link.source_names
        assert len(link.content_hashes) == len(link.document_version_ids)

        # The quote must actually appear in one of the cited documents, verbatim.
        cited = [documents[version_id] for version_id in link.document_version_ids]
        for _field_name, quote in link.quotes:
            assert any(quote in document.title or quote in document.body for document in cited), quote


def test_question_6_the_quoted_text_matches_the_value_the_merge_actually_used(effective_study) -> None:
    """A citation from a document that did not set the field would be traceable but wrong."""

    outage = _event_of(effective_study, "N01")
    assert outage.capacity_mw == 350.0, "更正后的容量"
    assert outage.revision_count == 4, "四个文档版本共同构成这一个事件"

    link = next(
        item for item in effective_study.package.evidence_links if item.event_id == outage.event_id
    )
    quotes = link.primary_quotes
    assert quotes["capacity_mw"] == "350", "引用必须来自定值的更正版本，而不是原始或转载版本"
    # The full citation list still keeps the superseded wording, for audit rather than display.
    assert ("capacity_mw", "500") in link.quotes
    assert quotes["effective_start_at"] == "2026-01-12T10:00:00Z", (
        "引用必须来自原始来源的绝对时间戳，而不是回顾报道里的“昨日 10:00”"
    )


def test_question_6_the_package_carries_method_version_and_input_output_hashes(effective_study) -> None:
    package = effective_study.package
    fingerprint = package.fingerprint()

    # Two separate questions, so two separate fingerprints: "did the conclusion change?"
    # and "was it produced the same way?". Only the second may move on a re-extraction.
    assert set(package.conclusion_fingerprint()) == {
        "analysis_hash",
        "event_hash",
        "feature_hash",
        "package_hash",
        "price_hash",
    }
    assert set(package.provenance_fingerprint()) == {"provenance_hash"}
    assert set(fingerprint) == set(package.conclusion_fingerprint()) | {"provenance_hash"}
    assert all(len(value) == 64 for value in fingerprint.values())

    method = package.analysis.method
    assert method.analysis_version and method.correction == "holm"
    assert method.window_hours == (1.0, 6.0, 24.0)
    assert package.events[0].merger_version
    assert package.features.feature_version


def test_question_6_the_rendered_report_states_the_limits_of_its_own_conclusions(effective_study) -> None:
    report = render_report(effective_study.package)

    assert "P2 事件—电价证据包" in report
    assert "方法声明" in report and "指纹" in report
    assert "不等于证明没有影响" in report, "报告必须说明未显著不等于没有影响"
    assert "事件关联也不等于因果效应" in report
    assert effective_study.package.fingerprint()["package_hash"] in report


def test_question_6_quality_report_precedes_and_summarises_the_inputs(effective_study) -> None:
    quality = effective_study.package.quality

    assert quality.document_count == 9
    assert quality.version_count == 10
    assert quality.merged_event_count == len(effective_study.view.events)
    assert quality.evidence_coverage == 1.0
    assert quality.after_the_fact_event_count >= 1, "至少有一个事件是事后才可用的，必须被点出来"


def _iso_z(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat().replace("+00:00", "Z")


def test_question_6_the_report_names_every_event_and_every_quarantined_document(
    effective_study, prices, tmp_path
) -> None:
    """A count alone lets a growing quarantine pass as a healthy run."""

    package = effective_study.package
    report = render_report(package)

    # Nothing may vanish between the quality section and the results.
    assert package.unexplained_events == (), "每个事件要么有结果，要么写明未纳入的原因"
    for event in package.events:
        assert event.event_id in report, f"{event.event_id} 未出现在报告中"
    assert "本次运行没有文档进入隔离区" in report

    # Now feed in a document the baseline cannot turn into a legal event, and check it is
    # named rather than merely counted.
    vague = json.loads(
        json.dumps(
            {
                "source_name": "测试市场运营方",
                "source_document_id": "N11",
                "source_ref": "fixture://news/N11",
                "version": 1,
                "title": "GEN_C 突发停运",
                "body": "区域 TEST_NORTH，资产 GEN_C 今日上午突发停运，不可用容量 200 MW。",
                "published_at": "2026-01-15T06:00:00+00:00",
                "collected_at": "2026-01-15T06:02:00+00:00",
                "updated_at": None,
                "language": "zh-CN",
                "market_tags": ["TEST_MARKET"],
                "metadata": {"fixture_id": "N11", "scenario": "vague_time"},
            }
        )
    )
    snapshot = tmp_path / "with_quarantine.jsonl"
    snapshot.write_text(
        NEWS.read_text(encoding="utf-8") + json.dumps(vague, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with_quarantine = run_news_price_study(
        adapter=JsonlCollectedNewsAdapter(snapshot), prices=prices, as_of=FINAL_AS_OF, axis="effective"
    )
    quarantined_report = render_report(with_quarantine.package)

    assert with_quarantine.package.quality.quarantined_count == 1
    assert with_quarantine.package.quality.quarantine_reasons == {"missing_effective_start": 1}
    assert "隔离区明细" in quarantined_report
    assert "短期事件没有可解析的生效开始时间" in quarantined_report
    assert with_quarantine.package.quarantined[0].document_version_id in quarantined_report

    # The quarantined document contributed no event, and the fingerprint reflects its presence.
    assert len(with_quarantine.view.events) == len(effective_study.view.events)
    assert with_quarantine.package.fingerprint() != package.fingerprint()


# ---------------------------------------------------------------------------
# 研究问题 6（续）：端点不可复现时，结论指纹还能不能回答“结论变了吗”？
# ---------------------------------------------------------------------------

DRIFT_BODY = (
    "区域 TEST_NORTH 电力供需紧张，2026-01-12T10:00:00Z 全网用电负荷创历史新高。"
)
DRIFT_EVIDENCE_FIELDS = (
    "relevance",
    "event_type",
    "status",
    "physical_effect",
    "affected_regions",
    "effective_start_at",
)


def _drift_payload(region: str) -> dict:
    """One event, told twice in different words. Only `region` changes."""

    return {
        "disposition": "event",
        "events": [
            {
                "relevance": "short_term",
                "event_type": "demand_shock",
                "status": "occurred",
                "physical_effect": "demand_up",
                "affected_regions": [region],
                "affected_assets": [],
                "quantity": None,
                "time_precision": "instant",
                "time_text": "2026-01-12T10:00:00Z",
                "event_instant": {"iso": "2026-01-12T10:00:00+00:00", "basis": "stated_absolute"},
                "event_end_instant": None,
                "confidence": 0.95,
                "evidence": [
                    {"field_name": name, "text_field": "body", "quote": DRIFT_BODY}
                    for name in DRIFT_EVIDENCE_FIELDS
                ],
            }
        ],
    }


class _RewordingGateway:
    """Stands in for the real endpoint, which answers differently on every call."""

    model_name = "drift-model"

    def __init__(self, region: str) -> None:
        self.region = region

    def invoke_structured(self, *, messages, schema):
        return schema.model_validate(_drift_payload(self.region))


class _SingleRecordAdapter:
    def __init__(self, record) -> None:
        self._record = record

    def load(self):
        return (self._record,)


def _drift_study(prices, region: str):
    record = CollectedNewsRecord(
        source_name="测试来源",
        source_document_id="DRIFT-01",
        source_ref="fixture://news/DRIFT-01",
        title="用电负荷创历史新高",
        body=DRIFT_BODY,
        published_at="2026-01-12T09:00:00+00:00",
        collected_at="2026-01-12T09:30:00+00:00",
        language="zh-CN",
        market_tags=("TEST_MARKET",),
    )
    return run_news_price_study(
        adapter=_SingleRecordAdapter(record),
        prices=prices,
        as_of=FINAL_AS_OF,
        axis="effective",
        extractor=StructuredNewsEventExtractor(_RewordingGateway(region), market_timezone="UTC"),
    )


def test_question_6_a_reworded_re_extraction_keeps_the_same_conclusion_fingerprint(prices) -> None:
    """The endpoint is not reproducible: 9 of 10 holdout documents hashed differently每次.

    A conclusion fingerprint that moved with the model's raw output could never answer the
    only question it is asked — did anything about this conclusion change?
    """

    plain = _drift_study(prices, "TEST_NORTH")
    with_grid_suffix = _drift_study(prices, "TEST_NORTH电网")

    plain_traces = [trace.output_hash for event in plain.view.events for trace in event.extraction_traces]
    reworded_traces = [
        trace.output_hash for event in with_grid_suffix.view.events for trace in event.extraction_traces
    ]
    assert plain_traces and plain_traces != reworded_traces, (
        "前提不成立：两次抽取的模型原始输出必须不同，否则这个测试什么也没证明"
    )
    assert plain.view.events[0].affected_regions != with_grid_suffix.view.events[0].affected_regions
    assert plain.view.events[0].region_keys == with_grid_suffix.view.events[0].region_keys

    assert plain.package.conclusion_fingerprint() == with_grid_suffix.package.conclusion_fingerprint()


def test_question_6_provenance_still_moves_when_the_model_output_does(prices) -> None:
    """Provenance is not discarded, only held apart; it must still register the difference."""

    plain = _drift_study(prices, "TEST_NORTH")
    with_grid_suffix = _drift_study(prices, "TEST_NORTH电网")

    assert plain.package.provenance_fingerprint() != with_grid_suffix.package.provenance_fingerprint()
    assert plain.package.fingerprint() != with_grid_suffix.package.fingerprint()


def test_question_6_a_changed_conclusion_still_moves_the_conclusion_fingerprint(prices) -> None:
    """The stability must come from ignoring noise, not from ignoring everything."""

    baseline = _drift_study(prices, "TEST_NORTH")
    other_region = _drift_study(prices, "TEST_SOUTH")

    assert baseline.package.conclusion_fingerprint() != other_region.package.conclusion_fingerprint()
