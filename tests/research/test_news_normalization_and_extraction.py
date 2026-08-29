"""P2 acceptance tests for news normalization and explicit-event extraction."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.research.news import (
    CollectedNewsRecord,
    EventExtractionResult,
    ExternalNewsAdapterNotConfigured,
    ExternalNewsApiPlaceholder,
    ExtractionQuarantine,
    JsonlCollectedNewsAdapter,
    NewsNormalizer,
    ObviousNewsEventExtractor,
    TimeResolution,
)

FIXTURE_DIRECTORY = Path(__file__).resolve().parents[1] / "fixtures" / "news_price"
NEWS_FIXTURE = FIXTURE_DIRECTORY / "synthetic_news.jsonl"
EXPECTED_EVENTS_FIXTURE = FIXTURE_DIRECTORY / "expected_events.json"


def _load_records():
    return JsonlCollectedNewsAdapter(NEWS_FIXTURE).load()


def _load_documents():
    return NewsNormalizer().normalize_many(_load_records())


def test_external_news_api_is_an_explicit_fail_closed_placeholder() -> None:
    with pytest.raises(ExternalNewsAdapterNotConfigured, match="仅为占位符"):
        ExternalNewsApiPlaceholder().load()


def test_collected_news_is_normalized_with_stable_identity_and_three_source_times() -> None:
    records = _load_records()
    normalizer = NewsNormalizer()

    first = normalizer.normalize_many(records)
    second = normalizer.normalize_many(records)

    assert first == second
    assert len(first) == 10
    # N01 and its 12:00 correction are two versions of ONE document, so identities are 9.
    assert len({document.document_id for document in first}) == 9
    assert len({document.document_version_id for document in first}) == len(first)

    for document in first:
        assert document.source_name
        assert document.source_document_id
        assert document.source_ref.startswith("fixture://news/")
        assert document.version >= 1
        assert len(document.content_hash) == 64
        assert document.available_at == max(document.published_at, document.first_seen_at)
        assert document.published_at.utcoffset().total_seconds() == 0
        assert document.first_seen_at.utcoffset().total_seconds() == 0
        if document.updated_at is not None:
            assert document.updated_at.utcoffset().total_seconds() == 0

    updated = next(document for document in first if document.source_document_id == "N08")
    assert updated.published_at == datetime.fromisoformat("2026-01-27T06:00:00+00:00")
    assert updated.first_seen_at == datetime.fromisoformat("2026-01-27T06:05:00+00:00")
    assert updated.updated_at == datetime.fromisoformat("2026-01-27T06:10:00+00:00")

    converted = next(document for document in first if document.source_document_id == "N06")
    assert converted.published_at == datetime.fromisoformat("2026-01-21T09:00:00+00:00")
    assert converted.first_seen_at == datetime.fromisoformat("2026-01-21T09:03:00+00:00")


def test_content_hash_ignores_insignificant_whitespace_but_changes_with_meaningful_content() -> None:
    record = _load_records()[0]
    normalizer = NewsNormalizer()
    original = normalizer.normalize(record)
    whitespace_variant = record.model_copy(
        update={
            "title": f"  {record.title}  ",
            "body": f"\r\n  {record.body.replace('500 MW', '500     MW')}  \r\n",
        }
    )
    changed_variant = record.model_copy(update={"body": record.body.replace("500 MW", "450 MW")})

    assert normalizer.normalize(whitespace_variant).content_hash == original.content_hash
    assert normalizer.normalize(changed_variant).content_hash != original.content_hash


def test_collection_contract_rejects_naive_or_reversed_source_times() -> None:
    payload = _load_records()[0].model_dump()
    payload["published_at"] = "2026-01-12T10:05:00"
    with pytest.raises(ValidationError, match="timezone offset"):
        CollectedNewsRecord.model_validate(payload)

    payload = _load_records()[0].model_dump()
    payload["collected_at"] = "2026-01-12T10:04:00+00:00"
    with pytest.raises(ValidationError, match="must not be before"):
        CollectedNewsRecord.model_validate(payload)


def test_obvious_news_extracts_all_expected_event_fields_and_exact_evidence() -> None:
    expected_by_id = json.loads(EXPECTED_EVENTS_FIXTURE.read_text(encoding="utf-8"))
    extractor = ObviousNewsEventExtractor()

    for document in _load_documents():
        # Keyed by fixture id, not source id: the correction shares its source document.
        expected = expected_by_id[document.raw_metadata["fixture_id"]]
        first = extractor.extract(document)
        second = extractor.extract(document)
        assert first == second
        assert first.document_version_id == document.document_version_id
        assert first.quarantine is None
        assert len(first.events) == 1

        event = first.events[0]
        assert event.relevance == expected["relevance"]
        assert event.event_type == expected["event_type"]
        assert list(event.affected_regions) == expected["affected_regions"]
        assert list(event.affected_assets) == expected["affected_assets"]
        assert event.capacity_mw == expected["capacity_mw"]
        assert event.direction == expected["direction"]
        assert _iso_z(event.effective_start_at) == expected["effective_start_at"]
        assert _iso_z(event.effective_end_at) == expected["effective_end_at"]
        basis = event.time_resolution.basis if event.time_resolution is not None else None
        assert basis == expected["time_basis"]
        if basis == "derived_from_publication":
            assert event.time_resolution.anchor_at == document.published_at
            assert event.time_resolution.market_timezone == "UTC"

        for span in event.evidence:
            source_text = getattr(document, span.text_field)
            assert source_text[span.start_char : span.end_char] == span.quote
            assert span.document_version_id == document.document_version_id

        evidence_fields = {span.field_name for span in event.evidence}
        if event.event_type != "unknown":
            assert {"relevance", "event_type", "direction"}.issubset(evidence_fields)
        if event.affected_regions:
            assert "affected_regions" in evidence_fields
        if event.affected_assets:
            assert "affected_assets" in evidence_fields
        if event.capacity_mw is not None:
            assert "capacity_mw" in evidence_fields
        if event.effective_start_at is not None:
            assert "effective_start_at" in evidence_fields
        if event.effective_end_at is not None:
            assert "effective_end_at" in evidence_fields


def test_unrecognized_news_does_not_fabricate_event_fields() -> None:
    raw = CollectedNewsRecord(
        source_name="测试来源",
        source_document_id="UNKNOWN-1",
        source_ref="fixture://news/UNKNOWN-1",
        title="季度例行信息",
        body="本季度办公室完成了常规设备盘点。",
        published_at="2026-01-30T01:00:00+00:00",
        collected_at="2026-01-30T01:01:00+00:00",
        language="zh-CN",
        market_tags=("TEST_MARKET",),
    )
    document = NewsNormalizer().normalize(raw)

    result = ObviousNewsEventExtractor().extract(document)

    assert result.quarantine is None
    event = result.events[0]
    assert event.relevance == "irrelevant"
    assert event.event_type == "unknown"
    assert event.affected_regions == ()
    assert event.affected_assets == ()
    assert event.capacity_mw is None
    assert event.effective_start_at is None
    assert event.effective_end_at is None
    assert event.evidence == ()


def _iso_z(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat().replace("+00:00", "Z")


def _fixture_document(title: str, body: str, source_document_id: str = "PROBE-1"):
    return NewsNormalizer().normalize(
        CollectedNewsRecord(
            source_name="测试来源",
            source_document_id=source_document_id,
            source_ref=f"fixture://news/{source_document_id}",
            title=title,
            body=body,
            published_at="2026-01-12T10:05:00+00:00",
            collected_at="2026-01-12T10:07:00+00:00",
            language="zh-CN",
            market_tags=("TEST_MARKET",),
        )
    )


def test_short_term_news_without_explicit_start_time_is_quarantined_not_fabricated() -> None:
    """Vague wording is the real-world norm; it must not crash and must not invent a time."""

    document = _fixture_document(
        "GEN_A 突发停运",
        "区域 TEST_NORTH，资产 GEN_A 今日上午突发停运，不可用容量 500 MW，预计傍晚恢复。",
    )

    result = ObviousNewsEventExtractor().extract(document)

    assert result.events == ()
    assert result.quarantine is not None
    assert result.quarantine.reason_code == "missing_effective_start"
    assert result.quarantine.document_version_id == document.document_version_id
    assert result.quarantine.extractor_id == ObviousNewsEventExtractor.extractor_id


def test_one_unusable_document_does_not_discard_the_usable_ones() -> None:
    """A quarantined document must not abort the batch around it."""

    usable = _fixture_document(
        "GEN_A 突发停运",
        "区域 TEST_NORTH，资产 GEN_A 突发停运，不可用容量 500 MW。事件开始：2026-01-12T10:00:00Z。",
        "BATCH-OK",
    )
    unusable = _fixture_document(
        "GEN_C 突发停运",
        "区域 TEST_NORTH，资产 GEN_C 今日突发停运。",
        "BATCH-BAD",
    )

    batch = ObviousNewsEventExtractor().extract_many((usable, unusable, usable))

    assert len(batch.results) == 3
    assert len(batch.events) == 2
    assert len(batch.quarantined) == 1
    assert batch.quarantined[0].document_version_id == unusable.document_version_id


def test_capacity_ignores_unqualified_numbers_and_prefers_the_body() -> None:
    """A nameplate figure and an outage figure look identical to a regex; only the qualifier separates them."""

    distractor = _fixture_document(
        "GEN_A 突发停运",
        "资产 GEN_A（总装机 1200 MW）在区域 TEST_NORTH 突发停运，不可用容量 500 MW。"
        "事件开始：2026-01-12T10:00:00Z。",
        "CAP-1",
    )
    headline_rounds = _fixture_document(
        "TEST_NORTH 进口能力减少 400 MW",
        "联络线 INTERCONNECTOR_NS 的进口能力减少，实际受影响 250 MW。事件开始：2026-01-20T09:00:00Z。",
        "CAP-2",
    )
    unqualified_only = _fixture_document(
        "GEN_A 突发停运",
        "区域 TEST_NORTH，资产 GEN_A 突发停运，全站装机 800 MW。事件开始：2026-01-12T10:00:00Z。",
        "CAP-3",
    )

    extractor = ObviousNewsEventExtractor()

    assert extractor.extract(distractor).events[0].capacity_mw == 500.0
    headline_event = extractor.extract(headline_rounds).events[0]
    assert headline_event.capacity_mw == 250.0
    assert next(span for span in headline_event.evidence if span.field_name == "capacity_mw").text_field == "body"
    assert extractor.extract(unqualified_only).events[0].capacity_mw is None


def test_multi_event_document_is_quarantined_instead_of_merged() -> None:
    """Two events in one document must not collapse into one record with both regions."""

    document = _fixture_document(
        "多事件通报",
        "区域 TEST_NORTH，资产 GEN_A 突发停运，不可用容量 500 MW；"
        "同时区域 TEST_SOUTH，资产 WIND_FLEET 的风电可用出力预计增加 700 MW。"
        "事件开始：2026-01-12T10:00:00Z。",
        "MULTI-1",
    )

    result = ObviousNewsEventExtractor().extract(document)

    assert result.events == ()
    assert result.quarantine is not None
    assert result.quarantine.reason_code == "ambiguous_multi_event"
    assert "generation_outage" in result.quarantine.message


def test_negated_long_horizon_wording_is_flagged_rather_than_misclassified() -> None:
    """The baseline cannot read negation, so overlapping keywords must surface, not silently pick one."""

    document = _fixture_document(
        "GEN_B 长期退役计划",
        "企业公告：区域 TEST_SOUTH，资产 GEN_B 计划在 2030 年关闭，本次公告不涉及任何非计划停运。"
        "事件开始：2030-01-01T00:00:00Z。",
        "NEG-1",
    )

    result = ObviousNewsEventExtractor().extract(document)

    assert result.quarantine is not None
    assert result.quarantine.reason_code == "ambiguous_multi_event"


def test_extraction_result_cannot_report_both_an_event_and_a_quarantine() -> None:
    """The event table and the quarantine list must stay mutually exclusive."""

    document = _fixture_document(
        "GEN_A 突发停运",
        "区域 TEST_NORTH，资产 GEN_A 突发停运，不可用容量 500 MW。事件开始：2026-01-12T10:00:00Z。",
        "EXCLUSIVE-1",
    )
    event = ObviousNewsEventExtractor().extract(document).events[0]
    quarantine = ExtractionQuarantine(
        document_version_id=document.document_version_id,
        reason_code="event_contract_violation",
        message="probe",
        extractor_id="probe",
        extractor_version="1.0.0",
    )

    with pytest.raises(ValidationError, match="either events or one quarantine"):
        EventExtractionResult(
            document_version_id=document.document_version_id,
            events=(event,),
            quarantine=quarantine,
        )
    with pytest.raises(ValidationError, match="either events or one quarantine"):
        EventExtractionResult(document_version_id=document.document_version_id)


def test_relative_day_and_clock_are_anchored_on_the_publication_date() -> None:
    """Bulletins date events against their own publication, which must resolve, not quarantine."""

    document = _fixture_document(
        "TEST_NORTH 次日需求激增",
        "区域 TEST_NORTH，资产 DEMAND_SYSTEM 的需求激增，预计增加 600 MW。生效时段：次日 14:00—19:00。",
        "REL-1",
    )

    event = ObviousNewsEventExtractor().extract(document).events[0]

    assert _iso_z(event.effective_start_at) == "2026-01-13T14:00:00Z"
    assert _iso_z(event.effective_end_at) == "2026-01-13T19:00:00Z"
    assert event.time_resolution.basis == "derived_from_publication"
    assert event.time_resolution.anchor_at == document.published_at
    assert event.time_resolution.market_timezone == "UTC"
    quotes = {span.field_name: span.quote for span in event.evidence}
    assert quotes["effective_start_at"] == "次日 14:00—19:00"


def test_market_timezone_decides_which_day_a_relative_expression_lands_on() -> None:
    """The market clock is an explicit setting, never an implicit default."""

    document = _fixture_document(
        "TEST_NORTH 次日需求激增",
        "区域 TEST_NORTH，资产 DEMAND_SYSTEM 的需求激增，预计增加 600 MW。生效时段：次日 14:00—19:00。",
        "REL-2",
    )

    utc_event = ObviousNewsEventExtractor(market_timezone="UTC").extract(document).events[0]
    shanghai_event = ObviousNewsEventExtractor(market_timezone="Asia/Shanghai").extract(document).events[0]

    assert _iso_z(utc_event.effective_start_at) == "2026-01-13T14:00:00Z"
    assert _iso_z(shanghai_event.effective_start_at) == "2026-01-13T06:00:00Z"
    assert shanghai_event.time_resolution.market_timezone == "Asia/Shanghai"

    with pytest.raises(ValueError, match="unknown market timezone"):
        ObviousNewsEventExtractor(market_timezone="Not/AZone")


def test_relative_window_that_reads_backwards_crosses_midnight() -> None:
    document = _fixture_document(
        "TEST_NORTH 次日需求激增",
        "区域 TEST_NORTH，资产 DEMAND_SYSTEM 的需求激增，预计增加 600 MW。生效时段：次日 22:00—02:00。",
        "REL-3",
    )

    event = ObviousNewsEventExtractor().extract(document).events[0]

    assert _iso_z(event.effective_start_at) == "2026-01-13T22:00:00Z"
    assert _iso_z(event.effective_end_at) == "2026-01-14T02:00:00Z"


def test_a_vague_time_period_is_quarantined_rather_than_invented() -> None:
    """"今日上午" has no defensible instant; picking one would be fabrication."""

    document = _fixture_document(
        "GEN_A 突发停运",
        "区域 TEST_NORTH，资产 GEN_A 今日上午突发停运，不可用容量 500 MW，预计傍晚恢复。",
        "REL-4",
    )

    result = ObviousNewsEventExtractor().extract(document)

    assert result.events == ()
    assert result.quarantine.reason_code == "missing_effective_start"


def test_several_relative_times_without_one_window_stay_ambiguous() -> None:
    document = _fixture_document(
        "GEN_A 突发停运",
        "区域 TEST_NORTH，资产 GEN_A 突发停运。通报于今日 09:00 发出，运维会议定在次日 15:00。",
        "REL-5",
    )

    result = ObviousNewsEventExtractor().extract(document)

    assert result.events == ()
    assert result.quarantine.reason_code == "ambiguous_event_time"


def test_a_derived_time_must_declare_its_anchor_and_market_clock() -> None:
    """A derived instant without its assumptions is indistinguishable from a stated one."""

    with pytest.raises(ValidationError, match="publication anchor and market timezone"):
        TimeResolution(basis="derived_from_publication")
    with pytest.raises(ValidationError, match="must not claim a derivation anchor"):
        TimeResolution(basis="stated_absolute", market_timezone="UTC")
