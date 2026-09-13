"""Regression tests for realistic P2 news semantics found during the 2026-09 review."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.research.news import (
    CollectedNewsRecord,
    EventExtractionResult,
    EventRecord,
    MarketClock,
    NewsNormalizer,
    NewsVersionStore,
    StructuredNewsEventExtractor,
    TimeResolution,
    build_event_features,
    merge_event_records,
)
from app.research.news.entity_resolution import _asset_source, resolve_entities
from app.research.news.model_extraction import (
    _asset_signature,
    _event_signature,
)


def _at(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 9, 1, hour, minute, tzinfo=UTC)


def _document(
    *,
    source_id: str = "REALISM-1",
    version: int = 1,
    body: str,
    collected_at: datetime | None = None,
    source_tier: str | None = None,
):
    return NewsNormalizer().normalize(
        CollectedNewsRecord(
            source_name="test",
            source_document_id=source_id,
            source_ref=f"https://example.invalid/{source_id}",
            version=version,
            title="Plant A outage",
            body=body,
            published_at=_at(10),
            collected_at=collected_at or _at(10),
            updated_at=collected_at if version > 1 else None,
            market_tags=("TEST",),
            metadata={"source_tier": source_tier} if source_tier else {},
        )
    )


BODY = (
    "Plant A in TEST tripped at 2026-09-01T10:00:00Z, losing 870 MW. "
    "Plant B was separately mentioned for 2026-09-01. "
    "The company was incorporated on 2000-01-01."
)


def _candidate(**changes):
    values = {
        "relevance": "short_term",
        "event_type": "generation_outage",
        "status": "occurred",
        "physical_effect": "supply_down",
        "affected_regions": ["TEST"],
        "affected_assets": ["Plant A"],
        "quantity": {
            "value": 870,
            "unit": "MW",
            "semantic": "generation_loss",
            "direction": "decrease",
            "raw_text": "870 MW",
        },
        "time_precision": "instant",
        "time_text": "2026-09-01T10:00:00Z",
        "event_instant": {
            "iso": "2026-09-01T10:00:00Z",
            "basis": "stated_absolute",
        },
        "event_end_instant": None,
        "confidence": 0.99,
    }
    values.update(changes)
    fields = ["relevance", "event_type", "status", "physical_effect"]
    if values["affected_regions"]:
        fields.append("affected_regions")
    if values["affected_assets"]:
        fields.append("affected_assets")
    if values["quantity"] is not None:
        fields.append("magnitude")
    if values["event_instant"] is not None:
        fields.append("effective_start_at")
    values["evidence"] = [
        {"field_name": field, "text_field": "body", "quote": BODY}
        for field in fields
    ]
    return values


class _Gateway:
    model_name = "realism-test-model"

    def __init__(self, payload):
        self.payload = payload

    def invoke_structured(self, *, messages, schema):
        return schema.model_validate(self.payload)


def _extract(*candidates):
    extractor = StructuredNewsEventExtractor(
        _Gateway({"disposition": "event", "events": list(candidates)}),
        market_timezone="UTC",
        extraction_passes=1,
    )
    return extractor.extract(_document(body=BODY))


def test_background_date_does_not_quarantine_a_valid_event() -> None:
    result = _extract(_candidate())

    assert result.quarantine is None
    assert result.candidate_quarantines == ()
    assert len(result.events) == 1


@pytest.mark.parametrize(
    "changes, reason",
    [
        ({"affected_assets": ["Invented Plant"]}, "invalid_evidence"),
        ({"affected_assets": ["Invented Power Station Unit 1"]}, "invalid_evidence"),
        (
            {
                "quantity": {
                    "value": 900,
                    "unit": "GW",
                    "semantic": "generation_loss",
                    "direction": "decrease",
                    "raw_text": "870 MW",
                }
            },
            "ambiguous_quantity",
        ),
        (
            {
                "event_instant": {
                    "iso": "2026-09-01T11:00:00Z",
                    "basis": "stated_absolute",
                }
            },
            "ambiguous_event_time",
        ),
    ],
)
def test_values_that_are_not_supported_by_the_source_are_rejected(changes, reason) -> None:
    result = _extract(_candidate(**changes))

    assert result.events == ()
    assert result.quarantine is not None
    assert result.quarantine.reason_code == reason


def test_compact_chinese_asset_mentions_do_not_authorize_an_invented_plant_name() -> None:
    quote = "太平岭核电厂1、2号机组因500kV输电线路故障进入厂用电运行工况。"

    assert _asset_source("太平岭核电厂1号机组", quote)
    assert _asset_source("太平岭核电厂2号机组", quote)
    assert not _asset_source("太平峪核电厂1号机组", quote)


def test_coded_region_may_add_a_chinese_grid_suffix_without_changing_identity() -> None:
    from app.research.news.contracts import EvidenceSpan

    doc = _document(body="区域 TEST_NORTH 电力供需紧张")
    span = EvidenceSpan(field_name="affected_regions", document_version_id=doc.document_version_id,
                        text_field="body", start_char=0, end_char=len(doc.body), quote=doc.body)
    resolution = resolve_entities(doc, ("TEST_NORTH电网",), (), (), [span])
    assert resolution.region_keys == ("test_north",)
    assert resolution.mentioned_regions[0].value == "TEST_NORTH"


def test_one_bad_candidate_does_not_discard_a_valid_sibling() -> None:
    vague = _candidate(
        affected_assets=["Plant B"],
        quantity=None,
        time_precision="day",
        time_text="2026-09-02",  # Fabricated evidence, not merely an imprecise real date.
        event_instant=None,
    )
    result = _extract(_candidate(), vague)

    assert result.quarantine is None
    assert len(result.events) == 1
    assert result.candidate_quarantines[0].reason_code == "invalid_evidence"


def test_identical_recollection_is_a_duplicate_even_when_collected_later() -> None:
    first = _document(body=BODY, collected_at=_at(10))
    later = _document(body=BODY, collected_at=_at(11))
    store = NewsVersionStore((first, later))

    assert first.document_version_id == later.document_version_id
    assert store.version_count == 1
    assert store.duplicates.duplicate_count == 1
    assert store.visible_at(_at(10, 30))[0].first_seen_at == _at(10)


def _event(
    document,
    suffix: int,
    *,
    start: int = 10,
    capacity: float | None = 500,
    status: str = "occurred",
    event_type: str = "generation_outage",
    asset: str = "Plant A",
    cleared_fields=(),
):
    return EventRecord(
        event_id=f"evt_{suffix:024x}",
        document_version_id=document.document_version_id,
        relevance="short_term",
        event_type=event_type,
        affected_regions=("TEST",),
        affected_assets=(asset,),
        capacity_mw=capacity,
        announcement_available_at=document.available_at,
        effective_start_at=_at(start),
        time_resolution=TimeResolution(basis="stated_absolute"),
        direction="down" if event_type == "generation_restore" else "up",
        status=status,
        physical_effect="supply_up" if event_type == "generation_restore" else "supply_down",
        extractor_id="test",
        extractor_version="1.0",
        cleared_fields=cleared_fields,
    )


def _feature_rows(events):
    return {
        row.interval_start: row
        for row in build_event_features(
            events,
            clock=MarketClock(market="TEST"),
            start_at=_at(10),
            end_at=_at(13),
            as_of=_at(13),
        ).rows
    }


def test_late_correction_does_not_rewrite_earlier_feature_rows() -> None:
    first_doc = _document(body="initial", collected_at=_at(10))
    corrected_doc = _document(body="corrected", version=2, collected_at=_at(12))
    event = merge_event_records(
        [(_event(first_doc, 1, capacity=500), first_doc), (_event(corrected_doc, 1, capacity=350), corrected_doc)],
        as_of=_at(13),
    )[0]
    rows = _feature_rows((event,))

    assert rows[_at(10)].active_capacity_mw == 500
    assert rows[_at(12)].active_capacity_mw == 350


def test_explicit_field_withdrawal_does_not_resurrect_the_old_value() -> None:
    first_doc = _document(body="initial", collected_at=_at(10))
    corrected_doc = _document(body="withdrawn", version=2, collected_at=_at(12))
    event = merge_event_records(
        [
            (_event(first_doc, 1, capacity=500), first_doc),
            (
                _event(
                    corrected_doc,
                    1,
                    capacity=None,
                    cleared_fields=("capacity_mw", "magnitude"),
                ),
                corrected_doc,
            ),
        ],
        as_of=_at(13),
    )[0]
    rows = _feature_rows((event,))

    assert event.capacity_mw is None
    assert rows[_at(10)].active_capacity_mw == 500
    assert rows[_at(12)].active_capacity_mw is None


def test_cancelled_and_unknown_capacity_have_distinct_feature_meanings() -> None:
    document = _document(body="event", collected_at=_at(10))
    cancelled = merge_event_records([(_event(document, 1, status="cancelled"), document)], as_of=_at(13))
    unknown = merge_event_records([(_event(document, 2, capacity=None), document)], as_of=_at(13))

    assert _feature_rows(cancelled)[_at(10)].active_capacity_mw == 0
    assert _feature_rows(cancelled)[_at(10)].active_event_count == 0
    assert _feature_rows(unknown)[_at(10)].active_capacity_mw is None
    assert _feature_rows(unknown)[_at(10)].active_event_count == 1
    assert _feature_rows(())[_at(10)].active_capacity_mw == 0


def test_restoration_closes_the_matching_open_outage() -> None:
    outage_doc = _document(source_id="OUTAGE", body="outage", collected_at=_at(10))
    restore_doc = _document(source_id="RESTORE", body="restore", collected_at=_at(11))
    events = merge_event_records(
        [
            (_event(outage_doc, 1), outage_doc),
            (_event(restore_doc, 2, start=11, event_type="generation_restore"), restore_doc),
        ],
        as_of=_at(13),
    )
    rows = _feature_rows(events)

    assert rows[_at(11)].event_type_counts["generation_outage"] == 0
    assert rows[_at(11, 30)].active_event_count == 0


def test_point_demand_and_dispatch_events_do_not_become_permanent_states() -> None:
    known_doc = _document(source_id="DEMAND", body="demand peak", collected_at=_at(10))
    late_doc = _document(source_id="DISPATCH", body="dispatch record", collected_at=_at(12))
    events = merge_event_records(
        [
            (_event(known_doc, 1, start=11, event_type="demand_shock"), known_doc),
            (_event(late_doc, 2, start=11, event_type="storage_dispatch"), late_doc),
        ],
        as_of=_at(13),
    )
    rows = _feature_rows(events)

    assert rows[_at(11)].event_type_counts["demand_shock"] == 1
    assert rows[_at(11, 30)].event_type_counts["demand_shock"] == 0
    assert all(row.event_type_counts["storage_dispatch"] == 0 for row in rows.values())
    assert rows[_at(12)].new_announcement_count == 1


def test_same_asset_sibling_events_survive_a_document_revision() -> None:
    first_doc = _document(body="initial", collected_at=_at(10))
    corrected_doc = _document(body="corrected", version=2, collected_at=_at(12))
    events = merge_event_records(
        [
            (_event(first_doc, 1, start=10), first_doc),
            (_event(first_doc, 2, start=12), first_doc),
            (_event(corrected_doc, 1, start=10), corrected_doc),
            (_event(corrected_doc, 2, start=12), corrected_doc),
        ],
        as_of=_at(13),
    )

    assert len(events) == 2


def test_consensus_fingerprint_keeps_each_asset_attached_to_its_event() -> None:
    document = _document(body="events", collected_at=_at(10))
    first = EventExtractionResult(
        document_version_id=document.document_version_id,
        events=(
            _event(document, 1, asset="Plant A"),
            _event(document, 2, start=11, asset="Plant B", event_type="generation_restore"),
        ),
    )
    swapped = EventExtractionResult(
        document_version_id=document.document_version_id,
        events=(
            _event(document, 3, asset="Plant B"),
            _event(document, 4, start=11, asset="Plant A", event_type="generation_restore"),
        ),
    )

    assert _event_signature(first) == _event_signature(swapped)
    assert _asset_signature(first) != _asset_signature(swapped)


def test_later_primary_source_outranks_an_earlier_secondary_report() -> None:
    secondary = _document(
        source_id="MEDIA",
        body="media report",
        collected_at=_at(10),
        source_tier="secondary_media",
    )
    primary = _document(
        source_id="OPERATOR",
        body="operator notice",
        collected_at=_at(11),
        source_tier="primary_operator",
    )
    event = merge_event_records(
        [
            (_event(secondary, 1, capacity=400), secondary),
            (_event(primary, 1, capacity=500), primary),
        ],
        as_of=_at(13),
    )[0]
    rows = _feature_rows((event,))

    assert event.capacity_mw == 500
    assert rows[_at(10)].active_capacity_mw == 400
    assert rows[_at(11)].active_capacity_mw == 500
