"""Independent, hand-labelled entity rules and their downstream consequences.

The 50 positive cases below are synthetic rule-validation samples, not a real-model
blind benchmark. Expected keys are literals and do not call the production normalizer.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from app.research.news import CollectedNewsRecord, EvidenceSpan, NewsNormalizer, StructuredNewsEventExtractor
from app.research.news.clock import MarketClock
from app.research.news.contracts import EventRecord
from app.research.news.entity_resolution import (
    EntityResolution,
    EntityResolutionError,
    entity_sources_are_valid,
    resolve_entities,
)
from app.research.news.features import build_event_features
from app.research.news.merging import merge_event_records
from app.research.news.versioning import NewsVersionError, NewsVersionStore


def document(body, tags=("GB",), title="Operational notice", source_id="independent", version=1, hour=0):
    return NewsNormalizer().normalize(
        CollectedNewsRecord(
            source_name="Independent rule samples",
            source_document_id=source_id,
            source_ref=f"fixture://entity-rules/{source_id}",
            title=title,
            body=body,
            published_at=datetime(2026, 1, 1, tzinfo=UTC),
            collected_at=datetime(2026, 1, 1, hour, tzinfo=UTC),
            version=version,
            market_tags=tags,
        )
    )


def resolve(doc, regions=(), assets=(), groups=()):
    spans = [
        EvidenceSpan(
            field_name=field,
            document_version_id=doc.document_version_id,
            text_field="body",
            start_char=0,
            end_char=len(doc.body),
            quote=doc.body,
        )
        for field in ("affected_regions", "affected_assets", "asset_groups")
    ]
    return resolve_entities(doc, regions, assets, groups, spans)


@pytest.mark.parametrize(
    "text,key",
    [
        ("Britain", "GB"),
        ("Great Britain", "GB"),
        ("Queensland", "NEM-QLD"),
        ("QLD", "NEM-QLD"),
        ("辽宁电网", "CN-LIAONING"),
        ("江苏", "CN-JIANGSU"),
        ("广东省", "CN-GUANGDONG"),
        ("安徽", "CN-ANHUI"),
        ("山东电网", "CN-SHANDONG"),
        ("福建", "CN-FUJIAN"),
    ],
)
def test_ten_explicit_regions(text, key):
    doc = document(f"{text} reported an operational event.", tags=(key,))
    result = resolve(doc, regions=(text,))
    assert result.region_keys == (key,)
    assert result.mentioned_regions[0].value == text
    assert result.raw_entity_mentions[0].source_type == "source_text"
    assert entity_sources_are_valid(result, {doc.document_version_id: doc})


@pytest.mark.parametrize(
    "tag,body",
    [
        ("NEM-QLD", "Both generating units tripped."),
        ("GB", "Demand increased during the evening."),
        ("ERCOT", "A generator returned to service."),
        ("CN-JIANGSU", "1号机组发生跳闸。"),
        ("CN-LIAONING", "电网最大负荷再创新高。"),
        ("CN-ANHUI", "高温推动用电需求。"),
        ("CN-GUANGDONG", "输电线路检修已结束。"),
        ("CN-SHANDONG", "风电出力下降。"),
        ("CN-FUJIAN", "燃料供应恢复。"),
        ("CN-ZHEJIANG", "机组进入计划检修。"),
    ],
)
def test_ten_tag_only_regions(tag, body):
    doc = document(body, tags=(tag,))
    # Even when a model copies the tag into its region list, it cannot become text evidence.
    result = resolve(doc, regions=(tag,))
    assert result.mentioned_regions == result.raw_entity_mentions == ()
    assert result.market_keys == (tag,)
    assert result.market_scope[0].source_ref == "market_tags[0]"
    assert result.market_scope[0].source_type == "market_tag"


@pytest.mark.parametrize(
    "asset,key",
    [
        ("Alpha Power Station Unit 1", "alphapowerstationunit1"),
        ("Beta Plant Unit 2", "betaplantunit2"),
        ("Gamma Power Plant", "gammapowerplant"),
        ("Delta Station Unit 10", "deltastationunit10"),
        ("Echo Power Station Generating Unit 3", "echopowerstationunit3"),
        ("国信沙洲电厂1号机组", "国信沙洲电厂1号机组"),
        ("常熟电厂6号机组", "常熟电厂6号机组"),
        ("太平岭核电厂2号机组", "太平岭核电厂2号机组"),
        ("甲乙500kV线路", "甲乙500kv线路"),
        ("海风电站", "海风电站"),
    ],
)
def test_ten_specific_assets(asset, key):
    result = resolve(document(f"{asset} is unavailable."), assets=(asset,))
    assert result.asset_keys == (key,)
    assert result.asset_groups == ()


@pytest.mark.parametrize(
    "group,key",
    [
        ("AGRs", "AGR"),
        ("7 AGRs", "AGR"),
        ("advanced gas cooled reactors", "AGR"),
        ("advanced gas cooled reactor stations", "AGR"),
        ("coal fleet", "coal_fleet"),
        ("coal plants", "coal_fleet"),
        ("wind generation", "wind_generation"),
        ("wind farms", "wind_generation"),
        ("solar generation", "solar_generation"),
        ("燃煤机组", "coal_fleet"),
    ],
)
def test_ten_groups_cannot_be_named_assets(group, key):
    doc = document(f"{group} are affected.")
    wrong_column = resolve(doc, assets=(group,))
    correct_column = resolve(doc, groups=(group,))
    assert wrong_column.asset_keys == correct_column.asset_keys == ()
    assert wrong_column.group_keys == correct_column.group_keys == (key,)
    assert wrong_column.raw_entity_mentions[0].quote == group


@pytest.mark.parametrize(
    "phrase,expected",
    [
        ("Alpha Power Station Units 1 and 2", ("alphapowerstationunit1", "alphapowerstationunit2")),
        ("Beta Plant Units 2 and 4", ("betaplantunit2", "betaplantunit4")),
        ("Gamma Station Generating Units 1 and 3", ("gammastationunit1", "gammastationunit3")),
        (
            "Delta Power Plant Units 1, 2 and 3",
            ("deltapowerplantunit1", "deltapowerplantunit2", "deltapowerplantunit3"),
        ),
        ("Echo Plant Units 1, 2, and 3", ("echoplantunit1", "echoplantunit2", "echoplantunit3")),
        ("Foxtrot Plant Units 10 and 11", ("foxtrotplantunit10", "foxtrotplantunit11")),
        ("松林电厂1、2号机组", ("松林电厂1号机组", "松林电厂2号机组")),
        ("江口电站2和3号机组", ("江口电站2号机组", "江口电站3号机组")),
        ("长河电厂1，3号机组", ("长河电厂1号机组", "长河电厂3号机组")),
        ("海湾核电厂1及2号机组", ("海湾核电厂1号机组", "海湾核电厂2号机组")),
    ],
)
def test_ten_enumerations_have_literal_expected_keys_and_common_backlinks(phrase, expected):
    doc = document(f"{phrase} tripped.")
    results = [resolve(doc, assets=(phrase,)) for _ in range(3)]
    assert all(result.asset_keys == expected for result in results)
    result = results[0]
    assert len(result.raw_entity_mentions) == 1
    assert result.raw_entity_mentions[0].quote == phrase
    assert all(
        item.source_type == "rule_derived"
        and item.rule_id == "explicit_unit_list"
        and item.mention_ids == (result.raw_entity_mentions[0].mention_id,)
        for item in result.affected_assets
    )
    # Already-expanded outputs normalize to the same keys without adding omitted siblings.
    separate = resolve(doc, assets=tuple(item.value for item in result.affected_assets))
    assert separate.asset_keys == expected
    assert len(resolve(doc, assets=(result.affected_assets[0].value,)).asset_keys) == 1


@pytest.mark.parametrize(
    "phrase",
    [
        "Alpha Plant Units 1-2",
        "Alpha Plant Units 1 or 2",
        "Units 1 and 2",
        "Alpha Plant Units 1/2",
        "Alpha Plant Units 1 and 1",
        "7 reactors",
        "海湾电厂1至3号机组",
        "海湾电厂1或2号机组",
        "海湾电厂1、1号机组",
        "Power Station Units 1 and 2",
        "该电厂1、2号机组",
        "Unit 1",
        "1号机组",
    ],
)
def test_ambiguous_collections_never_silently_become_specific_assets(phrase):
    with pytest.raises(EntityResolutionError):
        resolve(document(f"{phrase} may be unavailable."), assets=(phrase,))


@pytest.mark.parametrize(
    "invented,source",
    [
        ("Alpha Plant Unit 1", "Alpha Plant Unit 10 tripped."),
        ("Alpha Plant Unit 3", "Alpha Plant Units 1 and 2 tripped."),
        ("Beta Plant Unit 1", "Alpha Plant Units 1 and 2 tripped."),
        ("Alpha Plant Unit 1", "Alpha Plant Units 1 and 2 or 3 may trip."),
        ("海湾电厂3号机组", "海湾电厂1、2号机组跳闸，3日开始检修。"),
    ],
)
def test_tokens_elsewhere_in_a_quote_cannot_validate_an_invented_asset(invented, source):
    with pytest.raises(EntityResolutionError):
        resolve(document(source), assets=(invented,))


def candidate(doc, assets=(), regions=(), groups=(), short=False):
    fields = [
        "relevance",
        "event_type",
        "status",
        "physical_effect",
        "affected_regions",
        "affected_assets",
        "asset_groups",
    ]
    if short:
        fields += ["effective_start_at", "magnitude"]
    return {
        "relevance": "short_term" if short else "long_horizon",
        "event_type": "generation_outage" if short else "policy_long_horizon",
        "status": "occurred" if short else "planned",
        "physical_effect": "supply_down",
        "affected_regions": list(regions),
        "affected_assets": list(assets),
        "asset_groups": list(groups),
        "confidence": 0.99,
        "time_precision": "instant" if short else "vague",
        "time_text": "2026-01-01T01:00:00Z" if short else "this decade",
        "event_instant": {"iso": "2026-01-01T01:00:00+00:00", "basis": "stated_absolute"} if short else None,
        "quantity": {
            "value": 870,
            "unit": "MW",
            "semantic": "generation_loss",
            "direction": "decrease",
            "raw_text": "870 MW",
        }
        if short
        else None,
        "evidence": [
            {"field_name": field, "text_field": text_field, "quote": getattr(doc, text_field)}
            for field in fields
            for text_field in ("title", "body")
        ],
    }


class Gateway:
    model_name = "entity-rules-scripted"

    def __init__(self, candidates):
        self.candidates = iter(candidates)

    def invoke_structured(self, *, messages, schema):
        return schema.model_validate({"disposition": "event", "events": [next(self.candidates)]})


def extract(doc, candidates):
    return StructuredNewsEventExtractor(
        Gateway(candidates), market_timezone="UTC", extraction_passes=len(candidates)
    ).extract(doc)


def test_three_different_output_shapes_agree_and_count_capacity_once():
    phrase = "Independent Power Station Generating Units 1 and 2"
    doc = document(f"{phrase} tripped at 2026-01-01T01:00:00Z, losing 870 MW.", tags=("NEM-QLD",))
    expanded = ("Independent Power Station Unit 1", "Independent Power Station Unit 2")
    variants = [
        candidate(doc, assets=(phrase,), short=True),
        candidate(doc, assets=expanded, short=True),
        candidate(doc, assets=tuple(reversed(expanded)), regions=("QLD",), short=True),
    ]
    independent = [extract(doc, [variant]).events[0] for variant in variants]
    assert len({event.event_id for event in independent}) == 1
    consensus = extract(doc, variants)
    assert consensus.quarantine is None
    event = consensus.events[0]
    assert event.mentioned_regions == ()
    assert not any(span.field_name == "affected_regions" for span in event.evidence)
    cutoff = datetime(2026, 1, 1, 2, tzinfo=UTC)
    merged = merge_event_records([(event, doc)], as_of=cutoff)
    assert len(merged) == 1 and len(merged[0].asset_keys) == 2
    snapshot = build_event_features(
        merged, clock=MarketClock("NEM-QLD"), start_at=event.effective_start_at, end_at=cutoff, as_of=cutoff
    )
    assert all(row.active_event_count == 1 and row.active_capacity_mw == 870 for row in snapshot.rows)
    assert merged[0].state_history[0].entity_resolution == event.entity_resolution


def test_unnamed_long_term_events_do_not_globally_collapse_and_revisions_keep_their_id():
    docs = [
        document("Britain's 7 AGRs will retire this decade.", source_id=source, version=version, hour=hour)
        for source, version, hour in (("first", 1, 0), ("second", 1, 1), ("first", 2, 2))
    ]
    records = [(extract(doc, [candidate(doc, assets=("AGRs",), regions=("Britain",))]).events[0], doc) for doc in docs]
    assert records[0][0].event_id != records[1][0].event_id
    assert records[0][0].event_id == records[2][0].event_id
    assert records[0][0].asset_keys == () and records[0][0].group_keys == ("AGR",)
    assert any(m.quote == "7 AGRs" for m in records[0][0].raw_entity_mentions)
    early = merge_event_records(records, as_of=docs[0].available_at)
    late = merge_event_records(records, as_of=docs[-1].available_at)
    assert len(early) == 1 and len(late) == 2
    revised = next(event for event in late if event.event_id == early[0].event_id)
    assert revised.revision_count == 2
    assert revised.state_history[0].entity_resolution == early[0].entity_resolution
    assert all(
        entity_sources_are_valid(event.entity_resolution, {d.document_version_id: d for d in docs}) for event in late
    )


@pytest.mark.parametrize(
    "tags,region,target,allowed",
    [
        (("GB",), "", "GB", True),
        (("GB", "NEM-QLD"), "", "GB", False),
        (("GB",), "Queensland", "GB", False),
        (("GB", "NEM-QLD"), "Britain", "GB", True),
        (("GB", "NEM-QLD"), "Britain", "NEM-QLD", False),
    ],
)
def test_market_metadata_is_a_candidate_scope_not_an_invented_affected_region(tags, region, target, allowed):
    doc = document(f"{region} reported a trip.", tags=tags)
    result = resolve(doc, regions=(region,) if region else ())
    assert result.matches_market(target) is allowed
    assert result.market_keys == tuple(sorted(tags))


def test_legacy_serialized_events_keep_their_original_identity_and_keys():
    doc = document("Britain's AGRs will retire this decade.")
    event = EventRecord(
        event_id="evt_" + "a" * 24,
        document_version_id=doc.document_version_id,
        relevance="long_horizon",
        event_type="policy_long_horizon",
        affected_regions=("Britain",),
        affected_assets=("AGRs",),
        announcement_available_at=doc.available_at,
        extractor_id="legacy",
        extractor_version="1.2.0",
    )
    payload = event.model_dump(mode="json", exclude={"entity_resolution"})
    restored = EventRecord.model_validate_json(json.dumps(payload))
    assert restored.entity_resolution is None
    assert restored.event_id == event.event_id and restored.asset_keys == ("agrs",)
    merged = merge_event_records([(restored, doc)], as_of=doc.available_at)
    assert merged[0].event_id == event.event_id and merged[0].asset_keys == ("agrs",)


def test_saved_resolution_survives_roundtrip_and_detects_false_provenance():
    doc = document("Britain's AGRs will retire.")
    result = resolve(doc, regions=("Britain",), groups=("AGRs",))
    saved = EntityResolution.model_validate_json(result.model_dump_json())
    assert saved == result
    false_scope = saved.market_scope[0].model_copy(update={"raw_value": "NEM-QLD"})
    forged = saved.model_copy(update={"market_scope": (false_scope,)})
    assert not entity_sources_are_valid(forged, {doc.document_version_id: doc})


def test_true_quote_cannot_support_a_forged_derived_asset():
    doc = document("Alpha Plant Units 1 and 2 tripped.")
    result = resolve(doc, assets=("Alpha Plant Units 1 and 2",))
    invented = result.affected_assets[0].model_copy(update={"value": "Alpha Plant Unit 3", "key": "alphaplantunit3"})
    forged = result.model_copy(update={"affected_assets": (invented,)})
    assert not entity_sources_are_valid(forged, {doc.document_version_id: doc})


def test_identical_document_version_cannot_silently_change_its_market_provenance():
    first = document("Both units tripped.", tags=("GB",))
    conflicting = document("Both units tripped.", tags=("NEM-QLD",), hour=1)
    store = NewsVersionStore([first])
    with pytest.raises(NewsVersionError, match="市场标签"):
        store.add(conflicting)
    assert store.version_count == 1
    correction = document("Both units tripped.", tags=("NEM-QLD",), version=2, hour=1)
    assert store.add(correction)
    assert store.visible_at(first.available_at) == (first,)


def test_named_units_cannot_be_hidden_in_the_group_column():
    doc = document("Alpha Plant Units 1 and 2 tripped.")
    misplaced = resolve(doc, groups=("Alpha Plant Units 1 and 2",))
    assert misplaced.asset_keys == ("alphaplantunit1", "alphaplantunit2")
    assert misplaced.group_keys == ()


def test_quarantined_enumeration_keeps_its_validated_source_quote():
    doc = document("Alpha Plant Units 1 or 2 may retire this decade.")
    result = extract(doc, [candidate(doc, assets=("Alpha Plant Units 1 or 2",))])
    assert result.quarantine is not None
    assert result.quarantine.reason_code == "invalid_evidence"
    assert any(mention.quote == doc.body for mention in result.quarantine.raw_entity_mentions)


def test_explicit_group_withdrawal_preserves_the_old_state_without_resurrecting_the_group():
    old = document("Britain's AGRs will retire this decade.")
    new = document("Britain withdrew the previously stated reactor group this decade.", version=2, hour=1)
    first = extract(old, [candidate(old, regions=("Britain",), groups=("AGRs",))]).events[0]
    payload = candidate(new, regions=("Britain",))
    payload["cleared_fields"] = ["asset_groups"]
    second = extract(new, [payload]).events[0]
    early = merge_event_records([(first, old), (second, new)], as_of=old.available_at)
    late = merge_event_records([(first, old), (second, new)], as_of=new.available_at)
    assert len(early) == len(late) == 1
    assert early[0].event_id == late[0].event_id
    assert early[0].group_keys == ("AGR",) and late[0].group_keys == ()
    assert late[0].state_history[0].group_keys == ("AGR",)
    assert late[0].state_history[-1].group_keys == ()
