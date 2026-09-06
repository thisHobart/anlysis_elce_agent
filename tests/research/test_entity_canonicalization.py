"""Canonicalization must absorb wording drift without merging genuinely different entities."""

from __future__ import annotations

from datetime import UTC, datetime

from app.research.news import (
    AsOfEventAssembler,
    CollectedNewsRecord,
    NewsNormalizer,
    NewsVersionStore,
    StructuredNewsEventExtractor,
)
from app.research.news.entities import (
    canonical_asset,
    canonical_assets,
    canonical_region,
    canonical_regions,
    entity_identity,
    is_region_mention,
    split_entity_keys,
)

BODY = "2026年8月3日11时06分，辽宁电网最大用电负荷冲至4775万千瓦。"
TIME_TEXT = "2026年8月3日11时06分"


def test_the_observed_run_to_run_region_wobble_folds_to_one_key() -> None:
    """The holdout run alternated between 辽宁 and 辽宁电网 for the same grid."""

    keys = {canonical_region(value) for value in ("辽宁", "辽宁电网", "辽宁省", "辽宁电网公司")}
    assert keys == {"CN-LIAONING"}


def test_regions_that_are_actually_different_keep_different_keys() -> None:
    assert canonical_region("辽宁电网") != canonical_region("山东电网")
    assert canonical_region("华东") != canonical_region("华北")


def test_english_region_aliases_fold_together() -> None:
    """The development set failed only because the model wrote Britain, not Great Britain."""

    assert canonical_region("Great Britain") == canonical_region("Britain") == "GB"
    assert canonical_region("Queensland") == canonical_region("QLD") == "NEM-QLD"


def test_an_unlisted_region_still_folds_deterministically() -> None:
    """Stability must not depend on the alias table being complete."""

    assert canonical_region("新疆电网") == canonical_region("新疆") == canonical_region("新疆自治区")
    assert canonical_region("新疆") not in {"", "CN-LIAONING"}


def test_order_and_duplicates_do_not_change_the_identity() -> None:
    first = canonical_regions(("辽宁电网", "山东"))
    second = canonical_regions(("山东电网", "辽宁", "辽宁省"))
    assert first == second


def test_assets_keep_the_words_that_tell_units_apart() -> None:
    assert canonical_asset("国信沙洲电厂1号机组") != canonical_asset("国信沙洲电厂2号机组")


def test_assets_tolerate_spacing_case_and_fullwidth_digits() -> None:
    assert canonical_asset("Millmerran Power Station Unit 1") == canonical_asset(
        "millmerran power station  unit1"
    )
    assert canonical_asset("１号机组") == canonical_asset("1号机组")


def test_blank_and_punctuation_only_mentions_are_dropped() -> None:
    assert canonical_regions(("", "  ", "、")) == ()
    assert canonical_assets(("",)) == ()


def test_entity_identity_exposes_both_halves_in_canonical_form() -> None:
    identity = entity_identity(("辽宁电网",), ("国信沙洲电厂1号机组",))
    assert identity["affected_regions"] == ("CN-LIAONING",)
    assert identity["affected_assets"] == (canonical_asset("国信沙洲电厂1号机组"),)


def test_a_grid_name_counts_as_a_region_wherever_the_model_filed_it() -> None:
    """The holdout run put 天津电网 under assets on one pass and left assets empty on the next."""

    filed_as_asset = split_entity_keys(("天津电网",), ("天津电网",))
    filed_only_as_region = split_entity_keys(("天津电网",), ())

    assert filed_as_asset == filed_only_as_region == (("CN-TIANJIN",), ())


def test_a_name_the_vocabulary_does_not_know_stays_an_asset() -> None:
    """Reclassification is driven by the controlled vocabulary, not by guessing."""

    assert not is_region_mention("国信沙洲电厂6号机组")
    regions, assets = split_entity_keys(("江苏",), ("国信沙洲电厂6号机组",))
    assert regions == ("CN-JIANGSU",)
    assert assets == (canonical_asset("国信沙洲电厂6号机组"),)


def test_a_region_named_only_in_the_asset_slot_is_still_found() -> None:
    regions, assets = split_entity_keys((), ("辽宁电网",))
    assert regions == ("CN-LIAONING",)
    assert assets == ()


def _document(source_id: str = "ENTITY-DRIFT-01", hour: int = 6):
    record = CollectedNewsRecord(
        source_name="测试来源",
        source_document_id=source_id,
        source_ref="https://example.invalid/load-record",
        title="用电负荷创历史新高",
        body=BODY,
        published_at=datetime(2026, 8, 6, hour, 7, tzinfo=UTC),
        collected_at=datetime(2026, 8, 6, hour + 1, 0, tzinfo=UTC),
        language="zh-CN",
        market_tags=("CN-LIAONING",),
        metadata={"published_time_precision": "minute"},
    )
    return NewsNormalizer().normalize(record)


def _payload(region: str) -> dict:
    fields = ("relevance", "event_type", "status", "physical_effect", "affected_regions")
    evidence = [
        {"field_name": name, "text_field": "body", "quote": BODY} for name in fields
    ]
    evidence.append({"field_name": "effective_start_at", "text_field": "body", "quote": BODY})
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
                "time_text": TIME_TEXT,
                "event_instant": {"iso": "2026-08-03T11:06:00+08:00", "basis": "stated_absolute"},
                "event_end_instant": None,
                "confidence": 0.95,
                "evidence": evidence,
            }
        ],
    }


class _OneShotGateway:
    model_name = "entity-drift-model"

    def __init__(self, region: str) -> None:
        self.region = region

    def invoke_structured(self, *, messages, schema):
        return schema.model_validate(_payload(self.region))


def _extract(region: str):
    extractor = StructuredNewsEventExtractor(
        _OneShotGateway(region),
        market_timezone="Asia/Shanghai",
    )
    result = extractor.extract(_document())
    assert result.quarantine is None, result.quarantine
    return result.events[0]


def test_the_same_grid_written_two_ways_produces_one_event_identity() -> None:
    """This is the drift that broke deduplication: 辽宁 on one run, 辽宁电网 on the next."""

    short_form = _extract("辽宁")
    long_form = _extract("辽宁电网")

    assert short_form.event_id == long_form.event_id
    assert short_form.region_keys == long_form.region_keys == ("CN-LIAONING",)


def test_the_source_wording_is_still_preserved_for_evidence() -> None:
    """Canonicalization is for identity only; the record must still quote what was written."""

    assert _extract("辽宁电网").affected_regions == ("辽宁电网",)
    assert _extract("辽宁").affected_regions == ("辽宁",)


def test_a_region_absent_from_both_source_and_market_metadata_is_rejected() -> None:
    extractor = StructuredNewsEventExtractor(
        _OneShotGateway("华东"),
        market_timezone="Asia/Shanghai",
    )

    result = extractor.extract(_document())

    assert result.quarantine is not None
    assert result.quarantine.reason_code == "invalid_evidence"


class _PerDocumentGateway:
    """Answer with a different region spelling per source document, as reposts really do."""

    model_name = "repost-model"

    def __init__(self, regions: dict[str, str]) -> None:
        self.regions = regions

    def invoke_structured(self, *, messages, schema):
        import json as _json

        for message in messages:
            try:
                payload = _json.loads(message.content)
            except (ValueError, TypeError):
                continue
            if isinstance(payload, dict) and "source_document_id" in payload:
                return schema.model_validate(_payload(self.regions[payload["source_document_id"]]))
        raise AssertionError("no document payload message found")


def test_two_reposts_spelling_the_region_differently_merge_into_one_event() -> None:
    """The downstream payoff: without canonical identity these stay two separate events."""

    first = _document(source_id="REPOST-A", hour=6)
    second = _document(source_id="REPOST-B", hour=8)
    extractor = StructuredNewsEventExtractor(
        _PerDocumentGateway({"REPOST-A": "辽宁", "REPOST-B": "辽宁电网"}),
        market_timezone="Asia/Shanghai",
    )

    view = AsOfEventAssembler(
        NewsVersionStore((first, second)),
        extractor=extractor,
    ).view_at(datetime(2026, 8, 10, tzinfo=UTC))

    assert len(view.events) == 1
    assert view.events[0].revision_count == 2
    assert view.events[0].region_keys == ("CN-LIAONING",)


def test_a_two_character_name_is_not_stripped_down_to_one() -> None:
    """沙市 is a place, not 沙 plus a suffix; stripping it would collide with any other 沙*."""

    assert canonical_region("沙市") == canonical_region("沙市")
    assert canonical_region("沙市") != canonical_region("沙")
    assert canonical_region("天津市") == "CN-TIANJIN"
