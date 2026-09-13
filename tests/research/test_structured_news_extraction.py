"""Structured model extraction is useful only after deterministic safety gates pass."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.llm.gateway import ModelGatewayError, ModelResponseError, ModelTransientError
from app.research.news import (
    AsOfEventAssembler,
    CollectedNewsRecord,
    JsonlCollectedNewsAdapter,
    NewsNormalizer,
    NewsVersionStore,
    StructuredNewsEventExtractor,
    benchmark_real_news_extractor,
    build_model_news_extraction_messages,
    collect_evidence_spans,
    load_real_news_gold,
)
from app.research.news.model_extraction import (
    MODEL_EXTRACTION_SCHEMA_VERSION,
    ModelNewsExtraction,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "news_realistic"
NEWS = FIXTURES / "source_news.jsonl"
GOLD = FIXTURES / "gold_manifest.yaml"


def test_model_schema_version_literal_matches_the_published_schema_version():
    result = ModelNewsExtraction.model_validate(
        {
            "schema_version": MODEL_EXTRACTION_SCHEMA_VERSION,
            "disposition": "uncertain",
            "uncertainty_reason": "test",
        }
    )

    assert result.schema_version == "1.4.0"


def _document_payload(messages) -> dict:
    """Find the document payload message; a repair retry appends other messages after it."""

    for message in messages:
        try:
            candidate = json.loads(message.content)
        except (ValueError, TypeError):
            continue
        if isinstance(candidate, dict) and "source_document_id" in candidate:
            return candidate
    raise AssertionError("no document payload message found")


class ScriptedGateway:
    """A native structured-output test double; it never parses prose or reads gold labels."""

    model_name = "scripted-news-model"

    def __init__(self, outputs: dict[str, dict]) -> None:
        self.outputs = outputs
        self.calls = 0
        self.last_messages = None

    def invoke_structured(self, *, messages, schema):
        self.calls += 1
        self.last_messages = messages
        payload = _document_payload(messages)
        return schema.model_validate(self.outputs[payload["source_document_id"]])


class FailingGateway:
    model_name = "failing-news-model"

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls = 0

    def invoke_structured(self, *, messages, schema):
        self.calls += 1
        raise self.error


class RepairGateway:
    """Fail the first call with a schema error, then honor the corrected retry."""

    model_name = "repair-news-model"

    def __init__(self, error: Exception, good_output: dict, *, always_fail: bool = False) -> None:
        self.error = error
        self.good_output = good_output
        self.always_fail = always_fail
        self.calls = 0
        self.received_error_feedback = False

    def invoke_structured(self, *, messages, schema):
        self.calls += 1
        if self.calls == 1 or self.always_fail:
            raise self.error
        self.received_error_feedback = any("校验错误" in message.content for message in messages)
        return schema.model_validate(self.good_output)


def _claims(text_field: str, quote: str, *field_names: str) -> list[dict]:
    return [
        {
            "field_name": field_name,
            "text_field": text_field,
            "quote": quote,
        }
        for field_name in field_names
    ]


def _candidate(
    *,
    event_type: str,
    relevance: str,
    status: str,
    physical_effect: str,
    text_field: str,
    quote: str,
    regions: list[str] | None = None,
    assets: list[str] | None = None,
    quantity: dict | None = None,
    effective_start_at: str | None = None,
    effective_end_at: str | None = None,
    time_basis: str | None = None,
    time_precision: str = "unknown",
    time_text: str | None = None,
    extra_evidence: list[dict] | None = None,
) -> dict:
    evidence = _claims(
        text_field,
        quote,
        "relevance",
        "event_type",
        "status",
        "physical_effect",
    )
    if regions:
        evidence.extend(_claims(text_field, quote, "affected_regions"))
    if assets:
        evidence.extend(_claims(text_field, quote, "affected_assets"))
    if quantity is not None:
        evidence.extend(_claims(text_field, quote, "magnitude"))
    if effective_start_at is not None:
        evidence.extend(_claims(text_field, quote, "effective_start_at"))
    if effective_end_at is not None:
        evidence.extend(_claims(text_field, quote, "effective_end_at"))
    evidence.extend(extra_evidence or [])
    # The model wire shape carries an instant only at instant/hour precision; the local code
    # builds the datetime. The helper keeps the old kwargs and adapts to that nested shape.
    event_instant = (
        {"iso": effective_start_at, "basis": time_basis} if effective_start_at is not None else None
    )
    event_end_instant = (
        {"iso": effective_end_at, "basis": time_basis} if effective_end_at is not None else None
    )
    return {
        "relevance": relevance,
        "event_type": event_type,
        "status": status,
        "physical_effect": physical_effect,
        "affected_regions": regions or [],
        "affected_assets": assets or [],
        "quantity": quantity,
        "event_instant": event_instant,
        "event_end_instant": event_end_instant,
        "time_precision": time_precision,
        "time_text": time_text,
        "confidence": 0.99,
        "evidence": evidence,
    }


def _real_source_outputs() -> dict[str, dict]:
    r01 = "5月24日，国信沙洲电厂1号机组因EH油系统漏油，机组跳闸。"
    r02_title = "太平岭核电厂1、2号机组因500kV输电线路故障进入厂用电运行工况运行事件"
    r02_body = "电网线路恢复后，1、2号机组重新并网，事件结束。"
    r03 = "7月10日，全国用电负荷达到15.18亿千瓦，较7月初上涨超1.5亿千瓦。"
    r04 = "新能源最大出力屡创新高，最高达1.71亿千瓦，单日最大发电量超18亿千瓦时。"
    r05 = "Britain's 7 AGRs are due to reach the end of their operational lives this decade."
    r06_title = "Trip of Millmerran Power Station Generating Units 1 and 2 on 23 December 2013"
    r06_body = "At 0856 hrs both generating units tripped, resulting in the loss of 870 MW of generation."
    r07 = (
        "Low wind, reduced gas generation, high demand, adverse interconnector flows and network "
        "constraints created a complex operational challenge."
    )
    r08 = "ERCOT set an all-time peak demand record of 85,464 MW on August 10, 2023."
    r09 = "光坡核电厂1号机组已于本轮高温前顺利并网。"
    r10 = "服务队在37个社区开展爱心电力活动，长期提供管家式服务。"

    return {
        "REAL-CN-JS-20260622": {
            "disposition": "event",
            "events": [
                _candidate(
                    event_type="generation_outage",
                    relevance="short_term",
                    status="occurred",
                    physical_effect="supply_down",
                    text_field="body",
                    quote=r01,
                    regions=["江苏"],
                    assets=["国信沙洲电厂1号机组"],
                    time_precision="day",
                    time_text="5月24日",
                )
            ],
        },
        "REAL-CN-NNSA-20260804": {
            "disposition": "event",
            "events": [
                _candidate(
                    event_type="transmission_constraint",
                    relevance="short_term",
                    status="occurred",
                    physical_effect="transfer_down",
                    text_field="title",
                    quote=r02_title,
                    regions=["广东"],
                    assets=["太平岭核电厂1号机组", "太平岭核电厂2号机组"],
                    time_precision="day",
                ),
                _candidate(
                    event_type="generation_restore",
                    relevance="short_term",
                    status="restored",
                    physical_effect="supply_up",
                    text_field="body",
                    quote=r02_body,
                    regions=["广东"],
                    assets=["太平岭核电厂1号机组", "太平岭核电厂2号机组"],
                    time_precision="day",
                    extra_evidence=_claims("title", r02_title, "affected_regions", "affected_assets"),
                ),
            ],
        },
        "REAL-CN-NEA-LOAD-20260710": {
            "disposition": "event",
            "events": [
                _candidate(
                    event_type="demand_shock",
                    relevance="short_term",
                    status="occurred",
                    physical_effect="demand_up",
                    text_field="body",
                    quote=r03,
                    regions=["全国"],
                    assets=["全国用电负荷"],
                    quantity={
                        "value": 1.5,
                        "unit": "亿千瓦",
                        "semantic": "demand_change",
                        "direction": "increase",
                        "raw_text": "1.5亿千瓦",
                    },
                    time_precision="day",
                    time_text="7月10日",
                )
            ],
        },
        "REAL-CN-EAST-RENEWABLE-20260806": {
            "disposition": "event",
            "events": [
                _candidate(
                    event_type="renewable_supply_change",
                    relevance="short_term",
                    status="occurred",
                    physical_effect="supply_up",
                    text_field="body",
                    quote=r04,
                    regions=["华东"],
                    assets=["华东新能源"],
                    quantity={
                        "value": 1.71,
                        "unit": "亿千瓦",
                        "semantic": "output_level",
                        "direction": "unknown",
                        "raw_text": "1.71亿千瓦",
                    },
                    time_precision="day",
                )
            ],
        },
        "REAL-GB-AGR-20210623": {
            "disposition": "event",
            "events": [
                _candidate(
                    event_type="policy_long_horizon",
                    relevance="long_horizon",
                    status="planned",
                    physical_effect="supply_down",
                    text_field="body",
                    quote=r05,
                    regions=["Great Britain"],
                    assets=["AGRs"],
                    time_precision="vague",
                    time_text="this decade",
                )
            ],
        },
        "REAL-AEMO-MILLMERRAN-20140326": {
            "disposition": "event",
            "events": [
                _candidate(
                    event_type="generation_outage",
                    relevance="short_term",
                    status="occurred",
                    physical_effect="supply_down",
                    text_field="body",
                    quote=r06_body,
                    regions=["QLD"],
                    assets=["Millmerran Power Station Unit 1", "Millmerran Power Station Unit 2"],
                    quantity={
                        "value": 870,
                        "unit": "MW",
                        "semantic": "generation_loss",
                        "direction": "decrease",
                        "raw_text": "870 MW",
                    },
                    effective_start_at="2013-12-23T08:56:00+10:00",
                    time_basis="stated_components",
                    time_precision="instant",
                    time_text="0856 hrs",
                    extra_evidence=_claims(
                        "title",
                        r06_title,
                        "affected_regions",
                        "affected_assets",
                        "effective_start_at",
                    ),
                )
            ],
        },
        "REAL-NESO-EXTREME-WEATHER-20260713": {
            "disposition": "event",
            "events": [
                _candidate(
                    event_type=event_type,
                    relevance="short_term",
                    status="occurred",
                    physical_effect=effect,
                    text_field="body",
                    quote=r07,
                    regions=["Great Britain"],
                    time_precision="vague",
                )
                for event_type, effect in (
                    ("renewable_supply_change", "supply_down"),
                    ("fuel_supply_change", "supply_down"),
                    ("demand_shock", "demand_up"),
                    ("transmission_constraint", "transfer_down"),
                )
            ],
        },
        "REAL-ERCOT-PEAK-20230914": {
            "disposition": "event",
            "events": [
                _candidate(
                    event_type="demand_shock",
                    relevance="short_term",
                    status="occurred",
                    physical_effect="demand_up",
                    text_field="body",
                    quote=r08,
                    regions=["Texas"],
                    assets=["ERCOT system load"],
                    quantity={
                        "value": 85_464,
                        "unit": "MW",
                        "semantic": "demand_level",
                        "direction": "unknown",
                        "raw_text": "85,464 MW",
                    },
                    time_precision="day",
                    time_text="August 10, 2023",
                )
            ],
        },
        "REAL-CN-GX-20260608": {
            "disposition": "event",
            "events": [
                _candidate(
                    event_type="generation_restore",
                    relevance="short_term",
                    status="restored",
                    physical_effect="supply_up",
                    text_field="body",
                    quote=r09,
                    regions=["广西"],
                    assets=["光坡核电厂1号机组"],
                    time_precision="vague",
                    time_text="本轮高温前",
                )
            ],
        },
        "REAL-CN-SASAC-COMMUNITY-20250429": {
            "disposition": "irrelevant",
            "document_evidence": _claims("body", r10, "relevance"),
        },
    }


def _document(*, source_id: str, title: str, body: str, market: str = "TEST"):
    return NewsNormalizer().normalize(
        CollectedNewsRecord(
            source_name="test-source",
            source_document_id=source_id,
            source_ref=f"https://example.invalid/{source_id}",
            title=title,
            body=body,
            published_at=datetime(2026, 8, 30, tzinfo=UTC),
            collected_at=datetime(2026, 8, 30, 1, tzinfo=UTC),
            market_tags=(market,),
        )
    )


def test_prompt_payload_is_label_free_and_keeps_the_three_time_semantics() -> None:
    record = JsonlCollectedNewsAdapter(NEWS).load()[0]
    document = NewsNormalizer().normalize(record)

    messages = build_model_news_extraction_messages(document, market_timezone="Asia/Shanghai")
    payload = json.loads(messages[-1].content)

    assert "corpus_id" not in messages[-1].content
    assert "gold_reason" not in messages[-1].content
    assert payload["published_at"] == document.published_at.isoformat()
    assert payload["available_at"] == document.available_at.isoformat()
    assert payload["published_time_precision"] == "minute"
    assert "事件发生时间" not in payload


def test_scripted_structured_model_qualifies_on_the_real_source_development_set() -> None:
    records = JsonlCollectedNewsAdapter(NEWS).load()
    manifest = load_real_news_gold(GOLD)
    gateway = ScriptedGateway(_real_source_outputs())

    report = benchmark_real_news_extractor(
        records,
        manifest,
        extractor_factory=lambda timezone: StructuredNewsEventExtractor(
            gateway,
            market_timezone=timezone,
        extraction_passes=1,
        ),
    )

    assert report.qualified
    assert report.extractable_event_recall == 1.0
    assert report.quarantine_recall == 1.0
    assert report.irrelevant_accuracy == 1.0
    assert report.unknown_rate == 0.0
    assert report.full_case_accuracy == 1.0
    assert gateway.calls == 10


def test_exact_event_preserves_quantity_time_evidence_and_reproducibility_trace() -> None:
    record = next(
        item
        for item in JsonlCollectedNewsAdapter(NEWS).load()
        if item.source_document_id == "REAL-AEMO-MILLMERRAN-20140326"
    )
    document = NewsNormalizer().normalize(record)
    gateway = ScriptedGateway(_real_source_outputs())
    extractor = StructuredNewsEventExtractor(gateway, market_timezone="Australia/Brisbane", extraction_passes=1)

    result = extractor.extract(document)
    event = result.events[0]

    assert result.quarantine is None
    assert event.capacity_mw == 870.0
    assert event.magnitude is not None
    assert event.magnitude.semantic == "generation_loss"
    assert event.magnitude.raw_text == "870 MW"
    assert event.effective_start_at == datetime(2013, 12, 22, 22, 56, tzinfo=UTC)
    assert event.time_resolution is not None
    assert event.time_resolution.basis == "stated_components"
    assert event.physical_effect == "supply_down"
    assert event.direction == "up"
    assert event.extraction_trace is not None
    assert event.extraction_trace.model_name == "scripted-news-model"
    assert len(event.extraction_trace.input_hash) == 64
    assert len(event.extraction_trace.output_hash) == 64
    assert {span.field_name for span in event.evidence} >= {
        "event_type",
        "capacity_mw",
        "effective_start_at",
        "direction",
    }
    assert extractor.extract(document) is result
    assert gateway.calls == 1


def test_model_evidence_must_be_an_exact_source_substring() -> None:
    record = next(
        item
        for item in JsonlCollectedNewsAdapter(NEWS).load()
        if item.source_document_id == "REAL-AEMO-MILLMERRAN-20140326"
    )
    document = NewsNormalizer().normalize(record)
    outputs = _real_source_outputs()
    outputs[record.source_document_id]["events"][0]["evidence"][0]["quote"] = "not present in source"

    result = StructuredNewsEventExtractor(
        ScriptedGateway(outputs),
        market_timezone="Australia/Brisbane",
        extraction_passes=1,
    ).extract(document)

    assert result.events == ()
    assert result.quarantine is not None
    assert result.quarantine.reason_code == "invalid_evidence"


def test_multi_pass_extraction_reports_each_document_pass() -> None:
    record = next(
        item
        for item in JsonlCollectedNewsAdapter(NEWS).load()
        if item.source_document_id == "REAL-AEMO-MILLMERRAN-20140326"
    )
    document = NewsNormalizer().normalize(record)
    progress: list[tuple[str, int, int]] = []
    gateway = ScriptedGateway(_real_source_outputs())

    result = StructuredNewsEventExtractor(
        gateway,
        market_timezone="Australia/Brisbane",
        extraction_passes=3,
        progress=lambda item, number, total: progress.append(
            (item.document_version_id, number, total)
        ),
    ).extract(document)

    assert result.quarantine is None
    assert gateway.calls == 3
    assert progress == [
        (document.document_version_id, 1, 3),
        (document.document_version_id, 2, 3),
        (document.document_version_id, 3, 3),
    ]


def test_stated_time_components_must_use_the_market_offset() -> None:
    record = next(
        item
        for item in JsonlCollectedNewsAdapter(NEWS).load()
        if item.source_document_id == "REAL-AEMO-MILLMERRAN-20140326"
    )
    document = NewsNormalizer().normalize(record)
    outputs = _real_source_outputs()
    # A stated-components time that uses UTC instead of the Brisbane offset must be caught.
    outputs[record.source_document_id]["events"][0]["event_instant"] = {
        "iso": "2013-12-23T08:56:00Z",
        "basis": "stated_components",
    }

    result = StructuredNewsEventExtractor(
        ScriptedGateway(outputs),
        market_timezone="Australia/Brisbane",
        extraction_passes=1,
    ).extract(document)

    assert result.events == ()
    assert result.quarantine is not None
    assert result.quarantine.reason_code == "ambiguous_event_time"


def test_quantity_levels_remain_out_of_backward_compatible_capacity_feature() -> None:
    body = "At 2026-08-30T01:00:00Z renewable output reached 85,464 MW."
    document = _document(source_id="level", title="Renewable output record", body=body)
    outputs = {
        "level": {
            "disposition": "event",
            "events": [
                _candidate(
                    event_type="renewable_supply_change",
                    relevance="short_term",
                    status="occurred",
                    physical_effect="supply_up",
                    text_field="body",
                    quote=body,
                    quantity={
                        "value": 85_464,
                        "unit": "MW",
                        "semantic": "output_level",
                        "direction": "unknown",
                        "raw_text": "85,464 MW",
                    },
                    effective_start_at="2026-08-30T01:00:00Z",
                    time_basis="stated_absolute",
                    time_precision="instant",
                    time_text="2026-08-30T01:00:00Z",
                )
            ],
        }
    }

    event = StructuredNewsEventExtractor(
        ScriptedGateway(outputs),
        market_timezone="UTC",
        extraction_passes=1,
    ).extract(document).events[0]

    assert event.magnitude is not None
    assert event.magnitude.normalized_mw == 85_464
    assert event.capacity_mw is None


def test_sibling_events_are_not_merged_and_model_is_not_called_twice_for_evidence() -> None:
    body = (
        "Unit A tripped at 2026-08-30T01:00:00Z. "
        "Unit B tripped at 2026-08-30T02:00:00Z."
    )
    document = _document(source_id="siblings", title="Two unit trips", body=body)
    outputs = {
        "siblings": {
            "disposition": "event",
            "events": [
                _candidate(
                    event_type="generation_outage",
                    relevance="short_term",
                    status="occurred",
                    physical_effect="supply_down",
                    text_field="body",
                    quote=quote,
                    assets=[asset],
                    effective_start_at=start,
                    time_basis="stated_absolute",
                    time_precision="instant",
                    time_text=start,
                )
                for asset, start, quote in (
                    ("Unit A", "2026-08-30T01:00:00Z", "Unit A tripped at 2026-08-30T01:00:00Z."),
                    ("Unit B", "2026-08-30T02:00:00Z", "Unit B tripped at 2026-08-30T02:00:00Z."),
                )
            ],
        }
    }
    gateway = ScriptedGateway(outputs)
    extractor = StructuredNewsEventExtractor(gateway, market_timezone="UTC", extraction_passes=1)
    store = NewsVersionStore((document,))

    view = AsOfEventAssembler(store, extractor=extractor).view_at(
        datetime(2026, 8, 31, tzinfo=UTC)
    )
    spans = collect_evidence_spans(
        (document,),
        extraction_results=view.extraction_results,
    )

    assert len(view.events) == 2
    assert {event.affected_assets for event in view.events} == {("Unit A",), ("Unit B",)}
    assert gateway.calls == 1
    assert document.document_version_id in spans
    assert {span.event_id for span in spans[document.document_version_id]} == {
        event_id for event in view.events for event_id in event.source_event_ids
    }


def test_a_bad_model_response_quarantines_one_document_without_crashing_the_batch() -> None:
    document = _document(source_id="failure", title="Unit event", body="Unit event body")

    # A response the model produced but that cannot be used for THIS document is a per-document
    # problem: quarantine it and let the batch continue, the same way the rule extractor does.
    result = StructuredNewsEventExtractor(
        FailingGateway(ModelResponseError("bad schema")),
        market_timezone="UTC",
    ).extract(document)
    assert result.events == ()
    assert result.quarantine is not None
    assert result.quarantine.reason_code == "model_response_invalid"

    transient = StructuredNewsEventExtractor(
        FailingGateway(ModelTransientError("provider 500")),
        market_timezone="UTC",
    ).extract(document)
    assert transient.events == ()
    assert transient.quarantine is not None
    assert transient.quarantine.reason_code == "model_unavailable"

    # A broken endpoint is global, not this document's fault, so it still fails closed.
    with pytest.raises(ModelGatewayError, match="offline"):
        StructuredNewsEventExtractor(
            FailingGateway(ModelGatewayError("offline")),
            market_timezone="UTC",
        ).extract(document)


def _good_single_event_output() -> dict:
    return {
        "disposition": "event",
        "events": [
            _candidate(
                event_type="generation_outage",
                relevance="short_term",
                status="occurred",
                physical_effect="supply_down",
                text_field="body",
                quote="Unit A tripped at 2026-08-30T01:00:00Z.",
                assets=["Unit A"],
                effective_start_at="2026-08-30T01:00:00Z",
                time_basis="stated_absolute",
                time_precision="instant",
                time_text="2026-08-30T01:00:00Z",
            )
        ],
    }


def test_a_schema_failure_is_repaired_by_feeding_the_error_back() -> None:
    document = _document(
        source_id="repair", title="Unit event", body="Unit A tripped at 2026-08-30T01:00:00Z."
    )
    gateway = RepairGateway(
        ModelResponseError("events.0.quantity.unit Input should be 'MW' [input_value='台机组']"),
        _good_single_event_output(),
    )

    # Default max_repair_attempts=1: the first parse fails, the error is fed back, the
    # corrected retry succeeds.
    result = StructuredNewsEventExtractor(gateway, market_timezone="UTC", extraction_passes=1).extract(document)

    assert gateway.calls == 2
    assert gateway.received_error_feedback, "第二次调用必须带上一次的校验错误"
    assert result.quarantine is None
    assert len(result.events) == 1


def test_repair_is_bounded_and_can_be_disabled() -> None:
    document = _document(source_id="norepair", title="Unit event", body="Unit A tripped.")

    # Disabled: one call, straight to quarantine.
    off = RepairGateway(ModelResponseError("bad schema"), _good_single_event_output())
    off_result = StructuredNewsEventExtractor(
        off, market_timezone="UTC", max_repair_attempts=0, extraction_passes=1
    ).extract(document)
    assert off.calls == 1
    assert off_result.quarantine is not None

    # Persistently failing: bounded to 1 retry (two calls total), then quarantine.
    stubborn = RepairGateway(
        ModelResponseError("bad schema"), _good_single_event_output(), always_fail=True
    )
    stubborn_result = StructuredNewsEventExtractor(
        stubborn, market_timezone="UTC", max_repair_attempts=1, extraction_passes=1
    ).extract(document)
    assert stubborn.calls == 2
    assert stubborn_result.quarantine is not None
    assert stubborn_result.quarantine.reason_code == "model_response_invalid"

    with pytest.raises(ValueError, match="max_repair_attempts"):
        StructuredNewsEventExtractor(off, market_timezone="UTC", max_repair_attempts=-1)


def test_a_long_validation_error_is_clipped_instead_of_crashing_the_quarantine() -> None:
    """A multi-error schema failure must not exceed the quarantine message cap and crash."""

    document = _document(source_id="verbose", title="Unit event", body="Unit event body")
    gateway = FailingGateway(ModelResponseError("字段错误 " * 4000))

    result = StructuredNewsEventExtractor(
        gateway, market_timezone="UTC", max_repair_attempts=0
    ).extract(document)

    assert result.quarantine is not None
    assert result.quarantine.reason_code == "model_response_invalid"
    assert len(result.quarantine.message) <= 2048


def test_one_unusable_response_does_not_discard_the_usable_ones() -> None:
    good = _document(source_id="good", title="Unit event", body="Unit A tripped at 2026-08-30T01:00:00Z.")
    bad = _document(source_id="bad", title="Unit event", body="Unit event body")

    class MixedGateway:
        model_name = "mixed-news-model"

        def invoke_structured(self, *, messages, schema):
            payload = _document_payload(messages)
            if payload["source_document_id"] == "bad":
                raise ModelResponseError("bad schema for one document")
            return schema.model_validate(
                {
                    "disposition": "event",
                    "events": [
                        _candidate(
                            event_type="generation_outage",
                            relevance="short_term",
                            status="occurred",
                            physical_effect="supply_down",
                            text_field="body",
                            quote="Unit A tripped at 2026-08-30T01:00:00Z.",
                            assets=["Unit A"],
                            effective_start_at="2026-08-30T01:00:00Z",
                            time_basis="stated_absolute",
                            time_precision="instant",
                            time_text="2026-08-30T01:00:00Z",
                        )
                    ],
                }
            )

    batch = StructuredNewsEventExtractor(MixedGateway(), market_timezone="UTC").extract_many((good, bad))

    assert len(batch.results) == 2
    assert len(batch.events) == 1
    assert len(batch.quarantined) == 1
    assert batch.quarantined[0].reason_code == "model_response_invalid"


def test_unexpected_programming_error_is_not_hidden_as_quarantine() -> None:
    document = _document(source_id="bug", title="Unit event", body="Unit event body")
    extractor = StructuredNewsEventExtractor(
        FailingGateway(TypeError("programming bug")),
        market_timezone="UTC",
    )

    with pytest.raises(TypeError, match="programming bug"):
        extractor.extract(document)
