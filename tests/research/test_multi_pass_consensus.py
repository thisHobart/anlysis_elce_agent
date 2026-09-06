"""Independent passes turn an unreproducible endpoint into an explicit review decision.

The holdout run showed the model flipping on exactly the documents that sit on a decision
boundary: one sentence describing two record peaks was read as two events on some runs and
one on others, and a cumulative outage statistic was called irrelevant four times out of
five. Both are silent losses under a single pass. These tests pin the behaviour that turns
them into quarantine.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.research.news import (
    CollectedNewsRecord,
    NewsNormalizer,
    StructuredNewsEventExtractor,
)

BODY = "8月3日11时06分，辽宁电网最大用电负荷冲至4775万千瓦。"
TIME_TEXT = "8月3日11时06分"
# No power figures and no operational vocabulary, so a terminal "irrelevant" is believable.
SOCIAL_BODY = "服务队在37个社区开展爱心服务活动，长期提供管家式陪伴。"


def _document(body: str = BODY):
    record = CollectedNewsRecord(
        source_name="测试来源",
        source_document_id="CONSENSUS-01",
        source_ref="https://example.invalid/consensus",
        title="社区服务活动" if body == SOCIAL_BODY else "用电负荷创历史新高",
        body=body,
        published_at=datetime(2026, 8, 6, 6, 7, tzinfo=UTC),
        collected_at=datetime(2026, 8, 6, 7, 0, tzinfo=UTC),
        language="zh-CN",
        market_tags=("CN-LIAONING",),
        metadata={"published_time_precision": "minute"},
    )
    return NewsNormalizer().normalize(record)


def _evidence(*field_names: str) -> list[dict]:
    return [
        {"field_name": name, "text_field": "body", "quote": BODY} for name in field_names
    ]


def _event_payload(*, region: str = "辽宁电网", precision: str = "instant") -> dict:
    """One usable event; at day precision the local time gate quarantines it instead."""

    fields = ["relevance", "event_type", "status", "physical_effect", "affected_regions"]
    evidence = _evidence(*fields)
    instant = None
    if precision == "instant":
        evidence.extend(_evidence("effective_start_at"))
        instant = {"iso": "2026-08-03T11:06:00+08:00", "basis": "stated_absolute"}
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
                "time_precision": precision,
                "time_text": TIME_TEXT if precision == "instant" else None,
                "event_instant": instant,
                "event_end_instant": None,
                "confidence": 0.95,
                "evidence": evidence,
            }
        ],
    }


def _irrelevant_payload(body: str = BODY) -> dict:
    return {
        "disposition": "irrelevant",
        "events": [],
        "document_evidence": [
            {"field_name": "relevance", "text_field": "body", "quote": body}
        ],
    }


class _ScriptedPasses:
    """Return a different scripted answer per pass, in order."""

    model_name = "consensus-test-model"

    def __init__(self, *payloads: dict) -> None:
        self.payloads = payloads
        self.calls = 0

    def invoke_structured(self, *, messages, schema):
        payload = self.payloads[min(self.calls, len(self.payloads) - 1)]
        self.calls += 1
        return schema.model_validate(payload)


def _extract(*payloads: dict, passes: int, body: str = BODY):
    gateway = _ScriptedPasses(*payloads)
    extractor = StructuredNewsEventExtractor(
        gateway,
        market_timezone="Asia/Shanghai",
        extraction_passes=passes,
    )
    return extractor.extract(_document(body)), gateway


def test_agreeing_passes_accept_the_event() -> None:
    result, gateway = _extract(_event_payload(), _event_payload(), passes=3)

    assert gateway.calls == 3
    assert result.quarantine is None
    assert result.events[0].event_type == "demand_shock"


def test_wording_drift_between_passes_is_not_treated_as_disagreement() -> None:
    """辽宁 and 辽宁电网 are one grid; only canonical content decides agreement."""

    result, _ = _extract(
        _event_payload(region="辽宁电网"),
        _event_payload(region="辽宁"),
        _event_payload(region="辽宁省"),
        passes=3,
    )

    assert result.quarantine is None
    assert result.events[0].region_keys == ("CN-LIAONING",)


def test_a_pass_that_finds_the_event_and_a_pass_that_does_not_go_to_review() -> None:
    """This is the 安徽 case: one reading yields an event, another holds it back."""

    result, _ = _extract(
        _event_payload(precision="instant"),
        _event_payload(precision="day"),
        _event_payload(precision="instant"),
        passes=3,
    )

    assert result.quarantine is not None
    assert result.quarantine.reason_code == "inconsistent_extraction"


def test_a_lone_irrelevant_verdict_cannot_silently_drop_the_document() -> None:
    """The dangerous direction: irrelevant is never reviewed, quarantine is."""

    result, _ = _extract(
        _irrelevant_payload(),
        _irrelevant_payload(),
        _event_payload(),
        passes=3,
    )

    assert result.quarantine is not None
    assert result.quarantine.reason_code == "inconsistent_extraction"


def test_unanimous_irrelevant_is_still_accepted() -> None:
    """A genuinely irrelevant article still gets a terminal verdict, without needing review."""

    result, _ = _extract(
        _irrelevant_payload(body=SOCIAL_BODY),
        passes=3,
        body=SOCIAL_BODY,
    )

    assert result.quarantine is None
    assert result.events[0].event_type == "irrelevant"


def test_unanimous_quarantine_keeps_the_real_reason_instead_of_an_inconsistency_label() -> None:
    result, _ = _extract(_event_payload(precision="day"), passes=3)

    assert result.quarantine is not None
    assert result.quarantine.reason_code == "missing_effective_start"


def test_events_that_disagree_on_content_go_to_review() -> None:
    result, _ = _extract(
        _event_payload(region="辽宁"),
        _event_payload(region="华东"),
        passes=2,
    )

    assert result.quarantine is not None
    assert result.quarantine.reason_code == "inconsistent_extraction"


def test_three_passes_are_the_default_because_a_flag_nobody_sets_protects_nobody() -> None:
    """The stability this buys has to be what you get, not what you get if you remember to ask."""

    extractor = StructuredNewsEventExtractor(
        _ScriptedPasses(_event_payload()),
        market_timezone="Asia/Shanghai",
    )

    assert extractor.extraction_passes == 3


def test_a_single_pass_is_still_available_and_calls_the_model_once() -> None:
    result, gateway = _extract(_event_payload(), passes=1)

    assert gateway.calls == 1
    assert result.quarantine is None


def test_extraction_passes_must_be_at_least_one() -> None:
    with pytest.raises(ValueError, match="extraction_passes"):
        StructuredNewsEventExtractor(
            _ScriptedPasses(_event_payload()),
            market_timezone="Asia/Shanghai",
            extraction_passes=0,
        )


# ---------------------------------------------------------------------------
# 确定性闸门：模型稳定判错时，一致性闸门看到的是“一致”，只能靠原文兜底
# ---------------------------------------------------------------------------

MULTI_BODY = "安徽全省最大用电负荷继8月3日首创新高后，在4日午间12时45分再创历史新高，达6999万千瓦。"
TALLY_BODY = "截至2026年7月份，全省100MW及以上统调机组异常停运48次。"


def test_three_passes_agreeing_on_one_of_two_events_still_go_to_review() -> None:
    """Agreement is not correctness: every pass can miss the same event."""

    payload = _event_payload()
    payload["events"][0]["time_text"] = "4日午间12时45分"
    payload["events"][0]["event_instant"]["iso"] = "2026-08-04T12:45:00+08:00"
    for span in payload["events"][0]["evidence"]:
        span["quote"] = MULTI_BODY

    result, gateway = _extract(payload, passes=3, body=MULTI_BODY)

    assert gateway.calls == 3, "三趟都跑了，且三趟一致"
    assert result.quarantine is None
    assert len(result.events) == 1
    assert result.candidate_quarantines[0].reason_code == "ambiguous_multi_event"
    assert "2" in result.candidate_quarantines[0].message


def test_a_unanimous_irrelevant_verdict_is_refused_when_the_text_says_otherwise() -> None:
    """The outage tally the model dismissed as irrelevant on two runs out of three."""

    result, _ = _extract(
        _irrelevant_payload(body=TALLY_BODY),
        passes=3,
        body=TALLY_BODY,
    )

    assert result.events == ()
    assert result.quarantine is not None
    assert result.quarantine.reason_code == "disputed_irrelevance"
    assert "100MW" in result.quarantine.message


def test_the_extractor_records_what_the_passes_cost() -> None:
    """Three passes is a real bill; it has to be measurable, not inferred from the flag."""

    gateway = _ScriptedPasses(_event_payload())
    extractor = StructuredNewsEventExtractor(
        gateway,
        market_timezone="Asia/Shanghai",
        extraction_passes=3,
    )
    extractor.extract(_document())

    assert extractor.model_calls == 3
    assert extractor.model_seconds >= 0.0


ASSET_BODY = (
    "8月3日11时06分，辽宁电网通报 AGRs (advanced gas cooled reactor stations)、"
    "Unit A、Unit B 和 Unit C 的运行情况。"
)


def _payload_with_asset(asset: str) -> dict:
    payload = _event_payload()
    payload["events"][0]["affected_assets"] = [asset]
    for span in payload["events"][0]["evidence"]:
        span["quote"] = ASSET_BODY
    payload["events"][0]["evidence"].append(
        {"field_name": "affected_assets", "text_field": "body", "quote": ASSET_BODY}
    )
    return payload


def test_the_same_asset_described_two_ways_takes_the_majority_spelling() -> None:
    """Two passes wrote AGRs where a third spelled the reactors out; that is one asset."""

    result, _ = _extract(
        _payload_with_asset("AGRs"),
        _payload_with_asset("advanced gas cooled reactor stations"),
        _payload_with_asset("AGRs"),
        passes=3,
        body=ASSET_BODY,
    )

    assert result.quarantine is None
    assert result.events[0].affected_assets == ("AGRs",)


def test_three_different_asset_names_have_no_majority_and_go_to_review() -> None:
    """Without a majority there is nothing to prefer, so the fail-safe still applies."""

    result, _ = _extract(
        _payload_with_asset("Unit A"),
        _payload_with_asset("Unit B"),
        _payload_with_asset("Unit C"),
        passes=3,
        body=ASSET_BODY,
    )

    assert result.quarantine is not None
    assert result.quarantine.reason_code == "inconsistent_extraction"
    assert "资产名" in result.quarantine.message


def test_a_disagreement_about_the_conclusion_is_never_settled_by_majority() -> None:
    """Only the description is a vote; content still requires every pass to agree."""

    result, _ = _extract(
        _event_payload(),
        _event_payload(),
        _event_payload(precision="day"),
        passes=3,
    )

    assert result.quarantine is not None
    assert result.quarantine.reason_code == "inconsistent_extraction"
