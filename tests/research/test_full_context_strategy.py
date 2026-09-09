import json
from datetime import UTC, datetime

from app.llm.gateway import ModelMessage
from app.research.news import CollectedNewsRecord, FullContextNewsExtractor, ModelRoute, NewsNormalizer


def _claims(quote: str, asset: str, instant: str) -> list[dict]:
    return [
        {"field_name": field, "text_field": "body", "quote": quote, "occurrence_index": 0}
        for field in ("relevance", "event_type", "status", "physical_effect")
    ] + [
        {"field_name": "affected_assets", "text_field": "body", "quote": asset, "occurrence_index": 0},
        {"field_name": "effective_start_at", "text_field": "body", "quote": instant, "occurrence_index": 0},
    ]


class _FunctionalGateway:
    model_name = "functional-full-context"

    def __init__(self) -> None:
        self.calls = 0
        self.bodies: list[str] = []

    @property
    def enabled(self) -> bool:
        return True

    def invoke_structured(self, *, messages: list[ModelMessage], schema):
        self.calls += 1
        payload = json.loads(messages[-1].content)
        body = payload["body"]
        self.bodies.append(body)
        quote = "华能测试电厂1号机组于2026-05-24T02:30:00+08:00紧急停机。"
        asset = "华能测试电厂1号机组"
        instant = "2026-05-24T02:30:00+08:00"
        if quote not in body:
            return schema.model_validate(
                {
                    "disposition": "irrelevant",
                    "document_evidence": [
                        {"field_name": "relevance", "text_field": "body", "quote": body[:20]}
                    ],
                }
            )
        return schema.model_validate(
            {
                "disposition": "event",
                "events": [
                    {
                        "relevance": "short_term",
                        "event_type": "generation_outage",
                        "status": "occurred",
                        "physical_effect": "supply_down",
                        "affected_assets": [asset],
                        "time_precision": "instant",
                        "time_text": instant,
                        "event_instant": {"iso": instant, "basis": "stated_absolute"},
                        "confidence": 0.99,
                        "evidence": _claims(quote, asset, instant),
                    }
                ],
            }
        )


def _document(repetitions: int = 100):
    tail = "华能测试电厂1号机组于2026-05-24T02:30:00+08:00紧急停机。"
    body = ("这是背景材料，不包含运行事件。" * repetitions) + tail
    return NewsNormalizer().normalize(
        CollectedNewsRecord(
            source_name="functional-test",
            source_document_id="long-tail-event",
            source_ref="fixture://long-tail-event",
            title="电力运行消息",
            body=body,
            published_at=datetime(2026, 5, 24, 3, tzinfo=UTC),
            collected_at=datetime(2026, 5, 24, 3, 1, tzinfo=UTC),
            market_tags=("TEST",),
            language="zh-CN",
        )
    )


def test_full_context_routes_to_first_model_that_fits_and_keeps_tail_event() -> None:
    document = _document()
    small = _FunctionalGateway()
    large = _FunctionalGateway()
    extractor = FullContextNewsExtractor(
        (
            ModelRoute("small", small, context_window_tokens=600, reserved_output_tokens=100, safety_tokens=50),
            ModelRoute("large", large, context_window_tokens=20_000, reserved_output_tokens=100, safety_tokens=50),
        ),
        market_timezone="Asia/Shanghai",
    )

    result = extractor.extract(document)

    assert small.calls == 0
    assert large.calls == 1
    assert large.bodies == [document.body]
    assert result.events[0].event_type == "generation_outage"
    assert result.coverage is not None and result.coverage.complete
    assert extractor.last_route_name == "large"


def test_full_context_fails_closed_when_no_model_can_fit_the_whole_document() -> None:
    document = _document(500)
    gateway = _FunctionalGateway()
    result = FullContextNewsExtractor(
        (ModelRoute("small", gateway, context_window_tokens=300, reserved_output_tokens=100, safety_tokens=50),),
        market_timezone="Asia/Shanghai",
    ).extract(document)

    assert gateway.calls == 0
    assert result.quarantine is not None
    assert result.quarantine.reason_code == "context_limit_exceeded"
    assert result.coverage is not None and not result.coverage.complete
