"""Deterministic functional benchmark shared by all long-context strategy branches."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.llm.context_safety import structured_request_budget
from app.llm.gateway import (
    ModelContextLimitError,
    ModelMessage,
    ModelOutputTruncatedError,
    ModelResponseError,
)
from app.research.news import CollectedNewsRecord, NewsNormalizer
from app.research.news.long_context import (
    LONG_CONTEXT_STRATEGY,
    build_long_context_extractor,
)
from app.research.news.model_extraction import (
    ModelNewsExtraction,
    build_model_news_extraction_messages,
)

DEFAULT_CASES = ROOT / "tests" / "fixtures" / "news_long_context" / "gold_cases.yaml"

OUTAGE = "华能测试电厂1号机组于2026-05-24T02:30:00+08:00紧急停机，影响出力约60万千瓦。"
CORRECTION = "更正：上述华能测试电厂1号机组影响出力应为30万千瓦，事件时间仍为2026-05-24T02:30:00+08:00。"
TAIL = "大唐尾部电厂2号机组于2026-05-24T04:00:00+08:00发生跳闸。"
RESTORE = "国电恢复电厂5号机组于2026-05-24T05:00:00+08:00恢复并网。"
CROSS_ASSET = "华能跨段电厂3号机组发生紧急停机。"
CROSS_DETAIL = "本次事件影响出力20万千瓦，发生于2026-05-24T06:00:00+08:00。"
RETRACTION = (
    "撤回：有关华能测试电厂1号机组于2026-05-24T02:30:00+08:00紧急停机、影响出力60万千瓦"
    "的消息不实，该机组正常运行。"
)
BOUNDARY = "华能边界电厂6号机组于2026-05-24T06:30:00+08:00紧急停机，影响出力10万千瓦。"
TRUNCATE_OUTPUT = "TRUNCATE_OUTPUT"


def _claims(
    *,
    fact_quote: str,
    asset: str,
    instant: str,
    quantity_quote: str | None = None,
    status_quote: str | None = None,
) -> list[dict[str, Any]]:
    claims = [
        {"field_name": field, "text_field": "body", "quote": fact_quote}
        for field in ("relevance", "event_type", "physical_effect")
    ]
    claims.append(
        {"field_name": "status", "text_field": "body", "quote": status_quote or fact_quote}
    )
    claims.extend(
        (
            {"field_name": "affected_assets", "text_field": "body", "quote": asset},
            {"field_name": "effective_start_at", "text_field": "body", "quote": instant},
        )
    )
    if quantity_quote:
        claims.append({"field_name": "magnitude", "text_field": "body", "quote": quantity_quote})
    return claims


def _candidate(
    *,
    event_type: str,
    status: str,
    physical_effect: str,
    fact_quote: str,
    asset: str,
    instant: str,
    quantity_value: float | None = None,
    quantity_quote: str | None = None,
    quantity_semantic: str = "generation_loss",
    status_quote: str | None = None,
) -> dict[str, Any]:
    return {
        "relevance": "short_term",
        "event_type": event_type,
        "status": status,
        "physical_effect": physical_effect,
        "affected_assets": [asset],
        "quantity": (
            {
                "value": quantity_value,
                "unit": "万千瓦",
                "semantic": quantity_semantic,
                "direction": "increase" if physical_effect == "supply_up" else "decrease",
                "raw_text": quantity_quote,
            }
            if quantity_value is not None
            else None
        ),
        "time_precision": "instant",
        "time_text": instant,
        "event_instant": {"iso": instant, "basis": "stated_absolute"},
        "confidence": 0.99,
        "evidence": _claims(
            fact_quote=fact_quote,
            asset=asset,
            instant=instant,
            quantity_quote=quantity_quote,
            status_quote=status_quote,
        ),
    }


def _outage(capacity: int = 60, *, corrected: bool = False) -> dict[str, Any]:
    if corrected:
        return _candidate(
            event_type="generation_outage",
            status="corrected",
            physical_effect="supply_down",
            fact_quote=OUTAGE,
            status_quote=CORRECTION,
            asset="华能测试电厂1号机组",
            instant="2026-05-24T02:30:00+08:00",
            quantity_value=capacity,
            quantity_quote=f"{capacity}万千瓦",
        )
    return _candidate(
        event_type="generation_outage",
        status="occurred",
        physical_effect="supply_down",
        fact_quote=OUTAGE,
        asset="华能测试电厂1号机组",
        instant="2026-05-24T02:30:00+08:00",
        quantity_value=capacity,
        quantity_quote=f"{capacity}万千瓦",
    )


def _tail() -> dict[str, Any]:
    return _candidate(
        event_type="generation_outage",
        status="occurred",
        physical_effect="supply_down",
        fact_quote=TAIL,
        asset="大唐尾部电厂2号机组",
        instant="2026-05-24T04:00:00+08:00",
    )


def _restore() -> dict[str, Any]:
    return _candidate(
        event_type="generation_restore",
        status="restored",
        physical_effect="supply_up",
        fact_quote=RESTORE,
        asset="国电恢复电厂5号机组",
        instant="2026-05-24T05:00:00+08:00",
    )


def _cross() -> dict[str, Any]:
    candidate = _candidate(
        event_type="generation_outage",
        status="occurred",
        physical_effect="supply_down",
        fact_quote=CROSS_ASSET,
        asset="华能跨段电厂3号机组",
        instant="2026-05-24T06:00:00+08:00",
        quantity_value=20,
        quantity_quote="20万千瓦",
    )
    for claim in candidate["evidence"]:
        if claim["field_name"] in {"effective_start_at", "magnitude"}:
            claim["quote"] = (
                "2026-05-24T06:00:00+08:00"
                if claim["field_name"] == "effective_start_at"
                else "20万千瓦"
            )
    return candidate


def _cancelled_outage() -> dict[str, Any]:
    return _candidate(
        event_type="generation_outage",
        status="cancelled",
        physical_effect="unknown",
        fact_quote=RETRACTION,
        status_quote=RETRACTION,
        asset="华能测试电厂1号机组",
        instant="2026-05-24T02:30:00+08:00",
        quantity_value=60,
        quantity_quote="60万千瓦",
    )


def _boundary() -> dict[str, Any]:
    return _candidate(
        event_type="generation_outage",
        status="occurred",
        physical_effect="supply_down",
        fact_quote=BOUNDARY,
        asset="华能边界电厂6号机组",
        instant="2026-05-24T06:30:00+08:00",
        quantity_value=10,
        quantity_quote="10万千瓦",
    )


class FunctionalLongContextGateway:
    """A deterministic model substitute; it sees only the text each strategy sends."""

    model_name = "functional-long-context-model"

    def __init__(self) -> None:
        self.calls = 0

    @property
    def enabled(self) -> bool:
        return True

    def invoke_structured(self, *, messages: list[ModelMessage], schema):
        self.calls += 1
        payload = next(
            json.loads(message.content)
            for message in messages
            if message.role == "user" and message.content.lstrip().startswith("{")
        )
        if "segment" in payload:
            return self._incremental(payload, schema)
        return self._document_or_chunk(payload["body"], schema)

    def _document_or_chunk(self, text: str, schema):
        if "FAIL_STEP" in text:
            raise ModelResponseError("injected functional failure")
        if TRUNCATE_OUTPUT in text:
            raise ModelOutputTruncatedError("injected maximum output token failure")
        candidates: list[dict[str, Any]] = []
        if RETRACTION in text:
            candidates.append(_cancelled_outage())
        elif CORRECTION in text and OUTAGE in text:
            candidates.append(_outage(30, corrected=True))
        elif OUTAGE in text:
            candidates.append(_outage())
        elif CORRECTION in text:
            candidates.append(_outage(30, corrected=True))
        if TAIL in text:
            candidates.append(_tail())
        if RESTORE in text:
            candidates.append(_restore())
        if CROSS_ASSET in text and CROSS_DETAIL in text:
            candidates.append(_cross())
        if BOUNDARY in text:
            candidates.append(_boundary())
        if candidates:
            return schema.model_validate({"disposition": "event", "events": candidates})
        quote = text[: min(30, len(text))]
        return schema.model_validate(
            {
                "disposition": "irrelevant",
                "document_evidence": [
                    {"field_name": "relevance", "text_field": "body", "quote": quote}
                ],
            }
        )

    def _incremental(self, payload: dict[str, Any], schema):
        text = payload["segment"]["text"]
        state = payload["validated_state"]
        if "FAIL_STEP" in text:
            raise ModelResponseError("injected functional failure")
        if TRUNCATE_OUTPUT in text:
            raise ModelOutputTruncatedError("injected maximum output token failure")
        operations: list[dict[str, Any]] = []
        if OUTAGE in text:
            operations.append({"action": "upsert", "candidate": _outage()})
        if CORRECTION in text:
            if not state:
                return schema.model_validate({"uncertainty_reason": "更正前没有已验证状态"})
            operations.append({"action": "correct", "candidate": _outage(30, corrected=True)})
        if TAIL in text:
            operations.append({"action": "upsert", "candidate": _tail()})
        if RESTORE in text:
            operations.append({"action": "upsert", "candidate": _restore()})
        if RETRACTION in text:
            if not state and OUTAGE not in text:
                return schema.model_validate({"uncertainty_reason": "撤回前没有已验证状态"})
            operations.append({"action": "withdraw", "candidate": _cancelled_outage()})
        if BOUNDARY in text:
            operations.append({"action": "upsert", "candidate": _boundary()})
        if operations:
            return schema.model_validate({"operations": operations})
        quote = text[: min(30, len(text))]
        return schema.model_validate(
            {
                "document_evidence": [
                    {"field_name": "relevance", "text_field": "body", "quote": quote}
                ]
            }
        )


class FaultInjectingGateway:
    """Apply the same deterministic failed-range scenario to real provider runs."""

    def __init__(self, delegate) -> None:
        self.delegate = delegate

    @property
    def model_name(self):
        return self.delegate.model_name

    @property
    def enabled(self):
        return self.delegate.enabled

    def invoke_structured(self, *, messages: list[ModelMessage], schema):
        payload = next(
            json.loads(message.content)
            for message in messages
            if message.role == "user" and message.content.lstrip().startswith("{")
        )
        text = payload["segment"]["text"] if "segment" in payload else payload["body"]
        if "FAIL_STEP" in text:
            raise ModelResponseError("injected live functional failure")
        if TRUNCATE_OUTPUT in text:
            raise ModelOutputTruncatedError("injected live maximum output token failure")
        return self.delegate.invoke_structured(messages=messages, schema=schema)


def load_cases(path: Path = DEFAULT_CASES) -> list[dict[str, Any]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    cases = []
    for item in payload["cases"]:
        body = "\n\n".join(
            part["text"] for part in item["body_parts"] for _ in range(part.get("repeat", 1))
        )
        cases.append({**item, "body": body})
    return cases


def _document(case: dict[str, Any]):
    return NewsNormalizer().normalize(
        CollectedNewsRecord(
            source_name="long-context-functional-corpus",
            source_document_id=case["case_id"],
            source_ref=f"fixture://long-context/{case['case_id']}",
            # Gold descriptions explain the expected failure shape and must never leak
            # into model-visible input.
            title=f"长文本功能样本 {case['case_id']}",
            body=case["body"],
            published_at=datetime(2026, 5, 24, 7, tzinfo=UTC),
            collected_at=datetime(2026, 5, 24, 7, 1, tzinfo=UTC),
            language="zh-CN",
            market_tags=("TEST",),
            metadata={"corpus_id": case["case_id"]},
        )
    )


def _price_admissible(event) -> bool:
    return (
        event.analysis_eligibility == "eligible"
        and event.status != "cancelled"
        and event.relevance == "short_term"
        and event.event_type not in {"irrelevant", "unknown"}
        and event.effective_start_at is not None
    )


def _percentile_95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def run_context_boundary_checks() -> dict[str, Any]:
    """Exercise below/equal/above-limit behavior for one complete structured request."""

    document = _document(load_cases()[0])
    messages = build_model_news_extraction_messages(document, market_timezone="Asia/Shanghai")
    measured = structured_request_budget(
        messages,
        ModelNewsExtraction,
        context_window_tokens=10**12,
        reserved_output_tokens=100,
        safety_tokens=50,
    )
    required = measured.required_tokens
    profiles = (
        ("request_below_limit", required + 1, True),
        ("request_equal_limit", required, True),
        ("request_above_limit", required - 1, False),
    )
    outcomes = []
    for name, limit, expected_acceptance in profiles:
        try:
            structured_request_budget(
                messages,
                ModelNewsExtraction,
                context_window_tokens=limit,
                reserved_output_tokens=100,
                safety_tokens=50,
            )
            accepted = True
        except ModelContextLimitError:
            accepted = False
        outcomes.append(
            {
                "profile": name,
                "required_tokens": required,
                "context_window_tokens": limit,
                "accepted": accepted,
                "passed": accepted == expected_acceptance,
            }
        )
    return {
        "passed": all(item["passed"] for item in outcomes),
        "profiles": outcomes,
    }


def run_benchmark(
    path: Path = DEFAULT_CASES,
    *,
    gateway_factory=FunctionalLongContextGateway,
    unit_tokens: int = 40,
) -> dict[str, Any]:
    outcomes: list[dict[str, Any]] = []
    gold_total = actual_total = true_positive = 0
    evidence_valid = evidence_total = unsafe_price_admissions = 0
    total_calls = total_tokens = 0
    latencies: list[float] = []
    for case in load_cases(path):
        gateway = gateway_factory()
        extractor = build_long_context_extractor(
            gateway,
            market_timezone="Asia/Shanghai",
            context_window_tokens=case.get("context_window_tokens", 30_000),
            unit_tokens=unit_tokens,
        )
        document = _document(case)
        case_started = time.perf_counter()
        result = extractor.extract(document)
        latency = time.perf_counter() - case_started
        latencies.append(latency)
        events = [event for event in result.events if event.event_type != "irrelevant"]
        actual_types = Counter(event.event_type for event in events)
        expected_types = Counter(case["expected_event_types"])
        actual_statuses = Counter(event.status for event in events)
        expected_statuses = Counter(case["expected_statuses"])
        matched = sum((actual_types & expected_types).values())
        if case["expectation"] == "complete":
            gold_total += sum(expected_types.values())
            actual_total += sum(actual_types.values())
            true_positive += matched
        elif any(_price_admissible(event) for event in events):
            unsafe_price_admissions += 1
        for event in events:
            for span in event.evidence:
                evidence_total += 1
                source = getattr(document, span.text_field)
                evidence_valid += source[span.start_char : span.end_char] == span.quote
        coverage = result.coverage
        quarantine_reason_codes = [
            item.reason_code
            for item in (
                *((result.quarantine,) if result.quarantine is not None else ()),
                *result.candidate_quarantines,
            )
        ]
        if coverage is not None:
            total_calls += coverage.model_calls
            total_tokens += coverage.estimated_input_tokens
        capacities = sorted(event.capacity_mw for event in events if event.capacity_mw is not None)
        price_admissions = sum(_price_admissible(event) for event in events)
        outcomes.append(
            {
                "case_id": case["case_id"],
                "expectation": case["expectation"],
                "expected_event_types": list(expected_types.elements()),
                "actual_event_types": list(actual_types.elements()),
                "expected_statuses": list(expected_statuses.elements()),
                "actual_statuses": list(actual_statuses.elements()),
                "expected_capacities_mw": sorted(case["expected_capacities_mw"]),
                "actual_capacities_mw": capacities,
                "coverage_complete": bool(coverage and coverage.complete),
                "quarantine_reason_codes": quarantine_reason_codes,
                "price_eligible_events": price_admissions,
                "latency_seconds": round(latency, 4),
                "passed": (
                    actual_types == expected_types
                    and actual_statuses == expected_statuses
                    and capacities == sorted(case["expected_capacities_mw"])
                    and bool(coverage and coverage.complete)
                    and price_admissions == case["expected_price_admissions"]
                    if case["expectation"] == "complete"
                    else price_admissions == 0
                ),
            }
        )
    precision = true_positive / actual_total if actual_total else 0.0
    recall = true_positive / gold_total if gold_total else 0.0
    return {
        "scorer_version": "strict-v2",
        "strategy": LONG_CONTEXT_STRATEGY,
        "corpus": str(path),
        "case_count": len(outcomes),
        "passed_cases": sum(item["passed"] for item in outcomes),
        "event_precision": round(precision, 4),
        "event_recall": round(recall, 4),
        "evidence_integrity": round(evidence_valid / evidence_total, 4) if evidence_total else 1.0,
        "unsafe_price_admissions": unsafe_price_admissions,
        "model_calls": total_calls,
        "estimated_input_tokens": total_tokens,
        "average_latency_seconds": round(mean(latencies), 4) if latencies else 0.0,
        "p95_latency_seconds": round(_percentile_95(latencies), 4),
        "output_tokens_available": False,
        "estimated_cost_available": False,
        "context_boundary_checks": run_context_boundary_checks(),
        "cases": outcomes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--live", action="store_true", help="Use the configured real model endpoint")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--unit-tokens", type=int, default=800)
    args = parser.parse_args()
    if args.live:
        from app.llm import build_model_gateway

        if args.repeats < 1:
            parser.error("--repeats must be positive")
        runs = [
            run_benchmark(
                args.cases,
                gateway_factory=lambda: FaultInjectingGateway(build_model_gateway()),
                unit_tokens=args.unit_tokens,
            )
            for _ in range(args.repeats)
        ]
        signatures = [
            [
                (
                    tuple(case["actual_event_types"]),
                    tuple(case["actual_statuses"]),
                    tuple(case["actual_capacities_mw"]),
                    case["coverage_complete"],
                    tuple(case["quarantine_reason_codes"]),
                    case["price_eligible_events"],
                )
                for case in run["cases"]
            ]
            for run in runs
        ]
        stable = sum(
            len({run_signature[index] for run_signature in signatures}) == 1
            for index in range(len(signatures[0]))
        )
        report = {
            "mode": "live",
            "scorer_version": "strict-v2",
            "strategy": LONG_CONTEXT_STRATEGY,
            "repeats": args.repeats,
            "stable_case_rate": round(stable / len(signatures[0]), 4),
            "average_passed_cases": round(
                sum(run["passed_cases"] for run in runs) / len(runs), 4
            ),
            "average_model_calls": round(
                sum(run["model_calls"] for run in runs) / len(runs), 4
            ),
            "average_estimated_input_tokens": round(
                sum(run["estimated_input_tokens"] for run in runs) / len(runs), 4
            ),
            "average_latency_seconds": round(
                mean(case["latency_seconds"] for run in runs for case in run["cases"]), 4
            ),
            "p95_latency_seconds": round(
                _percentile_95(
                    [case["latency_seconds"] for run in runs for case in run["cases"]]
                ),
                4,
            ),
            "output_tokens_available": False,
            "estimated_cost_available": False,
            "context_boundary_checks": runs[0]["context_boundary_checks"],
            "runs": runs,
        }
    else:
        report = run_benchmark(args.cases)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
