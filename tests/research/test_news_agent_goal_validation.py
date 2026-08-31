"""Evaluation and validation tests for one fixed P2 goal-mode Agent run."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from app.research.news import (
    P2_AGENT_GOAL_PROMPT_VERSION,
    P2_AGENT_GOAL_USER_PROMPT,
    AgentGoalDisposition,
    AgentGoalFinding,
    JsonlCollectedNewsAdapter,
    MarketClock,
    NewsPriceStudy,
    Phase2GoalAgentResult,
    build_phase2_goal_input,
    evaluate_phase2_goal_agent,
    load_price_csv,
    run_news_price_study,
    run_phase2_goal_agent,
)
from app.research.news.agent_validation import AgentEvidenceQuote

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "news_price"
CLOCK = MarketClock(market="TEST_MARKET", timezone="UTC", interval_minutes=30)
FINAL_AS_OF = datetime.fromisoformat("2026-03-01T23:30:00+00:00")


@pytest.fixture(scope="module")
def study() -> NewsPriceStudy:
    prices = load_price_csv(FIXTURES / "synthetic_prices.csv", CLOCK)
    return run_news_price_study(
        adapter=JsonlCollectedNewsAdapter(FIXTURES / "synthetic_news.jsonl"),
        prices=prices,
        as_of=FINAL_AS_OF,
        axis="effective",
    )


@pytest.fixture(scope="module")
def gold_effects() -> dict[str, float]:
    manifest = yaml.safe_load(
        (FIXTURES / "fixture_manifest.yaml").read_text(encoding="utf-8")
    )
    return {
        item["fixture_id"]: float(item["delta"])
        for item in manifest["effects"]
    }


def test_goal_prompt_and_model_payload_do_not_expose_the_fixture_answer_key(
    study: NewsPriceStudy,
) -> None:
    payload = build_phase2_goal_input(study)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)

    assert "目标模式任务" in P2_AGENT_GOAL_USER_PROMPT
    assert "fixture_id" not in encoded
    assert "injected_delta" not in encoded
    assert "expected_outcomes" not in encoded
    assert "20260112" not in encoded


def test_faithful_agent_answer_passes_independent_evaluation_and_gold_validation(
    study: NewsPriceStudy,
    gold_effects: dict[str, float],
) -> None:
    gateway = _Gateway(_perfect_result(study))
    run = run_phase2_goal_agent(gateway, study)
    validation = evaluate_phase2_goal_agent(
        study,
        run,
        gold_effects=gold_effects,
    )

    assert validation.passed
    assert validation.overall_assessment == "ready_to_share"
    assert validation.failed_checks == ()
    assert {check.group for check in validation.checks} == {
        "evaluation",
        "validation",
    }
    assert gateway.schema is Phase2GoalAgentResult
    assert len(gateway.messages) == 2
    assert run.input_hash != run.output_hash


@pytest.mark.parametrize(
    ("mutation", "failed_code"),
    [
        ("missing_finding", "agent_finding_coverage"),
        ("wrong_value", "agent_finding_values"),
        ("invented_quote", "agent_evidence_fidelity"),
        ("wrong_fingerprint", "agent_fingerprint_fidelity"),
        ("missing_caveat", "agent_required_caveats"),
        ("causal_claim", "agent_claim_boundary"),
    ],
)
def test_evaluator_rejects_material_agent_answer_defects(
    study: NewsPriceStudy,
    gold_effects: dict[str, float],
    mutation: str,
    failed_code: str,
) -> None:
    answer = _mutate(_perfect_result(study), mutation)
    run = run_phase2_goal_agent(_Gateway(answer), study)
    validation = evaluate_phase2_goal_agent(
        study,
        run,
        gold_effects=gold_effects,
    )

    assert not validation.passed
    assert validation.overall_assessment == "needs_revision"
    assert failed_code in {check.code for check in validation.failed_checks}


def test_same_evidence_builds_the_same_agent_input_hash(study: NewsPriceStudy) -> None:
    answer = _perfect_result(study)

    first = run_phase2_goal_agent(_Gateway(answer), study)
    second = run_phase2_goal_agent(_Gateway(answer), study)

    assert first.input_hash == second.input_hash
    assert first.output_hash == second.output_hash


class _Gateway:
    def __init__(self, result: Phase2GoalAgentResult) -> None:
        self.result = result
        self.messages = []
        self.schema = None

    @property
    def enabled(self) -> bool:
        return True

    @property
    def model_name(self) -> str:
        return "deterministic-goal-agent"

    def invoke_structured(self, *, messages, schema):
        self.messages = messages
        self.schema = schema
        return self.result

    def invoke_text(self, *, messages):  # pragma: no cover - protocol completeness
        raise AssertionError(messages)

    def invoke_tool_calls(self, *, messages, tools):  # pragma: no cover
        raise AssertionError((messages, tools))


def _perfect_result(study: NewsPriceStudy) -> Phase2GoalAgentResult:
    events = {event.event_id: event for event in study.view.events}
    findings = []
    for link in study.package.evidence_links:
        if link.window_label != "[0,1h]":
            continue
        event = events[link.event_id]
        findings.append(
            AgentGoalFinding(
                event_id=link.event_id,
                event_type=link.event_type,
                direction=event.direction,
                window_label=link.window_label,
                conclusion=link.conclusion,
                deviation=link.deviation,
                corrected_p_value=link.corrected_p_value,
                document_version_ids=link.document_version_ids,
                content_hashes=link.content_hashes,
                evidence_quotes=(
                    AgentEvidenceQuote(
                        field_name=link.quotes[0][0],
                        quote=link.quotes[0][1],
                    ),
                ),
                interpretation=(
                    "该窗口与新闻方向一致，但仅是描述性关联。"
                    if link.conclusion == "association_consistent_with_expected_direction"
                    else "当前数据未提供该方向短期关联的支持。"
                ),
            )
        )
    dispositions = tuple(
        AgentGoalDisposition(event_id=event_id, status="excluded", reason=reason)
        for event_id, reason in study.analysis.excluded_events
    ) + tuple(
        AgentGoalDisposition(
            event_id=event_id,
            status="not_analyzed",
            reason=reason,
        )
        for event_id, reason in study.package.events_not_analyzed
    )
    fingerprints = study.package.fingerprint()
    return Phase2GoalAgentResult(
        prompt_version=P2_AGENT_GOAL_PROMPT_VERSION,
        overall_verdict="ready_with_caveats",
        scope="synthetic_p2_internal_validity",
        claim_boundary="descriptive_association_only",
        as_of=study.as_of,
        axis="effective",
        market=study.clock.market,
        market_timezone=study.clock.timezone,
        interval_minutes=study.clock.interval_minutes,
        focus_window="[0,1h]",
        method_version=study.analysis.method.analysis_version,
        input_quality_passed=study.result_quality.passed,
        evidence_coverage=study.package.quality.evidence_coverage,
        findings=tuple(findings),
        dispositions=dispositions,
        analysis_hash=fingerprints["analysis_hash"],
        event_hash=fingerprints["event_hash"],
        feature_hash=fingerprints["feature_hash"],
        package_hash=fingerprints["package_hash"],
        price_hash=fingerprints["price_hash"],
        caveat_codes=(
            "synthetic_only",
            "association_not_causation",
            "external_validity_unverified",
            "not_predictive_increment",
            "no_live_news_api",
        ),
        caveats=(
            "仅使用合成新闻与合成价格。",
            "事件关联不等于因果效应。",
            "真实语料和真实市场的外部有效性尚未验证。",
            "本结果没有验证新闻特征的样本外预测增量。",
            "本次运行读取冻结快照，没有调用实时新闻 API。",
        ),
        summary="四类注入事件方向一致，一个恢复事件当前不受支持；结论仅限合成数据关联。",
    )


def _mutate(answer: Phase2GoalAgentResult, mutation: str) -> Phase2GoalAgentResult:
    if mutation == "missing_finding":
        return answer.model_copy(update={"findings": answer.findings[1:]})
    if mutation == "wrong_value":
        changed = answer.findings[0].model_copy(
            update={"deviation": -answer.findings[0].deviation}
        )
        return answer.model_copy(update={"findings": (changed, *answer.findings[1:])})
    if mutation == "invented_quote":
        changed = answer.findings[0].model_copy(
            update={
                "evidence_quotes": (
                    AgentEvidenceQuote(
                        field_name="event_type",
                        quote="这段话不在任何新闻原文中",
                    ),
                )
            }
        )
        return answer.model_copy(update={"findings": (changed, *answer.findings[1:])})
    if mutation == "wrong_fingerprint":
        return answer.model_copy(update={"analysis_hash": "0" * 64})
    if mutation == "missing_caveat":
        return answer.model_copy(
            update={
                "caveat_codes": tuple(
                    code
                    for code in answer.caveat_codes
                    if code != "external_validity_unverified"
                )
            }
        )
    if mutation == "causal_claim":
        return answer.model_copy(update={"summary": "这些新闻事件导致了电价变化。"})
    raise AssertionError(mutation)
