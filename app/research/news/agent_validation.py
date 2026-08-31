"""Qualification prompt and independent validation for a P2 Agent answer.

The deterministic news-price pipeline remains the source of facts.  This module only
asks a model to turn the evidence package into a bounded answer, then checks that answer
against the package and (optionally) a fixture answer key the model never receives.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.llm.gateway import ModelGateway, ModelMessage
from app.research.news.pipeline import NewsPriceStudy

P2_AGENT_GOAL_PROMPT_VERSION = "p2-agent-goal-v1"
P2_AGENT_EVALUATOR_VERSION = "p2-agent-evaluator-v1"
DEFAULT_FOCUS_WINDOW = "[0,1h]"

P2_AGENT_GOAL_SYSTEM_PROMPT = """角色
你是 P2 新闻—电价研究结果解释 Agent。确定性流水线已经完成计算，你只负责依据输入证据形成结构化结论；你不重新计算统计量，也不补写输入中不存在的事实。

可信来源与边界
- 只能使用 evidence_package 中的字段。goal_prompt 是任务要求，不是事实来源。
- focus_window 中每个 analysis_results 事件必须恰好输出一条 finding，不得漏报、重复或增加事件。
- excluded_events 与 events_not_analyzed 必须逐项保留，不能把它们改写成有效应。
- 数值、事件 ID、文档版本 ID、内容哈希、方法版本和五类指纹必须原样抄录。
- conclusion 为 not_supported_by_current_data 时，只能表达“当前数据未提供支持”，不能表达方向性影响。
- 本次证据只支持合成数据上的描述性关联，不支持因果、真实市场泛化、预测增量或交易建议。
- runtime_quality.passed 为 false 时 overall_verdict 必须是 blocked；为 true 时仍只能是 ready_with_caveats。

输出
只返回符合 Phase2GoalAgentResult schema 的对象。caveat_codes 必须完整包含 schema 允许的五项边界，summary 和 interpretation 使用简洁中文。"""

P2_AGENT_GOAL_USER_PROMPT = """目标模式任务：
基于本次 P2 新闻—电价证据包，判断在事件生效轴的 [0,1h] 窗口中，哪些新闻事件出现了与新闻所述方向一致的短期价格关联，哪些事件当前不受支持，以及哪些事件因规则或价格覆盖范围未进入分析。

完成条件：
1. 覆盖证据包中 [0,1h] 的全部分析结果，不多报也不少报；
2. 每条结论保留事件类型、方向、偏差、校正后 p 值、新闻版本、内容哈希和至少一段原文证据；
3. 保留全部排除项与未分析项；
4. 原样返回 as_of、市场时钟、方法版本和五类运行指纹；
5. 明确说明这是合成数据内部验证、关联不等于因果、尚未验证真实语料外部有效性、尚未验证预测增量，且未调用实时新闻 API。

禁止把相关性写成因果、预测能力或交易建议。"""

CaveatCode = Literal[
    "synthetic_only",
    "association_not_causation",
    "external_validity_unverified",
    "not_predictive_increment",
    "no_live_news_api",
]
_REQUIRED_CAVEATS: frozenset[CaveatCode] = frozenset(
    {
        "synthetic_only",
        "association_not_causation",
        "external_validity_unverified",
        "not_predictive_increment",
        "no_live_news_api",
    }
)
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_FORBIDDEN_CLAIMS = (
    "证明了因果",
    "证明存在因果",
    "导致了电价",
    "造成了电价",
    "能够预测未来电价",
    "可以预测未来电价",
    "提高电价预测",
    "建议买入",
    "建议卖出",
)


class AgentEvidenceQuote(BaseModel):
    """One exact source fragment cited by the Agent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    field_name: str = Field(min_length=1)
    quote: str = Field(min_length=1)


class AgentGoalFinding(BaseModel):
    """The Agent's bounded restatement of one deterministic window result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str = Field(pattern=r"^evt_[a-f0-9]{24}$")
    event_type: str = Field(min_length=1)
    direction: Literal["up", "down", "mixed", "unknown"]
    window_label: str = Field(min_length=1)
    conclusion: str = Field(min_length=1)
    deviation: float
    corrected_p_value: float = Field(ge=0, le=1)
    document_version_ids: tuple[str, ...] = Field(min_length=1)
    content_hashes: tuple[str, ...] = Field(min_length=1)
    evidence_quotes: tuple[AgentEvidenceQuote, ...] = Field(min_length=1)
    interpretation: str = Field(min_length=1)


class AgentGoalDisposition(BaseModel):
    """An event deliberately excluded from statistics or not analyzed by rule."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str = Field(pattern=r"^evt_[a-f0-9]{24}$")
    status: Literal["excluded", "not_analyzed"]
    reason: str = Field(min_length=1)


class Phase2GoalAgentResult(BaseModel):
    """Structured answer produced by the model for the fixed P2 goal."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt_version: Literal["p2-agent-goal-v1"]
    overall_verdict: Literal["ready_with_caveats", "blocked"]
    scope: Literal["synthetic_p2_internal_validity"]
    claim_boundary: Literal["descriptive_association_only"]
    as_of: datetime
    axis: Literal["effective"]
    market: str = Field(min_length=1)
    market_timezone: str = Field(min_length=1)
    interval_minutes: int = Field(gt=0)
    focus_window: Literal["[0,1h]"]
    method_version: str = Field(min_length=1)
    input_quality_passed: bool
    evidence_coverage: float = Field(ge=0, le=1)
    findings: tuple[AgentGoalFinding, ...]
    dispositions: tuple[AgentGoalDisposition, ...]
    analysis_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    event_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    feature_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    package_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    price_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    caveat_codes: tuple[CaveatCode, ...]
    caveats: tuple[str, ...] = Field(min_length=1)
    summary: str = Field(min_length=1)


class Phase2GoalAgentRun(BaseModel):
    """One inspectable model call, including the exact prompt and hashes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt_version: str
    goal_prompt: str
    input_payload: dict[str, Any]
    input_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    result: Phase2GoalAgentResult
    output_hash: str = Field(pattern=r"^[a-f0-9]{64}$")


class AgentValidationCheck(BaseModel):
    """One independent evaluation or underlying-data validation assertion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(min_length=1)
    group: Literal["evaluation", "validation"]
    severity: Literal["blocker", "warning"]
    passed: bool
    detail: str = Field(min_length=1)


class Phase2AgentValidation(BaseModel):
    """Shareability verdict for one P2 goal-mode Agent run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evaluator_version: str
    prompt_version: str
    checks: tuple[AgentValidationCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks if check.severity == "blocker")

    @property
    def overall_assessment(self) -> Literal["ready_to_share", "needs_revision"]:
        return "ready_to_share" if self.passed else "needs_revision"

    @property
    def failed_checks(self) -> tuple[AgentValidationCheck, ...]:
        return tuple(check for check in self.checks if not check.passed)


def build_phase2_goal_input(
    study: NewsPriceStudy,
    *,
    focus_window: str = DEFAULT_FOCUS_WINDOW,
) -> dict[str, Any]:
    """Build the model-visible evidence; fixture IDs and injected effects stay out."""

    events = {event.event_id: event for event in study.view.events}
    findings: list[dict[str, Any]] = []
    for link in study.package.evidence_links:
        if link.window_label != focus_window:
            continue
        event = events[link.event_id]
        findings.append(
            {
                "event_id": link.event_id,
                "event_type": link.event_type,
                "direction": event.direction,
                "window_label": link.window_label,
                "conclusion": link.conclusion,
                "conclusion_reason": link.conclusion_reason,
                "deviation": link.deviation,
                "corrected_p_value": link.corrected_p_value,
                "document_version_ids": list(link.document_version_ids),
                "content_hashes": list(link.content_hashes),
                "source_names": list(link.source_names),
                "extraction_traces": [
                    trace.model_dump(mode="json") for trace in link.extraction_traces
                ],
                "evidence_quotes": [
                    {"field_name": field_name, "quote": quote}
                    for field_name, quote in link.quotes
                ],
            }
        )

    quality = study.result_quality
    return {
        "scope": quality.scope,
        "as_of": study.as_of.isoformat(),
        "clock": {
            "market": study.clock.market,
            "timezone": study.clock.timezone,
            "interval_minutes": study.clock.interval_minutes,
        },
        "analysis": {
            "axis": study.analysis.axis,
            "focus_window": focus_window,
            "method_version": study.analysis.method.analysis_version,
            "results": findings,
            "excluded_events": [
                {"event_id": event_id, "reason": reason}
                for event_id, reason in study.analysis.excluded_events
            ],
            "events_not_analyzed": [
                {"event_id": event_id, "reason": reason}
                for event_id, reason in study.package.events_not_analyzed
            ],
        },
        "runtime_quality": {
            "passed": quality.passed,
            "evidence_coverage": study.package.quality.evidence_coverage,
            "checks": [
                {
                    "code": check.code,
                    "passed": check.passed,
                    "detail": check.detail,
                }
                for check in quality.checks
            ],
        },
        "fingerprints": study.package.fingerprint(),
        "method_notes": list(study.package.method_notes),
    }


def run_phase2_goal_agent(
    gateway: ModelGateway,
    study: NewsPriceStudy,
    *,
    goal_prompt: str = P2_AGENT_GOAL_USER_PROMPT,
) -> Phase2GoalAgentRun:
    """Call the configured model once and retain the exact evaluation inputs."""

    payload = build_phase2_goal_input(study)
    input_hash = _hash_json(payload)
    result = gateway.invoke_structured(
        messages=[
            ModelMessage(role="system", content=P2_AGENT_GOAL_SYSTEM_PROMPT),
            ModelMessage(
                role="user",
                content=json.dumps(
                    {"goal_prompt": goal_prompt, "evidence_package": payload},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            ),
        ],
        schema=Phase2GoalAgentResult,
    )
    output_hash = _hash_json(result.model_dump(mode="json"))
    return Phase2GoalAgentRun(
        prompt_version=P2_AGENT_GOAL_PROMPT_VERSION,
        goal_prompt=goal_prompt,
        input_payload=payload,
        input_hash=input_hash,
        result=result,
        output_hash=output_hash,
    )


def evaluate_phase2_goal_agent(
    study: NewsPriceStudy,
    run: Phase2GoalAgentRun,
    *,
    gold_effects: Mapping[str, float] | None = None,
) -> Phase2AgentValidation:
    """Independently evaluate the Agent answer and validate its source evidence."""

    result = run.result
    checks: list[AgentValidationCheck] = []

    def add(
        code: str,
        group: Literal["evaluation", "validation"],
        passed: bool,
        detail: str,
        *,
        severity: Literal["blocker", "warning"] = "blocker",
    ) -> None:
        checks.append(
            AgentValidationCheck(
                code=code,
                group=group,
                severity=severity,
                passed=passed,
                detail=detail,
            )
        )

    failed_source_checks = [check.code for check in study.result_quality.failed_checks]
    add(
        "source_runtime_quality",
        "validation",
        study.result_quality.passed,
        (
            "P2 运行时九项质量门禁全部通过"
            if not failed_source_checks
            else f"P2 运行时门禁失败：{', '.join(failed_source_checks)}"
        ),
    )
    _add_gold_checks(add, study, gold_effects)

    metadata_ok = (
        result.prompt_version == P2_AGENT_GOAL_PROMPT_VERSION
        and result.scope == study.result_quality.scope
        and result.as_of == study.as_of
        and result.axis == study.analysis.axis
        and result.market == study.clock.market
        and result.market_timezone == study.clock.timezone
        and result.interval_minutes == study.clock.interval_minutes
        and result.focus_window == DEFAULT_FOCUS_WINDOW
        and result.method_version == study.analysis.method.analysis_version
        and result.input_quality_passed == study.result_quality.passed
        and math.isclose(
            result.evidence_coverage,
            study.package.quality.evidence_coverage,
            abs_tol=1e-9,
        )
    )
    add(
        "agent_metadata_fidelity",
        "evaluation",
        metadata_ok,
        "Agent 必须原样保留 Prompt、时点、时间轴、市场、方法和输入质量口径",
    )

    expected_links = {
        link.event_id: link
        for link in study.package.evidence_links
        if link.window_label == DEFAULT_FOCUS_WINDOW
    }
    finding_ids = [finding.event_id for finding in result.findings]
    coverage_ok = (
        len(finding_ids) == len(set(finding_ids))
        and set(finding_ids) == set(expected_links)
    )
    add(
        "agent_finding_coverage",
        "evaluation",
        coverage_ok,
        (
            f"应覆盖 {len(expected_links)} 个一小时窗口事件，实际 {len(finding_ids)} 条，"
            "且不得重复或虚构事件"
        ),
    )

    events = {event.event_id: event for event in study.view.events}
    value_errors: list[str] = []
    evidence_errors: list[str] = []
    for finding in result.findings:
        link = expected_links.get(finding.event_id)
        event = events.get(finding.event_id)
        if link is None or event is None:
            continue
        if not (
            finding.event_type == link.event_type
            and finding.direction == event.direction
            and finding.window_label == link.window_label
            and finding.conclusion == link.conclusion
            and math.isclose(finding.deviation, link.deviation, abs_tol=1e-6)
            and math.isclose(
                finding.corrected_p_value,
                link.corrected_p_value,
                abs_tol=1e-6,
            )
        ):
            value_errors.append(finding.event_id)
        expected_quotes = set(link.quotes)
        cited_quotes = {
            (quote.field_name, quote.quote) for quote in finding.evidence_quotes
        }
        if not (
            tuple(finding.document_version_ids) == link.document_version_ids
            and tuple(finding.content_hashes) == link.content_hashes
            and cited_quotes
            and cited_quotes.issubset(expected_quotes)
        ):
            evidence_errors.append(finding.event_id)
    add(
        "agent_finding_values",
        "evaluation",
        not value_errors,
        (
            "事件类型、方向、结论、偏差和校正后 p 值与确定性结果一致"
            if not value_errors
            else f"字段或数值不一致：{', '.join(value_errors)}"
        ),
    )
    add(
        "agent_evidence_fidelity",
        "evaluation",
        not evidence_errors,
        (
            "每条结论保留完整文档版本、内容哈希和真实引文"
            if not evidence_errors
            else f"证据引用不一致：{', '.join(evidence_errors)}"
        ),
    )

    expected_dispositions = {
        event_id: "excluded" for event_id, _ in study.analysis.excluded_events
    }
    expected_dispositions.update(
        {event_id: "not_analyzed" for event_id, _ in study.package.events_not_analyzed}
    )
    actual_dispositions = {item.event_id: item.status for item in result.dispositions}
    disposition_ok = (
        len(result.dispositions) == len(actual_dispositions)
        and actual_dispositions == expected_dispositions
    )
    add(
        "agent_disposition_coverage",
        "evaluation",
        disposition_ok,
        "排除项和按规则未分析项必须完整保留且不得改写为有效应",
    )

    expected_fingerprints = study.package.fingerprint()
    actual_fingerprints = {
        "analysis_hash": result.analysis_hash,
        "event_hash": result.event_hash,
        "feature_hash": result.feature_hash,
        "package_hash": result.package_hash,
        "price_hash": result.price_hash,
    }
    fingerprints_ok = actual_fingerprints == expected_fingerprints and all(
        _SHA256.fullmatch(value) for value in actual_fingerprints.values()
    )
    add(
        "agent_fingerprint_fidelity",
        "evaluation",
        fingerprints_ok,
        "Agent 必须原样保留五类 SHA-256 指纹",
    )

    caveat_set = set(result.caveat_codes)
    caveats_ok = _REQUIRED_CAVEATS.issubset(caveat_set)
    add(
        "agent_required_caveats",
        "evaluation",
        caveats_ok,
        (
            "已声明合成、非因果、外部有效性、非预测增量和非实时 API 五项边界"
            if caveats_ok
            else f"缺少边界：{', '.join(sorted(_REQUIRED_CAVEATS - caveat_set))}"
        ),
    )

    narrative = "\n".join(
        [result.summary, *result.caveats, *(finding.interpretation for finding in result.findings)]
    )
    forbidden = [claim for claim in _FORBIDDEN_CLAIMS if claim in narrative]
    claim_boundary_ok = (
        result.claim_boundary == "descriptive_association_only" and not forbidden
    )
    add(
        "agent_claim_boundary",
        "evaluation",
        claim_boundary_ok,
        (
            "答案保持描述性关联边界"
            if not forbidden
            else f"发现越界措辞：{', '.join(forbidden)}"
        ),
    )

    expected_verdict = (
        "ready_with_caveats" if study.result_quality.passed else "blocked"
    )
    add(
        "agent_overall_verdict",
        "evaluation",
        result.overall_verdict == expected_verdict,
        f"源质量对应的期望判定为 {expected_verdict}",
    )

    return Phase2AgentValidation(
        evaluator_version=P2_AGENT_EVALUATOR_VERSION,
        prompt_version=P2_AGENT_GOAL_PROMPT_VERSION,
        checks=tuple(checks),
    )


def _add_gold_checks(
    add: Any,
    study: NewsPriceStudy,
    gold_effects: Mapping[str, float] | None,
) -> None:
    if gold_effects is None:
        add(
            "gold_effect_recovery",
            "validation",
            True,
            "本次未提供独立金标准；仅验证运行时质量与 Agent 忠实度",
            severity="warning",
        )
        return

    failures: list[str] = []
    for fixture_id, injected_delta in gold_effects.items():
        document = next(
            (
                item
                for item in study.documents
                if item.raw_metadata.get("fixture_id") == fixture_id
            ),
            None,
        )
        if document is None:
            failures.append(f"{fixture_id}:missing_document")
            continue
        event = next(
            (
                item
                for item in study.view.events
                if any(
                    ref.document_version_id == document.document_version_id
                    for ref in item.document_refs
                )
            ),
            None,
        )
        if event is None:
            failures.append(f"{fixture_id}:missing_event")
            continue
        result = next(
            (
                item
                for item in study.analysis.for_event(event.event_id)
                if item.metrics.window_label == DEFAULT_FOCUS_WINDOW
            ),
            None,
        )
        if result is None or not (
            result.conclusion == "association_consistent_with_expected_direction"
            and result.metrics.deviation * injected_delta > 0
            and math.isclose(
                result.metrics.deviation,
                injected_delta,
                rel_tol=0.35,
                abs_tol=1e-9,
            )
        ):
            failures.append(fixture_id)
    add(
        "gold_effect_recovery",
        "validation",
        not failures,
        (
            f"独立金标准中的 {len(gold_effects)} 个注入效应均恢复方向和量级"
            if not failures
            else f"金标准恢复失败：{', '.join(failures)}"
        ),
    )


def _hash_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
