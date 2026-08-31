"""Post-execution quality assessment for a P2 news-price study result."""

from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass

from app.research.news.contracts import NewsDocument
from app.research.news.evidence import ResearchPackage
from app.research.news.prices import PriceObservations

QUALITY_ASSESSMENT_VERSION = "1.0.0"
_SHA256 = re.compile(r"^[a-f0-9]{64}$")


@dataclass(frozen=True)
class ResultQualityCheck:
    """One decision-relevant assertion about a completed study result."""

    code: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class ResultQualityAssessment:
    """A compact, inspectable verdict; passing means synthetic evidence is internally sound."""

    assessment_version: str
    scope: str
    checks: tuple[ResultQualityCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failed_checks(self) -> tuple[ResultQualityCheck, ...]:
        return tuple(check for check in self.checks if not check.passed)


def evaluate_result_quality(
    *,
    package: ResearchPackage,
    documents: tuple[NewsDocument, ...],
    prices: PriceObservations,
) -> ResultQualityAssessment:
    """Validate data, calculations, evidence and narrative inputs after execution."""

    analysis_clock = package.analysis.clock
    features = package.features
    clock_ok = features is not None and (
        features.market,
        features.market_timezone,
        features.interval_minutes,
    ) == (analysis_clock.market, analysis_clock.timezone, analysis_clock.interval_minutes)

    wrong_market = tuple(
        document.source_document_id
        for document in documents
        if analysis_clock.market not in document.market_tags
    )

    price_grid_ok = all(
        current - previous == prices.clock.interval
        for previous, current in itertools.pairwise(prices.timestamps)
    ) and all(prices.clock.floor(moment) == moment for moment in prices.timestamps)
    finite_prices = all(math.isfinite(value) for value in prices.prices)

    event_ids = [event.event_id for event in package.events]
    unique_events = len(event_ids) == len(set(event_ids))

    invalid_statistics = tuple(
        result.event_id
        for result in package.analysis.results
        if not (
            math.isfinite(result.metrics.deviation)
            and math.isfinite(result.permutation_p_value)
            and math.isfinite(result.corrected_p_value)
            and 0 <= result.permutation_p_value <= 1
            and result.permutation_p_value <= result.corrected_p_value <= 1
            and result.control_sample_size >= 0
        )
    )

    untraceable = package.untraceable_links()
    evidence_ok = not package.events or (
        package.quality.evidence_coverage == 1.0 and not untraceable
    )

    events_by_id = {event.event_id: event for event in package.events}
    leaked_rows = 0
    unknown_feature_events = 0
    if features is not None:
        for row in features.rows:
            for event_id in row.source_event_ids:
                event = events_by_id.get(event_id)
                if event is None:
                    unknown_feature_events += 1
                elif event.announcement_available_at > row.interval_start:
                    leaked_rows += 1

    fingerprints = package.fingerprint()
    fingerprints_ok = bool(fingerprints) and all(_SHA256.fullmatch(value) for value in fingerprints.values())

    checks = (
        ResultQualityCheck(
            code="single_market_clock",
            passed=clock_ok and analysis_clock == prices.clock,
            detail="分析、特征和价格必须使用同一市场、时区与结算间隔",
        ),
        ResultQualityCheck(
            code="document_market_coverage",
            passed=not wrong_market,
            detail=("所有新闻都属于价格市场" if not wrong_market else f"市场不匹配：{', '.join(wrong_market)}"),
        ),
        ResultQualityCheck(
            code="price_grid_and_values",
            passed=price_grid_ok and finite_prices,
            detail="价格必须连续落格且全部为有限数值",
        ),
        ResultQualityCheck(
            code="unique_event_identity",
            passed=unique_events,
            detail="as_of 事件视图不得包含重复 event_id",
        ),
        ResultQualityCheck(
            code="valid_statistics",
            passed=not invalid_statistics,
            detail=("统计量与 p 值合法" if not invalid_statistics else f"非法结果：{', '.join(invalid_statistics)}"),
        ),
        ResultQualityCheck(
            code="complete_evidence_chain",
            passed=evidence_ok,
            detail=(
                "所有结论都有关键字段原文证据"
                if evidence_ok
                else f"不可追溯结论 {len(untraceable)} 条，覆盖率 {package.quality.evidence_coverage:.1%}"
            ),
        ),
        ResultQualityCheck(
            code="feature_availability",
            passed=leaked_rows == 0 and unknown_feature_events == 0,
            detail=f"泄漏行 {leaked_rows}，未知事件引用 {unknown_feature_events}",
        ),
        ResultQualityCheck(
            code="complete_event_accounting",
            passed=not package.unexplained_events,
            detail=(
                "每个事件都有结果或排除原因"
                if not package.unexplained_events
                else f"未解释事件：{', '.join(package.unexplained_events)}"
            ),
        ),
        ResultQualityCheck(
            code="reproducible_fingerprints",
            passed=fingerprints_ok,
            detail="分析、事件、特征、价格和证据包均须有 SHA-256 指纹",
        ),
    )
    return ResultQualityAssessment(
        assessment_version=QUALITY_ASSESSMENT_VERSION,
        scope="synthetic_p2_internal_validity",
        checks=checks,
    )
