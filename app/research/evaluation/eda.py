"""Deterministic evaluation of evidence produced by an approved EDA plan."""

from __future__ import annotations

import hashlib
import itertools
import json
from typing import Any

from app.research.agent.schemas import AgendaScope, AgentEvaluation, EDAPlan, EvaluationCheck, HypothesisAssessment
from app.research.evaluation.agenda import resolve_agenda_item
from app.research.evaluation.criteria import (
    CALENDAR_EFFECT_THRESHOLD,
    DISTRIBUTION_KURTOSIS_THRESHOLD,
    DISTRIBUTION_SKEW_THRESHOLD,
    DRIVER_DRIFT_STANDARD_DEVIATIONS,
    MINIMUM_DRIVER_COVERAGE,
    SEASONAL_STRENGTH_THRESHOLD,
    VOLATILITY_REGIME_RATIO,
    group_variance_share,
)
from app.research.evaluation.issues import issue_line
from app.research.evaluation.wording import GROUP_LABELS, stationarity_text, transform_text
from app.research.planning.variables import build_variable_recommendations
from app.research.schemas.feedback import FeedbackPacket
from app.research.schemas.results import DataQualityReport
from app.research.tools.catalog import FUNCTION_CATALOG


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _share(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.2f}%"


def _segment_pair_label(comparison: dict[str, Any], contrast: dict[str, Any]) -> str:
    segments = comparison.get("segments", {})
    left_id = str(contrast.get("left_segment_id") or "")
    right_id = str(contrast.get("right_segment_id") or "")
    left = str((segments.get(left_id) or {}).get("label") or left_id)
    right = str((segments.get(right_id) or {}).get("label") or right_id)
    return f"{left}与{right}"


def _truncated_lag_scans(relationships: dict[str, Any]) -> tuple[list[str], int | None]:
    """Find variables whose lag scan ran past the point where pairs remain, and the last usable lag."""

    truncated: list[str] = []
    usable_ceiling: int | None = None
    for name, result in relationships.items():
        profile = result.get("lag_profile") or []
        usable = [int(row["lag"]) for row in profile if row.get("correlation") is not None]
        starved = [row for row in profile if row.get("correlation") is None]
        if not usable or not starved:
            continue
        truncated.append(name)
        ceiling = max(usable)
        usable_ceiling = ceiling if usable_ceiling is None else min(usable_ceiling, ceiling)
    return truncated, usable_ceiling


def _variables_without_relationship_evidence(relationships: dict[str, Any]) -> list[str]:
    """Find selected variables that produced no usable correlation at any lag."""

    empty: list[str] = []
    for name, result in relationships.items():
        contemporaneous = result.get("contemporaneous") or {}
        if any(item.get("correlation") is not None for item in contemporaneous.values()):
            continue
        profile = result.get("lag_profile") or []
        if any(row.get("correlation") is not None for row in profile):
            continue
        if not contemporaneous and not profile:
            continue
        empty.append(name)
    return empty


def _drifting_variables(series: dict[str, Any]) -> list[str] | None:
    """Return drivers whose whole-sample drift exceeds one of their own standard deviations."""

    drifting: list[str] = []
    judged = False
    for name, item in series.items():
        slope = item.get("trend_slope_per_interval")
        observations = item.get("observations")
        dispersion = item.get("std")
        if slope is None or not observations or not dispersion:
            continue
        judged = True
        if (
            abs(float(slope)) * int(observations)
            >= DRIVER_DRIFT_STANDARD_DEVIATIONS * float(dispersion)
        ):
            drifting.append(name)
    return drifting if judged else None


def _is_monotonic(rows: list[dict[str, Any]]) -> bool:
    """Check whether bucketed target means move in one direction across feature quartiles."""

    means = [row["target_mean"] for row in rows if row.get("target_mean") is not None]
    if len(means) < 3:
        return True
    steps = list(itertools.pairwise(means))
    ascending = all(later >= earlier for earlier, later in steps)
    descending = all(later <= earlier for earlier, later in steps)
    return ascending or descending


def _missing_scope(
    required_functions: tuple[str, ...],
    enabled_functions: set[str],
    *,
    approval_message: str,
    data_message: str,
) -> tuple[AgendaScope, str]:
    if not any(name in enabled_functions for name in required_functions):
        return "needs_approval", approval_message
    return "needs_data", data_message


def _grouped_relationship_assessment(
    relationships: dict[str, Any],
    enabled_functions: set[str],
    *,
    group_key: str,
    item_id: str,
    function_name: str,
    label: str,
) -> tuple[str, str, str, AgendaScope, str | None]:
    """Compare one grouped relationship view across its own groups."""

    spreads: list[float] = []
    for result in relationships.get("series", {}).values():
        rows = [row for row in result.get(group_key, []) if row.get("correlation") is not None]
        if len(rows) >= 2:
            values = [float(row["correlation"]) for row in rows]
            spreads.append(max(values) - min(values))
    if not spreads:
        scope, remediation = _missing_scope(
            (function_name,),
            enabled_functions,
            approval_message=f"需要批准加入分{label}关系函数后才能验证该假设。",
            data_message=f"已启用分{label}关系函数，但分组样本不足以形成可比证据。",
        )
        return item_id, "not_tested", f"本轮没有可比较的分{label}关系证据。", scope, remediation
    widest = max(spreads)
    status = "candidate_support" if widest >= 0.2 else "not_supported"
    return (
        item_id,
        status,
        f"分{label}相关系数的最大组间跨度为 {_fmt(widest)}。",
        "inherent",
        None,
    )


def _driver_stationarity_assessment(
    exogenous: dict[str, Any],
    enabled_functions: set[str],
) -> tuple[str, str, str, AgendaScope, str | None]:
    """Judge driver stationarity from the drivers' own tests, never from the target's."""

    driver_stationarity = exogenous.get("driver_stationarity")
    if not driver_stationarity:
        scope, remediation = _missing_scope(
            ("exogenous_stationarity_tests",),
            enabled_functions,
            approval_message="需要批准加入驱动平稳性检验后才能验证该假设。",
            data_message="已启用驱动平稳性检验，但有效样本不足。",
        )
        return "exogenous.stationarity", "not_tested", "本轮未执行驱动平稳性检验。", scope, remediation
    non_stationary = driver_stationarity.get("non_stationary_variables") or []
    status = "candidate_support" if non_stationary else "not_supported"
    return (
        "exogenous.stationarity",
        status,
        f"{len(non_stationary)} 个驱动变量被判定为非平稳。",
        "inherent",
        None,
    )

def _advanced_assessment(
    item_id: str,
    *,
    price: dict[str, Any],
    exogenous: dict[str, Any],
    relationships: dict[str, Any],
    enabled_functions: set[str],
) -> tuple[str, str, str, AgendaScope, str | None] | None:
    """Assess one agenda item using the evidence its own function produced."""

    if item_id == "relationships.by_hour":
        return _grouped_relationship_assessment(
            relationships,
            enabled_functions,
            group_key="by_hour",
            item_id="relationships.by_hour",
            function_name="relationship_pearson_by_hour",
            label="交割小时",
        )
    if item_id == "relationships.by_month":
        return _grouped_relationship_assessment(
            relationships,
            enabled_functions,
            group_key="by_month",
            item_id="relationships.by_month",
            function_name="relationship_pearson_by_month",
            label="月份",
        )
    if item_id == "price.distribution":
        distribution = price.get("distribution")
        if not distribution:
            scope, remediation = _missing_scope(
                ("price_descriptive_distribution",),
                enabled_functions,
                approval_message="需要批准加入电价分布画像后才能验证该假设。",
                data_message="已启用电价分布画像，但当前没有可用分布证据。",
            )
            return "price.distribution", "not_tested", "本轮未执行电价分布画像。", scope, remediation
        skewness = distribution.get("skewness")
        kurtosis = distribution.get("kurtosis")
        skewed = skewness is not None and abs(float(skewness)) >= DISTRIBUTION_SKEW_THRESHOLD
        heavy = kurtosis is not None and float(kurtosis) >= DISTRIBUTION_KURTOSIS_THRESHOLD
        return (
            "price.distribution",
            "candidate_support" if skewed or heavy else "not_supported",
            f"偏度为 {_fmt(skewness)}，超额峰度为 {_fmt(kurtosis)}。",
            "inherent",
            None,
        )
    if item_id == "price.volatility":
        volatility = price.get("volatility")
        rolling = price.get("rolling_statistics") or {}
        one_day = rolling.get("one_day") or {}
        if not volatility or one_day.get("maximum_std") is None:
            scope, remediation = _missing_scope(
                ("price_rolling_mean_std",),
                enabled_functions,
                approval_message="需要批准加入电价滚动波动函数后才能验证该假设。",
                data_message="已启用滚动波动函数，但可用滚动窗口不足。",
            )
            return "price.volatility", "not_tested", "本轮未执行电价滚动波动分析。", scope, remediation
        lowest = one_day.get("minimum_std")
        if lowest is None:
            return (
                "price.volatility",
                "inconclusive",
                "滚动窗口没有给出可比较的最低波动水平。",
                "needs_data",
                "需要更长或缺口更少的目标序列才能比较波动状态。",
            )
        if float(lowest) == 0.0:
            # A flat day (price held at a cap, floor, or a constant fill) makes the
            # ratio undefined, and is itself evidence that the swings are not uniform.
            return (
                "price.volatility",
                "candidate_support",
                "存在整整一天价格几乎没有变化的时段，而另一些时段波动很大，说明波动不是全期稳定的。",
                "inherent",
                None,
            )
        ratio = float(one_day["maximum_std"]) / float(lowest)
        return (
            "price.volatility",
            "candidate_support" if ratio >= VOLATILITY_REGIME_RATIO else "not_supported",
            f"一天窗口内最高与最低滚动标准差之比为 {ratio:.2f}。",
            "inherent",
            None,
        )
    if item_id == "exogenous.coverage":
        profiles = exogenous.get("series") or {}
        if not profiles:
            scope, remediation = _missing_scope(
                ("exogenous_descriptive_distribution",),
                enabled_functions,
                approval_message="需要批准加入影响因素分布画像后才能验证该假设。",
                data_message="已启用影响因素分布画像，但当前没有可用画像证据。",
            )
            return "exogenous.coverage", "not_tested", "本轮未执行影响因素分布画像。", scope, remediation
        sparse = [
            name
            for name, item in profiles.items()
            if item.get("coverage_rate") is not None
            and float(item["coverage_rate"]) < MINIMUM_DRIVER_COVERAGE
        ]
        return (
            "exogenous.coverage",
            "candidate_support" if sparse else "not_supported",
            (
                f"{'、'.join(sparse)} 的覆盖率低于 {MINIMUM_DRIVER_COVERAGE:.0%}。"
                if sparse
                else f"全部 {len(profiles)} 个变量的覆盖率都不低于 {MINIMUM_DRIVER_COVERAGE:.0%}。"
            ),
            "inherent",
            None,
        )
    if item_id == "exogenous.trend":
        trends = exogenous.get("series") or {}
        drifting = _drifting_variables(trends)
        if drifting is None:
            scope, remediation = _missing_scope(
                ("exogenous_linear_index_trend", "exogenous_descriptive_distribution"),
                enabled_functions,
                approval_message="需要批准加入影响因素漂移与分布画像后才能验证该假设。",
                data_message="已启用漂移分析，但缺少判断漂移量级所需的离散程度。",
            )
            return "exogenous.trend", "not_tested", "本轮没有可判断漂移量级的证据。", scope, remediation
        return (
            "exogenous.trend",
            "candidate_support" if drifting else "not_supported",
            (
                f"{'、'.join(drifting)} 的全期漂移超过自身一个标准差。"
                if drifting
                else "各变量的全期漂移都在自身一个标准差以内。"
            ),
            "inherent",
            None,
        )
    if item_id == "relationships.quantile_response":
        responses = {
            name: result["feature_quantile_response"]
            for name, result in relationships.get("series", {}).items()
            if result.get("feature_quantile_response")
        }
        if not responses:
            scope, remediation = _missing_scope(
                ("relationship_feature_quartile_response",),
                enabled_functions,
                approval_message="需要批准加入分位响应函数后才能验证该假设。",
                data_message="已启用分位响应函数，但分位样本不足以形成响应曲线。",
            )
            return "relationships.quantile_response", "not_tested", "本轮没有可用分位响应证据。", scope, remediation
        non_monotonic = [name for name, rows in responses.items() if not _is_monotonic(rows)]
        return (
            "relationships.quantile_response",
            "candidate_support" if non_monotonic else "not_supported",
            (
                f"{'、'.join(non_monotonic)} 的分位响应不是单调的。"
                if non_monotonic
                else f"全部 {len(responses)} 个变量的分位响应保持单调。"
            ),
            "inherent",
            None,
        )
    if item_id == "exogenous.stationarity":
        return _driver_stationarity_assessment(exogenous, enabled_functions)
    if item_id == "price.stationarity":
        stationarity = price.get("stationarity")
        if not stationarity:
            scope, remediation = _missing_scope(
                ("price_stationarity_tests",),
                enabled_functions,
                approval_message="需要批准加入电价平稳性检验后才能验证该假设。",
                data_message="已启用平稳性检验，但有效样本不足以完成 ADF/KPSS。",
            )
            return "price.stationarity", "not_tested", "本轮未执行平稳性检验。", scope, remediation
        verdict = str(stationarity.get("verdict", "inconclusive"))
        status = "candidate_support" if verdict in {"unit_root", "trend_or_break_suspected"} else "not_supported"
        scope = "inherent"
        remediation = None
        if verdict == "inconclusive":
            status, scope = "inconclusive", "needs_data"
            remediation = "ADF 与 KPSS 均未给出明确结论，需要更长样本或噪声更低的目标序列后重新检验。"
        return (
            "price.stationarity",
            status,
            (
                f"平稳性检验的结论是{stationarity_text(verdict)}；"
                f"{transform_text(stationarity.get('recommended_transform'))}。"
            ),
            scope,
            remediation,
        )
    if item_id == "price.decomposition":
        decomposition = price.get("decomposition")
        if not decomposition:
            scope, remediation = _missing_scope(
                ("price_seasonal_decomposition",),
                enabled_functions,
                approval_message="需要批准加入趋势季节分解后才能验证该假设。",
                data_message="已启用趋势季节分解，但样本不足两个完整日周期。",
            )
            return "price.decomposition", "not_tested", "本轮未执行趋势季节分解。", scope, remediation
        strengths = [value for value in (decomposition.get("seasonal_strength") or {}).values() if value is not None]
        strongest = max(strengths, default=None)
        status = "candidate_support" if strongest is not None and strongest >= 0.3 else "not_supported"
        return (
            "price.decomposition",
            status,
            (
                f"最强季节成分强度为 {_fmt(strongest)}，残差方差占比 "
                f"{_share((decomposition.get('variance_share') or {}).get('remainder'))}。"
            ),
            "inherent",
            None,
        )
    if item_id == "price.partial_autocorrelation":
        memory = price.get("partial_autocorrelation")
        if not memory:
            scope, remediation = _missing_scope(
                ("price_partial_autocorrelation",),
                enabled_functions,
                approval_message="需要批准加入偏自相关分析后才能验证该假设。",
                data_message="已启用偏自相关分析，但有效样本不足。",
            )
            return "price.partial_autocorrelation", "not_tested", "本轮未执行偏自相关分析。", scope, remediation
        order = int(memory.get("suggested_autoregressive_order", 0))
        status = "candidate_support" if order > 0 else "not_supported"
        return (
            "price.partial_autocorrelation",
            status,
            f"显著偏自相关滞后共 {memory.get('significant_lag_count', 0)} 个，最大显著滞后为 {order}。",
            "inherent",
            None,
        )
    if item_id == "price.spike_regime":
        regime = price.get("spike_regime")
        if not regime:
            scope, remediation = _missing_scope(
                ("price_spike_regime_profile",),
                enabled_functions,
                approval_message="需要批准加入尖峰状态画像后才能验证该假设。",
                data_message="已启用尖峰状态画像，但当前没有可用状态证据。",
            )
            return "price.spike_regime", "not_tested", "本轮未执行尖峰状态画像。", scope, remediation
        ratio = (regime.get("clustering") or {}).get("persistence_ratio")
        status = "candidate_support" if ratio is not None and ratio > 1.5 else "not_supported"
        return (
            "price.spike_regime",
            status,
            (
                f"尖峰占比 {_share(regime.get('high_spike_share'))}，尖峰后仍是尖峰的概率是随机水平的 {_fmt(ratio)} 倍，"
                f"最长尖峰段 {regime.get('high_episodes', {}).get('max_duration_intervals', 0)} 个间隔。"
            ),
            "inherent",
            None,
        )
    if item_id == "price.duration_curve":
        curve = price.get("duration_curve")
        if not curve:
            scope, remediation = _missing_scope(
                ("price_duration_curve",),
                enabled_functions,
                approval_message="需要批准加入电价持续曲线后才能验证该假设。",
                data_message="已启用电价持续曲线，但当前没有可用证据。",
            )
            return "price.duration_curve", "not_tested", "本轮未执行电价持续曲线。", scope, remediation
        concentration = curve.get("top_5_percent_share_of_positive_value")
        status = "candidate_support" if concentration is not None and concentration >= 0.15 else "not_supported"
        return (
            "price.duration_curve",
            status,
            (
                f"最高 5% 时段占正价格总量的 {_share(concentration)}，高于均值的时段占比 "
                f"{_share(curve.get('share_above_mean'))}。"
            ),
            "inherent",
            None,
        )
    if item_id == "price.variance_stabilization":
        stabilization = price.get("variance_stabilization")
        if not stabilization:
            scope, remediation = _missing_scope(
                ("price_variance_stabilization_check",),
                enabled_functions,
                approval_message="需要批准加入方差稳定检查后才能验证该假设。",
                data_message="已启用方差稳定检查，但当前没有可用证据。",
            )
            return "price.variance_stabilization", "not_tested", "本轮未执行方差稳定检查。", scope, remediation
        recommended = str(stabilization.get("recommended_transform", "none"))
        status = "candidate_support" if recommended != "none" else "not_supported"
        return (
            "price.variance_stabilization",
            status,
            (
                "稳健标准化后超额峰度 "
                f"{_fmt((stabilization.get('raw_standardized') or {}).get('excess_kurtosis'))}，"
                f"建议变换：{recommended}。"
            ),
            "inherent",
            None,
        )
    if item_id == "price.naive_baselines":
        baselines = price.get("naive_baselines")
        if not baselines:
            scope, remediation = _missing_scope(
                ("price_naive_baseline_benchmark",),
                enabled_functions,
                approval_message="需要批准加入朴素基线基准后才能验证该假设。",
                data_message="已启用朴素基线基准，但有效样本不足。",
            )
            return "price.naive_baselines", "not_tested", "本轮未执行朴素基线基准。", scope, remediation
        return (
            "price.naive_baselines",
            "candidate_support",
            (
                f"误差最低的朴素基线是{baselines.get('best_baseline_label')}，"
                f"平均绝对误差 {_fmt(baselines.get('best_mae'))}。"
            ),
            "inherent",
            None,
        )
    if item_id == "exogenous.multicollinearity":
        multicollinearity = exogenous.get("multicollinearity")
        if not multicollinearity:
            scope, remediation = _missing_scope(
                ("exogenous_variance_inflation",),
                enabled_functions,
                approval_message="需要批准加入 VIF 共线性分析后才能验证该假设。",
                data_message="已启用 VIF 共线性分析，但完整样本不足。",
            )
            return "exogenous.multicollinearity", "not_tested", "本轮未执行 VIF 共线性分析。", scope, remediation
        severe = multicollinearity.get("severe_variables") or []
        moderate = multicollinearity.get("moderate_variables") or []
        status = "candidate_support" if severe or moderate else "not_supported"
        return (
            "exogenous.multicollinearity",
            status,
            f"严重冗余 {len(severe)} 个、中度冗余 {len(moderate)} 个变量。",
            "inherent",
            None,
        )
    if item_id == "relationships.mutual_information":
        information = relationships.get("mutual_information")
        if not information:
            scope, remediation = _missing_scope(
                ("relationship_mutual_information_scan",),
                enabled_functions,
                approval_message="需要批准加入互信息扫描后才能验证该假设。",
                data_message="已启用互信息扫描，但成对有效样本不足。",
            )
            return "relationships.mutual_information", "not_tested", "本轮未执行互信息扫描。", scope, remediation
        candidates = information.get("nonlinear_candidates") or []
        status = "candidate_support" if candidates else "not_supported"
        return (
            "relationships.mutual_information",
            status,
            f"{len(candidates)} 个变量在低线性相关下仍出现较高互信息。",
            "inherent",
            None,
        )
    if item_id == "relationships.granger":
        precedence = relationships.get("granger_precedence")
        if not precedence:
            scope, remediation = _missing_scope(
                ("relationship_granger_causality_scan",),
                enabled_functions,
                approval_message="需要批准加入前置性检验后才能验证该假设。",
                data_message="已启用前置性检验，但样本不足以拟合受限与非受限回归。",
            )
            return "relationships.granger", "not_tested", "本轮未执行前置性检验。", scope, remediation
        preceding = precedence.get("variables_with_precedence") or []
        status = "candidate_support" if preceding else "not_supported"
        return (
            "relationships.granger",
            status,
            f"{len(preceding)} 个变量在样本内表现出对电价的前置性（非因果结论）。",
            "inherent",
            None,
        )
    if item_id == "relationships.rolling_stability":
        stability = relationships.get("rolling_stability")
        if not stability:
            scope, remediation = _missing_scope(
                ("relationship_rolling_correlation_stability",),
                enabled_functions,
                approval_message="需要批准加入关系稳定性分析后才能验证该假设。",
                data_message="已启用关系稳定性分析，但可用滚动窗口不足。",
            )
            return "relationships.rolling_stability", "not_tested", "本轮未执行关系稳定性分析。", scope, remediation
        unstable = stability.get("unstable_variables") or []
        status = "candidate_support" if unstable else "not_supported"
        return (
            "relationships.rolling_stability",
            status,
            f"{len(unstable)} 个变量的滚动相关出现明显漂移或反号。",
            "inherent",
            None,
        )
    return None


def _assess_hypotheses(plan: EDAPlan, summary: dict[str, Any]) -> list[HypothesisAssessment]:
    """Assess every planned hypothesis, keeping one row per agenda item.

    A plan carries both the model's own wording and the wording each selected
    function registers, so two differently phrased hypotheses regularly resolve
    to the same agenda item and the same evidence. Only the first wording is
    kept; otherwise the report and the revision agenda would count one question
    twice.
    """

    assessments: dict[str, HypothesisAssessment] = {}
    price = summary.get("price", {})
    exogenous = summary.get("exogenous", {})
    relationships = summary.get("relationships", {}).get("series", {})
    comparisons = summary.get("comparisons", {})
    enabled_functions = {step.function for step in plan.enabled_steps}
    for hypothesis in plan.hypotheses:
        item_id = resolve_agenda_item(hypothesis) or ""
        scope: AgendaScope = "inherent"
        remediation: str | None = None
        advanced = _advanced_assessment(
            item_id,
            price=price,
            exogenous=exogenous,
            relationships=summary.get("relationships", {}),
            enabled_functions=enabled_functions,
        )
        if advanced is not None:
            item_id, status, evidence, scope, remediation = advanced
        elif item_id == "price.extremes":
            extremes = price.get("extremes")
            if not extremes:
                status, evidence = "not_tested", "本轮未执行电价极端值方法。"
                scope, remediation = _missing_scope(
                    ("price_tukey_outer_fence",),
                    enabled_functions,
                    approval_message="需要批准加入电价极端值函数后才能验证该假设。",
                    data_message="已启用电价极端值函数，但当前没有可用极端值证据。",
                )
            else:
                count = int(extremes.get("high_spike_count", 0)) + int(extremes.get("extreme_low_count", 0))
                status = "candidate_support" if count else "not_supported"
                evidence = f"IQR 规则识别到 {count} 个极端高低价观测。"
        elif item_id == "price.autocorrelation":
            rows = [row for row in price.get("autocorrelation", []) if row.get("correlation") is not None]
            strongest = max((abs(float(row["correlation"])) for row in rows), default=None)
            if strongest is None:
                status, evidence = "not_tested", "本轮没有可用自相关结果。"
                scope, remediation = _missing_scope(
                    ("price_lag_autocorrelation",),
                    enabled_functions,
                    approval_message="需要批准加入电价自相关函数后才能验证该假设。",
                    data_message="已启用电价自相关函数，但当前有效样本不足以形成自相关证据。",
                )
            else:
                status = "candidate_support" if strongest >= 0.2 else "not_supported"
                evidence = f"扫描范围内最大绝对自相关为 {strongest:.3f}。"
        elif item_id == "price.seasonality":
            seasonality = price.get("seasonality")
            if not seasonality:
                status, evidence = "not_tested", "本轮未执行季节性方法。"
                scope, remediation = _missing_scope(
                    ("price_calendar_group_profile",),
                    enabled_functions,
                    approval_message="需要批准加入日历分组函数后才能验证该假设。",
                    data_message="已启用日历分组函数，但当前没有足够季节分组证据。",
                )
            else:
                hour_means = [row.get("mean") for row in seasonality.get("hour_of_day", []) if row.get("mean") is not None]
                spread = max(hour_means) - min(hour_means) if hour_means else None
                strengths = [
                    value
                    for value in ((price.get("decomposition") or {}).get("seasonal_strength") or {}).values()
                    if value is not None
                ]
                spread_text = f"小时均值最大差为 {spread:.3f}；" if spread is not None else ""
                shares = {
                    label: share
                    for label in ("hour_of_day", "day_of_week", "month")
                    if (share := group_variance_share(seasonality.get(label))) is not None
                }
                if strengths:
                    strongest = max(strengths)
                    status = "candidate_support" if strongest >= SEASONAL_STRENGTH_THRESHOLD else "not_supported"
                    evidence = f"{spread_text}最强季节成分强度为 {strongest:.3f}。"
                elif shares:
                    label, widest = max(shares.items(), key=lambda item: item[1])
                    status = (
                        "candidate_support" if widest >= CALENDAR_EFFECT_THRESHOLD else "not_supported"
                    )
                    evidence = f"{spread_text}按{GROUP_LABELS.get(label, label)}分组能解释 {widest:.1%} 的电价波动。"
                else:
                    status, evidence = "not_tested", "季节分组样本不足。"
                    scope, remediation = _missing_scope(
                        ("price_calendar_group_profile",),
                        enabled_functions,
                        approval_message="需要批准加入日历分组函数后才能验证该假设。",
                        data_message="已启用日历分组函数，但当前分组样本不足。",
                    )
        elif item_id == "exogenous.collinearity":
            pairs = exogenous.get("strong_collinearity_pairs")
            if pairs is None:
                status, evidence = "not_tested", "本轮未执行共线性方法。"
                scope, remediation = _missing_scope(
                    ("exogenous_pearson_collinearity",),
                    enabled_functions,
                    approval_message="需要批准加入外生变量共线性函数后才能验证该假设。",
                    data_message="已启用共线性函数，但当前没有可用变量冗余证据。",
                )
            else:
                status = "candidate_support" if pairs else "not_supported"
                evidence = f"有 {len(pairs)} 对变量彼此高度相关。"
        elif item_id == "exogenous.outliers":
            series = exogenous.get("series", {})
            counts = sum(int(item.get("outlier_count", 0)) for item in series.values())
            tested = any("outlier_count" in item for item in series.values())
            status = "candidate_support" if counts else "not_supported" if tested else "not_tested"
            evidence = f"所选变量共标记 {counts} 个 IQR 异常观测。" if tested else "本轮未执行变量异常值方法。"
            if not tested:
                scope, remediation = _missing_scope(
                    ("exogenous_iqr_outliers",),
                    enabled_functions,
                    approval_message="需要批准加入外生变量异常值函数后才能验证该假设。",
                    data_message="已启用变量异常值函数，但当前没有可用异常值证据。",
                )
        elif item_id in {"relationships.contemporaneous", "relationships.monotonic"}:
            coefficients = [
                abs(float(metric["correlation"]))
                for result in relationships.values()
                for metric in result.get("contemporaneous", {}).values()
                if metric.get("correlation") is not None
            ]
            strongest = max(coefficients, default=None)
            if strongest is None:
                status, evidence = "not_tested", "本轮没有可用同期关系结果。"
                scope, remediation = _missing_scope(
                    ("relationship_scipy_pearson_pairwise", "relationship_scipy_spearman_pairwise"),
                    enabled_functions,
                    approval_message="需要批准加入同期关系函数后才能验证该假设。",
                    data_message="已启用同期关系函数，但当前有效样本不足以形成同期关系证据。",
                )
            else:
                status = "candidate_support" if strongest >= 0.1 else "not_supported"
                evidence = f"同一时刻的最大绝对相关系数为 {strongest:.3f}，尚未排除共同趋势和季节规律。"
        elif item_id == "relationships.lead_lag":
            best_rows = [
                result.get("best_absolute_lag") or {}
                for result in relationships.values()
                if result.get("best_absolute_lag")
            ]
            positive = [row for row in best_rows if int(row.get("lag", 0)) > 0]
            if not best_rows:
                status, evidence = "not_tested", "本轮没有可用滞后扫描结果。"
                scope, remediation = _missing_scope(
                    ("relationship_pearson_positive_lead_scan",),
                    enabled_functions,
                    approval_message="需要批准加入领先滞后扫描函数后才能验证该假设。",
                    data_message="已启用领先滞后扫描函数，但当前没有足够关系证据。",
                )
            else:
                status = "candidate_support" if positive else "not_supported"
                evidence = f"{len(positive)}/{len(best_rows)} 个变量在正的提前量上相关最强。"
        elif item_id in {"comparisons.price_segments", "comparisons.relationship_segments"}:
            prefer_relationship = item_id == "comparisons.relationship_segments"
            price_contrasts = [
                (comparison, row)
                for comparison in comparisons.get("price", {}).values()
                for row in comparison.get("contrasts", [])
                if row.get("mean_difference_left_minus_right") is not None
            ]
            relationship_contrasts = [
                (comparison, variable, row)
                for comparison in comparisons.get("relationships", {}).values()
                for variable, result in comparison.get("series", {}).items()
                for row in result.get("contrasts", [])
                if row.get("correlation_difference_left_minus_right") is not None
            ]
            if price_contrasts and not (prefer_relationship and relationship_contrasts):
                comparison, largest = max(
                    price_contrasts,
                    key=lambda item: abs(item[1]["mean_difference_left_minus_right"]),
                )
                volatility = [
                    (item_comparison, row)
                    for item_comparison, row in price_contrasts
                    if row.get("std_difference_left_minus_right") is not None
                ]
                volatility_text = ""
                if volatility:
                    volatility_comparison, volatility_largest = max(
                        volatility,
                        key=lambda item: abs(item[1]["std_difference_left_minus_right"]),
                    )
                    volatility_text = (
                        f"；{_segment_pair_label(volatility_comparison, volatility_largest)}的标准差相差 "
                        f"{abs(float(volatility_largest['std_difference_left_minus_right'])):.3f}"
                    )
                status = "candidate_support"
                evidence = (
                    f"{_segment_pair_label(comparison, largest)}的均值相差 "
                    f"{abs(float(largest['mean_difference_left_minus_right'])):.3f}"
                    f"{volatility_text}，仍属于描述性比较。"
                )
            elif relationship_contrasts:
                comparison, variable, largest = max(
                    relationship_contrasts,
                    key=lambda item: abs(item[2]["correlation_difference_left_minus_right"]),
                )
                status = "candidate_support"
                evidence = (
                    f"{variable} 在{_segment_pair_label(comparison, largest)}的相关系数相差 "
                    f"{abs(float(largest['correlation_difference_left_minus_right'])):.3f}。"
                )
            else:
                status, evidence = "not_tested", "本轮没有可比较的分段证据。"
                scope, remediation = _missing_scope(
                    ("price_segment_distribution_comparison", "relationship_pearson_segment_comparison"),
                    enabled_functions,
                    approval_message="需要批准加入分段比较函数并明确分段定义后才能验证该假设。",
                    data_message="已启用分段比较函数，但当前分段样本不足或没有可比较证据。",
                )
        else:
            item_id = f"hypothesis:{hashlib.sha256(hypothesis.encode('utf-8')).hexdigest()[:12]}"
            status, evidence = "not_tested", "本轮的分析方法里没有能判定这条说法的检验。"
            scope = "needs_restatement"
            remediation = "请把它改写成本轮方法能判定的说法，或者确认不再跟踪它。"
        assessments.setdefault(
            item_id,
            HypothesisAssessment(
                item_id=item_id,
                hypothesis=hypothesis,
                status=status,
                evidence=evidence,
                scope=scope,
                remediation=remediation,
            ),
        )
    for hypothesis in plan.unverifiable_hypotheses:
        item_id = f"hypothesis:{hashlib.sha256(hypothesis.encode('utf-8')).hexdigest()[:12]}"
        assessments.setdefault(
            item_id,
            HypothesisAssessment(
                item_id=item_id,
                hypothesis=hypothesis,
                status="not_tested",
                evidence="这条说法没有对应的确定性研究函数，因此未进入本轮执行议程。",
                scope="needs_restatement",
                remediation="请把它改写为当前研究函数能判定的说法，或者确认不再跟踪它。",
            ),
        )
    return list(assessments.values())


def evaluate_agent_run(
    *,
    plan: EDAPlan,
    quality: DataQualityReport,
    summary: dict[str, Any],
) -> AgentEvaluation:
    """Check evidence coverage and produce concise, non-causal findings."""

    checks: list[EvaluationCheck] = []
    findings: list[str] = []
    warnings: list[str] = []
    followups: list[str] = []

    if plan.requested_statistics:
        distribution = (summary.get("price") or {}).get("distribution") or {}
        missing_statistics = [name for name in plan.requested_statistics if distribution.get(name) is None]
        checks.append(
            EvaluationCheck(
                name="描述统计交付完整性",
                status="warning" if missing_statistics else "pass",
                message=(
                    "缺少统计量：" + "、".join(missing_statistics)
                    if missing_statistics
                    else "用户要求的均值、最低值或最高值已经由确定性分布函数计算。"
                ),
                scope="needs_data" if missing_statistics else "inherent",
                remediation="检查电价分布函数输出和目标序列有效值。" if missing_statistics else None,
            )
        )

    if quality.usable_for_eda:
        checks.append(EvaluationCheck(name="目标数据可用性", status="pass", message="目标序列满足最小样本要求。"))
    else:
        checks.append(
            EvaluationCheck(
                name="目标数据可用性",
                status="fail",
                message="目标序列不足以支持当前 EDA。",
                scope="needs_data",
                remediation="请补充目标电价数据、扩大有效研究窗口或降低最低样本门槛后重新开始。",
            )
        )

    target_report = quality.series[summary["study"]["target"]]
    if target_report.aligned_coverage_rate >= 0.98:
        checks.append(
            EvaluationCheck(
                name="目标覆盖率",
                status="pass",
                message=f"目标数据覆盖率为 {target_report.aligned_coverage_rate:.2%}。",
            )
        )
    else:
        checks.append(
            EvaluationCheck(
                name="目标覆盖率",
                status="warning",
                message=f"目标数据覆盖率为 {target_report.aligned_coverage_rate:.2%}，解释时需考虑缺口。",
                scope="needs_data",
                remediation="请补充目标数据覆盖率后重新开始研究。",
            )
        )

    uses_exogenous = any(step.enabled and FUNCTION_CATALOG[step.function].uses_variables for step in plan.steps)
    in_scope_series = {summary["study"]["target"]}
    if uses_exogenous:
        in_scope_series.update(plan.selected_variables)
    severe_issues = [
        issue
        for issue in quality.issues
        if issue.severity in {"critical", "high"} and (issue.series is None or issue.series in in_scope_series)
    ]
    if severe_issues:
        blocking_quality_codes = {
            "series_has_no_values",
            "very_low_coverage",
            "low_coverage",
            "incomplete_coverage",
            "invalid_timestamps",
            "non_numeric_values",
        }
        data_scope: AgendaScope = (
            "needs_data"
            if any(issue.code in blocking_quality_codes for issue in severe_issues)
            else "inherent"
        )
        checks.append(
            EvaluationCheck(
                name="数据风险",
                status="warning",
                message=f"检测到 {len(severe_issues)} 个高优先级数据或可获得性风险。",
                scope=data_scope,
                remediation=(
                    "请先处理高优先级数据风险，再继续研究。"
                    if data_scope == "needs_data"
                    else "当前风险作为可获得性或方法限制记录，不阻止本轮描述性 EDA。"
                ),
            )
        )
        warnings.extend(issue_line(issue) for issue in severe_issues[:6])
    else:
        checks.append(EvaluationCheck(name="数据风险", status="pass", message="未检测到高优先级数据风险。"))

    missing_sections = sorted({
        section
        for step in plan.enabled_steps
        if (section := FUNCTION_CATALOG[step.function].result_key) != "data_quality" and section not in summary
    })
    checks.append(
        EvaluationCheck(
            name="计划执行完整性",
            status="fail" if missing_sections else "pass",
            message=(
                f"缺少计划结果：{', '.join(missing_sections)}。"
                if missing_sections
                else f"已完成 {len(plan.enabled_steps)} 个确认步骤。"
            ),
            scope="within_envelope" if missing_sections else "inherent",
            remediation=(
                f"补齐缺少的确定性结果步骤：{', '.join(missing_sections)}，然后重新执行。"
                if missing_sections
                else None
            ),
        )
    )

    price = summary.get("price")
    if price:
        distribution = price.get("distribution")
        extremes = price.get("extremes")
        if distribution and extremes:
            findings.append(
                f"多数时候电价在 {_fmt(distribution.get('median'))} 附近，"
                f"平均值 {_fmt(distribution.get('mean'))}，"
                f"另有 {extremes.get('high_spike_count', 0)} 个极端高价时点。"
            )
        elif distribution:
            findings.append(
                f"多数时候电价在 {_fmt(distribution.get('median'))} 附近，"
                f"平均值 {_fmt(distribution.get('mean'))}。"
            )
        elif extremes:
            findings.append(f"本轮识别到 {extremes.get('high_spike_count', 0)} 个极端高价时点。")

        stationarity = price.get("stationarity")
        if stationarity:
            findings.append(
                f"电价{stationarity_text(stationarity.get('verdict'))}，"
                f"{transform_text(stationarity.get('recommended_transform'))}。"
            )
            if stationarity.get("verdict") != "stationary":
                checks.append(
                    EvaluationCheck(
                        name="平稳性",
                        status="warning",
                        message="电价本身不围绕固定水平波动，同期相关里可能混进了共同趋势。",
                        scope="inherent",
                    )
                )
                followups.append("建模前先对电价做差分或去掉季节成分，再重新评估这些关系的强度。")

        decomposition = price.get("decomposition")
        if decomposition:
            remainder = (decomposition.get("variance_share") or {}).get("remainder")
            dominant = {"day": "日内周期", "week": "周周期"}.get(
                str(decomposition.get("dominant_seasonality")), "未识别"
            )
            findings.append(
                f"价格里最明显的重复规律是{dominant}；拆掉趋势和周期之后，"
                f"仍有 {_share(remainder)} 的波动无法解释。"
            )

        regime = price.get("spike_regime")
        if regime:
            findings.append(
                f"价格尖峰占 {_share(regime.get('high_spike_share'))} 的时段，"
                f"负价占 {_share(regime.get('negative_share'))}，"
                f"最长的一段尖峰连续了 "
                f"{regime.get('high_episodes', {}).get('max_duration_intervals', 0)} 个时段。"
            )

        baselines = price.get("naive_baselines")
        if baselines:
            findings.append(
                f"直接照抄历史价格中效果最好的是「{baselines.get('best_baseline_label')}」，"
                f"平均绝对误差 {_fmt(baselines.get('best_mae'))}；后续模型必须低于这个数字。"
            )
            checks.append(
                EvaluationCheck(
                    name="可预测性基线",
                    status="pass",
                    message=str(baselines.get("error_floor_note", "已给出朴素基线误差底线。")),
                    scope="inherent",
                )
            )
            if not baselines.get("percentage_errors_reliable", True):
                warnings.append("电价里有接近零的取值，用百分比表示的误差指标（如 MAPE）在这条序列上会失真。")

        stabilization = price.get("variance_stabilization")
        if stabilization and stabilization.get("recommended_transform") != "none":
            followups.append("训练模型之前，先做一次稳健标准化和 asinh 变换来压缩极端值。")

    exogenous_evidence = summary.get("exogenous", {})
    multicollinearity = exogenous_evidence.get("multicollinearity")
    if multicollinearity:
        severe = multicollinearity.get("severe_variables") or []
        checks.append(
            EvaluationCheck(
                name="变量冗余",
                status="warning" if severe else "pass",
                message=(
                    f"{len(severe)} 个变量的方差膨胀因子超过 {multicollinearity.get('severe_threshold')}。"
                    if severe
                    else "所选变量之间没有出现严重的多重共线性。"
                ),
                scope="inherent",
            )
        )
        if severe:
            warnings.append(
                f"{'、'.join(severe[:5])} 提供的信息和其他变量大量重叠，建模时保留一个代表即可。"
            )

    driver_stationarity = exogenous_evidence.get("driver_stationarity")
    if driver_stationarity:
        non_stationary = driver_stationarity.get("non_stationary_variables") or []
        if non_stationary:
            warnings.append(
                f"{len(non_stationary)} 个驱动变量自身非平稳；与非平稳电价的同期相关可能是伪回归。"
            )

    dependence = summary.get("relationships", {})
    information = dependence.get("mutual_information")
    if information:
        candidates = information.get("nonlinear_candidates") or []
        findings.append(
            f"互信息扫描标记出 {len(candidates)} 个线性相关较弱但仍存在依赖的变量。"
            if candidates
            else "互信息扫描未发现线性相关之外的额外依赖。"
        )
    precedence = dependence.get("granger_precedence")
    if precedence:
        preceding = precedence.get("variables_with_precedence") or []
        findings.append(
            f"{len(preceding)}/{len(precedence.get('series', {}))} 个变量在样本内改善了电价自回归拟合（非因果结论）。"
        )
    stability = dependence.get("rolling_stability")
    if stability:
        unstable = stability.get("unstable_variables") or []
        if unstable:
            warnings.append(
                f"{'、'.join(unstable[:5])} 与电价的关系随时间明显变化，全时段的相关系数代表不了各个阶段。"
            )
            followups.append("对关系不稳定的变量，按时间分段重新评估，或者改用系数随时间变化的模型。")

    relationships = summary.get("relationships", {}).get("series", {})
    ranked: list[tuple[str, str, float, float | None, int | None]] = []
    for name, result in relationships.items():
        pearson_result = result.get("contemporaneous", {}).get("pearson", {})
        spearman_result = result.get("contemporaneous", {}).get("spearman", {})
        pearson = pearson_result.get("correlation")
        spearman = spearman_result.get("correlation")
        best = result.get("best_absolute_lag") or {}
        if pearson is not None:
            ranked.append((name, "同期 Pearson", float(pearson), pearson_result.get("p_value"), None))
        if spearman is not None:
            ranked.append((name, "同期 Spearman", float(spearman), spearman_result.get("p_value"), None))
        if best.get("correlation") is not None:
            ranked.append((name, "领先滞后扫描", float(best["correlation"]), best.get("p_value"), best.get("lag")))
    ranked.sort(key=lambda row: abs(row[2]), reverse=True)
    if ranked:
        for name, method, correlation, p_value, lag in ranked[:3]:
            if lag:
                position = f"提前 {lag} 个时段时"
            elif method == "同期 Spearman":
                position = "在同一时刻按大小排序比较"
            else:
                position = "在同一时刻"
            if p_value is None:
                p_text = ""
            elif p_value == 0:
                p_text = "（p 远小于 0.001，未做多重比较校正）"
            else:
                p_text = f"（p={p_value:.3g}，未做多重比较校正）"
            findings.append(f"{name} 与电价{position}的相关系数为 {correlation:.3f}{p_text}。")
        meaningful = [row for row in ranked if abs(row[2]) >= 0.1]
        checks.append(
            EvaluationCheck(
                name="关系证据",
                status="pass" if meaningful else "warning",
                message=(
                    f"{len({row[0] for row in meaningful})} 个变量形成了需要进一步验证的探索性关系。"
                    if meaningful
                    else "已计算关系指标，但当前效应均较弱。"
                ),
                scope="inherent",
            )
        )
        truncated, usable_ceiling = _truncated_lag_scans(relationships)
        if truncated and usable_ceiling is not None:
            checks.append(
                EvaluationCheck(
                    name="滞后扫描样本",
                    status="warning",
                    message=(
                        f"{len(truncated)} 个变量的滞后扫描超出了成对样本能支撑的范围："
                        f"{'、'.join(truncated[:4])}。"
                    ),
                    scope="within_envelope",
                    remediation=f"把 max_lag 收缩到 {usable_ceiling} 个间隔后重新执行，不要新增函数或变量。",
                )
            )
        without_evidence = _variables_without_relationship_evidence(relationships)
        if without_evidence and len(without_evidence) < len(relationships):
            checks.append(
                EvaluationCheck(
                    name="变量关系证据",
                    status="warning",
                    message=(
                        f"{len(without_evidence)} 个变量在任何滞后上都没有形成可用关系证据："
                        f"{'、'.join(without_evidence[:4])}。"
                    ),
                    scope="within_envelope",
                    remediation=(
                        f"从 selected_variables 中移除 {'、'.join(without_evidence)} 后重新执行，"
                        "不要新增函数或变量。"
                    ),
                )
            )
        relationship_methods = set(summary.get("relationships", {}).get("methods", []))
        if relationship_methods.intersection({"pearson", "spearman", "lag_scan"}):
            checks.append(
                EvaluationCheck(
                    name="时间序列混杂控制",
                    status="warning",
                    message="当前关系尚未控制共同趋势、季节性与序列自相关，只能作为候选关系。",
                    scope="inherent",
                )
            )
            warnings.append("这些相关系数还没有排除共同趋势、日内和周内规律，以及电价自身的延续性。")
        if "lag_scan" in relationship_methods:
            warnings.append("最合适的提前量是在多个位置里挑出来的，p 值没有做多重比较校正，显著性会被高估。")
            followups.append("先给候选变量去掉趋势和季节成分，再重新检查提前量是否稳定，并对显著性做多重比较校正。")
    elif "relationships" in summary:
        checks.append(
            EvaluationCheck(
                name="关系证据",
                status="warning",
                message="所选变量未形成满足样本门槛的关系证据。",
                scope="needs_data",
                remediation="请检查变量覆盖率、研究窗口和最小样本门槛后重新开始关系分析。",
            )
        )
        followups.append("先检查变量的数据完整度、研究时段和最小样本要求，再重新安排关系分析。")

    comparisons = summary.get("comparisons", {})
    comparison_count = 0
    comparison_warnings: list[str] = []
    for comparison_id, comparison in comparisons.get("price", {}).items():
        comparison_count += 1
        insufficient = [
            segment_id
            for segment_id, row in comparison.get("segments", {}).items()
            if not row.get("sufficient_observations", False)
        ]
        overlaps = [row for row in comparison.get("contrasts", []) if int(row.get("overlap_rows", 0)) > 0]
        if insufficient:
            comparison_warnings.append(f"{comparison_id} 分段样本不足：{', '.join(insufficient)}。")
        if overlaps:
            comparison_warnings.append(f"{comparison_id} 存在重叠子样本，差值不是独立组比较。")
        usable = [
            row
            for row in comparison.get("contrasts", [])
            if row.get("mean_difference_left_minus_right") is not None
        ]
        if usable:
            largest = max(usable, key=lambda row: abs(row["mean_difference_left_minus_right"]))
            finding = (
                f"分段对比 {comparison_id} 中，{_segment_pair_label(comparison, largest)}的均值相差 "
                f"{abs(float(largest['mean_difference_left_minus_right'])):.3f}"
            )
            volatility = [
                row
                for row in comparison.get("contrasts", [])
                if row.get("std_difference_left_minus_right") is not None
            ]
            if volatility:
                volatility_largest = max(
                    volatility,
                    key=lambda row: abs(row["std_difference_left_minus_right"]),
                )
                finding += (
                    f"；{_segment_pair_label(comparison, volatility_largest)}的标准差相差 "
                    f"{abs(float(volatility_largest['std_difference_left_minus_right'])):.3f}"
                )
            findings.append(f"{finding}。")
    for comparison_id, comparison in comparisons.get("relationships", {}).items():
        comparison_count += 1
        usable = [
            row
            for result in comparison.get("series", {}).values()
            for row in result.get("contrasts", [])
            if row.get("correlation_difference_left_minus_right") is not None
        ]
        overlaps = [
            row
            for result in comparison.get("series", {}).values()
            for row in result.get("contrasts", [])
            if int(row.get("overlap_rows", 0)) > 0
        ]
        if overlaps:
            comparison_warnings.append(f"{comparison_id} 存在重叠子样本，相关差异需谨慎解释。")
        if usable:
            largest = max(usable, key=lambda row: abs(row["correlation_difference_left_minus_right"]))
            variable = next(
                (
                    name
                    for name, result in comparison.get("series", {}).items()
                    if largest in result.get("contrasts", [])
                ),
                "变量",
            )
            findings.append(
                f"分段关系 {comparison_id} 中，{variable} 在不同分段的最大绝对相关系数差为 "
                f"{abs(float(largest['correlation_difference_left_minus_right'])):.3f}。"
            )
    if comparison_count:
        checks.append(
            EvaluationCheck(
                name="分段比较证据",
                status="warning" if comparison_warnings else "pass",
                message=("；".join(comparison_warnings) if comparison_warnings else f"已完成 {comparison_count} 个同快照分段比较。"),
                scope=(
                    "needs_data"
                    if any("分段样本不足" in item for item in comparison_warnings)
                    else "inherent"
                ),
                remediation=(
                    "请补充分段样本或调整研究窗口后重新进行分段比较。"
                    if any("分段样本不足" in item for item in comparison_warnings)
                    else None
                ),
            )
        )
        warnings.extend(comparison_warnings)
        warnings.append("分段之间的差异只是直接比较的结果，还没有排除趋势、季节以及同时发生的其他因素。")

    if any(issue.code == "lookahead_risk" for issue in quality.issues):
        followups.append("进入预测实验前，去掉只有事后观测值的变量，或者补上它们真实的可获得时间。")
    if relationships:
        followups.append("把证据最强的关系做成候选特征，并在按时间顺序切分的实验里验证它是否真的提升精度。")
    if not findings:
        findings.append("本轮主要完成数据可用性审查；尚未执行足以支持关系结论的分析步骤。")

    variable_recommendations: dict[str, Any] | None = None
    if plan.variable_selection_mode == "auto_recommend" and plan.variable_selection_stage == "screening":
        variable_recommendations = build_variable_recommendations(
            selected_variables=plan.selected_variables,
            quality=quality,
            summary=summary,
            limit=plan.variable_recommendation_limit,
        )
        recommended_names = variable_recommendations["recommended_variables"]
        recommendation_text = (
            f"推荐优先深入分析：{'、'.join(recommended_names)}。"
            if recommended_names
            else "当前筛查没有形成达到关系门槛的优先变量。"
        )
        checks.append(
            EvaluationCheck(
                name="外生变量推荐确认",
                status="warning",
                message=recommendation_text,
                scope="needs_approval",
                remediation="请确认推荐变量、修改变量范围，或选择分析全部合格变量后再执行深入方法。",
            )
        )
        findings.append(
            f"已完成 {variable_recommendations['screened_variable_count']} 个外生变量的确定性筛查；"
            f"{recommendation_text}"
        )
        followups.append("确认推荐的变量之后，再做分小时、分月份、提前量、非线性和稳定性这几项分析。")

    hypothesis_assessments = _assess_hypotheses(plan, summary)

    def check_is_open(check: EvaluationCheck) -> bool:
        return check.status != "pass" and check.scope != "inherent"

    def hypothesis_is_open(item: HypothesisAssessment) -> bool:
        return item.status in {"not_tested", "inconclusive"} and item.scope != "inherent"

    within_items = [
        check.name for check in checks if check_is_open(check) and check.scope == "within_envelope"
    ] + [
        item.hypothesis for item in hypothesis_assessments
        if hypothesis_is_open(item) and item.scope == "within_envelope"
    ]
    user_items = [
        check.name for check in checks if check_is_open(check) and check.scope != "within_envelope"
    ] + [
        item.hypothesis for item in hypothesis_assessments
        if hypothesis_is_open(item) and item.scope != "within_envelope"
    ]
    needs_approval_items = [
        check.name for check in checks if check_is_open(check) and check.scope == "needs_approval"
    ] + [
        item.hypothesis for item in hypothesis_assessments
        if hypothesis_is_open(item) and item.scope == "needs_approval"
    ]
    needs_data_items = [
        check.name for check in checks if check_is_open(check) and check.scope == "needs_data"
    ] + [
        item.hypothesis for item in hypothesis_assessments
        if hypothesis_is_open(item) and item.scope == "needs_data"
    ]
    needs_restatement_items = [
        check.name for check in checks if check_is_open(check) and check.scope == "needs_restatement"
    ] + [
        item.hypothesis for item in hypothesis_assessments
        if hypothesis_is_open(item) and item.scope == "needs_restatement"
    ]
    def listed(items: list[str], limit: int = 3) -> str:
        """Join items as one clause: they are sentences, so their periods must go."""

        cleaned = [item.strip().rstrip("。") for item in items[:limit] if item.strip()]
        suffix = f"（另有 {len(items) - limit} 项）" if len(items) > limit else ""
        return "；".join(cleaned) + suffix

    resolved_hypotheses = sum(
        item.status in {"candidate_support", "not_supported"} for item in hypothesis_assessments
    )

    if within_items:
        decision = "revise"
        decision_text = (
            f"本轮还有 {len(within_items)} 项可以在你已批准的范围内自动补齐：{listed(within_items, 4)}。"
        )
    elif user_items:
        decision = "need_user"
        parts = [f"本轮有 {len(user_items)} 项待你决定。"]
        if needs_approval_items:
            parts.append(
                f"需要你同意扩大分析范围（{len(needs_approval_items)} 项）："
                f"{listed(needs_approval_items)}。"
            )
        if needs_data_items:
            parts.append(
                f"需要补充或修复数据（{len(needs_data_items)} 项）：{listed(needs_data_items)}。"
            )
        if needs_restatement_items:
            parts.append(
                f"需要改写或确认不再跟踪的说法（{len(needs_restatement_items)} 项）："
                f"{listed(needs_restatement_items)}。"
            )
        parts.append("逐条依据见 methods.md 的「假设验收」。")
        decision_text = "".join(parts)
    else:
        decision = "accept"
        unsettled = [
            item for item in hypothesis_assessments if item.status in {"not_tested", "inconclusive"}
        ]
        total_hypotheses = len(hypothesis_assessments)
        if unsettled:
            # The header already carries the run's status; repeating it here says nothing.
            decision_text = (
                f"本轮登记的 {total_hypotheses} 条判断中，{resolved_hypotheses} 条得到了明确结论，"
                f"{len(unsettled)} 条受方法本身限制无法在本轮判定。逐条依据见 methods.md 的「假设验收」。"
            )
        else:
            decision_text = (
                f"本轮登记的 {total_hypotheses} 条判断全部得到明确结论。"
                "逐条依据见 methods.md 的「假设验收」。"
            )

    if variable_recommendations is not None:
        decision = "need_user"
        recommended_names = variable_recommendations["recommended_variables"]
        decision_text = (
            "外生变量自动筛查已完成。"
            + (
                f"推荐优先深入分析 {'、'.join(recommended_names)}；深入方法尚未执行。"
                if recommended_names
                else "没有变量达到优先推荐门槛；请查看探索性候选并决定是否继续。"
            )
        )

    agenda_payload = [
        {
            "item_id": f"check:{check.name}",
            "status": check.status,
            "scope": check.scope,
        }
        for check in checks
    ] + [
        {
            "item_id": f"hypothesis:{item.item_id or item.hypothesis}",
            "status": item.status,
            "scope": item.scope,
        }
        for item in hypothesis_assessments
    ]
    agenda_fingerprint = hashlib.sha256(
        json.dumps(sorted(agenda_payload, key=lambda item: item["item_id"]), ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()[:24]

    def scope_recommendation(scope: AgendaScope) -> str:
        return {
            "within_envelope": "在原审批范围内修订 variables 或收缩 max_lag。",
            "needs_approval": "请用户明确批准新增函数、变量或分段范围后开启新的研究 Episode。",
            "needs_data": "请补充或修复数据后重新开始研究。",
            "needs_restatement": "请把该假设改写为可由已授权函数验证的表述，或确认不再追踪它。",
            "inherent": "作为方法论限制记录，不驱动自动修订。",
        }[scope]

    feedback_packets: list[FeedbackPacket] = []
    for index, check in enumerate(checks):
        if not check_is_open(check):
            continue
        feedback_packets.append(
            FeedbackPacket(
                feedback_id=hashlib.sha256(
                    f"{plan.plan_id}:evaluation:check:{index}:{check.status}:{check.scope}:{check.message}".encode()
                ).hexdigest()[:32],
                source="evaluator",
                code=f"evaluation_check_{index}_{check.status}",
                severity="error" if check.status == "fail" else "warning",
                message=check.message,
                observed={"status": check.status, "scope": check.scope},
                expected="pass",
                recommendation=check.remediation or scope_recommendation(check.scope),
                retryable=check.scope == "within_envelope",
                requires_user=check.scope != "within_envelope",
                created_at=plan.created_at,
            )
        )
    for index, item in enumerate(hypothesis_assessments):
        if not hypothesis_is_open(item):
            continue
        feedback_packets.append(
            FeedbackPacket(
                feedback_id=hashlib.sha256(
                    f"{plan.plan_id}:evaluation:hypothesis:{index}:{item.status}:{item.scope}:{item.hypothesis}".encode()
                ).hexdigest()[:32],
                source="evaluator",
                code=f"evaluation_hypothesis_{index}_{item.status}",
                severity="warning",
                message=f"{item.hypothesis}：{item.evidence}",
                observed={"status": item.status, "scope": item.scope},
                expected=["candidate_support", "not_supported"],
                recommendation=item.remediation or scope_recommendation(item.scope),
                retryable=item.scope == "within_envelope",
                requires_user=item.scope != "within_envelope",
                created_at=plan.created_at,
            )
        )
    return AgentEvaluation(
        decision=decision,
        summary=decision_text,
        checks=checks,
        hypothesis_assessments=hypothesis_assessments,
        findings=findings,
        warnings=warnings,
        suggested_followups=list(dict.fromkeys(followups)),
        variable_recommendations=variable_recommendations,
        feedback_packets=feedback_packets,
        agenda_fingerprint=agenda_fingerprint,
    )
