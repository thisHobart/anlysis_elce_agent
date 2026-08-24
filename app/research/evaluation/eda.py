"""Deterministic evaluation of evidence produced by an approved EDA plan."""

from __future__ import annotations

import hashlib
from typing import Any

from app.research.agent.schemas import AgentEvaluation, EDAPlan, EvaluationCheck, HypothesisAssessment
from app.research.schemas.feedback import FeedbackPacket
from app.research.schemas.results import DataQualityReport


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _issue_message(code: str, fallback: str) -> str:
    messages = {
        "series_has_no_values": "统一时间轴上没有可用数值。",
        "very_low_coverage": "可用数据不足统一时间轴的一半。",
        "low_coverage": "缺失时间点可能影响时序和关系分析。",
        "incomplete_coverage": "统一时间轴上存在数据缺口。",
        "invalid_timestamps": "部分非法时间戳已在对齐前排除。",
        "non_numeric_values": "部分非空值无法转换为数值。",
        "duplicate_timestamps": "重复时间戳已按配置规则聚合。",
        "availability_unknown": "预测时点的数据可获得性未知，当前关系只能用于描述。",
        "forecast_vintage_unknown": "缺少预测发布或可获得时间，无法确认该变量在预测时点可用。",
        "lookahead_risk": "实际观测值在未来预测时点可能不可用。",
    }
    return messages.get(code, fallback)


def _assess_hypotheses(plan: EDAPlan, summary: dict[str, Any]) -> list[HypothesisAssessment]:
    assessments: list[HypothesisAssessment] = []
    price = summary.get("price", {})
    exogenous = summary.get("exogenous", {})
    relationships = summary.get("relationships", {}).get("series", {})
    for hypothesis in plan.hypotheses:
        if any(word in hypothesis for word in ("尖峰", "极端", "负价")):
            extremes = price.get("extremes")
            if not extremes:
                status, evidence = "not_tested", "本轮未执行电价极端值方法。"
            else:
                count = int(extremes.get("high_spike_count", 0)) + int(extremes.get("extreme_low_count", 0))
                status = "candidate_support" if count else "not_supported"
                evidence = f"IQR 规则识别到 {count} 个极端高低价观测。"
        elif "自相关" in hypothesis:
            rows = [row for row in price.get("autocorrelation", []) if row.get("correlation") is not None]
            strongest = max((abs(float(row["correlation"])) for row in rows), default=None)
            if strongest is None:
                status, evidence = "not_tested", "本轮没有可用自相关结果。"
            else:
                status = "candidate_support" if strongest >= 0.2 else "not_supported"
                evidence = f"扫描范围内最大绝对自相关为 {strongest:.3f}。"
        elif any(word in hypothesis for word in ("日内", "周内", "月份结构")):
            seasonality = price.get("seasonality")
            if not seasonality:
                status, evidence = "not_tested", "本轮未执行季节性方法。"
            else:
                hour_means = [row.get("mean") for row in seasonality.get("hour_of_day", []) if row.get("mean") is not None]
                spread = max(hour_means) - min(hour_means) if hour_means else None
                status = "inconclusive"
                evidence = (
                    f"小时均值最大差为 {spread:.3f}，仍需季节显著性和跨时期稳定性检验。"
                    if spread is not None
                    else "季节分组样本不足。"
                )
        elif "共线性" in hypothesis:
            pairs = exogenous.get("strong_collinearity_pairs")
            if pairs is None:
                status, evidence = "not_tested", "本轮未执行共线性方法。"
            else:
                status = "candidate_support" if pairs else "not_supported"
                evidence = f"检测到 {len(pairs)} 对超过阈值的变量。"
        elif "异常观测" in hypothesis:
            series = exogenous.get("series", {})
            counts = sum(int(item.get("outlier_count", 0)) for item in series.values())
            tested = any("outlier_count" in item for item in series.values())
            status = "candidate_support" if counts else "not_supported" if tested else "not_tested"
            evidence = f"所选变量共标记 {counts} 个 IQR 异常观测。" if tested else "本轮未执行变量异常值方法。"
        elif any(word in hypothesis for word in ("同期线性", "单调关系")):
            coefficients = [
                abs(float(metric["correlation"]))
                for result in relationships.values()
                for metric in result.get("contemporaneous", {}).values()
                if metric.get("correlation") is not None
            ]
            strongest = max(coefficients, default=None)
            if strongest is None:
                status, evidence = "not_tested", "本轮没有可用同期关系结果。"
            else:
                status = "candidate_support" if strongest >= 0.1 else "not_supported"
                evidence = f"最大绝对同期关系系数为 {strongest:.3f}，尚未控制时间序列混杂。"
        elif "领先间隔" in hypothesis:
            best_rows = [
                result.get("best_absolute_lag") or {}
                for result in relationships.values()
                if result.get("best_absolute_lag")
            ]
            positive = [row for row in best_rows if int(row.get("lag", 0)) > 0]
            if not best_rows:
                status, evidence = "not_tested", "本轮没有可用滞后扫描结果。"
            else:
                status = "candidate_support" if positive else "not_supported"
                evidence = f"{len(positive)}/{len(best_rows)} 个变量的绝对相关峰值出现在正领先位置。"
        else:
            status, evidence = "inconclusive", "已生成相关描述性结果，但当前没有预定义的确定性验收规则。"
        assessments.append(HypothesisAssessment(hypothesis=hypothesis, status=status, evidence=evidence))
    return assessments


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

    if quality.usable_for_eda:
        checks.append(EvaluationCheck(name="目标数据可用性", status="pass", message="目标序列满足最小样本要求。"))
    else:
        checks.append(EvaluationCheck(name="目标数据可用性", status="fail", message="目标序列不足以支持当前 EDA。"))

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
            )
        )

    uses_exogenous = any(
        step.enabled and step.tool in {"exogenous_profile", "relationship_analysis"} for step in plan.steps
    )
    in_scope_series = {summary["study"]["target"]}
    if uses_exogenous:
        in_scope_series.update(plan.selected_variables)
    severe_issues = [
        issue
        for issue in quality.issues
        if issue.severity in {"critical", "high"} and (issue.series is None or issue.series in in_scope_series)
    ]
    if severe_issues:
        checks.append(
            EvaluationCheck(
                name="数据风险",
                status="warning",
                message=f"检测到 {len(severe_issues)} 个高优先级数据或可获得性风险。",
            )
        )
        warnings.extend(
            f"{issue.series or '研究'}：{_issue_message(issue.code, issue.message)}" for issue in severe_issues[:6]
        )
    else:
        checks.append(EvaluationCheck(name="数据风险", status="pass", message="未检测到高优先级数据风险。"))

    expected_sections = {
        "price_profile": "price",
        "exogenous_profile": "exogenous",
        "relationship_analysis": "relationships",
    }
    missing_sections = [
        section
        for step in plan.enabled_steps
        if (section := expected_sections.get(step.tool)) is not None and section not in summary
    ]
    checks.append(
        EvaluationCheck(
            name="计划执行完整性",
            status="fail" if missing_sections else "pass",
            message=(
                f"缺少计划结果：{', '.join(missing_sections)}。"
                if missing_sections
                else f"已完成 {len(plan.enabled_steps)} 个确认步骤。"
            ),
        )
    )

    price = summary.get("price")
    if price:
        distribution = price.get("distribution")
        extremes = price.get("extremes")
        if distribution and extremes:
            findings.append(
                "电价均值/中位数为 "
                f"{_fmt(distribution.get('mean'))}/{_fmt(distribution.get('median'))}，"
                f"高价尖峰 {extremes.get('high_spike_count', 0)} 个。"
            )
        elif distribution:
            findings.append(
                f"电价均值/中位数为 {_fmt(distribution.get('mean'))}/{_fmt(distribution.get('median'))}。"
            )
        elif extremes:
            findings.append(f"本轮识别到高价尖峰 {extremes.get('high_spike_count', 0)} 个。")

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
            lag_text = f"，位置为 {lag} 个间隔" if lag is not None else ""
            if p_value is None:
                p_text = ""
            elif p_value == 0:
                p_text = "，未校正 p<1e-12"
            else:
                p_text = f"，未校正 p={p_value:.3g}"
            findings.append(f"{name} 的{method}系数为 {correlation:.3f}{lag_text}{p_text}。")
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
            )
        )
        relationship_methods = set(summary.get("relationships", {}).get("methods", []))
        if relationship_methods.intersection({"pearson", "spearman", "lag_scan"}):
            checks.append(
                EvaluationCheck(
                    name="时间序列混杂控制",
                    status="warning",
                    message="当前关系尚未控制共同趋势、季节性与序列自相关，只能作为候选关系。",
                )
            )
            warnings.append("关系系数可能受到共同趋势、日内/周内季节性和序列自相关影响。")
        if "lag_scan" in relationship_methods:
            warnings.append("最佳滞后来自多位置扫描，当前 p 值未进行多重检验校正。")
            followups.append("对候选变量先去趋势、去季节性，再进行带多重检验控制的滞后稳定性分析。")
    elif "relationships" in summary:
        checks.append(
            EvaluationCheck(name="关系证据", status="warning", message="所选变量未形成满足样本门槛的关系证据。")
        )
        followups.append("检查变量覆盖率、研究窗口和最小样本门槛后重新规划关系分析。")

    if any(issue.code == "lookahead_risk" for issue in quality.issues):
        followups.append("进入预测实验前，排除 observed_only 变量或补齐真实可获得时间。")
    if relationships:
        followups.append("将最有证据的关系转化为候选特征，并在滚动时间切分中验证增量价值。")
    if not findings:
        findings.append("本轮主要完成数据可用性审查；尚未执行足以支持关系结论的分析步骤。")

    statuses = {check.status for check in checks}
    if "fail" in statuses:
        decision = "revise"
        decision_text = "执行证据不完整，需要修订方案。"
    elif "warning" in statuses:
        decision = "revise"
        decision_text = "分析已完成，但存在风险；将在预算内尝试受控修订。"
    else:
        decision = "accept"
        decision_text = "分析计划执行完整，结果可作为下一阶段输入。"
    feedback_packets = [
        FeedbackPacket(
            feedback_id=hashlib.sha256(
                f"{plan.plan_id}:evaluation:{index}:{check.status}:{check.message}".encode()
            ).hexdigest()[:32],
            source="evaluator",
            code=f"evaluation_{index}_{check.status}",
            severity="error" if check.status == "fail" else "warning",
            message=check.message,
            observed=check.status,
            expected="pass",
            recommendation=(followups[index] if index < len(followups) else "根据评估结果收缩或修订研究方案。"),
            retryable=True,
            requires_user=False,
            created_at=plan.created_at,
        )
        for index, check in enumerate(checks)
        if check.status != "pass"
    ]
    return AgentEvaluation(
        decision=decision,
        summary=decision_text,
        checks=checks,
        hypothesis_assessments=_assess_hypotheses(plan, summary),
        findings=findings,
        warnings=warnings,
        suggested_followups=list(dict.fromkeys(followups)),
        feedback_packets=feedback_packets,
    )
