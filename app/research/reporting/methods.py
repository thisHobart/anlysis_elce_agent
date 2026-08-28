"""Build the method and verification ledger that stands behind one report.

``report.md`` shows evidence; this document carries everything a reviewer needs
to judge it: why these steps were chosen, how each function was parameterised
and read, which thresholds applied, how every planned hypothesis was resolved,
and which validity checks the evaluator ran. Both files are rendered from the
same executed metadata so they cannot drift apart.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from app.research.agent.schemas import AgentEvaluation, EDAPlan
from app.research.evaluation.criteria import DECISION_CRITERIA, DecisionCriterion
from app.research.skills.contracts import ResearchProtocol
from app.research.tools.catalog import FUNCTION_CATALOG, STAGE_TITLES

CRITERIA_BY_FUNCTION: dict[str, tuple[str, ...]] = {
    "price_descriptive_distribution": ("distribution_shape",),
    "price_rolling_mean_std": ("volatility_regime",),
    "price_calendar_group_profile": ("calendar_effect",),
    "price_seasonal_decomposition": ("seasonal_strength",),
    "exogenous_descriptive_distribution": ("driver_coverage",),
    "exogenous_linear_index_trend": ("driver_drift",),
}

REPORT_LOCATIONS: dict[str, str] = {
    "data_quality": "数据说明 · 研究时段与数据列",
    "price_descriptive_distribution": "电价自身规律 · 电价水平与分布",
    "price_rolling_mean_std": "电价自身规律 · 电价走势与波动",
    "price_tukey_outer_fence": "电价自身规律 · 极端值识别",
    "price_spike_regime_profile": "电价自身规律 · 尖峰与负价状态",
    "price_duration_curve": "电价自身规律 · 电价持续曲线",
    "price_calendar_group_profile": "电价自身规律 · 日历规律",
    "price_seasonal_decomposition": "电价自身规律 · 趋势与季节分解",
    "price_stationarity_tests": "电价自身规律 · 平稳性检验",
    "price_lag_autocorrelation": "电价自身规律 · 记忆结构",
    "price_partial_autocorrelation": "电价自身规律 · 记忆结构",
    "exogenous_descriptive_distribution": "影响因素质量 · 覆盖率与取值范围",
    "exogenous_iqr_outliers": "影响因素质量 · 覆盖率与取值范围",
    "exogenous_linear_index_trend": "影响因素质量 · 覆盖率与取值范围",
    "exogenous_pearson_collinearity": "影响因素质量 · 变量之间的重复",
    "exogenous_variance_inflation": "影响因素质量 · 变量之间的重复",
    "exogenous_stationarity_tests": "影响因素质量 · 因素自身的平稳性",
    "relationship_scipy_pearson_pairwise": "电价与影响因素的关系 · 同期与领先关系",
    "relationship_scipy_spearman_pairwise": "电价与影响因素的关系 · 同期与领先关系",
    "relationship_pearson_positive_lead_scan": "电价与影响因素的关系 · 同期与领先关系",
    "relationship_mutual_information_scan": "电价与影响因素的关系 · 非线性依赖",
    "relationship_granger_causality_scan": "电价与影响因素的关系 · 样本内预测前置性",
    "relationship_rolling_correlation_stability": "电价与影响因素的关系 · 关系随时间的稳定性",
    "price_segment_distribution_comparison": "分段对比 · 电价分段",
    "relationship_pearson_segment_comparison": "分段对比 · 关系分段",
    "price_naive_baseline_benchmark": "可预测性基线 · 朴素基线误差底线",
    "price_variance_stabilization_check": "可预测性基线 · 方差稳定变换检查",
}

CHECK_LABELS = {"pass": "通过", "warning": "注意", "fail": "未通过"}

HYPOTHESIS_LABELS = {
    "candidate_support": "有证据支持",
    "not_supported": "证据不支持",
    "inconclusive": "证据不足以下结论",
    "not_tested": "本轮未检验",
}

SCOPE_LABELS = {
    "within_envelope": "可在原审批范围内自动修订",
    "needs_approval": "需要你批准新增函数或范围",
    "needs_data": "需要补充或修复数据",
    "needs_restatement": "需要改写该表述",
    "inherent": "方法论限制，不驱动修订",
}


@dataclass(frozen=True)
class ExecutedMethod:
    stage: str
    stage_title: str
    function: str
    title: str
    version: str
    description: str
    answers: str
    rationale: str
    parameters: dict[str, Any]
    interpretation_rule: str
    report_location: str | None
    evidence_location: str
    call_id: str
    work_id: str
    duration_ms: float | None
    output_hash: str


@dataclass(frozen=True)
class MethodDocument:
    run_id: str
    question: str
    objective: str
    revision: int
    revision_reason: str | None
    assumptions: tuple[str, ...]
    unverifiable_hypotheses: tuple[str, ...]
    skill_name: str
    skill_version: str
    protocol_id: str | None
    protocol_version: str | None
    methods: tuple[ExecutedMethod, ...]
    criteria: tuple[DecisionCriterion, ...]
    invariants: tuple[str, ...]
    evaluation: AgentEvaluation | None = None


def _protocol_rules(protocol: ResearchProtocol | None) -> dict[str, str]:
    if protocol is None:
        return {}
    return {
        function_name: rule
        for stage in protocol.stages
        for function_name, rule in stage.function_rules.items()
    }


def _evidence_location(function_name: str) -> str:
    spec = FUNCTION_CATALOG[function_name]
    if function_name == "data_quality":
        return "evidence/data_quality.json"
    pointer = f"/{spec.result_key}"
    if spec.evidence_field:
        pointer += f"/{spec.evidence_field}"
    return f"evidence/eda_summary.json#{pointer}"


def build_method_document(
    *,
    run_id: str,
    plan: EDAPlan,
    protocol: ResearchProtocol | None,
    execution_trace: list[dict[str, Any]],
    evaluation: AgentEvaluation | None = None,
) -> MethodDocument:
    """Create one immutable documentation model for Markdown and HTML renderers."""

    if protocol is not None and (
        protocol.protocol_id != plan.research_protocol_id or protocol.version != plan.research_protocol_version
    ):
        raise ValueError("研究方法说明使用的协议版本与锁定计划不一致")
    rules = _protocol_rules(protocol)
    trace_by_function = {str(item.get("function")): item for item in execution_trace}
    methods: list[ExecutedMethod] = []
    criterion_ids: set[str] = set()
    for step in plan.enabled_steps:
        spec = FUNCTION_CATALOG[step.function]
        trace = trace_by_function.get(step.function, {})
        criterion_ids.update(CRITERIA_BY_FUNCTION.get(step.function, ()))
        methods.append(
            ExecutedMethod(
                stage=spec.stage,
                stage_title=STAGE_TITLES.get(spec.stage, spec.stage),
                function=step.function,
                title=step.title,
                version=step.function_version,
                description=spec.description,
                answers=spec.answers,
                rationale=step.rationale,
                parameters=dict(trace.get("parameters") or step.parameters),
                interpretation_rule=rules.get(step.function, "当前 Skill 未声明额外判读规则。"),
                report_location=REPORT_LOCATIONS.get(step.function),
                evidence_location=_evidence_location(step.function),
                call_id=str(trace.get("call_id") or "未记录"),
                work_id=str(trace.get("work_id") or "未记录"),
                duration_ms=(float(trace["duration_ms"]) if trace.get("duration_ms") is not None else None),
                output_hash=str(trace.get("output_hash") or "未记录"),
            )
        )
    criteria = tuple(item for item in DECISION_CRITERIA if item.criterion_id in criterion_ids)
    return MethodDocument(
        run_id=run_id,
        question=plan.question,
        objective=plan.objective,
        revision=plan.revision,
        revision_reason=plan.revision_reason,
        assumptions=tuple(plan.assumptions),
        unverifiable_hypotheses=tuple(plan.unverifiable_hypotheses),
        skill_name=plan.skill_name,
        skill_version=plan.skill_version,
        protocol_id=plan.research_protocol_id,
        protocol_version=plan.research_protocol_version,
        methods=tuple(methods),
        criteria=criteria,
        invariants=tuple(protocol.invariants if protocol is not None else ()),
        evaluation=evaluation,
    )


def _parameter_text(parameters: dict[str, Any]) -> str:
    if not parameters:
        return "（无额外参数）"
    return "；".join(
        f"{key} = {json.dumps(value, ensure_ascii=False, sort_keys=True)}"
        for key, value in parameters.items()
        if key != "max_lag_limit"
    ) or "（无额外参数）"


def _cell(value: Any) -> str:
    text = "—" if value in (None, "") else str(value)
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ").strip() or "—"


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    if not rows:
        return []
    return [
        "",
        "| " + " | ".join(_cell(name) for name in headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *("| " + " | ".join(_cell(value) for value in row) + " |" for row in rows),
    ]


def _design_section(document: MethodDocument) -> list[str]:
    lines = [
        "",
        "## 1 本轮研究设计",
        "",
        f"**研究问题**：{document.question}",
        f"**研究目标**：{document.objective}",
    ]
    if document.revision > 1 and document.revision_reason:
        lines.append(f"**本版修订**：v{document.revision} — {document.revision_reason}")
    if document.assumptions:
        lines.extend(["", "**方案前提**", ""])
        lines.extend(f"- {item}" for item in document.assumptions)
    if document.evaluation is not None and document.evaluation.hypothesis_assessments:
        lines.extend(
            [
                "",
                (
                    f"本轮共登记 {len(document.evaluation.hypothesis_assessments)} 条待验证判断，"
                    "逐条验收结果见 §5。"
                ),
            ]
        )
    if document.unverifiable_hypotheses:
        lines.extend(["", "以下说法本轮没有对应的确定性检验，因此没有列入议程：", ""])
        lines.extend(f"- {item}" for item in document.unverifiable_hypotheses)
    grouped: dict[str, list[ExecutedMethod]] = {}
    for method in document.methods:
        grouped.setdefault(method.stage_title, []).append(method)
    lines.extend(["", "### 1.1 为什么选这些步骤"])
    for stage_title, methods in grouped.items():
        lines.extend(["", f"**{stage_title}**"])
        lines.extend(
            _table(
                ["分析步骤", "回答的问题", "选择理由"],
                [[method.title, method.answers, method.rationale] for method in methods],
            )
        )
    return lines


def _methods_section(document: MethodDocument) -> list[str]:
    lines = ["", "## 2 逐个方法"]
    previous_stage = ""
    for position, method in enumerate(document.methods, start=1):
        if method.stage != previous_stage:
            lines.extend(["", f"### {method.stage_title}"])
            previous_stage = method.stage
        duration = "未记录"
        if method.duration_ms is not None:
            duration = f"{method.duration_ms:,.0f} 毫秒"
        lines.extend(
            [
                "",
                f"#### {position}. {method.title} · `{method.function}` v{method.version}",
                "",
                f"- **做什么**：{method.description}",
                f"- **回答**：{method.answers}",
                f"- **本轮参数**：{_parameter_text(method.parameters)}",
                f"- **判读口径**：{method.interpretation_rule}",
            ]
        )
        if method.report_location:
            lines.append(f"- **报告位置**：report.md「{method.report_location}」")
        lines.extend(
            [
                f"- **结果位置**：`{method.evidence_location}`",
                (
                    f"- **执行标识**：call_id `{method.call_id}`；work_id `{method.work_id}`；"
                    f"output_hash `{method.output_hash}`；耗时 {duration}"
                ),
            ]
        )
    return lines


def _criteria_section(document: MethodDocument) -> list[str]:
    if not document.criteria:
        return []
    return [
        "",
        "## 3 判读阈值",
        "",
        "报告里每一句「越过 / 未越过阈值」都来自下表。阈值在执行前锁定，不随结果调整。",
        *_table(
            ["判据", "规则", "为什么这样定"],
            [[item.label, item.rule, item.rationale] for item in document.criteria],
        ),
    ]


def _validity_section(document: MethodDocument) -> list[str]:
    evaluation = document.evaluation
    if evaluation is None or not evaluation.checks:
        return []
    rows = [
        [
            check.name,
            CHECK_LABELS.get(check.status, check.status),
            check.message,
            check.remediation or "—",
        ]
        for check in evaluation.checks
    ]
    return [
        "",
        "## 4 有效性核验",
        "",
        "本轮结论成立的前提是否满足，由确定性评估器逐项检查；未通过项会进入下一轮修订议程。",
        *_table(["校验项", "结果", "说明", "处理方式"], rows),
    ]


def _hypothesis_section(document: MethodDocument) -> list[str]:
    evaluation = document.evaluation
    if evaluation is None or not evaluation.hypothesis_assessments:
        return []
    rows = [
        [
            item.hypothesis,
            HYPOTHESIS_LABELS.get(item.status, item.status),
            item.evidence,
            (
                item.remediation or SCOPE_LABELS.get(item.scope, item.scope)
                if item.status in {"not_tested", "inconclusive"}
                else "—"
            ),
        ]
        for item in evaluation.hypothesis_assessments
    ]
    return [
        "",
        "## 5 假设验收",
        "",
        (
            "每条判断由固定的证据位置和阈值判定，不由模型自行断言。"
            "「证据不支持」表示当前证据不足以支持该表述，不等于反向结论成立；"
            "措辞不同但指向同一议程条目的判断只保留一条。"
        ),
        *_table(["判断", "结论", "依据", "未决时如何处理"], rows),
    ]


def _boundaries_section(document: MethodDocument) -> list[str]:
    if not document.invariants:
        return []
    return ["", "## 6 方法边界", "", *[f"- {item}" for item in document.invariants]]


def render_methods_markdown(document: MethodDocument) -> str:
    lines = [
        "# 分析方法与验收",
        "",
        f"对应报告：[report.md](report.md) ｜ 运行编号：`{document.run_id}`",
        f"研究方法包：`{document.skill_name}@{document.skill_version}`",
        (
            f"领域协议：`{document.protocol_id}@{document.protocol_version}`"
            if document.protocol_id
            else "领域协议：未声明"
        ),
        f"本轮执行 {len(document.methods)} 个分析函数。",
        "",
        (
            "本文件回答三个问题：本轮为什么这样做（§1）、每一步具体怎么做和怎么读（§2、§3）、"
            "结论经过了哪些检查（§4、§5）。报告只呈现证据和结论，方法层面的依据全部在这里。"
        ),
        *_design_section(document),
        *_methods_section(document),
        *_criteria_section(document),
        *_validity_section(document),
        *_hypothesis_section(document),
        *_boundaries_section(document),
        "",
        "## 复核指引",
        "",
        "- 结论与图表：`report.md`",
        "- 数字证据：`evidence/`，每个函数的具体位置见 §2",
        "- 调用、数据指纹与输出哈希：`provenance/execution_trace.json`",
        "- 代码版本、运行环境及全部文件哈希：`manifest.json`",
        "",
    ]
    return "\n".join(lines)
