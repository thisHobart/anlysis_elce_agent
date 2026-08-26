"""Render an Agent-approved EDA plan and its evidence as Markdown."""

from __future__ import annotations

from typing import Any

from app.research.agent.schemas import AgentEvaluation, EDAPlan
from app.research.schemas.results import DataQualityReport
from app.research.schemas.study import StudyConfig


def _fmt(value: Any, decimals: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.{decimals}f}"
    return str(value)


def build_agent_eda_report(
    *,
    config: StudyConfig,
    quality: DataQualityReport,
    summary: dict[str, Any],
    plan: EDAPlan,
    evaluation: AgentEvaluation,
    figure_names: set[str],
) -> str:
    """Create a report that distinguishes Agent choices from tool evidence."""

    lines = [
        f"# {config.study.name}: Agent-led EDA",
        "",
        f"> 研究问题：{plan.question}",
        "",
        "> 本报告为描述性探索结果；相关性、滞后关系和分组差异均不构成因果证据。",
        "",
        "## Agent 研究方案",
        "",
        f"- 方案编号：`{plan.plan_id}`",
        f"- 数据指纹：`{plan.data_fingerprint or '未记录'}`",
        f"- 方案版本：`v{plan.revision}`",
        f"- 父方案：`{plan.parent_plan_id or '无'}`",
        f"- 规划方式：`{plan.planner}`",
        f"- 规划模型：`{plan.planning_model or '未记录'}`",
        f"- 提示词版本：`{plan.planning_prompt_version}`",
        f"- 激活 Skill：`{plan.skill_name}@{plan.skill_version}`",
        (
            f"- 领域研究协议：`{plan.research_protocol_id}@{plan.research_protocol_version}`"
            if plan.research_protocol_id
            else "- 领域研究协议：`未记录`"
        ),
        f"- 研究目标：{plan.objective}",
        f"- 所选变量：{', '.join(plan.selected_variables) if plan.selected_variables else '无'}",
        "",
        "| 执行 | 研究函数 | 调用名 | 函数版本 | Agent 理由 | 参数 |",
        "|---|---|---|---|---|---|",
    ]
    for step in plan.steps:
        lines.append(
            f"| {'是' if step.enabled else '否'} | {step.title} | `{step.tool}` | `{step.tool_version}` | "
            f"{step.rationale} | "
            f"`{step.parameters}` |"
        )

    if plan.hypotheses:
        lines.extend(["", "### 待验证假设", ""])
        lines.extend(f"- {item}" for item in plan.hypotheses)
    if plan.planning_notes:
        lines.extend(["", "### 规划说明", ""])
        lines.extend(f"- {item}" for item in plan.planning_notes)
    if plan.revision_reason:
        lines.extend(["", "### 本版修订", "", f"- {plan.revision_reason}"])

    lines.extend(
        [
            "",
            "## 数据范围与质量",
            "",
            f"- 市场：`{config.study.market}`",
            f"- 目标：`{config.target.name}` ({config.target.unit})",
            f"- 时间范围：`{quality.alignment.start_time}` 至 `{quality.alignment.end_time}`",
            f"- 频率/时区：`{config.study.frequency}` / `{config.study.timezone}`",
            f"- 对齐时间点：`{quality.alignment.expected_rows:,}`",
            "",
            "| 变量 | 覆盖率 | 有效值 | 缺失间隔 | 异常值 | 可获得性 |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for report in quality.series.values():
        lines.append(
            f"| {report.name} | {report.aligned_coverage_rate:.2%} | {report.aligned_non_null_rows:,} | "
            f"{report.missing_interval_count:,} | {report.outlier_count:,} | {report.availability_type} |"
        )

    price = summary.get("price")
    if price:
        lines.extend(
            [
                "",
                "## 电价结构分析",
                "",
                f"- 观测数/覆盖率：**{price.get('observations', 0):,} / {price.get('coverage_rate', 0):.2%}**。",
                f"- 本轮方法：**{', '.join(price.get('methods', [])) or '完整画像'}**。",
            ]
        )
        distribution = price.get("distribution")
        if distribution:
            lines.extend(
                [
                    f"- 均值/中位数：**{_fmt(distribution['mean'])} / {_fmt(distribution['median'])}**。",
                    f"- 最小值/最大值：**{_fmt(distribution['min'])} / {_fmt(distribution['max'])}**。",
                ]
            )
        extremes = price.get("extremes")
        if extremes:
            lines.append(
                f"- 高价尖峰：**{extremes['high_spike_count']:,}**；"
                f"负价：**{price.get('signs', {}).get('negative_count', 0):,}**。"
            )
        for key, title in (
            ("price_timeseries", "电价时序"),
            ("price_distribution", "电价分布"),
            ("seasonal_patterns", "季节性"),
        ):
            if key in figure_names:
                lines.extend(["", f"![{title}](figures/{key}.svg)"])

    exogenous = summary.get("exogenous")
    if exogenous:
        lines.extend(["", "## 外生变量画像", ""])
        lines.append(f"- 已分析变量数：**{len(exogenous['series'])}**。")
        lines.append(f"- 强共线性变量对：**{len(exogenous['strong_collinearity_pairs'])}**。")

    relationships = summary.get("relationships", {}).get("series", {})
    if relationships:
        lines.extend(
            [
                "",
                "## 电价—外生变量关系",
                "",
                "| 变量 | Pairwise N | Pearson | Spearman | 最强领先间隔 | 领先相关系数 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for name, result in relationships.items():
            pearson = result.get("contemporaneous", {}).get("pearson", {})
            spearman = result.get("contemporaneous", {}).get("spearman", {})
            best = result.get("best_absolute_lag") or {}
            lines.append(
                f"| {name} | {int(pearson.get('observations', spearman.get('observations', 0))):,} | "
                f"{_fmt(pearson.get('correlation'))} | {_fmt(spearman.get('correlation'))} | "
                f"{_fmt(best.get('lag'))} | {_fmt(best.get('correlation'))} |"
            )
        if "correlation_matrix" in figure_names:
            lines.extend(["", "![相关矩阵](figures/correlation_matrix.svg)"])
        if "lag_relationships" in figure_names:
            lines.extend(["", "![滞后关系](figures/lag_relationships.svg)"])

    lines.extend(["", "## 评估器结论", "", f"**{evaluation.decision.upper()}** — {evaluation.summary}", ""])
    lines.extend(f"- [{check.status}] {check.name}：{check.message}" for check in evaluation.checks)
    if evaluation.hypothesis_assessments:
        lines.extend(
            [
                "",
                "### 假设验收",
                "",
                "| 状态 | 假设 | 证据 |",
                "|---|---|---|",
            ]
        )
        lines.extend(
            f"| {item.status} | {item.hypothesis} | {item.evidence} |"
            for item in evaluation.hypothesis_assessments
        )
    lines.extend(["", "### 主要发现", ""])
    lines.extend(f"- {finding}" for finding in evaluation.findings)
    if evaluation.warnings:
        lines.extend(["", "### 风险与限制", ""])
        lines.extend(f"- {warning}" for warning in evaluation.warnings)
    if evaluation.suggested_followups:
        lines.extend(["", "### 下一步建议", ""])
        lines.extend(f"- {item}" for item in evaluation.suggested_followups)

    lines.extend(
        [
            "",
            "## 可复现研究包",
            "",
            "- `conversation.json`：触发本轮研究的对话",
            "- `research_plan.json`：用户确认后的工具、变量和参数",
            "- `execution_trace.json`：逐步骤运行状态和耗时",
            "- `data_quality.json`：数据质量证据",
            "- `eda_summary.json`：工具计算的结构化结果",
            "- `agent_evaluation.json`：评估器判断和后续建议",
            "- `aligned_data.parquet`：统一时间轴数据",
            "- `manifest.json`：输入、输出、代码和运行环境哈希",
            "",
        ]
    )
    return "\n".join(lines)
