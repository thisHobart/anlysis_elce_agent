"""Render one complete research report as Markdown from deterministic evidence.

The report answers one question only: what does this run's evidence show? Every
section is a figure or a table plus the reading of it. Method definitions,
parameters, thresholds, hypothesis verdicts and validity checks belong to
``methods.md``; keeping them out of here is what stops the same fact from being
told three times.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.research.agent.schemas import AgentEvaluation, EDAPlan
from app.research.evaluation.issues import issue_line
from app.research.evaluation.wording import stationarity_text
from app.research.reporting import readings
from app.research.reporting.capabilities import FIGURE_TITLES
from app.research.reporting.readings import format_statistic
from app.research.schemas.results import DataQualityReport
from app.research.schemas.study import StudyConfig

DECISION_LABELS = {
    "accept": "研究已收敛",
    "revise": "需要再迭代一轮",
    "need_user": "需要你来决定",
    "reject": "本轮结果不可用",
}

AVAILABILITY_LABELS = {
    "known_at_timestamp": "预测时点已知",
    "forecast": "预测值",
    "observed_only": "仅事后观测",
    "unknown": "可获得性未知",
}

SEGMENT_KIND_LABELS = {
    "hours": "日内时段",
    "months": "月份/季节",
    "time_range": "时间区间",
}

PACKAGE_FILES = (
    ("report.md", "本报告"),
    ("methods.md", "本轮方法、参数、判读阈值、假设验收与方法边界"),
    ("figures/", "报告中全部图表的 SVG 源文件"),
    ("evidence/", "全部结构化统计结果、数据可用性核验与评估记录"),
    ("provenance/", "锁定的方案、逐函数执行记录、对话与循环上下文"),
    ("data/aligned_data.parquet", "统一时间轴后的分析数据"),
    ("manifest.json", "输入输出 SHA-256、代码版本与运行环境"),
)


@dataclass(frozen=True)
class Block:
    """One titled piece of evidence: its figures, its reading, and its numbers."""

    title: str
    lines: list[str]


def _fmt(value: Any, unit: str = "") -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return format_statistic(value, unit)
    return "—" if value in (None, "") else str(value)


def _percent(value: Any) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "—"
    return f"{float(value) * 100:.2f}%"


def _cell(value: Any) -> str:
    """Keep any value inside one Markdown table cell."""

    text = "—" if value in (None, "") else str(value)
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ").strip() or "—"


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    if not rows:
        return []
    lines = [
        "| " + " | ".join(_cell(name) for name in headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(_cell(value) for value in row) + " |" for row in rows)
    return ["", *lines]


def _bullets(items: list[str]) -> list[str]:
    return ["", *[f"- {str(item).strip()}" for item in items if str(item).strip()]]


def _text(value: str) -> list[str]:
    """Emit one paragraph, or nothing when the evidence produced no sentence."""

    return ["", value.strip()] if value and value.strip() else []


def _reading(value: str) -> list[str]:
    return ["", f"**读图**：{value.strip()}"] if value and value.strip() else []


def _reading_of_table(value: str) -> list[str]:
    return ["", f"**读表**：{value.strip()}"] if value and value.strip() else []


def _figure(figure_names: set[str], key: str) -> list[str]:
    """Reference a figure only when its SVG was actually written to the package."""

    if key not in figure_names:
        return []
    # Each SVG draws its own title, so the alt text only has to stand in when it cannot load.
    return ["", f"![{FIGURE_TITLES.get(key, key)}](figures/{key}.svg)"]


def _figures(figure_names: set[str], *keys: str) -> list[str]:
    lines: list[str] = []
    for key in keys:
        lines.extend(_figure(figure_names, key))
    return lines


def _block(title: str, *parts: list[str]) -> list[Block]:
    """Keep a subsection only when something real ended up inside it."""

    lines = [line for part in parts for line in part]
    if not any(line.strip() for line in lines):
        return []
    if not any(line.startswith("![") for line in lines):
        # The figure this reading was written for was not produced this run.
        lines = [line.replace("**读图**：", "**读表**：") for line in lines]
    return [Block(title=title, lines=lines)]


def _conclusion(evaluation: AgentEvaluation, quality: DataQualityReport, in_scope: set[str]) -> list[Block]:
    limits = list(evaluation.warnings)
    limits.extend(
        issue_line(issue)
        for issue in quality.issues
        if issue.severity in {"critical", "high"} and (issue.series is None or issue.series in in_scope)
    )
    return [
        *_block("本轮的判断", _bullets(list(evaluation.findings))),
        *_block("结论的适用边界", _bullets(list(dict.fromkeys(limits)))),
        *_block("建议的下一步", _bullets(list(evaluation.suggested_followups))),
    ]


def _scope(config: StudyConfig, quality: DataQualityReport, in_scope: set[str]) -> list[Block]:
    target = quality.series[config.target.name]
    alignment = quality.alignment
    unit = config.target.unit if config.target.unit not in {"", "unknown"} else "未声明单位"
    summary_text = (
        f"统一时间轴为 {str(alignment.start_time)[:16]} → {str(alignment.end_time)[:16]}"
        f"（{alignment.timezone}，{alignment.frequency}），共 {alignment.expected_rows:,} 个时间点。"
        f"目标序列 {config.target.name}（{unit}）有效值 {target.aligned_non_null_rows:,} 个，"
        f"完整度 {_percent(target.aligned_coverage_rate)}，缺口 {target.missing_interval_count:,} 个间隔。"
        f"下表只列本轮实际进入分析的数据列；「预测时可用性」决定它们能否进入后续预测实验。"
    )
    rows = [
        [
            name,
            "目标电价" if name == config.target.name else "影响因素",
            _percent(report.aligned_coverage_rate),
            f"{report.aligned_non_null_rows:,}",
            f"{report.missing_interval_count:,}",
            AVAILABILITY_LABELS.get(report.availability_type, report.availability_type),
        ]
        for name, report in quality.series.items()
        if name in in_scope
    ]
    return _block(
        "研究时段与数据列",
        _text(summary_text),
        _table(["数据列", "角色", "完整度", "有效值", "缺口", "预测时可用性"], rows),
    )


def _price(summary: dict[str, Any], figure_names: set[str], unit: str) -> list[Block]:
    price = summary.get("price") or {}
    if not price:
        return []
    blocks: list[Block] = []

    distribution = price.get("distribution") or {}
    if distribution:
        blocks.extend(
            _block(
                "电价水平与分布",
                _figure(figure_names, "price_distribution"),
                _reading(readings.distribution_reading(distribution, unit)),
                _table(
                    ["均值", "中位数", "标准差", "最小值", "最大值", "偏度", "超额峰度"],
                    [
                        [
                            _fmt(distribution.get("mean"), unit),
                            _fmt(distribution.get("median"), unit),
                            _fmt(distribution.get("std")),
                            _fmt(distribution.get("min")),
                            _fmt(distribution.get("max")),
                            _fmt(distribution.get("skewness")),
                            _fmt(distribution.get("kurtosis")),
                        ]
                    ],
                ),
            )
        )

    blocks.extend(
        _block(
            "电价走势与波动",
            _figure(figure_names, "price_timeseries"),
            _reading(readings.timeseries_reading(price, unit)),
        )
    )

    extremes = price.get("extremes") or {}
    if extremes:
        blocks.extend(_block("极端值识别", _reading_of_table(readings.extremes_reading(extremes))))

    regime = price.get("spike_regime") or {}
    if regime:
        hours = (regime.get("concentration") or {}).get("top_hours") or []
        concentration = (
            _text("尖峰最集中的交割小时：" + "、".join(f"{row['group']} 时（{row['spike_count']} 次）" for row in hours))
            if hours
            else []
        )
        blocks.extend(
            _block(
                "尖峰与负价状态",
                _reading_of_table(readings.spike_regime_reading(regime)),
                _table(
                    ["高价尖峰占比", "负价占比", "尖峰段数", "最长尖峰", "尖峰后仍是尖峰的概率", "最长平静段"],
                    [
                        [
                            _percent(regime.get("high_spike_share")),
                            _percent(regime.get("negative_share")),
                            regime.get("high_episodes", {}).get("episode_count", 0),
                            f"{regime.get('high_episodes', {}).get('max_duration_intervals', 0):,} 个间隔",
                            _percent((regime.get("clustering") or {}).get("probability_spike_follows_spike")),
                            f"{regime.get('longest_calm_streak_intervals', 0):,} 个间隔",
                        ]
                    ],
                ),
                concentration,
            )
        )

    duration_curve = price.get("duration_curve") or {}
    blocks.extend(
        _block(
            "电价持续曲线",
            _figure(figure_names, "price_duration_curve"),
            _reading(readings.duration_curve_reading(duration_curve)),
        )
    )

    seasonality = price.get("seasonality") or {}
    blocks.extend(
        _block(
            "日历规律",
            _figures(figure_names, "seasonal_patterns", "price_weekday_profile", "price_month_profile"),
            _reading(readings.calendar_reading(seasonality)),
        )
    )

    decomposition = price.get("decomposition") or {}
    blocks.extend(
        _block(
            "趋势与季节分解",
            _figures(figure_names, "price_decomposition_variance", "price_daily_shape"),
            _reading(readings.decomposition_reading(decomposition)),
        )
    )

    stationarity = price.get("stationarity") or {}
    if stationarity:
        rows = []
        for label, key in (("原序列", "level"), ("一阶差分", "first_difference")):
            block = stationarity.get(key) or {}
            adf = block.get("adf") or {}
            kpss_result = block.get("kpss") or {}
            rows.append(
                [
                    label,
                    _fmt(adf.get("statistic")),
                    _fmt(adf.get("p_value")),
                    _fmt(kpss_result.get("statistic")),
                    _fmt(kpss_result.get("p_value")),
                    stationarity_text(block.get("verdict")),
                ]
            )
        blocks.extend(
            _block(
                "平稳性检验",
                _table(["序列", "ADF 统计量", "ADF p 值", "KPSS 统计量", "KPSS p 值", "结论"], rows),
                _reading_of_table(readings.stationarity_reading(stationarity)),
            )
        )

    blocks.extend(
        _block(
            "记忆结构",
            _figures(figure_names, "price_autocorrelation", "price_partial_autocorrelation"),
            _reading(readings.autocorrelation_reading(price.get("autocorrelation") or [])),
            _reading_of_table(
                readings.partial_autocorrelation_reading(price.get("partial_autocorrelation") or {})
            ),
        )
    )
    return blocks


def _drivers(summary: dict[str, Any], figure_names: set[str]) -> list[Block]:
    exogenous = summary.get("exogenous") or {}
    if not exogenous:
        return []
    blocks: list[Block] = []
    series = exogenous.get("series") or {}
    rows = [
        [
            name,
            _percent(item.get("coverage_rate")),
            _fmt(item.get("mean")),
            _fmt(item.get("std")),
            _fmt(item.get("min")),
            _fmt(item.get("max")),
            item.get("outlier_count", "—"),
        ]
        for name, item in series.items()
    ]
    blocks.extend(
        _block(
            "覆盖率与取值范围",
            _table(["影响因素", "完整度", "均值", "标准差", "最小值", "最大值", "异常点"], rows),
            _reading_of_table(readings.driver_coverage_reading(series)),
        )
    )

    pairs = exogenous.get("strong_collinearity_pairs") or []
    blocks.extend(
        _block(
            "变量之间的重复",
            _figure(figure_names, "exogenous_multicollinearity"),
            _reading(readings.collinearity_reading(exogenous)),
            _table(
                ["因素 A", "因素 B", "相关系数", "样本数"],
                [
                    [row["left"], row["right"], _fmt(row["correlation"]), f"{row['observations']:,}"]
                    for row in pairs[:12]
                ],
            ),
        )
    )

    driver_stationarity = exogenous.get("driver_stationarity") or {}
    if driver_stationarity:
        blocks.extend(
            _block(
                "因素自身的平稳性",
                _table(
                    ["影响因素", "结论", "说明"],
                    [
                        [name, stationarity_text(item.get("verdict")), item.get("verdict_text", "")]
                        for name, item in (driver_stationarity.get("series") or {}).items()
                    ],
                ),
                _reading_of_table(readings.driver_stationarity_reading(driver_stationarity)),
            )
        )
    return blocks


def _relationships(summary: dict[str, Any], figure_names: set[str]) -> list[Block]:
    relationships = summary.get("relationships") or {}
    if not relationships:
        return []
    series = relationships.get("series") or {}
    rows = []
    for name, result in series.items():
        contemporaneous = result.get("contemporaneous") or {}
        pearson = contemporaneous.get("pearson") or {}
        spearman = contemporaneous.get("spearman") or {}
        best = result.get("best_absolute_lag") or {}
        rows.append(
            [
                name,
                f"{int(pearson.get('observations') or spearman.get('observations') or 0):,}",
                _fmt(pearson.get("correlation")),
                _fmt(spearman.get("correlation")),
                f"{int(best['lag']):,}" if best.get("lag") is not None else "—",
                _fmt(best.get("correlation")),
            ]
        )
    return [
        *_block(
            "同期与领先关系",
            _figures(figure_names, "correlation_ranking", "lag_relationships", "correlation_matrix"),
            _reading(readings.relationship_reading(series)),
            _table(
                ["影响因素", "有效样本", "同期 Pearson", "同期 Spearman", "最强提前量（时段）", "该提前量的相关"],
                rows,
            ),
        ),
        *_block(
            "非线性依赖",
            _figure(figure_names, "mutual_information"),
            _reading(readings.mutual_information_reading(relationships.get("mutual_information") or {})),
        ),
        *_block(
            "样本内预测前置性",
            _figure(figure_names, "granger_precedence"),
            _reading(readings.granger_reading(relationships.get("granger_precedence") or {})),
        ),
        *_block(
            "关系随时间的稳定性",
            _figure(figure_names, "rolling_stability"),
            _reading(readings.rolling_stability_reading(relationships.get("rolling_stability") or {})),
        ),
    ]


def _comparisons(summary: dict[str, Any], figure_names: set[str], unit: str) -> list[Block]:
    comparisons = summary.get("comparisons") or {}
    price_comparisons = comparisons.get("price") or {}
    relationship_comparisons = comparisons.get("relationships") or {}
    if not price_comparisons and not relationship_comparisons:
        return []

    blocks: list[Block] = []
    for comparison_id, comparison in price_comparisons.items():
        grouped: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        for segment_id, row in (comparison.get("segments") or {}).items():
            kind = str((row.get("selector") or {}).get("kind") or "unknown")
            grouped.setdefault(kind, []).append((segment_id, row))
        tables: list[str] = []
        for kind, segments in grouped.items():
            tables.extend(_text(f"**{SEGMENT_KIND_LABELS.get(kind, kind)}**"))
            tables.extend(
                _table(
                    ["分段", "有效样本", "完整度", "均值", "中位数", "标准差", "25% 分位", "75% 分位"],
                    [
                        [
                            row.get("label") or segment_id,
                            f"{int(row.get('observations') or 0):,}",
                            _percent(row.get("coverage_rate")),
                            _fmt(row.get("mean"), unit),
                            _fmt(row.get("median"), unit),
                            _fmt(row.get("std")),
                            _fmt(row.get("q25")),
                            _fmt(row.get("q75")),
                        ]
                        for segment_id, row in segments
                    ],
                )
            )
        blocks.extend(
            _block(
                "电价分段",
                _text(f"**{comparison_id}**"),
                _figure(figure_names, "segment_price_comparison"),
                _reading(readings.segment_price_reading(comparison, unit)),
                tables,
            )
        )

    for comparison_id, comparison in relationship_comparisons.items():
        rows = []
        for variable, result in (comparison.get("series") or {}).items():
            for segment_id, row in (result.get("segments") or {}).items():
                kind = str((row.get("selector") or {}).get("kind") or "unknown")
                rows.append(
                    [
                        variable,
                        SEGMENT_KIND_LABELS.get(kind, kind),
                        row.get("label") or segment_id,
                        f"{int(row.get('observations') or 0):,}",
                        _fmt(row.get("correlation")),
                        _fmt(row.get("p_value")),
                    ]
                )
        blocks.extend(
            _block(
                "关系分段",
                _text(f"**{comparison_id}**"),
                _figure(figure_names, "segment_relationship_comparison"),
                _table(["影响因素", "分组维度", "分段", "有效样本", "Pearson", "p 值"], rows),
            )
        )
    return blocks


def _readiness(summary: dict[str, Any], figure_names: set[str], unit: str) -> list[Block]:
    price = summary.get("price") or {}
    baselines = price.get("naive_baselines") or {}
    stabilization = price.get("variance_stabilization") or {}
    blocks: list[Block] = []
    if baselines:
        rows = [
            [
                row["label"],
                row["lag_intervals"],
                _fmt(row.get("mae"), unit),
                _fmt(row.get("rmse")),
                _fmt(row.get("mae_ratio_to_persistence")),
                f"{row.get('observations', 0):,}",
            ]
            for row in baselines.get("baselines", [])
        ]
        blocks.extend(
            _block(
                "朴素基线误差底线",
                _figure(figure_names, "price_naive_baselines"),
                _reading(readings.baseline_reading(baselines, unit)),
                _table(["基线方法", "往前取几个时段", "平均绝对误差", "均方根误差", "与上一时刻法之比", "样本数"], rows),
            )
        )
    if stabilization:
        raw = stabilization.get("raw_standardized") or {}
        transformed = stabilization.get("asinh_transformed") or {}
        blocks.extend(
            _block(
                "方差稳定变换检查",
                _table(
                    ["处理", "偏度", "超额峰度", "超出 5 倍稳健标准差的占比"],
                    [
                        [
                            "稳健标准化",
                            _fmt(raw.get("skewness")),
                            _fmt(raw.get("excess_kurtosis")),
                            _percent(raw.get("share_beyond_5_robust_sd")),
                        ],
                        [
                            "asinh 变换后",
                            _fmt(transformed.get("skewness")),
                            _fmt(transformed.get("excess_kurtosis")),
                            _percent(transformed.get("share_beyond_5_robust_sd")),
                        ],
                    ],
                ),
                _reading_of_table(readings.variance_stabilization_reading(stabilization)),
            )
        )
    return blocks


def _appendix(plan: EDAPlan, config: StudyConfig, run_id: str) -> list[Block]:
    rows = [
        ["运行编号", f"`{run_id}`"],
        ["研究方案编号", f"`{plan.plan_id}`（v{plan.revision}，上一版 `{plan.parent_plan_id or '无'}`）"],
        ["数据指纹", f"`{plan.data_fingerprint or '未记录'}`"],
        ["规划方式 / 模型", f"`{plan.planner}` / {plan.planning_model or '未记录'}"],
        ["提示词版本", f"`{plan.planning_prompt_version}`"],
        ["研究方法包", f"`{plan.skill_name}@{plan.skill_version}`"],
        [
            "领域研究协议",
            f"`{plan.research_protocol_id or '未记录'}@{plan.research_protocol_version or ''}`",
        ],
        ["市场 / 时区 / 频率", f"{config.study.market} / {config.study.timezone} / {config.study.frequency}"],
    ]
    return [
        *_block("本轮运行标识", _table(["项目", "值"], rows)),
        *_block(
            "研究包文件",
            _bullets([f"`{name}` — {description}" for name, description in PACKAGE_FILES]),
        ),
    ]


def _render(sections: list[tuple[str, str, list[Block]]]) -> list[str]:
    """Number only the sections this run actually produced."""

    lines: list[str] = []
    index = 0
    for title, lede, blocks in sections:
        if not blocks:
            continue
        index += 1
        lines.extend(["", f"## {index} {title}", "", f"*{lede}*"])
        for position, block in enumerate(blocks, start=1):
            lines.extend(["", f"### {index}.{position} {block.title}", *block.lines])
    return lines


def build_agent_eda_report(
    *,
    config: StudyConfig,
    quality: DataQualityReport,
    summary: dict[str, Any],
    plan: EDAPlan,
    evaluation: AgentEvaluation,
    figure_names: set[str],
    run_id: str,
    created_at: datetime,
) -> str:
    """Compose the full report; only sections backed by real evidence are emitted."""

    unit = config.target.unit if config.target.unit not in {"", "unknown"} else ""
    in_scope = {config.target.name, *(summary.get("selected_variables") or [])}
    focus_steps = [step.title for step in plan.enabled_steps if step.function != "data_quality"]
    title = (
        f"本轮分析：{'、'.join(focus_steps)}"
        if plan.revision > 1 and focus_steps
        else plan.objective or plan.question
    )
    revision_context = (
        f" ｜ 本版修订：{plan.revision_reason}" if plan.revision > 1 and plan.revision_reason else ""
    )
    header = [
        f"# {title}",
        "",
        f"**研究问题**：{plan.question}{revision_context}",
        "",
        " ｜ ".join(
            [
                DECISION_LABELS.get(evaluation.decision, evaluation.decision),
                config.study.name,
                f"{str(quality.alignment.start_time)[:10]} → {str(quality.alignment.end_time)[:10]}",
                config.study.frequency,
                f"{len(plan.enabled_steps)} 个分析步骤",
                f"{len(figure_names)} 张图表",
                created_at.astimezone().strftime("%Y-%m-%d %H:%M"),
            ]
        ),
        "",
        (
            "> 本报告只描述数据里看到的规律：相关、提前量和分组差异都不能说明谁导致谁，"
            "也不代表放进模型就能提高预测精度。"
        ),
        "> 本轮用了哪些方法、参数、判读标准，以及每条判断的验收结果，见 [methods.md](methods.md)。",
    ]
    sections: list[tuple[str, str, list[Block]]] = [
        ("结论", evaluation.summary, _conclusion(evaluation, quality, in_scope)),
        ("数据说明", "本轮分析实际使用的时间范围、数据列，以及它们在预测的时候能不能拿到。", _scope(config, quality, in_scope)),
        ("电价自身规律", "只看电价这一条序列本身能得到的规律。", _price(summary, figure_names, unit)),
        ("影响因素质量", "候选影响因素的数据质量、彼此之间的重复程度，以及它们自身是否平稳。", _drivers(summary, figure_names)),
        (
            "电价与影响因素的关系",
            "以下都是描述性的关系证据：可以说明两者一起变化，但不能说明谁导致谁，也不代表放进模型就能提高预测精度。",
            _relationships(summary, figure_names),
        ),
        (
            "分段对比",
            "小时、月份和事件区间只在各自的维度内比较；均值和波动的差异都只是直接比较的结果。",
            _comparisons(summary, figure_names, unit),
        ),
        (
            "可预测性基线",
            "后续模型必须超过的误差底线，以及建模前的预处理建议。",
            _readiness(summary, figure_names, unit),
        ),
    ]
    lines = [*header, *_render(sections)]
    appendix = _appendix(plan, config, run_id)
    if appendix:
        lines.extend(["", "## 附录", "", "*复核与复现所需的标识与文件清单。*"])
        for position, block in enumerate(appendix, start=1):
            lines.extend(["", f"### A.{position} {block.title}", *block.lines])
    lines.extend(
        [
            "",
            "---",
            "",
            "本报告由确定性统计函数生成，未包含模型自由撰写的数值。",
            "",
        ]
    )
    return "\n".join(lines)
