"""Render one self-contained HTML research report from deterministic evidence."""

from __future__ import annotations

from datetime import datetime
from html import escape
from typing import Any

from app.research.agent.schemas import AgentEvaluation, EDAPlan
from app.research.reporting.report_charts import FIGURE_TITLES
from app.research.reporting.svg_charts import format_number
from app.research.schemas.results import DataQualityReport
from app.research.schemas.study import StudyConfig
from app.research.tools.catalog import FUNCTION_CATALOG, STAGE_TITLES

DECISION_LABELS = {
    "accept": ("研究已收敛", "ok"),
    "revise": ("需要再迭代一轮", "warn"),
    "need_user": ("需要你来决定", "warn"),
    "reject": ("本轮结果不可用", "bad"),
}

CHECK_LABELS = {"pass": ("通过", "ok"), "warning": ("注意", "warn"), "fail": ("未通过", "bad")}

HYPOTHESIS_LABELS = {
    "candidate_support": ("有证据支持", "ok"),
    "not_supported": ("证据不支持", "muted"),
    "inconclusive": ("证据不足以下结论", "warn"),
    "not_tested": ("本轮未检验", "muted"),
}

AVAILABILITY_LABELS = {
    "known_at_timestamp": "预测时点已知",
    "forecast": "预测值",
    "observed_only": "仅事后观测",
    "unknown": "可获得性未知",
}

STYLE = """
:root{color-scheme:light;--ink:#1F2430;--muted:#667085;--faint:#98A2B3;--line:#E7EAEF;
--panel:#FFFFFF;--page:#F5F6F8;--accent:#3F51B5;--accent-soft:#EEF0FB;--ok:#0F766E;
--ok-soft:#E6F2F0;--warn:#B45309;--warn-soft:#FDF3E4;--bad:#BE123C;--bad-soft:#FCEBEF;}
*{box-sizing:border-box;}
body{margin:0;background:var(--page);color:var(--ink);
font-family:'Segoe UI','Microsoft YaHei',-apple-system,BlinkMacSystemFont,'Helvetica Neue',Arial,sans-serif;
font-size:15px;line-height:1.65;}
.page{max-width:1080px;margin:0 auto;padding:32px 24px 80px;}
header.masthead{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:26px 28px;}
.eyebrow{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--faint);margin:0 0 8px;}
h1{font-size:26px;line-height:1.35;margin:0 0 10px;font-weight:650;}
.question{margin:0;color:var(--muted);font-size:15px;}
.badges{display:flex;flex-wrap:wrap;gap:8px;margin-top:16px;}
.badge{font-size:12.5px;padding:4px 11px;border-radius:999px;background:var(--accent-soft);color:var(--accent);}
.badge.ok{background:var(--ok-soft);color:var(--ok);}
.badge.warn{background:var(--warn-soft);color:var(--warn);}
.badge.bad{background:var(--bad-soft);color:var(--bad);}
.badge.muted{background:#F1F2F4;color:var(--muted);}
nav.toc{display:flex;flex-wrap:wrap;gap:8px;margin:20px 0 4px;}
nav.toc a{font-size:13px;text-decoration:none;color:var(--muted);background:var(--panel);
border:1px solid var(--line);border-radius:999px;padding:5px 13px;}
nav.toc a:hover{color:var(--accent);border-color:var(--accent);}
section{background:var(--panel);border:1px solid var(--line);border-radius:14px;
padding:24px 28px;margin-top:20px;}
section>h2{font-size:19px;margin:0 0 4px;font-weight:640;}
section>p.lede{margin:0 0 18px;color:var(--muted);font-size:13.5px;}
h3{font-size:15px;margin:24px 0 10px;font-weight:600;}
h3:first-of-type{margin-top:6px;}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));gap:12px;}
.tile{background:var(--page);border-radius:11px;padding:14px 16px;}
.tile .label{font-size:12px;color:var(--muted);}
.tile .value{font-size:21px;font-weight:640;margin-top:3px;letter-spacing:-.01em;}
.tile .hint{font-size:11.5px;color:var(--faint);margin-top:2px;}
ul.plain{margin:0;padding-left:20px;}
ul.plain li{margin:5px 0;}
ul.callouts{list-style:none;margin:0;padding:0;}
ul.callouts li{border-left:3px solid var(--line);padding:7px 0 7px 14px;margin:8px 0;}
ul.callouts li.ok{border-color:var(--ok);}
ul.callouts li.warn{border-color:var(--warn);}
ul.callouts li.bad{border-color:var(--bad);}
.scroll{overflow-x:auto;margin:6px 0 2px;}
table{border-collapse:collapse;width:100%;font-size:13.5px;min-width:520px;}
th,td{text-align:left;padding:9px 12px;border-bottom:1px solid var(--line);vertical-align:top;}
th{font-weight:600;color:var(--muted);font-size:12.5px;white-space:nowrap;}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;}
tr:last-child td{border-bottom:none;}
figure{margin:18px 0 6px;}
figure svg{display:block;border:1px solid var(--line);border-radius:10px;}
figcaption{font-size:12.5px;color:var(--muted);margin-top:7px;}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:18px;}
details{margin-top:14px;border-top:1px solid var(--line);padding-top:12px;}
summary{cursor:pointer;font-size:13.5px;color:var(--muted);}
details[open] summary{margin-bottom:10px;}
code{font-family:Consolas,'SFMono-Regular',Menlo,monospace;font-size:12.5px;
background:var(--page);padding:1px 5px;border-radius:4px;}
footer{margin-top:26px;color:var(--faint);font-size:12px;text-align:center;}
@media print{body{background:#FFF;}section,header.masthead{break-inside:avoid;border-radius:0;}nav.toc{display:none;}}
"""


def _fmt(value: Any, unit: str = "") -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return format_number(float(value), unit=unit)
    return "—" if value in (None, "") else escape(str(value))


def _percent(value: Any) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "—"
    return f"{float(value) * 100:.2f}%"


def _table(headers: list[str], rows: list[list[str]], *, numeric_from: int | None = None) -> str:
    if not rows:
        return ""

    def cell(tag: str, index: int, value: str) -> str:
        klass = ' class="num"' if numeric_from is not None and index >= numeric_from else ""
        return f"<{tag}{klass}>{value}</{tag}>"

    head = "".join(cell("th", index, escape(name)) for index, name in enumerate(headers))
    body = "".join(
        "<tr>" + "".join(cell("td", index, value) for index, value in enumerate(row)) + "</tr>" for row in rows
    )
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _figure(figures: dict[str, str], key: str, caption: str = "") -> str:
    markup = figures.get(key)
    if not markup:
        return ""
    text = caption or FIGURE_TITLES.get(key, "")
    return f"<figure>{markup}<figcaption>{escape(text)}</figcaption></figure>"


def _section(identifier: str, title: str, lede: str, body: str) -> str:
    if not body.strip():
        return ""
    return (
        f'<section id="{identifier}"><h2>{escape(title)}</h2>'
        f'<p class="lede">{escape(lede)}</p>{body}</section>'
    )


def _overview_tiles(
    config: StudyConfig,
    quality: DataQualityReport,
    summary: dict[str, Any],
) -> str:
    unit = config.target.unit if config.target.unit not in {"", "unknown"} else ""
    price = summary.get("price") or {}
    tiles: list[tuple[str, str, str]] = [
        (
            "研究时段",
            f"{str(quality.alignment.start_time)[:10]} → {str(quality.alignment.end_time)[:10]}",
            f"{quality.alignment.expected_rows:,} 个时间点 · {config.study.frequency}",
        ),
        (
            "目标数据完整度",
            _percent(quality.series[config.target.name].aligned_coverage_rate),
            f"有效值 {quality.series[config.target.name].aligned_non_null_rows:,}",
        ),
    ]
    distribution = price.get("distribution") or {}
    if distribution:
        tiles.append(
            (
                "电价中位数",
                _fmt(distribution.get("median"), unit),
                f"均值 {_fmt(distribution.get('mean'))} · 标准差 {_fmt(distribution.get('std'))}",
            )
        )
    regime = price.get("spike_regime") or {}
    if regime:
        tiles.append(
            (
                "尖峰时段占比",
                _percent(regime.get("high_spike_share")),
                f"负价时段 {_percent(regime.get('negative_share'))}",
            )
        )
    stationarity = price.get("stationarity") or {}
    if stationarity:
        verdict_text = {
            "stationary": "围绕稳定水平波动",
            "unit_root": "存在单位根，需差分",
            "trend_or_break_suspected": "疑似趋势或结构突变",
            "inconclusive": "结论不明确",
        }.get(str(stationarity.get("verdict")), "—")
        tiles.append(("平稳性", verdict_text, f"建议变换：{stationarity.get('recommended_transform')}"))
    baselines = price.get("naive_baselines") or {}
    if baselines:
        tiles.append(
            (
                "朴素基线误差",
                _fmt(baselines.get("best_mae"), unit),
                f"最优基线：{baselines.get('best_baseline_label')}",
            )
        )
    decomposition = price.get("decomposition") or {}
    if decomposition:
        remainder = (decomposition.get("variance_share") or {}).get("remainder")
        tiles.append(("难以解释的残差占比", _percent(remainder), "越低说明规律越强"))
    if summary.get("selected_variables"):
        tiles.append(
            ("纳入研究的影响因素", str(len(summary["selected_variables"])), "、".join(summary["selected_variables"][:3]))
        )
    cells = "".join(
        f'<div class="tile"><div class="label">{escape(label)}</div>'
        f'<div class="value">{value}</div><div class="hint">{escape(hint)}</div></div>'
        for label, value, hint in tiles
    )
    return f'<div class="tiles">{cells}</div>'


def _conclusion_section(evaluation: AgentEvaluation) -> str:
    findings = "".join(f"<li>{escape(item)}</li>" for item in evaluation.findings)
    body = f'<h3>主要发现</h3><ul class="plain">{findings}</ul>' if findings else ""
    if evaluation.warnings:
        items = "".join(f'<li class="warn">{escape(item)}</li>' for item in evaluation.warnings)
        body += f'<h3>需要留意的限制</h3><ul class="callouts">{items}</ul>'
    if evaluation.suggested_followups:
        items = "".join(f"<li>{escape(item)}</li>" for item in evaluation.suggested_followups)
        body += f'<h3>建议的下一步</h3><ul class="plain">{items}</ul>'
    checks = [
        [
            (
                f'<span class="badge {CHECK_LABELS.get(check.status, ("", "muted"))[1]}">'
                f"{CHECK_LABELS.get(check.status, (check.status, ''))[0]}</span>"
            ),
            escape(check.name),
            escape(check.message),
        ]
        for check in evaluation.checks
    ]
    if checks:
        body += "<h3>门禁检查</h3>" + _table(["结果", "检查项", "说明"], checks)
    assessments = [
        [
            (
                f'<span class="badge {HYPOTHESIS_LABELS.get(item.status, ("", "muted"))[1]}">'
                f"{HYPOTHESIS_LABELS.get(item.status, (item.status, ''))[0]}</span>"
            ),
            escape(item.hypothesis),
            escape(item.evidence),
        ]
        for item in evaluation.hypothesis_assessments
    ]
    if assessments:
        body += "<h3>逐条假设的验收结果</h3>" + _table(["结论", "假设", "依据"], assessments)
    return body


def _plan_section(plan: EDAPlan) -> str:
    grouped: dict[str, list[Any]] = {}
    for step in plan.enabled_steps:
        grouped.setdefault(FUNCTION_CATALOG[step.function].stage, []).append(step)
    body = ""
    for stage, steps in grouped.items():
        rows = [
            [escape(step.title), escape(FUNCTION_CATALOG[step.function].answers), escape(step.rationale)]
            for step in steps
        ]
        body += f"<h3>{escape(STAGE_TITLES.get(stage, stage))}</h3>" + _table(
            ["分析步骤", "回答的问题", "选择理由"], rows
        )
    if plan.hypotheses:
        items = "".join(f"<li>{escape(item)}</li>" for item in plan.hypotheses)
        body += f'<h3>本轮想验证的假设</h3><ul class="plain">{items}</ul>'
    if plan.assumptions:
        items = "".join(f"<li>{escape(item)}</li>" for item in plan.assumptions)
        body += f'<h3>方案前提</h3><ul class="plain">{items}</ul>'
    return body


def _quality_section(config: StudyConfig, quality: DataQualityReport, summary: dict[str, Any]) -> str:
    in_scope = {config.target.name, *(summary.get("selected_variables") or [])}
    rows = [
        [
            escape(name),
            "目标电价" if name == config.target.name else "影响因素",
            _percent(report.aligned_coverage_rate),
            f"{report.aligned_non_null_rows:,}",
            f"{report.missing_interval_count:,}",
            escape(AVAILABILITY_LABELS.get(report.availability_type, report.availability_type)),
        ]
        for name, report in quality.series.items()
        if name in in_scope
    ]
    body = _table(
        ["数据列", "角色", "完整度", "有效值", "缺口", "预测时可用性"], rows, numeric_from=2
    )
    issues = [issue for issue in quality.issues if issue.severity in {"critical", "high"}]
    if issues:
        items = "".join(
            f'<li class="warn">{escape(issue.series or "整体")}：{escape(issue.message)}</li>' for issue in issues[:10]
        )
        body += f'<h3>高优先级数据风险</h3><ul class="callouts">{items}</ul>'
    return body


def _price_section(summary: dict[str, Any], figures: dict[str, str], unit: str) -> str:
    price = summary.get("price") or {}
    if not price:
        return ""
    body = ""
    distribution = price.get("distribution")
    if distribution:
        rows = [
            [
                _fmt(distribution.get("mean"), unit),
                _fmt(distribution.get("median"), unit),
                _fmt(distribution.get("std")),
                _fmt(distribution.get("min")),
                _fmt(distribution.get("max")),
                _fmt(distribution.get("skewness")),
                _fmt(distribution.get("kurtosis")),
            ]
        ]
        body += _table(
            ["均值", "中位数", "标准差", "最小值", "最大值", "偏度", "峰度"], rows, numeric_from=0
        )
    body += _figure(figures, "price_timeseries")
    body += '<div class="grid2">'
    body += _figure(figures, "price_distribution")
    body += _figure(figures, "price_duration_curve")
    body += "</div>"

    regime = price.get("spike_regime")
    if regime:
        body += "<h3>尖峰与负价状态</h3>"
        body += _table(
            ["高价尖峰占比", "负价占比", "尖峰段数", "最长尖峰", "尖峰后仍是尖峰的概率", "最长平静段"],
            [
                [
                    _percent(regime.get("high_spike_share")),
                    _percent(regime.get("negative_share")),
                    str(regime.get("high_episodes", {}).get("episode_count", 0)),
                    f"{regime.get('high_episodes', {}).get('max_duration_intervals', 0)} 个间隔",
                    _percent((regime.get("clustering") or {}).get("probability_spike_follows_spike")),
                    f"{regime.get('longest_calm_streak_intervals', 0)} 个间隔",
                ]
            ],
            numeric_from=0,
        )
        hours = (regime.get("concentration") or {}).get("top_hours") or []
        if hours:
            body += (
                "<p>尖峰最集中的交割小时："
                + escape("、".join(f"{row['group']} 时（{row['spike_count']} 次）" for row in hours))
                + "</p>"
            )

    body += '<div class="grid2">'
    body += _figure(figures, "seasonal_patterns")
    body += _figure(figures, "price_weekday_profile")
    body += "</div>"
    body += _figure(figures, "price_month_profile")

    stationarity = price.get("stationarity")
    if stationarity:
        body += "<h3>平稳性检验</h3>"
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
                    escape(str(block.get("verdict", ""))),
                ]
            )
        body += _table(
            ["序列", "ADF 统计量", "ADF p 值", "KPSS 统计量", "KPSS p 值", "结论"], rows, numeric_from=1
        )
        body += f"<p>{escape(str(stationarity.get('verdict_text', '')))} 建议变换：<code>{escape(str(stationarity.get('recommended_transform')))}</code></p>"

    body += _figure(figures, "price_decomposition_variance")
    body += '<div class="grid2">'
    body += _figure(figures, "price_daily_shape")
    body += _figure(figures, "price_autocorrelation")
    body += "</div>"
    body += _figure(figures, "price_partial_autocorrelation")
    body += _figure(figures, "segment_price_comparison")
    return body


def _driver_section(summary: dict[str, Any], figures: dict[str, str]) -> str:
    exogenous = summary.get("exogenous") or {}
    if not exogenous:
        return ""
    body = ""
    series = exogenous.get("series") or {}
    rows = [
        [
            escape(name),
            _percent(item.get("coverage_rate")),
            _fmt(item.get("mean")),
            _fmt(item.get("std")),
            _fmt(item.get("min")),
            _fmt(item.get("max")),
            str(item.get("outlier_count", "—")),
        ]
        for name, item in series.items()
    ]
    if rows:
        body += _table(
            ["影响因素", "完整度", "均值", "标准差", "最小值", "最大值", "异常点"], rows, numeric_from=1
        )
    pairs = exogenous.get("strong_collinearity_pairs") or []
    if pairs:
        body += "<h3>高度重复的因素对</h3>"
        body += _table(
            ["因素 A", "因素 B", "相关系数", "样本数"],
            [
                [escape(row["left"]), escape(row["right"]), _fmt(row["correlation"]), f"{row['observations']:,}"]
                for row in pairs[:12]
            ],
            numeric_from=2,
        )
    body += _figure(figures, "exogenous_multicollinearity")
    driver_stationarity = exogenous.get("driver_stationarity")
    if driver_stationarity:
        rows = [
            [escape(name), escape(str(item.get("verdict", ""))), escape(str(item.get("verdict_text", "")))]
            for name, item in (driver_stationarity.get("series") or {}).items()
        ]
        body += "<h3>因素自身的平稳性</h3>" + _table(["影响因素", "结论", "说明"], rows)
    return body


def _relationship_section(summary: dict[str, Any], figures: dict[str, str]) -> str:
    relationships = summary.get("relationships") or {}
    if not relationships:
        return ""
    body = ""
    series = relationships.get("series") or {}
    rows = []
    for name, result in series.items():
        contemporaneous = result.get("contemporaneous") or {}
        pearson = contemporaneous.get("pearson") or {}
        spearman = contemporaneous.get("spearman") or {}
        best = result.get("best_absolute_lag") or {}
        rows.append(
            [
                escape(name),
                f"{int(pearson.get('observations') or spearman.get('observations') or 0):,}",
                _fmt(pearson.get("correlation")),
                _fmt(spearman.get("correlation")),
                _fmt(best.get("lag")),
                _fmt(best.get("correlation")),
            ]
        )
    if rows:
        body += _table(
            ["影响因素", "有效样本", "同期 Pearson", "同期 Spearman", "最强领先间隔", "该间隔相关"],
            rows,
            numeric_from=1,
        )
    body += _figure(figures, "correlation_ranking")
    body += _figure(figures, "lag_relationships")
    body += _figure(figures, "correlation_matrix")
    body += '<div class="grid2">'
    body += _figure(figures, "mutual_information")
    body += _figure(figures, "granger_precedence")
    body += "</div>"
    body += _figure(figures, "rolling_stability")
    body += _figure(figures, "segment_relationship_comparison")
    return body


def _readiness_section(summary: dict[str, Any], figures: dict[str, str], unit: str) -> str:
    price = summary.get("price") or {}
    baselines = price.get("naive_baselines")
    stabilization = price.get("variance_stabilization")
    if not baselines and not stabilization:
        return ""
    body = ""
    if baselines:
        rows = [
            [
                escape(str(row["label"])),
                f"{row['lag_intervals']}",
                _fmt(row.get("mae"), unit),
                _fmt(row.get("rmse")),
                _fmt(row.get("mae_ratio_to_persistence")),
                f"{row.get('observations', 0):,}",
            ]
            for row in baselines.get("baselines", [])
        ]
        body += _table(
            ["基线方法", "滞后间隔", "MAE", "RMSE", "相对持续法", "样本数"], rows, numeric_from=1
        )
        body += f"<p>{escape(str(baselines.get('error_floor_note', '')))}</p>"
        body += _figure(figures, "price_naive_baselines")
    if stabilization:
        raw = stabilization.get("raw_standardized") or {}
        transformed = stabilization.get("asinh_transformed") or {}
        body += "<h3>方差稳定变换检查</h3>"
        body += _table(
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
            numeric_from=1,
        )
        body += f"<p>{escape(str(stabilization.get('rationale', '')))}</p>"
    return body


def _technical_details(plan: EDAPlan, evaluation: AgentEvaluation, config: StudyConfig) -> str:
    rows = [
        ["研究方案编号", f"<code>{escape(plan.plan_id)}</code>"],
        ["方案版本", f"v{plan.revision}"],
        ["上一版方案", f"<code>{escape(plan.parent_plan_id or '无')}</code>"],
        ["数据指纹", f"<code>{escape(plan.data_fingerprint or '未记录')}</code>"],
        ["规划模型", escape(plan.planning_model or "未记录")],
        ["提示词版本", f"<code>{escape(plan.planning_prompt_version)}</code>"],
        ["研究方法包", f"<code>{escape(plan.skill_name)}@{escape(plan.skill_version)}</code>"],
        [
            "领域研究协议",
            f"<code>{escape(plan.research_protocol_id or '未记录')}@{escape(plan.research_protocol_version or '')}</code>",
        ],
        ["评估议程指纹", f"<code>{escape(evaluation.agenda_fingerprint or '未记录')}</code>"],
        ["市场", escape(config.study.market)],
        ["时区 / 采样频率", f"{escape(config.study.timezone)} / {escape(config.study.frequency)}"],
    ]
    function_rows = [
        [escape(step.title), f"<code>{escape(step.function)}</code>", f"<code>{escape(step.function_version)}</code>",
         f"<code>{escape(str(step.parameters))}</code>"]
        for step in plan.enabled_steps
    ]
    return (
        "<details><summary>技术细节与可复现信息（面向复核人员）</summary>"
        + _table(["项目", "值"], rows)
        + "<h3>执行的函数与参数</h3>"
        + _table(["分析步骤", "函数名", "版本", "参数"], function_rows)
        + "<h3>研究包文件</h3>"
        + '<ul class="plain">'
        + "".join(
            f"<li><code>{escape(name)}</code> — {escape(description)}</li>"
            for name, description in (
                ("report.html", "本报告"),
                ("report.md", "同一内容的 Markdown 版本"),
                ("research_plan.json", "锁定执行的方案、函数与参数"),
                ("eda_summary.json", "全部结构化统计结果"),
                ("data_quality.json", "数据体检结果"),
                ("agent_evaluation.json", "评估结论与门禁"),
                ("execution_trace.json", "逐函数执行记录与耗时"),
                ("aligned_data.parquet", "统一时间轴后的分析数据"),
                ("figures/", "报告中全部图表的 SVG 源文件"),
                ("manifest.json", "输入输出 SHA-256、代码版本与运行环境"),
            )
        )
        + "</ul></details>"
    )


def build_html_report(
    *,
    config: StudyConfig,
    quality: DataQualityReport,
    summary: dict[str, Any],
    plan: EDAPlan,
    evaluation: AgentEvaluation,
    figures: dict[str, str],
    run_id: str,
    created_at: datetime,
) -> str:
    """Compose the full report; only sections backed by real evidence are emitted."""

    unit = config.target.unit if config.target.unit not in {"", "unknown"} else ""
    decision_text, decision_class = DECISION_LABELS.get(evaluation.decision, (evaluation.decision, "muted"))
    sections = [
        _section(
            "overview",
            "研究概览",
            "本轮研究的关键数字，全部来自确定性统计函数的输出。",
            _overview_tiles(config, quality, summary),
        ),
        _section("conclusion", "结论与限制", evaluation.summary, _conclusion_section(evaluation)),
        _section("plan", "本轮做了哪些分析", "每个步骤都对应一个固定的统计过程，参数在执行前已锁定。", _plan_section(plan)),
        _section("quality", "数据体检", "分析前对时间轴、完整度和预测时可用性的检查结果。", _quality_section(config, quality, summary)),
        _section("price", "电价自身规律", "只看目标电价序列能得到的结构性证据。", _price_section(summary, figures, unit)),
        _section("drivers", "影响因素质量", "候选影响因素的数据质量、冗余与自身平稳性。", _driver_section(summary, figures)),
        _section(
            "relationships",
            "电价与影响因素的关系",
            "以下均为描述性关系证据，不构成因果结论，也不等于样本外预测增益。",
            _relationship_section(summary, figures),
        ),
        _section("readiness", "可预测性与建模就绪", "后续算法实验的误差底线与预处理建议。", _readiness_section(summary, figures, unit)),
    ]
    nav_targets = [
        ("overview", "概览"),
        ("conclusion", "结论"),
        ("plan", "分析步骤"),
        ("quality", "数据体检"),
        ("price", "电价规律"),
        ("drivers", "影响因素"),
        ("relationships", "关系证据"),
        ("readiness", "可预测性"),
    ]
    rendered = [item for item in sections if item]
    present = {item.split('id="', 1)[1].split('"', 1)[0] for item in rendered}
    nav = "".join(
        f'<a href="#{identifier}">{escape(label)}</a>' for identifier, label in nav_targets if identifier in present
    )
    badges = "".join(
        [
            f'<span class="badge {decision_class}">{escape(decision_text)}</span>',
            f'<span class="badge">{escape(config.study.name)}</span>',
            f'<span class="badge">{len(plan.enabled_steps)} 个分析步骤</span>',
            f'<span class="badge">{len(figures)} 张图表</span>',
            f'<span class="badge muted">{escape(created_at.astimezone().strftime("%Y-%m-%d %H:%M"))}</span>',
        ]
    )
    return (
        "<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{escape(config.study.name)} 电价研究报告</title>"
        f"<style>{STYLE}</style></head><body><div class=\"page\">"
        '<header class="masthead">'
        '<p class="eyebrow">电价预测前置研究</p>'
        f"<h1>{escape(plan.objective or plan.question)}</h1>"
        f'<p class="question">研究问题：{escape(plan.question)}</p>'
        f'<div class="badges">{badges}</div></header>'
        f'<nav class="toc">{nav}</nav>'
        + "".join(rendered)
        + '<section id="appendix"><h2>附录</h2><p class="lede">复核与复现所需的标识、版本与文件清单。</p>'
        + _technical_details(plan, evaluation, config)
        + "</section>"
        + f"<footer>运行编号 {escape(run_id)} · 本报告由确定性统计函数生成，未包含模型自由撰写的数值。</footer>"
        "</div></body></html>"
    )
