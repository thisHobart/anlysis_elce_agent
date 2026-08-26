"""Turn structured EDA evidence into the ordered figure set used by the report."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.research.reporting import svg_charts as svg
from app.research.schemas.study import StudyConfig

MAX_TIMELINE_POINTS = 1_400
WEEKDAY_LABELS = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")

FIGURE_TITLES: dict[str, str] = {
    "price_timeseries": "电价时间序列",
    "price_distribution": "电价分布",
    "price_duration_curve": "电价持续曲线",
    "seasonal_patterns": "分小时平均电价",
    "price_weekday_profile": "分星期平均电价",
    "price_month_profile": "分月份平均电价",
    "price_autocorrelation": "电价自相关",
    "price_partial_autocorrelation": "电价偏自相关",
    "price_decomposition_variance": "趋势与季节成分方差占比",
    "price_daily_shape": "分解得到的日内季节形态",
    "price_naive_baselines": "朴素基线误差底线",
    "exogenous_multicollinearity": "影响因素方差膨胀因子",
    "correlation_ranking": "电价—因素同期相关排序",
    "correlation_matrix": "变量相关矩阵",
    "lag_relationships": "因素领先电价的相关曲线",
    "mutual_information": "非线性依赖强度（归一化互信息）",
    "granger_precedence": "加入变量历史后的残差方差下降",
    "rolling_stability": "关系随时间的稳定性",
    "segment_price_comparison": "分段电价均值对比",
    "segment_relationship_comparison": "分段关系强度对比",
}


def _percent_subtitle(label: str, value: object) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return label
    return f"{label} {float(value) * 100:.1f}%"


def _timeline_points(values: pd.Series) -> tuple[list[tuple[float, float]], str]:
    clean = values.dropna()
    if clean.empty:
        return [], ""
    plotted = values
    grain = "原始粒度"
    if len(values) > MAX_TIMELINE_POINTS:
        for rule, label in (("1h", "小时均值"), ("6h", "6 小时均值"), ("1D", "日均值")):
            reduced = values.resample(rule).mean()
            if len(reduced) <= MAX_TIMELINE_POINTS:
                plotted, grain = reduced, label
                break
        else:
            plotted, grain = values.resample("1D").mean(), "日均值"
    plotted = plotted.dropna()
    if plotted.empty:
        return [], grain
    total = max(1, len(plotted) - 1)
    points = [(index / total, float(value)) for index, value in enumerate(plotted.to_numpy())]
    return points, grain


def _histogram(values: pd.Series) -> tuple[list[float], list[int]]:
    clean = values.dropna().to_numpy(dtype=float)
    if clean.size == 0:
        return [], []
    bins = int(min(60, max(12, round(np.sqrt(clean.size)))))
    counts, edges = np.histogram(clean, bins=bins)
    return [float(edge) for edge in edges], [int(count) for count in counts]


def _group_rows(rows: list[dict[str, Any]], *, labels: tuple[str, ...] | None = None) -> tuple[list[str], list[float]]:
    ordered = sorted(rows, key=lambda row: int(row.get("group", 0)))
    names = [
        labels[int(row["group"])] if labels and 0 <= int(row["group"]) < len(labels) else str(row.get("label") or row.get("group"))
        for row in ordered
    ]
    return names, [row.get("mean") for row in ordered]


def _correlation_matrix_values(
    frame: pd.DataFrame, names: list[str], minimum: int
) -> list[list[float | None]]:
    correlation = frame[names].corr(min_periods=minimum)
    return [
        [None if pd.isna(correlation.loc[row, column]) else float(correlation.loc[row, column]) for column in names]
        for row in names
    ]


def build_report_figures(
    frame: pd.DataFrame,
    config: StudyConfig,
    summary: dict[str, Any],
) -> dict[str, str]:
    """Render only the figures the executed evidence actually supports."""

    figures: dict[str, str] = {}
    unit = config.target.unit if config.target.unit not in {"", "unknown"} else ""
    price = summary.get("price") or {}
    exogenous = summary.get("exogenous") or {}
    relationships = summary.get("relationships") or {}
    comparisons = summary.get("comparisons") or {}
    series = relationships.get("series") or {}
    selected = list(summary.get("selected_variables") or [])

    if price:
        points, grain = _timeline_points(frame[config.target.name])
        if points:
            figures["price_timeseries"] = svg.line_chart(
                points,
                title=FIGURE_TITLES["price_timeseries"],
                subtitle=f"{grain} · {price.get('observations', 0):,} 个有效时点",
                unit=unit,
                caption=f"{price.get('start_time', '')} → {price.get('end_time', '')}（{config.study.timezone}）",
            )

    if price.get("distribution") or price.get("extremes"):
        edges, counts = _histogram(frame[config.target.name])
        if counts:
            distribution = price.get("distribution") or {}
            figures["price_distribution"] = svg.histogram_chart(
                edges,
                counts,
                title=FIGURE_TITLES["price_distribution"],
                subtitle=(
                    f"中位数 {svg.format_number(distribution.get('median'))} · "
                    f"偏度 {svg.format_number(distribution.get('skewness'))}"
                ),
                unit=unit,
            )

    duration = price.get("duration_curve")
    if duration and duration.get("points"):
        rows = duration["points"]
        figures["price_duration_curve"] = svg.line_chart(
            [(float(row["exceedance_share"]), float(row["price"])) for row in rows if row.get("price") is not None],
            title=FIGURE_TITLES["price_duration_curve"],
            subtitle=_percent_subtitle("高于均值的时段占比", duration.get("share_above_mean")),
            unit=unit,
            caption="横轴为超越概率（0 = 最高价时段，1 = 最低价时段）",
            fill=True,
        )

    seasonality = price.get("seasonality") or {}
    if seasonality.get("hour_of_day"):
        labels, values = _group_rows(seasonality["hour_of_day"])
        figures["seasonal_patterns"] = svg.bar_chart(
            labels,
            values,
            title=FIGURE_TITLES["seasonal_patterns"],
            subtitle=f"时区 {config.study.timezone}",
            unit=unit,
            caption="交割小时",
        )
    if seasonality.get("day_of_week"):
        labels, values = _group_rows(seasonality["day_of_week"], labels=WEEKDAY_LABELS)
        figures["price_weekday_profile"] = svg.bar_chart(
            labels,
            values,
            title=FIGURE_TITLES["price_weekday_profile"],
            unit=unit,
            caption="星期",
        )
    month_rows = seasonality.get("month") or []
    if len(month_rows) > 1:
        labels, values = _group_rows(month_rows)
        figures["price_month_profile"] = svg.bar_chart(
            [f"{label}月" for label in labels],
            values,
            title=FIGURE_TITLES["price_month_profile"],
            unit=unit,
            caption="月份",
        )

    autocorrelation = price.get("autocorrelation") or []
    if autocorrelation:
        figures["price_autocorrelation"] = svg.stem_chart(
            [int(row["lag"]) for row in autocorrelation],
            [row.get("correlation") for row in autocorrelation],
            title=FIGURE_TITLES["price_autocorrelation"],
            subtitle=f"最大滞后 {autocorrelation[-1]['lag']} 个间隔",
        )

    partial = price.get("partial_autocorrelation")
    if partial and partial.get("series"):
        rows = partial["series"]
        figures["price_partial_autocorrelation"] = svg.stem_chart(
            [int(row["lag"]) for row in rows],
            [row.get("partial_correlation") for row in rows],
            title=FIGURE_TITLES["price_partial_autocorrelation"],
            subtitle=f"建议自回归阶数 {partial.get('suggested_autoregressive_order', 0)}",
            band=partial.get("significance_band"),
            caption="阴影区为 95% 显著性带；超出阴影的滞后视为显著",
        )

    decomposition = price.get("decomposition")
    if decomposition:
        shares = decomposition.get("variance_share") or {}
        pretty = {
            "trend": "趋势",
            "seasonal_day": "日内季节",
            "seasonal_week": "周季节",
            "remainder": "残差",
        }
        figures["price_decomposition_variance"] = svg.stacked_share_chart(
            [(pretty.get(key, key), value) for key, value in shares.items() if value],
            title=FIGURE_TITLES["price_decomposition_variance"],
            subtitle=f"{decomposition.get('algorithm')} · 分析粒度 {decomposition.get('analysis_resolution')}",
            caption="残差占比越高，说明可重复的日周规律越难解释全部价格波动",
        )
        shape = (decomposition.get("daily_shape") or {}).get("hourly_profile") or []
        if shape:
            figures["price_daily_shape"] = svg.bar_chart(
                [str(row["hour"]) for row in shape],
                [row.get("value") for row in shape],
                title=FIGURE_TITLES["price_daily_shape"],
                subtitle=(
                    f"峰值 {(decomposition.get('daily_shape') or {}).get('peak_hour')} 时 · "
                    f"谷值 {(decomposition.get('daily_shape') or {}).get('trough_hour')} 时"
                ),
                unit=unit,
                caption="交割小时（已剔除趋势与周季节）",
                diverging=True,
            )

    baselines = price.get("naive_baselines")
    if baselines and baselines.get("baselines"):
        figures["price_naive_baselines"] = svg.ranked_bar_chart(
            [(str(row["label"]), row.get("mae")) for row in baselines["baselines"]],
            title=FIGURE_TITLES["price_naive_baselines"],
            subtitle="平均绝对误差（越低越难被超越）",
            symmetric=False,
            unit=unit,
            caption="后续模型必须在同一时间切分下低于最优基线的误差",
        )

    multicollinearity = exogenous.get("multicollinearity")
    if multicollinearity and multicollinearity.get("variables"):
        figures["exogenous_multicollinearity"] = svg.ranked_bar_chart(
            [
                (str(row["variable"]), row.get("variance_inflation_factor"))
                for row in multicollinearity["variables"]
            ],
            title=FIGURE_TITLES["exogenous_multicollinearity"],
            subtitle=f"严重阈值 {multicollinearity.get('severe_threshold')}",
            symmetric=False,
            caption="VIF 越高说明该变量越能被其余变量共同解释",
        )

    pearson_rows = [
        (name, ((result.get("contemporaneous") or {}).get("pearson") or {}).get("correlation"))
        for name, result in series.items()
    ]
    if any(value is not None for _name, value in pearson_rows):
        figures["correlation_ranking"] = svg.ranked_bar_chart(
            pearson_rows,
            title=FIGURE_TITLES["correlation_ranking"],
            subtitle="同期 Pearson 相关系数",
            caption="相关不等于因果，也不等于样本外预测增益",
        )

    wants_matrix = bool(series) or bool(exogenous.get("correlation_matrix"))
    matrix_names = [config.target.name, *(selected or [spec.name for spec in config.exogenous])]
    matrix_names = [name for name in dict.fromkeys(matrix_names) if name in frame.columns][:12]
    if wants_matrix and len(matrix_names) > 1:
        figures["correlation_matrix"] = svg.heatmap_chart(
            matrix_names,
            _correlation_matrix_values(frame, matrix_names, config.analysis.min_relationship_observations),
            title=FIGURE_TITLES["correlation_matrix"],
            subtitle="成对有效样本",
        )

    lag_series = [
        (
            name,
            [
                (
                    float(row["lag"]) / max(1, result["lag_profile"][-1]["lag"]),
                    float(row["correlation"]),
                )
                for row in result.get("lag_profile", [])
                if row.get("correlation") is not None
            ],
        )
        for name, result in series.items()
        if result.get("lag_profile")
    ]
    lag_series = [item for item in lag_series if item[1]][:5]
    if lag_series:
        longest = max(
            (result["lag_profile"][-1]["lag"] for result in series.values() if result.get("lag_profile")),
            default=0,
        )
        figures["lag_relationships"] = svg.multi_line_chart(
            lag_series,
            title=FIGURE_TITLES["lag_relationships"],
            subtitle=f"扫描范围 0 – {longest} 个间隔",
            unit="相关系数",
            caption="正滞后表示因素领先电价；横轴按最大滞后归一化",
        )

    information = relationships.get("mutual_information")
    if information and information.get("series"):
        rows = [
            (name, ((item.get("best_lag") or {}).get("normalized_mutual_information")))
            for name, item in information["series"].items()
        ]
        if any(value is not None for _name, value in rows):
            figures["mutual_information"] = svg.ranked_bar_chart(
                rows,
                title=FIGURE_TITLES["mutual_information"],
                subtitle="各变量在最佳滞后处的归一化互信息",
                symmetric=False,
                caption="互信息高但线性相关低，提示存在非线性依赖",
            )

    precedence = relationships.get("granger_precedence")
    if precedence and precedence.get("series"):
        rows = [
            (name, ((item.get("strongest") or {}).get("residual_variance_reduction")))
            for name, item in precedence["series"].items()
        ]
        if any(value is not None for _name, value in rows):
            figures["granger_precedence"] = svg.ranked_bar_chart(
                rows,
                title=FIGURE_TITLES["granger_precedence"],
                subtitle="相对纯自回归基线的残差方差下降比例",
                symmetric=False,
                caption="样本内前置性，不是因果关系，也不是样本外增益",
            )

    stability = relationships.get("rolling_stability")
    if stability and stability.get("series"):
        rows = [
            {
                "name": name,
                "mean": item.get("mean_correlation"),
                "min": item.get("min_correlation"),
                "max": item.get("max_correlation"),
            }
            for name, item in stability["series"].items()
            if item.get("mean_correlation") is not None
        ]
        if rows:
            figures["rolling_stability"] = svg.range_chart(
                rows,
                title=FIGURE_TITLES["rolling_stability"],
                subtitle=f"滚动窗口 {stability.get('window_days')} 天",
            )

    for comparison_id, comparison in (comparisons.get("price") or {}).items():
        segments = comparison.get("segments") or {}
        labels = [str(row.get("label") or key) for key, row in segments.items()]
        values = [row.get("mean") for row in segments.values()]
        if any(value is not None for value in values):
            figures["segment_price_comparison"] = svg.bar_chart(
                labels,
                values,
                title=FIGURE_TITLES["segment_price_comparison"],
                subtitle=f"对比集合 {comparison_id}",
                unit=unit,
                caption="同一数据快照内的子样本均值",
                rotate_labels=len(labels) > 6,
            )
        break

    for comparison_id, comparison in (comparisons.get("relationships") or {}).items():
        rows = [
            (f"{variable} · {segment.get('label') or segment_id}", segment.get("correlation"))
            for variable, result in (comparison.get("series") or {}).items()
            for segment_id, segment in (result.get("segments") or {}).items()
        ]
        if any(value is not None for _name, value in rows):
            figures["segment_relationship_comparison"] = svg.ranked_bar_chart(
                rows,
                title=FIGURE_TITLES["segment_relationship_comparison"],
                subtitle=f"对比集合 {comparison_id}",
                caption="每个子样本内的同期 Pearson 相关系数",
            )
        break

    return figures
