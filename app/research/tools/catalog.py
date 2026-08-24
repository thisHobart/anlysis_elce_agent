"""Declarative allow-list for deterministic EDA tools and selectable methods."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EDAMethodSpec:
    key: str
    label: str
    description: str
    implementation_id: str
    version: str = "1.0.0"


@dataclass(frozen=True)
class EDAToolSpec:
    key: str
    title: str
    description: str
    version: str = "1.0.0"
    methods: tuple[EDAMethodSpec, ...] = ()


TOOL_CATALOG: dict[str, EDAToolSpec] = {
    "data_quality": EDAToolSpec(
        key="data_quality",
        title="数据质量与时间对齐",
        description="检查覆盖率、缺失、重复、异常与可获得时间风险",
    ),
    "price_profile": EDAToolSpec(
        key="price_profile",
        title="电价结构分析",
        description="按研究问题选择电价分布、波动、极端值、自相关与季节性",
        methods=(
            EDAMethodSpec(
                "distribution",
                "分布画像",
                "描述位置、离散程度、偏度和分位数",
                "price.descriptive_distribution",
            ),
            EDAMethodSpec(
                "volatility",
                "波动与趋势",
                "检查差分波动、滚动均值和滚动标准差",
                "price.rolling_mean_std",
            ),
            EDAMethodSpec(
                "extremes",
                "尖峰与负价",
                "使用 Tukey outer fence 识别极端高低价、负价和典型时点",
                "price.tukey_outer_fence",
            ),
            EDAMethodSpec(
                "autocorrelation",
                "自相关",
                "使用成对有效样本检查不同滞后的序列自相关",
                "price.lag_autocorrelation",
            ),
            EDAMethodSpec(
                "seasonality",
                "日历分组季节性",
                "比较小时、星期、工作日和月份分组统计",
                "price.calendar_group_profile",
            ),
        ),
    ),
    "exogenous_profile": EDAToolSpec(
        key="exogenous_profile",
        title="外生变量画像",
        description="按研究问题选择变量分布、异常、趋势与共线性",
        methods=(
            EDAMethodSpec(
                "distribution",
                "变量分布",
                "描述覆盖率、位置、离散程度和取值范围",
                "exogenous.descriptive_distribution",
            ),
            EDAMethodSpec(
                "outliers",
                "IQR 异常值",
                "使用稳健 IQR 规则标记异常观测",
                "exogenous.iqr_outliers",
            ),
            EDAMethodSpec(
                "trend",
                "线性索引趋势",
                "检查相对时间位置的线性趋势和一阶差分波动",
                "exogenous.linear_index_trend",
            ),
            EDAMethodSpec(
                "collinearity",
                "Pearson 共线性",
                "检查所选变量之间的强同期 Pearson 相关",
                "exogenous.pearson_collinearity",
            ),
        ),
    ),
    "relationship_analysis": EDAToolSpec(
        key="relationship_analysis",
        title="电价—外生变量关系",
        description="按研究问题选择同期、秩相关、领先滞后、分时段和分位数组关系",
        methods=(
            EDAMethodSpec(
                "pearson",
                "成对 Pearson 同期关系",
                "使用 SciPy Pearson 和成对有效样本衡量线性同期关系",
                "relationship.scipy_pearson_pairwise",
            ),
            EDAMethodSpec(
                "spearman",
                "成对 Spearman 秩关系",
                "使用 SciPy Spearman 和成对有效样本衡量单调同期关系",
                "relationship.scipy_spearman_pairwise",
            ),
            EDAMethodSpec(
                "lag_scan",
                "Pearson 正向领先扫描",
                "比较 feature[t-lag] 与 target[t] 的候选间隔",
                "relationship.pearson_positive_lead_scan",
            ),
            EDAMethodSpec(
                "hour_segments",
                "分小时 Pearson",
                "按小时分组比较 Pearson 关系差异",
                "relationship.pearson_by_hour",
            ),
            EDAMethodSpec(
                "month_segments",
                "分月份 Pearson",
                "按月份分组比较 Pearson 关系差异",
                "relationship.pearson_by_month",
            ),
            EDAMethodSpec(
                "quantile_response",
                "变量四分位数组响应",
                "比较变量四分位数组中的目标均值和中位数",
                "relationship.feature_quartile_response",
            ),
        ),
    ),
}


def tool_metadata() -> dict[str, tuple[str, str]]:
    return {key: (spec.title, spec.description) for key, spec in TOOL_CATALOG.items()}


def method_keys(tool: str) -> tuple[str, ...]:
    spec = TOOL_CATALOG.get(tool)
    return tuple(method.key for method in spec.methods) if spec is not None else ()


def method_labels(tool: str) -> dict[str, str]:
    spec = TOOL_CATALOG.get(tool)
    return {method.key: method.label for method in spec.methods} if spec is not None else {}


def method_versions(tool: str) -> dict[str, str]:
    spec = TOOL_CATALOG.get(tool)
    return {method.key: method.version for method in spec.methods} if spec is not None else {}


def selected_implementation_versions(tool: str, selected_methods: list[str]) -> dict[str, str]:
    spec = TOOL_CATALOG.get(tool)
    if spec is None:
        return {}
    selected = set(selected_methods)
    return {
        method.implementation_id: method.version
        for method in spec.methods
        if method.key in selected
    }


def validate_method_versions(tool: str, selected_methods: list[str], versions: dict[str, str]) -> None:
    """Reject plans whose recorded implementation version differs from the installed catalog."""

    spec = TOOL_CATALOG.get(tool)
    by_key = {method.key: method for method in spec.methods} if spec is not None else {}
    for method in selected_methods:
        definition = by_key.get(method)
        if definition is None:
            raise ValueError(f"unknown method {tool}.{method}")
        recorded = versions.get(definition.implementation_id)
        if recorded != definition.version:
            raise ValueError(
                f"method version mismatch for {definition.implementation_id}: "
                f"plan={recorded or 'missing'}, installed={definition.version}"
            )
