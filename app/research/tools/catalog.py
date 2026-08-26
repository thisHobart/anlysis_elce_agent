"""Single-source catalog for atomic deterministic research functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ResearchFunctionName = Literal[
    "data_quality",
    "price_descriptive_distribution",
    "price_rolling_mean_std",
    "price_tukey_outer_fence",
    "price_lag_autocorrelation",
    "price_calendar_group_profile",
    "price_segment_distribution_comparison",
    "price_stationarity_tests",
    "price_seasonal_decomposition",
    "price_partial_autocorrelation",
    "price_spike_regime_profile",
    "price_duration_curve",
    "price_variance_stabilization_check",
    "price_naive_baseline_benchmark",
    "exogenous_descriptive_distribution",
    "exogenous_iqr_outliers",
    "exogenous_linear_index_trend",
    "exogenous_pearson_collinearity",
    "exogenous_variance_inflation",
    "exogenous_stationarity_tests",
    "relationship_scipy_pearson_pairwise",
    "relationship_scipy_spearman_pairwise",
    "relationship_pearson_positive_lead_scan",
    "relationship_pearson_by_hour",
    "relationship_pearson_by_month",
    "relationship_feature_quartile_response",
    "relationship_pearson_segment_comparison",
    "relationship_mutual_information_scan",
    "relationship_granger_causality_scan",
    "relationship_rolling_correlation_stability",
]

ResearchStage = Literal[
    "data_readiness",
    "target_structure",
    "driver_readiness",
    "relationship_evidence",
    "forecast_readiness",
]

FUNCTION_CATALOG_VERSION = "eda-functions-v5"

STAGE_TITLES: dict[str, str] = {
    "data_readiness": "数据体检",
    "target_structure": "电价自身规律",
    "driver_readiness": "影响因素质量",
    "relationship_evidence": "电价与因素的关系",
    "forecast_readiness": "可预测性判断",
}


@dataclass(frozen=True)
class ResearchFunctionSpec:
    """Metadata shared by model discovery, plan compilation, UI and execution."""

    key: ResearchFunctionName
    title: str
    description: str
    category: Literal["quality", "price", "exogenous", "relationship"]
    result_key: Literal["data_quality", "price", "exogenous", "relationships", "comparisons"]
    stage: ResearchStage
    answers: str
    version: str = "1.0.0"
    evidence_field: str | None = None
    uses_variables: bool = False
    uses_max_lag: bool = False
    uses_segments: bool = False
    uses_spike_multiplier: bool = False
    uses_outlier_multiplier: bool = False
    min_variables: int = 1
    max_calls_per_plan: int = 1
    batch_parameter: str | None = None
    planning_guidance: str = "每个计划最多调用一次。"


FUNCTION_CATALOG: dict[str, ResearchFunctionSpec] = {
    "data_quality": ResearchFunctionSpec(
        key="data_quality",
        title="数据质量与时间对齐",
        description="检查覆盖率、缺失、重复、异常与可获得时间风险。",
        category="quality",
        result_key="data_quality",
        stage="data_readiness",
        answers="这批数据能不能用来做分析，有没有缺口和时间错位。",
    ),
    "price_descriptive_distribution": ResearchFunctionSpec(
        key="price_descriptive_distribution",
        title="电价水平与分布",
        description="计算位置、离散程度、偏度、峰度和分位数。",
        category="price",
        result_key="price",
        stage="target_structure",
        evidence_field="distribution",
        answers="电价整体处在什么水平，波动范围有多大。",
    ),
    "price_rolling_mean_std": ResearchFunctionSpec(
        key="price_rolling_mean_std",
        title="电价滚动均值与波动",
        description="计算电价差分波动以及日、周滚动均值和滚动标准差。",
        category="price",
        result_key="price",
        stage="target_structure",
        evidence_field="volatility",
        version="1.1.0",
        answers="电价的波动是稳定的，还是分阶段忽大忽小。",
        batch_parameter="内置一天、七天窗口",
        planning_guidance="一次调用已返回一天和七天滚动窗口；不要按窗口重复调用。",
    ),
    "price_tukey_outer_fence": ResearchFunctionSpec(
        key="price_tukey_outer_fence",
        title="电价极端值识别",
        description="使用 Tukey outer fence 识别极端高低价、负价及典型时点。",
        category="price",
        result_key="price",
        stage="target_structure",
        evidence_field="extremes",
        answers="有多少极端高价和负价，分别出现在什么时候。",
        uses_spike_multiplier=True,
    ),
    "price_lag_autocorrelation": ResearchFunctionSpec(
        key="price_lag_autocorrelation",
        title="电价滞后自相关",
        description="使用成对有效样本计算指定最大滞后内的电价自相关。",
        category="price",
        result_key="price",
        stage="target_structure",
        evidence_field="autocorrelation",
        answers="过去的价格对当前价格还有多大参考价值。",
        uses_max_lag=True,
        batch_parameter="max_lag",
        planning_guidance="一次调用扫描 1 到 max_lag 的完整范围；不要按单个滞后重复调用。",
    ),
    "price_calendar_group_profile": ResearchFunctionSpec(
        key="price_calendar_group_profile",
        title="电价日历规律",
        description="按小时、星期、工作日或周末及月份统计电价分组画像。",
        category="price",
        result_key="price",
        stage="target_structure",
        evidence_field="seasonality",
        answers="一天之内、一周之内、一年之内的价格规律是什么。",
        batch_parameter="内置小时、星期、工作日/周末和月份全分组",
        planning_guidance="一次调用已返回全部内置日历分组；不要按小时或月份重复调用。",
    ),
    "price_stationarity_tests": ResearchFunctionSpec(
        key="price_stationarity_tests",
        title="电价平稳性检验",
        description="对原序列与一阶差分同时执行 ADF 与 KPSS 单位根检验并给出一致性结论。",
        category="price",
        result_key="price",
        stage="target_structure",
        evidence_field="stationarity",
        answers="电价是围绕一个稳定水平波动，还是持续漂移、需要差分处理。",
    ),
    "price_seasonal_decomposition": ResearchFunctionSpec(
        key="price_seasonal_decomposition",
        title="趋势与季节成分分解",
        description="使用 MSTL/STL 把电价拆成趋势、日内季节、周季节和残差，并量化各成分强度。",
        category="price",
        result_key="price",
        stage="target_structure",
        evidence_field="decomposition",
        answers="价格里有多少是可重复的日周规律，多少是无法解释的随机部分。",
    ),
    "price_partial_autocorrelation": ResearchFunctionSpec(
        key="price_partial_autocorrelation",
        title="电价偏自相关与白噪声检验",
        description="计算 PACF 直接记忆结构，并用 Ljung-Box 检验残余自相关。",
        category="price",
        result_key="price",
        stage="target_structure",
        evidence_field="partial_autocorrelation",
        answers="自回归基线大概需要几阶，价格是否已经接近随机游走。",
        uses_max_lag=True,
        batch_parameter="max_lag",
        planning_guidance="一次调用扫描 1 到 max_lag 的完整范围；不要按单个滞后重复调用。",
    ),
    "price_spike_regime_profile": ResearchFunctionSpec(
        key="price_spike_regime_profile",
        title="尖峰与负价状态画像",
        description="统计负价、零价、高低尖峰占比、持续时长、聚集性和出现时段。",
        category="price",
        result_key="price",
        stage="target_structure",
        evidence_field="spike_regime",
        answers="极端价格是零散出现还是成片爆发，集中在哪些小时和月份。",
        uses_spike_multiplier=True,
    ),
    "price_duration_curve": ResearchFunctionSpec(
        key="price_duration_curve",
        title="电价持续曲线",
        description="按超越概率排序生成价格持续曲线，并统计极端区间对价值的集中度。",
        category="price",
        result_key="price",
        stage="target_structure",
        evidence_field="duration_curve",
        answers="多少时间处于高价区间，价格价值集中在多小比例的时段上。",
    ),
    "price_variance_stabilization_check": ResearchFunctionSpec(
        key="price_variance_stabilization_check",
        title="方差稳定变换检查",
        description="比较 median/MAD 稳健标准化前后与 asinh 变换后的偏度、峰度和尾部占比。",
        category="price",
        result_key="price",
        stage="forecast_readiness",
        evidence_field="variance_stabilization",
        answers="建模前是否需要做方差稳定变换来压制厚尾。",
    ),
    "price_naive_baseline_benchmark": ResearchFunctionSpec(
        key="price_naive_baseline_benchmark",
        title="朴素基线误差底线",
        description="计算持续法、日naive 与周naive 基线的 MAE/RMSE，给出后续模型必须超越的误差底线。",
        category="price",
        result_key="price",
        stage="forecast_readiness",
        evidence_field="naive_baselines",
        answers="不建模、只照抄历史能做到多准，后续模型至少要好过多少。",
    ),
    "price_segment_distribution_comparison": ResearchFunctionSpec(
        key="price_segment_distribution_comparison",
        title="电价分段对比",
        description=(
            "在同一数据快照中按 hours、months 或 time_range 定义的多个子样本，"
            "比较电价样本量、覆盖率、均值、中位数、波动和分位数。"
        ),
        category="price",
        result_key="comparisons",
        evidence_field="price",
        stage="target_structure",
        answers="峰段谷段、不同季节或事件前后的价格差异有多大。",
        uses_segments=True,
        batch_parameter="segments",
        planning_guidance="把峰谷、季节或事件前后所有子样本放入一次 segments 调用；不要重复调用本函数。",
    ),
    "exogenous_descriptive_distribution": ResearchFunctionSpec(
        key="exogenous_descriptive_distribution",
        title="影响因素分布画像",
        description="计算所选外生变量的覆盖率、位置、离散程度和取值范围。",
        category="exogenous",
        result_key="exogenous",
        stage="driver_readiness",
        evidence_field="series",
        answers="每个候选影响因素的数据质量和取值范围是否正常。",
        uses_variables=True,
        batch_parameter="variables",
        planning_guidance="一次调用用 variables 覆盖全部候选变量。",
    ),
    "exogenous_iqr_outliers": ResearchFunctionSpec(
        key="exogenous_iqr_outliers",
        title="影响因素异常值",
        description="使用稳健 IQR 规则标记所选外生变量的异常观测。",
        category="exogenous",
        result_key="exogenous",
        stage="driver_readiness",
        evidence_field="series",
        answers="哪些因素存在采集异常或极端读数。",
        uses_outlier_multiplier=True,
        uses_variables=True,
        batch_parameter="variables",
        planning_guidance="一次调用用 variables 覆盖全部候选变量。",
    ),
    "exogenous_linear_index_trend": ResearchFunctionSpec(
        key="exogenous_linear_index_trend",
        title="影响因素长期漂移",
        description="检查所选外生变量相对时间位置的线性趋势和一阶差分波动。",
        category="exogenous",
        result_key="exogenous",
        stage="driver_readiness",
        evidence_field="series",
        answers="哪些因素在研究窗口内出现了长期漂移或口径变化。",
        uses_variables=True,
        batch_parameter="variables",
        planning_guidance="一次调用用 variables 覆盖全部候选变量。",
    ),
    "exogenous_pearson_collinearity": ResearchFunctionSpec(
        key="exogenous_pearson_collinearity",
        title="影响因素两两相关",
        description="检查所选外生变量之间的强同期 Pearson 相关。",
        category="exogenous",
        result_key="exogenous",
        stage="driver_readiness",
        evidence_field="strong_collinearity_pairs",
        answers="哪两个因素之间高度重复，可能只需要保留一个。",
        uses_variables=True,
        batch_parameter="variables",
        planning_guidance="一次调用用 variables 覆盖全部候选变量。",
    ),
    "exogenous_variance_inflation": ResearchFunctionSpec(
        key="exogenous_variance_inflation",
        title="影响因素多重共线性 (VIF)",
        description="用方差膨胀因子衡量每个变量被其余变量共同解释的程度，识别多变量冗余。",
        category="exogenous",
        result_key="exogenous",
        stage="driver_readiness",
        evidence_field="multicollinearity",
        answers="除了两两相关，是否存在多个变量合起来互相解释的冗余结构。",
        min_variables=2,
        uses_variables=True,
        batch_parameter="variables",
        planning_guidance="一次调用用 variables 覆盖全部候选变量；至少需要两个变量。",
    ),
    "exogenous_stationarity_tests": ResearchFunctionSpec(
        key="exogenous_stationarity_tests",
        title="影响因素平稳性检验",
        description="对每个所选外生变量执行 ADF 与 KPSS 检验，标记与电价共同非平稳的伪回归风险。",
        category="exogenous",
        result_key="exogenous",
        stage="driver_readiness",
        evidence_field="driver_stationarity",
        answers="哪些因素本身在漂移，可能与电价形成虚假的共同趋势相关。",
        uses_variables=True,
        batch_parameter="variables",
        planning_guidance="一次调用用 variables 覆盖全部候选变量。",
    ),
    "relationship_scipy_pearson_pairwise": ResearchFunctionSpec(
        key="relationship_scipy_pearson_pairwise",
        title="电价—因素同期线性相关",
        description="使用 SciPy Pearson 和成对有效样本衡量线性同期关系。",
        category="relationship",
        result_key="relationships",
        stage="relationship_evidence",
        evidence_field="series",
        answers="同一时刻，因素涨落与电价涨落的线性同步程度。",
        uses_variables=True,
        batch_parameter="variables",
        planning_guidance="一次调用用 variables 覆盖全部候选变量。",
    ),
    "relationship_scipy_spearman_pairwise": ResearchFunctionSpec(
        key="relationship_scipy_spearman_pairwise",
        title="电价—因素同期秩相关",
        description="使用 SciPy Spearman 和成对有效样本衡量单调同期关系。",
        category="relationship",
        result_key="relationships",
        stage="relationship_evidence",
        evidence_field="series",
        answers="不假设线性时，因素与电价的排序关系是否一致。",
        uses_variables=True,
        batch_parameter="variables",
        planning_guidance="一次调用用 variables 覆盖全部候选变量。",
    ),
    "relationship_pearson_positive_lead_scan": ResearchFunctionSpec(
        key="relationship_pearson_positive_lead_scan",
        title="因素领先电价扫描",
        description="比较 feature[t-lag] 与 target[t] 的 Pearson 候选领先间隔。",
        category="relationship",
        result_key="relationships",
        stage="relationship_evidence",
        evidence_field="series",
        answers="因素的变化会提前多久反映到电价上。",
        uses_variables=True,
        uses_max_lag=True,
        batch_parameter="max_lag, variables",
        planning_guidance="一次调用扫描 0 到 max_lag，并用 variables 覆盖全部变量；不要按滞后重复调用。",
    ),
    "relationship_pearson_by_hour": ResearchFunctionSpec(
        key="relationship_pearson_by_hour",
        title="分小时关系差异",
        description="按小时分组比较所选变量与电价的 Pearson 关系差异。",
        category="relationship",
        result_key="relationships",
        stage="relationship_evidence",
        evidence_field="series",
        answers="同一个因素在不同交割小时的影响是否一致。",
        uses_variables=True,
        batch_parameter="variables, 内置全部小时",
        planning_guidance="一次调用返回全部小时分组，并用 variables 覆盖全部变量。",
    ),
    "relationship_pearson_by_month": ResearchFunctionSpec(
        key="relationship_pearson_by_month",
        title="分月份关系差异",
        description="按月份分组比较所选变量与电价的 Pearson 关系差异。",
        category="relationship",
        result_key="relationships",
        stage="relationship_evidence",
        evidence_field="series",
        answers="关系是否随季节改变，甚至反号。",
        uses_variables=True,
        batch_parameter="variables, 内置全部月份",
        planning_guidance="一次调用返回全部月份分组，并用 variables 覆盖全部变量。",
    ),
    "relationship_feature_quartile_response": ResearchFunctionSpec(
        key="relationship_feature_quartile_response",
        title="因素分位响应曲线",
        description="比较变量四分位数组中的目标电价均值和中位数。",
        category="relationship",
        result_key="relationships",
        stage="relationship_evidence",
        evidence_field="series",
        answers="因素处于高位和低位时，电价水平分别是多少。",
        uses_variables=True,
        batch_parameter="variables",
        planning_guidance="一次调用用 variables 覆盖全部候选变量。",
    ),
    "relationship_mutual_information_scan": ResearchFunctionSpec(
        key="relationship_mutual_information_scan",
        title="非线性依赖扫描 (互信息)",
        description="按等频秩分箱计算各滞后的互信息，识别相关系数接近零但仍存在依赖的变量。",
        category="relationship",
        result_key="relationships",
        stage="relationship_evidence",
        evidence_field="mutual_information",
        answers="有没有相关系数很低、但其实存在非线性依赖的因素。",
        uses_variables=True,
        uses_max_lag=True,
        batch_parameter="max_lag, variables",
        planning_guidance="一次调用扫描 0 到 max_lag，并用 variables 覆盖全部变量。",
    ),
    "relationship_granger_causality_scan": ResearchFunctionSpec(
        key="relationship_granger_causality_scan",
        title="预测前置性检验 (Granger)",
        description="比较自回归基线与加入变量历史后的回归残差，用 F 检验判断样本内前置性。",
        category="relationship",
        result_key="relationships",
        stage="relationship_evidence",
        evidence_field="granger_precedence",
        answers="加入这个因素的历史，是否真的比只看电价自身更能解释电价。",
        uses_variables=True,
        uses_max_lag=True,
        batch_parameter="max_lag, variables",
        planning_guidance="一次调用用 variables 覆盖全部变量；函数内部自动选择受检滞后集合。",
    ),
    "relationship_rolling_correlation_stability": ResearchFunctionSpec(
        key="relationship_rolling_correlation_stability",
        title="关系随时间稳定性",
        description="用滚动窗口 Pearson 评估关系的均值、波动、极值和反号频率。",
        category="relationship",
        result_key="relationships",
        stage="relationship_evidence",
        evidence_field="rolling_stability",
        answers="这个关系是长期成立，还是只在某段时间成立。",
        uses_variables=True,
        batch_parameter="variables",
        planning_guidance="一次调用用 variables 覆盖全部候选变量。",
    ),
    "relationship_pearson_segment_comparison": ResearchFunctionSpec(
        key="relationship_pearson_segment_comparison",
        title="分段关系对比",
        description=(
            "在同一数据快照中按 hours、months 或 time_range 定义的多个子样本，"
            "比较所选变量与电价的 Pearson 关系。"
        ),
        category="relationship",
        result_key="comparisons",
        evidence_field="relationships",
        stage="relationship_evidence",
        answers="在你指定的几个时段里，同一个因素的影响差多少。",
        uses_variables=True,
        uses_segments=True,
        batch_parameter="segments, variables",
        planning_guidance="把全部子样本放入一次 segments，并用 variables 覆盖全部变量；不要重复调用本函数。",
    ),
}


# Read-only migration aliases for plans/checkpoints created before atomic functions.
LEGACY_METHOD_TO_FUNCTION: dict[tuple[str, str], ResearchFunctionName] = {
    ("price_profile", "distribution"): "price_descriptive_distribution",
    ("price_profile", "price.descriptive_distribution"): "price_descriptive_distribution",
    ("price_profile", "volatility"): "price_rolling_mean_std",
    ("price_profile", "price.rolling_mean_std"): "price_rolling_mean_std",
    ("price_profile", "extremes"): "price_tukey_outer_fence",
    ("price_profile", "price.tukey_outer_fence"): "price_tukey_outer_fence",
    ("price_profile", "autocorrelation"): "price_lag_autocorrelation",
    ("price_profile", "price.lag_autocorrelation"): "price_lag_autocorrelation",
    ("price_profile", "seasonality"): "price_calendar_group_profile",
    ("price_profile", "price.calendar_group_profile"): "price_calendar_group_profile",
    ("exogenous_profile", "distribution"): "exogenous_descriptive_distribution",
    ("exogenous_profile", "exogenous.descriptive_distribution"): "exogenous_descriptive_distribution",
    ("exogenous_profile", "outliers"): "exogenous_iqr_outliers",
    ("exogenous_profile", "exogenous.iqr_outliers"): "exogenous_iqr_outliers",
    ("exogenous_profile", "trend"): "exogenous_linear_index_trend",
    ("exogenous_profile", "exogenous.linear_index_trend"): "exogenous_linear_index_trend",
    ("exogenous_profile", "collinearity"): "exogenous_pearson_collinearity",
    ("exogenous_profile", "exogenous.pearson_collinearity"): "exogenous_pearson_collinearity",
    ("relationship_analysis", "pearson"): "relationship_scipy_pearson_pairwise",
    ("relationship_analysis", "relationship.scipy_pearson_pairwise"): "relationship_scipy_pearson_pairwise",
    ("relationship_analysis", "spearman"): "relationship_scipy_spearman_pairwise",
    ("relationship_analysis", "relationship.scipy_spearman_pairwise"): "relationship_scipy_spearman_pairwise",
    ("relationship_analysis", "lag_scan"): "relationship_pearson_positive_lead_scan",
    ("relationship_analysis", "relationship.pearson_positive_lead_scan"): "relationship_pearson_positive_lead_scan",
    ("relationship_analysis", "hour_segments"): "relationship_pearson_by_hour",
    ("relationship_analysis", "relationship.pearson_by_hour"): "relationship_pearson_by_hour",
    ("relationship_analysis", "month_segments"): "relationship_pearson_by_month",
    ("relationship_analysis", "relationship.pearson_by_month"): "relationship_pearson_by_month",
    ("relationship_analysis", "quantile_response"): "relationship_feature_quartile_response",
    ("relationship_analysis", "relationship.feature_quartile_response"): "relationship_feature_quartile_response",
}

FUNCTION_TO_ANALYSIS_METHOD: dict[str, str] = {
    function_name: legacy_method
    for (_legacy_tool, legacy_method), function_name in LEGACY_METHOD_TO_FUNCTION.items()
    if "." not in legacy_method
}


def function_metadata() -> dict[str, tuple[str, str]]:
    """Compatibility name for callers that consume function title/description metadata."""

    return {key: (spec.title, spec.description) for key, spec in FUNCTION_CATALOG.items()}


def optional_function_names() -> tuple[str, ...]:
    return tuple(name for name in FUNCTION_CATALOG if name != "data_quality")


def functions_using_variables() -> frozenset[str]:
    return frozenset(name for name, spec in FUNCTION_CATALOG.items() if spec.uses_variables)


def functions_using_max_lag() -> frozenset[str]:
    return frozenset(name for name, spec in FUNCTION_CATALOG.items() if spec.uses_max_lag)


def stage_of(function_name: str) -> ResearchStage:
    """Return the research stage a function belongs to, for grouping in UI and reports."""

    spec = FUNCTION_CATALOG.get(function_name)
    return spec.stage if spec is not None else "target_structure"


def expand_legacy_function_permissions(names: list[str]) -> list[str]:
    """Expand v1 aggregate Skill permissions into their atomic v2 functions."""

    legacy_categories = {
        "price_profile": "price",
        "exogenous_profile": "exogenous",
        "relationship_analysis": "relationship",
    }
    expanded: list[str] = []
    for name in names:
        category = legacy_categories.get(name)
        if category is None:
            expanded.append(name)
            continue
        expanded.extend(function_name for function_name, spec in FUNCTION_CATALOG.items() if spec.category == category)
    return list(dict.fromkeys(expanded))
