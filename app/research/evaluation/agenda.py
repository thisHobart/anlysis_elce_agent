"""Map one hypothesis wording to the agenda item that can decide it.

The planner writes hypotheses in its own words while the compiler registers a
canonical wording for every selected function. Both must resolve to the same
agenda item, or the same question ends up on the report twice with opposite
verdicts. This table is the single place that mapping lives: the compiler uses
it to keep one hypothesis per item, and the evaluator uses it to pick which
evidence rule decides that item.
"""

from __future__ import annotations

AgendaKeywords = tuple[str, tuple[str, ...], tuple[str, ...]]

# Ordered: the first matching row wins, so a more specific reading of a shared
# word ("自身非平稳" for a driver) must come before the general one ("平稳").
AGENDA_KEYWORDS: tuple[AgendaKeywords, ...] = (
    ("relationships.by_hour", ("关系可能存在时段差异", "分小时关系"), ()),
    ("relationships.by_month", ("关系可能存在月份结构", "分月份关系"), ()),
    ("price.distribution", ("偏斜", "分布形态", "偏度", "峰度"), ()),
    (
        "price.volatility",
        (
            "波动可能分阶段",
            "波动可能存在",
            "波动聚集",
            "方差随时间",
            "波动状态",
            "分阶段波动",
            "波动变化",
            "分阶段变化",
            "滚动均值",
            "滚动标准差",
            "波动是否",
        ),
        ("峰段", "谷段", "夏季", "冬季", "事件前", "事件后"),
    ),
    ("exogenous.coverage", ("覆盖率", "变量覆盖", "样本覆盖"), ()),
    ("exogenous.trend", ("长期漂移", "变量趋势", "驱动漂移"), ()),
    ("relationships.quantile_response", ("响应可能不是单调", "分位响应", "响应形状"), ()),
    ("exogenous.stationarity", ("自身非平稳", "驱动非平稳", "共同趋势"), ()),
    ("price.stationarity", ("平稳", "差分", "去趋势", "单位根", "稳定水平"), ()),
    ("price.decomposition", ("季节成分", "趋势成分", "成分分解", "残差占比"), ()),
    ("price.partial_autocorrelation", ("直接记忆", "偏自相关", "自回归阶", "随机游走"), ()),
    ("price.spike_regime", ("聚集", "成片", "状态画像", "负价频率"), ()),
    ("price.duration_curve", ("持续曲线", "高价时段", "价格集中度", "贡献主要价值"), ()),
    ("price.variance_stabilization", ("厚尾", "方差稳定", "变换"), ()),
    ("price.naive_baselines", ("朴素基线", "误差底线", "达标线"), ()),
    ("exogenous.multicollinearity", ("多重共线", "共线性冗余", "变量冗余结构"), ()),
    ("relationships.mutual_information", ("非线性依赖", "互信息"), ()),
    ("relationships.granger", ("前置性", "granger", "自回归拟合"), ()),
    ("relationships.rolling_stability", ("随时间漂移", "反号", "关系稳定性"), ()),
    ("price.extremes", ("尖峰", "极端", "负价", "峰值", "高价尖峰", "异常价格", "尾部"), ()),
    (
        "price.autocorrelation",
        ("自相关", "序列相关", "序列持续性", "价格持续性", "价格滞后结构"),
        (),
    ),
    (
        "price.seasonality",
        (
            "日内",
            "周内",
            "月份结构",
            "月度结构",
            "季节性",
            "季节结构",
            "小时结构",
            "日历结构",
            "日历分组",
            "日历画像",
            "分组画像",
            "工作日/周末",
            "小时层面",
            "月份层面",
        ),
        (),
    ),
    ("exogenous.collinearity", ("共线性", "变量冗余", "驱动冗余", "高度相关变量"), ()),
    ("exogenous.outliers", ("异常观测", "异常值", "离群", "变量异常"), ()),
    (
        "relationships.contemporaneous",
        ("同期线性", "同期关系", "同期相关", "线性关系", "单调关系"),
        (),
    ),
    (
        "relationships.lead_lag",
        ("领先间隔", "领先滞后", "领先关系", "滞后关系", "领先变化", "滞后变化"),
        (),
    ),
    (
        "comparisons.price_segments",
        ("峰段", "谷段", "夏季", "冬季", "事件前", "事件后", "分段", "分段差异", "时段差异"),
        (),
    ),
)


def contains_any(value: str, terms: tuple[str, ...]) -> bool:
    normalized = value.casefold()
    return any(term.casefold() in normalized for term in terms)


def _refine(item_id: str, hypothesis: str) -> str:
    """Split the two items that share a keyword set with their neighbour."""

    if item_id == "relationships.contemporaneous" and contains_any(hypothesis, ("单调关系",)):
        return "relationships.monotonic"
    if item_id == "comparisons.price_segments" and contains_any(hypothesis, ("关系差异",)):
        return "comparisons.relationship_segments"
    return item_id


def resolve_agenda_item(hypothesis: str) -> str | None:
    """Return the agenda item this wording belongs to, or None when nothing can decide it."""

    for item_id, keywords, excluded in AGENDA_KEYWORDS:
        if contains_any(hypothesis, keywords) and not (excluded and contains_any(hypothesis, excluded)):
            return _refine(item_id, hypothesis)
    return None
