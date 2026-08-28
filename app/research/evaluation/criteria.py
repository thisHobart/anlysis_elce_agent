"""Single-source deterministic decision criteria shared by evaluation and reporting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SEASONAL_STRENGTH_THRESHOLD = 0.3
SEASONAL_STRONG_THRESHOLD = 0.6
CALENDAR_EFFECT_THRESHOLD = 0.06
DISTRIBUTION_SKEW_THRESHOLD = 0.5
DISTRIBUTION_KURTOSIS_THRESHOLD = 1.0
VOLATILITY_REGIME_RATIO = 2.0
MINIMUM_DRIVER_COVERAGE = 0.9
DRIVER_DRIFT_STANDARD_DEVIATIONS = 1.0


def group_variance_share(rows: Any) -> float | None:
    """Return eta squared for one calendar grouping, from the group summary rows themselves."""

    usable = [
        row
        for row in (rows or [])
        if isinstance(row, dict) and row.get("mean") is not None and row.get("observations")
    ]
    if len(usable) < 2:
        return None
    total = sum(int(row["observations"]) for row in usable)
    grand_mean = sum(float(row["mean"]) * int(row["observations"]) for row in usable) / total
    between = sum(int(row["observations"]) * (float(row["mean"]) - grand_mean) ** 2 for row in usable)
    within = sum(
        max(int(row["observations"]) - 1, 0) * float(row["std"]) ** 2
        for row in usable
        if row.get("std") is not None
    )
    spread = between + within
    return between / spread if spread > 0 else None


@dataclass(frozen=True)
class DecisionCriterion:
    """One auditable threshold used by the deterministic evaluator."""

    criterion_id: str
    label: str
    rule: str
    rationale: str


DECISION_CRITERIA: tuple[DecisionCriterion, ...] = (
    DecisionCriterion(
        criterion_id="seasonal_strength",
        label="季节成分强度",
        rule=(
            f"≥ {SEASONAL_STRONG_THRESHOLD:.1f} 为强周期；"
            f"{SEASONAL_STRENGTH_THRESHOLD:.1f}–{SEASONAL_STRONG_THRESHOLD:.1f} 为中等；"
            f"< {SEASONAL_STRENGTH_THRESHOLD:.1f} 不支持稳定周期结构"
        ),
        rationale="使用趋势季节分解返回的 seasonal_strength 判读周期证据。",
    ),
    DecisionCriterion(
        criterion_id="calendar_effect",
        label="日历分组结构",
        rule=f"组间方差占比 ≥ {CALENDAR_EFFECT_THRESHOLD:.0%} 支持该日历层结构",
        rationale="在没有分解强度时，以分组均值的 eta-squared 衡量结构效应。",
    ),
    DecisionCriterion(
        criterion_id="distribution_shape",
        label="分布偏斜与厚尾",
        rule=(
            f"|偏度| ≥ {DISTRIBUTION_SKEW_THRESHOLD:.1f} 或超额峰度 ≥ "
            f"{DISTRIBUTION_KURTOSIS_THRESHOLD:.1f}"
        ),
        rationale="标记可能影响均值、方差和误差指标解释的非对称与厚尾结构。",
    ),
    DecisionCriterion(
        criterion_id="volatility_regime",
        label="波动分阶段",
        rule=f"一天窗口最大/最小滚动标准差比值 ≥ {VOLATILITY_REGIME_RATIO:.1f}",
        rationale="识别全样本波动统计量可能无法代表局部状态的情况。",
    ),
    DecisionCriterion(
        criterion_id="driver_coverage",
        label="驱动变量覆盖率",
        rule=f"覆盖率 < {MINIMUM_DRIVER_COVERAGE:.0%} 时不足以独立支撑关系结论",
        rationale="避免用稀疏变量形成不稳定的关系证据。",
    ),
    DecisionCriterion(
        criterion_id="driver_drift",
        label="驱动变量长期漂移",
        rule=(
            "|趋势斜率| × 观测数 ≥ "
            f"{DRIVER_DRIFT_STANDARD_DEVIATIONS:g} 个变量自身标准差"
        ),
        rationale="把趋势量级与变量自身波动尺度比较，识别共同趋势混杂。",
    ),
)
