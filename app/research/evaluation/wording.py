"""Say one verdict the same way everywhere it is shown.

Verdict codes such as ``trend_or_break_suspected`` are contracts between
functions, not sentences for a reader. Both the evaluator and the report
translate them here, so a conclusion never reaches the user as a raw enum.
"""

from __future__ import annotations

STATIONARITY_VERDICT_TEXT = {
    "stationary": "围绕一个稳定的水平上下波动",
    "unit_root": "会持续漂移，不围绕固定水平",
    "trend_or_break_suspected": "存在趋势，或某个时点发生过结构变化",
    "inconclusive": "两项检验都没有给出明确结论",
}

TRANSFORM_TEXT = {
    "none": "不需要额外处理",
    "first_difference": "建模前先取相邻时段的差值（一阶差分）",
    "seasonal_or_structural_review": "需要先处理季节成分或结构变化，只做差分还不够",
    "asinh_median_mad": "建模前先做稳健标准化和 asinh 变换来压缩极端值",
}

SEASONALITY_TEXT = {"day": "日周期", "week": "周周期"}

GROUP_LABELS = {
    "hour_of_day": "小时",
    "day_of_week": "星期",
    "day_type": "工作日与周末",
    "month": "月份",
}


def stationarity_text(verdict: str | None) -> str:
    return STATIONARITY_VERDICT_TEXT.get(str(verdict), "结论不明确")


def transform_text(transform: str | None) -> str:
    return TRANSFORM_TEXT.get(str(transform), str(transform or "未给出建议"))
