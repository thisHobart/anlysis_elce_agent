"""Single-source reader-facing wording for data-quality and availability issues."""

from __future__ import annotations

from app.research.schemas.results import QualityIssue

ISSUE_MESSAGES: dict[str, str] = {
    "series_has_no_values": "在统一后的时间轴上没有任何可用数值。",
    "very_low_coverage": "可用数据不到统一时间轴的一半。",
    "low_coverage": "缺失的时间点较多，可能影响时序分析和关系分析的结论。",
    "incomplete_coverage": "在统一后的时间轴上存在数据缺口。",
    "invalid_timestamps": "有一些时间戳无法解析，已在对齐之前排除。",
    "non_numeric_values": "有一些取值不是数字，无法参与计算。",
    "duplicate_timestamps": "同一时刻出现了多条记录，已按配置的规则合并。",
    "availability_unknown": "不确定预测的时候能不能拿到这个变量，相关结论只能用于描述。",
    "forecast_vintage_unknown": "没有记录这个预测值的发布时间，无法确认预测的时候它已经拿得到。",
    "lookahead_risk": "这是事后才有的实测值，做预测的那一刻可能还拿不到。",
}


def issue_message(code: str, fallback: str) -> str:
    """Translate one issue code; fall back to the raw message for unmapped codes."""

    return ISSUE_MESSAGES.get(code, fallback)


def issue_line(issue: QualityIssue) -> str:
    """Render one issue as the single sentence both the evaluator and the report use."""

    return f"{issue.series or '研究'}：{issue_message(issue.code, issue.message)}"
