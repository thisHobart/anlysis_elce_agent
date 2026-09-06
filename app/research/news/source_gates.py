"""Deterministic checks read straight from the source text, independent of the model.

The consistency gate can only catch an *unstable* mistake: three passes that disagree get
sent to review. A model that is confidently and repeatably wrong sails through it, because
what the gate sees is agreement. These checks are the answer to that — they read the article
itself, so they hold whether the model wavered or not.

Every function here is pure and regex-driven. None of them decides what an event *is*; they
only say what the source demonstrably mentions, so a downstream gate can notice that an
extraction failed to account for it.
"""

from __future__ import annotations

import re

SOURCE_GATE_VERSION = "1.0.0"

# A number carrying a power (or energy) unit. Bare "W" is deliberately excluded: on its own
# it collides with ordinary text far more often than it names a quantity.
_POWER_QUANTITY = re.compile(
    r"\d[\d,]*(?:\.\d+)?\s*(?:万|亿)?\s*(?:[千干]?瓦(?:时)?|[kKmMgG][wW][hH]?)"
)

# Words that only appear when an article is discussing power-system operation. Kept narrow
# on purpose: 电力 and 发电 alone show up in company names and licence notices, so they are
# not here. Anything added must be checked against the irrelevant cases in both corpora.
_OPERATIONAL_TERMS: tuple[str, ...] = (
    "停运",
    "跳闸",
    "检修",
    "故障",
    "事故",
    "停电",
    "限电",
    "负荷",
    "出力",
    "发电量",
    "装机",
    "并网",
    "解列",
    "非计划",
    "电量",
    "outage",
    "tripped",
    "curtailment",
    "blackout",
    "derate",
)

# One whole date/time expression per match. The alternatives are ordered longest-first so
# that "2026年8月3日11时06分" counts once rather than as a date plus a clock time — the
# difference between one anchor and two is exactly what the multi-event gate turns on.
_TIME_ANCHOR = re.compile(
    r"(?:\d{4}\s*年\s*)?\d{1,2}\s*月\s*\d{1,2}\s*日"
    r"(?:\s*(?:凌晨|上午|中午|午间|下午|傍晚|晚间|夜间)?\s*\d{1,2}\s*[:：时点]\s*\d{1,2}\s*分?)?"
    r"|\d{1,2}\s*日\s*(?:凌晨|上午|中午|午间|下午|傍晚|晚间|夜间)?\s*\d{1,2}\s*[:：时点]\s*\d{1,2}\s*分?"
    r"|\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?Z?"
    # 份 is required: it separates "2026年7月份" from the "7月" inside "7月初".
    r"|(?:\d{4}\s*年\s*)?\d{1,2}\s*月份"
)


def power_quantities(text: str) -> tuple[str, ...]:
    """Every number-with-a-power-unit the source states, in order of appearance."""

    return tuple(match.group(0).strip() for match in _POWER_QUANTITY.finditer(text))


def operational_terms(text: str) -> tuple[str, ...]:
    """Operational vocabulary present in the source, deduplicated and ordered."""

    lowered = text.casefold()
    return tuple(term for term in _OPERATIONAL_TERMS if term.casefold() in lowered)


def operational_signals(text: str) -> tuple[str, ...]:
    """Anything that makes 'this article is not about power system operation' hard to believe."""

    return power_quantities(text) + operational_terms(text)


def time_anchors(text: str) -> tuple[str, ...]:
    """Distinct date/time expressions stated in the source, in order of appearance."""

    return tuple(match.group(0).strip() for match in _TIME_ANCHOR.finditer(text))
