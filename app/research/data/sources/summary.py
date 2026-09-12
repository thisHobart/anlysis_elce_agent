"""Plain-language description of one research dataset, rendered verbatim by the UI.

The desktop panel and the confirmation card both read this structure and nothing
else. Every string is already formatted for a power-market analyst, so the two
surfaces cannot drift apart and the same dataset always reads the same way.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime
from typing import Any, Literal

from app.research.data.sources.naming import label_variable

VariableKind = Literal["actual", "forecast", "unknown"]


@dataclass(frozen=True)
class VariableLabel:
    """One exogenous variable as the analyst should read it."""

    display: str
    resolved: bool = True
    kind: VariableKind = "unknown"


@dataclass(frozen=True)
class DataSummary:
    """Everything the UI is allowed to say about a dataset, already in Chinese.

    Fields are left empty rather than guessed while exploration is still running,
    which is how the panel knows an item is not settled yet.
    """

    price_label: str = ""
    variables: list[VariableLabel] = field(default_factory=list)
    start_date: str = ""
    end_date: str = ""
    granularity_text: str = ""
    gap_text: str | None = None
    fetched_at_text: str = ""


_FREQUENCY_MINUTES: dict[str, int] = {
    "15T": 15,
    "15min": 15,
    "30T": 30,
    "30min": 30,
    "H": 60,
    "1H": 60,
    "h": 60,
    "1h": 60,
    "5T": 5,
    "5min": 5,
}


def format_date(value: Any) -> str:
    """Render one timestamp as 「2026年1月1日」, never as an ISO string."""

    day = value if isinstance(value, date) and not isinstance(value, datetime) else _as_datetime(value)
    if day is None:
        return ""
    return f"{day.year}年{day.month}月{day.day}日"


def format_moment(value: Any, *, now: datetime | None = None) -> str:
    """Render a fetch time relative to today, as 「今天 15:32」."""

    moment = _as_datetime(value)
    if moment is None:
        return ""
    reference = (now or datetime.now(UTC)).astimezone()
    if moment.tzinfo is not None:
        moment = moment.astimezone(reference.tzinfo)
    days = (reference.date() - moment.date()).days
    clock = f"{moment.hour:02d}:{moment.minute:02d}"
    if days == 0:
        return f"今天 {clock}"
    if days == 1:
        return f"昨天 {clock}"
    return f"{moment.year}年{moment.month}月{moment.day}日 {clock}"


def format_granularity(frequency: str) -> str:
    """Turn a pandas offset alias into the sampling interval an analyst names."""

    minutes = _FREQUENCY_MINUTES.get(str(frequency).strip())
    if minutes is None:
        return ""
    if minutes % 60 == 0:
        hours = minutes // 60
        return "每小时一个点" if hours == 1 else f"每 {hours} 小时一个点"
    return f"每 {minutes} 分钟一个点"


def format_gap(missing_points: int) -> str | None:
    """Describe missing samples, or return None when the series is continuous."""

    return f"缺 {missing_points:,} 个点" if missing_points > 0 else None


def build_partial_summary(
    *,
    market: str,
    target_name: str,
    exogenous_names: list[str],
    frequency: str = "",
    table_names: dict[str, str] | None = None,
    variable_kinds: Mapping[str, VariableKind] | None = None,
) -> DataSummary:
    """Name what is already known before the data itself has been read.

    Everything still being established is left empty rather than guessed, which is
    how the panel knows to show 「待定」 for it.
    """

    lookup = table_names or {}
    kinds = variable_kinds or {}
    price_label = label_variable(target_name, table=lookup.get(target_name)).display
    if market and market not in {"unspecified", "unknown"}:
        price_label = f"{market} {price_label}"
    return DataSummary(
        price_label=price_label,
        variables=[
            label_variable(name, table=lookup.get(name), kind=kinds.get(name, "unknown"))
            for name in exogenous_names
        ],
        granularity_text=format_granularity(frequency) if frequency else "",
    )


def build_summary(
    *,
    market: str,
    target_name: str,
    exogenous_names: list[str],
    start_time: Any,
    end_time: Any,
    frequency: str,
    missing_points: int = 0,
    fetched_at: Any = None,
    table_names: dict[str, str] | None = None,
    variable_kinds: Mapping[str, VariableKind] | None = None,
) -> DataSummary:
    """Assemble the display summary deterministically, so it never varies per run.

    ``table_names`` maps a column to the table it came from and only steers the
    name lookup; the table name itself never reaches the returned strings.
    """

    known = build_partial_summary(
        market=market,
        target_name=target_name,
        exogenous_names=exogenous_names,
        frequency=frequency,
        table_names=table_names,
        variable_kinds=variable_kinds,
    )
    return replace(
        known,
        start_date=format_date(start_time),
        end_date=format_date(end_time),
        gap_text=format_gap(missing_points),
        fetched_at_text=format_moment(fetched_at or datetime.now(UTC).astimezone()),
    )


def summary_payload(summary: DataSummary | None) -> dict[str, Any] | None:
    """Render a summary as the plain dict a snapshot, a graph state, or a session stores."""

    if summary is None:
        return None
    return {
        "price_label": summary.price_label,
        "variables": [
            {"display": item.display, "resolved": item.resolved, "kind": item.kind}
            for item in summary.variables
        ],
        "start_date": summary.start_date,
        "end_date": summary.end_date,
        "granularity_text": summary.granularity_text,
        "gap_text": summary.gap_text,
        "fetched_at_text": summary.fetched_at_text,
    }


def parse_summary(value: Any) -> DataSummary:
    """Rebuild a summary from the dict a snapshot or a saved conversation stored.

    Anything unreadable becomes an empty summary rather than an error: the panel
    can show 「待定」, but it cannot show a traceback.
    """

    if isinstance(value, DataSummary):
        return value
    if not isinstance(value, dict):
        return DataSummary()
    raw_variables = value.get("variables")
    variables = [
        VariableLabel(
            display=str(item.get("display", "")),
            resolved=bool(item.get("resolved", True)),
            kind=(
                item.get("kind")
                if item.get("kind") in {"actual", "forecast", "unknown"}
                else "unknown"
            ),
        )
        for item in (raw_variables if isinstance(raw_variables, list) else [])
        if isinstance(item, dict)
    ]
    return DataSummary(
        price_label=str(value.get("price_label", "")),
        variables=variables,
        start_date=str(value.get("start_date", "")),
        end_date=str(value.get("end_date", "")),
        granularity_text=str(value.get("granularity_text", "")),
        gap_text=value.get("gap_text") or None,
        fetched_at_text=str(value.get("fetched_at_text", "")),
    )


def unresolved_variables(summary: DataSummary | None) -> list[str]:
    """Name the variables still shown under their original column name."""

    if summary is None:
        return []
    return [item.display for item in summary.variables if not item.resolved]


def _as_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None
