"""Validate desktop data selections and build their runtime study context."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Literal

from app.desktop.session import InputRole, ResearchSession
from app.research.data.inference import infer_study_context
from app.research.schemas.study import StudyConfig

ROLE_LABELS: dict[InputRole, str] = {
    "target": "目标电价",
    "actuals": "实际外生变量",
    "forecasts": "预测外生变量",
}
FILE_FILTERS: dict[InputRole, str] = {
    "target": "数据文件 (*.csv *.parquet *.pq)",
    "actuals": "数据文件 (*.csv *.parquet *.pq)",
    "forecasts": "数据文件 (*.csv *.parquet *.pq)",
}

_DATE_TIME = re.compile(
    r"(?<!\d)(?P<year>20\d{2})\s*(?:年|[-/.])\s*(?P<month>\d{1,2})"
    r"\s*(?:月|[-/.])\s*(?P<day>\d{1,2})\s*日?"
    r"(?:(?:T|\s+)(?P<hour>[01]?\d|2[0-3])(?::|时)(?P<minute>[0-5]\d)?\s*分?)?"
)
_RANGE_WORDS = ("时间范围", "分析区间", "数据区间", "选择时间", "仅分析", "只分析")
_RESET_WORDS = (
    "恢复全部时间范围",
    "使用全部时间范围",
    "使用完整时间范围",
    "清除时间范围",
    "取消时间范围",
    "分析全部数据",
)


@dataclass(frozen=True)
class ChatTimeRangeRequest:
    """A deterministic, explicit study-window instruction found in one chat turn."""

    action: Literal["set", "clear"]
    start_time: datetime | None = None
    end_time: datetime | None = None


def parse_chat_time_range(question: str) -> ChatTimeRangeRequest | None:
    """Read only explicit full-date ranges; never guess a year or missing boundary."""

    compact = " ".join(question.split())
    if any(word in compact for word in _RESET_WORDS):
        return ChatTimeRangeRequest(action="clear")
    matches = list(_DATE_TIME.finditer(compact))
    if len(matches) < 2:
        if any(word in compact for word in _RANGE_WORDS) and matches:
            raise ValueError("请同时写明开始日期和结束日期，例如：仅分析 2026-01-01 到 2026-03-31")
        return None
    connector = compact[matches[0].end() : matches[1].start()]
    if not any(word in connector for word in ("到", "至", "~", "～", "—", "–")):
        return None

    def moment(match: re.Match[str], *, end: bool) -> datetime:
        hour = match.group("hour")
        if hour is None:
            clock = time.max if end else time.min
        else:
            clock = time(int(hour), int(match.group("minute") or 0))
        return datetime.combine(
            date(
                int(match.group("year")),
                int(match.group("month")),
                int(match.group("day")),
            ),
            clock,
        )

    try:
        start = moment(matches[0], end=False)
        end = moment(matches[1], end=True)
    except ValueError as exc:
        raise ValueError("时间范围中的日期或时间无效，请使用完整日期") from exc
    if start > end:
        raise ValueError("开始时间不能晚于结束时间")
    return ChatTimeRangeRequest(action="set", start_time=start, end_time=end)


def validate_input_path(role: InputRole, path: str | Path) -> Path:
    """Validate only existence and format; filenames are deliberately unrestricted."""

    if role not in ROLE_LABELS:
        raise ValueError(f"不支持的数据文件角色：{role}")
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"文件不存在：{resolved}")
    if resolved.suffix.casefold() not in {".csv", ".parquet", ".pq"}:
        raise ValueError(f"{ROLE_LABELS[role]}只支持 CSV/Parquet 文件")
    return resolved


def build_runtime_study(session: ResearchSession, *, output_directory: Path | None = None) -> StudyConfig:
    """Infer the internal execution contract from the three desktop data roles."""

    if not session.inputs["target"].path:
        raise ValueError("执行 EDA 前至少需要选择目标电价文件")
    target = validate_input_path("target", session.inputs["target"].path)
    actuals = (
        validate_input_path("actuals", session.inputs["actuals"].path)
        if session.inputs["actuals"].path
        else None
    )
    forecasts = (
        validate_input_path("forecasts", session.inputs["forecasts"].path)
        if session.inputs["forecasts"].path
        else None
    )
    config = infer_study_context(
        target_path=target,
        actuals_path=actuals,
        forecasts_path=forecasts,
        output_directory=output_directory,
    )
    study_updates = {
        "start_time": (
            datetime.fromisoformat(session.analysis_start_time)
            if session.analysis_start_time
            else None
        ),
        "end_time": (
            datetime.fromisoformat(session.analysis_end_time)
            if session.analysis_end_time
            else None
        ),
    }
    config = config.model_copy(update={"study": config.study.model_copy(update=study_updates)})
    if session.source_kind == "database":
        config = config.model_copy(
            update={
                "study": config.study.model_copy(
                    update={
                        "name": f"{session.region_id}_actual_price_study",
                        "market": session.region_market,
                        "timezone": session.region_timezone,
                        "frequency": session.database_fetch_details.get("frequency", config.study.frequency),
                    }
                )
            }
        )
    return config
