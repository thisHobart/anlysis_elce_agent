"""Deterministic baseline extraction for deliberately explicit P2 test news."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import ValidationError

from app.research.news.contracts import (
    EventDirection,
    EventExtractionBatch,
    EventExtractionResult,
    EventRecord,
    EvidenceSpan,
    ExtractionQuarantine,
    NewsDocument,
    NewsEventType,
    NewsRelevance,
    QuarantineReason,
    TimeResolution,
)

EXTRACTOR_ID = "obvious-news-rule-baseline"
EXTRACTOR_VERSION = "1.1.0"

_ISO_INSTANT = r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:\d{2})"
_START_PATTERN = re.compile(rf"(?:事件开始|生效开始|事件发生时间)[:：]\s*(?P<value>{_ISO_INSTANT})")
_END_PATTERN = re.compile(rf"(?:事件结束|生效结束|预计结束)[:：]\s*(?P<value>{_ISO_INSTANT})")
# Real bulletins date events against their own publication ("次日 14:00"). Only a relative
# DAY plus an explicitly stated CLOCK time is resolved. A vague period such as "今日上午"
# has no defensible instant, so it is left for review instead of being invented.
_RELATIVE_DAY_OFFSETS = {
    "前日": -2,
    "昨日": -1,
    "昨天": -1,
    "今日": 0,
    "今天": 0,
    "当日": 0,
    "本日": 0,
    "次日": 1,
    "翌日": 1,
    "明日": 1,
    "明天": 1,
    "后日": 2,
}
_DAY_WORDS = "|".join(sorted(_RELATIVE_DAY_OFFSETS, key=len, reverse=True))
_CLOCK = r"\d{1,2}[:：]\d{2}"
_RANGE_SEPARATOR = r"(?:[—–\-－~～]|至|到)"
_RELATIVE_RANGE_PATTERN = re.compile(
    rf"(?P<day>{_DAY_WORDS})\s*(?P<start>{_CLOCK})\s*{_RANGE_SEPARATOR}\s*(?P<end>{_CLOCK})"
)
_RELATIVE_POINT_PATTERN = re.compile(rf"(?P<day>{_DAY_WORDS})\s*(?P<value>{_CLOCK})")

_REGION_PATTERN = re.compile(r"\bTEST_(?:NORTH|SOUTH|EAST|WEST|CENTRAL)\b")
_ASSET_PATTERN = re.compile(r"(?:资产|机组|联络线)[:：\s]+(?P<value>[A-Z][A-Z0-9_-]{1,31})\b")

# A bare "1200 MW" is never taken as the affected magnitude: a nameplate figure and an
# outage figure look identical to a regex. Only an explicit change/affected qualifier
# standing immediately in front of the number is trusted; anything else leaves capacity
# unknown rather than guessing.
_MAGNITUDE_QUALIFIER = (
    r"(?:不可用容量|受影响(?:容量)?|(?:可用)?容量恢复"
    r"|(?:进口|出口|输电|输送|送电)?能力(?:减少|降低|下降|增加|提升)"
    r"|出力(?:预计)?(?:增加|减少|下降)"
    r"|需求(?:预计)?(?:增加|减少)"
    r"|预计(?:增加|减少)|增加|减少|恢复)"
)
_CAPACITY_PATTERN = re.compile(
    rf"{_MAGNITUDE_QUALIFIER}[^0-9]{{0,4}}(?P<value>\d+(?:\.\d+)?)\s*MW\b",
    re.IGNORECASE,
)

_VALUELESS_TYPES = frozenset({"irrelevant", "unknown"})


@dataclass(frozen=True)
class _EventRule:
    event_type: NewsEventType
    relevance: NewsRelevance
    direction: EventDirection
    patterns: tuple[re.Pattern[str], ...]


_RULES = (
    _EventRule("irrelevant", "irrelevant", "unknown", (re.compile(r"慈善长跑"), re.compile(r"员工运动会"))),
    _EventRule(
        "generation_restore",
        "short_term",
        "down",
        (re.compile(r"已恢复运行"), re.compile(r"恢复并网")),
    ),
    _EventRule(
        "generation_outage",
        "short_term",
        "up",
        (re.compile(r"突发停运"), re.compile(r"非计划停运")),
    ),
    _EventRule(
        "transmission_constraint",
        "short_term",
        "up",
        (re.compile(r"进口能力减少"), re.compile(r"输电能力降低")),
    ),
    _EventRule(
        "demand_shock",
        "short_term",
        "up",
        (re.compile(r"需求激增"), re.compile(r"需求预计增加")),
    ),
    _EventRule(
        "renewable_supply_change",
        "short_term",
        "down",
        (re.compile(r"风电可用出力预计增加"), re.compile(r"光伏可用出力预计增加")),
    ),
    _EventRule(
        "fuel_supply_change",
        "short_term",
        "up",
        (re.compile(r"燃料供应中断"), re.compile(r"天然气供应减少")),
    ),
    _EventRule(
        "policy_long_horizon",
        "long_horizon",
        "unknown",
        (re.compile(r"计划在\s*20\d{2}\s*年关闭"), re.compile(r"长期退役计划")),
    ),
)

_UNKNOWN_RULE = _EventRule("unknown", "irrelevant", "unknown", ())


@dataclass(frozen=True)
class _LocatedMatch:
    text_field: Literal["title", "body"]
    match: re.Match[str]


@dataclass(frozen=True)
class _ResolvedTimes:
    """Event window plus how it was obtained, or the reason it could not be pinned down."""

    effective_start_at: datetime | None = None
    effective_end_at: datetime | None = None
    resolution: TimeResolution | None = None
    evidence: tuple[EvidenceSpan, ...] = ()
    ambiguity: str | None = None


def _canonical_hash(value) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _search(pattern: re.Pattern[str], document: NewsDocument, fields: tuple[str, ...]) -> _LocatedMatch | None:
    for text_field in fields:
        match = pattern.search(getattr(document, text_field))
        if match is not None:
            return _LocatedMatch(text_field=text_field, match=match)
    return None


def _first_match(pattern: re.Pattern[str], document: NewsDocument) -> _LocatedMatch | None:
    """Locate a classification keyword; a headline is an acceptable place to state it."""

    return _search(pattern, document, ("title", "body"))


def _first_value_match(pattern: re.Pattern[str], document: NewsDocument) -> _LocatedMatch | None:
    """Locate a field value. The body is authoritative; headlines round and abbreviate."""

    return _search(pattern, document, ("body", "title"))


def _all_matches(pattern: re.Pattern[str], document: NewsDocument) -> tuple[_LocatedMatch, ...]:
    matches: list[_LocatedMatch] = []
    for text_field in ("title", "body"):
        matches.extend(
            _LocatedMatch(text_field=text_field, match=match)
            for match in pattern.finditer(getattr(document, text_field))
        )
    return tuple(matches)


def _distinct_by_value(
    matches: tuple[_LocatedMatch, ...], *, group: str | int
) -> tuple[tuple[str, _LocatedMatch], ...]:
    """Keep the first span of each distinct captured value so evidence stays one per value."""

    seen: dict[str, _LocatedMatch] = {}
    for located in matches:
        seen.setdefault(located.match.group(group), located)
    return tuple(seen.items())


def _distinct_relative_points(document: NewsDocument) -> tuple[_LocatedMatch, ...]:
    """Collect distinct relative day/clock phrases so a document naming several stays ambiguous."""

    seen: dict[str, _LocatedMatch] = {}
    for text_field in ("body", "title"):
        for match in _RELATIVE_POINT_PATTERN.finditer(getattr(document, text_field)):
            key = f"{match.group('day')}|{match.group('value').replace('：', ':')}"
            seen.setdefault(key, _LocatedMatch(text_field=text_field, match=match))
    return tuple(seen.values())


def _evidence(
    document: NewsDocument,
    *,
    field_name: str,
    located: _LocatedMatch,
    group_name: str | None = None,
) -> EvidenceSpan:
    start, end = located.match.span(group_name) if group_name else located.match.span()
    source_text = getattr(document, located.text_field)
    return EvidenceSpan(
        field_name=field_name,
        document_version_id=document.document_version_id,
        text_field=located.text_field,
        start_char=start,
        end_char=end,
        quote=source_text[start:end],
    )


def _parse_utc_instant(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("event timestamp must include a timezone offset")
    return parsed.astimezone(UTC)


def _parse_clock(value: str) -> time | None:
    hour_text, minute_text = value.replace("：", ":").split(":")
    hour, minute = int(hour_text), int(minute_text)
    if hour > 23 or minute > 59:
        return None
    return time(hour=hour, minute=minute)


def _resolve_relative_instant(
    day_word: str, clock: str, *, anchor_at: datetime, market_zone: ZoneInfo
) -> datetime | None:
    """Anchor a relative day on the publication date as it reads in the market timezone."""

    parsed_clock = _parse_clock(clock)
    if parsed_clock is None:
        return None
    local_day: date = anchor_at.astimezone(market_zone).date() + timedelta(days=_RELATIVE_DAY_OFFSETS[day_word])
    return datetime.combine(local_day, parsed_clock, tzinfo=market_zone).astimezone(UTC)


class ObviousNewsEventExtractor:
    """Parse explicit fixture wording; this is a test baseline, not a production NLP claim."""

    extractor_id = EXTRACTOR_ID
    extractor_version = EXTRACTOR_VERSION

    def __init__(self, *, market_timezone: str = "UTC") -> None:
        """The market clock decides which calendar day "次日" lands on, so it stays explicit."""

        try:
            self._market_zone = ZoneInfo(market_timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"unknown market timezone: {market_timezone}") from exc
        self.market_timezone = market_timezone

    def extract(self, document: NewsDocument) -> EventExtractionResult:
        matched = self._match_rules(document)
        if len(matched) > 1:
            types = "、".join(sorted(rule.event_type for rule, _ in matched))
            return self._quarantine(
                document,
                reason_code="ambiguous_multi_event",
                message=f"文档同时命中多个事件类别（{types}）；规则基线不猜测主次，需要人工复核或模型抽取",
            )

        rule, rule_match = matched[0] if matched else (_UNKNOWN_RULE, None)
        evidence: list[EvidenceSpan] = []

        if rule_match is not None:
            evidence.extend(
                _evidence(document, field_name=field_name, located=rule_match)
                for field_name in ("relevance", "event_type", "direction")
            )

        carries_values = rule.event_type not in _VALUELESS_TYPES

        region_matches = (
            _distinct_by_value(_all_matches(_REGION_PATTERN, document), group=0) if carries_values else ()
        )
        regions = tuple(value for value, _ in region_matches)
        evidence.extend(
            _evidence(document, field_name="affected_regions", located=located) for _, located in region_matches
        )

        asset_matches = (
            _distinct_by_value(_all_matches(_ASSET_PATTERN, document), group="value") if carries_values else ()
        )
        assets = tuple(value for value, _ in asset_matches)
        evidence.extend(
            _evidence(document, field_name="affected_assets", located=located, group_name="value")
            for _, located in asset_matches
        )

        capacity_match = _first_value_match(_CAPACITY_PATTERN, document) if carries_values else None
        capacity_mw = float(capacity_match.match.group("value")) if capacity_match is not None else None
        if capacity_match is not None:
            evidence.append(_evidence(document, field_name="capacity_mw", located=capacity_match, group_name="value"))

        times = self._resolve_times(document) if carries_values else _ResolvedTimes()
        if times.ambiguity is not None:
            return self._quarantine(document, reason_code="ambiguous_event_time", message=times.ambiguity)
        effective_start_at = times.effective_start_at
        effective_end_at = times.effective_end_at
        evidence.extend(times.evidence)

        if rule.relevance == "short_term" and effective_start_at is None:
            return self._quarantine(
                document,
                reason_code="missing_effective_start",
                message=(
                    f"识别为短期事件（{rule.event_type}）但正文没有可解析的生效开始时间；"
                    "只接受绝对时间戳或“相对日期 + 明确时刻”，模糊时段不予推断"
                ),
            )

        identity = {
            "affected_assets": assets,
            "affected_regions": regions,
            "effective_start_at": effective_start_at.isoformat() if effective_start_at is not None else None,
            "event_type": rule.event_type,
        }
        if rule.event_type in _VALUELESS_TYPES:
            identity["document_id"] = document.document_id
        event_id = f"evt_{_canonical_hash(identity)[:24]}"

        try:
            event = EventRecord(
                event_id=event_id,
                document_version_id=document.document_version_id,
                relevance=rule.relevance,
                event_type=rule.event_type,
                affected_regions=regions,
                affected_assets=assets,
                capacity_mw=capacity_mw,
                effective_start_at=effective_start_at,
                effective_end_at=effective_end_at,
                direction=rule.direction,
                time_resolution=times.resolution,
                extractor_id=self.extractor_id,
                extractor_version=self.extractor_version,
                evidence=tuple(evidence),
            )
        except ValidationError as exc:
            return self._quarantine(
                document,
                reason_code="event_contract_violation",
                message=f"抽取候选未通过事件契约校验（{exc.error_count()} 项）：{exc.errors()[0]['msg']}",
            )

        return EventExtractionResult(document_version_id=document.document_version_id, events=(event,))

    def extract_many(self, documents: tuple[NewsDocument, ...]) -> EventExtractionBatch:
        """One unusable document is quarantined; it never discards the usable ones."""

        return EventExtractionBatch(results=tuple(self.extract(document) for document in documents))

    def _resolve_times(self, document: NewsDocument) -> _ResolvedTimes:
        """Prefer a stated absolute stamp; fall back to a relative day anchored on publication."""

        start_match = _first_value_match(_START_PATTERN, document)
        if start_match is not None:
            end_match = _first_value_match(_END_PATTERN, document)
            evidence = [
                _evidence(document, field_name="effective_start_at", located=start_match, group_name="value")
            ]
            if end_match is not None:
                evidence.append(
                    _evidence(document, field_name="effective_end_at", located=end_match, group_name="value")
                )
            return _ResolvedTimes(
                effective_start_at=_parse_utc_instant(start_match.match.group("value")),
                effective_end_at=(
                    _parse_utc_instant(end_match.match.group("value")) if end_match is not None else None
                ),
                resolution=TimeResolution(basis="stated_absolute"),
                evidence=tuple(evidence),
            )

        anchor_at = document.published_at
        derived = TimeResolution(
            basis="derived_from_publication",
            anchor_at=anchor_at,
            market_timezone=self.market_timezone,
        )

        range_match = _first_value_match(_RELATIVE_RANGE_PATTERN, document)
        if range_match is not None:
            start = self._relative(range_match.match.group("day"), range_match.match.group("start"), anchor_at)
            end = self._relative(range_match.match.group("day"), range_match.match.group("end"), anchor_at)
            if start is not None and end is not None:
                if end < start:
                    # A window that reads backwards spans midnight into the following day.
                    end += timedelta(days=1)
                return _ResolvedTimes(
                    effective_start_at=start,
                    effective_end_at=end,
                    resolution=derived,
                    evidence=(
                        _evidence(document, field_name="effective_start_at", located=range_match),
                        _evidence(document, field_name="effective_end_at", located=range_match),
                    ),
                )

        point_matches = _distinct_relative_points(document)
        if len(point_matches) > 1:
            quoted = "、".join(located.match.group(0) for located in point_matches)
            return _ResolvedTimes(
                ambiguity=f"正文出现多个相对时间表述（{quoted}）且没有明确的生效区间；规则基线不猜测哪一个是事件时间"
            )
        if len(point_matches) == 1:
            located = point_matches[0]
            start = self._relative(located.match.group("day"), located.match.group("value"), anchor_at)
            if start is not None:
                return _ResolvedTimes(
                    effective_start_at=start,
                    resolution=derived,
                    evidence=(_evidence(document, field_name="effective_start_at", located=located),),
                )
        return _ResolvedTimes()

    def _relative(self, day_word: str, clock: str, anchor_at: datetime) -> datetime | None:
        return _resolve_relative_instant(day_word, clock, anchor_at=anchor_at, market_zone=self._market_zone)

    def _quarantine(
        self, document: NewsDocument, *, reason_code: QuarantineReason, message: str
    ) -> EventExtractionResult:
        return EventExtractionResult(
            document_version_id=document.document_version_id,
            quarantine=ExtractionQuarantine(
                document_version_id=document.document_version_id,
                reason_code=reason_code,
                message=message,
                extractor_id=self.extractor_id,
                extractor_version=self.extractor_version,
            ),
        )

    @staticmethod
    def _match_rules(document: NewsDocument) -> tuple[tuple[_EventRule, _LocatedMatch], ...]:
        """Collect every distinct rule the text triggers so ambiguity stays visible."""

        matched: list[tuple[_EventRule, _LocatedMatch]] = []
        for rule in _RULES:
            for pattern in rule.patterns:
                located = _first_match(pattern, document)
                if located is not None:
                    matched.append((rule, located))
                    break
        return tuple(matched)
