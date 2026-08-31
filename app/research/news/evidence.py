"""End-to-end evidence chain: conclusion → price window → event → news quote → hashes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

from app.research.news.analysis import AnalysisResult, EventWindowResult
from app.research.news.contracts import (
    DocumentRef,
    EventFeatureSnapshot,
    EvidenceSpan,
    ExtractionQuarantine,
    ExtractionTrace,
    MergedEvent,
    NewsDocument,
)
from app.research.news.merging import EventView

PACKAGE_VERSION = "1.0.0"


class EvidenceChainError(ValueError):
    """Raised when a conclusion cannot be traced back to the text it rests on."""


@dataclass(frozen=True)
class EvidenceLink:
    """One conclusion with the whole path back to the sentence that produced it."""

    conclusion: str
    conclusion_reason: str
    event_id: str
    event_type: str
    axis: str
    window_label: str
    window_start_at: datetime
    window_end_at: datetime
    deviation: float
    permutation_p_value: float
    corrected_p_value: float
    control_sample_size: int
    announcement_available_at: datetime
    effective_start_at: datetime | None
    document_version_ids: tuple[str, ...]
    content_hashes: tuple[str, ...]
    source_names: tuple[str, ...]
    extraction_traces: tuple[ExtractionTrace, ...] = ()
    quotes: tuple[tuple[str, str], ...] = ()

    @property
    def is_traceable(self) -> bool:
        return bool(self.document_version_ids) and bool(self.quotes)

    @property
    def primary_quotes(self) -> dict[str, str]:
        """One quote per field: the wording of the version that actually set that value.

        `quotes` keeps every citation for full provenance and is ordered by authority, but
        reading it with `dict()` would silently take the LAST (least authoritative) entry.
        Callers that want "the" quote for a field must use this.
        """

        chosen: dict[str, str] = {}
        for field_name, quote in self.quotes:
            chosen.setdefault(field_name, quote)
        return chosen


@dataclass(frozen=True)
class DataQualityReport:
    """Input health, stated before any price statistic is allowed to be read."""

    document_count: int
    version_count: int
    duplicate_version_count: int
    quarantined_count: int
    quarantine_reasons: dict[str, int] = field(default_factory=dict)
    merged_event_count: int = 0
    events_with_full_evidence: int = 0
    after_the_fact_event_count: int = 0

    @property
    def evidence_coverage(self) -> float:
        if self.merged_event_count == 0:
            return 0.0
        return round(self.events_with_full_evidence / self.merged_event_count, 6)


@dataclass(frozen=True)
class ResearchPackage:
    """The reproducible bundle: quality, events, results, evidence links and fingerprints."""

    package_version: str
    generated_for_as_of: datetime
    quality: DataQualityReport
    events: tuple[MergedEvent, ...]
    analysis: AnalysisResult
    evidence_links: tuple[EvidenceLink, ...]
    features: EventFeatureSnapshot | None = None
    method_notes: tuple[str, ...] = ()
    events_not_analyzed: tuple[tuple[str, str], ...] = ()
    quarantined: tuple[ExtractionQuarantine, ...] = ()

    @property
    def unexplained_events(self) -> tuple[str, ...]:
        """Events that neither produced a result nor stated why; there must never be any."""

        analyzed = {link.event_id for link in self.evidence_links}
        explained = analyzed | {event_id for event_id, _ in self.analysis.excluded_events}
        explained |= {event_id for event_id, _ in self.events_not_analyzed}
        return tuple(sorted(event.event_id for event in self.events if event.event_id not in explained))

    def fingerprint(self) -> dict[str, str]:
        return {
            "analysis_hash": self.analysis.content_hash(),
            "event_hash": self.analysis.event_hash,
            "feature_hash": self.features.content_hash if self.features is not None else "",
            "package_hash": self._package_hash(),
            "price_hash": self.analysis.price_hash,
        }

    def _package_hash(self) -> str:
        payload = json.dumps(
            {
                "analysis_hash": self.analysis.content_hash(),
                "as_of": self.generated_for_as_of.isoformat(),
                "evidence": [
                    {
                        "conclusion": link.conclusion,
                        "event_id": link.event_id,
                        "extraction_traces": [
                            trace.model_dump(mode="json") for trace in link.extraction_traces
                        ],
                        "quotes": [list(quote) for quote in link.quotes],
                        "window": link.window_label,
                    }
                    for link in self.evidence_links
                ],
                "package_version": self.package_version,
                "quality": {
                    "documents": self.quality.document_count,
                    # The quarantine list joins the fingerprint: a run that silently starts
                    # dropping documents must not produce the same hash as one that did not.
                    "quarantined": [
                        [item.document_version_id, item.reason_code] for item in self.quarantined
                    ],
                    "quarantined_count": self.quality.quarantined_count,
                    "versions": self.quality.version_count,
                },
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def untraceable_links(self) -> tuple[EvidenceLink, ...]:
        return tuple(link for link in self.evidence_links if not link.is_traceable)


def _refs_by_authority(event: MergedEvent) -> tuple[DocumentRef, ...]:
    """Order citations the way the merge decided the values, most authoritative first.

    This has to mirror `merge_event_records`: the originating lineage at its newest version
    is what set each field, so its wording is what a reviewer must see. Quoting a later
    third-party article would be traceable but would not be the text the value came from.
    """

    primary_lineage = min(
        event.document_refs, key=lambda ref: (ref.available_at, ref.document_version_id)
    ).document_id
    return tuple(
        sorted(
            event.document_refs,
            key=lambda ref: (
                ref.document_id != primary_lineage,
                -ref.version,
                -ref.available_at.timestamp(),
                ref.document_version_id,
            ),
        )
    )


def _quotes_for(
    event: MergedEvent, spans_by_version: dict[str, tuple[EvidenceSpan, ...]]
) -> tuple[tuple[str, str], ...]:
    quotes: list[tuple[str, str]] = []
    for ref in _refs_by_authority(event):
        for span in _spans_for_event(event, ref.document_version_id, spans_by_version):
            quotes.append((span.field_name, span.quote))
    return tuple(dict.fromkeys(quotes))


def _spans_for_event(
    event: MergedEvent,
    document_version_id: str,
    spans_by_version: dict[str, tuple[EvidenceSpan, ...]],
) -> tuple[EvidenceSpan, ...]:
    """Keep sibling events in one document from borrowing each other's evidence."""

    spans = spans_by_version.get(document_version_id, ())
    if not event.source_event_ids:
        return spans
    source_ids = set(event.source_event_ids)
    return tuple(span for span in spans if span.event_id is None or span.event_id in source_ids)


def build_evidence_links(
    analysis: AnalysisResult,
    events: Sequence[MergedEvent],
    spans_by_version: dict[str, tuple[EvidenceSpan, ...]],
) -> tuple[EvidenceLink, ...]:
    events_by_id = {event.event_id: event for event in events}
    links: list[EvidenceLink] = []
    for result in analysis.results:
        event = events_by_id.get(result.event_id)
        if event is None:
            raise EvidenceChainError(f"分析结果引用了不存在的事件：{result.event_id}")
        links.append(_link(result, event, spans_by_version))
    return tuple(links)


def _link(
    result: EventWindowResult,
    event: MergedEvent,
    spans_by_version: dict[str, tuple[EvidenceSpan, ...]],
) -> EvidenceLink:
    return EvidenceLink(
        conclusion=result.conclusion,
        conclusion_reason=result.conclusion_reason,
        event_id=event.event_id,
        event_type=event.event_type,
        axis=result.axis,
        window_label=result.metrics.window_label,
        window_start_at=result.metrics.start_at,
        window_end_at=result.metrics.end_at,
        deviation=result.metrics.deviation,
        permutation_p_value=result.permutation_p_value,
        corrected_p_value=result.corrected_p_value,
        control_sample_size=result.control_sample_size,
        announcement_available_at=event.announcement_available_at,
        effective_start_at=event.effective_start_at,
        document_version_ids=tuple(ref.document_version_id for ref in event.document_refs),
        content_hashes=tuple(ref.content_hash for ref in event.document_refs),
        source_names=tuple(dict.fromkeys(ref.source_name for ref in event.document_refs)),
        extraction_traces=event.extraction_traces,
        quotes=_quotes_for(event, spans_by_version),
    )


def build_quality_report(
    *,
    documents: Sequence[NewsDocument],
    version_count: int,
    duplicate_version_count: int,
    quarantined: Sequence[ExtractionQuarantine],
    view: EventView,
    spans_by_version: dict[str, tuple[EvidenceSpan, ...]],
) -> DataQualityReport:
    reasons: dict[str, int] = {}
    for item in quarantined:
        reasons[item.reason_code] = reasons.get(item.reason_code, 0) + 1
    full_evidence = sum(1 for event in view.events if _has_required_evidence(event, spans_by_version))
    after_the_fact = sum(
        1
        for event in view.events
        if event.lead_time_hours is not None and event.lead_time_hours < 0
    )
    return DataQualityReport(
        document_count=len({document.document_id for document in documents}),
        version_count=version_count,
        duplicate_version_count=duplicate_version_count,
        quarantined_count=len(quarantined),
        quarantine_reasons=reasons,
        merged_event_count=len(view.events),
        events_with_full_evidence=full_evidence,
        after_the_fact_event_count=after_the_fact,
    )


def _has_required_evidence(
    event: MergedEvent,
    spans_by_version: dict[str, tuple[EvidenceSpan, ...]],
) -> bool:
    """Whether every populated critical field has at least one source text span."""

    required = {"event_type", "relevance", "direction"}
    if event.affected_regions:
        required.add("affected_regions")
    if event.affected_assets:
        required.add("affected_assets")
    if event.capacity_mw is not None:
        required.add("capacity_mw")
    if event.effective_start_at is not None:
        required.add("effective_start_at")
    if event.effective_end_at is not None:
        required.add("effective_end_at")

    available = {
        span.field_name
        for ref in event.document_refs
        for span in _spans_for_event(event, ref.document_version_id, spans_by_version)
    }
    return bool(event.document_refs) and required.issubset(available)


_REPORT_QUOTE_FIELDS = ("event_type", "capacity_mw", "effective_start_at", "effective_end_at")
_CONCLUSIONS_NEEDING_REASON = frozenset({"contradicts_expected_direction", "insufficient_sample"})


def _report_quotes(link: EvidenceLink) -> tuple[tuple[str, str], ...]:
    """Show what the event claims, not just the keyword that classified it.

    A reader checking a conclusion wants the capacity and the window in the source's own
    words; the classification keyword alone is traceable but not informative.
    """

    by_field = link.primary_quotes
    chosen = tuple(
        (field_name, by_field[field_name]) for field_name in _REPORT_QUOTE_FIELDS if field_name in by_field
    )
    return chosen or link.quotes[:1]


CONCLUSION_LABELS = {
    "association_consistent_with_expected_direction": "与预期方向一致",
    "not_supported_by_current_data": "当前数据未提供支持",
    "insufficient_sample": "样本量不足",
    "contradicts_expected_direction": "与预期方向相反或不稳定",
}
DIRECTION_LABELS = {"up": "上行压力", "down": "下行压力", "mixed": "方向混合", "unknown": "方向未知"}
AXIS_LABELS = {"announcement": "公告可用轴", "effective": "事件生效轴"}
QUARANTINE_LABELS = {
    "ambiguous_multi_event": "同一文档命中多个事件类别",
    "ambiguous_event_time": "多个相对时间表述且无明确生效区间",
    "missing_effective_start": "短期事件没有可解析的生效开始时间",
    "event_contract_violation": "未通过事件契约校验",
}
QUOTE_FIELD_LABELS = {
    "relevance": "相关性",
    "event_type": "事件类型",
    "direction": "方向",
    "affected_regions": "区域",
    "affected_assets": "资产",
    "capacity_mw": "容量",
    "effective_start_at": "生效开始",
    "effective_end_at": "生效结束",
}


def _stamp(value: datetime | None) -> str:
    return "—" if value is None else value.isoformat().replace("+00:00", "Z")


def _render_header(package: ResearchPackage) -> list[str]:
    analysis = package.analysis
    clock = analysis.clock
    axis = AXIS_LABELS.get(analysis.axis, analysis.axis)
    return [
        "# P2 事件—电价证据包",
        "",
        f"- 视图时点（as_of）：{_stamp(package.generated_for_as_of)}",
        f"- 分析时间轴：{axis}（`{analysis.axis}`）",
        f"- 市场时钟：{clock.market} / {clock.timezone} / {clock.interval_minutes} 分钟结算",
        f"- 方法版本：{analysis.method.analysis_version}，多重比较校正：{analysis.method.correction}",
        "",
    ]


def _render_quality(package: ResearchPackage) -> list[str]:
    quality = package.quality
    lines = [
        "## 一、输入质量",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| 文档 / 版本 | {quality.document_count} 篇 / {quality.version_count} 个 |",
        f"| 完全相同的重复采集 | {quality.duplicate_version_count} 次（已丢弃，未计入） |",
        f"| 进入隔离区的文档 | {quality.quarantined_count} 篇 |",
        f"| 合并后事件 | {quality.merged_event_count} 个 |",
        f"| 关键字段证据覆盖率 | {quality.evidence_coverage:.0%} |",
        f"| 事后才可用的事件 | {quality.after_the_fact_event_count} 个（新闻晚于事件开始） |",
        "",
    ]
    lines.extend(_render_quarantine(package))
    return lines


def _render_quarantine(package: ResearchPackage) -> list[str]:
    """Name every document that was set aside.

    A count alone lets a growing quarantine pass as a healthy run; the whole point of the
    isolation bin is that someone can look inside it.
    """

    if not package.quarantined:
        return ["> 本次运行没有文档进入隔离区。", ""]

    lines = ["### 隔离区明细", "", "| 文档版本 | 原因 | 说明 |", "|---|---|---|"]
    lines.extend(
        f"| `{item.document_version_id}` | {QUARANTINE_LABELS.get(item.reason_code, item.reason_code)}"
        f"（`{item.reason_code}`） | {item.message} |"
        for item in package.quarantined
    )
    lines.extend(
        [
            "",
            "> 隔离的文档不产生事件，也不进入电价分析或事件特征表；它们在这里留痕以便复核。",
            "",
        ]
    )
    return lines


def _render_summary(package: ResearchPackage) -> list[str]:
    counts: dict[str, int] = {}
    for link in package.evidence_links:
        counts[link.conclusion] = counts.get(link.conclusion, 0) + 1
    total = len(package.evidence_links)
    event_count = len({link.event_id for link in package.evidence_links})
    window_count = len({link.window_label for link in package.evidence_links})

    skipped = len(package.events) - event_count
    tail = f"；另有 {skipped} 个事件未纳入分析，原因见第四节" if skipped else ""
    lines = [
        "## 二、结论摘要",
        "",
        f"共 {total} 项检验（进入分析的 {event_count} 个事件 × {window_count} 个预注册窗口）{tail}。",
        "",
        "| 结论 | 项数 |",
        "|---|---:|",
    ]
    for code, label in CONCLUSION_LABELS.items():
        lines.append(f"| {label} | {counts.get(code, 0)} |")
    lines.extend(
        [
            "",
            (
                "> 结论口径：未达显著并不等于证明没有影响；即便显著，事件关联也不等于因果效应。"
                "“样本量不足”与“当前数据未提供支持”是两回事，因此分开计数。"
            ),
            "",
        ]
    )
    return lines


def _render_event_section(
    index: int, links: list[EvidenceLink], event: MergedEvent | None
) -> list[str]:
    """One section per event, so its metadata and quotes are stated once, not per window."""

    head = links[0]
    title = head.event_type
    if event is not None and (event.affected_regions or event.affected_assets):
        parts = list(event.affected_regions) + list(event.affected_assets)
        title = f"{title} · {' · '.join(parts)}"

    lines = [f"### {index}. {title}", ""]
    lines.append(f"- 事件 ID：`{head.event_id}`")
    lines.append(
        f"- 公告可用：{_stamp(head.announcement_available_at)}｜生效："
        f"{_stamp(head.effective_start_at)} ~ {_stamp(event.effective_end_at) if event else '—'}"
    )
    if event is not None:
        if event.lead_time_hours is not None:
            posture = "事件开始前已可用" if event.lead_time_hours > 0 else "事件开始后才可用"
            lines.append(f"- 提前量：{event.lead_time_hours:+.2f} 小时（{posture}）")
        capacity = "未知" if event.capacity_mw is None else f"{event.capacity_mw:g} MW"
        lines.append(
            f"- 受影响量值：{capacity}｜事件语义预期：{DIRECTION_LABELS.get(event.direction, event.direction)}"
        )
    lines.append(f"- 来源：{'、'.join(head.source_names)}（{len(head.document_version_ids)} 个文档版本）")
    for trace in head.extraction_traces:
        lines.append(
            "- 抽取方法："
            f"{trace.model_name}｜Prompt {trace.prompt_version}｜Schema {trace.output_schema_version}｜"
            f"输入 `{trace.input_hash}`｜输出 `{trace.output_hash}`"
        )

    quotes = _report_quotes(head)
    if quotes:
        lines.extend(["- 原文依据："])
        lines.extend(
            f"  - {QUOTE_FIELD_LABELS.get(field_name, field_name)}：{quote}" for field_name, quote in quotes
        )
    lines.extend(
        [
            "",
            "| 窗口 | 偏离基线 | 原始 p | 校正后 p | 对照样本 | 结论 |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    lines.extend(
        f"| {link.window_label} | {link.deviation:+.2f} | {link.permutation_p_value:.4f} | "
        f"{link.corrected_p_value:.4f} | {link.control_sample_size} | "
        f"{CONCLUSION_LABELS.get(link.conclusion, link.conclusion)} |"
        for link in links
    )
    # Spell out only the outcomes a reader cannot infer from the numbers above.
    needs_reason = [link for link in links if link.conclusion in _CONCLUSIONS_NEEDING_REASON]
    if needs_reason:
        lines.append("")
        lines.extend(f"- {link.window_label}：{link.conclusion_reason}" for link in needs_reason)
    lines.append("")
    return lines


def render_report(package: ResearchPackage) -> str:
    """A human-readable summary; the machine-readable truth stays in the objects.

    Grouped by event rather than by test: one event tested over three windows used to
    repeat its whole provenance block three times, which buried the part that differs.
    """

    events_by_id = {event.event_id: event for event in package.events}
    grouped: dict[str, list[EvidenceLink]] = {}
    for link in package.evidence_links:
        grouped.setdefault(link.event_id, []).append(link)

    lines = _render_header(package)
    lines.extend(_render_quality(package))
    lines.extend(_render_summary(package))
    lines.extend(["## 三、逐事件证据", ""])
    for index, (event_id, links) in enumerate(grouped.items(), start=1):
        lines.extend(_render_event_section(index, links, events_by_id.get(event_id)))

    lines.extend(["## 四、未纳入分析的事件", ""])
    skipped = tuple(package.events_not_analyzed) + tuple(package.analysis.excluded_events)
    if skipped:
        lines.extend(f"- `{event_id}`：{reason}" for event_id, reason in skipped)
    else:
        lines.append("- 无")
    unexplained = package.unexplained_events
    if unexplained:
        lines.append("")
        lines.extend(f"- **{event_id}：既无结果也无原因，属于报告缺口**" for event_id in unexplained)
    lines.append("")

    lines.extend(["## 五、方法声明", ""])
    lines.extend(f"- {note}" for note in package.analysis.notes)
    method = package.analysis.method
    windows = "、".join(f"[0,{hours:g}h]" for hours in method.window_hours)
    placebos = "、".join(f"{days:+d} 天" for days in method.placebo_offset_days)
    lines.extend(
        [
            "",
            (
                f"- 预注册窗口：{windows}；安慰剂偏移：{placebos}；"
                f"置换抽样 {method.permutation_samples} 次，随机种子 {method.seed}。"
            ),
            "",
        ]
    )

    lines.extend(["## 六、指纹", "", "| 项 | 值 |", "|---|---|"])
    lines.extend(f"| {key} | `{value}` |" for key, value in sorted(package.fingerprint().items()) if value)
    return "\n".join(lines)
