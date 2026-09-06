"""The one path from collected news to a reproducible event-price evidence package."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime

from app.research.news.adapters import CollectedNewsAdapter
from app.research.news.analysis import AnalysisMethod, AnalysisResult, EventPriceAnalyzer
from app.research.news.clock import MarketClock, TimeAxis, lead_time_table
from app.research.news.contracts import (
    EventExtractionResult,
    EventFeatureSnapshot,
    EvidenceSpan,
    ExtractionQuarantine,
    NewsDocument,
)
from app.research.news.evidence import (
    PACKAGE_VERSION,
    ResearchPackage,
    build_evidence_links,
    build_quality_report,
)
from app.research.news.extraction import NewsEventExtractor, ObviousNewsEventExtractor
from app.research.news.features import build_event_features
from app.research.news.merging import AsOfEventAssembler, EventView
from app.research.news.normalization import NewsNormalizer
from app.research.news.prices import PriceObservations
from app.research.news.quality import ResultQualityAssessment, evaluate_result_quality
from app.research.news.versioning import NewsVersionStore


class NewsPipelineError(ValueError):
    """Raised when inputs disagree on a boundary that would make results misleading."""


@dataclass(frozen=True)
class NewsPriceStudy:
    """Everything one run produced, with the intermediate stages kept inspectable."""

    as_of: datetime
    clock: MarketClock
    documents: tuple[NewsDocument, ...]
    store: NewsVersionStore
    view: EventView
    analysis: AnalysisResult
    features: EventFeatureSnapshot
    package: ResearchPackage
    result_quality: ResultQualityAssessment

    @property
    def lead_times(self):
        return lead_time_table(self.view.events)


def collect_evidence_spans(
    documents: tuple[NewsDocument, ...],
    extractor: NewsEventExtractor | None = None,
    *,
    extraction_results: tuple[EventExtractionResult, ...] | None = None,
) -> dict[str, tuple[EvidenceSpan, ...]]:
    """Map exact field evidence without invoking a model twice in one study."""

    spans: dict[str, tuple[EvidenceSpan, ...]] = {}
    if extraction_results is None:
        if extractor is None:
            raise ValueError("extractor is required when extraction_results are not provided")
        extraction_results = tuple(extractor.extract(document) for document in documents)
    for result in extraction_results:
        spans[result.document_version_id] = tuple(
            span for event in result.events for span in event.evidence
        )
    return spans


def run_news_price_study(
    *,
    adapter: CollectedNewsAdapter,
    prices: PriceObservations,
    as_of: datetime,
    clock: MarketClock | None = None,
    axis: TimeAxis = "effective",
    method: AnalysisMethod | None = None,
    extractor: NewsEventExtractor | None = None,
    include_irrelevant: bool = False,
) -> NewsPriceStudy:
    """Collected news in, evidence package out, with every gate applied in order.

    The order is not incidental. Documents are versioned before they are extracted, events
    are merged and gated to `as_of` before any price is touched, and the features are built
    from that same gated view — so nothing downstream can reach a document that had not yet
    arrived.
    """

    market_clock = clock or prices.clock
    if market_clock != prices.clock:
        raise NewsPipelineError(
            "研究时钟必须与价格时钟完全一致："
            f"研究={market_clock.market}/{market_clock.timezone}/{market_clock.interval_minutes}m，"
            f"价格={prices.clock.market}/{prices.clock.timezone}/{prices.clock.interval_minutes}m"
        )
    extractor = extractor or ObviousNewsEventExtractor(market_timezone=market_clock.timezone)
    extractor_timezone = getattr(extractor, "market_timezone", market_clock.timezone)
    if extractor_timezone != market_clock.timezone:
        raise NewsPipelineError(
            "新闻抽取器时区必须与市场时钟一致："
            f"抽取器={extractor_timezone}，市场={market_clock.timezone}"
        )

    documents = NewsNormalizer().normalize_many(adapter.load())
    store = NewsVersionStore(documents)
    matched_documents = tuple(
        document
        for document in documents
        if document.market_tags and market_clock.market in document.market_tags
    )
    extraction_store = NewsVersionStore(matched_documents)
    routed_out = tuple(
        ExtractionQuarantine(
            document_version_id=document.document_version_id,
            reason_code="market_mismatch",
            message=(
                f"新闻缺少市场标签，无法确认是否属于 {market_clock.market}"
                if not document.market_tags
                else f"新闻市场 {', '.join(document.market_tags)} 与目标市场 {market_clock.market} 不一致"
            ),
            extractor_id="news-market-router",
            extractor_version="1.0.0",
        )
        for document in store.visible_at(as_of)
        if not document.market_tags or market_clock.market not in document.market_tags
    )
    assembler = AsOfEventAssembler(extraction_store, extractor=extractor)
    view = assembler.view_at(as_of)
    if routed_out:
        view = replace(
            view,
            quarantined=(*routed_out, *view.quarantined),
            visible_document_count=len(store.visible_at(as_of)),
        )

    skipped_types = set() if include_irrelevant else {"irrelevant", "unknown"}

    def analysis_exclusion(event):
        if event.event_type in skipped_types:
            return f"事件类型为 `{event.event_type}`，按预注册规则不进入电价分析"
        if event.relevance != "short_term":
            return "长期事件只进入长期研究，不进入短期电价窗口"
        if event.status == "cancelled":
            return "事件已取消，不作为已发生的价格冲击"
        if event.effective_start_at is None:
            return "没有可对齐的生效时刻"
        return None

    analyzable = [event for event in view.events if analysis_exclusion(event) is None]
    # Anything dropped here must still be named in the report, or the event count silently
    # shrinks between the quality section and the results.
    not_analyzed = tuple(
        (event.event_id, reason)
        for event in view.events
        if (reason := analysis_exclusion(event)) is not None
    )
    analysis = EventPriceAnalyzer(method).analyze(prices, analyzable, axis=axis)

    features = build_event_features(
        view.events,
        clock=market_clock,
        start_at=prices.start_at,
        end_at=prices.end_at,
        as_of=as_of,
    )

    spans_by_version = collect_evidence_spans(
        documents,
        extraction_results=view.extraction_results,
    )
    quality = build_quality_report(
        documents=documents,
        version_count=store.version_count,
        duplicate_version_count=store.duplicates.duplicate_count,
        quarantined=view.quarantined,
        view=view,
        spans_by_version=spans_by_version,
    )
    links = build_evidence_links(analysis, view.events, spans_by_version)

    package = ResearchPackage(
        package_version=PACKAGE_VERSION,
        generated_for_as_of=as_of,
        quality=quality,
        events=view.events,
        analysis=analysis,
        evidence_links=links,
        features=features,
        method_notes=analysis.notes,
        events_not_analyzed=not_analyzed,
        quarantined=view.quarantined,
    )
    result_quality = evaluate_result_quality(package=package, documents=documents, prices=prices)
    return NewsPriceStudy(
        as_of=as_of,
        clock=market_clock,
        documents=documents,
        store=store,
        view=view,
        analysis=analysis,
        features=features,
        package=package,
        result_quality=result_quality,
    )
