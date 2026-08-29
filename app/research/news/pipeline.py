"""The one path from collected news to a reproducible event-price evidence package."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from app.research.news.adapters import CollectedNewsAdapter
from app.research.news.analysis import AnalysisMethod, AnalysisResult, EventPriceAnalyzer
from app.research.news.clock import MarketClock, TimeAxis, lead_time_table
from app.research.news.contracts import EventFeatureSnapshot, EvidenceSpan, NewsDocument
from app.research.news.evidence import (
    PACKAGE_VERSION,
    ResearchPackage,
    build_evidence_links,
    build_quality_report,
)
from app.research.news.extraction import ObviousNewsEventExtractor
from app.research.news.features import build_event_features
from app.research.news.merging import AsOfEventAssembler, EventView
from app.research.news.normalization import NewsNormalizer
from app.research.news.prices import PriceObservations
from app.research.news.versioning import NewsVersionStore


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

    @property
    def lead_times(self):
        return lead_time_table(self.view.events)


def collect_evidence_spans(
    documents: tuple[NewsDocument, ...],
    extractor: ObviousNewsEventExtractor,
) -> dict[str, tuple[EvidenceSpan, ...]]:
    """Map each document version to the exact text spans its event fields rest on."""

    spans: dict[str, tuple[EvidenceSpan, ...]] = {}
    for document in documents:
        result = extractor.extract(document)
        for event in result.events:
            spans[document.document_version_id] = event.evidence
    return spans


def run_news_price_study(
    *,
    adapter: CollectedNewsAdapter,
    prices: PriceObservations,
    as_of: datetime,
    clock: MarketClock | None = None,
    axis: TimeAxis = "effective",
    method: AnalysisMethod | None = None,
    extractor: ObviousNewsEventExtractor | None = None,
    include_irrelevant: bool = False,
) -> NewsPriceStudy:
    """Collected news in, evidence package out, with every gate applied in order.

    The order is not incidental. Documents are versioned before they are extracted, events
    are merged and gated to `as_of` before any price is touched, and the features are built
    from that same gated view — so nothing downstream can reach a document that had not yet
    arrived.
    """

    market_clock = clock or prices.clock
    extractor = extractor or ObviousNewsEventExtractor(market_timezone=market_clock.timezone)

    documents = NewsNormalizer().normalize_many(adapter.load())
    store = NewsVersionStore(documents)
    assembler = AsOfEventAssembler(store, extractor=extractor)
    view = assembler.view_at(as_of)

    skipped_types = set() if include_irrelevant else {"irrelevant", "unknown"}
    analyzable = [event for event in view.events if event.event_type not in skipped_types]
    # Anything dropped here must still be named in the report, or the event count silently
    # shrinks between the quality section and the results.
    not_analyzed = tuple(
        (event.event_id, f"事件类型为 `{event.event_type}`，按预注册规则不进入电价分析")
        for event in view.events
        if event.event_type in skipped_types
    )
    analysis = EventPriceAnalyzer(method).analyze(prices, analyzable, axis=axis)

    features = build_event_features(
        view.events,
        clock=market_clock,
        start_at=prices.start_at,
        end_at=prices.end_at,
        as_of=as_of,
    )

    quality = build_quality_report(
        documents=documents,
        version_count=store.version_count,
        duplicate_version_count=store.duplicates.duplicate_count,
        quarantined=view.quarantined,
        view=view,
    )
    links = build_evidence_links(analysis, view.events, collect_evidence_spans(documents, extractor))

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
    return NewsPriceStudy(
        as_of=as_of,
        clock=market_clock,
        documents=documents,
        store=store,
        view=view,
        analysis=analysis,
        features=features,
        package=package,
    )
