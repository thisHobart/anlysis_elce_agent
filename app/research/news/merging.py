"""Merge extracted event candidates across reposts and revisions into one event per view."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from app.research.news.contracts import (
    EventRecord,
    ExtractionQuarantine,
    MergedEvent,
    NewsDocument,
)
from app.research.news.extraction import ObviousNewsEventExtractor
from app.research.news.versioning import NewsVersionStore, document_ref

MERGER_VERSION = "1.0.0"


@dataclass(frozen=True)
class EventView:
    """Everything one `as_of` instant can legitimately see."""

    as_of: datetime
    events: tuple[MergedEvent, ...] = ()
    quarantined: tuple[ExtractionQuarantine, ...] = ()
    visible_document_count: int = 0

    def by_id(self, event_id: str) -> MergedEvent | None:
        return next((event for event in self.events if event.event_id == event_id), None)

    @property
    def short_term_events(self) -> tuple[MergedEvent, ...]:
        return tuple(event for event in self.events if event.relevance == "short_term")


def merge_event_records(
    records: Iterable[tuple[EventRecord, NewsDocument]], *, as_of: datetime
) -> tuple[MergedEvent, ...]:
    """Group candidates by event identity, then decide which document gets to state each field.

    Three rules carry the weight here.

    The announcement time is the EARLIEST contributing document, because that is when the
    market could first have known.

    Field values come from the ORIGINATING lineage — the document that first reported the
    event — at its newest visible version. That is what lets a 12:00 correction replace the
    10:00 figure while a third party's later article cannot overwrite the operator's own
    number with a stale one.

    A field the originating lineage leaves empty may be filled from another contributor, in
    availability order. A repost that says nothing new therefore adds provenance and nothing
    else.
    """

    grouped: dict[str, list[tuple[EventRecord, NewsDocument]]] = {}
    for record, document in records:
        if record.document_version_id != document.document_version_id:
            raise ValueError("event record and document version must match")
        if document.available_at > as_of:
            continue
        grouped.setdefault(record.event_id, []).append((record, document))

    merged: list[MergedEvent] = []
    for event_id, members in grouped.items():
        ordered = sorted(members, key=lambda item: (item[1].available_at, item[1].version))
        earliest_document = ordered[0][1]
        primary_lineage = earliest_document.document_id
        lineage_members = [item for item in ordered if item[1].document_id == primary_lineage]
        primary_record = lineage_members[-1][0]
        fallbacks = [record for record, _ in ordered if record is not primary_record]

        def stated(field_name: str, *, _primary=primary_record, _fallbacks=fallbacks):
            value = getattr(_primary, field_name)
            if value not in (None, ()):
                return value
            for candidate in _fallbacks:
                other = getattr(candidate, field_name)
                if other not in (None, ()):
                    return other
            return value

        refs = tuple(
            sorted(
                (document_ref(document) for _, document in ordered),
                key=lambda ref: (ref.available_at, ref.document_version_id),
            )
        )
        effective_start_at = stated("effective_start_at")
        merged.append(
            MergedEvent(
                merger_version=MERGER_VERSION,
                event_id=event_id,
                as_of=as_of,
                relevance=primary_record.relevance,
                event_type=primary_record.event_type,
                affected_regions=stated("affected_regions"),
                affected_assets=stated("affected_assets"),
                capacity_mw=stated("capacity_mw"),
                announcement_available_at=earliest_document.available_at,
                effective_start_at=effective_start_at,
                effective_end_at=stated("effective_end_at") if effective_start_at is not None else None,
                direction=primary_record.direction,
                document_refs=refs,
                revision_count=len(refs),
            )
        )
    return tuple(sorted(merged, key=lambda event: (event.announcement_available_at, event.event_id)))


class AsOfEventAssembler:
    """Rebuild the event table exactly as it stood at any instant."""

    merger_version = MERGER_VERSION

    def __init__(
        self,
        store: NewsVersionStore,
        *,
        extractor: ObviousNewsEventExtractor | None = None,
    ) -> None:
        self._store = store
        self._extractor = extractor or ObviousNewsEventExtractor()

    def view_at(self, as_of: datetime) -> EventView:
        history = self._store.history_at(as_of)
        pairs: list[tuple[EventRecord, NewsDocument]] = []
        quarantined: list[ExtractionQuarantine] = []
        for document in history:
            result = self._extractor.extract(document)
            if result.quarantine is not None:
                quarantined.append(result.quarantine)
                continue
            pairs.extend((event, document) for event in result.events)
        return EventView(
            as_of=as_of,
            events=merge_event_records(pairs, as_of=as_of),
            quarantined=tuple(quarantined),
            visible_document_count=len(self._store.visible_at(as_of)),
        )

    def replay(self) -> tuple[EventView, ...]:
        """One view per instant at which the visible document set changes."""

        return tuple(self.view_at(instant) for instant in self._store.availability_timeline())
