"""Merge extracted event candidates across reposts and revisions into one event per view."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from app.research.news.contracts import (
    EventExtractionResult,
    EventRecord,
    EventStateRevision,
    ExtractionQuarantine,
    MergedEvent,
    NewsDocument,
)
from app.research.news.entity_resolution import EntityResolution
from app.research.news.extraction import NewsEventExtractor, ObviousNewsEventExtractor
from app.research.news.versioning import NewsVersionStore, document_ref

MERGER_VERSION = "1.3.0"


def _source_authority(document: NewsDocument) -> int:
    """Lower is more authoritative; unknown sources remain usable but cannot outrank primary data."""

    tier = str(document.raw_metadata.get("source_tier", "")).casefold()
    if tier.startswith("primary_") or any(
        word in tier for word in ("operator", "regulator", "government", "official")
    ):
        return 0
    if any(word in tier for word in ("secondary", "media", "repost", "aggregator")):
        return 2
    return 1


@dataclass(frozen=True)
class EventView:
    """Everything one `as_of` instant can legitimately see."""

    as_of: datetime
    events: tuple[MergedEvent, ...] = ()
    quarantined: tuple[ExtractionQuarantine, ...] = ()
    extraction_results: tuple[EventExtractionResult, ...] = ()
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

    Field values prefer the newest visible revision from the most authoritative source.
    That lets a 12:00 operator correction replace the 10:00 figure while a less
    authoritative repost cannot overwrite it with a stale value. The earliest contributor
    still determines when the event first became knowable.

    A field the chosen source leaves empty may be filled from another contributor unless an
    authoritative revision explicitly clears it. A repost that says nothing new therefore
    adds provenance and nothing else.
    """

    candidates: list[tuple[EventRecord, NewsDocument]] = []
    for record, document in records:
        if record.document_version_id != document.document_version_id:
            raise ValueError("event record and document version must match")
        if document.available_at > as_of:
            continue
        candidates.append((record, document))

    # Event fields such as effective_start_at are correctable and therefore cannot be the
    # only grouping key. Connect records either by their extracted identity (reposts) or by
    # source document lineage plus event type (revisions of one source document). The
    # earliest member's event_id remains the canonical ID so an as_of replay is stable.
    parent = list(range(len(candidates)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    # Give same-type siblings a stable ordinal within each document version. Without it, two
    # outages of the same asset are joined transitively through the next article revision.
    sibling_groups: dict[tuple[str, str, tuple[str, ...], tuple[str, ...]], list[int]] = {}
    for index, (record, document) in enumerate(candidates):
        sibling_groups.setdefault(
            (document.document_version_id, record.event_type, record.asset_keys,
             (record.region_keys, record.group_keys, record.market_keys)),
            [],
        ).append(index)
    sibling_ordinal: dict[int, int] = {}
    for indices in sibling_groups.values():
        for ordinal, index in enumerate(
            sorted(
                indices,
                key=lambda item: (
                    candidates[item][0].effective_start_at or datetime.min.replace(tzinfo=as_of.tzinfo),
                    candidates[item][0].effective_end_at or datetime.max.replace(tzinfo=as_of.tzinfo),
                    candidates[item][0].event_id,
                ),
            )
        ):
            sibling_ordinal[index] = ordinal

    by_event_id: dict[str, int] = {}
    by_lineage: dict[tuple[str, str, tuple[str, ...], tuple[str, ...], int], int] = {}
    for index, (record, document) in enumerate(candidates):
        event_owner = by_event_id.setdefault(record.event_id, index)
        lineage_key = (
            document.document_id,
            record.event_type,
            record.asset_keys,
            (record.region_keys, record.group_keys, record.market_keys),
            sibling_ordinal[index],
        )
        lineage_owner = by_lineage.setdefault(lineage_key, index)
        union(index, event_owner)
        # One article may legitimately contain two events of the same type. The lineage
        # fallback exists for corrections across versions, not to collapse sibling events
        # produced from the same document version.
        if candidates[lineage_owner][1].document_version_id != document.document_version_id:
            union(index, lineage_owner)

    # Clearing a group changes its identity key. Link that explicit withdrawal only
    # when each earlier source version has one unambiguous matching event.
    for index, (record, document) in enumerate(candidates):
        if "asset_groups" not in record.cleared_fields:
            continue
        prior_versions: dict[str, list[int]] = {}
        for prior_index, (prior, prior_document) in enumerate(candidates):
            if (prior_document.document_id == document.document_id
                    and prior_document.document_version_id != document.document_version_id
                    and prior_document.available_at < document.available_at
                    and prior.event_type == record.event_type and prior.asset_keys == record.asset_keys
                    and prior.region_keys == record.region_keys and prior.market_keys == record.market_keys):
                prior_versions.setdefault(prior_document.document_version_id, []).append(prior_index)
        for matches in prior_versions.values():
            if len(matches) == 1:
                union(index, matches[0])

    grouped: dict[int, list[tuple[EventRecord, NewsDocument]]] = {}
    for index, candidate in enumerate(candidates):
        grouped.setdefault(find(index), []).append(candidate)

    merged: list[MergedEvent] = []
    for members in grouped.values():
        ordered = sorted(members, key=lambda item: (item[1].available_at, item[1].version))
        event_id = ordered[0][0].event_id
        earliest_document = ordered[0][1]

        def selected_fields(
            visible: list[tuple[EventRecord, NewsDocument]],
        ) -> tuple[EventRecord, dict[str, object]]:
            lineages = tuple(dict.fromkeys(document.document_id for _, document in visible))
            visible_primary_lineage = min(
                lineages,
                key=lambda lineage: min(
                    (
                        _source_authority(document),
                        document.available_at,
                        document.document_id,
                    )
                    for _, document in visible
                    if document.document_id == lineage
                ),
            )
            primary_history = [
                item for item in visible if item[1].document_id == visible_primary_lineage
            ]
            visible_primary = primary_history[-1][0]
            # First inherit an older value from the same source lineage. If it never stated the
            # field, use the newest version of the best available contributing source.
            fallbacks = [record for record, _ in reversed(primary_history[:-1])]
            other_latest = [
                [item for item in visible if item[1].document_id == lineage][-1]
                for lineage in lineages
                if lineage != visible_primary_lineage
            ]
            fallbacks.extend(
                record
                for record, _ in sorted(
                    other_latest,
                    key=lambda item: (
                        _source_authority(item[1]),
                        item[1].available_at,
                        item[1].document_id,
                    ),
                )
            )

            def stated(field_name: str):
                value = getattr(visible_primary, field_name)
                if field_name in visible_primary.cleared_fields:
                    return value
                if value not in (None, ()):
                    return value
                for candidate in fallbacks:
                    if field_name in candidate.cleared_fields:
                        # Withdrawal is a persistent tombstone, not an omitted value.
                        # A later omission must never resurrect the pre-withdrawal state.
                        return getattr(candidate, field_name)
                    other = getattr(candidate, field_name)
                    if other not in (None, ()):
                        return other
                return value

            fields = {
                field_name: stated(field_name)
                for field_name in (
                    "affected_regions",
                    "affected_assets",
                    "capacity_mw",
                    "magnitude",
                    "effective_start_at",
                    "effective_end_at",
                )
            }
            # Resolve each entity family from the same donor as its raw field; retain
            # donor mentions together with keys, including at historical cutoffs.
            resolution = visible_primary.entity_resolution
            if resolution is not None:
                parts = {}
                mentions = {}
                for part, raw_field in (("mentioned_regions", "affected_regions"),
                                        ("affected_assets", "affected_assets"),
                                        ("asset_groups", "asset_groups")):
                    selected = ()
                    for donor in (visible_primary, *fallbacks):
                        if raw_field in donor.cleared_fields:
                            break
                        if donor.entity_resolution is None:
                            continue
                        selected = getattr(donor.entity_resolution, part)
                        if selected:
                            ids = {mid for item in selected for mid in item.mention_ids}
                            mentions.update({m.mention_id: m for m in donor.raw_entity_mentions
                                             if m.mention_id in ids})
                            break
                    parts[part] = selected
                fields["entity_resolution"] = EntityResolution(
                    market_scope=resolution.market_scope,
                    raw_entity_mentions=tuple(mentions[k] for k in sorted(mentions)), **parts)
                fields["affected_regions"] = tuple(item.value for item in parts["mentioned_regions"])
                fields["affected_assets"] = tuple(item.value for item in parts["affected_assets"])
            else:
                fields["entity_resolution"] = None
            return visible_primary, fields

        primary_record, fields = selected_fields(ordered)

        refs = tuple(
            sorted(
                (document_ref(document) for _, document in ordered),
                key=lambda ref: (ref.available_at, ref.document_version_id),
            )
        )
        trace_list = []
        for record, _ in ordered:
            if record.extraction_trace is not None and record.extraction_trace not in trace_list:
                trace_list.append(record.extraction_trace)
        extraction_traces = tuple(trace_list)
        effective_start_at = fields["effective_start_at"]
        revision_instants = tuple(dict.fromkeys(document.available_at for _, document in ordered))
        state_history: list[EventStateRevision] = []
        for instant in revision_instants:
            visible = [item for item in ordered if item[1].available_at <= instant]
            state_record, state_fields = selected_fields(visible)
            state_start = state_fields["effective_start_at"]
            state_history.append(
                EventStateRevision(
                    available_at=instant,
                    relevance=state_record.relevance,
                    event_type=state_record.event_type,
                    affected_regions=state_fields["affected_regions"],
                    affected_assets=state_fields["affected_assets"],
                    entity_resolution=state_fields["entity_resolution"],
                    capacity_mw=state_fields["capacity_mw"],
                    effective_start_at=state_start,
                    effective_end_at=(
                        state_fields["effective_end_at"] if state_start is not None else None
                    ),
                    direction=state_record.direction,
                    status=state_record.status,
                    physical_effect=state_record.physical_effect,
                    review_status=state_record.review_status,
                )
            )
        merged.append(
            MergedEvent(
                merger_version=MERGER_VERSION,
                event_id=event_id,
                as_of=as_of,
                relevance=primary_record.relevance,
                event_type=primary_record.event_type,
                affected_regions=fields["affected_regions"],
                affected_assets=fields["affected_assets"],
                entity_resolution=fields["entity_resolution"],
                capacity_mw=fields["capacity_mw"],
                magnitude=fields["magnitude"],
                announcement_available_at=earliest_document.available_at,
                effective_start_at=effective_start_at,
                effective_end_at=fields["effective_end_at"] if effective_start_at is not None else None,
                direction=primary_record.direction,
                status=primary_record.status,
                physical_effect=primary_record.physical_effect,
                confidence=primary_record.confidence,
                time_resolution=primary_record.time_resolution,
                review_status=primary_record.review_status,
                extraction_traces=extraction_traces,
                source_event_ids=tuple(dict.fromkeys(record.event_id for record, _ in ordered)),
                document_refs=refs,
                revision_count=len(refs),
                state_history=tuple(state_history),
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
        extractor: NewsEventExtractor | None = None,
    ) -> None:
        self._store = store
        self._extractor = extractor or ObviousNewsEventExtractor()

    def view_at(self, as_of: datetime) -> EventView:
        history = self._store.history_at(as_of)
        pairs: list[tuple[EventRecord, NewsDocument]] = []
        quarantined: list[ExtractionQuarantine] = []
        extraction_results: list[EventExtractionResult] = []
        for document in history:
            result = self._extractor.extract(document)
            extraction_results.append(result)
            if result.quarantine is not None:
                quarantined.append(result.quarantine)
                quarantined.extend(result.candidate_quarantines)
                continue
            quarantined.extend(result.candidate_quarantines)
            pairs.extend((event, document) for event in result.events)
        resolutions = {result.supersedes_document_version_id: result.review_status for result in extraction_results
                       if result.supersedes_document_version_id and result.review_status != "unreviewed"}
        quarantined = [item.model_copy(update={"review_status": resolutions[item.document_version_id]})
                       if item.document_version_id in resolutions else item for item in quarantined]
        return EventView(
            as_of=as_of,
            events=merge_event_records(pairs, as_of=as_of),
            quarantined=tuple(quarantined),
            extraction_results=tuple(extraction_results),
            visible_document_count=len(self._store.visible_at(as_of)),
        )

    def replay(self) -> tuple[EventView, ...]:
        """One view per instant at which the visible document set changes."""

        return tuple(self.view_at(instant) for instant in self._store.availability_timeline())
