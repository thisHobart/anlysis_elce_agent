"""Document version history and point-in-time (`as_of`) reconstruction."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from app.research.news.contracts import DocumentRef, NewsDocument

VERSION_STORE_VERSION = "1.0.0"


class NewsVersionError(ValueError):
    """Raised when a version history cannot be trusted for point-in-time replay."""


def document_ref(document: NewsDocument) -> DocumentRef:
    return DocumentRef(
        document_id=document.document_id,
        document_version_id=document.document_version_id,
        version=document.version,
        source_name=document.source_name,
        content_hash=document.content_hash,
        available_at=document.available_at,
    )


@dataclass(frozen=True)
class DuplicateReport:
    """Exact re-collections that must not be counted twice."""

    duplicate_version_ids: tuple[str, ...] = ()
    duplicate_content_documents: tuple[tuple[str, str], ...] = ()

    @property
    def duplicate_count(self) -> int:
        return len(self.duplicate_version_ids)


class NewsVersionStore:
    """Hold every collected version and answer "what was visible at instant t?".

    A revision never overwrites its predecessor: both versions stay, and the view for a
    given instant picks the newest one that had already arrived by then. That is what
    makes a later correction unable to rewrite what an earlier analysis could see.
    """

    store_version = VERSION_STORE_VERSION

    def __init__(self, documents: Iterable[NewsDocument] = ()) -> None:
        self._versions: dict[str, dict[str, NewsDocument]] = {}
        self._duplicate_version_ids: list[str] = []
        self._duplicate_content: list[tuple[str, str]] = []
        for document in documents:
            self.add(document)

    def add(self, document: NewsDocument) -> bool:
        """Return True when the version is new; an identical re-collection is dropped."""

        versions = self._versions.setdefault(document.document_id, {})
        existing = versions.get(document.document_version_id)
        if existing is not None:
            if existing != document:
                raise NewsVersionError(
                    f"{document.document_version_id} 已存在且内容不同；版本身份必须随内容变化"
                )
            self._duplicate_version_ids.append(document.document_version_id)
            return False

        same_content = [
            other
            for other in versions.values()
            if other.content_hash == document.content_hash and other.version != document.version
        ]
        if same_content:
            self._duplicate_content.append((document.document_version_id, same_content[0].document_version_id))

        clashing = [other for other in versions.values() if other.version == document.version]
        if clashing:
            raise NewsVersionError(
                f"文档 {document.document_id} 的版本 {document.version} 出现两份不同内容；来源版本号必须唯一"
            )
        versions[document.document_version_id] = document
        return True

    def extend(self, documents: Iterable[NewsDocument]) -> None:
        for document in documents:
            self.add(document)

    @property
    def document_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._versions))

    @property
    def version_count(self) -> int:
        return sum(len(versions) for versions in self._versions.values())

    @property
    def duplicates(self) -> DuplicateReport:
        return DuplicateReport(
            duplicate_version_ids=tuple(self._duplicate_version_ids),
            duplicate_content_documents=tuple(self._duplicate_content),
        )

    def versions_of(self, document_id: str) -> tuple[NewsDocument, ...]:
        versions = self._versions.get(document_id, {})
        return tuple(sorted(versions.values(), key=lambda document: (document.version, document.available_at)))

    def visible_at(self, as_of: datetime) -> tuple[NewsDocument, ...]:
        """The one version of each document a reader could legitimately hold at `as_of`."""

        _require_utc(as_of)
        visible: list[NewsDocument] = []
        for document_id in self.document_ids:
            candidates = [
                document for document in self.versions_of(document_id) if document.available_at <= as_of
            ]
            if candidates:
                visible.append(max(candidates, key=lambda document: (document.version, document.available_at)))
        return tuple(sorted(visible, key=lambda document: (document.available_at, document.document_version_id)))

    def history_at(self, as_of: datetime) -> tuple[NewsDocument, ...]:
        """Every version that had arrived by `as_of`, superseded ones included.

        Event assembly needs the full history, not just the current view: a 12:00
        correction changes what the event says, but it must not change when the market
        first learned of it, and the superseded version stays part of the provenance.
        """

        _require_utc(as_of)
        return tuple(document for document in self.all_documents() if document.available_at <= as_of)

    def all_documents(self) -> tuple[NewsDocument, ...]:
        return tuple(
            sorted(
                (document for versions in self._versions.values() for document in versions.values()),
                key=lambda document: (document.available_at, document.document_version_id),
            )
        )

    def availability_timeline(self) -> tuple[datetime, ...]:
        """Every instant at which the visible set changes; useful for replaying history."""

        return tuple(sorted({document.available_at for document in self.all_documents()}))


def _require_utc(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise NewsVersionError("as_of 必须带时区")
    if value.utcoffset().total_seconds() != 0:
        raise NewsVersionError("as_of 必须归一化为 UTC")
