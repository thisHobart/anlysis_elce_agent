"""Deterministic normalization for already-collected news records."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from app.research.news.contracts import CollectedNewsRecord, NewsDocument

NORMALIZER_VERSION = "1.0.0"


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_news_text(value: str) -> str:
    """Normalize Unicode, line endings and insignificant whitespace for stable identity."""

    normalized = unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n")
    lines = [" ".join(line.split()) for line in normalized.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def _utc(value: datetime | None) -> datetime | None:
    return value.astimezone(UTC) if value is not None else None


class NewsNormalizer:
    """Convert provider-neutral collection records to immutable canonical documents."""

    normalizer_version = NORMALIZER_VERSION

    def normalize(self, record: CollectedNewsRecord) -> NewsDocument:
        title = normalize_news_text(record.title)
        body = normalize_news_text(record.body)
        published_at = _utc(record.published_at)
        first_seen_at = _utc(record.collected_at)
        updated_at = _utc(record.updated_at)
        assert published_at is not None and first_seen_at is not None

        content_hash = _canonical_hash({"body": body, "title": title})
        document_id_hash = _canonical_hash(
            {
                "source_document_id": record.source_document_id,
                "source_name": record.source_name.casefold(),
            }
        )
        document_id = f"news_{document_id_hash[:24]}"
        document_version_hash = _canonical_hash(
            {
                "content_hash": content_hash,
                "document_id": document_id,
                "first_seen_at": first_seen_at.isoformat(),
                "published_at": published_at.isoformat(),
                "updated_at": updated_at.isoformat() if updated_at is not None else None,
                "version": record.version,
            }
        )

        return NewsDocument(
            normalizer_version=self.normalizer_version,
            document_id=document_id,
            document_version_id=f"newsv_{document_version_hash[:24]}",
            source_name=record.source_name,
            source_document_id=record.source_document_id,
            source_ref=record.source_ref,
            version=record.version,
            title=title,
            body=body,
            published_at=published_at,
            first_seen_at=first_seen_at,
            updated_at=updated_at,
            available_at=max(published_at, first_seen_at),
            language=record.language,
            market_tags=tuple(sorted(set(record.market_tags))),
            content_hash=content_hash,
            raw_metadata=dict(record.metadata),
        )

    def normalize_many(self, records: Iterable[CollectedNewsRecord]) -> tuple[NewsDocument, ...]:
        return tuple(self.normalize(record) for record in records)
