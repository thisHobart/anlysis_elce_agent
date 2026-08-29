"""Input adapters for collected-news snapshots."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from app.research.news.contracts import CollectedNewsRecord


class CollectedNewsAdapterError(RuntimeError):
    """Raised when a collected-news input cannot be loaded safely."""


class ExternalNewsAdapterNotConfigured(CollectedNewsAdapterError):
    """Raised until the real external API contract is supplied."""


class CollectedNewsAdapter(Protocol):
    """Boundary implemented by fixture and future external-API adapters."""

    def load(self) -> Sequence[CollectedNewsRecord]: ...


@dataclass(frozen=True)
class ExternalNewsApiPlaceholder:
    """Fail-closed placeholder; it performs no network operation."""

    adapter_name: str = "external-news-api"

    def load(self) -> Sequence[CollectedNewsRecord]:
        raise ExternalNewsAdapterNotConfigured(
            f"{self.adapter_name} 仅为占位符；提供样例响应和字段映射后才能启用生产新闻输入"
        )


@dataclass(frozen=True)
class JsonlCollectedNewsAdapter:
    """Load deterministic collected-news fixtures from a local JSONL snapshot."""

    path: Path

    def load(self) -> Sequence[CollectedNewsRecord]:
        resolved = self.path.resolve()
        if not resolved.is_file():
            raise CollectedNewsAdapterError(f"新闻快照不存在：{resolved}")

        records: list[CollectedNewsRecord] = []
        with resolved.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                    if not isinstance(payload, dict):
                        raise TypeError("JSONL 行必须是对象")
                    records.append(CollectedNewsRecord.model_validate(payload))
                except (json.JSONDecodeError, TypeError, ValidationError) as exc:
                    raise CollectedNewsAdapterError(f"新闻快照第 {line_number} 行无效：{exc}") from exc
        return tuple(records)

