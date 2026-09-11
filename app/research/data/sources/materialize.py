"""Freeze one dataset read so an approved plan keeps running on the data it was approved for.

Without this, execution re-reads the sources and a change made between approval
and execution either moves the conclusions or aborts the run. Neither is what the
analyst agreed to: they approved a plan *over a particular dataset*, so that
dataset is written down once and every later step reads the copy.

A snapshot is addressed by the study fingerprint, which already covers the
configuration and the content of every source, so writing the same dataset twice
is a no-op and two different datasets can never share a directory.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from app.research.data.loader import LoadedSeries
from app.research.data.sources.summary import (
    DataSummary,
    build_summary,
    parse_summary,
    summary_payload,
)
from app.research.schemas.results import DataQualityReport
from app.research.schemas.study import SeriesSpec, StudyConfig
from app.runtime_paths import application_data_directory

# Bumping this abandons older snapshots instead of misreading them; execution then
# falls back to reading the sources, which is the behaviour that existed before.
SNAPSHOT_FORMAT_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
DEFAULT_RETAINED_SNAPSHOTS = 8

SERIES_COUNTER_FIELDS = (
    "raw_rows",
    "invalid_timestamp_rows",
    "non_numeric_rows",
    "duplicate_timestamp_rows",
    "duplicate_timestamp_keys",
)


@dataclass(frozen=True)
class MaterializedSnapshot:
    """One frozen dataset read, plus everything needed to describe where it came from."""

    fingerprint: str
    path: Path
    created_at: str
    row_count: int
    input_manifest: list[dict[str, Any]]
    summary: DataSummary


@dataclass(frozen=True)
class RestoredDataset:
    """Series read back from a snapshot, shaped exactly as the loader would return them."""

    target: LoadedSeries
    exogenous: list[LoadedSeries]
    snapshot: MaterializedSnapshot


def snapshot_root() -> Path:
    """Return the internal directory holding frozen datasets.

    This lives beside the other runtime state rather than in the user's research
    output: a snapshot is machinery, not a deliverable, and deleting it only costs
    a re-read.
    """

    return application_data_directory() / "research" / "snapshots"


def snapshot_path(fingerprint: str) -> Path:
    return snapshot_root() / str(fingerprint)


def materialize_dataset(
    *,
    config: StudyConfig,
    target: LoadedSeries,
    exogenous: list[LoadedSeries],
    quality: DataQualityReport,
    fingerprint: str,
    input_manifest: list[dict[str, Any]],
    fetched_at: datetime | None = None,
    retain: int = DEFAULT_RETAINED_SNAPSHOTS,
) -> MaterializedSnapshot:
    """Write this dataset down once and return how the analyst should read it.

    An existing snapshot for the same fingerprint is returned untouched, so the
    recorded fetch time keeps pointing at when the data was actually read rather
    than at the most recent time somebody asked a question about it.
    """

    existing = read_snapshot(fingerprint)
    if existing is not None:
        return existing

    moment = (fetched_at or datetime.now(UTC)).astimezone()
    summary = build_summary(
        market=config.study.market,
        target_name=config.target.name,
        exogenous_names=[spec.name for spec in config.exogenous],
        start_time=quality.alignment.start_time,
        end_time=quality.alignment.end_time,
        frequency=config.study.frequency,
        missing_points=max(0, quality.alignment.expected_rows - quality.alignment.complete_case_rows),
        fetched_at=moment,
    )
    series = [("target", target), *(("exogenous", item) for item in exogenous)]
    entries: list[dict[str, Any]] = []
    root = snapshot_root()
    root.mkdir(parents=True, exist_ok=True)
    staging = root / f".writing-{fingerprint}-{uuid4().hex[:8]}"
    staging.mkdir(parents=True)
    try:
        for index, (role, loaded) in enumerate(series):
            filename = f"series-{index:03d}.parquet"
            loaded.frame.to_parquet(staging / filename, index=False)
            entries.append(
                {
                    "role": role,
                    "name": loaded.spec.name,
                    "file": filename,
                    "rows": len(loaded.frame),
                    **{field: int(getattr(loaded, field)) for field in SERIES_COUNTER_FIELDS},
                }
            )
        manifest = {
            "format_version": SNAPSHOT_FORMAT_VERSION,
            "fingerprint": fingerprint,
            "created_at": moment.isoformat(),
            "row_count": sum(int(entry["rows"]) for entry in entries),
            "input_manifest": input_manifest,
            "summary": summary_payload(summary),
            "series": entries,
        }
        (staging / MANIFEST_FILENAME).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        destination = snapshot_path(fingerprint)
        try:
            os.replace(staging, destination)
        except OSError:
            # Another proposal froze the same dataset first. Content is addressed by
            # fingerprint, so whatever is already there says the same thing.
            shutil.rmtree(staging, ignore_errors=True)
            restored = read_snapshot(fingerprint)
            if restored is None:
                raise
            return restored
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    prune_snapshots(retain=retain, keep=(fingerprint,))
    return MaterializedSnapshot(
        fingerprint=fingerprint,
        path=destination,
        created_at=manifest["created_at"],
        row_count=int(manifest["row_count"]),
        input_manifest=list(input_manifest),
        summary=summary,
    )


def read_snapshot(fingerprint: str) -> MaterializedSnapshot | None:
    """Describe a stored snapshot without reading the data itself."""

    manifest = _read_manifest(fingerprint)
    if manifest is None:
        return None
    return MaterializedSnapshot(
        fingerprint=str(manifest.get("fingerprint", fingerprint)),
        path=snapshot_path(fingerprint),
        created_at=str(manifest.get("created_at", "")),
        row_count=int(manifest.get("row_count", 0)),
        input_manifest=list(manifest.get("input_manifest") or []),
        summary=parse_summary(manifest.get("summary")),
    )


def restore_dataset(fingerprint: str | None, *, config: StudyConfig) -> RestoredDataset | None:
    """Read a frozen dataset back, or return None so the caller reads the sources.

    Every failure path returns None rather than raising: a missing or unreadable
    snapshot is a cache miss, and falling back to the sources is exactly the
    behaviour that existed before snapshots.
    """

    if not fingerprint:
        return None
    manifest = _read_manifest(fingerprint)
    if manifest is None:
        return None
    entries = manifest.get("series")
    if not isinstance(entries, list) or not entries:
        return None
    specs = _specs_by_role(config)
    if [str(entry.get("name")) for entry in entries] != [spec.name for spec in specs]:
        return None
    directory = snapshot_path(fingerprint)
    restored: list[LoadedSeries] = []
    for entry, spec in zip(entries, specs, strict=True):
        loaded = _restore_series(directory, entry, spec)
        if loaded is None:
            return None
        restored.append(loaded)
    snapshot = read_snapshot(fingerprint)
    if snapshot is None:
        return None
    return RestoredDataset(target=restored[0], exogenous=restored[1:], snapshot=snapshot)


def prune_snapshots(*, retain: int = DEFAULT_RETAINED_SNAPSHOTS, keep: tuple[str, ...] = ()) -> int:
    """Keep the newest snapshots and drop the rest; return how many were removed.

    A pruned snapshot an approved plan still points at costs a re-read, and the
    fingerprint check then catches a source that moved in the meantime.
    """

    root = snapshot_root()
    if not root.is_dir():
        return 0
    protected = set(keep)
    candidates: list[tuple[float, Path]] = []
    for path in root.iterdir():
        if not path.is_dir() or path.name.startswith(".writing-"):
            continue
        if path.name in protected:
            continue
        try:
            created = (path / MANIFEST_FILENAME).stat().st_mtime
        except OSError:
            created = 0.0
        candidates.append((created, path))
    surplus = len(candidates) + len(protected) - max(0, int(retain))
    if surplus <= 0:
        return 0
    removed = 0
    for _created, path in sorted(candidates)[:surplus]:
        shutil.rmtree(path, ignore_errors=True)
        removed += 1
    return removed


def _specs_by_role(config: StudyConfig) -> list[SeriesSpec]:
    return [config.target, *config.exogenous]


def _read_manifest(fingerprint: str) -> dict[str, Any] | None:
    path = snapshot_path(fingerprint) / MANIFEST_FILENAME
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(manifest, dict):
        return None
    if int(manifest.get("format_version", 0)) != SNAPSHOT_FORMAT_VERSION:
        return None
    return manifest


def _restore_series(directory: Path, entry: dict[str, Any], spec: SeriesSpec) -> LoadedSeries | None:
    filename = str(entry.get("file", ""))
    if not filename:
        return None
    try:
        frame = pd.read_parquet(directory / filename)
    except (OSError, ValueError):
        return None
    if "timestamp" not in frame.columns or "value" not in frame.columns:
        return None
    return LoadedSeries(
        spec=spec,
        frame=frame,
        **{field: int(entry.get(field, 0)) for field in SERIES_COUNTER_FIELDS},
    )
