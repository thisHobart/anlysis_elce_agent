"""Freezing a dataset, reading it back, and what happens when the frozen copy is gone."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from app.research.application.planning import (
    freeze_research_data,
    prepare_research_data,
    restore_prepared_data,
)
from app.research.data.snapshot import input_file_manifest, study_fingerprint
from app.research.data.sources.materialize import (
    MANIFEST_FILENAME,
    materialize_dataset,
    prune_snapshots,
    read_snapshot,
    snapshot_path,
    snapshot_root,
)
from app.research.schemas.study import load_study_config


def _frozen(study: Path):
    """Load a study, freeze it, and hand back everything the assertions need."""

    config = load_study_config(study)
    prepared = prepare_research_data(config)
    manifest = input_file_manifest(config)
    fingerprint = study_fingerprint(config, manifest)
    snapshot = freeze_research_data(prepared, fingerprint=fingerprint, input_manifest=manifest)
    assert snapshot is not None
    return config, prepared, fingerprint, snapshot


def test_a_frozen_dataset_reads_back_exactly_as_it_was_read(synthetic_study: Path):
    config, prepared, fingerprint, _snapshot = _frozen(synthetic_study)

    restored = restore_prepared_data(fingerprint, config)

    assert restored is not None
    rebuilt, _stored = restored
    pd.testing.assert_frame_equal(rebuilt.aligned.frame, prepared.aligned.frame)
    assert rebuilt.quality == prepared.quality
    assert rebuilt.target.raw_rows == prepared.target.raw_rows
    assert rebuilt.target.duplicate_timestamp_keys == prepared.target.duplicate_timestamp_keys
    assert [item.spec.name for item in rebuilt.exogenous] == [item.spec.name for item in prepared.exogenous]


def test_the_frozen_copy_ignores_a_source_that_moves_afterwards(synthetic_study: Path):
    config, prepared, fingerprint, _snapshot = _frozen(synthetic_study)
    frame = pd.read_csv(config.target.path)
    frame.loc[0, config.target.value_column] = 12345.0
    frame.to_csv(config.target.path, index=False)

    restored = restore_prepared_data(fingerprint, config)

    assert restored is not None
    rebuilt, _stored = restored
    pd.testing.assert_frame_equal(rebuilt.aligned.frame, prepared.aligned.frame)


def test_freezing_the_same_data_twice_keeps_the_first_fetch_time(synthetic_study: Path):
    config = load_study_config(synthetic_study)
    prepared = prepare_research_data(config)
    manifest = input_file_manifest(config)
    fingerprint = study_fingerprint(config, manifest)
    earlier = datetime(2026, 3, 1, 9, 30).astimezone()

    first = materialize_dataset(
        config=config,
        target=prepared.target,
        exogenous=prepared.exogenous,
        quality=prepared.quality,
        fingerprint=fingerprint,
        input_manifest=manifest,
        fetched_at=earlier,
    )
    second = materialize_dataset(
        config=config,
        target=prepared.target,
        exogenous=prepared.exogenous,
        quality=prepared.quality,
        fingerprint=fingerprint,
        input_manifest=manifest,
        fetched_at=earlier + timedelta(days=2),
    )

    assert second.created_at == first.created_at
    assert second.summary.fetched_at_text == first.summary.fetched_at_text
    assert "2026年3月1日 09:30" == first.summary.fetched_at_text


def test_the_summary_written_with_the_snapshot_speaks_the_panel_language(synthetic_study: Path):
    _config, _prepared, fingerprint, snapshot = _frozen(synthetic_study)

    stored = read_snapshot(fingerprint)

    assert stored is not None
    assert stored.summary == snapshot.summary
    assert stored.summary.granularity_text == "每小时一个点"
    # 「气温」 comes from the word list; the two columns it does not name keep their own.
    assert [item.display for item in stored.summary.variables] == ["load", "wind", "气温"]
    assert [item.resolved for item in stored.summary.variables] == [False, False, True]
    assert stored.summary.start_date.endswith("日")
    assert stored.input_manifest == snapshot.input_manifest


@pytest.mark.parametrize("damage", ["missing", "truncated", "future_format", "renamed_series"])
def test_an_unusable_snapshot_falls_back_to_reading_the_sources(synthetic_study: Path, damage: str):
    config, _prepared, fingerprint, _snapshot = _frozen(synthetic_study)
    directory = snapshot_path(fingerprint)
    manifest_path = directory / MANIFEST_FILENAME

    if damage == "missing":
        (directory / "series-000.parquet").unlink()
    elif damage == "truncated":
        manifest_path.write_text("{", encoding="utf-8")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if damage == "future_format":
            manifest["format_version"] = 9999
        else:
            manifest["series"][1]["name"] = "a_variable_this_study_never_had"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    assert restore_prepared_data(fingerprint, config) is None


def test_pruning_keeps_the_newest_snapshots_and_never_the_one_just_written(synthetic_study: Path):
    _config, _prepared, fingerprint, _snapshot = _frozen(synthetic_study)
    root = snapshot_root()
    for index in range(5):
        older = root / f"{index:012d}"
        older.mkdir(parents=True)
        (older / MANIFEST_FILENAME).write_text("{}", encoding="utf-8")

    removed = prune_snapshots(retain=3, keep=(fingerprint,))

    survivors = {path.name for path in root.iterdir() if path.is_dir()}
    assert removed == 3
    assert fingerprint in survivors
    assert len(survivors) == 3
