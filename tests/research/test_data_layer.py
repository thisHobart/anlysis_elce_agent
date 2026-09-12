"""Tests for configuration, loading, time alignment, and quality evidence."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.research.application.planning import prepare_research_data
from app.research.data.alignment import align_loaded_series
from app.research.data.loader import load_series
from app.research.data.quality import build_quality_report
from app.research.data.snapshot import input_file_manifest, study_fingerprint
from app.research.schemas.study import SeriesSpec, StudyConfig, StudyDefinition, load_study_config


def test_config_resolves_csv_and_parquet_paths(synthetic_study: Path):
    config = load_study_config(synthetic_study)
    assert config.target.path.is_absolute()
    assert config.target.path.name == "price.csv"
    assert config.exogenous[-1].path.suffix == ".parquet"
    assert config.study.frequency == "1h"


def test_loader_preserves_bad_row_evidence(tmp_path: Path):
    path = tmp_path / "dirty.csv"
    pd.DataFrame(
        {
            "when": ["2025-01-01 00:00", "bad", "2025-01-01 01:00", "2025-01-01 01:00"],
            "value": [1, 2, "not-a-number", 4],
        }
    ).to_csv(path, index=False)
    loaded = load_series(
        SeriesSpec(name="dirty", path=path, timestamp_column="when", value_column="value"),
        study_timezone="Asia/Shanghai",
    )
    assert loaded.invalid_timestamp_rows == 1
    assert loaded.non_numeric_rows == 1
    assert loaded.duplicate_timestamp_rows == 2
    assert loaded.duplicate_timestamp_keys == 1
    assert str(loaded.frame["timestamp"].dt.tz) == "Asia/Shanghai"


def test_alignment_and_quality_use_target_grid(synthetic_study: Path):
    config = load_study_config(synthetic_study)
    options = {"study_timezone": config.study.timezone}
    target = load_series(config.target, **options)
    features = [load_series(spec, **options) for spec in config.exogenous]
    aligned = align_loaded_series(target, features, config)
    quality = build_quality_report(target, features, aligned, config)
    assert aligned.frame.shape == (24 * 30, 4)
    assert aligned.frame.index.tz is not None
    assert quality.series["price"].missing_interval_count == 1
    assert quality.series["wind"].missing_interval_count == 3
    assert quality.usable_for_eda is True
    assert any(issue.code == "forecast_vintage_unknown" for issue in quality.issues)


def test_input_manifest_hashes_each_shared_file_once(synthetic_study: Path):
    manifest = input_file_manifest(load_study_config(synthetic_study))
    assert len(manifest) == 3
    shared = next(item for item in manifest if item["path"].endswith("features.csv"))
    assert shared["series"] == ["load", "wind"]
    assert len(shared["sha256"]) == 64


def test_data_preparation_reads_each_shared_file_once(monkeypatch, synthetic_study: Path):
    from app.research.application import planning

    reads = []
    original = planning._read_frame

    def counted(spec):
        reads.append(spec.path)
        return original(spec)

    monkeypatch.setattr(planning, "_read_frame", counted)

    prepared = prepare_research_data(load_study_config(synthetic_study))

    assert len(prepared.exogenous) == 3
    assert len(reads) == len(set(reads)) == 3


def test_study_fingerprint_ignores_physical_cache_directory(synthetic_study: Path, tmp_path: Path):
    config = load_study_config(synthetic_study)
    copied_paths = {}
    for spec in (config.target, *config.exogenous):
        copied = tmp_path / spec.path.name
        if spec.path not in copied_paths:
            copied.write_bytes(spec.path.read_bytes())
            copied_paths[spec.path] = copied
    moved = config.model_copy(
        update={
            "target": config.target.model_copy(update={"path": copied_paths[config.target.path]}),
            "exogenous": [
                spec.model_copy(update={"path": copied_paths[spec.path]})
                for spec in config.exogenous
            ],
        }
    )

    original = study_fingerprint(config, input_file_manifest(config))
    relocated = study_fingerprint(moved, input_file_manifest(moved))

    assert relocated == original


def test_sum_aggregation_keeps_empty_intervals_missing(tmp_path: Path):
    path = tmp_path / "sparse.csv"
    pd.DataFrame(
        {
            "datetime": ["2025-01-01 00:00", "2025-01-01 02:00"],
            "value": [1.0, 2.0],
        }
    ).to_csv(path, index=False)
    spec = SeriesSpec(
        name="price",
        path=path,
        timestamp_column="datetime",
        value_column="value",
        aggregation="sum",
    )
    config = StudyConfig(study=StudyDefinition(name="sum-test", market="test", frequency="1h"), target=spec)
    loaded = load_series(spec, study_timezone=config.study.timezone)
    aligned = align_loaded_series(loaded, [], config)
    assert aligned.frame["price"].tolist()[0] == 1.0
    assert pd.isna(aligned.frame["price"].tolist()[1])
    assert aligned.frame["price"].tolist()[2] == 2.0


def test_very_sparse_target_is_not_marked_usable(tmp_path: Path):
    path = tmp_path / "sparse-target.csv"
    pd.DataFrame(
        {
            "datetime": pd.date_range("2025-01-01", periods=5, freq="24h"),
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    ).to_csv(path, index=False)
    spec = SeriesSpec(name="price", path=path, timestamp_column="datetime", value_column="value")
    config = StudyConfig(
        study=StudyDefinition(name="sparse-test", market="test", frequency="1h"),
        target=spec,
        analysis={"min_relationship_observations": 3, "min_target_coverage_rate": 0.5},
    )
    loaded = load_series(spec, study_timezone=config.study.timezone)
    aligned = align_loaded_series(loaded, [], config)
    quality = build_quality_report(loaded, [], aligned, config)

    assert quality.series["price"].aligned_non_null_rows == 5
    assert quality.series["price"].aligned_coverage_rate < 0.1
    assert quality.usable_for_eda is False


def test_factor_without_rows_in_selected_window_is_reported_not_fatal(tmp_path: Path):
    target_path = tmp_path / "target.parquet"
    factor_path = tmp_path / "factor.parquet"
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=96, freq="15min"),
            "price": range(96),
        }
    ).to_parquet(target_path, index=False)
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-04-01", periods=4, freq="15min"),
            "solar": range(4),
        }
    ).to_parquet(factor_path, index=False)
    config = StudyConfig(
        study=StudyDefinition(
            name="window-test",
            market="test",
            frequency="15min",
            start_time="2026-01-01",
            end_time="2026-01-01 23:59:59",
        ),
        target=SeriesSpec(
            name="price",
            path=target_path,
            timestamp_column="timestamp",
            value_column="price",
        ),
        exogenous=[
            SeriesSpec(
                name="solar",
                path=factor_path,
                timestamp_column="timestamp",
                value_column="solar",
            )
        ],
    )

    prepared = prepare_research_data(config)

    assert prepared.aligned.frame["solar"].isna().all()
    assert any(
        issue.code == "series_has_no_values" and issue.series == "solar"
        for issue in prepared.quality.issues
    )
