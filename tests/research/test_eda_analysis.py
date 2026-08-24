"""Tests for price profiles, exogenous summaries, and lag semantics."""

from __future__ import annotations

from pathlib import Path

from app.research.data.alignment import align_loaded_series
from app.research.data.loader import load_series
from app.research.schemas.study import load_study_config
from app.research.tools.eda.exogenous import analyze_exogenous
from app.research.tools.eda.price import analyze_price
from app.research.tools.eda.relationships import analyze_relationships


def _aligned(synthetic_study: Path):
    config = load_study_config(synthetic_study)
    options = {"study_timezone": config.study.timezone}
    target = load_series(config.target, **options)
    features = [load_series(spec, **options) for spec in config.exogenous]
    return config, align_loaded_series(target, features, config).frame


def test_price_analysis_retains_negative_prices_and_spikes(synthetic_study: Path):
    config, frame = _aligned(synthetic_study)
    result = analyze_price(
        frame[config.target.name],
        unit=config.target.unit,
        frequency=config.study.frequency,
        max_lag=config.analysis.max_lag,
        spike_iqr_multiplier=config.analysis.spike_iqr_multiplier,
    )
    assert result["signs"]["negative_count"] == 1
    assert result["extremes"]["high_spike_count"] >= 1
    assert len(result["seasonality"]["hour_of_day"]) == 24
    assert len(result["seasonality"]["day_type"]) == 2
    assert result["rolling_statistics"]["one_day"]["window_intervals"] == 24
    assert len(result["autocorrelation"]) == config.analysis.max_lag


def test_relationship_positive_lag_means_feature_leads_target(synthetic_study: Path):
    _config, frame = _aligned(synthetic_study)
    result = analyze_relationships(
        frame,
        target_name="price",
        exogenous_names=["load"],
        max_lag=8,
        min_observations=12,
    )
    best = result["series"]["load"]["best_absolute_lag"]
    assert best["lag"] == 2
    assert best["correlation"] > 0.9
    assert "leads" in result["series"]["load"]["lag_semantics"]


def test_exogenous_analysis_reports_collinearity_without_dropping_columns(synthetic_study: Path):
    config, frame = _aligned(synthetic_study)
    names = [series.name for series in config.exogenous]
    result = analyze_exogenous(
        frame,
        names=names,
        units={series.name: series.unit for series in config.exogenous},
        outlier_iqr_multiplier=1.5,
    )
    assert set(result["series"]) == set(names)
    assert set(result["correlation_matrix"]) == set(names)
