"""Deterministic test datasets for the Phase 1 research pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


@pytest.fixture
def synthetic_study(tmp_path: Path) -> Path:
    rng = np.random.default_rng(20260823)
    index = pd.date_range("2025-01-01 00:00:00", periods=24 * 30, freq="1h")
    hour = np.arange(len(index))
    leading_signal = np.sin(2 * np.pi * hour / 24) + rng.normal(0, 0.08, len(index))
    wind = np.cos(2 * np.pi * hour / (24 * 7)) + rng.normal(0, 0.12, len(index))
    target = 250 + 90 * pd.Series(leading_signal).shift(2).bfill().to_numpy() + rng.normal(0, 3, len(index))
    target[10] = -50
    target[100] = 900
    target[200] = np.nan
    wind[300:303] = np.nan

    pd.DataFrame({"datetime": index, "price": target}).to_csv(tmp_path / "price.csv", index=False)
    pd.DataFrame({"datetime": index, "load": leading_signal, "wind": wind}).to_csv(
        tmp_path / "features.csv", index=False
    )
    pd.DataFrame({"datetime": index, "temperature": 15 + 8 * np.sin(2 * np.pi * hour / (24 * 30))}).to_parquet(
        tmp_path / "weather.parquet", index=False
    )

    config = {
        "study": {
            "name": "synthetic_price_eda",
            "market": "synthetic_test_market",
            "timezone": "Asia/Shanghai",
            "frequency": "1h",
            "start_time": None,
            "end_time": None,
        },
        "target": {
            "name": "price",
            "path": "price.csv",
            "timestamp_column": "datetime",
            "value_column": "price",
            "unit": "test_unit",
            "availability_type": "observed_only",
        },
        "exogenous": [
            {
                "name": "load",
                "path": "features.csv",
                "timestamp_column": "datetime",
                "value_column": "load",
                "unit": "test_unit",
                "availability_type": "forecast",
            },
            {
                "name": "wind",
                "path": "features.csv",
                "timestamp_column": "datetime",
                "value_column": "wind",
                "unit": "test_unit",
                "availability_type": "forecast",
            },
            {
                "name": "temperature",
                "path": "weather.parquet",
                "timestamp_column": "datetime",
                "value_column": "temperature",
                "unit": "Celsius",
                "availability_type": "forecast",
            },
        ],
        "analysis": {
            "max_lag": 24,
            "min_relationship_observations": 12,
            "outlier_iqr_multiplier": 1.5,
            "spike_iqr_multiplier": 3.0,
            "output_directory": "artifacts",
        },
    }
    config_path = tmp_path / "study.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return config_path
