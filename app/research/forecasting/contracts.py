"""Versioned public contracts for the minimal Shandong forecast workflow."""

from __future__ import annotations

from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

FORECAST_ALGORITHM_ID = "shandong-rt-ctm-similar-day"
FORECAST_ALGORITHM_VERSION = "1.2.0+operational_backtest_origin"


class ForecastNewsFeatureSource(BaseModel):
    """Audited P2→P3 numeric feature hand-off frozen with a forecast plan."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_run_id: str = Field(min_length=1)
    source_p1_run_id: str | None = None
    source_path: Path
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    selected_columns: tuple[str, ...] = ()
    excluded_columns: dict[str, str] = Field(default_factory=dict)
    selection_cutoff: datetime
    availability_type: Literal["known_at_timestamp"] = "known_at_timestamp"
    selection_reason: str = Field(min_length=1)


class CTMTrainingConfig(BaseModel):
    """Frozen reference parameters approved with every forecast plan."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    lookback_steps: int = 96
    horizon_steps: int = 96
    internal_ticks: int = 6
    memory_length: int = 6
    hidden_size: int = 80
    sync_pairs: int = 48
    dropout: float = 0.10
    max_epochs: int = 55
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    gradient_clip: float = 1.0
    early_stopping_patience: int = 8
    minimum_training_rows: int = 96 * 21
    training_stride: int = 2
    maximum_training_samples: int = 4500
    seed: int = 42
    price_floor: float = -120.0
    price_cap: float = 1500.0
    similar_day_top_k: int = 7
    similar_day_windows: tuple[int, ...] = (30, 90, 240)


class ForecastSnapshotSpec(BaseModel):
    """One immutable point-in-time model input attached to a plan."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["backtest", "future"]
    as_of: datetime
    target_start: datetime
    target_end: datetime
    path: Path
    fingerprint: str = Field(pattern=r"^[a-f0-9]{12}$")
    truth_path: Path | None = None
    truth_fingerprint: str | None = Field(default=None, pattern=r"^[a-f0-9]{12}$")
    news_feature_columns: tuple[str, ...] = ()
    expected_baseline_common_observations: int = Field(default=0, ge=0, le=96)


class ForecastPlan(BaseModel):
    """Fixed-scope plan shown to the user before any model training starts."""

    model_config = ConfigDict(extra="forbid")

    plan_kind: Literal["forecast"] = "forecast"
    plan_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    question: str = Field(min_length=1)
    region_id: Literal["shandong"] = "shandong"
    region_label: str = "山东"
    market: str = "山东电网"
    target_name: Literal["rt_price"] = "rt_price"
    timezone: Literal["Asia/Shanghai"] = "Asia/Shanghai"
    frequency: Literal["15min"] = "15min"
    forecast_start: datetime
    forecast_end: datetime
    horizon_steps: Literal[96] = 96
    algorithm_id: Literal["shandong-rt-ctm-similar-day"] = FORECAST_ALGORITHM_ID
    algorithm_version: str = FORECAST_ALGORITHM_VERSION
    training: CTMTrainingConfig = Field(default_factory=CTMTrainingConfig)
    snapshots: list[ForecastSnapshotSpec] = Field(min_length=4, max_length=4)
    data_fingerprint: str = Field(pattern=r"^[a-f0-9]{12}$")
    source_data_fingerprint: str | None = Field(default=None, pattern=r"^[a-f0-9]{12}$")
    preflight_warnings: list[str] = Field(default_factory=list)
    news_feature_sources: tuple[ForecastNewsFeatureSource, ...] = ()
    read_only: Literal[True] = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def validate_shape(self) -> ForecastPlan:
        folds = [item for item in self.snapshots if item.role == "backtest"]
        future = [item for item in self.snapshots if item.role == "future"]
        if len(folds) != 3 or len(future) != 1:
            raise ValueError("forecast plan requires exactly three backtests and one future snapshot")
        if any(item.truth_path is None or item.truth_fingerprint is None for item in folds):
            raise ValueError("every backtest snapshot requires independently hashed truth")
        if future[0].truth_path is not None or future[0].truth_fingerprint is not None:
            raise ValueError("future snapshot cannot contain future truth")
        anchors = sorted(item.as_of for item in folds)
        if any((right - left).days < 7 for left, right in pairwise(anchors)):
            raise ValueError("backtest anchors must be at least seven days apart")
        if self.forecast_end <= self.forecast_start:
            raise ValueError("forecast_end must be after forecast_start")
        return self


class ForecastMetricSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observations: int = Field(ge=0)
    mae: float
    rmse: float
    bias: float


class ForecastFoldResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    anchor: datetime
    target_start: datetime
    target_end: datetime
    model: ForecastMetricSet
    persistence: ForecastMetricSet
    day_naive: ForecastMetricSet
    week_naive: ForecastMetricSet
    prediction_path: Path
    snapshot_fingerprint: str
    common_observations: int = Field(default=0, ge=0)
    news_feature_columns: tuple[str, ...] = ()


class ForecastRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    status: Literal["completed"] = "completed"
    plan: ForecastPlan
    folds: list[ForecastFoldResult] = Field(min_length=3, max_length=3)
    aggregate: dict[str, ForecastMetricSet]
    diagnostics: dict[str, object] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    prediction_path: Path
    backtest_path: Path
    metrics_path: Path
    report_path: Path
    artifact_directory: Path
    figure_paths: dict[str, Path]
    output_hash: str = Field(pattern=r"^[a-f0-9]{64}$")

    @property
    def baseline_verified(self) -> bool:
        model = self.aggregate["model"].mae
        comparable = all(item.common_observations >= 72 for item in self.folds)
        return comparable and model is not None and model < min(
            self.aggregate["day_naive"].mae,
            self.aggregate["week_naive"].mae,
        )

    @property
    def evaluation_status(self) -> Literal["verified", "not_verified", "not_evaluable"]:
        if any(item.common_observations < 72 for item in self.folds):
            return "not_evaluable"
        return "verified" if self.baseline_verified else "not_verified"
