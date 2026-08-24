"""Align target and exogenous series to one explicit, timezone-aware grid."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.research.data.loader import LoadedSeries, ResearchDataError
from app.research.schemas.study import StudyConfig


@dataclass(frozen=True)
class AlignmentResult:
    """Canonical analysis frame and coverage evidence."""

    frame: pd.DataFrame
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    coverage: dict[str, float]


def _aggregate(loaded: LoadedSeries, frequency: str) -> pd.Series:
    values = loaded.frame.set_index("timestamp")["value"]
    resampler = values.resample(frequency)
    if loaded.spec.aggregation == "sum":
        return resampler.sum(min_count=1)
    return getattr(resampler, loaded.spec.aggregation)()


def align_loaded_series(target: LoadedSeries, exogenous: list[LoadedSeries], config: StudyConfig) -> AlignmentResult:
    """Use the target window as the population and left-align all explanatory series."""

    target_aggregated = _aggregate(target, config.study.frequency)
    if target_aggregated.empty:
        raise ResearchDataError("target series is empty after frequency aggregation")

    start = target_aggregated.index.min()
    end = target_aggregated.index.max()
    grid = pd.date_range(start=start, end=end, freq=config.study.frequency, tz=config.study.timezone)
    if grid.empty:
        raise ResearchDataError("configured study window produced an empty time grid")

    frame = pd.DataFrame(index=grid)
    frame.index.name = "timestamp"
    frame[target.spec.name] = target_aggregated.reindex(grid)
    for loaded in exogenous:
        frame[loaded.spec.name] = _aggregate(loaded, config.study.frequency).reindex(grid)

    coverage = {name: round(float(frame[name].notna().mean()), 6) for name in frame.columns}
    return AlignmentResult(frame=frame, start_time=start, end_time=end, coverage=coverage)
