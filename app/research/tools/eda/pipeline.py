"""Run the complete deterministic Phase 1 EDA pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.research.data.alignment import align_loaded_series
from app.research.data.loader import ResearchDataError, load_series
from app.research.data.quality import build_quality_report
from app.research.data.snapshot import input_file_manifest, study_fingerprint
from app.research.reporting.artifacts import write_research_package
from app.research.schemas.results import PipelineRunResult
from app.research.schemas.study import StudyConfig, load_study_config
from app.research.tools.eda.exogenous import analyze_exogenous
from app.research.tools.eda.price import analyze_price
from app.research.tools.eda.relationships import analyze_relationships
from app.runtime_paths import source_worktree


def _load_all_series(config: StudyConfig):
    load_options = {
        "study_timezone": config.study.timezone,
        "start_time": config.study.start_time,
        "end_time": config.study.end_time,
    }
    target = load_series(config.target, **load_options)
    exogenous = [load_series(series, **load_options) for series in config.exogenous]
    return target, exogenous


def _build_summary(config: StudyConfig, aligned_frame) -> dict[str, Any]:
    exogenous_names = [series.name for series in config.exogenous]
    return {
        "study": {
            "name": config.study.name,
            "market": config.study.market,
            "timezone": config.study.timezone,
            "frequency": config.study.frequency,
            "target": config.target.name,
            "exogenous": exogenous_names,
        },
        "methodology": {
            "missing_value_policy": "pairwise complete for relationships; no implicit imputation",
            "outlier_policy": "retain observations and report robust IQR flags",
            "lag_semantics": "positive lag compares feature[t-lag] with target[t]",
            "causal_claims": False,
        },
        "price": analyze_price(
            aligned_frame[config.target.name],
            unit=config.target.unit,
            frequency=config.study.frequency,
            max_lag=config.analysis.max_lag,
            spike_iqr_multiplier=config.analysis.spike_iqr_multiplier,
        ),
        "exogenous": analyze_exogenous(
            aligned_frame,
            names=exogenous_names,
            units={series.name: series.unit for series in config.exogenous},
            outlier_iqr_multiplier=config.analysis.outlier_iqr_multiplier,
        ),
        "relationships": analyze_relationships(
            aligned_frame,
            target_name=config.target.name,
            exogenous_names=exogenous_names,
            max_lag=config.analysis.max_lag,
            min_observations=config.analysis.min_relationship_observations,
        ),
    }


def run_eda_pipeline(
    config: StudyConfig | str | Path,
    *,
    output_directory: str | Path | None = None,
    run_id: str | None = None,
) -> PipelineRunResult:
    """Execute ingestion, alignment, checks, EDA, and artifact generation."""

    resolved_config = load_study_config(config) if isinstance(config, (str, Path)) else config
    if output_directory is not None:
        output_path = Path(output_directory).resolve()
        resolved_config = resolved_config.model_copy(
            update={
                "analysis": resolved_config.analysis.model_copy(update={"output_directory": output_path}),
            }
        )

    target, exogenous = _load_all_series(resolved_config)
    aligned = align_loaded_series(target, exogenous, resolved_config)
    quality = build_quality_report(target, exogenous, aligned, resolved_config)
    if not quality.usable_for_eda:
        raise ResearchDataError("target data does not meet the minimum observation requirement for EDA")

    summary = _build_summary(resolved_config, aligned.frame)
    inputs = input_file_manifest(resolved_config)
    fingerprint = study_fingerprint(resolved_config, inputs)
    bundle = write_research_package(
        config=resolved_config,
        quality=quality,
        summary=summary,
        aligned_frame=aligned.frame,
        input_manifest=inputs,
        fingerprint=fingerprint,
        worktree=source_worktree(),
        run_id=run_id,
    )
    return PipelineRunResult(
        run_id=bundle.run_id,
        artifact_directory=bundle.directory,
        report_path=bundle.report_path,
        aligned_rows=len(aligned.frame),
        quality_report=quality,
        eda_summary=summary,
    )
