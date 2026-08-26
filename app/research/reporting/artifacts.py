"""Write all EDA evidence into one checksummed, reproducible research package."""

from __future__ import annotations

import json
import platform
import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy
import pandas
import pyarrow
import scipy
import statsmodels

from app.research.data.snapshot import sha256_file
from app.research.reporting.charts import render_eda_charts
from app.research.reporting.eda_report import build_eda_report
from app.research.schemas.results import DataQualityReport
from app.research.schemas.study import StudyConfig

SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


@dataclass(frozen=True)
class ArtifactBundle:
    """Paths produced by one completed research-package write."""

    run_id: str
    directory: Path
    report_path: Path
    manifest_path: Path
    figure_paths: dict[str, Path]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _git_state(worktree: Path | None) -> tuple[str | None, bool | None]:
    if worktree is None:
        return None, None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=worktree,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=worktree,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def _write_figures(directory: Path, figures: dict[str, Any]) -> dict[str, Path]:
    """Persist every inline SVG figure so the package stays readable without the HTML."""

    if not figures:
        return {}
    directory.mkdir(parents=True, exist_ok=False)
    paths: dict[str, Path] = {}
    for key, markup in figures.items():
        path = directory / f"{key}.svg"
        path.write_text(markup, encoding="utf-8")
        paths[key] = path
    return paths


def _output_manifest(directory: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(directory).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    ]


def write_research_package(
    *,
    config: StudyConfig,
    quality: DataQualityReport,
    summary: dict[str, Any],
    aligned_frame: pandas.DataFrame,
    input_manifest: list[dict[str, Any]],
    fingerprint: str,
    worktree: Path | None,
    run_id: str | None = None,
) -> ArtifactBundle:
    """Persist JSON evidence, aligned data, charts, narrative, and hashes."""

    created_at = datetime.now(UTC)
    resolved_run_id = run_id or f"{created_at.strftime('%Y%m%dT%H%M%S%fZ')}-{fingerprint}"
    if not SAFE_RUN_ID.fullmatch(resolved_run_id):
        raise ValueError("run_id must contain only letters, numbers, '.', '_' or '-'")

    directory = (config.analysis.output_directory / resolved_run_id).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    figure_paths = render_eda_charts(aligned_frame, config, summary, directory / "figures")

    _write_json(directory / "study_config.json", config.model_dump(mode="json"))
    _write_json(directory / "data_quality.json", quality.model_dump(mode="json"))
    _write_json(directory / "eda_summary.json", summary)
    aligned_frame.reset_index().to_parquet(directory / "aligned_data.parquet", index=False)

    report_path = directory / "report.md"
    report_path.write_text(build_eda_report(config, quality, summary), encoding="utf-8")

    commit, dirty = _git_state(worktree)
    manifest = {
        "run_id": resolved_run_id,
        "study_fingerprint": fingerprint,
        "created_at": created_at.isoformat(),
        "git": {"commit": commit, "dirty": dirty},
        "runtime": {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "matplotlib": matplotlib.__version__,
            "pyarrow": pyarrow.__version__,
        },
        "inputs": input_manifest,
        "outputs": _output_manifest(directory),
    }
    manifest_path = directory / "manifest.json"
    _write_json(manifest_path, manifest)
    return ArtifactBundle(
        run_id=resolved_run_id,
        directory=directory,
        report_path=report_path,
        manifest_path=manifest_path,
        figure_paths=figure_paths,
    )


def write_agent_research_package(
    *,
    config: StudyConfig,
    quality: DataQualityReport,
    summary: dict[str, Any],
    aligned_frame: pandas.DataFrame,
    input_manifest: list[dict[str, Any]],
    fingerprint: str,
    worktree: Path | None,
    plan: Any,
    evaluation: Any,
    conversation: list[Any],
    execution_trace: list[dict[str, Any]],
    loop_context: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> ArtifactBundle:
    """Persist an Agent conversation, approved plan, evidence and evaluation."""

    from app.research.reporting.agent_report import build_agent_eda_report
    from app.research.reporting.html_report import build_html_report
    from app.research.reporting.report_charts import build_report_figures

    created_at = datetime.now(UTC)
    resolved_run_id = run_id or f"agent-{created_at.strftime('%Y%m%dT%H%M%S%fZ')}-{fingerprint}"
    if not SAFE_RUN_ID.fullmatch(resolved_run_id):
        raise ValueError("run_id must contain only letters, numbers, '.', '_' or '-'")

    directory = (config.analysis.output_directory / resolved_run_id).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    figures = build_report_figures(aligned_frame, config, summary)
    figure_paths = _write_figures(directory / "figures", figures)

    _write_json(directory / "study_config.json", config.model_dump(mode="json"))
    _write_json(directory / "conversation.json", [item.model_dump(mode="json") for item in conversation])
    _write_json(directory / "research_plan.json", plan.model_dump(mode="json"))
    _write_json(directory / "execution_trace.json", execution_trace)
    _write_json(directory / "data_quality.json", quality.model_dump(mode="json"))
    _write_json(directory / "eda_summary.json", summary)
    _write_json(directory / "agent_evaluation.json", evaluation.model_dump(mode="json"))
    if loop_context is not None:
        _write_json(directory / "research_loop.json", loop_context)
    aligned_frame.reset_index().to_parquet(directory / "aligned_data.parquet", index=False)

    report_path = directory / "report.html"
    report_path.write_text(
        build_html_report(
            config=config,
            quality=quality,
            summary=summary,
            plan=plan,
            evaluation=evaluation,
            figures=figures,
            run_id=resolved_run_id,
            created_at=created_at,
        ),
        encoding="utf-8",
    )
    (directory / "report.md").write_text(
        build_agent_eda_report(
            config=config,
            quality=quality,
            summary=summary,
            plan=plan,
            evaluation=evaluation,
            figure_names=set(figure_paths),
        ),
        encoding="utf-8",
    )

    commit, dirty = _git_state(worktree)
    manifest = {
        "run_id": resolved_run_id,
        "study_fingerprint": fingerprint,
        "created_at": created_at.isoformat(),
        "research_agent": {
            "plan_id": plan.plan_id,
            "data_fingerprint": plan.data_fingerprint,
            "parent_plan_id": plan.parent_plan_id,
            "plan_revision": plan.revision,
            "revision_source": plan.revision_source,
            "planner": plan.planner,
            "planning_model": plan.planning_model,
            "planning_prompt_version": plan.planning_prompt_version,
            "skill": {"name": plan.skill_name, "version": plan.skill_version},
            "research_protocol": (
                {
                    "protocol_id": plan.research_protocol_id,
                    "version": plan.research_protocol_version,
                    "function_order": plan.research_protocol_function_order,
                }
                if plan.research_protocol_id
                else None
            ),
            "approved_functions": [step.tool for step in plan.enabled_steps],
            "function_versions": {step.tool: step.tool_version for step in plan.enabled_steps},
        },
        "research_loop": loop_context,
        "git": {"commit": commit, "dirty": dirty},
        "runtime": {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "matplotlib": matplotlib.__version__,
            "pyarrow": pyarrow.__version__,
        },
        "inputs": input_manifest,
        "outputs": _output_manifest(directory),
    }
    manifest_path = directory / "manifest.json"
    _write_json(manifest_path, manifest)
    return ArtifactBundle(
        run_id=resolved_run_id,
        directory=directory,
        report_path=report_path,
        manifest_path=manifest_path,
        figure_paths=figure_paths,
    )
