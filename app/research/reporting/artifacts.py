"""Write all EDA evidence into one checksummed, reproducible research package."""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy
import pandas
import pyarrow
import scipy
import statsmodels

from app.research.data.snapshot import sha256_file
from app.research.evidence import CallEvidenceLedger
from app.research.reporting.layout import ArtifactLayout
from app.research.reporting.methods import build_method_document, render_methods_markdown
from app.research.schemas.results import DataQualityReport
from app.research.schemas.study import StudyConfig
from app.research.skills.contracts import ResearchProtocol

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


def _build_evidence_figures(
    *,
    aligned_frame: Any,
    config: StudyConfig,
    evidence: CallEvidenceLedger,
) -> dict[str, str]:
    """Keep legacy figures and add call-specific figures for repeated functions."""

    from app.research.reporting.report_charts import build_report_figures

    figures = build_report_figures(aligned_frame, config, evidence)
    for function_name in dict.fromkeys(item.function for item in evidence.successful_calls()):
        runs = evidence.calls_for(function_name)
        if len(runs) < 2:
            continue
        for run in runs:
            call_id = run.call_id
            for key, markup in build_report_figures(
                aligned_frame,
                config,
                evidence.only(call_id),
            ).items():
                figures[f"{key}__{call_id[:8]}"] = markup
    return figures


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


def _load_completed_bundle(
    directory: Path,
    *,
    run_id: str,
    fingerprint: str,
    plan: Any,
) -> ArtifactBundle:
    """Validate and reuse a package committed by an earlier identical attempt."""

    layout = ArtifactLayout(directory)
    if not layout.manifest.is_file():
        raise FileExistsError(f"研究包目录已存在但没有完成标记：{directory}")
    try:
        manifest = json.loads(layout.manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"已有研究包 manifest 无法读取：{directory}") from exc
    agent = manifest.get("research_agent") or {}
    expected_identity = {
        "run_id": run_id,
        "study_fingerprint": fingerprint,
        "plan_id": plan.plan_id,
        "data_fingerprint": plan.data_fingerprint,
    }
    observed_identity = {
        "run_id": manifest.get("run_id"),
        "study_fingerprint": manifest.get("study_fingerprint"),
        "plan_id": agent.get("plan_id"),
        "data_fingerprint": agent.get("data_fingerprint"),
    }
    if observed_identity != expected_identity:
        raise FileExistsError(
            f"研究包 run_id 已被不同运行占用：expected={expected_identity}, observed={observed_identity}"
        )
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise ValueError(f"已有研究包缺少输出清单：{directory}")
    for item in outputs:
        if not isinstance(item, dict) or not item.get("path"):
            raise ValueError(f"已有研究包包含无效输出清单项：{directory}")
        relative = Path(str(item["path"]))
        source = (directory / relative).resolve()
        if relative.is_absolute() or not source.is_relative_to(directory):
            raise ValueError(f"已有研究包包含越界输出路径：{relative}")
        if not source.is_file():
            raise FileNotFoundError(f"已有研究包缺少输出：{relative}")
        if source.stat().st_size != int(item.get("size_bytes", -1)):
            raise ValueError(f"已有研究包输出大小不匹配：{relative}")
        if sha256_file(source) != item.get("sha256"):
            raise ValueError(f"已有研究包输出哈希不匹配：{relative}")
    figure_paths = {
        path.stem: path
        for path in sorted(layout.figures.glob("*.svg"))
        if path.is_file()
    } if layout.figures.is_dir() else {}
    return ArtifactBundle(
        run_id=run_id,
        directory=directory,
        report_path=layout.report_markdown,
        manifest_path=layout.manifest,
        figure_paths=figure_paths,
    )


def write_agent_research_package(
    *,
    config: StudyConfig,
    quality: DataQualityReport,
    evidence: CallEvidenceLedger,
    compatibility_summary: dict[str, Any],
    aligned_frame: pandas.DataFrame,
    input_manifest: list[dict[str, Any]],
    fingerprint: str,
    worktree: Path | None,
    plan: Any,
    evaluation: Any,
    conversation: list[Any],
    execution_trace: list[dict[str, Any]],
    research_protocol: ResearchProtocol | None,
    loop_context: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> ArtifactBundle:
    """Persist an Agent conversation, approved plan, evidence and evaluation."""

    from app.research.reporting.agent_report import build_agent_eda_report

    created_at = datetime.now(UTC)
    resolved_run_id = run_id or f"agent-{created_at.strftime('%Y%m%dT%H%M%S%fZ')}-{fingerprint}"
    if not SAFE_RUN_ID.fullmatch(resolved_run_id):
        raise ValueError("run_id must contain only letters, numbers, '.', '_' or '-'")

    output_root = config.analysis.output_directory.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    directory = (output_root / resolved_run_id).resolve()
    if directory.parent != output_root:
        raise ValueError("run_id resolved outside the configured output directory")
    if directory.exists():
        if not directory.is_dir():
            raise FileExistsError(f"研究包路径已被文件占用：{directory}")
        return _load_completed_bundle(
            directory,
            run_id=resolved_run_id,
            fingerprint=fingerprint,
            plan=plan,
        )

    staging = Path(
        tempfile.mkdtemp(
            prefix=f".research-package-{resolved_run_id[-16:]}-",
            dir=output_root,
        )
    ).resolve()
    layout = ArtifactLayout(staging)
    try:
        layout.create_directories()
        figures = _build_evidence_figures(
            aligned_frame=aligned_frame,
            config=config,
            evidence=evidence,
        )
        figure_paths = _write_figures(layout.figures, figures)

        _write_json(layout.study_context, config.model_dump(mode="json"))
        _write_json(layout.conversation, [item.model_dump(mode="json") for item in conversation])
        _write_json(layout.research_plan, plan.model_dump(mode="json"))
        _write_json(layout.execution_trace, execution_trace)
        _write_json(layout.data_quality, quality.model_dump(mode="json"))
        _write_json(layout.call_evidence, evidence.artifact_payload())
        _write_json(layout.eda_summary, compatibility_summary)
        _write_json(layout.agent_evaluation, evaluation.model_dump(mode="json"))
        if loop_context is not None:
            _write_json(layout.research_loop, loop_context)
        aligned_frame.reset_index().to_parquet(layout.aligned_data, index=False)

        method_document = build_method_document(
            run_id=resolved_run_id,
            plan=plan,
            protocol=research_protocol,
            execution_trace=execution_trace,
            evaluation=evaluation,
        )
        layout.methods_markdown.write_text(render_methods_markdown(method_document), encoding="utf-8")

        layout.report_markdown.write_text(
            build_agent_eda_report(
                config=config,
                quality=quality,
                evidence=evidence,
                plan=plan,
                evaluation=evaluation,
                figure_names=set(figure_paths),
                run_id=resolved_run_id,
                created_at=created_at,
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
                "approved_functions": [step.function for step in plan.enabled_steps],
                "function_versions": {step.function: step.function_version for step in plan.enabled_steps},
            },
            "research_loop": (
                {"path": layout.research_loop.relative_to(staging).as_posix()}
                if loop_context is not None
                else None
            ),
            "git": {"commit": commit, "dirty": dirty},
            "runtime": {
                "python": platform.python_version(),
                "numpy": numpy.__version__,
                "pandas": pandas.__version__,
                "scipy": scipy.__version__,
                "statsmodels": statsmodels.__version__,
                "pyarrow": pyarrow.__version__,
            },
            "inputs": input_manifest,
            "outputs": _output_manifest(staging),
        }
        _write_json(layout.manifest, manifest)

        try:
            os.replace(staging, directory)
        except OSError:
            # A concurrent or crash-recovery attempt may have committed the
            # same deterministic run while this staging package was built.
            if directory.is_dir():
                return _load_completed_bundle(
                    directory,
                    run_id=resolved_run_id,
                    fingerprint=fingerprint,
                    plan=plan,
                )
            raise
        committed_layout = ArtifactLayout(directory)
        committed_figures = {
            key: directory / path.relative_to(staging)
            for key, path in figure_paths.items()
        }
        return ArtifactBundle(
            run_id=resolved_run_id,
            directory=directory,
            report_path=committed_layout.report_markdown,
            manifest_path=committed_layout.manifest,
            figure_paths=committed_figures,
        )
    finally:
        if staging.is_dir() and staging.parent == output_root:
            shutil.rmtree(staging, ignore_errors=True)
