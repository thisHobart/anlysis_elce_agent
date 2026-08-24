"""End-to-end tests for CLI-compatible EDA research packages."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from PIL import Image

from app.research.cli import main
from app.research.tools.eda.pipeline import run_eda_pipeline


def test_pipeline_writes_complete_reproducible_package(synthetic_study: Path):
    first = run_eda_pipeline(synthetic_study, run_id="test-run-one")
    second = run_eda_pipeline(synthetic_study, run_id="test-run-two")
    expected = {
        "study_config.json",
        "data_quality.json",
        "eda_summary.json",
        "aligned_data.parquet",
        "report.md",
        "manifest.json",
    }
    assert expected.issubset({path.name for path in first.artifact_directory.iterdir()})
    assert (first.artifact_directory / "eda_summary.json").read_bytes() == (
        second.artifact_directory / "eda_summary.json"
    ).read_bytes()
    aligned = pd.read_parquet(first.artifact_directory / "aligned_data.parquet")
    assert aligned.shape == (24 * 30, 5)
    assert str(aligned["timestamp"].dt.tz) == "Asia/Shanghai"
    assert "Correlation" in first.report_path.read_text(encoding="utf-8")

    manifest = json.loads((first.artifact_directory / "manifest.json").read_text(encoding="utf-8"))
    for output in manifest["outputs"]:
        path = first.artifact_directory / output["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == output["sha256"]

    figures = list((first.artifact_directory / "figures").glob("*.png"))
    assert len(figures) == 5
    assert all(Image.open(path).width >= 800 for path in figures)


def test_cli_runs_with_one_command_contract(synthetic_study: Path, capsys):
    exit_code = main(["--config", str(synthetic_study), "--run-id", "cli-run"])
    captured = capsys.readouterr()
    response = json.loads(captured.out)
    assert exit_code == 0
    assert response["status"] == "completed"
    assert Path(response["report_path"]).is_file()
