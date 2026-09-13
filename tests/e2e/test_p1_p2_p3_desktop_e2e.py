"""Executable synthetic P1→P2→P1→desktop approval→P3→P1 feedback test."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e


def test_complete_flow_uses_desktop_approval_and_stops_after_one_feedback(tmp_path: Path):
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.validate_p1_p2_p3",
            "--synthetic",
            "--output",
            str(tmp_path),
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    run_directory = next(tmp_path.glob("p1-p2-p3-synthetic-*"))
    evidence = json.loads((run_directory / "evidence.json").read_text(encoding="utf-8"))

    assert evidence["status"] == "feedback_complete"
    assert evidence["run_kinds"] == ["eda", "news", "eda", "forecast", "feedback"]
    assert evidence["desktop_run_kinds"] == ["eda", "news", "eda", "forecast", "feedback"]
    assert evidence["forecast_runs_used"] == 1
    assert evidence["feedback_runs_used"] == 1
    assert evidence["p3_desktop_approval"]["shown"] is True
    assert evidence["p3_desktop_approval"]["method"] == "forecast_plan_card_click"
    assert evidence["p3_desktop_approval"]["status_before_click"] == "awaiting_plan_approval"
    assert evidence["p3_desktop_approval"]["status_after_run"] == "completed"
    assert evidence["p3_desktop_approval"]["worker_idle"] is True
    assert evidence["p3_desktop_approval"]["result_card_count"] == 1
    assert evidence["p1_anomaly_to_p2_events"]
    assert evidence["p3_news_input_contract"]["selected_fields"]
    assert all(
        snapshot["fields"]
        for snapshot in evidence["p3_news_input_contract"]["snapshots"]
    )
    expected_common = [
        snapshot["expected_baseline_common_observations"]
        for snapshot in evidence["p3_news_input_contract"]["snapshots"]
        if snapshot["role"] == "backtest"
    ]
    assert evidence["forecast_feedback"]["fold_common_observations"] == expected_common
    assert all(points >= 72 for points in expected_common)
    assert "停止" in evidence["stop_reason"]
