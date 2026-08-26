"""Fresh-process checks for research package import boundaries."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "app.research.planning.compiler",
        "app.research.agent.orchestrator",
        "app.research.agent.subagents.eda",
    ],
)
def test_research_modules_import_without_order_dependencies(module_name: str) -> None:
    """Each public implementation module must import in a clean interpreter."""

    repository_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, "-c", f"import importlib; importlib.import_module({module_name!r})"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    ("package_name", "export_name"),
    [
        ("app.research.agent", "MainResearchAgent"),
        ("app.research.agent", "EDASubagent"),
        ("app.research.agent.subagents", "EDASubagent"),
        ("app.research.agent.subagents", "ModelEDAPlanner"),
    ],
)
def test_package_shortcuts_are_lazy_and_backward_compatible(package_name: str, export_name: str) -> None:
    """Removing eager imports must not break the existing package-level API."""

    repository_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, "-c", f"from {package_name} import {export_name}"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
