"""Fresh-process checks for research package import boundaries."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "app.research.planning.compiler",
        "app.research.agent.orchestrator",
        "app.research.agent.retrieval",
        "app.research.agent.subagents.eda",
        "app.research.news",
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


def test_news_domain_does_not_import_p1_eda_domain_modules() -> None:
    """P2 may share infrastructure later, but must not depend on P1 domain contracts."""

    repository_root = Path(__file__).resolve().parents[2]
    news_root = repository_root / "app" / "research" / "news"
    forbidden_prefixes = (
        "app.research.agent",
        "app.research.planning",
        "app.research.schemas.study",
        "app.research.tools",
    )
    violations: list[str] = []

    for path in news_root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported_modules: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.append(node.module)
        for module_name in imported_modules:
            if module_name.startswith(forbidden_prefixes):
                violations.append(f"{path.name}: {module_name}")

    assert violations == []
