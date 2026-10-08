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
        "app.research.news.analysis",
        "app.research.news.pipeline",
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


def _imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


def test_p1_domain_does_not_import_the_p2_news_domain() -> None:
    """Domain modules stay separate; the explicit full-flow application service may coordinate both."""

    repository_root = Path(__file__).resolve().parents[2]
    research_root = repository_root / "app" / "research"
    violations: list[str] = []

    for path in research_root.rglob("*.py"):
        parts = path.relative_to(research_root).parts
        if "news" in parts or "full_flow" in parts:
            continue
        for module_name in _imported_modules(path):
            if module_name.startswith("app.research.news"):
                violations.append(f"{path.relative_to(repository_root)}: {module_name}")

    assert violations == [], f"P1 侧不得依赖 P2 新闻领域：{violations}"


def test_the_price_analysis_cannot_see_the_fixture_answer_key() -> None:
    """If the statistics could read the injected effects, recovering them would prove nothing."""

    repository_root = Path(__file__).resolve().parents[2]
    news_root = repository_root / "app" / "research" / "news"
    answer_key_modules = {"app.research.news.synthetic"}
    analysis_side = ("analysis.py", "clock.py", "prices.py", "features.py", "evidence.py")
    violations: list[str] = []

    for file_name in analysis_side:
        path = news_root / file_name
        for module_name in _imported_modules(path):
            if module_name in answer_key_modules:
                violations.append(f"{file_name}: {module_name}")
        source = path.read_text(encoding="utf-8")
        for forbidden in ("fixture_manifest", "EffectSpec", "effect_spec"):
            assert forbidden not in source, f"{file_name} 不得引用夹具答案：{forbidden}"

    assert violations == [], f"分析代码不得导入夹具生成器：{violations}"


def test_the_fixture_generator_is_never_pulled_in_by_the_news_package_import() -> None:
    """Importing the domain must not drag the answer key into the process at all."""

    repository_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import app.research.news; "
                "print('app.research.news.synthetic' in sys.modules)"
            ),
        ],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "False", "app.research.news 不应连带导入夹具生成器"
