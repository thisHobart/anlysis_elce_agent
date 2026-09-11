"""Build the PySide6 desktop application with PyInstaller on Windows."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build PriceResearchAgent.exe with PyInstaller.")
    parser.add_argument(
        "--onefile",
        action="store_true",
        help="Create one executable instead of the recommended diagnostic-friendly one-folder bundle.",
    )
    return parser


def sanitized_build_path(path_value: str, *, windows_root: Path) -> str:
    """Remove foreign ICU directories that can poison Qt dependency analysis.

    Qt uses the Windows ICU shim on this build.  If an unrelated application
    puts its own ``icuuc.dll`` on PATH, PyInstaller may bundle that incompatible
    binary beside Qt6Core and the resulting EXE fails with WinError 127.
    """

    system32 = (windows_root / "System32").resolve()
    retained: list[str] = []
    for value in path_value.split(os.pathsep):
        if not value:
            continue
        directory = Path(value)
        try:
            resolved = directory.resolve()
        except OSError:
            retained.append(value)
            continue
        foreign_icu = (resolved / "icuuc.dll").is_file() and resolved != system32
        if not foreign_icu:
            retained.append(value)
    return os.pathsep.join(retained)


def validate_built_executable(executable: Path, *, build_root: Path) -> dict[str, object]:
    """Start the freshly built desktop and verify packaged production resources."""

    build_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="exe-smoke-", dir=build_root) as temporary_value:
        temporary = Path(temporary_value)
        result_path = temporary / "result.json"
        environment = os.environ.copy()
        environment.update(
            {
                "QT_QPA_PLATFORM": "offscreen",
                "PRICE_RESEARCH_APP_DATA_DIRECTORY": str(temporary / "app-data"),
                "PRICE_RESEARCH_OUTPUT_DIRECTORY": str(temporary / "research"),
            }
        )
        process = subprocess.Popen(
            [str(executable), f"--smoke-test={result_path}"],
            env=environment,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        deadline = time.monotonic() + 180
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        if process.poll() is None:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                capture_output=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            raise SystemExit("Built EXE smoke test timed out after 180 seconds.")
        if process.returncode != 0:
            detail = result_path.read_text(encoding="utf-8") if result_path.is_file() else "no result file"
            raise SystemExit(f"Built EXE smoke test failed with exit code {process.returncode}: {detail}")
        if not result_path.is_file():
            raise SystemExit("Built EXE smoke test did not create its result file.")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        skill_names = {
            str(item.get("name"))
            for item in payload.get("builtin_skills", [])
            if isinstance(item, dict)
        }
        required = {"price-exogenous-eda", "price-forecastability-audit"}
        if payload.get("status") != "passed" or not required.issubset(skill_names):
            raise SystemExit(f"Built EXE smoke result is incomplete: {payload}")
        if payload.get("function_count") != 30 or not payload.get("window_constructed"):
            raise SystemExit(f"Built EXE did not load the complete Phase 1 runtime: {payload}")
        expected_app_data = (temporary / "app-data").resolve()
        expected_research = (temporary / "research").resolve()
        session_store = Path(str(payload.get("session_store_path", ""))).resolve()
        research_output = Path(str(payload.get("research_output_directory", ""))).resolve()
        if expected_app_data not in session_store.parents or research_output != expected_research:
            raise SystemExit(f"Built EXE ignored isolated user runtime directories: {payload}")
        return payload


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if sys.platform != "win32":
        raise SystemExit("Windows EXE must be built on Windows.")
    try:
        import PyInstaller.__main__
    except ImportError as exc:
        raise SystemExit('PyInstaller is not installed. Run: python -m pip install -e ".[build]"') from exc

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from app.research.data.sources.naming import WORD_LIST_FILENAME

    entry = root / "app" / "desktop" / "__main__.py"
    mode = "--onefile" if args.onefile else "--onedir"
    desktop_excludes = (
        "qdrant_client",
        "sentence_transformers",
        "transformers",
        "torch",
        "torchvision",
        "torchaudio",
        "sklearn",
        "sympy",
        "pypdf",
        "pymysql",
        "pytest",
        "IPython",
        "notebook",
    )
    exclude_args = [f"--exclude-module={name}" for name in desktop_excludes]
    skill_root = root / "app" / "research" / "skills"
    builtin_skills = sorted(path for path in skill_root.iterdir() if (path / "SKILL.md").is_file())
    if not builtin_skills:
        raise SystemExit("No builtin Skill bundle was found to package.")
    skill_args = [
        f"--add-data={path};app/research/skills/{path.name}" for path in builtin_skills
    ]
    # The variable word list ships beside its loader so a packaged build resolves
    # it the same way a source checkout does.
    word_list = root / "configs" / WORD_LIST_FILENAME
    if not word_list.is_file():
        raise SystemExit(f"No variable word list was found at {word_list}.")
    word_list_arg = f"--add-data={word_list};app/research/data/sources"
    original_path = os.environ.get("PATH", "")
    os.environ["PATH"] = sanitized_build_path(
        original_path,
        windows_root=Path(os.environ.get("SystemRoot", r"C:\Windows")),
    )
    try:
        PyInstaller.__main__.run(
            [
                str(entry),
                "--name=PriceResearchAgent",
                "--windowed",
                mode,
                "--noconfirm",
                "--clean",
                "--noupx",
                "--log-level=WARN",
                f"--paths={root}",
                f"--distpath={root / 'dist'}",
                f"--workpath={root / 'build' / 'pyinstaller'}",
                f"--specpath={root / 'build'}",
                "--collect-data=statsmodels",
                "--hidden-import=statsmodels.tsa.seasonal",
                "--hidden-import=statsmodels.tsa.stattools",
                "--hidden-import=statsmodels.stats.diagnostic",
                # Freezing a dataset writes Parquet, and pandas reaches pyarrow
                # lazily, so a packaged build has to be told to carry it.
                "--hidden-import=pyarrow",
                "--hidden-import=pyarrow.parquet",
                *skill_args,
                word_list_arg,
                *exclude_args,
            ]
        )
    finally:
        os.environ["PATH"] = original_path
    output = root / "dist" / ("PriceResearchAgent.exe" if args.onefile else "PriceResearchAgent")
    external_skill_root = output.parent / "skills" if args.onefile else output / "skills"
    external_skill_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / "skills" / "README.md", external_skill_root / "README.md")
    executable = output if args.onefile else output / "PriceResearchAgent.exe"
    smoke = validate_built_executable(executable, build_root=root / "build")
    print(f"Build completed: {output}")
    print(
        "EXE smoke passed: "
        f"{len(smoke['builtin_skills'])} builtin Skills, {smoke['function_count']} research functions"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
