"""Build the PySide6 desktop application with PyInstaller on Windows."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build PriceResearchAgent.exe with PyInstaller.")
    parser.add_argument(
        "--onefile",
        action="store_true",
        help="Create one executable instead of the recommended diagnostic-friendly one-folder bundle.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if sys.platform != "win32":
        raise SystemExit("Windows EXE must be built on Windows.")
    try:
        import PyInstaller.__main__
    except ImportError as exc:
        raise SystemExit('PyInstaller is not installed. Run: python -m pip install -e ".[build]"') from exc

    root = Path(__file__).resolve().parents[1]
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
            *skill_args,
            *exclude_args,
        ]
    )
    output = root / "dist" / ("PriceResearchAgent.exe" if args.onefile else "PriceResearchAgent")
    external_skill_root = output.parent / "skills" if args.onefile else output / "skills"
    external_skill_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / "skills" / "README.md", external_skill_root / "README.md")
    print(f"Build completed: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
