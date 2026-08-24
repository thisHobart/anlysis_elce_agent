"""Command-line entry point for reproducible Phase 1 EDA runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from app.research.data.loader import ResearchDataError
from app.research.tools.eda.pipeline import run_eda_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze electricity price and exogenous time series.")
    parser.add_argument("--config", required=True, type=Path, help="Path to the research YAML configuration.")
    parser.add_argument("--output", type=Path, help="Override the configured artifact root directory.")
    parser.add_argument("--run-id", help="Optional deterministic run identifier, useful for controlled validation.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_eda_pipeline(
            args.config,
            output_directory=args.output,
            run_id=args.run_id,
        )
    except (FileExistsError, FileNotFoundError, OSError, ResearchDataError, ValidationError, ValueError) as exc:
        print(f"EDA failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": result.status,
                "run_id": result.run_id,
                "aligned_rows": result.aligned_rows,
                "artifact_directory": str(result.artifact_directory),
                "report_path": str(result.report_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
