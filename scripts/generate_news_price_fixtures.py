"""Regenerate the synthetic price fixture from its manifest.

Run this after changing `fixture_manifest.yaml`. A regression test regenerates the series
in memory and compares it to the committed CSV, so the two can never drift apart silently.

    python scripts/generate_news_price_fixtures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from app.research.news.synthetic import config_from_manifest, generate_price_series

FIXTURE_DIRECTORY = REPOSITORY_ROOT / "tests" / "fixtures" / "news_price"
MANIFEST_PATH = FIXTURE_DIRECTORY / "fixture_manifest.yaml"
PRICES_PATH = FIXTURE_DIRECTORY / "synthetic_prices.csv"


def render_csv(rows: tuple[tuple[str, ...], ...]) -> str:
    return "\n".join(",".join(row) for row in rows) + "\n"


def main() -> int:
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    series = generate_price_series(config_from_manifest(manifest))
    PRICES_PATH.write_text(render_csv(series.to_csv_rows()), encoding="utf-8")
    print(f"wrote {PRICES_PATH.relative_to(REPOSITORY_ROOT)}")
    print(f"  intervals: {len(series.timestamps)}")
    print(f"  window:    {series.start_at.isoformat()} .. {series.end_at.isoformat()}")
    print(f"  min/max:   {min(series.prices):.4f} / {max(series.prices):.4f}")
    print(f"  hash:      {series.content_hash()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
