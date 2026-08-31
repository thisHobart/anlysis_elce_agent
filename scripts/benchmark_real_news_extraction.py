"""Benchmark the current news extractor on frozen excerpts from real sources."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from app.llm.openai_compatible import ResearchModelGateway
from app.research.news import (
    JsonlCollectedNewsAdapter,
    NewsNormalizer,
    StructuredNewsEventExtractor,
    benchmark_real_news_extractor,
    load_real_news_gold,
)


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    fixtures = root / "tests" / "fixtures" / "news_realistic"
    parser = argparse.ArgumentParser(
        description="Measure current extraction accuracy on traceable real-news excerpts."
    )
    parser.add_argument("--news", type=Path, default=fixtures / "source_news.jsonl")
    parser.add_argument("--gold", type=Path, default=fixtures / "gold_manifest.yaml")
    parser.add_argument("--output", type=Path, default=root / "artifacts" / "acceptance")
    parser.add_argument(
        "--extractor",
        choices=("rule", "structured-model"),
        default="rule",
        help="Use the deterministic baseline or the explicitly configured structured model.",
    )
    parser.add_argument(
        "--require-qualified",
        action="store_true",
        help="Return a non-zero exit code when the extractor misses a qualification threshold.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    records = JsonlCollectedNewsAdapter(args.news).load()
    manifest = load_real_news_gold(args.gold)
    extractor_factory = None
    if args.extractor == "structured-model":
        gateway = ResearchModelGateway()
        extractor_factory = lambda timezone: StructuredNewsEventExtractor(
            gateway,
            market_timezone=timezone,
        )
    report = benchmark_real_news_extractor(
        records,
        manifest,
        extractor_factory=extractor_factory,
    )
    documents = NewsNormalizer().normalize_many(records)
    payload = {
        "status": "qualified" if report.qualified else "not_qualified",
        "generated_at": datetime.now(UTC).isoformat(),
        "news_path": str(args.news.resolve()),
        "gold_path": str(args.gold.resolve()),
        "selected_extractor": args.extractor,
        "source_content_hashes": {
            str(document.raw_metadata["corpus_id"]): document.content_hash
            for document in documents
        },
        "report": {
            **report.model_dump(mode="json"),
            "qualified": report.qualified,
            "failed_check_codes": [check.code for check in report.failed_checks],
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / f"real-news-extractor-{uuid4().hex[:12]}.json"
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    printable = {
        "status": payload["status"],
        "extractor": f"{report.extractor_id}@{report.extractor_version}",
        "document_count": report.document_count,
        "extractable_event_recall": report.extractable_event_recall,
        "quarantine_recall": report.quarantine_recall,
        "irrelevant_accuracy": report.irrelevant_accuracy,
        "unknown_rate": report.unknown_rate,
        "full_case_accuracy": report.full_case_accuracy,
        "failed_check_codes": payload["report"]["failed_check_codes"],
        "artifact_path": str(output_path.resolve()),
    }
    print(json.dumps(printable, ensure_ascii=True, indent=2))
    return 1 if args.require_qualified and not report.qualified else 0


if __name__ == "__main__":
    raise SystemExit(main())
