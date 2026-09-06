"""Benchmark the current news extractor on frozen excerpts from real sources."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from app.llm.factory import build_model_gateway
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
        "--repeat",
        type=int,
        default=1,
        help=(
            "Run the whole benchmark N times and report per-case agreement. Stable headline "
            "metrics can hide cases that flip and cancel each other out, so instability is "
            "measured here rather than assumed away."
        ),
    )
    parser.add_argument(
        "--extractor",
        choices=("rule", "structured-model"),
        default="rule",
        help="Use the deterministic baseline or the explicitly configured structured model.",
    )
    parser.add_argument(
        "--extraction-passes",
        type=int,
        default=3,
        help=(
            "Independent structured-model passes per document. Above 1, a document is only "
            "accepted when every pass reaches the same verdict; disagreement is quarantined."
        ),
    )
    parser.add_argument(
        "--require-qualified",
        action="store_true",
        help="Return a non-zero exit code when the extractor misses a qualification threshold.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print a human-readable Chinese summary and per-item diagnostics.",
    )
    parser.add_argument(
        "--show-model-output",
        action="store_true",
        help="Print the model's raw function-call arguments and parsing error for debugging.",
    )
    return parser


_DISPOSITION_LABELS = {
    "event": "形成事件",
    "quarantine": "进入隔离",
    "irrelevant": "无关新闻",
    "invalid": "结果无效",
}


def _concise_error_lines(message: str | None, *, limit: int = 8) -> list[str]:
    """Keep actionable validation details while removing noisy URLs and value dumps."""

    if not message:
        return []
    lines = [line.strip() for line in message.splitlines() if line.strip()]
    useful: list[str] = []
    field_pattern = re.compile(r"^(?:events\.\d+\.)?[a-z_][a-z0-9_.]*$")
    for index, line in enumerate(lines):
        if line.startswith("For further information visit ") or "input_value=" in line:
            continue
        if index == 0 or "validation error" in line:
            useful.append(line)
        elif field_pattern.fullmatch(line):
            detail = lines[index + 1].strip() if index + 1 < len(lines) else ""
            useful.append(f"{line}: {detail}" if detail else line)
        elif not useful:
            useful.append(line)
        if len(useful) >= limit:
            break
    return useful


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _model_output_collector():
    current_corpus_id: str | None = None
    captured: dict[str, list[dict]] = {}

    def select(corpus_id: str) -> None:
        nonlocal current_corpus_id
        current_corpus_id = corpus_id

    def observe(payload: dict) -> None:
        corpus_id = current_corpus_id or "unassigned"
        captured.setdefault(corpus_id, []).append(payload)

    return observe, select, captured


class _DiagnosticExtractor:
    """Associate every model attempt, including repair attempts, with its source document."""

    def __init__(self, delegate, select: Callable[[str], None]) -> None:
        self.delegate = delegate
        self.select = select
        self.extractor_id = delegate.extractor_id
        self.extractor_version = delegate.extractor_version

    def extract(self, document):
        self.select(str(document.raw_metadata.get("corpus_id", "unassigned")))
        return self.delegate.extract(document)


def _print_human_report(
    payload: dict,
    report,
    output_path: Path,
    latest_path: Path,
    *,
    titles: dict[str, str],
    model_debug: dict[str, list[dict]],
) -> None:
    status_label = "通过" if report.qualified else "未通过"
    print("\nP2 真实新闻抽取资格报告")
    print("=" * 72)
    print(f"总体结果：{status_label}（{payload['status']}）")
    print(f"抽取器：{report.extractor_id}@{report.extractor_version}")
    print(f"样本数：{report.document_count}")
    print("\n质量指标")
    print(f"  可抽取事件召回率：{_percent(report.extractable_event_recall)}")
    print(f"  应隔离新闻召回率：{_percent(report.quarantine_recall)}")
    print(f"  无关新闻正确率：  {_percent(report.irrelevant_accuracy)}")
    print(f"  unknown 比率：     {_percent(report.unknown_rate)}")
    print(f"  逐案例完整准确率：{_percent(report.full_case_accuracy)}")
    failed = "、".join(check.code for check in report.failed_checks) or "无"
    print(f"  未通过检查：{failed}")

    print("\n逐条诊断")
    print("-" * 72)
    for case in report.cases:
        mark = "通过" if case.passed else "失败"
        expected = _DISPOSITION_LABELS.get(case.expected_disposition, case.expected_disposition)
        actual = _DISPOSITION_LABELS.get(case.actual_disposition, case.actual_disposition)
        print(f"{case.corpus_id} [{mark}] {titles.get(case.corpus_id, '')}")
        print(f"  1. 基准预期：{expected}")
        traces = model_debug.get(case.corpus_id) or []
        if traces:
            print(f"  2. 模型调用：{len(traces)} 次（包含 schema 修复重试）")
            for attempt, trace in enumerate(traces, start=1):
                calls = trace.get("tool_calls") or []
                raw_content = trace.get("raw_content") or ""
                provider = trace.get("provider") or "unknown"
                transport = trace.get("transport") or "unknown"
                method = trace.get("structured_output_method") or "unknown"
                enforcement = trace.get("schema_enforcement") or "unknown"
                print(
                    f"     第 {attempt} 次：provider={provider}；transport={transport}；"
                    f"method={method}；schema_enforcement={enforcement}"
                )
                if raw_content:
                    print("       raw content：")
                    for line in str(raw_content).splitlines():
                        print(f"         {line}")
                elif calls:
                    print("       raw function arguments：")
                    for line in json.dumps(
                        calls[0].get("arguments", {}),
                        ensure_ascii=False,
                        indent=2,
                        default=str,
                    ).splitlines():
                        print(f"         {line}")
                else:
                    print("       raw：无文本内容且未返回函数调用")
                if trace.get("parsed") is not None:
                    print("       LangChain 解析：成功")
                else:
                    print("       LangChain 解析：失败")
                    for line in _concise_error_lines(trace.get("parsing_error")):
                        print(f"         - {line}")
        else:
            print("  2. 模型调用：未请求显示或调用前已失败")

        print(f"  3. 本地处理：{actual}")
        if case.actual_event_type:
            print(
                "     事件："
                f"{case.actual_event_type}；区域={list(case.actual_regions)}；"
                f"资产={list(case.actual_assets)}；容量MW={case.actual_capacity_mw}；"
                f"开始={case.actual_start_at}"
            )
            print(
                f"     证据：{case.evidence_span_count} 段；"
                f"完整性={case.evidence_integrity}"
            )
        if case.quarantine_reason:
            print(f"     隔离原因：{case.quarantine_reason}")
        for line in _concise_error_lines(case.quarantine_message):
            print(f"       - {line}")
        print(f"  4. 基准判定：{mark}")
        print()

    print(f"本次 JSON 报告：{output_path.resolve()}")
    print(f"最新 JSON 报告：{latest_path.resolve()}")


def _cost_summary(runs: list[dict[str, float]], *, document_count: int) -> dict[str, float]:
    """What N passes actually cost, so the pass count is never the only thing on record."""

    if not runs:
        return {}
    calls = sum(run["model_calls"] for run in runs)
    model_seconds = sum(run["model_seconds"] for run in runs)
    wall_seconds = sum(run["wall_seconds"] for run in runs)
    documents = max(1, document_count * len(runs))
    return {
        "model_calls": calls,
        "model_calls_per_document": round(calls / documents, 2),
        "model_seconds": round(model_seconds, 3),
        "seconds_per_document": round(wall_seconds / documents, 3),
        "wall_seconds": round(wall_seconds, 3),
    }


_STABILITY_FIELDS = (
    "actual_disposition",
    "actual_event_type",
    "actual_regions",
    "actual_assets",
    "actual_start_at",
    "quarantine_reason",
    "passed",
)


def _stability_summary(runs: list) -> dict:
    """Per-case agreement across repeats.

    Headline metrics can sit perfectly still while individual cases flip, because two cases
    that trade places leave the totals unchanged. That reads as stability and is not, so the
    per-case verdicts are compared directly.
    """

    per_case: dict[str, dict[str, list[str]]] = {}
    for report in runs:
        for case in report.cases:
            record = per_case.setdefault(case.corpus_id, {field: [] for field in _STABILITY_FIELDS})
            payload = case.model_dump(mode="json")
            for field in _STABILITY_FIELDS:
                record[field].append(json.dumps(payload[field], ensure_ascii=False, sort_keys=True))

    cases = {}
    for corpus_id, fields in sorted(per_case.items()):
        varying = {name: sorted(set(values)) for name, values in fields.items() if len(set(values)) > 1}
        cases[corpus_id] = {
            "stable": not varying,
            "pass_count": fields["passed"].count("true"),
            "run_count": len(runs),
            "varying_fields": varying,
        }
    unstable = sorted(cid for cid, item in cases.items() if not item["stable"])
    verdict_unstable = sorted(
        cid for cid, item in cases.items() if 0 < item["pass_count"] < item["run_count"]
    )
    return {
        "run_count": len(runs),
        "qualified_runs": sum(1 for report in runs if report.qualified),
        # A metric identical in every run is not evidence of stability while cases still flip.
        "metric_stability_is_incidental": bool(verdict_unstable),
        "unstable_cases": unstable,
        "unstable_verdicts": verdict_unstable,
        "cases": cases,
    }


def main() -> int:
    args = build_parser().parse_args()
    if (args.verbose or args.show_model_output) and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    records = JsonlCollectedNewsAdapter(args.news).load()
    manifest = load_real_news_gold(args.gold)
    extractor_factory = None
    model_debug: dict[str, list[dict]] = {}
    built: list[StructuredNewsEventExtractor] = []
    cost: list[dict[str, float]] = []
    if args.extractor == "structured-model":
        observer = None
        select_corpus = None
        if args.show_model_output:
            observer, select_corpus, model_debug = _model_output_collector()
        gateway = build_model_gateway(structured_output_observer=observer)

        def extractor_factory(timezone):
            delegate = StructuredNewsEventExtractor(
                gateway,
                market_timezone=timezone,
                extraction_passes=args.extraction_passes,
            )
            built.append(delegate)
            return (
                _DiagnosticExtractor(delegate, select_corpus)
                if select_corpus is not None
                else delegate
            )

    repeats = max(1, args.repeat)
    runs = []
    for attempt in range(1, repeats + 1):
        if repeats > 1:
            print()
            print(f"===== 第 {attempt}/{repeats} 次基准运行 =====")
        built.clear()
        started = time.perf_counter()
        runs.append(
            benchmark_real_news_extractor(
                records,
                manifest,
                extractor_factory=extractor_factory,
            )
        )
        wall_seconds = time.perf_counter() - started
        cost.append(
            {
                "model_calls": sum(item.model_calls for item in built),
                "model_seconds": round(sum(item.model_seconds for item in built), 3),
                "wall_seconds": round(wall_seconds, 3),
            }
        )
    report = runs[-1]
    documents = NewsNormalizer().normalize_many(records)
    titles = {
        str(document.raw_metadata["corpus_id"]): document.title
        for document in documents
    }
    generated_at = datetime.now(UTC)
    payload = {
        "status": "qualified" if report.qualified else "not_qualified",
        "generated_at": generated_at.isoformat(),
        "news_path": str(args.news.resolve()),
        "gold_path": str(args.gold.resolve()),
        "selected_extractor": args.extractor,
        "extraction_passes": args.extraction_passes,
        "repeat": repeats,
        "cost": _cost_summary(cost, document_count=len(records)),
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
    if repeats > 1:
        payload["stability"] = _stability_summary(runs)
    if model_debug:
        payload["model_debug"] = model_debug
    args.output.mkdir(parents=True, exist_ok=True)
    run_stamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
    output_path = args.output / (
        f"real-news-extractor-{args.extractor}-{run_stamp}-{uuid4().hex[:8]}.json"
    )
    latest_path = args.output / f"real-news-extractor-{args.extractor}-latest.json"
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    output_path.write_text(serialized, encoding="utf-8")
    latest_path.write_text(serialized, encoding="utf-8")
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
        "latest_artifact_path": str(latest_path.resolve()),
    }
    if args.verbose or args.show_model_output:
        _print_human_report(
            payload,
            report,
            output_path,
            latest_path,
            titles=titles,
            model_debug=model_debug,
        )
    else:
        print(json.dumps(printable, ensure_ascii=False, indent=2))
    return 1 if args.require_qualified and not report.qualified else 0


if __name__ == "__main__":
    raise SystemExit(main())
