"""Build one immutable Shandong P2 news-study snapshot from read-only databases."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from app.config import get_settings
from app.llm.factory import build_model_gateway
from app.research.news import (
    JsonlCollectedNewsAdapter,
    MarketClock,
    StructuredNewsEventExtractor,
    collect_evidence_spans,
    load_price_csv,
    run_news_price_study,
)
from app.research.news.features import FEATURE_EVENT_TYPES
from app.research.news.model_extraction import (
    MODEL_EXTRACTION_PROMPT_VERSION,
    MODEL_EXTRACTION_SCHEMA_VERSION,
    MODEL_EXTRACTOR_VERSION,
)
from app.research.news.preparation import (
    INTERVAL_MINUTES,
    MARKET,
    MARKET_TIMEZONE,
    PROFILE_VERSION,
    SEARCH_CONFIG_VERSION,
    MySQLCredentials,
    PreparationError,
    apply_search_execution,
    build_p1_summary,
    build_search_plan,
    load_p1_frames,
    prepare_news_snapshot,
    profile_databases,
    sha256_file,
    write_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/shandong-news-study"))
    parser.add_argument("--run-id", help="Immutable run directory name; defaults to UTC timestamp")
    parser.add_argument("--as-of", required=True, help="Unified information cutoff with timezone offset")
    parser.add_argument("--search-results", type=Path, help="Frozen collected search-result JSONL")
    parser.add_argument("--search-executions", type=Path, help="Per-query execution evidence JSONL")
    parser.add_argument("--search-service", default="manual-reviewed-import")
    parser.add_argument("--extractor", choices=("none", "structured-model"), default="none")
    parser.add_argument("--extraction-passes", type=int, default=3)
    parser.add_argument("--extraction-blocker", help="Recorded reason a required real-news extraction could not run")
    return parser


def _timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise PreparationError("--as-of 必须包含时区偏移")
    return parsed.astimezone(UTC)


def _export_price(frame: pd.DataFrame, path: Path) -> None:
    output = frame[["timestamp", "value"]].copy()
    output["timestamp"] = pd.to_datetime(output["timestamp"])
    output["price"] = pd.to_numeric(output.pop("value"), errors="raise")
    if output.timestamp.dt.tz is None:
        output["timestamp"] = output.timestamp.dt.tz_localize(ZoneInfo(MARKET_TIMEZONE))
    output["timestamp"] = output.timestamp.dt.tz_convert(UTC).map(
        lambda item: item.isoformat().replace("+00:00", "Z")
    )
    output = output.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
    output.to_csv(path, index=False, encoding="utf-8", columns=["timestamp", "price"])


def _feature_frame(study) -> pd.DataFrame:
    rows = []
    for row in study.features.rows:
        payload = {
            "timestamp": row.interval_start.isoformat().replace("+00:00", "Z"),
            "active_event_count": row.active_event_count,
            "active_capacity_mw": row.active_capacity_mw,
            "direction_up_count": row.direction_up_count,
            "direction_down_count": row.direction_down_count,
            "new_announcement_count": row.new_announcement_count,
            "next_effective_in_hours": row.next_effective_in_hours,
        }
        for event_type in FEATURE_EVENT_TYPES:
            payload[f"{event_type}_count"] = row.event_type_counts.get(event_type, 0)
        rows.append(payload)
    return pd.DataFrame(rows)


def _empty_feature_frame() -> pd.DataFrame:
    columns = [
        "timestamp",
        "active_event_count",
        "active_capacity_mw",
        "direction_up_count",
        "direction_down_count",
        "new_announcement_count",
        "next_effective_in_hours",
        *(f"{event_type}_count" for event_type in FEATURE_EVENT_TYPES),
    ]
    return pd.DataFrame(columns=columns)


def _write_event_outputs(study, output: Path) -> dict:
    spans = collect_evidence_spans(study.documents, extraction_results=study.view.extraction_results)
    accepted = []
    for event in study.view.events:
        source_ids = set(event.source_event_ids)
        evidence = [
            span.model_dump(mode="json")
            for document_spans in spans.values()
            for span in document_spans
            if span.event_id is None or span.event_id in source_ids
        ]
        accepted.append({"event": event.model_dump(mode="json"), "evidence": evidence})
    with (output / "accepted_events.jsonl").open("w", encoding="utf-8") as stream:
        for item in accepted:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    existing = (output / "quarantined_news.jsonl").read_text(encoding="utf-8")
    with (output / "quarantined_news.jsonl").open("w", encoding="utf-8") as stream:
        stream.write(existing)
        for item in study.view.quarantined:
            stream.write(
                json.dumps({"stage": "extraction", **item.model_dump(mode="json")}, ensure_ascii=False) + "\n"
            )
    features = _feature_frame(study)
    features.to_parquet(output / "event_features.parquet", index=False)
    irrelevant = sum(
        event.relevance == "irrelevant" or event.event_type == "irrelevant" for event in study.view.events
    )
    missing_time = sum(
        event.relevance == "short_term" and event.effective_start_at is None for event in study.view.events
    )
    return {
        "merged_event_count": len(study.view.events),
        "accepted_event_count": len(accepted),
        "extraction_quarantine_count": len(study.view.quarantined),
        "irrelevant_event_count": irrelevant,
        "missing_effective_time_count": missing_time,
        "evidence_span_count": sum(len(item["evidence"]) for item in accepted),
        "feature_rows": len(features),
        "nonzero_active_feature_rows": int((features.active_event_count > 0).sum()),
        "result_quality_passed": study.result_quality.passed,
        "result_quality_checks": [item.__dict__ for item in study.result_quality.checks],
    }


def _readiness_markdown(
    database_profile: dict,
    p1: dict,
    news: dict,
    event: dict | None,
    extraction_blocker: str | None = None,
) -> str:
    actual = database_profile["actual_price_conclusion"]["exists"]
    normalized = news.get("normalized_news_count", 0)
    event_complete = bool(event and event.get("status") == "complete")
    events = event["accepted_event_count"] if event_complete else None
    nonzero = event["nonzero_active_feature_rows"] if event_complete else None
    extraction_passed = bool(event_complete and event["result_quality_passed"])
    ready = bool(actual and normalized and events and nonzero and extraction_passed)
    event_label = "不可用（抽取未完成）" if events is None else str(events)
    feature_label = "不可用（抽取未完成）" if nonzero is None else str(nonzero)
    verdict = "可进入受限描述性回测" if ready else "尚不可进入无泄漏事件—价格回测"
    blocker_line = f"- 事件抽取阻断：{extraction_blocker}" if extraction_blocker else ""
    full_text = int(news.get("full_text_count", 0))
    return f"""# 山东 P2 新闻数据回测准备度

## 结论

**{verdict}。** 山东实际日前与实时出清电价已经确认存在；但本次历史新闻是在当前运行时首次取得，`available_at` 不会回填为历史发布时间。若历史价格区间内没有非零新闻特征，就不能把这些新闻冒充当时可用信息进行回测。

## 已验证数据边界

- 市场：山东省 / {MARKET}；时区：{MARKET_TIMEZONE}；结算粒度：{INTERVAL_MINUTES} 分钟。
- 实时实际电价：{p1['study_start_at']} 至 {p1['study_end_at']}。
- 实际电价表存在：{str(actual).lower()}；预测表未被当作实际值。
- 规范化新闻：{normalized}；合并事件：{event_label}；历史时间轴非零事件特征行：{feature_label}。
- 可用于完整证据抽取的全文新闻：{full_text}/{normalized}。
- 模型与本地证据门禁整体通过：{str(extraction_passed).lower()}。
{blocker_line}

## 主要缺口和风险

1. 本次检索结果没有历史采集系统的可信 `first_seen_at`，只能使用本次 `fetched_at`；这是当前回测的关键阻断项。
2. 数据库 `create_time` 显示存在历史回填，不能自动视为市场发布时间；需要业务方确认数据发布/可获得时间口径。
3. 外来电与输电预测候选表当前为空，新闻侧虽可收集线路事件，但 P1 缺少对应连续数值变量。
4. 煤炭指数频率低，无法支持 15 分钟短期事件窗口；适合作为低频背景变量。
5. 自动抽取通过不等同于生产资格；真实新闻独立盲测和第二人复核仍是准入门槛。

## 最小下一步

建立带可信首次入库时间的历史新闻归档，或取得第三方新闻 API 的历史 first-seen 审计字段；冻结后重新运行本命令。随后只对在事件生效前已公开且已被系统取得的事件执行描述性窗口检验，并继续避免因果表述。
"""


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    as_of = _timestamp(args.as_of)
    run_id = args.run_id or as_of.strftime("%Y%m%dT%H%M%SZ")
    output = (args.output_root / run_id).resolve()
    if output.exists():
        raise PreparationError(f"运行目录已存在，拒绝改写不可变快照：{output}")
    output.mkdir(parents=True)
    try:
        feature_credentials = MySQLCredentials.from_env("feature")
        analytics_credentials = MySQLCredentials.from_env("analytics")
        database_profile = profile_databases(feature_credentials, analytics_credentials, as_of=as_of)
        frames = load_p1_frames(feature_credentials, analytics_credentials, as_of=as_of)
        p1_summary = build_p1_summary(frames)
        plan = build_search_plan(p1_summary, as_of=as_of)
        _export_price(frames["real_time_price"], output / "actual_price.csv")
        write_json(output / "database_profile.json", database_profile)
        write_json(output / "p1_data_summary.json", p1_summary)

        news_quality: dict = {"search_result_count": 0, "normalized_news_count": 0}
        event_quality: dict = {
            "status": "not_run",
            "accepted_event_count": None,
            "extraction_quarantine_count": None,
            "irrelevant_event_count": None,
            "missing_effective_time_count": None,
            "feature_rows": 0,
            "nonzero_active_feature_rows": None,
            "result_quality_passed": False,
        }
        if args.search_results:
            shutil.copyfile(args.search_results, output / "search_results.jsonl")
            query_ids = {item["query_id"] for item in plan["queries"]}
            news_quality = prepare_news_snapshot(
                output / "search_results.jsonl", output=output, query_ids=query_ids
            )
        else:
            for name in (
                "search_results.jsonl",
                "normalized_news.jsonl",
                "collected_news.jsonl",
                "quarantined_news.jsonl",
                "accepted_events.jsonl",
            ):
                (output / name).write_text("", encoding="utf-8")
            _empty_feature_frame().to_parquet(output / "event_features.parquet", index=False)
        plan = apply_search_execution(
            plan,
            imported_result_counts=news_quality.get("query_result_counts", {}),
            execution_log=args.search_executions,
            import_service=args.search_service if args.search_results else None,
        )
        write_json(output / "search_plan.json", plan)
        extraction_blocker = args.extraction_blocker

        if args.extractor == "none":
            (output / "accepted_events.jsonl").write_text("", encoding="utf-8")
            _empty_feature_frame().to_parquet(output / "event_features.parquet", index=False)
            extraction_blocker = extraction_blocker or "本次运行未请求结构化事件抽取"
        if args.extractor == "structured-model":
            if not args.search_results:
                raise PreparationError("structured-model 需要 --search-results")
            if news_quality.get("full_text_count", 0) != news_quality.get("normalized_news_count", 0):
                extraction_blocker = "真实新闻抽取要求 content_scope=full_text；摘录或摘要只能进入准备度报告"
                event_quality["status"] = "blocked"
                (output / "accepted_events.jsonl").write_text("", encoding="utf-8")
                _empty_feature_frame().to_parquet(output / "event_features.parquet", index=False)
            else:
                settings = get_settings()
                extractor = StructuredNewsEventExtractor(
                    build_model_gateway(settings),
                    market_timezone=MARKET_TIMEZONE,
                    extraction_passes=args.extraction_passes,
                )
                clock = MarketClock(
                    market=MARKET,
                    timezone=MARKET_TIMEZONE,
                    interval_minutes=INTERVAL_MINUTES,
                )
                study = run_news_price_study(
                    adapter=JsonlCollectedNewsAdapter(output / "collected_news.jsonl"),
                    prices=load_price_csv(output / "actual_price.csv", clock),
                    as_of=as_of,
                    clock=clock,
                    extractor=extractor,
                )
                event_quality = {"status": "complete", **_write_event_outputs(study, output)}

        quality = {
            "database": database_profile["actual_price_conclusion"],
            "news": news_quality,
            "events": event_quality,
            "extraction_blocker": extraction_blocker,
        }
        write_json(output / "quality_report.json", quality)
        (output / "readiness_report.md").write_text(
            _readiness_markdown(
                database_profile,
                p1_summary,
                news_quality,
                event_quality,
                extraction_blocker=extraction_blocker,
            ),
            encoding="utf-8",
        )
        files = {
            str(path.relative_to(output)).replace("\\", "/"): sha256_file(path)
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        }
        manifest = {
            "status": "complete"
            if event_quality.get("status") == "complete" and event_quality["result_quality_passed"]
            else "needs_review",
            "run_id": run_id,
            "as_of": as_of.isoformat(),
            "market": MARKET,
            "market_timezone": MARKET_TIMEZONE,
            "database_profile_version": PROFILE_VERSION,
            "search_config_version": SEARCH_CONFIG_VERSION,
            "event_extraction": {
                "extractor": args.extractor,
                "blocker": extraction_blocker,
                "extractor_version": MODEL_EXTRACTOR_VERSION
                if args.extractor == "structured-model"
                else None,
                "prompt_version": MODEL_EXTRACTION_PROMPT_VERSION
                if args.extractor == "structured-model"
                else None,
                "schema_version": MODEL_EXTRACTION_SCHEMA_VERSION
                if args.extractor == "structured-model"
                else None,
                "passes": args.extraction_passes if args.extractor == "structured-model" else 0,
            },
            "input_sources": {
                "databases": [feature_credentials.safe_identity, analytics_credentials.safe_identity],
                "search_service": args.search_service if args.search_results else None,
                "search_execution_log": args.search_executions.name if args.search_executions else None,
            },
            "files": files,
        }
        write_json(output / "manifest.json", manifest)
        print(
            json.dumps(
                {
                    "output": str(output),
                    "status": manifest["status"],
                    "news": news_quality,
                    "events": event_quality,
                },
                ensure_ascii=False,
            )
        )
    finally:
        # A failed run intentionally has no manifest and is never a completed snapshot.
        pass
    return 0 if manifest["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
