"""P2 batch entry point: import, evidence-checked extraction, review and export."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from app.config import get_settings
from app.llm.factory import build_model_gateway
from app.research.news import (
    JsonlCollectedNewsAdapter,
    MarketClock,
    ObviousNewsEventExtractor,
    StructuredNewsEventExtractor,
    load_price_csv,
    run_news_price_study,
)
from app.research.news.analysis import AnalysisMethod
from app.research.news.workspace import (
    CachedNewsExtractor,
    NewsWorkspace,
    WorkspaceNewsAdapter,
    export_study,
    study_needs_review,
)


def build_parser():
    parser = argparse.ArgumentParser(description="新闻事件工作台：批处理、人工复核和可追溯导出")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="导入新闻并生成独立研究结果目录")
    run.add_argument("--workspace", type=Path, required=True)
    run.add_argument("--news", type=Path, required=True)
    run.add_argument("--prices", type=Path, required=True)
    run.add_argument("--market", required=True)
    run.add_argument("--timezone", required=True)
    run.add_argument("--interval-minutes", type=int, required=True)
    run.add_argument("--as-of", required=True, help="带时区的新闻可用时间截止点")
    run.add_argument("--axis", choices=("announcement", "effective"), default="announcement")
    run.add_argument(
        "--extractor",
        choices=("rule", "structured-model"),
        required=True,
        help="rule 仅为测试基线；真实新闻显式选择 structured-model",
    )
    run.add_argument("--extraction-passes", type=int, default=3)
    run.add_argument(
        "--control-matching", choices=("same_clock_day_type", "same_clock_weekday"), default="same_clock_day_type"
    )
    review = commands.add_parser("review", help="登记人工复核，下一次 run 会应用可见的复核版本")
    review.add_argument("--workspace", type=Path, required=True)
    review.add_argument("--cache-key", required=True, help="review-queue.json 中的 cache_key")
    review.add_argument("--decision", choices=("accepted", "corrected", "rejected"), required=True)
    review.add_argument("--reviewer", required=True)
    review.add_argument("--reason", required=True)
    review.add_argument("--candidate", type=Path, help="corrected 时填写符合 ModelNewsExtraction 的 JSON 文件")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    workspace = NewsWorkspace(args.workspace)
    market_file = workspace.directory / "market.json"
    if args.command == "review":
        if not market_file.is_file():
            parser.error("请先运行一次研究，建立市场配置和复核队列")
        market = json.loads(market_file.read_text(encoding="utf-8"))
        corrected = json.loads(args.candidate.read_text(encoding="utf-8")) if args.candidate else None
        review_id = workspace.review(
            args.cache_key,
            decision=args.decision,
            reviewer=args.reviewer,
            reason=args.reason,
            corrected=corrected,
            market_timezone=market["timezone"],
        )
        print(json.dumps({"review_id": review_id, "status": "recorded"}))
        return 0
    clock = MarketClock(market=args.market, timezone=args.timezone, interval_minutes=args.interval_minutes)
    cutoff = datetime.fromisoformat(args.as_of)
    if cutoff.tzinfo is None or cutoff.utcoffset() is None:
        parser.error("--as-of 必须带时区")
    cutoff = cutoff.astimezone(UTC)
    market = {"market": clock.market, "timezone": clock.timezone, "interval_minutes": clock.interval_minutes}
    if market_file.is_file() and json.loads(market_file.read_text(encoding="utf-8")) != market:
        parser.error("一个工作区只服务一个市场时钟；请换用独立工作区")
    market_file.write_text(json.dumps(market, ensure_ascii=False, indent=2), encoding="utf-8")
    configuration = {"market": market, "passes": args.extraction_passes}
    if args.extractor == "structured-model":
        settings = get_settings()
        configuration["model"] = settings.model_dump(mode="json", exclude={"llm_api_key"})
        delegate = StructuredNewsEventExtractor(
            build_model_gateway(settings), market_timezone=clock.timezone, extraction_passes=args.extraction_passes
        )
    else:
        delegate = ObviousNewsEventExtractor(market_timezone=clock.timezone)
        print("规则抽取器用于模板测试，结果不代表真实新闻模型资格。")
    study = run_news_price_study(
        adapter=WorkspaceNewsAdapter(JsonlCollectedNewsAdapter(args.news), workspace),
        prices=load_price_csv(args.prices, clock),
        as_of=cutoff,
        clock=clock,
        axis=args.axis,
        extractor=CachedNewsExtractor(delegate, workspace, configuration),
        method=AnalysisMethod(control_matching=args.control_matching),
    )
    output = export_study(study, workspace)
    print(
        json.dumps(
            {
                "output": str(output),
                "quality_passed": study.result_quality.passed,
                "events": len(study.view.events),
                "input_issues": len(study.package.input_issues),
                "quarantines": len(study.view.quarantined),
            },
            ensure_ascii=False,
        )
    )
    return 2 if study_needs_review(study) else 0


if __name__ == "__main__":
    raise SystemExit(main())
