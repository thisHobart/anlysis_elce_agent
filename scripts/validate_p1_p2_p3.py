"""Run the bounded P1→P2→P1→P3→P1-feedback acceptance flow."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
from PySide6.QtWidgets import QApplication

from app.desktop.main_window import MainWindow
from app.desktop.session import SessionStore
from app.llm.factory import build_model_gateway
from app.research.data.sources.regions import RegionPriceFetch
from app.research.forecasting.data import prepare_forecast_plan
from app.research.full_flow import FlowRunReference, FullFlowStore, FullResearchFlow
from app.research.news import ObviousNewsEventExtractor, StructuredNewsEventExtractor
from app.research.tools.eda.price import analyze_price
from scripts.validate_desktop_phase1 import _shutdown_window, _wait
from scripts.validate_phase1 import validation_settings


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--mcp-targets", type=Path, help="通过浏览器MCP采集的新闻目标JSON数组")
    parser.add_argument("--mcp-server", default="browser")
    parser.add_argument("--mcp-catalog", type=Path)
    parser.add_argument(
        "--news",
        type=Path,
        default=Path("data/news/shandong_p2_test_news.jsonl"),
        help="Frozen local Shandong JSONL",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/acceptance"))
    return parser


def _frozen_news_evidence(path: Path) -> dict[str, object]:
    """Verify an adjacent frozen-corpus manifest when one is present."""

    resolved = path.resolve()
    lines = [line for line in resolved.read_text(encoding="utf-8").splitlines() if line.strip()]
    evidence: dict[str, object] = {
        "path": str(resolved),
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
        "records": len(lines),
    }
    manifest_path = resolved.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        return evidence
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source_file") != resolved.name:
        raise RuntimeError("新闻冻结清单的 source_file 与输入文件不一致")
    if manifest.get("source_file_sha256") != evidence["sha256"]:
        raise RuntimeError("新闻冻结清单哈希与输入文件不一致")
    if manifest.get("record_count") != evidence["records"]:
        raise RuntimeError("新闻冻结清单条数与输入文件不一致")
    evidence["manifest_path"] = str(manifest_path)
    evidence["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    evidence["corpus_id"] = manifest.get("corpus_id")
    evidence["frozen_at"] = manifest.get("frozen_at")
    return evidence


def _synthetic_fetcher(instant: datetime):
    """Adapt the controlled P2 event-price fixture to P1/P3's 15-minute source contract."""

    source = pd.read_csv("tests/fixtures/news_price/synthetic_prices.csv")
    source["timestamp"] = pd.to_datetime(source["timestamp"], utc=True).dt.tz_localize(None)
    price = source.set_index("timestamp")["price"].resample("15min").interpolate("time")

    def fetch(profile, *, output_directory, now=None, progress=None, **_kwargs):
        cutoff = pd.Timestamp(now or instant).tz_localize(None)
        target_index = price.index
        factor_index = pd.date_range(
            target_index.min(), cutoff.normalize() + pd.Timedelta(days=2) - pd.Timedelta(minutes=15), freq="15min"
        )
        slot = factor_index.hour * 4 + factor_index.minute // 15
        load = 60000 + 4000 * pd.Series(np.sin(2 * np.pi * slot / 96), index=factor_index)
        target = pd.DataFrame(
            {"timestamp": target_index, "rt_price": price.values, "available_at": target_index + pd.Timedelta(minutes=15)}
        )
        actuals = pd.DataFrame(
            {"timestamp": factor_index, "actual_load": load.values, "available_at": factor_index + pd.Timedelta(minutes=15)}
        )
        forecasts = pd.DataFrame(
            {
                "timestamp": factor_index,
                "forecast_load": load.values,
                "available_at": factor_index.normalize() - pd.Timedelta(days=1),
            }
        )
        frames = [frame.loc[frame["available_at"].le(cutoff)].copy() for frame in (target, actuals, forecasts)]
        root = Path(output_directory)
        root.mkdir(parents=True, exist_ok=True)
        for name, frame in zip(("target", "actuals", "forecasts"), frames, strict=True):
            frame.to_parquet(root / f"{name}.parquet", index=False)
        return RegionPriceFetch(
            profile=profile,
            path=root / "target.parquet",
            fetched_at=now or instant,
            row_count=len(frames[0]),
            start_at=frames[0].timestamp.min().to_pydatetime(),
            end_at=frames[0].timestamp.max().to_pydatetime(),
            duplicate_timestamp_rows=0,
            invalid_rows=0,
            database_name="SYNTHETIC_NEWS_PRICE_FIXTURE",
            actuals_path=root / "actuals.parquet",
            forecasts_path=root / "forecasts.parquet",
            actual_variable_names=("actual_load",),
            forecast_variable_names=("forecast_load",),
        )

    return fetch


def main() -> int:
    args = _parser().parse_args()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    output = (args.output / f"p1-p2-p3-{'synthetic' if args.synthetic else 'real'}-{stamp}").resolve()
    output.mkdir(parents=True)
    os.environ["PRICE_RESEARCH_APP_DATA_DIRECTORY"] = str(output / "app-data")
    os.environ["PRICE_RESEARCH_OUTPUT_DIRECTORY"] = str(output / "research")
    settings = validation_settings(120)
    instant = (
        datetime(2026, 3, 1, 23, 30, tzinfo=ZoneInfo("Asia/Shanghai"))
        if args.synthetic
        else datetime.now(ZoneInfo("Asia/Shanghai"))
    )
    evidence: dict[str, object] = {
        "data_mode": "synthetic" if args.synthetic else "real",
        "model": "not used for deterministic synthetic P1/P2" if args.synthetic else settings.llm_model,
        "training": "2 epochs / 64 samples" if args.synthetic else "production defaults",
        "timings": {},
    }
    state = None
    app = QApplication.instance() or QApplication([])
    window = MainWindow(
        session_store=SessionStore(output / "sessions.json"),
        region_fetcher=_synthetic_fetcher(instant) if args.synthetic else None,
    )
    workspace = window.workspace

    def wait(label: str, timeout: int = 420) -> None:
        started = time.monotonic()
        _wait(app, lambda: not workspace.is_busy, timeout=timeout, label=label)
        evidence["timings"][label] = round(time.monotonic() - started, 2)  # type: ignore[index]

    try:
        workspace.select_region("shandong")
        wait("p1_data")
        if workspace.current_session.data_state != "ready":
            raise RuntimeError("P1山东数据未就绪")
        session = workspace.current_session
        if args.synthetic:
            started = time.monotonic()
            target = pd.read_parquet(session.inputs["target"].path)
            target["timestamp"] = pd.to_datetime(target["timestamp"], errors="coerce")
            price = target.set_index("timestamp")["rt_price"]
            p1_summary = {
                "research_question": "受控合成P1电价体检",
                "study": {"name": "synthetic_price_study", "market": "TEST_MARKET", "timezone": "UTC"},
                "price": analyze_price(
                    price,
                    unit="synthetic_currency/MWh",
                    frequency="15min",
                    max_lag=192,
                    spike_iqr_multiplier=3.0,
                ),
            }
            p1_root = output / "p1-synthetic"
            p1_root.mkdir(parents=True)
            p1_report = p1_root / "report.md"
            p1_report.write_text(
                "# 受控合成 P1 电价体检\n\n"
                f"观测数：{p1_summary['price']['observations']}；"
                f"均值：{p1_summary['price']['distribution']['mean']}；"
                f"最低：{p1_summary['price']['distribution']['min']}；"
                f"最高：{p1_summary['price']['distribution']['max']}。\n",
                encoding="utf-8",
            )
            (p1_root / "eda_summary.json").write_text(
                json.dumps(p1_summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            p1_ref = FlowRunReference(
                run_kind="eda",
                run_id="p1-synthetic",
                artifact_directory=p1_root,
                report_path=p1_report,
                data_fingerprint=hashlib.sha256(target.to_csv(index=False).encode()).hexdigest()[:12],
            )
            evidence["timings"]["p1_deterministic"] = round(time.monotonic() - started, 2)  # type: ignore[index]
        else:
            workspace.submit_question(
                "先分析山东实时电价分布、尖峰和负价，并检查日历规律；这是P1-P2-P3全流程测试。"
            )
            wait("p1_plan")
            if session.status != "awaiting_plan_approval":
                detail = session.messages[-1].content if session.messages else ""
                raise RuntimeError(f"P1没有到达方案确认：{detail}")
            workspace.conversation.current_plan_widget.run_button.click()
            wait("p1_execute")
            if not session.runs or session.latest_eda_summary is None:
                raise RuntimeError("P1没有产生可交接证据")
            p1 = session.runs[-1]
            p1_summary = session.latest_eda_summary
            p1_ref = FlowRunReference(
                run_kind="eda",
                run_id=p1.run_id,
                parent_run_id=p1.parent_run_id,
                artifact_directory=Path(p1.artifact_directory),
                report_path=Path(p1.report_path),
                data_fingerprint=p1.data_fingerprint,
            )
        flow = FullResearchFlow(
            store=FullFlowStore(output / "full-flow.json"),
            output_directory=output / "full-flow",
        )
        state = flow.start(
            p1_run=p1_ref,
            eda_summary=p1_summary,
            information_cutoff=instant,
        )
        if args.synthetic:
            state = state.model_copy(
                update={
                    "p1_request": state.p1_request.model_copy(
                        update={"market": "TEST_MARKET", "timezone": "UTC", "interval_minutes": 15}
                    )
                }
            )
            flow.store.save(state)
            news = Path("tests/fixtures/news_price/synthetic_news.jsonl").resolve()
            converted = Path(session.inputs["target"].path)
            extractor = ObviousNewsEventExtractor(market_timezone="UTC")
        else:
            if args.mcp_targets:
                from app.integrations.mcp import (
                    BrowserMCPNewsCollector,
                    BrowserNewsTarget,
                    MCPClient,
                    freeze_collected_news,
                    load_mcp_server_catalog,
                )

                target_payload = json.loads(args.mcp_targets.read_text(encoding="utf-8"))
                if not isinstance(target_payload, list):
                    raise ValueError("--mcp-targets 必须是JSON数组")
                servers = load_mcp_server_catalog(args.mcp_catalog)
                if args.mcp_server not in servers:
                    raise ValueError(f"MCP服务器不存在：{args.mcp_server}")
                collector = BrowserMCPNewsCollector(MCPClient(servers[args.mcp_server]))
                records = collector.collect(
                    [BrowserNewsTarget.model_validate(item) for item in target_payload]
                )
                news, _manifest = freeze_collected_news(
                    records,
                    output / "p2-browser-news.jsonl",
                    server_id=args.mcp_server,
                )
                evidence["p2_collection"] = {
                    "method": "browser_mcp",
                    "server_id": args.mcp_server,
                    "records": len(records),
                    "targets_path": str(args.mcp_targets.resolve()),
                }
            else:
                news = args.news.resolve()
                evidence["p2_collection"] = {"method": "frozen_local_news"}
            converted = Path(session.inputs["target"].path)
            extractor = StructuredNewsEventExtractor(
                build_model_gateway(settings),
                market_timezone="Asia/Shanghai",
                extraction_passes=3,
            )
        started = time.monotonic()
        state = flow.run_p2(
            state=state,
            news_path=news,
            target_path=converted,
            extractor=extractor,
        )
        evidence["timings"]["p2"] = round(time.monotonic() - started, 2)  # type: ignore[index]
        evidence["p2_phase"] = state.phase
        evidence["news_snapshot"] = _frozen_news_evidence(news)
        events = json.loads((state.runs[-1].artifact_directory / "events.json").read_text(encoding="utf-8"))
        traceable: list[dict[str, object]] = []
        for anomaly in state.p1_request.anomaly_windows:
            anomaly_at = pd.Timestamp(anomaly["timestamp"])
            for event in events:
                effective = event.get("effective_start_at")
                if effective and abs(pd.Timestamp(effective).tz_localize(None) - anomaly_at.tz_localize(None)) <= pd.Timedelta(hours=6):
                    traceable.append(
                        {
                            "anomaly": anomaly,
                            "event_id": event.get("event_id"),
                            "event_type": event.get("event_type"),
                            "effective_start_at": effective,
                        }
                    )
        evidence["p1_anomaly_to_p2_events"] = traceable
        if state.phase == "p2_needs_review":
            evidence["stop_reason"] = state.stop_reason
            evidence["run_kinds"] = [item.run_kind for item in state.runs]
            evidence["run_relationships"] = [item.model_dump(mode="json") for item in state.runs]
            (output / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
            (output / "summary.md").write_text(
                "# P1—P2—P3 真实数据验证\n\n"
                "P1 与 P2 已运行。P2 的确定性质量门禁发现未解决复核项，流程按约束暂停，"
                "没有静默批准 P3。\n\n"
                f"- 新闻快照：`{news}`\n"
                f"- 冻结哈希：`{evidence['news_snapshot']['sha256']}`\n"  # type: ignore[index]
                f"- 停止原因：{state.stop_reason}\n"
                f"- P2 报告：`{state.runs[-1].report_path}`\n",
                encoding="utf-8",
            )
            print(json.dumps({"status": "needs_review", "output": str(output)}, ensure_ascii=True))
            return 2
        state = flow.synthesize_p1(state=state)
        evidence["p1_feature_decisions"] = [item.model_dump(mode="json") for item in state.feature_decisions]
        workspace.update_full_flow_state(
            state,
            state_path=flow.store.path,
            output_directory=flow.output_directory,
        )

        prepare_kwargs = {
            "question": "预测山东明天实时电价",
            "profile": workspace.region_profiles["shandong"],
            "target_path": session.inputs["target"].path,
            "actuals_path": session.inputs["actuals"].path or None,
            "forecasts_path": session.inputs["forecasts"].path or None,
            "output_directory": output / "research" / "forecasting",
            "source_data_fingerprint": session.dataset_fingerprint,
            "now": instant if args.synthetic else None,
            "snapshot_fetcher": workspace.region_fetcher,
        }

        def short_prepare(**kwargs):
            plan = prepare_forecast_plan(**kwargs)
            return plan.model_copy(
                update={
                    "training": plan.training.model_copy(
                        update={"max_epochs": 2, "maximum_training_samples": 64}
                    )
                }
            )

        started = time.monotonic()
        if args.synthetic:
            with patch("app.research.full_flow.service.prepare_forecast_plan", short_prepare):
                state = flow.prepare_p3(state=state, **prepare_kwargs)
        else:
            state = flow.prepare_p3(state=state, **prepare_kwargs)
        evidence["timings"]["p3_prepare"] = round(time.monotonic() - started, 2)  # type: ignore[index]
        workspace.update_full_flow_state(
            state,
            state_path=flow.store.path,
            output_directory=flow.output_directory,
        )
        plan_widget = workspace.conversation.current_plan_widget
        if workspace.current_session.status != "awaiting_plan_approval" or plan_widget is None:
            raise RuntimeError("P3没有投影到桌面的独立确认卡片")
        evidence["p3_desktop_approval"] = {
            "shown": True,
            "method": "forecast_plan_card_click",
            "plan_id": str(state.forecast_plan["plan_id"]),
            "status_before_click": workspace.current_session.status,
        }
        forecast_plan = state.forecast_plan or {}
        selected_news = sorted(
            {
                column
                for snapshot in forecast_plan.get("snapshots", [])
                for column in snapshot.get("news_feature_columns", [])
            }
        )
        snapshot_values = []
        for snapshot in forecast_plan.get("snapshots", []):
            frame = pd.read_parquet(snapshot["path"])
            snapshot_values.append(
                {
                    "role": snapshot["role"],
                    "as_of": snapshot["as_of"],
                    "expected_baseline_common_observations": snapshot.get(
                        "expected_baseline_common_observations", 0
                    ),
                    "fields": {
                        name: {
                            "valid": int(pd.to_numeric(frame[name], errors="coerce").notna().sum()),
                            "distinct": int(pd.to_numeric(frame[name], errors="coerce").nunique(dropna=True)),
                        }
                        for name in selected_news
                    },
                }
            )
        evidence["p3_news_input_contract"] = {
            "sources": forecast_plan.get("news_feature_sources", []),
            "selected_fields": selected_news,
            "snapshots": snapshot_values,
        }
        started = time.monotonic()
        plan_widget.run_button.click()
        wait("p3_desktop_execute_and_feedback")
        state = flow.store.load()
        evidence["timings"]["p3_execute_and_feedback"] = round(time.monotonic() - started, 2)  # type: ignore[index]
        evidence["p3_desktop_approval"].update(  # type: ignore[union-attr]
            {
                "status_after_run": workspace.current_session.status,
                "worker_idle": not workspace.is_busy,
                "result_card_count": sum(
                    message.kind == "forecast_result" for message in workspace.current_session.messages
                ),
            }
        )
        evidence.update(
            {
                "status": state.phase,
                "run_kinds": [item.run_kind for item in state.runs],
                "desktop_run_kinds": [item.run_kind for item in workspace.current_session.runs],
                "forecast_runs_used": state.forecast_runs_used,
                "feedback_runs_used": state.feedback_runs_used,
                "forecast_feedback": state.feedback.model_dump(mode="json") if state.feedback else None,
                "run_relationships": [item.model_dump(mode="json") for item in state.runs],
                "stop_reason": state.stop_reason,
            }
        )
        (output / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        feedback = state.feedback
        (output / "summary.md").write_text(
            "# P1—P2—P3 一轮验证报告\n\n"
            f"最终状态：`{state.phase}`。{state.stop_reason}\n\n"
            f"- 阶段关系：{' → '.join(item.run_kind for item in state.runs)}\n"
            f"- P3 执行次数：{state.forecast_runs_used}\n"
            f"- P1 反馈次数：{state.feedback_runs_used}\n"
            f"- 新闻快照：`{news}`\n"
            f"- 实际新闻字段：{', '.join(feedback.used_news_features) if feedback else '无'}\n"
            f"- 三折共同有效点：{list(feedback.fold_common_observations) if feedback else '无'}\n"
            f"- 评价状态：{feedback.status if feedback else '无'}\n",
            encoding="utf-8",
        )
        print(json.dumps({"status": state.phase, "output": str(output)}, ensure_ascii=True))
        return 0
    except Exception as exc:
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        if state is not None:
            evidence["persisted_phase"] = flow.store.load().phase if flow.store.path.is_file() else state.phase
        (output / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
        raise
    finally:
        _shutdown_window(app, window, timeout=240)


if __name__ == "__main__":
    raise SystemExit(main())
