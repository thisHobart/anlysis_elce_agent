"""Persistent application service for one bounded P1→P2→P1→P3 flow."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from app.research.forecasting.contracts import ForecastPlan, ForecastRunResult
from app.research.forecasting.data import prepare_forecast_plan, select_news_forecast_features
from app.research.forecasting.workflow import execute_forecast_workflow
from app.research.full_flow.contracts import (
    FeatureDecision,
    FlowRunReference,
    ForecastFeedback,
    FullFlowState,
    P2ResearchRequest,
    P2ReviewDecision,
    P2ReviewItem,
    P2ReviewSummary,
    synchronize_flow_lifecycle,
)
from app.research.news import (
    JsonlCollectedNewsAdapter,
    MarketClock,
    PriceObservations,
    build_forecast_news_features,
    run_news_price_study,
)
from app.research.news.workspace import (
    CachedNewsExtractor,
    NewsWorkspace,
    WorkspaceNewsAdapter,
    export_study,
    study_needs_review,
)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _p2_extractor_configuration(extractor: Any, clock: MarketClock) -> dict[str, object]:
    """Use the same cache identity as the standalone P2 workbench."""

    configuration: dict[str, object] = {
        "market": {
            "market": clock.market,
            "timezone": clock.timezone,
            "interval_minutes": clock.interval_minutes,
        },
        "passes": getattr(extractor, "extraction_passes", 1),
    }
    settings = getattr(getattr(extractor, "gateway", None), "settings", None)
    if settings is not None:
        configuration["model"] = settings.model_dump(mode="json", exclude={"llm_api_key"})
    return configuration


class FullFlowStore:
    """Atomic JSON state; forecast/P2 services keep their own immutable artifacts."""

    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()

    def save(self, state: FullFlowState) -> FullFlowState:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        updated = synchronize_flow_lifecycle(
            state.model_copy(update={"updated_at": datetime.now(UTC)})
        )
        temporary = self.path.with_suffix(self.path.suffix + f".{os.getpid()}.tmp")
        temporary.write_text(updated.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(self.path)
        return updated

    def load(self) -> FullFlowState:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("schema_version", 1) == 1:
            payload["schema_version"] = 2
        return synchronize_flow_lifecycle(FullFlowState.model_validate(payload))


def build_p2_request(
    *,
    eda_summary: dict[str, Any],
    information_cutoff: datetime,
) -> P2ResearchRequest:
    """Translate deterministic P1 evidence into a bounded P2 research request."""

    price = eda_summary.get("price") or {}
    distribution = price.get("distribution") or {}
    extremes = price.get("extremes") or {}
    anomaly_windows: list[dict[str, object]] = []
    for kind, items in (
        ("high", extremes.get("highest_examples", [])),
        ("low", extremes.get("lowest_examples", [])),
    ):
        for item in items[:5] if isinstance(items, list) else []:
            if not isinstance(item, dict) or not item.get("timestamp"):
                continue
            anomaly_windows.append(
                {
                    "kind": kind,
                    "timestamp": item["timestamp"],
                    "price": item.get("value"),
                    "source": "p1.price.extremes",
                }
            )
    start = datetime.fromisoformat(str(price["start_time"]))
    end = datetime.fromisoformat(str(price["end_time"]))
    return P2ResearchRequest(
        price_start=start,
        price_end=end,
        information_cutoff=information_cutoff,
        anomaly_evidence={
            key: distribution.get(key)
            for key in ("mean", "min", "max", "skewness", "kurtosis")
            if distribution.get(key) is not None
        },
        anomaly_windows=tuple(anomaly_windows),
        questions=(
            "负荷冲击是否与异常电价时段对齐",
            "机组停运、恢复或检修是否提供可追溯解释",
            "新能源供给变化或输电约束是否与价格变化同窗出现",
            "市场政策信息是否只有长期背景意义而不具备短期时点资格",
        ),
        search_topics=(
            "山东 实时电价 尖峰 负电价",
            "山东 用电负荷 高温 寒潮",
            "山东 机组 出力 检修 停运 恢复",
            "山东 风电 光伏 新能源 出力",
            "山东 输电约束 电网 通道",
            "山东 电力市场 政策 交易规则",
        ),
    )


def _price_observations(path: str | Path, request: P2ResearchRequest) -> PriceObservations:
    source = Path(path)
    frame = pd.read_parquet(source) if source.suffix.casefold() in {".parquet", ".pq"} else pd.read_csv(source)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["rt_price"] = pd.to_numeric(frame["rt_price"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "rt_price"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    timestamps = frame.timestamp
    if timestamps.dt.tz is None:
        timestamps = timestamps.dt.tz_localize(request.timezone)
    clock = MarketClock(
        market=request.market,
        timezone=request.timezone,
        interval_minutes=request.interval_minutes,
    )
    return PriceObservations(
        clock=clock,
        timestamps=tuple(value.to_pydatetime() for value in timestamps),
        prices=tuple(float(value) for value in frame.rt_price),
        provenance={"source": str(source), "sha256": _hash_file(source)},
    )


class FullResearchFlow:
    """Run and resume one explicitly bounded cross-stage research experiment."""

    def __init__(self, *, store: FullFlowStore, output_directory: str | Path):
        self.store = store
        self.output_directory = Path(output_directory).resolve()

    def _record_stage_attempt(self, state: FullFlowState, stage: str) -> FullFlowState:
        attempts = dict(state.stage_attempts)
        attempts[stage] = attempts.get(stage, 0) + 1
        return self.store.save(state.model_copy(update={"stage_attempts": attempts}))

    def _load_state_for_flow(self, flow_id: str) -> FullFlowState:
        state = self.store.load()
        if state.flow_id != flow_id:
            raise ValueError("复核命令不属于当前全流程；请刷新会话")
        return state

    @staticmethod
    def _review_blocking_reasons(result: dict[str, Any]) -> tuple[str, ...]:
        reasons: list[str] = []
        quarantine = result.get("quarantine")
        if isinstance(quarantine, dict):
            reasons.append(str(quarantine.get("message") or quarantine.get("reason_code") or "抽取结果被隔离"))
        for candidate in result.get("candidate_quarantines") or ():
            if isinstance(candidate, dict):
                reasons.append(
                    str(candidate.get("message") or candidate.get("reason_code") or "候选事件存在抽取分歧")
                )
        for event in result.get("events") or ():
            if not isinstance(event, dict):
                continue
            eligibility = event.get("analysis_eligibility")
            if event.get("relevance") == "short_term" and not event.get("effective_start_at"):
                reasons.append("短期事件缺少可用于事件分析的精确生效时间")
            elif eligibility == "needs_coverage_review":
                reasons.append("新闻覆盖范围不完整，不能直接用于分析")
        return tuple(dict.fromkeys(reasons))

    def get_p2_review_summary(self, flow_id: str) -> P2ReviewSummary:
        """Project the authoritative SQLite review state for desktop and command routing."""

        state = self._load_state_for_flow(flow_id)
        workspace = NewsWorkspace(self.output_directory / state.flow_id / "p2")
        items: list[P2ReviewItem] = []
        required_pending = 0
        required_resolved = 0
        optional_unreviewed = 0
        for entry in workspace.review_queue():
            original = dict(entry["original_result"])
            displayed = dict(entry["reviewed_result"] or original)
            if not displayed.get("pass_results") and original.get("pass_results"):
                displayed["pass_results"] = original["pass_results"]
            blocking = self._review_blocking_reasons(original)
            status = str(entry["decision"] or "unreviewed")
            required = bool(blocking)
            if required and status == "unreviewed":
                required_pending += 1
            elif required:
                required_resolved += 1
            elif status == "unreviewed":
                optional_unreviewed += 1
            allows_background = bool(original.get("events")) and not original.get("quarantine") and not (
                original.get("candidate_quarantines") or ()
            ) and all(
                isinstance(event, dict)
                and event.get("analysis_eligibility") in {"eligible", "needs_time_review"}
                for event in original.get("events") or ()
            ) and any(
                isinstance(event, dict) and event.get("analysis_eligibility") == "needs_time_review"
                for event in original.get("events") or ()
            )
            document = dict(entry["document"])
            items.append(
                P2ReviewItem(
                    cache_key=str(entry["cache_key"]),
                    document_version_id=str(entry["document_version_id"]),
                    revision=str(entry["revision"]),
                    title=str(document.get("title") or entry["document_version_id"]),
                    source_name=str(document.get("source_name") or "未知来源"),
                    source_ref=str(document.get("source_ref") or ""),
                    body=str(document.get("body") or ""),
                    result=displayed,
                    blocking_reasons=blocking,
                    review_status=status,
                    review_use=(str(entry["usage"]) if entry["review_id"] is not None else None),
                    reviewer=(str(entry["reviewer"]) if entry["reviewer"] else None),
                    review_reason=(str(entry["reason"]) if entry["reason"] else None),
                    required=required,
                    allows_background_only=allows_background,
                )
            )
        news_run = next((run for run in reversed(state.runs) if run.run_kind == "news"), None)
        quality_failures: list[str] = []
        if news_run is not None:
            package_path = news_run.artifact_directory / "package.json"
            try:
                package = json.loads(package_path.read_text(encoding="utf-8"))
                checks = (package.get("result_quality") or {}).get("checks") or ()
                quality_failures = [
                    f"{check.get('code', 'quality')}: {check.get('detail', '质量检查未通过')}"
                    for check in checks
                    if isinstance(check, dict) and not check.get("passed")
                ]
            except (OSError, ValueError):
                quality_failures = ["quality_artifact: P2质量检查产物不可读取"]
        revision = _fingerprint(
            {
                "flow_id": state.flow_id,
                "phase": state.phase,
                "items": [(item.cache_key, item.revision) for item in items],
                "quality_failures": quality_failures,
            }
        )
        return P2ReviewSummary(
            flow_id=state.flow_id,
            phase=state.phase,
            revision=revision,
            items=tuple(items),
            required_pending=required_pending,
            required_resolved=required_resolved,
            optional_unreviewed=optional_unreviewed,
            quality_failures=tuple(quality_failures),
            report_path=news_run.report_path if news_run else None,
        )

    def submit_p2_review(self, command: P2ReviewDecision) -> P2ReviewSummary:
        """Validate and append one review decision without advancing any flow stage."""

        state = self._load_state_for_flow(command.flow_id)
        if state.phase != "p2_needs_review":
            raise ValueError("当前阶段不接受P2复核操作")
        workspace = NewsWorkspace(self.output_directory / state.flow_id / "p2")
        workspace.review(
            command.cache_key,
            decision=command.decision,
            reviewer=command.reviewer,
            reason=command.reason,
            corrected=command.corrected,
            market_timezone=state.p1_request.timezone,
            usage=command.use,
            expected_revision=command.expected_revision,
            request_id=command.request_id,
        )
        return self.get_p2_review_summary(state.flow_id)

    def request_p3(self, state: FullFlowState) -> FullFlowState:
        """Explicitly extend a completed P1/P2 flow to P3 without rerunning prior stages."""

        latest = self._load_state_for_flow(state.flow_id)
        if latest.phase != "p1_synthesis_complete":
            raise ValueError("只有P1综合分析完成后才能追加P3预测")
        if latest.requested_forecast:
            return latest
        requested_stages = tuple(
            dict.fromkeys((*latest.requested_stages, "p3_prepare", "p3_execute", "p1_feedback"))
        )
        return self.store.save(
            latest.model_copy(
                update={
                    "requested_forecast": True,
                    "requested_stages": requested_stages,
                    "stop_reason": None,
                    "blocked_reason": None,
                }
            )
        )

    def revalidate_p2(
        self,
        flow_id: str,
        expected_revision: str,
        *,
        target_path: str | Path,
        extractor: Any,
    ) -> FullFlowState:
        """Rebuild P2 from saved decisions, then let deterministic gates choose the phase."""

        state = self._load_state_for_flow(flow_id)
        if state.phase != "p2_needs_review":
            raise ValueError("当前阶段不需要P2复核校验")
        summary = self.get_p2_review_summary(flow_id)
        if summary.revision != expected_revision:
            raise ValueError("P2复核状态已更新；请刷新后重新校验")
        if summary.required_pending:
            raise ValueError(f"仍有 {summary.required_pending} 条必审项未处理")
        if state.p2_news_path is None:
            raise ValueError("P2原始新闻路径缺失，不能重新校验")
        return self.run_p2(
            state=state,
            news_path=state.p2_news_path,
            target_path=target_path,
            extractor=extractor,
        )

    def start(
        self,
        *,
        p1_run: FlowRunReference,
        eda_summary: dict[str, Any],
        information_cutoff: datetime,
        request_text: str = "",
        requested_forecast: bool = False,
    ) -> FullFlowState:
        state = FullFlowState(
            p1_request=build_p2_request(eda_summary=eda_summary, information_cutoff=information_cutoff),
            runs=(p1_run,),
            request_text=request_text,
            requested_forecast=requested_forecast,
            durable_goal=request_text,
            requested_stages=(
                ("p2", "p1_synthesis", "p3_prepare", "p3_execute", "p1_feedback")
                if requested_forecast
                else ("p2", "p1_synthesis")
            ),
            input_fingerprints=(
                {"p1_data": p1_run.data_fingerprint} if p1_run.data_fingerprint else {}
            ),
        )
        return self.store.save(state)

    def run_p2(
        self,
        *,
        state: FullFlowState,
        news_path: str | Path,
        target_path: str | Path,
        extractor: Any,
    ) -> FullFlowState:
        if state.phase not in {"p1_initial_complete", "p2_needs_review"}:
            raise ValueError(f"当前阶段不能运行P2：{state.phase}")
        state = self._record_stage_attempt(state, "p2")
        prices = _price_observations(target_path, state.p1_request)
        workspace = NewsWorkspace(self.output_directory / state.flow_id / "p2")
        review_records = workspace.review_records()
        p2_knowledge_cutoff = max(
            (
                state.p1_request.information_cutoff.astimezone(UTC),
                *((state.p2_knowledge_cutoff.astimezone(UTC),) if state.p2_knowledge_cutoff else ()),
                *(record.collected_at.astimezone(UTC) for record in review_records),
            )
        )
        market_file = workspace.directory / "market.json"
        market_file.write_text(
            json.dumps(
                {
                    "market": prices.clock.market,
                    "timezone": prices.clock.timezone,
                    "interval_minutes": prices.clock.interval_minutes,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        configured_extractor = CachedNewsExtractor(
            extractor,
            workspace,
            _p2_extractor_configuration(extractor, prices.clock),
        )
        study = run_news_price_study(
            adapter=WorkspaceNewsAdapter(JsonlCollectedNewsAdapter(Path(news_path)), workspace),
            prices=prices,
            as_of=p2_knowledge_cutoff,
            axis="effective",
            extractor=configured_extractor,
        )
        package = export_study(study, workspace)
        forecast_end = p2_knowledge_cutoff + timedelta(days=2)
        frame = build_forecast_news_features(
            study.view.events,
            clock=prices.clock,
            start_at=prices.start_at,
            end_at=forecast_end,
            knowledge_cutoff=p2_knowledge_cutoff,
        )
        feature_path = package / "forecast_features.csv"
        frame.to_csv(feature_path, index=False, encoding="utf-8")
        feature_manifest = package / "forecast_features.manifest.json"
        feature_manifest.write_text(
            json.dumps(
                {
                    "source_run_id": package.name,
                    "information_cutoff": p2_knowledge_cutoff.isoformat(),
                    "source_news": str(Path(news_path).resolve()),
                    "source_news_sha256": _hash_file(Path(news_path)),
                    "price_source": str(Path(target_path).resolve()),
                    "price_source_sha256": _hash_file(Path(target_path)),
                    "feature_sha256": _hash_file(feature_path),
                    "rows": len(frame),
                    "columns": list(frame.columns),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        p2_run = FlowRunReference(
            run_kind="news",
            run_id=package.name,
            parent_run_id=state.runs[-1].run_id,
            artifact_directory=package,
            report_path=package / "report.md",
            data_fingerprint=_hash_file(feature_path)[:12],
        )
        phase = "p2_needs_review" if study_needs_review(study) else "p2_ready"
        fingerprints = {
            **state.input_fingerprints,
            "p2_news": _hash_file(Path(news_path)),
            "p2_price": _hash_file(Path(target_path)),
            "p2_features": _hash_file(feature_path),
        }
        updated = state.model_copy(
            update={
                "phase": phase,
                "runs": (*state.runs, p2_run),
                "p2_news_path": Path(news_path).resolve(),
                "p2_feature_path": feature_path,
                "p2_manifest_path": feature_manifest,
                "p2_knowledge_cutoff": p2_knowledge_cutoff,
                "stop_reason": "P2存在未解决复核项" if phase == "p2_needs_review" else None,
                "error": None,
                "resume_from_phase": None,
                "failed_stage": None,
                "input_fingerprints": fingerprints,
            }
        )
        return self.store.save(updated)

    def synthesize_p1(self, *, state: FullFlowState) -> FullFlowState:
        if state.phase != "p2_ready" or state.p2_feature_path is None:
            raise ValueError("P2必须完成且通过复核门禁后才能进行P1综合分析")
        state = self._record_stage_attempt(state, "p1_synthesis")
        selection_cutoff = state.p2_knowledge_cutoff or state.p1_request.information_cutoff
        selected, excluded = select_news_forecast_features(
            state.p2_feature_path,
            selection_cutoff=selection_cutoff,
        )
        feature_frame = pd.read_csv(state.p2_feature_path)
        feature_frame["timestamp"] = pd.to_datetime(feature_frame["timestamp"], errors="coerce")
        feature_frame["available_at"] = pd.to_datetime(feature_frame["available_at"], errors="coerce")
        cutoff = pd.Timestamp(selection_cutoff).tz_localize(None)
        feature_frame["timestamp"] = feature_frame["timestamp"].dt.tz_localize(None)
        feature_frame["available_at"] = feature_frame["available_at"].dt.tz_localize(None)
        eligible = feature_frame.loc[
            feature_frame["timestamp"].lt(cutoff) & feature_frame["available_at"].le(cutoff)
        ]
        manifest = json.loads(state.p2_manifest_path.read_text(encoding="utf-8")) if state.p2_manifest_path else {}
        price_path = Path(str(manifest.get("price_source", "")))
        price_frame = (
            pd.read_parquet(price_path)
            if price_path.suffix.casefold() in {".parquet", ".pq"}
            else pd.read_csv(price_path)
        )
        price_frame["timestamp"] = pd.to_datetime(price_frame["timestamp"], errors="coerce").dt.tz_localize(None)
        price_frame["rt_price"] = pd.to_numeric(price_frame["rt_price"], errors="coerce")
        joined = eligible.merge(price_frame[["timestamp", "rt_price"]], on="timestamp", how="left")
        per_hour = max(1, round(60 / state.p1_request.interval_minutes))

        def correlation(left: pd.Series, right: pd.Series) -> float | None:
            paired = pd.concat([pd.to_numeric(left, errors="coerce"), pd.to_numeric(right, errors="coerce")], axis=1).dropna()
            if len(paired) < 3 or paired.iloc[:, 0].nunique() < 2 or paired.iloc[:, 1].nunique() < 2:
                return None
            return round(float(paired.iloc[:, 0].corr(paired.iloc[:, 1])), 6)

        def feature_evidence(name: str) -> dict[str, object]:
            source = name.removeprefix("news_")
            values = pd.to_numeric(eligible.get(source), errors="coerce")
            valid = values.dropna()
            relation_values = pd.to_numeric(joined.get(source), errors="coerce")
            price_values = pd.to_numeric(joined.get("rt_price"), errors="coerce")
            midpoint = len(joined) // 2
            return {
                "eligible_rows": len(eligible),
                "valid_rows": int(valid.size),
                "coverage": round(float(valid.size / len(eligible)), 6) if len(eligible) else 0.0,
                "distinct_values": int(valid.nunique()),
                "changes": int(valid.ne(valid.shift()).sum() - (1 if not valid.empty else 0)),
                "minimum": float(valid.min()) if not valid.empty else None,
                "maximum": float(valid.max()) if not valid.empty else None,
                "standard_deviation": float(valid.std()) if valid.size > 1 else None,
                "price_correlation_same_time": correlation(relation_values, price_values),
                "price_correlation_lag_1h": correlation(relation_values.shift(per_hour), price_values),
                "price_correlation_lag_24h": correlation(relation_values.shift(24 * per_hour), price_values),
                "price_correlation_first_half": correlation(relation_values.iloc[:midpoint], price_values.iloc[:midpoint]),
                "price_correlation_second_half": correlation(relation_values.iloc[midpoint:], price_values.iloc[midpoint:]),
            }

        decisions = tuple(
            [
                FeatureDecision(
                    name=name,
                    decision="selected",
                    reason="截止点前具有足够观测且发生变化",
                    evidence=feature_evidence(name),
                )
                for name in selected
            ]
            + [
                FeatureDecision(
                    name=name,
                    decision="excluded",
                    reason=reason,
                    evidence=feature_evidence(name),
                )
                for name, reason in excluded.items()
            ]
        )
        synthesis_id = f"p1-synthesis-{state.flow_id}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}"
        root = self.output_directory / state.flow_id / "p1-synthesis" / synthesis_id
        root.mkdir(parents=True, exist_ok=True)
        report = root / "report.md"
        report.write_text(
            "# P1 综合分析\n\n"
            + ("采用的新闻特征：" + "、".join(selected) if selected else "没有新闻特征通过预测输入门禁。")
            + "\n\n## 特征处置\n\n"
            + "\n".join(
                f"- `{item.name}`：{item.decision}；{item.reason}；"
                f"覆盖率 {float(item.evidence.get('coverage', 0)):.1%}，"
                f"变化次数 {item.evidence.get('changes', 0)}，"
                f"同期相关 {item.evidence.get('price_correlation_same_time')}，"
                f"1小时滞后相关 {item.evidence.get('price_correlation_lag_1h')}，"
                f"24小时滞后相关 {item.evidence.get('price_correlation_lag_24h')}"
                for item in decisions
            )
            + "\n\n同期、1小时滞后、24小时滞后及前后半段相关只用于P1描述；"
            "P3会在最早回测锚点前独立重做选择，不使用评估窗口答案。"
            + "\n",
            encoding="utf-8",
        )
        payload = root / "feature_decisions.json"
        payload.write_text(json.dumps([item.model_dump() for item in decisions], ensure_ascii=False, indent=2), encoding="utf-8")
        run = FlowRunReference(
            run_kind="eda",
            run_id=synthesis_id,
            parent_run_id=state.runs[-1].run_id,
            artifact_directory=root,
            report_path=report,
            data_fingerprint=_hash_file(state.p2_feature_path)[:12],
        )
        updated = state.model_copy(
            update={
                "phase": "p1_synthesis_complete",
                "runs": (*state.runs, run),
                "feature_decisions": decisions,
                "stop_reason": None,
                "error": None,
                "resume_from_phase": None,
                "failed_stage": None,
            }
        )
        return self.store.save(updated)

    def prepare_p3(self, *, state: FullFlowState, **kwargs: Any) -> FullFlowState:
        if state.phase != "p1_synthesis_complete" or state.p2_feature_path is None:
            raise ValueError("必须先完成P1综合分析")
        state = self._record_stage_attempt(state, "p3_prepare")
        plan = prepare_forecast_plan(
            **kwargs,
            news_features_path=state.p2_feature_path,
            news_source_run_id=state.runs[-2].run_id,
            news_source_p1_run_id=state.runs[-1].run_id,
        )
        payload = plan.model_dump(mode="json")
        updated = state.model_copy(
            update={
                "phase": "awaiting_forecast_approval",
                "forecast_plan": payload,
                "forecast_plan_fingerprint": _fingerprint(payload),
                "approved_forecast_plan_id": None,
                "stop_reason": None,
                "error": None,
                "resume_from_phase": None,
                "failed_stage": None,
            }
        )
        return self.store.save(updated)

    def approve_and_run_p3(
        self,
        *,
        state: FullFlowState,
        plan_id: str,
        plan_fingerprint: str,
        progress: Any = None,
    ) -> FullFlowState:
        if state.forecast_plan is None or state.phase not in {"awaiting_forecast_approval", "forecast_running"}:
            raise ValueError("当前没有可批准或恢复的P3方案")
        plan = ForecastPlan.model_validate(state.forecast_plan)
        if plan.plan_id != plan_id or state.forecast_plan_fingerprint != plan_fingerprint:
            raise ValueError("P3批准与当前方案身份或内容指纹不一致")
        if state.forecast_runs_used and state.approved_forecast_plan_id != plan_id:
            raise ValueError("本轮P3执行预算已经用完")
        state = self._record_stage_attempt(state, "p3_execute")
        running = state.model_copy(
            update={
                "phase": "forecast_running",
                "approved_forecast_plan_id": plan_id,
                "forecast_approved_at": datetime.now(UTC),
                "forecast_runs_used": 1,
            }
        )
        running = self.store.save(running)
        try:
            result = execute_forecast_workflow(plan, progress=progress)
        except Exception as exc:
            failed = running.model_copy(
                update={
                    "phase": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "stop_reason": "P3执行或产物校验失败；保留当前方案和已完成检查点",
                    "resume_from_phase": "forecast_running",
                    "failed_stage": "p3_execute",
                }
            )
            self.store.save(failed)
            raise
        return self._finish_forecast(running, result)

    def stop(self, state: FullFlowState, *, reason: str = "用户停止当前全流程") -> FullFlowState:
        """Persist a terminal user stop without discarding completed stage artifacts."""

        stopped = state.model_copy(
            update={
                "phase": "stopped",
                "stop_reason": reason,
                "error": None,
                "resume_from_phase": None,
                "failed_stage": None,
            }
        )
        return self.store.save(stopped)

    def _finish_forecast(self, state: FullFlowState, result: ForecastRunResult) -> FullFlowState:
        status = result.evaluation_status
        excluded = {
            name: reason
            for source in result.plan.news_feature_sources
            for name, reason in source.excluded_columns.items()
        }
        diagnosis: list[str] = []
        next_steps: list[str] = []
        error_by_hour: dict[str, float] = {}
        try:
            points = pd.read_parquet(result.backtest_path)
            points["timestamp"] = pd.to_datetime(points["timestamp"], errors="coerce")
            points["absolute_error"] = (
                pd.to_numeric(points["predicted_rt_price"], errors="coerce")
                - pd.to_numeric(points["actual_rt_price"], errors="coerce")
            ).abs()
            hourly = points.dropna(subset=["timestamp", "absolute_error"]).groupby(
                points["timestamp"].dt.hour
            )["absolute_error"].mean()
            error_by_hour = {f"{int(hour):02d}:00": round(float(value), 6) for hour, value in hourly.items()}
        except (KeyError, OSError, ValueError):
            pass
        if status == "not_evaluable":
            diagnosis.append("至少一个回测折的模型与三种基线共同有效点少于72个。")
            next_steps.append("核对历史价格版本的 available_at/create_time 语义和逐折缺失时点。")
        elif status == "not_verified":
            diagnosis.append("样本门禁通过，但模型聚合MAE没有同时优于日、周朴素基线。")
            next_steps.append("检查误差集中时段、新闻特征变化和强朴素基线，下一轮再决定是否调整模型。")
        if error_by_hour:
            worst = sorted(error_by_hour.items(), key=lambda item: item[1], reverse=True)[:3]
            diagnosis.append(
                "绝对误差最高的小时为："
                + "、".join(f"{hour}（MAE {value:.2f}）" for hour, value in worst)
                + "。"
            )
        if result.plan.news_feature_sources and not result.diagnostics.get("news_feature_columns"):
            diagnosis.append("P2证据已生成，但没有新闻特征通过训练前选择和可获得性门禁。")
        constant_count = sum("没有变化" in reason for reason in excluded.values())
        sparse_count = sum("有效观测少于" in reason for reason in excluded.values())
        if constant_count:
            diagnosis.append(f"{constant_count}个新闻候选字段在训练前选择区间内恒定，已排除。")
        if sparse_count:
            diagnosis.append(f"{sparse_count}个新闻候选字段在训练前有效观测不足，已排除。")
        if status == "not_verified":
            model_mae = result.aggregate["model"].mae
            stronger = [
                label
                for key, label in (("day_naive", "日前同刻"), ("week_naive", "周前同刻"))
                if result.aggregate[key].mae <= model_mae
            ]
            if stronger:
                diagnosis.append("较强基线：" + "、".join(stronger) + "的MAE不高于模型。")
        feedback = ForecastFeedback(
            forecast_run_id=result.run_id,
            plan_id=result.plan.plan_id,
            status=status,
            fold_common_observations=tuple(item.common_observations for item in result.folds),
            aggregate_metrics={name: value.model_dump(mode="json") for name, value in result.aggregate.items()},
            error_by_hour=error_by_hour,
            used_news_features=tuple(result.diagnostics.get("news_feature_columns", [])),
            excluded_news_features=excluded,
            warnings=tuple(result.warnings),
            diagnosis=tuple(diagnosis),
            next_steps=tuple(next_steps),
            prediction_path=result.prediction_path,
            backtest_path=result.backtest_path,
            metrics_path=result.metrics_path,
            report_path=result.report_path,
            artifact_directory=result.artifact_directory,
            figure_paths=result.figure_paths,
            output_hash=result.output_hash,
        )
        run = FlowRunReference(
            run_kind="forecast",
            run_id=result.run_id,
            parent_run_id=state.runs[-1].run_id,
            artifact_directory=result.artifact_directory,
            report_path=result.report_path,
            data_fingerprint=result.plan.data_fingerprint,
        )
        if status == "verified":
            updated = state.model_copy(
                update={
                    "phase": "forecast_verified",
                    "runs": (*state.runs, run),
                    "feedback": feedback,
                    "stop_reason": "P3达到逐折样本与日/周基线目标",
                }
            )
        else:
            attempts = dict(state.stage_attempts)
            attempts["p1_feedback"] = attempts.get("p1_feedback", 0) + 1
            feedback_root = self.output_directory / state.flow_id / "p1-feedback"
            feedback_root.mkdir(parents=True, exist_ok=True)
            report = feedback_root / "report.md"
            report.write_text(
                "# P3 预测结果反馈\n\n"
                + "\n".join(f"- {item}" for item in feedback.diagnosis)
                + "\n\n## 下一步\n\n"
                + "\n".join(f"- {item}" for item in feedback.next_steps)
                + "\n\n这是预测结果反馈，不是重新开始P1；本轮在反馈后停止，不再次执行P3。\n",
                encoding="utf-8",
            )
            feedback_run = FlowRunReference(
                run_kind="feedback",
                run_id=f"p1-feedback-{state.flow_id}",
                parent_run_id=result.run_id,
                artifact_directory=feedback_root,
                report_path=report,
                data_fingerprint=result.plan.data_fingerprint,
            )
            updated = state.model_copy(
                update={
                    "phase": "feedback_complete",
                    "runs": (*state.runs, run, feedback_run),
                    "feedback": feedback,
                    "feedback_runs_used": 1,
                    "stage_attempts": attempts,
                    "stop_reason": "P3未达标或不可评估；已完成一次P1反馈分析并停止",
                }
            )
        return self.store.save(updated)
