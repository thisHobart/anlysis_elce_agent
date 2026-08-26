"""Execute locked research plans with deterministic Python methods."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

from app.research.agent.errors import (
    DataFingerprintMismatchError,
    InsufficientDataError,
    PlanCompatibilityError,
    RepairablePlanError,
    ResearchPlanValidationError,
)
from app.research.agent.schemas import AgentRunResult, ConversationMessage, EDAPlan
from app.research.application.planning import noop_progress, prepare_research_data, resolve_config
from app.research.data.snapshot import input_file_manifest, study_fingerprint
from app.research.evaluation.eda import evaluate_agent_run
from app.research.reporting.artifacts import write_agent_research_package
from app.research.schemas.study import StudyConfig
from app.research.skills.registry import SkillRegistry
from app.research.tools.catalog import FUNCTION_CATALOG
from app.research.tools.contracts import ToolCall, ToolContext, ToolResult
from app.research.tools.eda.functions import build_eda_tool_registry
from app.research.tools.executor import ToolExecutor, output_fingerprint
from app.research.tools.policy import ToolPolicy
from app.research.tools.registry import ToolRegistry
from app.runtime_paths import source_worktree

ProgressCallback = Callable[[int, str], None]


@dataclass(frozen=True)
class PreparedExecution:
    config: StudyConfig
    prepared: Any
    input_manifest: list[dict[str, Any]]
    data_fingerprint: str
    policy: ToolPolicy


class EDAExecutionService:
    """Deterministic executor kept separate from model-led Agent roles."""

    def __init__(
        self,
        *,
        registry: ToolRegistry | None = None,
        skills: SkillRegistry | None = None,
        prepared_cache_size: int = 2,
    ) -> None:
        self.registry = registry or build_eda_tool_registry()
        self.skills = skills or SkillRegistry.default()
        self.executor = ToolExecutor(self.registry)
        self._prepared_cache_size = max(1, int(prepared_cache_size))
        self._prepared_cache: OrderedDict[str, PreparedExecution] = OrderedDict()
        self._prepared_cache_lock = RLock()

    @staticmethod
    def _prepared_cache_key(plan: EDAPlan, config: StudyConfig) -> str:
        config_payload = config.model_dump(mode="json", exclude={"analysis": {"output_directory"}})
        config_signature = hashlib.sha256(
            json.dumps(config_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]
        plan_signature = hashlib.sha256(
            json.dumps(plan.model_dump(mode="json"), ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]
        return (
            f"{plan.plan_id}:{plan.revision}:{plan.data_fingerprint or 'unlocked'}:"
            f"{plan_signature}:{config_signature}"
        )

    @staticmethod
    def _with_output_directory(prepared: PreparedExecution, output_directory: str | Path | None) -> PreparedExecution:
        if output_directory is None:
            return prepared
        output_path = Path(output_directory).resolve()
        config = prepared.config.model_copy(
            update={"analysis": prepared.config.analysis.model_copy(update={"output_directory": output_path})}
        )
        return PreparedExecution(
            config=config,
            prepared=prepared.prepared,
            input_manifest=prepared.input_manifest,
            data_fingerprint=prepared.data_fingerprint,
            policy=prepared.policy,
        )

    def evict_prepared(self, plan: EDAPlan | None = None) -> None:
        """Release a plan snapshot, or every cached snapshot when no plan is supplied."""

        with self._prepared_cache_lock:
            if plan is None:
                self._prepared_cache.clear()
                return
            prefix = f"{plan.plan_id}:{plan.revision}:"
            for key in [item for item in self._prepared_cache if item.startswith(prefix)]:
                self._prepared_cache.pop(key, None)

    @property
    def prepared_cache_entries(self) -> int:
        return len(self._prepared_cache)

    def prepare(
        self,
        *,
        plan: EDAPlan,
        config_path: str | Path | None = None,
        study_config: StudyConfig | None = None,
        output_directory: str | Path | None = None,
    ) -> PreparedExecution:
        """Validate all immutable execution inputs and reload canonical data."""

        config = resolve_config(config_path=config_path, study_config=study_config)
        if plan.data_fingerprint is None:
            raise PlanCompatibilityError("锁定执行前必须存在数据指纹，请重新生成并审批研究方案。")
        cache_key = self._prepared_cache_key(plan, config)
        with self._prepared_cache_lock:
            cached = self._prepared_cache.get(cache_key)
            if cached is not None:
                self._prepared_cache.move_to_end(cache_key)
                return self._with_output_directory(cached, output_directory)
        if plan.planner != "llm":
            raise PlanCompatibilityError("旧版本地规划方案不能继续执行，请由大模型重新生成方案。")
        skill = self.skills.get(plan.skill_name)
        if skill.version != plan.skill_version:
            raise PlanCompatibilityError(
                f"Skill 版本不匹配 {plan.skill_name}: plan={plan.skill_version}, installed={skill.version}"
            )
        policy = ToolPolicy(allowed_functions=frozenset(skill.allowed_functions))
        for step in plan.enabled_steps:
            try:
                installed_tool = self.registry.get(step.function)
            except ValueError as exc:
                raise PlanCompatibilityError(str(exc)) from exc
            if step.function_version != installed_tool.version:
                raise PlanCompatibilityError(
                    f"工具版本不匹配 {step.function}: "
                    f"plan={step.function_version}, installed={installed_tool.version}"
                )
            arguments = {key: value for key, value in step.parameters.items() if key != "max_lag_limit"}
            try:
                installed_tool.arguments_model.model_validate(arguments)
            except (TypeError, ValueError) as exc:
                raise RepairablePlanError(f"函数 {step.function} 参数无效：{exc}") from exc
        prepared = prepare_research_data(config)
        inputs = input_file_manifest(config)
        study_hash = study_fingerprint(config, inputs)
        if plan.data_fingerprint is not None and plan.data_fingerprint != study_hash:
            raise DataFingerprintMismatchError(
                "研究数据或自动识别的数据上下文在方案生成后发生变化；"
                "为避免在新数据上执行旧方案，请重新生成分析方案"
            )
        if not prepared.quality.usable_for_eda:
            raise InsufficientDataError("target data does not meet the minimum observation requirement for EDA")
        available_variables = {spec.name for spec in config.exogenous}
        selected = list(dict.fromkeys(plan.selected_variables))
        unknown = sorted(set(selected).difference(available_variables))
        if unknown:
            raise RepairablePlanError(f"plan contains unknown variables: {', '.join(unknown)}")
        needs_variables = any(step.enabled and FUNCTION_CATALOG[step.function].uses_variables for step in plan.steps)
        if needs_variables and not selected:
            raise RepairablePlanError(
                "at least one exogenous variable must be selected for the approved plan"
            )
        prepared_execution = PreparedExecution(
            config=config,
            prepared=prepared,
            input_manifest=inputs,
            data_fingerprint=study_hash,
            policy=policy,
        )
        with self._prepared_cache_lock:
            self._prepared_cache[cache_key] = prepared_execution
            self._prepared_cache.move_to_end(cache_key)
            while len(self._prepared_cache) > self._prepared_cache_size:
                self._prepared_cache.popitem(last=False)
        return self._with_output_directory(prepared_execution, output_directory)

    @staticmethod
    def _stable_work_id(plan: EDAPlan, payload: dict[str, Any]) -> str:
        if plan.data_fingerprint is None:
            raise PlanCompatibilityError("锁定研究函数前必须存在数据指纹")
        canonical = json.dumps(
            {
                "data_fingerprint": plan.data_fingerprint,
                "payload": payload,
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()[:24]

    @staticmethod
    def _stable_call_id(plan: EDAPlan, step_id: str, work_id: str) -> str:
        canonical = json.dumps(
            {
                "plan_id": plan.plan_id,
                "revision": plan.revision,
                "step_id": step_id,
                "work_id": work_id,
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()[:24]

    def compile_tool_queue(self, plan: EDAPlan) -> list[ToolCall]:
        calls: list[ToolCall] = []
        for step in plan.enabled_steps:
            arguments = {key: value for key, value in step.parameters.items() if key != "max_lag_limit"}
            payload = {
                "name": step.function,
                "version": step.function_version,
                "arguments": arguments,
            }
            work_id = self._stable_work_id(plan, payload)
            calls.append(
                ToolCall(
                    call_id=self._stable_call_id(plan, step.step_id, work_id),
                    work_id=work_id,
                    step_id=step.step_id,
                    name=step.function,
                    version=step.function_version,
                    arguments=arguments,
                )
            )
        return calls

    def execute_call(
        self,
        *,
        plan: EDAPlan,
        study_config: StudyConfig,
        call: ToolCall,
        validate_result: bool = True,
    ) -> ToolResult:
        prepared = self.prepare(plan=plan, study_config=study_config)
        context = ToolContext(
            config=prepared.config,
            frame=prepared.prepared.aligned.frame,
            quality=prepared.prepared.quality,
        )
        result = self.executor.execute(
            call,
            context=context,
            policy=prepared.policy,
            data_fingerprint=prepared.data_fingerprint,
        )
        if validate_result:
            self.validate_tool_result(plan=plan, call=call, result=result)
        return result

    def validate_tool_result(self, *, plan: EDAPlan, call: ToolCall, result: ToolResult) -> None:
        if (
            result.call.call_id != call.call_id
            or result.call.work_id != call.work_id
            or result.call.step_id != call.step_id
            or result.call.name != call.name
        ):
            raise ResearchPlanValidationError("工具结果与调用身份不匹配")
        expected_key = self.registry.get(call.name).result_key
        if result.output.result_key != expected_key:
            raise RepairablePlanError(
                f"工具 {call.name} 返回了错误结果键：{result.output.result_key}"
            )
        if result.tool_version != call.version:
            raise ResearchPlanValidationError("工具结果版本与锁定调用不匹配")
        if result.data_fingerprint != plan.data_fingerprint:
            raise ResearchPlanValidationError("工具结果数据指纹与锁定计划不匹配")
        if output_fingerprint(result.output) != result.output_hash:
            # A reused or checkpoint-restored payload must still hash to its recorded value.
            raise ResearchPlanValidationError(f"工具 {call.name} 的结果与记录的 output_hash 不一致")
        evidence_field = FUNCTION_CATALOG[call.name].evidence_field
        if evidence_field is not None and evidence_field not in result.output.value:
            raise RepairablePlanError(
                f"工具 {call.name} 的结果缺少证据字段 {evidence_field}"
            )

    @classmethod
    def _merge_evidence(cls, existing: Any, incoming: Any, *, field: str = "") -> Any:
        """Merge partial atomic-function evidence and reject contradictory shared metadata."""

        if existing is None:
            return incoming
        if incoming is None:
            return existing
        if isinstance(existing, dict) and isinstance(incoming, dict):
            merged = dict(existing)
            for key, value in incoming.items():
                merged[key] = cls._merge_evidence(merged.get(key), value, field=key)
            return merged
        if isinstance(existing, list) and isinstance(incoming, list):
            if field == "methods":
                return list(dict.fromkeys([*existing, *incoming]))
            if not existing:
                return incoming
            if not incoming or existing == incoming:
                return existing
            raise ResearchPlanValidationError(f"原子函数返回了冲突的列表证据：{field}")
        if existing == incoming:
            return existing
        raise ResearchPlanValidationError(f"原子函数返回了冲突的共享证据：{field}")

    def finalize(
        self,
        *,
        plan: EDAPlan,
        study_config: StudyConfig,
        tool_results: list[ToolResult],
        conversation: list[ConversationMessage | dict[str, Any]] | None = None,
        output_directory: str | Path | None = None,
        run_id: str | None = None,
        progress: ProgressCallback | None = None,
        loop_context: dict[str, Any] | None = None,
    ) -> AgentRunResult:
        callback = progress or noop_progress
        prepared_execution = self.prepare(
            plan=plan,
            study_config=study_config,
            output_directory=output_directory,
        )
        config = prepared_execution.config
        prepared = prepared_execution.prepared
        selected = list(dict.fromkeys(plan.selected_variables))
        summary: dict[str, Any] = {
            "study": {
                "name": config.study.name,
                "market": config.study.market,
                "timezone": config.study.timezone,
                "frequency": config.study.frequency,
                "target": config.target.name,
                "exogenous": [spec.name for spec in config.exogenous],
            },
            "research_question": plan.question,
            "selected_variables": selected,
            "methodology": {
                "missing_value_policy": "pairwise complete for relationships; no implicit imputation",
                "outlier_policy": "retain observations and report robust IQR flags",
                "lag_semantics": "positive lag compares feature[t-lag] with target[t]",
                "causal_claims": False,
            },
        }
        trace: list[dict[str, Any]] = []
        steps = {step.step_id: step for step in plan.enabled_steps}
        result_step_ids = [result.call.step_id for result in tool_results]
        if len(result_step_ids) != len(set(result_step_ids)):
            raise ResearchPlanValidationError("最终合并包含重复的计划步骤结果")
        if set(result_step_ids) != set(steps):
            missing = sorted(set(steps).difference(result_step_ids))
            unexpected = sorted(set(result_step_ids).difference(steps))
            raise ResearchPlanValidationError(
                f"最终工具证据与计划步骤不一致；缺失={missing}，越界={unexpected}"
            )
        tool_records = (loop_context or {}).get("tool_records", {})
        for result in tool_results:
            self.validate_tool_result(plan=plan, call=result.call, result=result)
            result_key = result.output.result_key
            if result_key != "data_quality":
                summary[result_key] = self._merge_evidence(summary.get(result_key), result.output.value)
            step = steps[result.call.step_id]
            record = tool_records.get(result.call.call_id, {})
            trace.append(
                {
                    "step_id": step.step_id,
                    "call_id": result.call.call_id,
                    "function": result.call.name,
                    "title": step.title,
                    "parameters": step.parameters,
                    "status": "completed",
                    "result_key": result_key,
                    "started_at": result.started_at or record.get("started_at"),
                    "finished_at": result.finished_at or record.get("finished_at"),
                    "duration_ms": result.duration_ms,
                    "provider": result.provider,
                    "function_version": result.tool_version,
                    "data_fingerprint": result.data_fingerprint,
                    "output_hash": result.output_hash,
                }
            )
        callback(78, "评估器检查结果与风险")
        evaluation = evaluate_agent_run(plan=plan, quality=prepared.quality, summary=summary)
        plan_payload = json.dumps(plan.model_dump(mode="json"), ensure_ascii=False, sort_keys=True).encode("utf-8")
        fingerprint = hashlib.sha256(
            prepared_execution.data_fingerprint.encode("ascii") + plan_payload
        ).hexdigest()[:12]
        messages = [
            item if isinstance(item, ConversationMessage) else ConversationMessage.model_validate(item)
            for item in (conversation or [])
        ]
        if not messages:
            messages = [ConversationMessage(role="user", content=plan.question)]
        callback(88, "生成可复现研究包")
        bundle = write_agent_research_package(
            config=config,
            quality=prepared.quality,
            summary=summary,
            aligned_frame=prepared.aligned.frame,
            input_manifest=prepared_execution.input_manifest,
            fingerprint=fingerprint,
            worktree=source_worktree(),
            plan=plan,
            evaluation=evaluation,
            conversation=messages,
            execution_trace=trace,
            loop_context=loop_context,
            run_id=run_id,
        )
        callback(100, "研究完成")
        completed = AgentRunResult(
            run_id=bundle.run_id,
            artifact_directory=bundle.directory,
            report_path=bundle.report_path,
            figure_paths=bundle.figure_paths,
            aligned_rows=len(prepared.aligned.frame),
            plan=plan,
            quality_report=prepared.quality,
            eda_summary=summary,
            evaluation=evaluation,
        )
        self.evict_prepared(plan)
        return completed

    def execute(
        self,
        *,
        plan: EDAPlan,
        config_path: str | Path | None = None,
        study_config: StudyConfig | None = None,
        conversation: list[ConversationMessage | dict[str, Any]] | None = None,
        output_directory: str | Path | None = None,
        run_id: str | None = None,
        progress: ProgressCallback | None = None,
    ) -> AgentRunResult:
        callback = progress or noop_progress
        config = resolve_config(config_path=config_path, study_config=study_config)
        callback(5, "重新加载数据，确保执行使用当前文件")
        calls = self.compile_tool_queue(plan)
        results: list[ToolResult] = []
        for index, call in enumerate(calls, start=1):
            callback(10 + round(65 * (index - 1) / max(len(calls), 1)), f"执行研究函数 {call.name}")
            results.append(self.execute_call(plan=plan, study_config=config, call=call))
        return self.finalize(
            plan=plan,
            study_config=config,
            tool_results=results,
            conversation=conversation,
            output_directory=output_directory,
            run_id=run_id,
            progress=callback,
        )
