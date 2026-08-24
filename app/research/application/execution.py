"""Execute locked research plans with deterministic Python methods."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.research.agent.errors import ResearchPlanValidationError
from app.research.agent.schemas import AgentRunResult, ConversationMessage, EDAPlan
from app.research.application.planning import noop_progress, prepare_research_data, resolve_config
from app.research.data.loader import ResearchDataError
from app.research.data.snapshot import input_file_manifest, study_fingerprint
from app.research.evaluation.eda import evaluate_agent_run
from app.research.reporting.artifacts import write_agent_research_package
from app.research.schemas.study import StudyConfig
from app.research.skills.registry import SkillRegistry
from app.research.tools.catalog import validate_method_versions
from app.research.tools.contracts import ToolCall, ToolContext, ToolResult
from app.research.tools.eda.functions import build_eda_tool_registry
from app.research.tools.executor import ToolExecutor
from app.research.tools.policy import ToolPolicy
from app.research.tools.registry import ToolRegistry

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
    ) -> None:
        self.registry = registry or build_eda_tool_registry()
        self.skills = skills or SkillRegistry.default()
        self.executor = ToolExecutor(self.registry)

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
        if plan.planner != "llm":
            raise ResearchPlanValidationError("旧版本地规划方案不能继续执行，请由大模型重新生成方案。")
        skill = self.skills.get(plan.skill_name)
        if skill.version != plan.skill_version:
            raise ResearchPlanValidationError(
                f"Skill 版本不匹配 {plan.skill_name}: plan={plan.skill_version}, installed={skill.version}"
            )
        policy = ToolPolicy(allowed_tools=frozenset(skill.allowed_tools))
        for step in plan.enabled_steps:
            try:
                installed_tool = self.registry.get(step.tool)
            except ValueError as exc:
                raise ResearchPlanValidationError(str(exc)) from exc
            if step.tool_version != installed_tool.version:
                raise ResearchPlanValidationError(
                    f"工具版本不匹配 {step.tool}: "
                    f"plan={step.tool_version}, installed={installed_tool.version}"
                )
            methods = list(step.parameters.get("methods", []))
            if methods:
                try:
                    validate_method_versions(step.tool, methods, step.method_versions)
                except ValueError as exc:
                    raise ResearchPlanValidationError(str(exc)) from exc
        if output_directory is not None:
            output_path = Path(output_directory).resolve()
            config = config.model_copy(
                update={"analysis": config.analysis.model_copy(update={"output_directory": output_path})}
            )
        prepared = prepare_research_data(config)
        inputs = input_file_manifest(config)
        study_hash = study_fingerprint(config, inputs)
        if plan.data_fingerprint is not None and plan.data_fingerprint != study_hash:
            raise ResearchDataError(
                "研究文件或配置在方案生成后发生变化；为避免在新数据上执行旧方案，请重新生成分析方案"
            )
        if not prepared.quality.usable_for_eda:
            raise ResearchDataError("target data does not meet the minimum observation requirement for EDA")
        available_variables = {spec.name for spec in config.exogenous}
        selected = list(dict.fromkeys(plan.selected_variables))
        unknown = sorted(set(selected).difference(available_variables))
        if unknown:
            raise ResearchPlanValidationError(f"plan contains unknown variables: {', '.join(unknown)}")
        needs_variables = any(
            step.enabled and step.tool in {"exogenous_profile", "relationship_analysis"} for step in plan.steps
        )
        if needs_variables and not selected:
            raise ResearchPlanValidationError(
                "at least one exogenous variable must be selected for the approved plan"
            )
        return PreparedExecution(
            config=config,
            prepared=prepared,
            input_manifest=inputs,
            data_fingerprint=study_hash,
            policy=policy,
        )

    @staticmethod
    def _stable_call_id(plan: EDAPlan, step_id: str, payload: dict[str, Any]) -> str:
        canonical = json.dumps(
            {
                "plan_id": plan.plan_id,
                "revision": plan.revision,
                "step_id": step_id,
                "data_fingerprint": plan.data_fingerprint,
                "payload": payload,
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
                "name": step.tool,
                "version": step.tool_version,
                "arguments": arguments,
            }
            calls.append(
                ToolCall(
                    call_id=self._stable_call_id(plan, step.step_id, payload),
                    name=step.tool,
                    version=step.tool_version,
                    arguments=arguments,
                )
            )
        return calls

    def execute_call(self, *, plan: EDAPlan, study_config: StudyConfig, call: ToolCall) -> ToolResult:
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
        self.validate_tool_result(plan=plan, call=call, result=result)
        return result

    def validate_tool_result(self, *, plan: EDAPlan, call: ToolCall, result: ToolResult) -> None:
        expected_keys = {
            "data_quality": "data_quality",
            "price_profile": "price",
            "exogenous_profile": "exogenous",
            "relationship_analysis": "relationships",
        }
        if result.call.call_id != call.call_id or result.call.name != call.name:
            raise ResearchPlanValidationError("工具结果与调用身份不匹配")
        if result.output.result_key != expected_keys[call.name]:
            raise ResearchPlanValidationError(
                f"工具 {call.name} 返回了错误结果键：{result.output.result_key}"
            )
        if result.tool_version != call.version:
            raise ResearchPlanValidationError("工具结果版本与锁定调用不匹配")
        if result.data_fingerprint != plan.data_fingerprint:
            raise ResearchPlanValidationError("工具结果数据指纹与锁定计划不匹配")

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
        steps = {step.tool: step for step in plan.enabled_steps}
        now = datetime.now(UTC).isoformat()
        for result in tool_results:
            self.validate_tool_result(plan=plan, call=result.call, result=result)
            result_key = result.output.result_key
            if result_key != "data_quality":
                summary[result_key] = result.output.value
            step = steps[result.call.name]
            trace.append(
                {
                    "step_id": step.step_id,
                    "call_id": result.call.call_id,
                    "tool": result.call.name,
                    "title": step.title,
                    "parameters": step.parameters,
                    "status": "completed",
                    "result_key": result_key,
                    "started_at": now,
                    "finished_at": now,
                    "duration_ms": result.duration_ms,
                    "provider": result.provider,
                    "tool_version": result.tool_version,
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
        worktree = Path(__file__).resolve().parents[3]
        bundle = write_agent_research_package(
            config=config,
            quality=prepared.quality,
            summary=summary,
            aligned_frame=prepared.aligned.frame,
            input_manifest=prepared_execution.input_manifest,
            fingerprint=fingerprint,
            worktree=worktree,
            plan=plan,
            evaluation=evaluation,
            conversation=messages,
            execution_trace=trace,
            loop_context=loop_context,
            run_id=run_id,
        )
        callback(100, "研究完成")
        return AgentRunResult(
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
            callback(10 + round(65 * (index - 1) / max(len(calls), 1)), f"执行 {call.name}")
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
