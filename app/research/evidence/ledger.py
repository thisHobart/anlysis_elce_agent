"""Validated, call-level evidence ledger used by evaluation and reporting."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.research.agent.errors import ResearchPlanValidationError
from app.research.agent.schemas import EDAPlan
from app.research.schemas.study import StudyConfig
from app.research.tools.contracts import ToolOutput, ToolResult
from app.research.tools.executor import output_fingerprint


class CallEvidence(BaseModel):
    """One immutable atomic-call result, including its provider correlation."""

    model_config = ConfigDict(extra="forbid")

    sequence: int = Field(ge=1)
    call_id: str = Field(min_length=1)
    provider_call_id: str | None = None
    origin: Literal["direct", "recipe", "system_preflight"] = "direct"
    work_id: str = Field(pattern=r"^[a-f0-9]{24}$")
    function: str = Field(min_length=1)
    function_version: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    result_key: str | None = None
    value: dict[str, Any] = Field(default_factory=dict)
    result_reference: dict[str, Any] | None = None
    data_fingerprint: str = Field(pattern=r"^[a-f0-9]{12}$")
    output_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    status: Literal["completed", "reused", "failed", "cancelled"]
    error: dict[str, Any] | None = None
    started_at: str | None = None
    finished_at: str | None = None
    duration_ms: float | None = Field(default=None, ge=0)
    provider: str | None = None

    @model_validator(mode="after")
    def validate_result(self) -> CallEvidence:
        if self.status in {"completed", "reused"}:
            if not self.result_key or not self.output_hash:
                raise ValueError("成功调用必须包含结果键和输出哈希")
            observed = output_fingerprint(ToolOutput(result_key=self.result_key, value=self.value))
            if observed != self.output_hash:
                raise ValueError(f"调用 {self.call_id} 的值与 output_hash 不一致")
        return self


def _merge_evidence(existing: Any, incoming: Any, *, field: str = "") -> Any:
    if existing is None:
        return incoming
    if incoming is None:
        return existing
    if isinstance(existing, dict) and isinstance(incoming, dict):
        merged = dict(existing)
        for key, value in incoming.items():
            merged[key] = _merge_evidence(merged.get(key), value, field=key)
        return merged
    if isinstance(existing, list) and isinstance(incoming, list):
        if field == "methods":
            return list(dict.fromkeys([*existing, *incoming]))
        if not existing or existing == incoming:
            return incoming or existing
        raise ResearchPlanValidationError(f"原子函数返回了冲突的列表证据：{field}")
    if existing == incoming:
        return existing
    raise ResearchPlanValidationError(f"原子函数返回了冲突的共享证据：{field}")


class CallEvidenceLedger(BaseModel):
    """Authoritative ordered evidence source for one frozen dataset."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    data_fingerprint: str = Field(pattern=r"^[a-f0-9]{12}$")
    calls: list[CallEvidence] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_ledger(self) -> CallEvidenceLedger:
        ordered = sorted(self.calls, key=lambda item: item.sequence)
        if ordered != self.calls:
            raise ValueError("调用级证据必须按 sequence 排序")
        call_ids = [item.call_id for item in self.calls]
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("调用级证据包含重复 call_id")
        sequences = [item.sequence for item in self.calls]
        if len(sequences) != len(set(sequences)):
            raise ValueError("调用级证据包含重复 sequence")
        if any(item.data_fingerprint != self.data_fingerprint for item in self.calls):
            raise ValueError("调用级证据包含其他数据快照的结果")
        return self

    @classmethod
    def from_tool_results(
        cls,
        *,
        plan: EDAPlan,
        tool_results: list[ToolResult],
        loop_context: dict[str, Any] | None = None,
    ) -> CallEvidenceLedger:
        if plan.data_fingerprint is None:
            raise ResearchPlanValidationError("执行计划缺少数据指纹，无法建立证据账本")
        canonical = {
            str(item.get("call_id")): item
            for item in (loop_context or {}).get("call_evidence", [])
            if isinstance(item, dict) and item.get("call_id")
        }
        tool_records = (loop_context or {}).get("tool_records", {})
        groups = (loop_context or {}).get("provider_call_groups", {})
        provider_by_call: dict[str, tuple[str | None, str]] = {}
        for provider_call_id, raw_group in groups.items():
            if not isinstance(raw_group, dict):
                continue
            origin = str(raw_group.get("origin") or "direct")
            for child_call_id in raw_group.get("child_call_ids", []):
                provider_by_call[str(child_call_id)] = (str(provider_call_id), origin)
        results_by_call = {result.call.call_id: result for result in tool_results}
        ordered_records: dict[str, dict[str, Any]] = {
            str(call_id): record
            for call_id, record in tool_records.items()
            if isinstance(record, dict) and isinstance(record.get("call"), dict)
        }
        for result in tool_results:
            ordered_records.setdefault(
                result.call.call_id,
                {"call": result.call.model_dump(mode="json"), "status": "completed"},
            )
        calls: list[CallEvidence] = []
        for call_id, record in sorted(
            ordered_records.items(),
            key=lambda item: int(str(item[1]["call"]["step_id"])[1:]),
        ):
            call = record["call"]
            result = results_by_call.get(call_id)
            metadata = canonical.get(call_id, {})
            provider_call_id, group_origin = provider_by_call.get(call_id, (None, "direct"))
            raw_status = str(metadata.get("status") or record.get("status") or "completed")
            if raw_status in {"pending", "running"}:
                raw_status = "cancelled"
            error = record.get("error")
            if hasattr(error, "model_dump"):
                error = error.model_dump(mode="json")
            calls.append(
                CallEvidence(
                    sequence=int(metadata.get("sequence", str(call["step_id"])[1:])),
                    call_id=call_id,
                    provider_call_id=metadata.get("provider_call_id") or provider_call_id,
                    origin=metadata.get(
                        "origin",
                        "system_preflight" if call["name"] == "data_quality" else group_origin,
                    ),
                    work_id=call["work_id"],
                    function=call["name"],
                    function_version=(result.tool_version if result is not None else call["version"]),
                    arguments=call.get("arguments", {}),
                    result_key=result.output.result_key if result is not None else None,
                    value=result.output.value if result is not None else {},
                    result_reference=metadata.get("result") or record.get("result"),
                    data_fingerprint=(
                        result.data_fingerprint if result is not None else plan.data_fingerprint
                    ),
                    output_hash=result.output_hash if result is not None else None,
                    status=raw_status,
                    error=error,
                    started_at=(result.started_at if result is not None else None)
                    or record.get("started_at"),
                    finished_at=(result.finished_at if result is not None else None)
                    or record.get("finished_at"),
                    duration_ms=result.duration_ms if result is not None else None,
                    provider=result.provider if result is not None else None,
                )
            )
        return cls(data_fingerprint=plan.data_fingerprint, calls=calls)

    def successful_calls(self) -> list[CallEvidence]:
        return [item for item in self.calls if item.status in {"completed", "reused"}]

    def calls_for(self, function: str) -> list[CallEvidence]:
        return [item for item in self.calls if item.function == function]

    def only(self, call_id: str) -> CallEvidenceLedger:
        selected = [item for item in self.calls if item.call_id == call_id]
        if not selected:
            raise KeyError(call_id)
        return self.model_copy(update={"calls": selected})

    def analysis_view(
        self,
        *,
        config: StudyConfig,
        research_question: str,
        selected_variables: list[str],
    ) -> dict[str, Any]:
        """Derive the internal evidence view without reading compatibility output."""

        summary: dict[str, Any] = {
            "study": {
                "name": config.study.name,
                "market": config.study.market,
                "timezone": config.study.timezone,
                "frequency": config.study.frequency,
                "target": config.target.name,
                "exogenous": [spec.name for spec in config.exogenous],
            },
            "research_question": research_question,
            "selected_variables": list(dict.fromkeys(selected_variables)),
            "methodology": {
                "missing_value_policy": "pairwise complete for relationships; no implicit imputation",
                "outlier_policy": "retain observations and report robust IQR flags",
                "lag_semantics": "positive lag compares feature[t-lag] with target[t]",
                "causal_claims": False,
            },
            "call_evidence": {},
            "runs_by_function": {},
        }
        merged_functions: set[str] = set()
        for item in self.calls:
            payload = {
                "sequence": item.sequence,
                "call_id": item.call_id,
                "provider_call_id": item.provider_call_id,
                "origin": item.origin,
                "work_id": item.work_id,
                "function": item.function,
                "function_version": item.function_version,
                "arguments": item.arguments,
                "result_key": item.result_key,
                "value": item.value,
                "result_reference": item.result_reference,
                "data_fingerprint": item.data_fingerprint,
                "output_hash": item.output_hash,
                "status": item.status,
            }
            summary["call_evidence"][item.call_id] = payload
            summary["runs_by_function"].setdefault(item.function, []).append(payload)
            if (
                item.status not in {"completed", "reused"}
                or item.result_key == "data_quality"
                or item.function in merged_functions
            ):
                continue
            summary[item.result_key] = _merge_evidence(summary.get(item.result_key), item.value)
            merged_functions.add(item.function)
        return summary

    def artifact_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "data_fingerprint": self.data_fingerprint,
            "call_order": [item.call_id for item in self.calls],
            "calls": {
                item.call_id: item.model_dump(mode="json")
                for item in self.calls
            },
        }
