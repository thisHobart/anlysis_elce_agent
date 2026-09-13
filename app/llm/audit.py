"""Prompt-free JSONL audit records for model capacity and recovery decisions."""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.llm.budget import ModelRequestPurpose, RequestBudget
from app.llm.context_safety import response_finish_reason
from app.runtime_paths import model_call_audit_path


def response_usage(response: Any) -> tuple[int | None, int | None]:
    candidates = [
        getattr(response, "usage_metadata", None),
        getattr(response, "response_metadata", None),
    ]
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        nested = candidate.get("token_usage") or candidate.get("usage")
        values = nested if isinstance(nested, dict) else candidate
        input_tokens = values.get("input_tokens", values.get("prompt_tokens"))
        output_tokens = values.get("output_tokens", values.get("completion_tokens"))
        if input_tokens is not None or output_tokens is not None:
            return (
                int(input_tokens) if input_tokens is not None else None,
                int(output_tokens) if output_tokens is not None else None,
            )
    return None, None


class ModelCallAudit:
    _lock = threading.Lock()

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else model_call_audit_path()

    def record(
        self,
        budget: RequestBudget,
        *,
        response: Any | None = None,
        outcome: str,
        error_type: str | None = None,
        recovery: str | None = None,
    ) -> None:
        actual_input, actual_output = response_usage(response)
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "purpose": budget.purpose.value,
            "route_key": budget.route_key,
            "profile_source": budget.profile_source,
            "profile_verified": budget.profile_verified,
            "estimated_input_tokens": budget.input_tokens,
            "actual_input_tokens": actual_input,
            "target_output_tokens": budget.target_output_tokens,
            "effective_output_tokens": budget.effective_output_tokens,
            "actual_output_tokens": actual_output,
            "finish_reason": response_finish_reason(response) if response is not None else "",
            "compaction_level": budget.compaction_level,
            "outcome": outcome,
            "error_type": error_type,
            "recovery": recovery
            or (
                "minimal_eda_plan_intent"
                if budget.purpose == ModelRequestPurpose.EDA_PLANNING_RECOVERY
                else None
            ),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._lock, self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(line)
