"""Compile model-visible arguments into trusted deterministic tool arguments."""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.research.agent.errors import ResearchPlanValidationError
from app.research.schemas.study import StudyConfig
from app.research.tools.catalog import FUNCTION_CATALOG


def _intervals_per_hour(frequency: str) -> float:
    offset = pd.tseries.frequencies.to_offset(frequency)
    seconds = offset.nanos / 1_000_000_000
    return 3600 / seconds


def max_lag_limit(frequency: str, *, days: int = 31) -> int:
    """Convert a duration safety limit to canonical intervals."""

    return max(1, round(days * 24 * _intervals_per_hour(frequency)))


def _parameter_int(parameters: dict[str, Any], key: str, default: int) -> int:
    value = parameters.get(key, default)
    if isinstance(value, bool):
        raise ResearchPlanValidationError(f"{key} 必须是整数")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ResearchPlanValidationError(f"{key} 必须是整数") from exc


def compile_function_parameters(
    function_name: str,
    parameters: dict[str, Any],
    *,
    selected_variables: list[str],
    config: StudyConfig,
    enabled: bool,
) -> dict[str, Any]:
    """Fill trusted arguments and reject fields unrelated to one atomic function."""

    spec = FUNCTION_CATALOG[function_name]
    supplied = dict(parameters)
    compiled: dict[str, Any] = {}

    if spec.uses_variables:
        requested_variables = supplied.pop("variables", selected_variables)
        if requested_variables != selected_variables:
            raise ResearchPlanValidationError(f"{function_name}.variables 必须与方案所选变量一致")
        if enabled and len(selected_variables) < spec.min_variables:
            raise ResearchPlanValidationError(
                f"模型启用了 {function_name}，但所选外生变量少于 {spec.min_variables} 个"
            )
        compiled["variables"] = selected_variables
    elif "variables" in supplied:
        raise ResearchPlanValidationError(f"{function_name} 不接受 variables 参数")

    if spec.uses_max_lag:
        maximum = max_lag_limit(config.study.frequency)
        max_lag = _parameter_int(supplied, "max_lag", config.analysis.max_lag)
        supplied.pop("max_lag", None)
        supplied.pop("max_lag_limit", None)
        if not 0 <= max_lag <= maximum:
            raise ResearchPlanValidationError(f"{function_name}.max_lag 必须在 0 到 {maximum} 之间")
        compiled.update({"max_lag": max_lag, "max_lag_limit": maximum})
    elif "max_lag" in supplied or "max_lag_limit" in supplied:
        raise ResearchPlanValidationError(f"{function_name} 不接受 max_lag 参数")

    if spec.uses_segments:
        comparison_id = supplied.pop("comparison_id", None)
        segments = supplied.pop("segments", None)
        if not comparison_id or not segments:
            raise ResearchPlanValidationError(f"{function_name} 必须提供 comparison_id 和至少两个 segments")
        compiled["comparison_id"] = comparison_id
        compiled["segments"] = segments
    elif "comparison_id" in supplied or "segments" in supplied:
        raise ResearchPlanValidationError(f"{function_name} 不接受分段参数")

    if spec.uses_spike_multiplier:
        supplied.pop("spike_iqr_multiplier", None)
        compiled["spike_iqr_multiplier"] = config.analysis.spike_iqr_multiplier
    if spec.uses_outlier_multiplier:
        supplied.pop("outlier_iqr_multiplier", None)
        compiled["outlier_iqr_multiplier"] = config.analysis.outlier_iqr_multiplier
    if spec.category == "relationship":
        supplied.pop("min_observations", None)
        compiled["min_observations"] = config.analysis.min_relationship_observations

    if supplied:
        raise ResearchPlanValidationError(
            f"{function_name} 包含不支持的参数：{', '.join(sorted(supplied))}"
        )
    return compiled
