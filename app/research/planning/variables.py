"""Deterministic variable screening and recommendation policies."""

from __future__ import annotations

from typing import Any, Literal

from app.research.schemas.results import DataQualityReport
from app.research.schemas.study import StudyConfig
from app.research.tools.catalog import FUNCTION_CATALOG

VariableSelectionMode = Literal["explicit", "all_eligible", "auto_recommend"]
VariableSelectionStage = Literal["direct", "screening", "recommended"]

# Screening deliberately uses bounded, interpretable statistics. Expensive
# lagged, grouped, nonlinear and rolling analyses remain deferred until the
# user has seen and approved the evidence-backed recommendation.
VARIABLE_SCREENING_FUNCTIONS: tuple[str, ...] = (
    "exogenous_descriptive_distribution",
    "exogenous_pearson_collinearity",
    "exogenous_stationarity_tests",
    "relationship_scipy_pearson_pairwise",
    "relationship_scipy_spearman_pairwise",
)


def eligible_exogenous_variables(
    config: StudyConfig,
    quality: DataQualityReport | None,
) -> list[str]:
    """Return candidates that satisfy the minimum deterministic data gate."""

    if quality is None:
        return [spec.name for spec in config.exogenous]
    minimum = config.analysis.min_relationship_observations
    return [
        spec.name
        for spec in config.exogenous
        if (report := quality.series.get(spec.name)) is not None
        and report.aligned_non_null_rows >= minimum
        and report.aligned_coverage_rate >= 0.5
    ]


def screening_function_set(
    requested_functions: set[str],
    *,
    allowed_functions: set[str],
) -> tuple[set[str], list[str]]:
    """Build a first-stage plan and retain deeper variable work for approval."""

    price_functions = {
        name
        for name in requested_functions
        if name in FUNCTION_CATALOG and not FUNCTION_CATALOG[name].uses_variables
    }
    requested_variable_functions = {
        name
        for name in requested_functions
        if name in FUNCTION_CATALOG and FUNCTION_CATALOG[name].uses_variables
    }
    screening = {
        name for name in VARIABLE_SCREENING_FUNCTIONS if name in allowed_functions
    }
    deferred = sorted(requested_variable_functions.difference(screening))
    return price_functions.union(screening), deferred


def build_variable_recommendations(
    *,
    selected_variables: list[str],
    quality: DataQualityReport,
    summary: dict[str, Any],
    limit: int = 8,
) -> dict[str, Any]:
    """Rank screened variables from validated quality and relationship evidence."""

    exogenous = summary.get("exogenous") or {}
    profiles = exogenous.get("series") or {}
    relationships = (summary.get("relationships") or {}).get("series") or {}
    redundant_pairs = exogenous.get("strong_collinearity_pairs") or []
    redundant_with: dict[str, str] = {}

    rows: list[dict[str, Any]] = []
    for name in selected_variables:
        quality_row = quality.series.get(name)
        profile = profiles.get(name) or {}
        relation = relationships.get(name) or {}
        contemporaneous = relation.get("contemporaneous") or {}
        coefficients = [
            float(result["correlation"])
            for result in contemporaneous.values()
            if isinstance(result, dict) and isinstance(result.get("correlation"), (int, float))
        ]
        best = max(coefficients, key=abs) if coefficients else None
        coverage = quality_row.aligned_coverage_rate if quality_row is not None else 0.0
        observations = quality_row.aligned_non_null_rows if quality_row is not None else 0
        standard_deviation = profile.get("std")
        availability_type = quality_row.availability_type if quality_row is not None else "unknown"
        prediction_ready = bool(
            availability_type == "known_at_timestamp"
            or (
                availability_type == "forecast"
                and quality_row is not None
                and quality_row.availability_timestamp_present
            )
        )
        exclusion_reason: str | None = None
        if quality_row is None or observations == 0:
            exclusion_reason = "在统一时间轴上没有可用观测。"
        elif coverage < 0.5:
            exclusion_reason = f"覆盖率仅为 {coverage:.1%}，不足以支持稳定筛查。"
        elif standard_deviation is None or float(standard_deviation) == 0:
            exclusion_reason = "变量没有足够变化，无法形成关系证据。"
        if exclusion_reason:
            tier = "excluded"
            reason = exclusion_reason
        elif best is not None and abs(best) >= 0.1:
            tier = "recommended"
            reason = f"覆盖率 {coverage:.1%}，最强同期关系系数为 {best:.3f}。"
        else:
            tier = "exploratory"
            relation_text = "尚未形成可用同期关系" if best is None else f"最强同期关系仅为 {best:.3f}"
            reason = f"覆盖率 {coverage:.1%}，{relation_text}，暂不进入优先深入分析。"
        rows.append(
            {
                "name": name,
                "tier": tier,
                "reason": reason,
                "coverage_rate": coverage,
                "observations": observations,
                "best_contemporaneous_relationship": best,
                "availability_type": availability_type,
                "prediction_ready": prediction_ready,
            }
        )

    by_name = {row["name"]: row for row in rows}
    for pair in redundant_pairs:
        left = by_name.get(str(pair.get("left") or ""))
        right = by_name.get(str(pair.get("right") or ""))
        if not left or not right or left["tier"] == "excluded" or right["tier"] == "excluded":
            continue
        left_score = abs(left["best_contemporaneous_relationship"] or 0.0)
        right_score = abs(right["best_contemporaneous_relationship"] or 0.0)
        loser, winner = (right, left) if left_score >= right_score else (left, right)
        redundant_with[loser["name"]] = winner["name"]
        if loser["tier"] == "recommended":
            loser["tier"] = "exploratory"
            loser["reason"] += f" 与 {winner['name']} 高度重复，优先保留后者。"

    rank = {"recommended": 0, "exploratory": 1, "excluded": 2}
    rows.sort(
        key=lambda row: (
            rank[row["tier"]],
            -abs(row["best_contemporaneous_relationship"] or 0.0),
            -float(row["coverage_rate"]),
            row["name"],
        )
    )
    recommended = [row for row in rows if row["tier"] == "recommended"][: max(1, limit)]
    recommended_names = [row["name"] for row in recommended]
    descriptive_only = [
        row["name"]
        for row in rows
        if row["tier"] != "excluded" and not row["prediction_ready"]
    ]
    return {
        "mode": "auto_recommend",
        "recommended_variables": recommended_names,
        "recommended": recommended,
        "exploratory": [row for row in rows if row["tier"] == "exploratory"],
        "excluded": [row for row in rows if row["tier"] == "excluded"],
        "descriptive_only_variables": descriptive_only,
        "redundant_with": redundant_with,
        "screened_variable_count": len(rows),
        "recommendation_limit": max(1, limit),
    }
