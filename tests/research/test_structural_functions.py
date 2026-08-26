"""Deterministic checks for the statsmodels-backed research functions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.research.application.execution import EDAExecutionService
from app.research.application.planning import prepare_research_data
from app.research.data.snapshot import input_file_manifest, study_fingerprint
from app.research.planning.compiler import EDAPlanCompiler
from app.research.planning.contracts import EDAPlanDraft
from app.research.schemas.study import load_study_config
from app.research.skills.registry import SkillRegistry
from app.research.tools.catalog import FUNCTION_CATALOG
from app.research.tools.eda import dependence, drivers, structure

SEGMENTS = [
    {"segment_id": "peak", "label": "峰段", "kind": "hours", "hours": [9, 10, 11, 18, 19, 20]},
    {"segment_id": "valley", "label": "谷段", "kind": "hours", "hours": [0, 1, 2, 3, 4, 5]},
]


@pytest.fixture
def hourly_price() -> pd.Series:
    rng = np.random.default_rng(20260825)
    index = pd.date_range("2025-01-01", periods=24 * 90, freq="1h", tz="Asia/Shanghai")
    hours = np.arange(len(index))
    values = (
        300
        + 70 * np.sin(2 * np.pi * hours / 24)
        + 25 * np.sin(2 * np.pi * hours / 168)
        + rng.normal(0, 18, len(index))
    )
    values[np.array([100, 900, 901, 902])] = 2400.0
    values[np.array([250, 1500])] = -70.0
    series = pd.Series(values, index=index)
    series.iloc[400:404] = np.nan
    return series


@pytest.fixture
def driver_frame(hourly_price: pd.Series) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    hours = np.arange(len(hourly_price))
    load = 1000 + 180 * np.sin(2 * np.pi * hours / 24) + rng.normal(0, 20, len(hours))
    return pd.DataFrame(
        {
            "price": hourly_price,
            "load": load,
            "load_duplicate": load * 1.002 + rng.normal(0, 0.4, len(hours)),
            "wind": 300 + 120 * np.cos(2 * np.pi * hours / 168) + rng.normal(0, 35, len(hours)),
        },
        index=hourly_price.index,
    )


def test_stationarity_reports_both_tests_and_a_single_verdict(hourly_price: pd.Series):
    result = structure.analyze_stationarity(hourly_price, unit="CNY/MWh", frequency="1h")

    evidence = result["stationarity"]
    assert evidence["level"]["adf"]["test"] == "Augmented Dickey-Fuller"
    assert evidence["level"]["kpss"]["test"] == "KPSS"
    assert evidence["first_difference"]["adf"]["p_value"] is not None
    assert evidence["verdict"] in {"stationary", "unit_root", "trend_or_break_suspected", "inconclusive"}
    assert evidence["recommended_transform"] in {"none", "first_difference", "seasonal_or_structural_review"}
    assert 0 <= evidence["interpolated_share_for_tests"] < 0.05


def test_seasonal_decomposition_finds_the_daily_cycle(hourly_price: pd.Series):
    result = structure.analyze_seasonal_decomposition(hourly_price, unit="CNY/MWh", frequency="1h")

    evidence = result["decomposition"]
    assert evidence["algorithm"] == "MSTL"
    assert evidence["periods"] == {"day": 24, "week": 168}
    assert evidence["seasonal_strength"]["day"] > evidence["seasonal_strength"]["week"]
    assert sum(evidence["variance_share"].values()) == pytest.approx(1.0, abs=1e-3)
    assert evidence["daily_shape"]["peak_hour"] != evidence["daily_shape"]["trough_hour"]


def test_partial_autocorrelation_reports_a_significance_band(hourly_price: pd.Series):
    result = structure.analyze_partial_autocorrelation(
        hourly_price, unit="CNY/MWh", frequency="1h", max_lag=48
    )

    evidence = result["partial_autocorrelation"]
    assert evidence["max_lag"] == 48
    assert len(evidence["series"]) == 48
    assert evidence["significance_band"] > 0
    assert evidence["significant_lag_count"] >= 1
    assert all(row["lag"] >= 1 for row in evidence["ljung_box"])


def test_spike_regime_detects_clustering_and_negative_prices(hourly_price: pd.Series):
    result = structure.analyze_spike_regime(hourly_price, unit="CNY/MWh", spike_iqr_multiplier=3.0)

    evidence = result["spike_regime"]
    assert evidence["high_episodes"]["max_duration_intervals"] >= 3
    assert evidence["negative_share"] > 0
    assert evidence["clustering"]["persistence_ratio"] > 1
    assert evidence["longest_calm_streak_intervals"] > 0


def test_duration_curve_is_monotonically_decreasing(hourly_price: pd.Series):
    result = structure.analyze_duration_curve(hourly_price, unit="CNY/MWh")

    prices = [row["price"] for row in result["duration_curve"]["points"]]
    assert prices == sorted(prices, reverse=True)
    assert 0 < result["duration_curve"]["share_above_zero"] <= 1


def test_naive_baselines_expose_the_error_floor(hourly_price: pd.Series):
    result = structure.analyze_naive_baselines(hourly_price, unit="CNY/MWh", frequency="1h")

    evidence = result["naive_baselines"]
    assert {row["baseline"] for row in evidence["baselines"]} == {
        "persistence",
        "daily_naive",
        "weekly_naive",
    }
    assert evidence["best_mae"] == min(row["mae"] for row in evidence["baselines"])
    assert "MAE" in evidence["error_floor_note"]


def test_variance_stabilization_recommends_asinh_for_heavy_tails(hourly_price: pd.Series):
    result = structure.analyze_variance_stabilization(hourly_price, unit="CNY/MWh")

    evidence = result["variance_stabilization"]
    assert evidence["recommended_transform"] == "asinh_median_mad"
    assert abs(evidence["asinh_transformed"]["excess_kurtosis"]) < abs(
        evidence["raw_standardized"]["excess_kurtosis"]
    )


def test_variance_inflation_flags_a_duplicated_driver(driver_frame: pd.DataFrame):
    result = drivers.analyze_variance_inflation(driver_frame, names=["load", "load_duplicate", "wind"])

    evidence = result["multicollinearity"]
    assert evidence["verdict"] == "severe_redundancy"
    assert set(evidence["severe_variables"]) == {"load", "load_duplicate"}
    assert evidence["variables"][0]["variance_inflation_factor"] > 10


def test_driver_stationarity_covers_every_selected_variable(driver_frame: pd.DataFrame):
    result = drivers.analyze_driver_stationarity(driver_frame, names=["load", "wind"], frequency="1h")

    series = result["driver_stationarity"]["series"]
    assert set(series) == {"load", "wind"}
    assert all("verdict_text" in item for item in series.values())


def test_mutual_information_recovers_a_lagged_dependence():
    rng = np.random.default_rng(5)
    index = pd.date_range("2025-03-01", periods=24 * 60, freq="1h", tz="Asia/Shanghai")
    driver = rng.normal(0, 1, len(index))
    price = np.roll(driver, 3) ** 2 + rng.normal(0, 0.2, len(index))
    frame = pd.DataFrame({"price": price, "driver": driver}, index=index)

    result = dependence.analyze_mutual_information(
        frame, target_name="price", variables=["driver"], max_lag=8, min_observations=48
    )

    evidence = result["mutual_information"]["series"]["driver"]
    assert evidence["best_lag"]["lag"] == 3
    assert evidence["best_lag"]["normalized_mutual_information"] > 0.05
    assert "driver" in result["mutual_information"]["nonlinear_candidates"]


def test_granger_precedence_separates_a_driver_from_noise():
    rng = np.random.default_rng(9)
    index = pd.date_range("2025-03-01", periods=24 * 60, freq="1h", tz="Asia/Shanghai")
    driver = rng.normal(0, 1, len(index))
    price = 0.8 * np.roll(driver, 1) + rng.normal(0, 0.3, len(index))
    frame = pd.DataFrame(
        {"price": price, "driver": driver, "noise": rng.normal(0, 1, len(index))},
        index=index,
    )

    result = dependence.analyze_granger_precedence(
        frame,
        target_name="price",
        variables=["driver", "noise"],
        max_lag=6,
        frequency="1h",
        min_observations=48,
    )

    evidence = result["granger_precedence"]
    assert evidence["variables_with_precedence"] == ["driver"]
    assert evidence["series"]["driver"]["strongest"]["residual_variance_reduction"] > 0.2


def test_rolling_stability_flags_a_relationship_that_flips_sign():
    index = pd.date_range("2025-01-01", periods=24 * 120, freq="1h", tz="Asia/Shanghai")
    rng = np.random.default_rng(4)
    driver = rng.normal(0, 1, len(index))
    half = len(index) // 2
    price = np.concatenate([driver[:half], -driver[half:]]) + rng.normal(0, 0.05, len(index))
    frame = pd.DataFrame({"price": price, "driver": driver}, index=index)

    result = dependence.analyze_rolling_stability(
        frame, target_name="price", variables=["driver"], frequency="1h", min_observations=48
    )

    evidence = result["rolling_stability"]["series"]["driver"]
    assert evidence["stability"] == "sign_unstable"
    assert evidence["sign_flip_share"] > 0
    assert "driver" in result["rolling_stability"]["unstable_variables"]


def test_each_generated_agenda_item_is_judged_by_its_own_rule(
    synthetic_study: Path, tmp_path: Path
):
    """Keyword matchers are an ordered if-chain, so distinct items must stay distinct."""

    from app.research.evaluation.eda import _assess_hypotheses
    from app.research.planning.compiler import FUNCTION_AGENDA_HYPOTHESES

    config = load_study_config(synthetic_study)
    skill = SkillRegistry.default().get("price-exogenous-eda")
    variables = [spec.name for spec in config.exogenous]
    steps = []
    for name, spec in FUNCTION_CATALOG.items():
        if name == "data_quality":
            continue
        parameters: dict[str, object] = {}
        if spec.uses_variables:
            parameters["variables"] = variables
        if spec.uses_max_lag:
            parameters["max_lag"] = 24
        if spec.uses_segments:
            parameters["comparison_id"] = "peak_vs_valley"
            parameters["segments"] = SEGMENTS
        steps.append({"function": name, "enabled": True, "rationale": spec.answers, "parameters": parameters})
    plan = EDAPlanCompiler().compile(
        EDAPlanDraft(objective="议程覆盖回归", selected_variables=variables, steps=steps),
        question="执行全部研究函数",
        config=config,
        skill=skill,
        model_name="regression",
        prompt_version="regression",
    )
    plan = plan.model_copy(
        update={"data_fingerprint": study_fingerprint(config, input_file_manifest(config))}
    )
    summary = EDAExecutionService().execute(
        plan=plan,
        study_config=config,
        output_directory=tmp_path / "agenda-artifacts",
        run_id="agenda-coverage",
    ).eda_summary

    owners: dict[str, list[str]] = {}
    for function_name, hypothesis in FUNCTION_AGENDA_HYPOTHESES.items():
        assessment = _assess_hypotheses(plan.model_copy(update={"hypotheses": [hypothesis]}), summary)[0]
        assert assessment.scope != "needs_restatement", function_name
        owners.setdefault(assessment.item_id, []).append(function_name)

    collisions = {item: names for item, names in owners.items() if len(names) > 1}
    assert not collisions, collisions
    assert len(owners) == len(FUNCTION_AGENDA_HYPOTHESES)


def test_driver_stationarity_is_judged_by_the_drivers_not_the_target(
    synthetic_study: Path, tmp_path: Path
):
    from app.research.evaluation.eda import _assess_hypotheses
    from app.research.planning.compiler import FUNCTION_AGENDA_HYPOTHESES

    config = load_study_config(synthetic_study)
    skill = SkillRegistry.default().get("price-exogenous-eda")
    variables = [spec.name for spec in config.exogenous]
    plan = EDAPlanCompiler().compile(
        EDAPlanDraft(
            objective="驱动平稳性",
            selected_variables=variables,
            steps=[
                {
                    "function": "exogenous_stationarity_tests",
                    "enabled": True,
                    "rationale": "检查驱动自身平稳性。",
                    "parameters": {"variables": variables},
                }
            ],
        ),
        question="驱动变量自己平稳吗",
        config=config,
        skill=skill,
        model_name="regression",
        prompt_version="regression",
    )
    plan = plan.model_copy(
        update={"data_fingerprint": study_fingerprint(config, input_file_manifest(config))}
    )
    summary = EDAExecutionService().execute(
        plan=plan,
        study_config=config,
        output_directory=tmp_path / "driver-stationarity",
        run_id="driver-stationarity",
    ).eda_summary

    hypothesis = FUNCTION_AGENDA_HYPOTHESES["exogenous_stationarity_tests"]
    assessment = _assess_hypotheses(plan.model_copy(update={"hypotheses": [hypothesis]}), summary)[0]

    assert assessment.item_id == "exogenous.stationarity"
    assert "驱动变量" in assessment.evidence


def test_every_registered_function_executes_and_reports_evidence(synthetic_study: Path, tmp_path: Path):
    config = load_study_config(synthetic_study)
    prepared = prepare_research_data(config)
    skill = SkillRegistry.default().get("price-exogenous-eda")
    variables = [spec.name for spec in config.exogenous]

    steps = []
    for name, spec in FUNCTION_CATALOG.items():
        if name == "data_quality":
            continue
        parameters: dict[str, object] = {}
        if spec.uses_variables:
            parameters["variables"] = variables
        if spec.uses_max_lag:
            parameters["max_lag"] = 24
        if spec.uses_segments:
            parameters["comparison_id"] = "peak_vs_valley"
            parameters["segments"] = SEGMENTS
        steps.append({"function": name, "enabled": True, "rationale": spec.answers, "parameters": parameters})

    plan = EDAPlanCompiler().compile(
        EDAPlanDraft(objective="全函数回归", selected_variables=variables, steps=steps),
        question="执行全部研究函数",
        config=config,
        skill=skill,
        model_name="regression",
        prompt_version="regression",
    )
    plan = plan.model_copy(
        update={"data_fingerprint": study_fingerprint(config, input_file_manifest(config))}
    )
    assert len(plan.enabled_steps) == len(FUNCTION_CATALOG)
    del prepared

    result = EDAExecutionService().execute(
        plan=plan,
        study_config=config,
        output_directory=tmp_path / "artifacts",
        run_id="all-functions",
    )

    summary = result.eda_summary
    for spec in FUNCTION_CATALOG.values():
        if spec.evidence_field is None:
            continue
        assert spec.evidence_field in summary[spec.result_key], spec.key
    assert set(summary["comparisons"]) == {"methods", "price", "relationships"}
    assert result.report_path.name == "report.html"
    report = result.report_path.read_text(encoding="utf-8")
    assert "电价自身规律" in report
    assert "朴素基线" in report
    assert json.loads((result.artifact_directory / "eda_summary.json").read_text(encoding="utf-8"))
