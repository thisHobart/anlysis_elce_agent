"""Regression checks for the real Phase 1 acceptance script."""

from scripts.validate_phase1 import enabled_function_names


def test_acceptance_evidence_reads_current_function_field() -> None:
    plan = {
        "steps": [
            {"function": "data_quality", "tool": "legacy-wrong-name", "enabled": True},
            {"function": "price_descriptive_distribution", "enabled": True},
            {"function": "price_lag_autocorrelation", "enabled": False},
        ]
    }

    assert enabled_function_names(plan) == [
        "data_quality",
        "price_descriptive_distribution",
    ]
