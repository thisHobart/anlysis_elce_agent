"""Regression tests for the leak-safe Shandong P2 preparation boundary."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime

import pandas as pd

from app.research.news.preparation import (
    MySQLCredentials,
    PreparationError,
    _query_frame,
    apply_search_execution,
    build_p1_summary,
    build_search_plan,
    business_timestamp,
    canonical_url,
    prepare_news_snapshot,
)


def _frame(start: str, values: list[float]) -> pd.DataFrame:
    timestamps = pd.date_range(start, periods=len(values), freq="15min")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "value": values,
            "available_at": timestamps + pd.Timedelta(hours=1),
        }
    )


def test_hour_24_rolls_to_next_midnight_and_rejects_nonzero_minutes():
    assert business_timestamp(date(2026, 1, 1), 24, 0) == datetime.fromisoformat("2026-01-02T00:00:00")
    try:
        business_timestamp(date(2026, 1, 1), 24, 15)
    except ValueError as exc:
        assert "hour=24" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("24:15 must be rejected")


def test_search_plan_is_driven_by_profile_window_and_anomalies():
    base = _frame("2026-01-01 00:15", [100.0] * (96 * 4))
    base.loc[96 * 2 : 96 * 3 - 1, "value"] = 500.0
    weather = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=24 * 4, freq="h"),
            "temperature": range(24 * 4),
            "wind_speed_eighty": [3.0] * (24 * 4),
            "hour_precipitation": [0.0] * (24 * 4),
            "solar_radiation": [100.0] * (24 * 4),
            "available_at": pd.date_range("2026-01-02", periods=24 * 4, freq="h"),
        }
    )
    frames = {
        name: base.copy()
        for name in (
            "real_time_price",
            "day_ahead_price",
            "load",
            "wind",
            "solar",
            "generation",
            "positive_reserve",
            "negative_reserve",
        )
    }
    frames["weather"] = weather
    frames["coal"] = _frame("2026-01-01", [700.0, 710.0]).assign(type_name="index", heat_value="5500")
    frames["maintenance"] = pd.DataFrame(
        {
            "p_date": ["2026-01-03"],
            "bgn_time": ["2026-01-03 01:00"],
            "end_time": ["2026-01-03 04:00"],
            "dev_name": ["Unit A"],
            "rep_type": ["planned"],
            "votage_level": ["500kV"],
            "available_at": ["2026-01-02 12:00"],
        }
    )
    summary = build_p1_summary(frames)
    plan = build_search_plan(summary, as_of=datetime(2026, 1, 5, tzinfo=UTC))

    assert plan["queries"][0]["start_date"] == "2026-01-01"
    assert any(
        item["layer"] == "anomaly_window" and "2026-01-03" in item["trigger"]
        for item in plan["queries"]
    )
    assert len({item["query_id"] for item in plan["queries"]}) == len(plan["queries"])


def test_news_snapshot_preserves_first_seen_and_deduplicates(tmp_path):
    plan_id = "qry_test"
    item = {
        "source_name": "山东权威来源",
        "source_url": "https://example.com/a?utm_source=test",
        "source_type": "government_regulator",
        "title": "山东电力事件",
        "content": "4月9日，某机组恢复并网。",
        "published_at": "2026-04-09T12:00:00+08:00",
        "fetched_at": "2026-09-10T12:00:00+08:00",
        "search_query_id": plan_id,
    }
    source = tmp_path / "input.jsonl"
    source.write_text("\n".join([json.dumps(item, ensure_ascii=False)] * 2), encoding="utf-8")
    result = prepare_news_snapshot(source, output=tmp_path, query_ids={plan_id})
    rows = [
        json.loads(line)
        for line in (tmp_path / "normalized_news.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert result["duplicate_count"] == 1
    assert result["normalized_news_count"] == 1
    assert rows[0]["first_seen_at"] == "2026-09-10T04:00:00Z"
    assert rows[0]["available_at"] == rows[0]["first_seen_at"]
    assert rows[0]["raw_metadata"]["content_scope"] == "excerpt"
    assert result["full_text_count"] == 0
    assert canonical_url(item["source_url"]) == "https://example.com/a"


def test_search_execution_is_not_invented_when_only_results_are_imported(tmp_path):
    plan = {
        "queries": [
            {"query_id": "qry_one"},
            {"query_id": "qry_two"},
        ]
    }
    updated = apply_search_execution(
        plan,
        imported_result_counts={"qry_one": 2},
        import_service="reviewed-import",
    )

    assert updated["queries"][0]["status"] == "results_imported_without_execution_log"
    assert updated["queries"][0]["executed_at"] is None
    assert updated["queries"][0]["imported_result_count"] == 2
    assert updated["queries"][1]["status"] == "not_run"
    assert updated["execution_summary"] == {
        "executed": 0,
        "results_imported_without_execution_log": 1,
        "not_run": 1,
    }


def test_execution_log_requires_timezone_and_known_query(tmp_path):
    log = tmp_path / "executions.jsonl"
    log.write_text(
        json.dumps(
            {
                "query_id": "qry_one",
                "executed_at": "2026-09-10T10:00:00",
                "search_service": "test",
                "returned_result_count": 2,
            }
        ),
        encoding="utf-8",
    )
    try:
        apply_search_execution(
            {"queries": [{"query_id": "qry_one"}]},
            imported_result_counts={},
            execution_log=log,
        )
    except PreparationError as exc:
        assert "timezone" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("naive execution timestamps must be rejected")


def test_credentials_repr_and_query_boundary_do_not_expose_or_modify():
    credentials = MySQLCredentials("db.example", 3306, "reader", "secret", "analytics")
    assert "secret" not in repr(credentials)

    try:
        _query_frame(object(), "DELETE FROM prices")
    except PreparationError as exc:
        assert "SELECT" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("non-SELECT statements must be rejected")
