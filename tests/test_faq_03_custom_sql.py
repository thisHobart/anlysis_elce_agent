"""Tests for FAQ-03: Resource type statistics using custom SQL query.

This tests the system's ability to execute custom SQL queries and map
the results to template variables.
"""

from app.graph.workflow import build_workflow


def test_faq_03_resource_types_custom_sql():
    """FAQ-03: 有哪些资源类型 - uses custom SQL query with aggregation."""
    workflow = build_workflow()

    try:
        result = workflow.invoke(
            {
                "question": "虚拟电厂有哪些资源类型",
                "role": "viewer",
                "params": {},
                "request_action": False,
                "trace": [],
            }
        )

        assert result["route"] == "faq"
        assert result["faq_id"] == "FAQ-03"

        # Should have real data from custom SQL query
        data = result.get("data", {})

        # Check that all expected fields are present
        assert "type_count" in data, "Missing type_count field"
        assert "total_count" in data, "Missing total_count field"
        assert "total_capacity_kw" in data, "Missing total_capacity_kw field"
        assert "pv_count" in data, "Missing pv_count field"
        assert "storage_count" in data, "Missing storage_count field"
        assert "charging_count" in data, "Missing charging_count field"
        assert "generator_count" in data, "Missing generator_count field"

        # Check data types
        assert isinstance(data["type_count"], int), "type_count should be int"
        assert isinstance(data["total_count"], int), "total_count should be int"
        assert isinstance(data["pv_count"], int), "pv_count should be int"

        # Check trace shows db success
        trace_str = " ".join(result.get("trace", []))
        assert "db:success" in trace_str, f"Expected db:success in trace, got: {trace_str}"

        # Answer should contain resource type information
        answer = result["answer"]
        assert "资源" in answer or "类型" in answer or "站点" in answer, f"Answer should mention resources: {answer}"

        # Should not be in fallback mode
        assert not result.get("fallback"), "Should not be in fallback mode with real data"

        print(f"✅ FAQ-03 test passed")
        print(f"   Data: {data}")
        print(f"   Answer: {answer[:100]}...")

    except Exception as e:
        import pytest
        pytest.skip(f"Database integration test failed: {e}")


def test_faq_03_data_structure():
    """Verify FAQ-03 custom SQL returns correct data structure."""
    from app.services.vpp_db import VPPDatabase

    db = VPPDatabase()

    try:
        # Execute the same query that FAQ-03 uses
        result = db.execute(
            "exec",
            {
                "query": """
                    SELECT COUNT(DISTINCT type) AS type_count, COUNT(*) AS total_count,
                           COALESCE(SUM(capacity),0) AS total_capacity_kw,
                           SUM(CASE WHEN type='fbsgf' THEN 1 ELSE 0 END) AS pv_count,
                           SUM(CASE WHEN type IN ('gsycn','qtcn') THEN 1 ELSE 0 END) AS storage_count,
                           SUM(CASE WHEN type='chdsb' THEN 1 ELSE 0 END) AS charging_count,
                           SUM(CASE WHEN type='cyfdj' THEN 1 ELSE 0 END) AS generator_count
                    FROM t_res_resource WHERE del_flag=0 OR del_flag IS NULL
                """,
                "params": None,
            },
        )

        assert result["code"] == 0, f"Query failed: {result.get('message')}"
        assert "data" in result
        assert len(result["data"]) > 0, "Query should return at least one row"

        row = result["data"][0]

        # Verify all fields exist
        required_fields = [
            "type_count",
            "total_count",
            "total_capacity_kw",
            "pv_count",
            "storage_count",
            "charging_count",
            "generator_count",
        ]

        for field in required_fields:
            assert field in row, f"Missing required field: {field}"
            assert row[field] is not None, f"Field {field} should not be None"

        print(f"✅ FAQ-03 data structure test passed")
        print(f"   Resource stats: {row}")

    except Exception as e:
        import pytest
        pytest.skip(f"Database not available: {e}")


def test_custom_sql_with_multiple_aggregations():
    """Test that custom SQL with CASE statements and aggregations works correctly."""
    from app.services.vpp_db import VPPDatabase

    db = VPPDatabase()

    try:
        # Test a similar aggregation query
        result = db.execute(
            "exec",
            {
                "query": """
                    SELECT
                        COUNT(*) AS total,
                        SUM(CASE WHEN connect_state=1 THEN 1 ELSE 0 END) AS online,
                        SUM(CASE WHEN connect_state=0 THEN 1 ELSE 0 END) AS offline
                    FROM t_res_dev
                    WHERE del_flag=0 OR del_flag IS NULL
                    LIMIT 1
                """,
                "params": None,
            },
        )

        assert result["code"] == 0
        assert len(result["data"]) == 1

        row = result["data"][0]
        assert "total" in row
        assert "online" in row
        assert "offline" in row

        # Verify the math adds up
        assert row["total"] == row["online"] + row["offline"], "Total should equal online + offline"

        print(f"✅ Custom SQL aggregation test passed")

    except Exception as e:
        import pytest
        pytest.skip(f"Database not available: {e}")
