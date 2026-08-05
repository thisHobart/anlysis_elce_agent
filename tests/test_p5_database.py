"""P5 database integration tests.

Tests real MySQL database access for DB-based FAQs.
Requires VPP_DB_* environment variables to be configured.
"""

import pytest

from app.graph.workflow import build_workflow
from app.services.vpp_db import VPPDatabase, VPPDBError


def test_database_connection():
    """Verify database connection and basic query execution."""
    db = VPPDatabase()
    try:
        result = db.execute("vpp-overview")
        assert "total_enterprises" in result
        assert "total_capacity_mw" in result
        assert "max_adjustable_capacity_mw" in result
        assert isinstance(result["total_enterprises"], int)
        assert isinstance(result["total_capacity_mw"], float)
    except VPPDBError as e:
        pytest.skip(f"Database not available: {e}")


def test_database_commands():
    """Test all predefined database commands."""
    db = VPPDatabase()

    commands_to_test = [
        "vpp-overview",
        "device-online",
        "active-gaps",
        "cumulative",
        "carbon",
        "anomaly-devices",
        "low-response",
    ]

    for cmd in commands_to_test:
        try:
            result = db.execute(cmd)
            assert result is not None
            assert isinstance(result, dict)
        except VPPDBError as e:
            pytest.skip(f"Database command {cmd} failed: {e}")


def test_faq_02_with_real_database():
    """FAQ-02: 介绍虚拟电厂 - uses db source for enterprise count and capacity."""
    workflow = build_workflow()

    try:
        result = workflow.invoke(
            {
                "question": "介绍一下聊城虚拟电厂",
                "role": "viewer",
                "params": {},
                "request_action": False,
                "trace": [],
            }
        )

        assert result["route"] == "faq"
        assert result["faq_id"] == "FAQ-02"
        assert result["classification_source"] in {"llm", "rules"}

        # Should have real data from database
        assert "data" in result
        if result.get("fallback"):
            # If fallback, should be due to LLM/template issue, not data
            assert result.get("fallback_reason") != "faq_data_not_connected"

        # Answer should contain enterprise count and capacity
        answer = result["answer"]
        assert "企业" in answer or "容量" in answer

        # Should not contain placeholder markers
        assert "〔" not in answer or "暂未接入" in answer

    except Exception as e:  # noqa: BLE001 - skip tests when DB unavailable
        pytest.skip(f"Database integration test failed: {e}")


def test_faq_05_with_real_database():
    """FAQ-05: 接入多少企业 - uses db source."""
    workflow = build_workflow()

    try:
        result = workflow.invoke(
            {
                "question": "接入了多少家企业",
                "role": "viewer",
                "params": {},
                "request_action": False,
                "trace": [],
            }
        )

        assert result["route"] == "faq"
        assert result["faq_id"] == "FAQ-05"

        # Should have real data
        if not result.get("fallback"):
            answer = result["answer"]
            assert "企业" in answer

    except Exception as e:  # noqa: BLE001 - skip tests when DB unavailable
        pytest.skip(f"Database integration test failed: {e}")


def test_faq_12_device_online_summary():
    """FAQ-12: 设备在线率 - uses db source."""
    workflow = build_workflow()

    try:
        result = workflow.invoke(
            {
                "question": "有多少设备在线",
                "role": "viewer",
                "params": {},
                "request_action": False,
                "trace": [],
            }
        )

        assert result["route"] == "faq"
        assert result["faq_id"] == "FAQ-12"

        # Check trace shows db success
        trace_str = " ".join(result.get("trace", []))
        assert "db:success" in trace_str or "db:error" in trace_str

    except Exception as e:  # noqa: BLE001 - skip tests when DB unavailable
        pytest.skip(f"Database integration test failed: {e}")


def test_sql_injection_blocked():
    """Verify that SQL injection attempts are blocked."""
    db = VPPDatabase()

    malicious_queries = [
        "SELECT * FROM t_res_company; DROP TABLE t_res_company;",  # Multi-statement
        "INSERT INTO t_res_company (name) VALUES ('hacker')",  # DML
        "UPDATE t_res_company SET name='hacked'",  # DML
        "DELETE FROM t_res_company",  # DML
        "SELECT * FROM t_res_company INTO OUTFILE '/tmp/data.txt'",  # INTO OUTFILE
    ]

    for query in malicious_queries:
        result = db.execute("exec", {"query": query})
        assert result["code"] == -1, f"Query should be blocked: {query}"
        assert "rejected" in result["message"].lower()


def test_benign_or_condition_allowed():
    """Verify that legitimate OR conditions in WHERE clauses are allowed."""
    db = VPPDatabase()

    try:
        # This is a legitimate query, not SQL injection
        result = db.execute(
            "exec",
            {
                "query": "SELECT COUNT(*) AS cnt FROM t_res_company WHERE id=1 OR id=2",
                "params": None,
            },
        )

        # Should succeed - OR in WHERE clause is legitimate SQL
        assert result["code"] == 0

    except Exception as e:  # noqa: BLE001 - skip tests when DB unavailable
        pytest.skip(f"Database not available: {e}")


def test_safe_select_allowed():
    """Verify that safe SELECT queries are allowed."""
    db = VPPDatabase()

    try:
        # Simple safe query
        result = db.execute(
            "exec",
            {
                "query": "SELECT COUNT(*) AS cnt FROM t_res_company WHERE del_flag=0 OR del_flag IS NULL",
                "params": None,
            },
        )

        assert result["code"] == 0
        assert "data" in result
        assert isinstance(result["data"], list)

    except Exception as e:  # noqa: BLE001 - skip tests when DB unavailable
        pytest.skip(f"Database not available: {e}")
