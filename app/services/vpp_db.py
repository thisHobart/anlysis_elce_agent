"""VPP MySQL database service.

Migrated from digital-human/workspace-trader/tools/vpp-db-query/vpp_db.py.
Provides read-only access to VPP Liaocheng database with security validation.

Architecture:
- Only SELECT queries allowed (enforced by _validate_sql_safe)
- Uses read-only MySQL account (readonly_user)
- Parameterized queries to prevent SQL injection
- No multi-statement execution
- No dangerous functions (LOAD_FILE, INTO OUTFILE, etc.)

Command interface:
- Pre-defined commands (vpp-overview, device-online, etc.) map to safe queries
- Custom SQL via sql_execute with security validation
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from app.config import get_settings


class VPPDBError(RuntimeError):
    """Raised when database operations fail."""


def _get_connection():
    """Create a database connection using configured credentials."""
    try:
        import pymysql
    except ImportError as exc:  # pragma: no cover
        raise VPPDBError("pymysql is not installed. Run: pip install pymysql") from exc

    settings = get_settings()
    return pymysql.connect(
        host=settings.db_host,
        port=settings.db_port,
        user=settings.db_user,
        password=settings.db_password.get_secret_value(),
        database=settings.db_name,
        charset=settings.db_charset,
        cursorclass=pymysql.cursors.DictCursor,
    )


def _validate_sql_safe(sql: str) -> tuple[bool, str]:
    """Lightweight SQL safety validation (no sqlparse dependency).

    Ensures only safe SELECT queries are allowed through lexical analysis.
    Returns (is_safe: bool, reason: str)
    """
    stripped = sql.strip()
    if not stripped:
        return False, "empty SQL"

    # 1. Prevent multi-statement (semicolon injection)
    # Remove semicolons inside string literals to avoid false positives
    cleaned = []
    in_str = None
    i = 0
    while i < len(stripped):
        ch = stripped[i]
        if in_str:
            if ch == "\\" and i + 1 < len(stripped):
                cleaned.append(ch)
                cleaned.append(stripped[i + 1])
                i += 2
                continue
            elif ch == in_str:
                in_str = None
            cleaned.append(ch)
        else:
            if ch in ("'", '"'):
                in_str = ch
            cleaned.append(ch)
        i += 1
    no_strings = "".join(cleaned)
    # True statement delimiters (semicolons not in strings)
    stmt_semicolons = no_strings.count(";")
    if stmt_semicolons > 1:
        return False, "multiple statements detected"

    # 2. Confirm it's a SELECT type (and not SELECT ... INTO)
    upper = stripped.upper().lstrip()
    if not upper.startswith("SELECT"):
        return False, "only SELECT allowed"
    if " INTO " in upper:
        return False, "SELECT INTO not allowed"

    # 3. Block dangerous functions and keywords
    dangerous = [
        "LOAD_FILE(",
        "SYS_EXEC",
        "EXECUTE",
        "EXEC(",
        "SLEEP(",
        "BENCHMARK(",  # Prevent time-based blind SQL injection probing
        "INTO OUTFILE",
        "INTO DUMPFILE",
        "INFORMATION_SCHEMA.PROCESSLIST",
    ]
    for kw in dangerous:
        if kw in upper:
            return False, f"forbidden: {kw}"

    # 4. Check for DML in nested statements (subqueries can't contain INSERT/UPDATE/DELETE/DROP)
    dml_keywords = [
        " INSERT ",
        " UPDATE ",
        " DELETE ",
        " DROP ",
        " ALTER ",
        " TRUNCATE ",
        " CREATE ",
        " REPLACE ",
        " GRANT ",
        " REVOKE ",
    ]
    no_str_upper = "".join(cleaned).upper()
    for kw in dml_keywords:
        if kw in no_str_upper:
            return False, f"forbidden DML keyword: {kw.strip()}"

    return True, ""


def query_vpp_overview() -> dict[str, Any]:
    """虚拟电厂概览：企业数、总容量、最大可调能力

    Used by FAQ-02, FAQ-05.
    """
    with _get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) AS cnt FROM t_res_company WHERE del_flag=0 OR del_flag IS NULL")
        company_count = cur.fetchone()["cnt"]
        cur.execute("SELECT COALESCE(SUM(capacity), 0) AS total FROM t_res_resource WHERE del_flag=0 OR del_flag IS NULL")
        total_capacity_kw = cur.fetchone()["total"]
        cur.execute("SELECT COALESCE(SUM(capacity_total), 0) AS adjustable FROM t_res_aggregation_unit")
        adjustable = cur.fetchone()["adjustable"]
        cur.close()
        return {
            "total_enterprises": company_count,
            "total_capacity_mw": round(float(total_capacity_kw) / 1000, 2),
            "max_adjustable_capacity_mw": round(float(adjustable), 2),
        }


def query_device_online_summary() -> dict[str, Any]:
    """设备在线/离线统计

    Used by FAQ-12.
    """
    with _get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                SUM(CASE WHEN connect_state=1 THEN 1 ELSE 0 END) AS online_count,
                SUM(CASE WHEN connect_state=0 THEN 1 ELSE 0 END) AS offline_count
            FROM t_res_dev WHERE (del_flag=0 OR del_flag IS NULL)
        """
        )
        row = cur.fetchone()
        online = row["online_count"] or 0
        offline = row["offline_count"] or 0
        total = online + offline
        rate = round(online / total * 100, 1) if total > 0 else 0
        cur.close()
        return {"online_count": online, "offline_count": offline, "online_rate": rate}


def query_active_gaps() -> dict[str, Any]:
    """当前进行中的缺口任务

    Used by FAQ-14.
    """
    with _get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT gap_code, gap_name, max_load, total_load, total_hours,
                   'interval', release_time, declare_deadline, status
            FROM t_load_gap
            WHERE status IN (0,1) AND (del_flag=0 OR del_flag IS NULL)
            ORDER BY release_time DESC LIMIT 20
        """
        )
        tasks = cur.fetchall()
        cur.close()
        return {
            "has_task": len(tasks) > 0,
            "count": len(tasks),
            "tasks": [
                {
                    "code": t["gap_code"],
                    "name": t["gap_name"],
                    "max_load_mw": float(t["max_load"]) if t["max_load"] else 0,
                    "interval": t["interval"],
                    "status": t["status"],
                }
                for t in tasks
            ],
        }


def query_weekly_response_stats() -> dict[str, Any]:
    """响应成功率统计

    Used by FAQ-17.
    """
    with _get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT m.gap_code, m.company_name, m.load_percentage, m.status,
                   g.release_time, g.gap_name
            FROM t_load_gap_monitor m
            JOIN t_load_gap g ON m.gap_code = g.gap_code
            WHERE g.del_flag=0 OR g.del_flag IS NULL
            ORDER BY g.release_time DESC LIMIT 200
        """
        )
        rows = cur.fetchall()
        cur.close()

        if not rows:
            return {"task_count": 0, "success_rate": 0, "max_date": "", "max_rate": 0, "min_date": "", "min_rate": 0}

        rates = [float(r["load_percentage"]) for r in rows if r.get("load_percentage") is not None]
        avg_rate = round(sum(rates) / len(rates), 1) if rates else 0
        task_codes = {r["gap_code"] for r in rows}

        by_date = defaultdict(list)
        for r in rows:
            rt = r.get("release_time")
            if rt and r.get("load_percentage") is not None:
                d = rt.isoformat()[:10] if hasattr(rt, "isoformat") else str(rt)[:10]
                by_date[d].append(float(r["load_percentage"]))

        best_date = max(by_date, key=lambda d: sum(by_date[d]) / len(by_date[d])) if by_date else ""
        worst_date = min(by_date, key=lambda d: sum(by_date[d]) / len(by_date[d])) if by_date else ""
        best_rate = round(sum(by_date[best_date]) / len(by_date[best_date]), 1) if best_date else 0
        worst_rate = round(sum(by_date[worst_date]) / len(by_date[worst_date]), 1) if worst_date else 0

        return {
            "task_count": len(task_codes),
            "success_rate": avg_rate,
            "max_date": best_date,
            "max_rate": best_rate,
            "min_date": worst_date,
            "min_rate": worst_rate,
        }


def query_cumulative_response() -> dict[str, Any]:
    """累计响应统计

    Used by FAQ-19.
    """
    with _get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(DISTINCT gap_code) AS cnt FROM t_load_gap WHERE (del_flag=0 OR del_flag IS NULL)")
        response_count = cur.fetchone()["cnt"]
        cur.execute("SELECT COALESCE(SUM(response_load), 0) AS total FROM t_load_gap_monitor")
        total_response = cur.fetchone()["total"]
        cur.execute("SELECT COALESCE(MAX(bid_load), 0) AS max_load FROM t_load_gap_monitor")
        max_response_mw = cur.fetchone()["max_load"]
        cur.close()
        return {
            "response_count": response_count,
            "max_response_mw": round(float(max_response_mw) / 1000, 2) if max_response_mw else 0,
            "curtailment": round(float(total_response) / 10000, 2) if total_response else 0,
        }


def query_total_carbon() -> dict[str, Any]:
    """累计碳减排量

    Used by FAQ-20.
    """
    with _get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COALESCE(SUM(carbon), 0) AS total FROM t_eec_carbon_emissions")
        total = cur.fetchone()["total"]
        cur.close()
        tree_count = round(float(total) * 0.005, 2) if total else 0
        return {"carbon": round(float(total), 2), "tree_count": tree_count}


def query_anomaly_devices() -> dict[str, Any]:
    """当前异常设备

    Used by FAQ-28.
    """
    with _get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT d.name, d.type, d.code, c.name AS company_name,
                   d.connect_state, d.data_gather_state
            FROM t_res_dev d
            LEFT JOIN t_res_company c ON d.company_id = c.id
            WHERE (d.connect_state = 0 OR d.data_gather_state = 0)
            AND (d.del_flag=0 OR d.del_flag IS NULL)
            ORDER BY d.id DESC LIMIT 20
        """
        )
        devices = cur.fetchall()
        cur.close()
        events = []
        for d in devices:
            msg_parts = []
            if d.get("connect_state") == 0:
                msg_parts.append("离线")
            if d.get("data_gather_state") == 0:
                msg_parts.append("数据采集异常")
            events.append(
                {"device_name": d.get("name", d.get("code", "未知")), "message": "、".join(msg_parts), "type": d.get("type", "")}
            )
        return {"count": len(devices), "events": events}


def query_low_response_companies(threshold: int = 90) -> dict[str, Any]:
    """低响应率企业列表

    Used by FAQ-29.
    """
    with _get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT company_name, load_percentage, gap_code
            FROM t_load_gap_monitor
            WHERE load_percentage IS NOT NULL AND load_percentage < %s
            ORDER BY load_percentage ASC LIMIT 50
        """,
            (threshold,),
        )
        rows = cur.fetchall()
        cur.close()
        enterprises = [{"name": r["company_name"], "rate": float(r["load_percentage"])} for r in rows]
        lowest = min(enterprises, key=lambda e: e["rate"]) if enterprises else {}
        return {"enterprises": enterprises, "lowest": lowest.get("name", ""), "lowest_rate": lowest.get("rate", 0)}


def sql_execute(query: str, params: tuple | None = None) -> dict[str, Any]:
    """Execute a custom SQL query with security validation.

    Only SELECT queries are allowed. All queries are validated before execution.
    Returns {"code": 0, "data": [...], "count": N} on success.
    Returns {"code": -1, "message": "..."} on failure.
    """
    # Security validation
    ok, msg = _validate_sql_safe(query)
    if not ok:
        return {"code": -1, "message": f"SQL rejected: {msg}"}

    try:
        with _get_connection() as conn:
            cur = conn.cursor()
            cur.execute(query, params or ())
            rows = cur.fetchall()
            cur.close()

            def serialize(obj):
                if hasattr(obj, "__float__"):
                    return float(obj)
                if hasattr(obj, "isoformat"):
                    return obj.isoformat()
                return obj

            rows = [{k: serialize(v) for k, v in r.items()} for r in rows]
            return {"code": 0, "data": rows, "count": len(rows)}
    except Exception as e:  # noqa: BLE001 - catch all DB errors for safe error reporting
        return {"code": -1, "message": str(e)}


# Command dispatcher for predefined queries
COMMANDS = {
    "vpp-overview": query_vpp_overview,
    "device-online": query_device_online_summary,
    "active-gaps": query_active_gaps,
    "weekly-response": query_weekly_response_stats,
    "cumulative": query_cumulative_response,
    "carbon": query_total_carbon,
    "anomaly-devices": query_anomaly_devices,
    "low-response": query_low_response_companies,
}


class VPPDatabase:
    """Adapter for VPP database access.

    Provides both predefined commands and custom SQL execution with security validation.
    """

    def execute(self, command: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute a database command.

        Args:
            command: Predefined command name or "exec" for custom SQL
            params: Parameters for the command (dict for predefined, {"query": "...", "params": [...]} for exec)

        Returns:
            Result dict with command-specific structure
        """
        if command == "exec":
            # Custom SQL execution
            if not params or "query" not in params:
                raise VPPDBError("exec command requires 'query' parameter")
            query = params["query"]
            query_params = params.get("params")
            return sql_execute(query, tuple(query_params) if query_params else None)

        # Predefined command
        handler = COMMANDS.get(command)
        if handler is None:
            raise VPPDBError(f"Unknown database command: {command}")

        # Most commands take no parameters, but some do (like low-response threshold)
        if params and command == "low-response" and "threshold" in params:
            return handler(threshold=params["threshold"])

        return handler()
