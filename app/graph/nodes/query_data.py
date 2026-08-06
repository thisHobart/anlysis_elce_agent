"""Query Data 节点 - 支持多步查询编排

P7: Multi-step query orchestration
支持依赖查询链，如 "企业名 → 企业ID → 企业数据"
"""

from __future__ import annotations

import re
from typing import Any

from app.graph.state import AgentState
from app.services.faq_service import FAQService
from app.services.knowledge import KnowledgeSearchError, get_knowledge_service, is_available
from app.services.vpp_api_mock import VPPAPIError, get_mock_api
from app.services.vpp_db import VPPDatabase, VPPDBError

FAQ = FAQService()
DB = VPPDatabase()
KNOWLEDGE = get_knowledge_service()
API = get_mock_api()  # P7: Mock API for multi-step orchestration


def _extract_entity_from_question(question: str, entity_type: str) -> str | None:
    """从问题中提取实体（简单正则匹配）"""
    if entity_type == "enterprise_name":
        # 匹配企业名称模式
        patterns = [
            # 带后缀的完整企业名
            r"([一-龥]{2,20}(?:公司|集团|企业|工厂|工业园|发电站|新能源|能源|电力))",
            # 不带后缀但有常见企业关键词（排除通用词，最少4个字）
            r"(?<![的当前实时可调总平台])([一-龥]{4,20})(?=的(?:实时负荷|可调能力|负荷|能力))",
            # 英文字母开头的企业名
            r"([A-Z]{2,}(?:公司|集团)?)",
        ]
        for pattern in patterns:
            match = re.search(pattern, question)
            if match:
                name = match.group(1)
                # 过滤掉明显不是企业名的词
                if name not in ["当前", "实时", "可调", "总", "平台", "企业", "的", "是"]:
                    return name
    return None


def _execute_multi_step_query(
    state: AgentState, entry: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """
    执行多步查询编排

    支持的依赖链:
    1. 企业名 → 企业ID → 企业数据
       - Step 1: 从问题提取企业名
       - Step 2: API/DB 查询企业ID
       - Step 3: 用企业ID查询实时数据

    Returns:
        (data_dict, trace_list)
    """
    question = state.get("question", "")
    faq_id = state.get("faq_id")
    data = {}
    trace = []

    # 检查是否需要多步编排（FAQ-09, FAQ-15）
    if faq_id not in ["FAQ-09", "FAQ-15"]:
        # 不需要多步编排，返回空
        return data, trace

    # Step 1: 提取企业名
    enterprise_name = _extract_entity_from_question(question, "enterprise_name")
    if not enterprise_name:
        trace.append("multi_step:no_enterprise_name_extracted")
        return data, trace

    trace.append(f"multi_step:extracted_name:{enterprise_name}")
    data["enterprise_name"] = enterprise_name

    # Step 2: 查询企业ID（先尝试 API，失败则 DB）
    company_id = None

    # 尝试 API
    try:
        result = API.execute("companies", {"name": enterprise_name})
        if result.get("code") == 0:
            company_id = result["data"]["company_id"]
            trace.append(f"multi_step:api:company_id:{company_id}")
    except VPPAPIError:
        pass

    # API 失败，尝试 DB
    if not company_id:
        try:
            # 使用数据库查询企业ID
            db_result = DB.execute(
                "exec",
                {
                    "query": "SELECT id FROM t_res_company WHERE company_name LIKE %s LIMIT 1",
                    "params": (f"%{enterprise_name}%",),
                },
            )
            if db_result.get("code") == 0 and db_result.get("data"):
                rows = db_result["data"]
                if len(rows) > 0:
                    company_id = rows[0].get("id")
                    trace.append(f"multi_step:db:company_id:{company_id}")
        except VPPDBError:
            pass

    if not company_id:
        trace.append("multi_step:company_id_not_found")
        return data, trace

    # Step 3: 根据 FAQ 类型查询不同数据
    if faq_id == "FAQ-09":
        # 企业实时负荷
        try:
            result = API.execute("realtime-load", {"company_id": company_id})
            if result.get("code") == 0:
                api_data = result["data"]
                data["load"] = api_data.get("load_kw", 0) / 1000  # 转为兆瓦
                data["ratio"] = api_data.get("load_ratio", 0)
                trace.append("multi_step:api:realtime_load:success")
        except VPPAPIError as e:
            trace.append(f"multi_step:api:realtime_load:error:{e}")

    elif faq_id == "FAQ-15":
        # 企业可调能力
        try:
            result = API.execute("adjustable-capacity", {"company_id": company_id})
            if result.get("code") == 0:
                api_data = result["data"]
                data["adjustable"] = api_data.get("adjustable_capacity_mw", 0)
                data["peak"] = api_data.get("peak_shaving_mw", 0)
                data["valley"] = api_data.get("valley_filling_mw", 0)
                data["level"] = api_data.get("supply_level", "未知")
                trace.append("multi_step:api:adjustable_capacity:success")
        except VPPAPIError as e:
            trace.append(f"multi_step:api:adjustable_capacity:error:{e}")

    return data, trace


def query_data(state: AgentState) -> dict:
    """P7 data query: database + knowledge + multi-step orchestration.

    Routes:
    - faq with db source → execute real database queries
    - faq with api source → execute mock API (P7) or multi-step orchestration
    - knowledge → execute RAG search (P6)
    - data → placeholder
    """
    route = state.get("route")
    trace = state.get("trace", [])

    if state.get("validation_errors"):
        return {
            "data": {},
            "fallback": True,
            "fallback_reason": "validation_failed",
            "trace": [*trace, "query:blocked"],
        }

    if route == "faq":
        entry = FAQ.get(state.get("faq_id")) if state.get("faq_id") else None
        if not entry:
            return {"data": {}, "trace": [*trace, "faq:not_found"]}

        # Check if this FAQ needs data
        needs_data = any(
            req.get("source") != "static" and req.get("variable") not in {None, "none"}
            for req in entry.get("data_requirements", [])
        )

        if not needs_data:
            # Static FAQ, no data needed
            return {"data": {}, "trace": [*trace, f"faq:{state.get('faq_id')}:static"]}

        # Determine data source
        data_sources = {req.get("source") for req in entry.get("data_requirements", [])}

        # P7: Multi-step orchestration for FAQ-09, FAQ-15
        if state.get("faq_id") in ["FAQ-09", "FAQ-15"]:
            try:
                data, multi_trace = _execute_multi_step_query(state, entry)
                if data:
                    return {
                        "data": data,
                        "trace": [*trace, *multi_trace, f"faq:{state.get('faq_id')}:multi_step:success"],
                    }
                else:
                    return {
                        "data": {},
                        "fallback": True,
                        "fallback_reason": "multi_step_failed",
                        "trace": [*trace, *multi_trace, f"faq:{state.get('faq_id')}:multi_step:failed"],
                    }
            except Exception as e:
                return {
                    "data": {},
                    "fallback": True,
                    "fallback_reason": "multi_step_error",
                    "trace": [*trace, f"faq:{state.get('faq_id')}:multi_step:error:{e!s}"],
                }

        if "db" in data_sources:
            # Execute real database queries
            try:
                data = {}
                db_reqs = [req for req in entry.get("data_requirements", []) if req.get("source") == "db"]

                for req in db_reqs:
                    # Check if this is a custom SQL query
                    if req.get("db_query"):
                        # Custom SQL execution
                        result = DB.execute("exec", {"query": req["db_query"], "params": None})
                        if result.get("code") == 0 and result.get("data"):
                            # Custom query returns a list of rows
                            # For single-row results (like FAQ-03), extract the first row
                            if len(result["data"]) > 0:
                                # Merge all fields from the first row into data
                                data.update(result["data"][0])
                        else:
                            # Query failed
                            raise VPPDBError(result.get("message", "Custom SQL query failed"))
                    elif req.get("db_command"):
                        # Predefined command execution
                        result = DB.execute(req["db_command"])
                        # Map result to variables expected by the FAQ template
                        data.update(result)

                return {"data": data, "trace": [*trace, f"faq:{state.get('faq_id')}:db:success"]}

            except VPPDBError as e:
                return {
                    "data": {},
                    "fallback": True,
                    "fallback_reason": "db_query_failed",
                    "trace": [*trace, f"faq:{state.get('faq_id')}:db:error:{e!s}"],
                }

        if "api" in data_sources:
            # API data source - try mock API
            try:
                data = {}
                api_reqs = [req for req in entry.get("data_requirements", []) if req.get("source") == "api"]

                for req in api_reqs:
                    vpp_command = req.get("vpp_command")
                    vpp_args = req.get("vpp_args", {})

                    # Map vpp_command to mock API command
                    mock_command = _map_vpp_command_to_mock(vpp_command)
                    if mock_command:
                        result = API.execute(mock_command, vpp_args)
                        if result.get("code") == 0:
                            data.update(result["data"])

                if data:
                    return {"data": data, "trace": [*trace, f"faq:{state.get('faq_id')}:api:success"]}
                else:
                    raise VPPAPIError("No data returned from mock API")

            except VPPAPIError as e:
                return {
                    "data": {},
                    "fallback": True,
                    "fallback_reason": "api_query_failed",
                    "trace": [*trace, f"faq:{state.get('faq_id')}:api:error:{e!s}"],
                }

        # Other sources (knowledge, static)
        return {
            "data": {},
            "fallback": True,
            "fallback_reason": "faq_data_not_connected",
            "trace": [*trace, f"faq:{state.get('faq_id')}:unknown_source"],
        }

    if route == "knowledge":
        # P6: Execute RAG knowledge search
        question = state.get("question", "")
        if not question:
            return {
                "data": {},
                "fallback": True,
                "fallback_reason": "knowledge_no_question",
                "trace": [*trace, "knowledge:no_question"],
            }

        # Check if knowledge service is available
        if not is_available():
            return {
                "data": {},
                "fallback": True,
                "fallback_reason": "knowledge_not_available",
                "trace": [*trace, "knowledge:not_available"],
            }

        try:
            result = KNOWLEDGE.execute("search", {"query": question})
            if result.get("code") == 0:
                # Store formatted XML in data for compose node
                return {
                    "data": {
                        "knowledge_results": result["data"]["results"],
                        "knowledge_formatted": result["data"]["formatted"],
                        "knowledge_count": result["data"]["count"],
                    },
                    "trace": [*trace, f"knowledge:success:{result['data']['count']}_results"],
                }
            else:
                return {
                    "data": {},
                    "fallback": True,
                    "fallback_reason": "knowledge_search_failed",
                    "trace": [*trace, f"knowledge:error:{result.get('error')}"],
                }

        except KnowledgeSearchError as e:
            return {
                "data": {},
                "fallback": True,
                "fallback_reason": "knowledge_search_error",
                "trace": [*trace, f"knowledge:error:{e!s}"],
            }

    if route == "data":
        return {
            "data": {},
            "fallback": True,
            "fallback_reason": "data_query_not_connected",
            "trace": [*trace, "data:placeholder"],
        }

    return {"data": {}, "trace": [*trace, f"query:skipped:{route}"]}


def _map_vpp_command_to_mock(vpp_command: str) -> str | None:
    """将 VPP 命令映射到 Mock API 命令"""
    mapping = {
        "realtime": "realtime-overall",
        "storage_soc": "storage-soc",
        # 可以继续添加更多映射
    }
    return mapping.get(vpp_command)
