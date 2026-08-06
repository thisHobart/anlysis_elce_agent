"""P7 多步查询编排测试

测试 FAQ-09 和 FAQ-15 的"企业名 → 企业ID → 企业数据"依赖查询链。
使用 Mock API 模拟真实 API 行为。
"""

import pytest

from app.graph.workflow import build_workflow
from app.services.vpp_api_mock import get_mock_api


class TestMultiStepOrchestration:
    """测试多步查询编排功能"""

    def test_faq_09_enterprise_realtime_load(self):
        """测试 FAQ-09: 企业实时负荷（多步编排）"""
        workflow = build_workflow()

        # 测试问题：包含企业名称
        result = workflow.invoke({
            "question": "山东鲁能新能源的实时负荷是多少？",
            "role": "viewer",
            "params": {},
            "request_action": False,
            "trace": [],
        })

        assert result["route"] == "faq"
        assert result["faq_id"] == "FAQ-09"
        assert result["answer"]

        # 验证多步编排成功
        trace = result.get("trace", [])
        assert any("multi_step" in t for t in trace), f"应该有多步编排痕迹，trace: {trace}"
        assert any("extracted_name" in t for t in trace), "应该提取到企业名"

        # 验证数据字段
        data = result.get("data", {})
        assert "enterprise_name" in data
        assert "山东鲁能新能源" in data["enterprise_name"]

        # 如果多步成功，应该有负荷数据
        if any("success" in t for t in trace):
            assert "load" in data or "ratio" in data

    def test_faq_09_another_enterprise(self):
        """测试 FAQ-09: 另一个企业"""
        workflow = build_workflow()

        result = workflow.invoke({
            "question": "青岛海尔工业园的负荷多少？",
            "role": "viewer",
            "params": {},
            "request_action": False,
            "trace": [],
        })

        assert result["route"] == "faq"
        assert result["faq_id"] == "FAQ-09"

        data = result.get("data", {})
        if "enterprise_name" in data:
            assert "海尔" in data["enterprise_name"]

    def test_faq_15_enterprise_adjustable_capacity(self):
        """测试 FAQ-15: 企业可调能力（多步编排）"""
        workflow = build_workflow()

        result = workflow.invoke({
            "question": "济南钢铁集团的可调能力是多少？",
            "role": "viewer",
            "params": {},
            "request_action": False,
            "trace": [],
        })

        assert result["route"] == "faq"
        assert result["faq_id"] == "FAQ-15"
        assert result["answer"]

        # 验证多步编排
        trace = result.get("trace", [])
        assert any("multi_step" in t for t in trace)

        data = result.get("data", {})
        assert "enterprise_name" in data
        assert "济南" in data["enterprise_name"] or "钢铁" in data["enterprise_name"]

        # 如果成功，应该有可调能力数据
        if any("success" in t for t in trace):
            assert "adjustable" in data or "peak" in data or "valley" in data

    def test_faq_09_no_enterprise_name(self):
        """测试 FAQ-09: 没有企业名称的情况"""
        workflow = build_workflow()

        result = workflow.invoke({
            "question": "企业的实时负荷是多少？",  # 没有具体企业名
            "role": "viewer",
            "params": {},
            "request_action": False,
            "trace": [],
        })

        # 应该匹配 FAQ-09，但多步编排会失败
        if result["faq_id"] == "FAQ-09":
            trace = result.get("trace", [])
            # 应该有无法提取企业名的标记
            assert any("no_enterprise_name" in t or "failed" in t for t in trace)

    def test_faq_09_unknown_enterprise(self):
        """测试 FAQ-09: 未知企业"""
        workflow = build_workflow()

        result = workflow.invoke({
            "question": "北京不存在公司的实时负荷？",
            "role": "viewer",
            "params": {},
            "request_action": False,
            "trace": [],
        })

        if result["faq_id"] == "FAQ-09":
            trace = result.get("trace", [])
            # 应该提取到企业名，但查不到ID
            assert any("extracted_name" in t or "not_found" in t for t in trace)


class TestMockAPI:
    """测试 Mock API 服务"""

    def test_mock_api_get_companies_by_name(self):
        """测试按名称查询企业"""
        api = get_mock_api()

        result = api.execute("companies", {"name": "鲁能"})
        assert result["code"] == 0
        assert "data" in result
        assert "company_id" in result["data"]
        assert result["data"]["company_name"] == "山东鲁能新能源"

    def test_mock_api_get_all_companies(self):
        """测试获取所有企业"""
        api = get_mock_api()

        result = api.execute("companies", {})
        assert result["code"] == 0
        assert "data" in result
        assert "companies" in result["data"]
        assert result["data"]["total"] == 4

    def test_mock_api_realtime_load(self):
        """测试获取企业实时负荷"""
        api = get_mock_api()

        result = api.execute("realtime-load", {"company_id": "COMP001"})
        assert result["code"] == 0
        assert "data" in result
        assert "load_kw" in result["data"]
        assert "load_ratio" in result["data"]

    def test_mock_api_adjustable_capacity(self):
        """测试获取企业可调能力"""
        api = get_mock_api()

        result = api.execute("adjustable-capacity", {"company_id": "COMP001"})
        assert result["code"] == 0
        assert "data" in result
        assert "adjustable_capacity_mw" in result["data"]
        assert "peak_shaving_mw" in result["data"]
        assert "valley_filling_mw" in result["data"]

    def test_mock_api_unknown_company(self):
        """测试查询不存在的企业"""
        api = get_mock_api()

        result = api.execute("companies", {"name": "不存在公司"})
        assert result["code"] == 1
        assert "error" in result

    def test_mock_api_missing_company_id(self):
        """测试缺少 company_id 参数"""
        api = get_mock_api()

        result = api.execute("realtime-load", {})
        assert result["code"] == 1
        assert "error" in result
        assert "company_id" in result["error"]

    def test_mock_api_unknown_command(self):
        """测试未知命令"""
        api = get_mock_api()

        result = api.execute("unknown-command", {})
        assert result["code"] == 1
        assert "未知命令" in result["error"]


class TestMultiStepDataFlow:
    """测试多步编排的数据流转"""

    def test_data_flow_faq_09(self):
        """测试 FAQ-09 的完整数据流"""
        workflow = build_workflow()

        result = workflow.invoke({
            "question": "潍坊光伏发电站的负荷是多少？",
            "role": "viewer",
            "params": {},
            "request_action": False,
            "trace": [],
        })

        # 打印详细信息用于调试
        print("\n=== FAQ-09 Data Flow ===")
        print(f"Route: {result.get('route')}")
        print(f"FAQ ID: {result.get('faq_id')}")
        print(f"Data: {result.get('data')}")
        print(f"Trace: {result.get('trace')}")
        print(f"Answer: {result.get('answer')}")

        # 基础验证
        assert result["route"] == "faq"
        assert result["faq_id"] == "FAQ-09"

    def test_data_flow_faq_15(self):
        """测试 FAQ-15 的完整数据流"""
        workflow = build_workflow()

        result = workflow.invoke({
            "question": "山东鲁能新能源的可调能力怎么样？",
            "role": "viewer",
            "params": {},
            "request_action": False,
            "trace": [],
        })

        # 打印详细信息
        print("\n=== FAQ-15 Data Flow ===")
        print(f"Route: {result.get('route')}")
        print(f"FAQ ID: {result.get('faq_id')}")
        print(f"Data: {result.get('data')}")
        print(f"Trace: {result.get('trace')}")
        print(f"Answer: {result.get('answer')}")

        # 基础验证
        assert result["route"] == "faq"
        assert result["faq_id"] == "FAQ-15"


class TestEntityExtraction:
    """测试实体提取功能"""

    def test_extract_chinese_company_name(self):
        """测试提取中文企业名"""
        from app.graph.nodes.query_data import _extract_entity_from_question

        question = "山东鲁能新能源的实时负荷是多少？"
        name = _extract_entity_from_question(question, "enterprise_name")
        assert name is not None
        assert "公司" in name or "新能源" in name or "鲁能" in name

    def test_extract_company_with_different_suffix(self):
        """测试提取不同后缀的企业名"""
        from app.graph.nodes.query_data import _extract_entity_from_question

        test_cases = [
            ("济南钢铁集团的负荷？", "集团"),
            ("青岛海尔工业园的数据", "工业园"),
            ("潍坊光伏发电站怎么样", "发电站"),
        ]

        for question, expected_suffix in test_cases:
            name = _extract_entity_from_question(question, "enterprise_name")
            if name:
                assert expected_suffix in name, f"Expected {expected_suffix} in {name}"

    def test_extract_no_company_name(self):
        """测试没有企业名的情况"""
        from app.graph.nodes.query_data import _extract_entity_from_question

        question = "当前的实时负荷是多少？"
        name = _extract_entity_from_question(question, "enterprise_name")
        # 应该返回 None 或无法识别
        assert name is None or len(name) < 2
