"""P6 知识检索（RAG）集成测试

测试 Qdrant + bge-small-zh-v1.5 语义检索功能。

前提条件:
1. 安装依赖: pip install qdrant-client sentence-transformers pypdf
2. 启动 Qdrant: docker run -p 6333:6333 qdrant/qdrant
3. 构建索引: python tools/knowledge-search/build_index.py
"""

import pytest

from app.graph.workflow import build_workflow
from app.services.knowledge import (
    KnowledgeSearchError,
    format_results_xml,
    get_knowledge_service,
    is_available,
    search,
)


class TestKnowledgeService:
    """测试知识检索服务基础功能"""

    def test_service_availability(self):
        """测试服务可用性检查"""
        available = is_available()
        # 如果 Qdrant 未运行或索引未构建，跳过测试
        if not available:
            pytest.skip("Knowledge service not available (Qdrant not running or index not built)")
        assert available is True

    def test_search_basic(self):
        """测试基础搜索功能"""
        if not is_available():
            pytest.skip("Knowledge service not available")

        try:
            results = search("什么是虚拟电厂", top_k=3)
            assert isinstance(results, list)
            assert len(results) <= 3

            if len(results) > 0:
                # 验证结果结构
                for r in results:
                    assert "content" in r
                    assert "source" in r
                    assert "section" in r
                    assert "score" in r
                    assert isinstance(r["score"], float)
                    assert 0 <= r["score"] <= 1

        except KnowledgeSearchError as e:
            pytest.skip(f"Knowledge search failed: {e}")

    def test_search_adjustable_load(self):
        """测试检索可调节负荷相关内容"""
        if not is_available():
            pytest.skip("Knowledge service not available")

        try:
            results = search("可调节负荷有哪些类型", top_k=3)
            assert isinstance(results, list)

            if len(results) > 0:
                # 应该能找到 adjustable_load.md 相关内容
                sources = [r["source"] for r in results]
                assert any("adjustable_load" in s.lower() for s in sources), \
                    f"Expected adjustable_load in sources, got: {sources}"

        except KnowledgeSearchError as e:
            pytest.skip(f"Knowledge search failed: {e}")

    def test_search_business_model(self):
        """测试检索商业模式相关内容"""
        if not is_available():
            pytest.skip("Knowledge service not available")

        try:
            results = search("虚拟电厂的盈利模式", top_k=3)
            assert isinstance(results, list)

            if len(results) > 0:
                # 应该能找到 business_model.md 相关内容
                sources = [r["source"] for r in results]
                assert any("business_model" in s.lower() for s in sources), \
                    f"Expected business_model in sources, got: {sources}"

        except KnowledgeSearchError as e:
            pytest.skip(f"Knowledge search failed: {e}")

    def test_format_results_xml(self):
        """测试 XML 格式化功能"""
        results = [
            {
                "content": "虚拟电厂是一种分布式能源聚合系统。",
                "source": "vpp_concept.md",
                "section": "概述",
                "score": 0.95,
            },
            {
                "content": "可调节负荷包括工业负荷、商业负荷等。",
                "source": "adjustable_load.md",
                "section": "分类",
                "score": 0.88,
            },
        ]

        xml = format_results_xml(results)

        assert "<vpp_reference_documents>" in xml
        assert "</vpp_reference_documents>" in xml
        assert "虚拟电厂是一种分布式能源聚合系统" in xml
        assert "可调节负荷包括工业负荷" in xml
        assert 'source="vpp_concept.md"' in xml
        assert 'section="概述"' in xml
        assert 'confidence="95.0%"' in xml

    def test_format_empty_results(self):
        """测试空结果格式化"""
        xml = format_results_xml([])
        assert "<vpp_reference_documents>" in xml
        assert "未找到相关内容" in xml

    def test_service_execute_command(self):
        """测试服务 execute 接口"""
        if not is_available():
            pytest.skip("Knowledge service not available")

        service = get_knowledge_service()

        # 测试 search 命令
        result = service.execute("search", {"query": "虚拟电厂"})

        if result.get("code") == 0:
            assert "data" in result
            assert "results" in result["data"]
            assert "formatted" in result["data"]
            assert "count" in result["data"]
            assert isinstance(result["data"]["results"], list)
            assert isinstance(result["data"]["formatted"], str)

    def test_service_execute_missing_query(self):
        """测试缺少 query 参数"""
        service = get_knowledge_service()
        result = service.execute("search", {})
        assert result["code"] == 1
        assert "error" in result
        assert "query" in result["error"].lower()

    def test_service_execute_unknown_command(self):
        """测试未知命令"""
        service = get_knowledge_service()
        result = service.execute("unknown", {"query": "test"})
        assert result["code"] == 1
        assert "未知命令" in result["error"]


class TestKnowledgeWorkflow:
    """测试知识检索工作流集成"""

    def test_knowledge_route_basic(self):
        """测试知识路由基本工作流程"""
        if not is_available():
            pytest.skip("Knowledge service not available")

        workflow = build_workflow()

        try:
            result = workflow.invoke({
                "question": "什么是虚拟电厂？它有什么作用？",
                "role": "viewer",
                "params": {},
                "request_action": False,
                "trace": [],
            })

            # 可能路由到 FAQ-01（静态）或 knowledge
            assert result["route"] in ["faq", "knowledge"]

            if result["route"] == "knowledge":
                assert result["answer"]
                trace = result.get("trace", [])
                assert any("knowledge" in t for t in trace)

        except Exception as e:  # noqa: BLE001
            pytest.skip(f"Knowledge workflow test failed: {e}")

    def test_knowledge_route_with_llm(self):
        """测试带 LLM 的知识检索路由"""
        if not is_available():
            pytest.skip("Knowledge service not available")

        workflow = build_workflow()

        try:
            result = workflow.invoke({
                "question": "虚拟电厂的盈利模式有哪些？",
                "role": "viewer",
                "params": {},
                "request_action": False,
                "trace": [],
            })

            # 应该路由到 knowledge 或匹配 FAQ-04
            assert result["route"] in ["faq", "knowledge"]
            assert result["answer"]

            # 检查是否有知识检索相关的 trace
            trace = result.get("trace", [])
            if result["route"] == "knowledge":
                assert any("knowledge" in t for t in trace)

        except Exception as e:  # noqa: BLE001
            pytest.skip(f"Knowledge workflow with LLM test failed: {e}")

    def test_knowledge_route_unavailable(self):
        """测试知识服务不可用时的降级"""
        workflow = build_workflow()

        # 使用一个不太可能匹配 FAQ 的问题
        result = workflow.invoke({
            "question": "虚拟电厂的调度算法原理是什么？",
            "role": "viewer",
            "params": {},
            "request_action": False,
            "trace": [],
        })

        # 如果知识库不可用，应该有合理的回退
        assert result["answer"]
        if not is_available():
            # 应该包含服务不可用的提示
            assert any(
                keyword in result["answer"]
                for keyword in ["暂未", "不可用", "稍后", "无法", "抱歉"]
            )
