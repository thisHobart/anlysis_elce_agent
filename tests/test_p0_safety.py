"""P0 safety tests: verify that LLM never hallucinates when data sources are not connected.

For a VPP production system, generating plausible-sounding but fabricated business data
is unacceptable. These tests ensure that when knowledge/data/faq_data are marked as
not connected, the system returns explicit "not connected" messages rather than allowing
the LLM to answer from its general knowledge.
"""


from app.graph.workflow import build_workflow
from app.schemas.llm import ClassificationResult


class FakeLLM:
    """A fake LLM that would hallucinate if given the chance."""

    enabled = True

    def __init__(self):
        self.compose_calls = []

    def classify(self, *, question: str, faq_catalog: list, history: list, context: dict):
        # Simulate LLM classifying questions
        # "什么是虚拟电厂" should match FAQ-01
        if "什么是虚拟电厂" in question:
            return ClassificationResult(intent="faq", faq_id="FAQ-01")
        # "工作原理" is knowledge query (not in FAQ)
        if "工作原理" in question or "vpp" in question.lower():
            return ClassificationResult(intent="knowledge_query", faq_id=None)
        if "功率" in question or "发电" in question:
            return ClassificationResult(intent="data_query", faq_id=None)
        if "天气" in question:
            return ClassificationResult(intent="out_of_scope", faq_id=None)
        return ClassificationResult(intent="chitchat", faq_id=None)

    def compose(self, *, payload: dict) -> str:
        # Record this call for verification
        self.compose_calls.append(payload)

        # This should NEVER be called when fallback_reason indicates not_connected
        fallback_reason = payload.get("fallback_reason")

        # If we reach here with a not_connected reason, return a hallucination marker
        if fallback_reason in {"knowledge_not_connected", "data_query_not_connected", "faq_data_not_connected"}:
            return "[HALLUCINATION] 虚拟电厂是一个智能电力管理系统，通过聚合分布式能源资源..."

        question = payload.get("question", "")
        route = payload.get("route", "")

        if "vpp" in question.lower():
            return "虚拟电厂（VPP）是一种创新的电力管理模式。"
        if route == "chitchat":
            return "您好！很高兴为您服务。"
        return "这是 LLM 生成的回答。"


def test_knowledge_not_connected_blocks_llm():
    """When knowledge base is not connected, must return fixed message, not LLM hallucination."""
    fake_llm = FakeLLM()
    workflow = build_workflow(llm_service=fake_llm)

    result = workflow.invoke(
        {
            "question": "请解释一下 VPP 的工作原理",
            "role": "viewer",
            "params": {},
            "request_action": False,
            "trace": [],
        }
    )

    assert result["route"] == "knowledge"
    assert result["fallback_reason"] == "knowledge_not_connected"
    assert result["compose_source"] == "fixed"
    assert "知识库服务暂未接入" in result["answer"]
    assert "[HALLUCINATION]" not in result["answer"]
    assert "composed:fixed:forced" in result["trace"]
    # Verify LLM compose was NOT called
    assert len(fake_llm.compose_calls) == 0


def test_data_query_not_connected_blocks_llm():
    """When data query is not connected, must return fixed message, not fabricated numbers."""
    fake_llm = FakeLLM()
    workflow = build_workflow(llm_service=fake_llm)

    result = workflow.invoke(
        {
            "question": "当前总发电功率是多少",
            "role": "viewer",
            "params": {},
            "request_action": False,
            "trace": [],
        }
    )

    assert result["route"] == "data"
    assert result["fallback_reason"] == "data_query_not_connected"
    assert result["compose_source"] == "fixed"
    assert "实时数据查询暂未接入" in result["answer"]
    assert "[HALLUCINATION]" not in result["answer"]
    assert "composed:fixed:forced" in result["trace"]
    # Verify LLM compose was NOT called
    assert len(fake_llm.compose_calls) == 0


def test_faq_needing_data_not_connected_blocks_llm():
    """When FAQ needs real-time data that's not available, must not fabricate numbers."""
    from app.graph.nodes.compose import compose
    from app.graph.state import AgentState

    fake_llm = FakeLLM()

    # Manually construct state as if FAQ-06 was matched but data is not connected
    state: AgentState = {
        "question": "当前发电功率",
        "role": "viewer",
        "params": {},
        "request_action": False,
        "messages": [],
        "context": {},
        "intent": "faq",
        "route": "faq",
        "faq_id": "FAQ-06",
        "entities": {},
        "classification_source": "llm",
        "compose_source": "fixed",
        "fallback": True,
        "fallback_reason": "faq_data_not_connected",
        "tool_name": None,
        "validation_errors": [],
        "data": {},
        "answer": "",
        "screen_action": None,
        "trace": ["faq:FAQ-06"],
    }

    result = compose(state, llm_service=fake_llm)

    assert result["compose_source"] == "fixed"
    assert "FAQ-06" in result["answer"] or "数据源暂未接入" in result["answer"]
    assert "[HALLUCINATION]" not in result["answer"]
    # Should not contain fabricated numbers
    assert "兆瓦" not in result["answer"] or "暂未接入" in result["answer"]
    # Verify LLM compose was NOT called
    assert len(fake_llm.compose_calls) == 0


def test_static_faq_still_uses_llm():
    """Static FAQ (no data requirements) should still use LLM for natural composition."""
    fake_llm = FakeLLM()
    workflow = build_workflow(llm_service=fake_llm)

    result = workflow.invoke(
        {
            "question": "什么是虚拟电厂",
            "role": "viewer",
            "params": {},
            "request_action": False,
            "trace": [],
        }
    )

    assert result["route"] == "faq"
    assert result["faq_id"] == "FAQ-01"
    # Static FAQ has no data requirements, so no not_connected fallback
    assert result.get("fallback_reason") not in {
        "knowledge_not_connected",
        "data_query_not_connected",
        "faq_data_not_connected",
    }
    # Should use LLM for natural composition
    assert result["compose_source"] in {"llm", "template"}
    # LLM compose SHOULD have been called for static FAQ
    if result["compose_source"] == "llm":
        assert len(fake_llm.compose_calls) > 0


def test_chitchat_uses_llm_normally():
    """Chitchat route should use LLM normally since it doesn't involve VPP data."""
    fake_llm = FakeLLM()
    workflow = build_workflow(llm_service=fake_llm)

    result = workflow.invoke(
        {
            "question": "你好",
            "role": "viewer",
            "params": {},
            "request_action": False,
            "trace": [],
        }
    )

    assert result["route"] == "chitchat"
    assert result["compose_source"] in {"llm", "fixed"}
    # Chitchat doesn't have not_connected issue
    assert result.get("fallback_reason") not in {
        "knowledge_not_connected",
        "data_query_not_connected",
        "faq_data_not_connected",
    }


def test_out_of_scope_returns_fixed_message():
    """Out of scope questions should return fixed rejection message."""
    fake_llm = FakeLLM()
    workflow = build_workflow(llm_service=fake_llm)

    result = workflow.invoke(
        {
            "question": "今天天气怎么样",
            "role": "viewer",
            "params": {},
            "request_action": False,
            "trace": [],
        }
    )

    assert result["route"] == "out_of_scope"
    assert "只能协助处理虚拟电厂相关问题" in result["answer"]
    assert result["compose_source"] == "fixed"
