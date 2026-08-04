from collections.abc import Callable
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.graph.workflow import build_workflow
from app.schemas.llm import ClassificationResult, ExtractedEntities
from app.services.llm import LLMServiceError


class FakeLLM:
    enabled = True

    def __init__(
        self,
        classifier: Callable[..., ClassificationResult],
        answer: str = "LLM 组织后的回答",
        compose_error: bool = False,
    ) -> None:
        self.classifier = classifier
        self.answer = answer
        self.compose_error = compose_error
        self.classify_calls: list[dict[str, Any]] = []
        self.compose_calls: list[dict[str, Any]] = []

    def classify(self, **kwargs: Any) -> ClassificationResult:
        self.classify_calls.append(kwargs)
        return self.classifier(**kwargs)

    def compose(self, *, payload: dict[str, Any]) -> str:
        self.compose_calls.append(payload)
        if self.compose_error:
            raise LLMServiceError("simulated local-model failure")
        return self.answer


def test_llm_classifies_and_composes_static_faq():
    fake = FakeLLM(
        lambda **_: ClassificationResult(intent="faq", faq_id="FAQ-01"),
        answer="虚拟电厂会聚合分布式资源并参与电网调度。",
    )

    result = build_workflow(llm_service=fake).invoke({"question": "给我解释一下VPP", "role": "viewer"})

    assert result["route"] == "faq"
    assert result["faq_id"] == "FAQ-01"
    assert result["classification_source"] == "llm"
    assert result["compose_source"] == "llm"
    assert result["answer"] == "虚拟电厂会聚合分布式资源并参与电网调度。"
    assert fake.compose_calls[0]["faq_template"]


def test_unknown_llm_faq_falls_back_to_p2_rules():
    fake = FakeLLM(lambda **_: ClassificationResult(intent="faq", faq_id="FAQ-99"))

    result = build_workflow(llm_service=fake).invoke({"question": "什么是虚拟电厂", "role": "viewer"})

    assert result["faq_id"] == "FAQ-01"
    assert result["classification_source"] == "rules"
    assert result["fallback"] is True
    assert result["fallback_reason"] == "classification_llm_error"


def test_compose_failure_uses_existing_faq_template():
    fake = FakeLLM(
        lambda **_: ClassificationResult(intent="faq", faq_id="FAQ-01"),
        compose_error=True,
    )

    result = build_workflow(llm_service=fake).invoke({"question": "什么是虚拟电厂", "role": "viewer"})

    assert result["compose_source"] == "template"
    assert "虚拟电厂" in result["answer"]
    assert "composed:template:fallback" in result["trace"]


@pytest.mark.parametrize(
    ("intent", "expected_route"),
    [
        ("knowledge_query", "knowledge"),
        ("data_query", "data"),
        ("screen_action", "screen_action"),
        ("chitchat", "chitchat"),
        ("out_of_scope", "out_of_scope"),
    ],
)
def test_expanded_llm_routes(intent, expected_route):
    fake = FakeLLM(lambda **_: ClassificationResult(intent=intent))

    result = build_workflow(llm_service=fake).invoke({"question": "测试问题", "role": "viewer"})

    assert result["route"] == expected_route
    assert result["classification_source"] == "llm"


def test_same_thread_preserves_multi_turn_context_and_messages():
    def classify_with_context(*, question: str, context: dict[str, Any], **_: Any) -> ClassificationResult:
        if "华能" in question:
            return ClassificationResult(
                intent="data_query",
                entities=ExtractedEntities(enterprise_name="华能公司"),
            )
        assert context["enterprise_name"] == "华能公司"
        return ClassificationResult(
            intent="data_query",
            entities=ExtractedEntities(time_range="最近一周"),
        )

    fake = FakeLLM(classify_with_context)
    workflow = build_workflow(llm_service=fake, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "session-001"}}

    workflow.invoke({"question": "查询华能公司的响应情况", "role": "viewer", "trace": []}, config=config)
    result = workflow.invoke({"question": "那它最近一周呢？", "role": "viewer", "trace": []}, config=config)

    assert result["context"] == {"enterprise_name": "华能公司", "time_range": "最近一周"}
    assert len(result["messages"]) == 4
    assert result["trace"][0] == "turn:prepared"


def test_different_threads_are_isolated():
    def classifier(*, context: dict[str, Any], **_: Any) -> ClassificationResult:
        assert context == {}
        return ClassificationResult(intent="chitchat")

    workflow = build_workflow(llm_service=FakeLLM(classifier), checkpointer=InMemorySaver())

    workflow.invoke(
        {"question": "你好", "role": "viewer", "trace": []},
        config={"configurable": {"thread_id": "session-a"}},
    )
    result = workflow.invoke(
        {"question": "你好", "role": "viewer", "trace": []},
        config={"configurable": {"thread_id": "session-b"}},
    )

    assert len(result["messages"]) == 2
