from langchain_core.messages import AIMessage

from app.config import Settings
from app.services.llm import LLMService


class StubModel:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)

    def invoke(self, _messages):
        return AIMessage(content=next(self.responses))


def configured_settings() -> Settings:
    return Settings(
        llm_enabled=True,
        llm_base_url="http://local.test/v1",
        llm_api_key="local-placeholder",
        llm_model="local-model",
    )


def test_llm_service_parses_json_prompt_classification():
    service = LLMService(configured_settings())
    service._model = StubModel(
        ['{"intent":"faq","faq_id":"FAQ-01","entities":{"enterprise_name":null}}']
    )

    result = service.classify(question="什么是虚拟电厂", faq_catalog=[], history=[], context={})

    assert result.intent == "faq"
    assert result.faq_id == "FAQ-01"


def test_llm_service_returns_composed_text():
    service = LLMService(configured_settings())
    service._model = StubModel(["这是本地模型组织后的回答。"])

    assert service.compose(payload={"question": "你好"}) == "这是本地模型组织后的回答。"
