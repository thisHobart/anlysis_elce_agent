"""Verify the preferred structured mode uses provider function calling."""

from pydantic import BaseModel

from app.config import Settings
from app.llm.openai_compatible import OpenAICompatibleGateway


class StructuredAnswer(BaseModel):
    value: str


class BoundModel:
    def invoke(self, _messages):
        return {"value": "ok"}


class RecordingModel:
    def __init__(self) -> None:
        self.method = None

    def with_structured_output(self, _schema, *, method):
        self.method = method
        return BoundModel()


def test_native_structured_mode_uses_function_calling():
    gateway = OpenAICompatibleGateway(
        Settings(
            llm_base_url="http://model.invalid/v1",
            llm_model="test-model",
            llm_structured_mode="native",
        )
    )
    model = RecordingModel()
    gateway._model = model

    result = gateway.invoke_structured(messages=[("human", "test")], schema=StructuredAnswer)

    assert result.value == "ok"
    assert model.method == "function_calling"
