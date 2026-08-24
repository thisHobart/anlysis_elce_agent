"""OpenAI-compatible implementation of the shared model gateway."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from app.config import Settings, get_settings
from app.llm.gateway import (
    ModelConfigurationError,
    ModelGatewayError,
    ModelMessage,
    ModelResponseError,
    StructuredResult,
)


def _response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(item.get("text", "") for item in content if isinstance(item, dict)).strip()
    return str(content).strip()


def _strip_json_fence(value: str) -> str:
    stripped = value.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()[1:]
    if lines and lines[-1].strip() == "```":
        lines.pop()
    return "\n".join(lines).strip()


class OpenAICompatibleGateway:
    """Create one lazy ChatOpenAI client for all Agent roles."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._model: Any | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.settings.llm_base_url and self.settings.llm_model)

    @property
    def model_name(self) -> str:
        return self.settings.llm_model

    def _get_model(self) -> Any:
        if not self.enabled:
            raise ModelConfigurationError("大模型尚未配置，请先填写 Base URL 和模型名称。")
        if self._model is None:
            try:
                from langchain_openai import ChatOpenAI
            except ImportError as exc:  # pragma: no cover - runtime dependency
                raise ModelConfigurationError("langchain-openai 尚未安装。") from exc
            options: dict[str, Any] = {
                "model": self.settings.llm_model,
                "base_url": self.settings.llm_base_url,
                "timeout": self.settings.llm_timeout_seconds,
                "max_retries": self.settings.llm_max_retries,
                "temperature": 0,
            }
            api_key = self.settings.llm_api_key.get_secret_value().strip()
            if api_key:
                options["api_key"] = api_key
            self._model = ChatOpenAI(**options)
        return self._model

    def invoke_structured(
        self,
        *,
        messages: list[ModelMessage],
        schema: type[StructuredResult],
    ) -> StructuredResult:
        model = self._get_model()
        try:
            if self.settings.llm_structured_mode == "native":
                result = model.with_structured_output(schema, method="function_calling").invoke(messages)
                return result if isinstance(result, schema) else schema.model_validate(result)
            response = model.invoke(messages)
            return schema.model_validate_json(_strip_json_fence(_response_text(response)))
        except (ValidationError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ModelResponseError(f"大模型结构化输出无法解析：{exc}") from exc
        except ModelGatewayError:
            raise
        except Exception as exc:
            raise ModelGatewayError(f"大模型调用失败：{type(exc).__name__}: {exc}") from exc

    def invoke_text(self, *, messages: list[ModelMessage]) -> str:
        try:
            answer = _response_text(self._get_model().invoke(messages))
        except ModelGatewayError:
            raise
        except Exception as exc:
            raise ModelGatewayError(f"大模型调用失败：{type(exc).__name__}: {exc}") from exc
        if not answer:
            raise ModelResponseError("大模型返回了空回复。")
        return answer

