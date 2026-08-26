"""OpenAI-compatible implementation of the shared model gateway."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

from pydantic import ValidationError

from app.config import Settings, get_settings
from app.llm.gateway import (
    ModelConfigurationError,
    ModelGatewayError,
    ModelMessage,
    ModelResponseError,
    ModelToolCall,
    StructuredResult,
)


def _response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(item.get("text", "") for item in content if isinstance(item, dict)).strip()
    return str(content).strip()


_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", flags=re.IGNORECASE | re.DOTALL)


def _reasoning_text(response: Any) -> str:
    """Read common provider reasoning fields without retaining the raw response."""

    direct = getattr(response, "reasoning_content", None)
    if direct:
        return str(direct).strip()
    additional = getattr(response, "additional_kwargs", None)
    if isinstance(additional, dict):
        for key in ("reasoning_content", "reasoning"):
            value = additional.get(key)
            if value:
                return str(value).strip()
    content = getattr(response, "content", None)
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            block_type = str(item.get("type", "")).casefold()
            if block_type in {"reasoning", "thinking"}:
                value = item.get("text") or item.get("content")
                if value:
                    return str(value).strip()
    return ""


def _reject_nonempty_thinking(response: Any) -> None:
    """Fail closed when a provider still emits a reasoning trace."""

    if _reasoning_text(response):
        raise ModelResponseError("模型仍返回 reasoning_content，思考模式禁用失败。")
    content = _response_text(response)
    if any(match.group(1).strip() for match in _THINK_BLOCK.finditer(content)):
        raise ModelResponseError("模型仍返回非空 <think> 内容，思考模式禁用失败。")


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

    @property
    def _is_deepseek_endpoint(self) -> bool:
        host = (urlparse(self.settings.llm_base_url).hostname or "").casefold()
        return host == "deepseek.com" or host.endswith(".deepseek.com")

    def _model_options(self) -> dict[str, Any]:
        """Build provider-specific request defaults for the two supported endpoint types."""

        options: dict[str, Any] = {
            "model": self.settings.llm_model,
            "base_url": self.settings.llm_base_url,
            "timeout": self.settings.llm_timeout_seconds,
            "max_retries": self.settings.llm_max_retries,
            "temperature": 0,
        }
        if self._is_deepseek_endpoint:
            options["extra_body"] = {"enable_thinking": False}
        else:
            options["extra_body"] = {
                "enable_thinking": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        api_key = self.settings.llm_api_key.get_secret_value().strip()
        if api_key:
            options["api_key"] = api_key
        return options

    def _get_model(self) -> Any:
        if not self.enabled:
            raise ModelConfigurationError("大模型尚未配置，请先填写 Base URL 和模型名称。")
        if self._model is None:
            try:
                from langchain_openai import ChatOpenAI
            except ImportError as exc:  # pragma: no cover - runtime dependency
                raise ModelConfigurationError("langchain-openai 尚未安装。") from exc
            self._model = ChatOpenAI(**self._model_options())
        return self._model

    def invoke_structured(
        self,
        *,
        messages: list[ModelMessage],
        schema: type[StructuredResult],
    ) -> StructuredResult:
        model = self._get_model()
        try:
            result = model.with_structured_output(
                schema,
                method="function_calling",
                include_raw=True,
            ).invoke(messages)
            if not isinstance(result, dict):
                raise ModelResponseError("大模型结构化输出缺少原始响应，无法检查思考模式。")
            raw = result.get("raw")
            if raw is None:
                raise ModelResponseError("大模型结构化输出缺少 raw，无法检查思考模式。")
            _reject_nonempty_thinking(raw)
            parsing_error = result.get("parsing_error")
            if parsing_error is not None:
                raise ModelResponseError(f"大模型结构化输出无法解析：{parsing_error}")
            parsed = result.get("parsed")
            return parsed if isinstance(parsed, schema) else schema.model_validate(parsed)
        except (ValidationError, ValueError, TypeError) as exc:
            raise ModelResponseError(f"大模型结构化输出无法解析：{exc}") from exc
        except ModelGatewayError:
            raise
        except Exception as exc:
            raise ModelGatewayError(f"大模型调用失败：{type(exc).__name__}: {exc}") from exc

    def invoke_text(self, *, messages: list[ModelMessage]) -> str:
        try:
            response = self._get_model().invoke(messages)
            _reject_nonempty_thinking(response)
            answer = _response_text(response)
        except ModelGatewayError:
            raise
        except Exception as exc:
            raise ModelGatewayError(f"大模型调用失败：{type(exc).__name__}: {exc}") from exc
        if not answer:
            raise ModelResponseError("大模型返回了空回复。")
        return answer

    def invoke_tool_calls(
        self,
        *,
        messages: list[ModelMessage],
        tools: list[dict[str, Any]],
    ) -> list[ModelToolCall]:
        """Return normalized calls only; discard response prose and provider reasoning metadata."""

        if not tools:
            raise ModelConfigurationError("没有可提供给大模型的研究函数。")
        try:
            response = self._get_model().bind_tools(
                tools,
                tool_choice="required",
                parallel_tool_calls=True,
            ).invoke(messages)
            _reject_nonempty_thinking(response)
            raw_calls = getattr(response, "tool_calls", None) or []
            calls = [
                ModelToolCall(
                    name=str(item.get("name", "")),
                    arguments=dict(item.get("args") or item.get("arguments") or {}),
                    call_id=str(item.get("id")) if item.get("id") else None,
                )
                for item in raw_calls
                if isinstance(item, dict)
            ]
        except ModelGatewayError:
            raise
        except Exception as exc:
            raise ModelGatewayError(f"大模型函数选择失败：{type(exc).__name__}: {exc}") from exc
        if not calls or any(not call.name for call in calls):
            raise ModelResponseError("大模型没有返回可执行的研究函数调用。")
        return calls

