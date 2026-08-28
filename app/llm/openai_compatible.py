"""Research-model protocol implementation for Chat Completions and Responses."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from app.config import Settings, get_settings
from app.llm import compat
from app.llm.gateway import (
    ModelConfigurationError,
    ModelGatewayError,
    ModelMessage,
    ModelProtocolError,
    ModelResponseError,
    ModelThinkingError,
    ModelToolCall,
    StructuredResult,
)


def _response_text(response: Any) -> str:
    """Return visible text blocks without treating reasoning blocks as answers."""

    content = getattr(response, "content", response)
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            block_type = str(item.get("type", "")).casefold()
            if block_type in {"reasoning", "thinking"}:
                continue
            value = item.get("text") or item.get("content")
            if value:
                parts.append(str(value))
        return "".join(parts).strip()
    return str(content).strip()


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
        raise ModelThinkingError("模型仍返回 reasoning_content，思考模式禁用失败。")
    if compat.thinking_fragments(_response_text(response)):
        raise ModelThinkingError("模型仍返回非空 <think> 内容，思考模式禁用失败。")


def _visible_text(response: Any) -> str:
    """Return the answer with any inline reasoning trace removed."""

    return compat.strip_thinking(_response_text(response))


def _normalized_calls(raw_calls: Any) -> list[ModelToolCall]:
    """Normalize native function calls while preserving their provider call id."""

    return [
        ModelToolCall(
            name=str(item.get("name", "")),
            arguments=item.get("args") or item.get("arguments") or {},
            call_id=str(item.get("id")) if item.get("id") else None,
        )
        for item in (raw_calls or [])
        if isinstance(item, dict)
    ]


def _tool_names(tools: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for tool in tools:
        function = tool.get("function", tool)
        if isinstance(function, dict) and function.get("name"):
            names.add(str(function["name"]))
    return names


def _transport_messages(messages: list[ModelMessage]) -> list[Any]:
    """Convert the Agent's canonical roles to LangChain's typed messages.

    ChatOpenAI owns the final Chat Completions or Responses wire encoding. Tuples
    and provider-shaped dictionaries are rejected here so API dialect details do
    not leak back into the research Agent.
    """

    try:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
    except ImportError as exc:  # pragma: no cover - installed with langchain-openai
        raise ModelConfigurationError("langchain-core 尚未安装。") from exc

    converted: list[Any] = []
    for message in messages:
        if not isinstance(message, ModelMessage):
            raise ModelConfigurationError("研究 Agent 消息必须使用 ModelMessage 协议。")
        if message.role == "system":
            converted.append(SystemMessage(content=message.content))
        elif message.role == "user":
            converted.append(HumanMessage(content=message.content))
        elif message.role == "assistant":
            converted.append(AIMessage(content=message.content))
        else:
            converted.append(
                ToolMessage(content=message.content, tool_call_id=message.tool_call_id or "")
            )
    return converted


class ResearchModelGateway:
    """Enforce the native model protocol required by the research Agent.

    Canonical messages, Pydantic schemas, and research functions are translated
    through the explicitly selected Chat Completions or Responses API. Core
    capabilities fail closed; only optional request controls may be removed when
    an endpoint explicitly rejects them.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._model: Any | None = None
        self._http_client: Any | None = None
        self._disabled: set[str] = set()

    @property
    def enabled(self) -> bool:
        return bool(self.settings.llm_base_url and self.settings.llm_model)

    @property
    def model_name(self) -> str:
        return self.settings.llm_model

    def _reasoning_options(self) -> dict[str, Any]:
        """Map one semantic effort setting onto the selected provider/API dialect."""

        if "provider_reasoning" in self._disabled:
            return {}
        provider = self.settings.llm_provider
        style = self.settings.llm_api_style
        configured_effort = self.settings.llm_reasoning_effort.strip().casefold()
        effort = configured_effort or ("none" if provider in {"deepseek", "qwen"} else "")
        if not effort:
            return {}
        if style == "responses":
            return {"reasoning": {"effort": effort}}
        if provider == "deepseek":
            options: dict[str, Any] = {
                "extra_body": {
                    "thinking": {"type": "disabled" if effort == "none" else "enabled"}
                }
            }
            if effort != "none":
                options["reasoning_effort"] = effort
            return options
        if provider == "qwen":
            options = {"extra_body": {"enable_thinking": effort != "none"}}
            if effort != "none":
                options["reasoning_effort"] = effort
            return options
        return {"reasoning_effort": effort}

    def _degradable_client_features(self) -> frozenset[str]:
        """Only auto-added provider controls may degrade; explicit custom settings may not."""

        features = {"temperature"}
        if self.settings.llm_provider != "custom":
            features.add("provider_reasoning")
        return frozenset(features)

    def _model_options(self) -> dict[str, Any]:
        """Build defaults for the explicitly selected API, minus optional refused fields."""

        options: dict[str, Any] = {
            "model": self.settings.llm_model,
            "base_url": self.settings.llm_base_url,
            "timeout": self.settings.llm_timeout_seconds,
            "max_retries": self.settings.llm_max_retries,
            "use_responses_api": self.settings.llm_api_style == "responses",
        }
        if "temperature" not in self._disabled:
            options["temperature"] = 0
        options.update(self._reasoning_options())
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
            options = self._model_options()
            if self._http_client is None:
                self._http_client = compat.build_compatible_http_client()
            if self._http_client is not None:
                options["http_client"] = self._http_client
            try:
                self._model = ChatOpenAI(**options)
            except Exception as exc:
                raise ModelConfigurationError(f"大模型客户端配置无效：{exc}") from exc
        return self._model

    def _disable(self, feature: str) -> None:
        """Drop one refused optional field and rebuild the client when needed."""

        self._disabled.add(feature)
        if feature in compat.CLIENT_LEVEL_FEATURES:
            self._model = None

    def _degrade(self, call: Callable[[], Any], *, allowed: frozenset[str]) -> Any:
        """Retry only after an endpoint rejects an explicitly allowed optional field."""

        for _ in range(len(compat.UNSUPPORTED_HINTS) + 1):
            try:
                return call()
            except ModelGatewayError:
                raise
            except Exception as exc:
                feature = compat.unsupported_feature(exc, self._disabled, allowed)
                if feature is None:
                    raise
                self._disable(feature)
        raise ModelResponseError("大模型端点拒绝了全部可选请求参数组合。")

    def _protocol_error(self, capability: str, detail: object) -> ModelProtocolError:
        profile = (
            f"Provider={self.settings.llm_provider}, "
            f"API Style={self.settings.llm_api_style}, "
            f"model={self.settings.llm_model}"
        )
        return ModelProtocolError(
            f"当前模型端点不支持研究 Agent 必需的原生{capability}协议（{profile}）：{detail}。"
            "请确认 API 形式和模型能力；系统不会用提示词 JSON 模拟该协议。"
        )

    def invoke_structured(
        self,
        *,
        messages: list[ModelMessage],
        schema: type[StructuredResult],
    ) -> StructuredResult:
        """Request one schema-validated native function call; never parse prose as JSON."""

        prepared = _transport_messages(messages)
        try:
            result = self._degrade(
                lambda: self._get_model()
                .with_structured_output(schema, method="function_calling", include_raw=True)
                .invoke(prepared),
                allowed=self._degradable_client_features(),
            )
            if not isinstance(result, dict):
                raise ModelResponseError("大模型结构化输出缺少原始响应，无法验证协议。")
            raw = result.get("raw")
            if raw is None:
                raise ModelResponseError("大模型结构化输出缺少 raw，无法验证协议。")
            self._guard_thinking(raw)
            parsed = result.get("parsed")
            if parsed is not None:
                return parsed if isinstance(parsed, schema) else schema.model_validate(parsed)
            parsing_error = result.get("parsing_error")
            if getattr(raw, "tool_calls", None):
                raise ModelResponseError(f"大模型函数参数不符合结构化 schema：{parsing_error}")
            raise self._protocol_error("结构化输出/Function Calling", parsing_error or "未返回函数调用")
        except (ModelConfigurationError, ModelThinkingError, ModelResponseError):
            raise
        except (ValidationError, ValueError, TypeError, AttributeError) as exc:
            raise ModelResponseError(f"大模型结构化函数参数无法解析：{exc}") from exc
        except Exception as exc:
            if compat.is_rejected_request(exc):
                raise self._protocol_error("结构化输出/Function Calling", exc) from exc
            raise ModelGatewayError(f"大模型调用失败：{type(exc).__name__}: {exc}") from exc

    def invoke_text(self, *, messages: list[ModelMessage]) -> str:
        prepared = _transport_messages(messages)
        try:
            response = self._degrade(
                lambda: self._get_model().invoke(prepared),
                allowed=self._degradable_client_features(),
            )
            self._guard_thinking(response)
            answer = _visible_text(response)
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
        """Return validated native calls; never reinterpret response prose as a call."""

        if not tools:
            raise ModelConfigurationError("没有可提供给大模型的研究函数。")
        allowed_names = _tool_names(tools)
        if not allowed_names:
            raise ModelConfigurationError("研究函数定义缺少 name。")
        prepared = _transport_messages(messages)
        try:
            response = self._degrade(
                lambda: self._bound_tools(tools).invoke(prepared),
                allowed=(
                    self._degradable_client_features()
                    | compat.OPTIONAL_TOOL_CALL_FEATURES
                ),
            )
            self._guard_thinking(response)
            calls = _normalized_calls(getattr(response, "tool_calls", None))
        except (ModelConfigurationError, ModelThinkingError):
            raise
        except ValidationError as exc:
            raise ModelResponseError(f"大模型函数参数无法解析：{exc}") from exc
        except Exception as exc:
            if compat.is_rejected_request(exc):
                raise self._protocol_error("Function Calling", exc) from exc
            raise ModelGatewayError(f"大模型函数选择失败：{type(exc).__name__}: {exc}") from exc
        if not calls or any(not call.name for call in calls):
            raise self._protocol_error("Function Calling", "未返回原生函数调用")
        unknown = sorted({call.name for call in calls} - allowed_names)
        if unknown:
            raise ModelResponseError(f"大模型调用了未提供的研究函数：{', '.join(unknown)}")
        return calls

    def _bound_tools(self, tools: list[dict[str, Any]]) -> Any:
        options: dict[str, Any] = {"tool_choice": "required"}
        if "parallel_tool_calls" not in self._disabled:
            options["parallel_tool_calls"] = True
        return self._get_model().bind_tools(tools, **options)

    def _guard_thinking(self, response: Any) -> None:
        """Apply the configured reasoning policy; ``strip`` simply discards the trace."""

        if self.settings.llm_thinking_policy == "reject":
            _reject_nonempty_thinking(response)
