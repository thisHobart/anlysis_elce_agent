"""Research-model protocol implementation for Chat Completions and Responses."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.config import Settings, get_settings
from app.llm import compat
from app.llm.audit import ModelCallAudit
from app.llm.budget import ModelBudgetManager, ModelRequestPurpose, RequestBudget
from app.llm.context_safety import (
    ensure_complete_response,
    is_output_truncation_error,
)
from app.llm.gateway import (
    ModelConfigurationError,
    ModelGatewayError,
    ModelMessage,
    ModelOutputTruncatedError,
    ModelProtocolError,
    ModelResponseError,
    ModelThinkingError,
    ModelToolCall,
    ModelToolTurn,
    ModelTransientError,
    StructuredResult,
)
from app.llm.langchain_support import (
    normalized_calls,
    reject_nonempty_thinking,
    response_metadata,
    response_text,
    tool_names,
    transport_messages,
    visible_text,
)
from app.llm.model_profiles import resolve_effective_model_profile
from app.llm.runtime_settings import LLMRuntimeSettings


def _prompt_json_messages(
    messages: list[ModelMessage],
    schema: type[StructuredResult],
) -> list[ModelMessage]:
    """Add the exact response contract for proxies that discard API-level schemas."""

    schema_json = json.dumps(
        schema.model_json_schema(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    instruction = (
        "\n\n[结构化输出兼容协议]\n"
        "上游代理不会传递原生 response schema。你必须只返回一个符合下列 JSON Schema 的"
        "JSON 对象，不要返回 Markdown、代码围栏、解释或额外字段。字段名、嵌套层级、枚举值和"
        "必填项必须完全一致；无法确定业务事实时使用 schema 允许的 uncertain 形式，不得编造。\n"
        f"JSON Schema：{schema_json}"
    )
    prepared = list(messages)
    for index, message in enumerate(prepared):
        if message.role == "system":
            prepared[index] = message.model_copy(update={"content": f"{message.content}{instruction}"})
            break
    else:
        prepared.insert(0, ModelMessage(role="system", content=instruction.lstrip()))
    return prepared


class _PromptToolCall(BaseModel):
    """One locally validated function proposal for explicit proxy compatibility mode."""

    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: dict[str, Any]


class _PromptToolCalls(BaseModel):
    """Envelope used only when prompt_json is explicitly selected."""

    model_config = ConfigDict(extra="forbid")

    calls: list[_PromptToolCall] = Field(min_length=1)


class _PromptToolTurn(BaseModel):
    """Compatibility envelope for one optional tool-selection turn."""

    model_config = ConfigDict(extra="forbid")

    content: str = ""
    calls: list[_PromptToolCall] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_turn(self) -> _PromptToolTurn:
        if not self.content.strip() and not self.calls:
            raise ValueError("content 和 calls 不能同时为空")
        return self


def _prompt_tool_call_messages(
    messages: list[ModelMessage],
    tools: list[dict[str, Any]],
) -> list[ModelMessage]:
    """Describe the exact tool whitelist when an OpenAI proxy discards ``tools``."""

    definitions = []
    for tool in tools:
        function = tool.get("function", tool)
        if isinstance(function, dict):
            definitions.append(
                {
                    "name": function.get("name"),
                    "description": function.get("description", ""),
                    "parameters": function.get("parameters", {}),
                }
            )
    instruction = (
        "\n\n[研究函数选择兼容协议]\n"
        "当前代理不会转发原生 tools。请在结构化结果的 calls 数组中选择完成任务所需的最少函数。"
        "name 必须逐字来自下列白名单，arguments 必须符合对应 parameters；不得返回白名单之外的函数，"
        "不得执行函数，不得在结构化对象之外输出说明。\n"
        "函数白名单：" + json.dumps(definitions, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    prepared = list(messages)
    for index, message in enumerate(prepared):
        if message.role == "system":
            prepared[index] = message.model_copy(update={"content": f"{message.content}{instruction}"})
            break
    else:
        prepared.insert(0, ModelMessage(role="system", content=instruction.lstrip()))
    return prepared


class ResearchModelGateway:
    """Enforce the native model protocol required by the research Agent.

    Canonical messages, Pydantic schemas, and research functions are translated
    through the explicitly selected Chat Completions or Responses API. Core
    capabilities fail closed; only optional request controls may be removed when
    an endpoint explicitly rejects them.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        structured_output_observer: Callable[[dict[str, Any]], None] | None = None,
        runtime_settings: LLMRuntimeSettings | None = None,
        call_audit: ModelCallAudit | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._model: Any | None = None
        self._http_client: Any | None = None
        self._disabled: set[str] = set()
        self._structured_output_observer = structured_output_observer
        resolved_profile = resolve_effective_model_profile(
            provider=self.settings.llm_provider,
            base_url=self.settings.llm_base_url,
            model=self.settings.llm_model,
            context_window_tokens=self.settings.llm_context_window_tokens,
            max_output_tokens=self.settings.llm_max_output_tokens,
            api_style=self.settings.llm_api_style,
            structured_output_method=self.settings.llm_structured_output_method,
        )
        self._budget_manager = ModelBudgetManager(
            resolved_profile=resolved_profile,
            runtime=runtime_settings,
        )
        self._call_audit = call_audit

    def _observe_structured_output(
        self,
        raw: Any,
        parsed: object,
        parsing_error: object,
    ) -> None:
        """Expose function arguments only when a caller explicitly requests diagnostics."""

        if self._structured_output_observer is None:
            return
        calls = normalized_calls(getattr(raw, "tool_calls", None))
        self._structured_output_observer(
            {
                "provider": self.settings.llm_provider,
                "transport": "openai_compatible",
                "model": self.model_name,
                "structured_output_method": self.settings.llm_structured_output_method,
                "schema_enforcement": (
                    "local" if self.settings.llm_structured_output_method == "prompt_json" else "provider"
                ),
                "raw_content": response_text(raw),
                "tool_calls": [call.model_dump(mode="json") for call in calls],
                "parsed": (parsed.model_dump(mode="json") if isinstance(parsed, BaseModel) else parsed),
                "parsing_error": str(parsing_error) if parsing_error is not None else None,
                "response_metadata": response_metadata(raw),
            }
        )

    @property
    def enabled(self) -> bool:
        return bool(self.settings.llm_base_url and self.settings.llm_model)

    @property
    def model_name(self) -> str:
        return self.settings.llm_model

    @property
    def budget_manager(self) -> ModelBudgetManager:
        return self._budget_manager

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
                "extra_body": {"thinking": {"type": "disabled" if effort == "none" else "enabled"}}
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

    @staticmethod
    def _purpose(value: ModelRequestPurpose | str) -> ModelRequestPurpose:
        try:
            return ModelRequestPurpose(value)
        except ValueError:
            return ModelRequestPurpose.GENERIC

    def _budget(
        self,
        messages: list[ModelMessage],
        *,
        purpose: ModelRequestPurpose | str,
        schema: type[BaseModel] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> RequestBudget:
        return self._budget_manager.budget(
            messages,
            purpose=self._purpose(purpose),
            schema=schema,
            tools=tools,
        )

    def _request_reasoning_options(self, purpose: ModelRequestPurpose) -> dict[str, Any]:
        if purpose not in {
            ModelRequestPurpose.DIALOGUE,
            ModelRequestPurpose.EDA_PLANNING,
            ModelRequestPurpose.EDA_PLANNING_RECOVERY,
        }:
            return {}
        if "provider_reasoning" in self._disabled:
            return {}
        if self.settings.llm_api_style == "responses":
            return {"reasoning": {"effort": "none"}}
        if self.settings.llm_provider == "deepseek":
            return {"extra_body": {"thinking": {"type": "disabled"}}}
        if self.settings.llm_provider == "qwen":
            return {"extra_body": {"enable_thinking": False}}
        if self.settings.llm_reasoning_effort:
            return {"reasoning_effort": "none"}
        return {}

    def _bind_request_options(self, runnable: Any, budget: RequestBudget) -> Any:
        bind = getattr(runnable, "bind", None)
        if not callable(bind):
            return runnable
        options: dict[str, Any] = {"max_tokens": budget.effective_output_tokens}
        options.update(self._request_reasoning_options(budget.purpose))
        return bind(**options)

    def _audit(
        self,
        budget: RequestBudget,
        *,
        response: Any | None,
        outcome: str,
        error: BaseException | None = None,
    ) -> None:
        if self._call_audit is not None:
            self._call_audit.record(
                budget,
                response=response,
                outcome=outcome,
                error_type=type(error).__name__ if error is not None else None,
            )

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
            "请确认 API 形式和模型能力；只有显式配置 prompt_json 时才允许使用"
            "提示词约束 JSON，并继续执行本地严格校验。"
        )

    def invoke_structured(
        self,
        *,
        messages: list[ModelMessage],
        schema: type[StructuredResult],
        purpose: ModelRequestPurpose | str = ModelRequestPurpose.GENERIC,
    ) -> StructuredResult:
        """Request structured output and validate every result against the local schema."""

        method = self.settings.llm_structured_output_method
        compatibility_mode = method == "prompt_json"
        request_messages = _prompt_json_messages(messages, schema) if compatibility_mode else messages
        budget = self._budget(
            request_messages,
            purpose=purpose,
            schema=None if compatibility_mode else schema,
        )
        prepared = transport_messages(request_messages)
        wire_method = "json_mode" if compatibility_mode else method
        raw: Any | None = None
        try:

            def invoke() -> Any:
                runnable = self._get_model().with_structured_output(
                    schema,
                    method=wire_method,
                    include_raw=True,
                )
                return self._bind_request_options(runnable, budget).invoke(prepared)

            result = self._degrade(
                invoke,
                allowed=self._degradable_client_features(),
            )
            if not isinstance(result, dict):
                raise ModelResponseError("大模型结构化输出缺少原始响应，无法验证协议。")
            raw = result.get("raw")
            if raw is None:
                raise ModelResponseError("大模型结构化输出缺少 raw，无法验证协议。")
            self._guard_thinking(raw)
            ensure_complete_response(raw)
            parsed = result.get("parsed")
            parsing_error = result.get("parsing_error")
            if parsing_error is not None:
                calls = normalized_calls(getattr(raw, "tool_calls", None))
                print("[结构化模型] Schema 解析失败，模型原始输出：")
                if calls:
                    print(
                        json.dumps(
                            [call.model_dump(mode="json") for call in calls],
                            ensure_ascii=False,
                            indent=2,
                            default=str,
                        )
                    )
                else:
                    print(response_text(raw) or "<empty>")
                print("[结构化模型] Schema 校验错误：")
                print(str(parsing_error))
            self._observe_structured_output(raw, parsed, parsing_error)
            if parsed is not None:
                resolved = parsed if isinstance(parsed, schema) else schema.model_validate(parsed)
                self._audit(budget, response=raw, outcome="completed")
                return resolved
            if getattr(raw, "tool_calls", None):
                raise ModelResponseError(f"大模型函数参数不符合结构化 schema：{parsing_error}")
            if compatibility_mode:
                raise ModelResponseError(f"Cherry 兼容 JSON 未通过本地结构化 schema：{parsing_error or 'parsed 为空'}")
            raise self._protocol_error(
                f"结构化输出（method={method}）",
                parsing_error
                or f"未返回可解析的结构化结果；该端点可能不支持 {method}，可用 "
                "scripts/probe_structured_output.py 探测其支持的原生方法",
            )
        except (ModelConfigurationError, ModelThinkingError, ModelResponseError) as exc:
            self._audit(budget, response=raw, outcome="failed", error=exc)
            raise
        except (ValidationError, ValueError, TypeError, AttributeError) as exc:
            self._audit(budget, response=raw, outcome="failed", error=exc)
            raise ModelResponseError(f"大模型结构化函数参数无法解析：{exc}") from exc
        except Exception as exc:
            self._audit(budget, response=raw, outcome="failed", error=exc)
            if is_output_truncation_error(exc):
                raise ModelOutputTruncatedError(f"模型输出达到 token 限制：{exc}") from exc
            if compat.is_transient_failure(exc):
                raise ModelTransientError(f"模型端点暂时不可用：{type(exc).__name__}: {exc}") from exc
            if compat.is_rejected_request(exc):
                raise self._protocol_error("结构化输出/Function Calling", exc) from exc
            raise ModelGatewayError(f"大模型调用失败：{type(exc).__name__}: {exc}") from exc

    def invoke_text(
        self,
        *,
        messages: list[ModelMessage],
        purpose: ModelRequestPurpose | str = ModelRequestPurpose.GENERIC,
    ) -> str:
        budget = self._budget(messages, purpose=purpose)
        prepared = transport_messages(messages)
        response: Any | None = None
        try:
            response = self._degrade(
                lambda: self._bind_request_options(self._get_model(), budget).invoke(prepared),
                allowed=self._degradable_client_features(),
            )
            self._guard_thinking(response)
            ensure_complete_response(response)
            answer = visible_text(response)
        except ModelGatewayError as exc:
            self._audit(budget, response=response, outcome="failed", error=exc)
            raise
        except Exception as exc:
            self._audit(budget, response=response, outcome="failed", error=exc)
            if is_output_truncation_error(exc):
                raise ModelOutputTruncatedError(f"模型输出达到 token 限制：{exc}") from exc
            if compat.is_transient_failure(exc):
                raise ModelTransientError(f"模型端点暂时不可用：{type(exc).__name__}: {exc}") from exc
            raise ModelGatewayError(f"大模型调用失败：{type(exc).__name__}: {exc}") from exc
        if not answer:
            raise ModelResponseError("大模型返回了空回复。")
        self._audit(budget, response=response, outcome="completed")
        return answer

    def invoke_tool_calls(
        self,
        *,
        messages: list[ModelMessage],
        tools: list[dict[str, Any]],
        purpose: ModelRequestPurpose | str = ModelRequestPurpose.GENERIC,
    ) -> list[ModelToolCall]:
        """Return validated native calls; never reinterpret response prose as a call."""

        if not tools:
            raise ModelConfigurationError("没有可提供给大模型的研究函数。")
        allowed_names = tool_names(tools)
        if not allowed_names:
            raise ModelConfigurationError("研究函数定义缺少 name。")
        if self.settings.llm_structured_output_method == "prompt_json":
            result = self.invoke_structured(
                messages=_prompt_tool_call_messages(messages, tools),
                schema=_PromptToolCalls,
                purpose=purpose,
            )
            calls = [
                ModelToolCall(
                    name=call.name,
                    arguments=call.arguments,
                    call_id=f"prompt-json-{index}",
                )
                for index, call in enumerate(result.calls, start=1)
            ]
            return self._validate_tool_calls(calls, allowed_names)
        budget = self._budget(messages, purpose=purpose, tools=tools)
        prepared = transport_messages(messages)
        response: Any | None = None
        try:
            response = self._degrade(
                lambda: self._bind_request_options(self._bound_tools(tools), budget).invoke(prepared),
                allowed=(self._degradable_client_features() | compat.OPTIONAL_TOOL_CALL_FEATURES),
            )
            self._guard_thinking(response)
            ensure_complete_response(response)
            calls = normalized_calls(getattr(response, "tool_calls", None))
        except (ModelConfigurationError, ModelThinkingError, ModelOutputTruncatedError) as exc:
            self._audit(budget, response=response, outcome="failed", error=exc)
            raise
        except ValidationError as exc:
            self._audit(budget, response=response, outcome="failed", error=exc)
            raise ModelResponseError(f"大模型函数参数无法解析：{exc}") from exc
        except Exception as exc:
            self._audit(budget, response=response, outcome="failed", error=exc)
            if is_output_truncation_error(exc):
                raise ModelOutputTruncatedError(f"模型输出达到 token 限制：{exc}") from exc
            if compat.is_transient_failure(exc):
                raise ModelTransientError(f"模型端点暂时不可用：{type(exc).__name__}: {exc}") from exc
            if compat.is_rejected_request(exc):
                raise self._protocol_error("Function Calling", exc) from exc
            raise ModelGatewayError(f"大模型函数选择失败：{type(exc).__name__}: {exc}") from exc
        if not calls or any(not call.name for call in calls):
            raise self._protocol_error("Function Calling", "未返回原生函数调用")
        validated = self._validate_tool_calls(calls, allowed_names)
        self._audit(budget, response=response, outcome="completed")
        return validated

    def invoke_tool_turn(
        self,
        *,
        messages: list[ModelMessage],
        tools: list[dict[str, Any]],
        purpose: ModelRequestPurpose | str = ModelRequestPurpose.GENERIC,
    ) -> ModelToolTurn:
        """Return one optional-tool assistant turn while preserving provider call ids."""

        if not tools:
            raise ModelConfigurationError("没有可提供给大模型的研究函数。")
        allowed_names = tool_names(tools)
        if not allowed_names:
            raise ModelConfigurationError("研究函数定义缺少 name。")
        if self.settings.llm_structured_output_method == "prompt_json":
            result = self.invoke_structured(
                messages=_prompt_tool_call_messages(messages, tools),
                schema=_PromptToolTurn,
                purpose=purpose,
            )
            history_key = json.dumps(
                [message.model_dump(mode="json") for message in messages],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            calls = [
                ModelToolCall(
                    name=call.name,
                    arguments=call.arguments,
                    call_id=(
                        "prompt-json-"
                        + hashlib.sha256(
                            f"{history_key}:{index}:{call.model_dump_json()}".encode()
                        ).hexdigest()[:20]
                    ),
                )
                for index, call in enumerate(result.calls, start=1)
            ]
            return ModelToolTurn(
                content=result.content,
                tool_calls=self._validate_tool_calls(calls, allowed_names),
                finish_reason="prompt_json",
            )
        budget = self._budget(messages, purpose=purpose, tools=tools)
        prepared = transport_messages(messages)
        response: Any | None = None
        try:
            response = self._degrade(
                lambda: self._bind_request_options(self._bound_tools_for_turn(tools), budget).invoke(prepared),
                allowed=(self._degradable_client_features() | compat.OPTIONAL_TOOL_CALL_FEATURES),
            )
            self._guard_thinking(response)
            ensure_complete_response(response)
            calls = normalized_calls(getattr(response, "tool_calls", None))
            content = visible_text(response)
        except (ModelConfigurationError, ModelThinkingError, ModelOutputTruncatedError) as exc:
            self._audit(budget, response=response, outcome="failed", error=exc)
            raise
        except ValidationError as exc:
            self._audit(budget, response=response, outcome="failed", error=exc)
            raise ModelResponseError(f"大模型函数参数无法解析：{exc}") from exc
        except Exception as exc:
            self._audit(budget, response=response, outcome="failed", error=exc)
            if is_output_truncation_error(exc):
                raise ModelOutputTruncatedError(f"模型输出达到 token 限制：{exc}") from exc
            if compat.is_transient_failure(exc):
                raise ModelTransientError(f"模型端点暂时不可用：{type(exc).__name__}: {exc}") from exc
            if compat.is_rejected_request(exc):
                raise self._protocol_error("Function Calling", exc) from exc
            raise ModelGatewayError(f"大模型函数选择失败：{type(exc).__name__}: {exc}") from exc
        validated = self._validate_tool_calls(calls, allowed_names)
        if any(not call.call_id for call in validated):
            raise ModelResponseError("大模型函数调用缺少 provider call_id。")
        finish_reason = response_metadata(response).get("finish_reason")
        turn = ModelToolTurn(
            content=content,
            tool_calls=validated,
            finish_reason=str(finish_reason) if finish_reason is not None else None,
        )
        self._audit(budget, response=response, outcome="completed")
        return turn

    @staticmethod
    def _validate_tool_calls(
        calls: list[ModelToolCall],
        allowed_names: set[str],
    ) -> list[ModelToolCall]:
        """Keep compatibility proposals inside the same whitelist as native calls."""

        unknown = sorted({call.name for call in calls} - allowed_names)
        if unknown:
            raise ModelResponseError(f"大模型调用了未提供的研究函数：{', '.join(unknown)}")
        return calls

    def _bound_tools(self, tools: list[dict[str, Any]]) -> Any:
        options: dict[str, Any] = {"tool_choice": "required"}
        if "parallel_tool_calls" not in self._disabled:
            options["parallel_tool_calls"] = True
        return self._get_model().bind_tools(tools, **options)

    def _bound_tools_for_turn(self, tools: list[dict[str, Any]]) -> Any:
        options: dict[str, Any] = {}
        if "parallel_tool_calls" not in self._disabled:
            options["parallel_tool_calls"] = True
        return self._get_model().bind_tools(tools, **options)

    def _guard_thinking(self, response: Any) -> None:
        """Apply the configured reasoning policy; ``strip`` simply discards the trace."""

        if self.settings.llm_thinking_policy == "reject":
            reject_nonempty_thinking(response)
