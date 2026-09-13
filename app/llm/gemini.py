"""Native Gemini and Vertex AI implementation of the shared model gateway."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ValidationError

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


class GeminiModelGateway:
    """Call Gemini with its native protocol instead of an OpenAI compatibility layer."""

    structured_output_method = "json_schema"

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
        self._structured_output_observer = structured_output_observer
        route_url = (
            f"vertex://{self.settings.llm_google_project}/{self.settings.llm_google_location}"
            if self.settings.llm_google_vertexai
            else ""
        )
        resolved_profile = resolve_effective_model_profile(
            provider="gemini",
            base_url=route_url,
            model=self.settings.llm_model,
            context_window_tokens=self.settings.llm_context_window_tokens,
            max_output_tokens=self.settings.llm_max_output_tokens,
            api_style="chat",
            structured_output_method="json_schema",
        )
        self._budget_manager = ModelBudgetManager(
            resolved_profile=resolved_profile,
            runtime=runtime_settings,
        )
        self._call_audit = call_audit

    @property
    def enabled(self) -> bool:
        # Authentication may come from VPP_LLM_API_KEY, GOOGLE_API_KEY, or Vertex ADC.
        return bool(self.settings.llm_model)

    @property
    def model_name(self) -> str:
        return self.settings.llm_model

    @property
    def budget_manager(self) -> ModelBudgetManager:
        return self._budget_manager

    def _model_options(self) -> dict[str, Any]:
        model = self.model_name.strip()
        if model.casefold().startswith("vertexai:"):
            raise ModelConfigurationError(
                "Gemini 原生适配器不能使用 Cherry Studio 的 vertexai: 模型前缀。"
                "请填写 Google 模型名（例如 gemini-2.5-flash）；使用 Vertex AI 时另设 "
                "桌面运行设置中的 Vertex 开关、项目和地区。"
            )
        options: dict[str, Any] = {
            "model": model,
            "temperature": 0,
            "max_retries": self.settings.llm_max_retries,
            "timeout": self.settings.llm_timeout_seconds,
            "include_thoughts": False,
            "vertexai": self.settings.llm_google_vertexai,
        }
        if model.casefold().startswith("gemini-2.5-flash"):
            options["thinking_budget"] = 0
        if self.settings.llm_google_vertexai:
            if project := self.settings.llm_google_project.strip():
                options["project"] = project
            if location := self.settings.llm_google_location.strip():
                options["location"] = location
        elif api_key := self.settings.llm_api_key.get_secret_value().strip():
            options["api_key"] = api_key
        return options

    def _get_model(self) -> Any:
        if not self.enabled:
            raise ModelConfigurationError("Gemini 尚未配置，请先填写模型名称和认证信息。")
        if self._model is None:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
            except ImportError as exc:  # pragma: no cover - runtime dependency
                raise ModelConfigurationError("langchain-google-genai 尚未安装，无法使用 Gemini 原生适配器。") from exc
            try:
                self._model = ChatGoogleGenerativeAI(**self._model_options())
            except Exception as exc:
                raise ModelConfigurationError(f"Gemini 客户端配置无效：{exc}") from exc
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

    @staticmethod
    def _bind_output_limit(runnable: Any, budget: RequestBudget) -> Any:
        bind = getattr(runnable, "bind", None)
        return bind(max_output_tokens=budget.effective_output_tokens) if callable(bind) else runnable

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

    def _observe_structured_output(
        self,
        raw: Any,
        parsed: object,
        parsing_error: object,
    ) -> None:
        if self._structured_output_observer is None:
            return
        calls = normalized_calls(getattr(raw, "tool_calls", None))
        self._structured_output_observer(
            {
                "provider": "gemini",
                "transport": "google_native",
                "model": self.model_name,
                "structured_output_method": self.structured_output_method,
                "schema_enforcement": "provider",
                "raw_content": response_text(raw),
                "tool_calls": [call.model_dump(mode="json") for call in calls],
                "parsed": (parsed.model_dump(mode="json") if isinstance(parsed, BaseModel) else parsed),
                "parsing_error": str(parsing_error) if parsing_error is not None else None,
                "response_metadata": response_metadata(raw),
            }
        )

    def invoke_structured(
        self,
        *,
        messages: list[ModelMessage],
        schema: type[StructuredResult],
        purpose: ModelRequestPurpose | str = ModelRequestPurpose.GENERIC,
    ) -> StructuredResult:
        """Use Gemini's native response schema and then validate the result locally."""

        budget = self._budget(messages, purpose=purpose, schema=schema)
        prepared = transport_messages(messages)
        raw: Any | None = None
        try:
            runnable = self._get_model().with_structured_output(
                schema,
                method=self.structured_output_method,
                include_raw=True,
            )
            result = self._bind_output_limit(runnable, budget).invoke(prepared)
            if not isinstance(result, dict):
                raise ModelResponseError("Gemini 结构化输出缺少诊断 envelope。")
            raw = result.get("raw")
            if raw is None:
                raise ModelResponseError("Gemini 结构化输出缺少 raw，无法核对模型原文。")
            self._guard_thinking(raw)
            ensure_complete_response(raw)
            parsed = result.get("parsed")
            parsing_error = result.get("parsing_error")
            self._observe_structured_output(raw, parsed, parsing_error)
            if parsing_error is not None or parsed is None:
                raise ModelResponseError(f"Gemini 原生 JSON Schema 响应无法解析：{parsing_error or 'parsed 为空'}")
            resolved = parsed if isinstance(parsed, schema) else schema.model_validate(parsed)
            self._audit(budget, response=raw, outcome="completed")
            return resolved
        except (ModelConfigurationError, ModelThinkingError, ModelResponseError) as exc:
            self._audit(budget, response=raw, outcome="failed", error=exc)
            raise
        except (ValidationError, ValueError, TypeError, AttributeError) as exc:
            self._audit(budget, response=raw, outcome="failed", error=exc)
            raise ModelResponseError(f"Gemini 响应未通过本地 schema 校验：{exc}") from exc
        except Exception as exc:
            self._audit(budget, response=raw, outcome="failed", error=exc)
            if is_output_truncation_error(exc):
                raise ModelOutputTruncatedError(f"Gemini 输出达到 token 限制：{exc}") from exc
            if compat.is_rejected_request(exc):
                raise ModelProtocolError(
                    f"当前 Gemini 模型拒绝了原生 JSON Schema 请求：{exc}。请确认模型支持结构化输出。"
                ) from exc
            raise ModelGatewayError(f"Gemini 调用失败：{type(exc).__name__}: {exc}") from exc

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
            response = self._bind_output_limit(self._get_model(), budget).invoke(prepared)
            self._guard_thinking(response)
            ensure_complete_response(response)
            answer = visible_text(response)
        except ModelGatewayError as exc:
            self._audit(budget, response=response, outcome="failed", error=exc)
            raise
        except Exception as exc:
            self._audit(budget, response=response, outcome="failed", error=exc)
            if is_output_truncation_error(exc):
                raise ModelOutputTruncatedError(f"Gemini 输出达到 token 限制：{exc}") from exc
            raise ModelGatewayError(f"Gemini 调用失败：{type(exc).__name__}: {exc}") from exc
        if not answer:
            raise ModelResponseError("Gemini 返回了空回复。")
        self._audit(budget, response=response, outcome="completed")
        return answer

    def invoke_tool_calls(
        self,
        *,
        messages: list[ModelMessage],
        tools: list[dict[str, Any]],
        purpose: ModelRequestPurpose | str = ModelRequestPurpose.GENERIC,
    ) -> list[ModelToolCall]:
        if not tools:
            raise ModelConfigurationError("没有可提供给 Gemini 的研究函数。")
        allowed_names = tool_names(tools)
        if not allowed_names:
            raise ModelConfigurationError("研究函数定义缺少 name。")
        budget = self._budget(messages, purpose=purpose, tools=tools)
        prepared = transport_messages(messages)
        response: Any | None = None
        try:
            runnable = self._get_model().bind_tools(tools, tool_choice="any")
            response = self._bind_output_limit(runnable, budget).invoke(prepared)
            self._guard_thinking(response)
            ensure_complete_response(response)
            calls = normalized_calls(getattr(response, "tool_calls", None))
        except (ModelConfigurationError, ModelThinkingError, ModelOutputTruncatedError) as exc:
            self._audit(budget, response=response, outcome="failed", error=exc)
            raise
        except ValidationError as exc:
            self._audit(budget, response=response, outcome="failed", error=exc)
            raise ModelResponseError(f"Gemini 函数参数无法解析：{exc}") from exc
        except Exception as exc:
            self._audit(budget, response=response, outcome="failed", error=exc)
            if is_output_truncation_error(exc):
                raise ModelOutputTruncatedError(f"Gemini 输出达到 token 限制：{exc}") from exc
            if compat.is_rejected_request(exc):
                raise ModelProtocolError(f"当前 Gemini 模型不支持原生函数调用：{exc}") from exc
            raise ModelGatewayError(f"Gemini 函数选择失败：{type(exc).__name__}: {exc}") from exc
        if not calls or any(not call.name for call in calls):
            raise ModelProtocolError("Gemini 未返回原生函数调用。")
        unknown = sorted({call.name for call in calls} - allowed_names)
        if unknown:
            raise ModelResponseError(f"Gemini 调用了未提供的研究函数：{', '.join(unknown)}")
        self._audit(budget, response=response, outcome="completed")
        return calls

    def _guard_thinking(self, response: Any) -> None:
        if self.settings.llm_thinking_policy == "reject":
            reject_nonempty_thinking(response)
