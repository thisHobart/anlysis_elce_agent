from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

from app.config import Settings, get_settings
from app.schemas.llm import ClassificationResult

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"


class LLMServiceError(RuntimeError):
    """Raised when an LLM call cannot produce a usable result."""


class AgentLLM(Protocol):
    """Small injectable boundary used by graph nodes and offline tests."""

    @property
    def enabled(self) -> bool: ...

    def classify(
        self,
        *,
        question: str,
        faq_catalog: list[dict[str, Any]],
        history: list[dict[str, str]],
        context: dict[str, Any],
    ) -> ClassificationResult: ...

    def compose(self, *, payload: dict[str, Any]) -> str: ...


def _read_prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8")


def _message_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [item.get("text", "") for item in content if isinstance(item, dict)]
        return "".join(parts).strip()
    return str(content).strip()


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped


class LLMService:
    """OpenAI-compatible local-model adapter with lazy client creation."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._model: Any | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.settings.llm_enabled and self.settings.llm_base_url and self.settings.llm_model)

    def _get_model(self) -> Any:
        if not self.enabled:
            raise LLMServiceError("LLM is disabled or incompletely configured.")
        if self._model is None:
            try:
                from langchain_openai import ChatOpenAI
            except ImportError as exc:  # pragma: no cover - depends on optional runtime installation
                raise LLMServiceError("langchain-openai is not installed.") from exc
            self._model = ChatOpenAI(
                model=self.settings.llm_model,
                base_url=self.settings.llm_base_url,
                api_key=self.settings.llm_api_key.get_secret_value() or "local-placeholder",
                timeout=self.settings.llm_timeout_seconds,
                max_retries=self.settings.llm_max_retries,
                temperature=0,
            )
        return self._model

    def classify(
        self,
        *,
        question: str,
        faq_catalog: list[dict[str, Any]],
        history: list[dict[str, str]],
        context: dict[str, Any],
    ) -> ClassificationResult:
        model = self._get_model()
        system_prompt = _read_prompt("classify_system.md")
        payload = {
            "question": question,
            "history": history,
            "context": context,
            "faq_candidates": faq_catalog,
            "output_schema": ClassificationResult.model_json_schema(),
        }
        messages = [("system", system_prompt), ("human", json.dumps(payload, ensure_ascii=False))]
        try:
            if self.settings.llm_structured_mode == "native":
                result = model.with_structured_output(ClassificationResult).invoke(messages)
                return result if isinstance(result, ClassificationResult) else ClassificationResult.model_validate(result)
            response = model.invoke(messages)
            parsed = json.loads(_strip_json_fence(_message_text(response)))
            return ClassificationResult.model_validate(parsed)
        except Exception as exc:
            raise LLMServiceError("The classification model returned an unusable result.") from exc

    def compose(self, *, payload: dict[str, Any]) -> str:
        model = self._get_model()
        messages = [
            ("system", _read_prompt("compose_system.md")),
            ("human", json.dumps(payload, ensure_ascii=False, default=str)),
        ]
        try:
            answer = _message_text(model.invoke(messages))
        except Exception as exc:
            raise LLMServiceError("The compose model call failed.") from exc
        if not answer:
            raise LLMServiceError("The compose model returned an empty answer.")
        return answer


@lru_cache
def get_llm_service() -> LLMService:
    return LLMService()
