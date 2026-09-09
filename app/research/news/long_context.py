"""Whole-document long-context strategy with explicit model-capacity routing."""

from __future__ import annotations

import time
from dataclasses import dataclass

from app.llm.context_safety import (
    RequestBudget,
    TokenCounter,
    conservative_token_count,
    structured_request_budget,
)
from app.llm.gateway import ModelContextLimitError, ModelGateway
from app.research.news.contracts import (
    CharacterRange,
    EventExtractionBatch,
    EventExtractionResult,
    ExtractionCoverage,
    ExtractionQuarantine,
    NewsDocument,
)
from app.research.news.model_extraction import (
    ModelNewsExtraction,
    StructuredNewsEventExtractor,
    build_model_news_extraction_messages,
)


@dataclass(frozen=True)
class ModelRoute:
    """One explicitly configured model capacity, ordered by routing preference."""

    name: str
    gateway: ModelGateway
    context_window_tokens: int
    reserved_output_tokens: int = 4096
    safety_tokens: int = 1024
    token_counter: TokenCounter = conservative_token_count

    def preflight(self, document: NewsDocument, market_timezone: str) -> RequestBudget:
        return structured_request_budget(
            build_model_news_extraction_messages(document, market_timezone=market_timezone),
            ModelNewsExtraction,
            context_window_tokens=self.context_window_tokens,
            reserved_output_tokens=self.reserved_output_tokens,
            safety_tokens=self.safety_tokens,
            token_counter=self.token_counter,
        )


class FullContextNewsExtractor:
    """Send every source character to one capacity-qualified model invocation path."""

    extractor_id = "full-context-news-extractor"
    extractor_version = "1.0.0"

    def __init__(
        self,
        routes: tuple[ModelRoute, ...],
        *,
        market_timezone: str,
        extraction_passes: int = 1,
        max_repair_attempts: int = 1,
        minimum_confidence: float = 0.7,
    ) -> None:
        if not routes:
            raise ValueError("at least one model route is required")
        if any(route.context_window_tokens <= 0 for route in routes):
            raise ValueError("every full-context route requires a positive context window")
        self.routes = routes
        self.market_timezone = market_timezone
        self.extraction_passes = extraction_passes
        self.max_repair_attempts = max_repair_attempts
        self.minimum_confidence = minimum_confidence
        self.last_route_name: str | None = None

    def extract(self, document: NewsDocument) -> EventExtractionResult:
        started = time.perf_counter()
        selected: ModelRoute | None = None
        selected_budget: RequestBudget | None = None
        failures: list[str] = []
        for route in self.routes:
            try:
                selected_budget = route.preflight(document, self.market_timezone)
            except ModelContextLimitError as exc:
                failures.append(f"{route.name}: {exc}")
                continue
            selected = route
            break

        full_range = CharacterRange(start_char=0, end_char=len(document.body))
        if selected is None or selected_budget is None:
            message = "所有已配置模型均无法容纳完整新闻；" + " | ".join(failures)
            return EventExtractionResult(
                document_version_id=document.document_version_id,
                quarantine=ExtractionQuarantine(
                    document_version_id=document.document_version_id,
                    reason_code="context_limit_exceeded",
                    message=message[:2048],
                    extractor_id=self.extractor_id,
                    extractor_version=self.extractor_version,
                ),
                coverage=ExtractionCoverage(
                    strategy="full_context",
                    total_characters=len(document.body),
                    failed_ranges=(full_range,),
                    estimated_input_tokens=0,
                    elapsed_seconds=time.perf_counter() - started,
                ),
            )

        self.last_route_name = selected.name
        delegate = StructuredNewsEventExtractor(
            selected.gateway,
            market_timezone=self.market_timezone,
            extraction_passes=self.extraction_passes,
            max_repair_attempts=self.max_repair_attempts,
            minimum_confidence=self.minimum_confidence,
        )
        result = delegate.extract(document)
        failed = bool(
            result.quarantine
            and result.quarantine.reason_code
            in {"context_limit_exceeded", "model_output_truncated", "model_response_invalid", "model_unavailable"}
        )
        coverage = ExtractionCoverage(
            strategy="full_context",
            total_characters=len(document.body),
            processed_ranges=() if failed else (full_range,),
            failed_ranges=(full_range,) if failed else (),
            model_calls=delegate.model_calls,
            estimated_input_tokens=selected_budget.input_tokens * max(1, delegate.model_calls),
            elapsed_seconds=time.perf_counter() - started,
        )
        return result.model_copy(update={"coverage": coverage})

    def extract_many(self, documents: tuple[NewsDocument, ...]) -> EventExtractionBatch:
        return EventExtractionBatch(results=tuple(self.extract(document) for document in documents))
