"""Narrow repairs for optional request fields and non-standard HTTP envelopes.

These helpers never emulate message roles, structured output, or function calls.
Those are core research-model capabilities and an incompatible endpoint must fail
closed at the gateway boundary.
"""

from __future__ import annotations

import json
import re
from typing import Any

THINK_BLOCK = re.compile(r"<think>.*?(?:</think>|$)", flags=re.IGNORECASE | re.DOTALL)
_THINK_CONTENT = re.compile(r"<think>(.*?)</think>", flags=re.IGNORECASE | re.DOTALL)

#: Request options the gateway sets itself and can therefore drop, ordered from
#: most to least specific. Every entry maps a feature name onto the wording
#: providers use to refuse it. Core protocol fields such as ``tools``,
#: ``tool_choice`` and ``response_format`` are absent on purpose: refusing one
#: means the endpoint is not compatible with the research Agent.
UNSUPPORTED_HINTS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "provider_reasoning",
        re.compile(
            r"enable_thinking|chat_template_kwargs|reasoning[_ .]effort|['\"](?:thinking|reasoning)['\"]",
            re.IGNORECASE,
        ),
    ),
    ("parallel_tool_calls", re.compile(r"parallel[_ ]tool[_ ]calls", re.IGNORECASE)),
    ("temperature", re.compile(r"\btemperature\b", re.IGNORECASE)),
)

#: Features that live on the client constructor; changing one rebuilds the client.
CLIENT_LEVEL_FEATURES = frozenset({"provider_reasoning", "temperature"})

#: Features that only exist while binding research functions to a request.
OPTIONAL_TOOL_CALL_FEATURES = frozenset({"parallel_tool_calls"})

_REJECTED_REQUEST_ERRORS = frozenset(
    {"BadRequestError", "UnprocessableEntityError", "NotFoundError"}
)
_TRANSPORT_ERRORS = frozenset(
    {
        "APITimeoutError",
        "APIConnectionError",
        "AuthenticationError",
        "PermissionDeniedError",
        "RateLimitError",
        "InternalServerError",
    }
)


def is_rejected_request(exc: BaseException) -> bool:
    """Report whether the endpoint refused the request itself, not the connection."""

    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status in {400, 404, 422}
    return type(exc).__name__ in _REJECTED_REQUEST_ERRORS


def is_transport_failure(exc: BaseException) -> bool:
    """Report whether retrying with different request options is pointless."""

    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status >= 500 or status in {401, 403, 408, 429}
    return type(exc).__name__ in _TRANSPORT_ERRORS


def is_transient_failure(exc: BaseException) -> bool:
    """Report invocation-scoped failures that may succeed on a later request."""

    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status >= 500 or status in {408, 429}
    return type(exc).__name__ in {
        "APITimeoutError",
        "APIConnectionError",
        "RateLimitError",
        "InternalServerError",
    }


def unsupported_feature(
    exc: BaseException,
    disabled: frozenset[str] | set[str],
    allowed: frozenset[str],
) -> str | None:
    """Name the droppable request option an endpoint just refused, if any."""

    if not is_rejected_request(exc):
        return None
    text = str(exc)
    for feature, pattern in UNSUPPORTED_HINTS:
        if feature in allowed and feature not in disabled and pattern.search(text):
            return feature
    return None


def strip_thinking(text: str) -> str:
    """Remove ``<think>`` traces so only the answer reaches the research loop."""

    return THINK_BLOCK.sub("", text).strip()


def thinking_fragments(text: str) -> list[str]:
    """Return the non-empty ``<think>`` payloads a response still carries."""

    return [
        fragment
        for match in _THINK_CONTENT.finditer(text)
        if (fragment := match.group(1).strip())
    ]


def unwrap_double_encoded_body(raw: bytes) -> bytes | None:
    """Return the inner document when a gateway JSON-encodes its response twice."""

    try:
        payload = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(payload, str):
        return None
    try:
        inner = json.loads(payload)
    except ValueError:
        return None
    if not isinstance(inner, dict | list):
        return None
    return payload.encode("utf-8")


def build_compatible_http_client() -> Any | None:
    """Build the HTTP client that repairs non-standard response envelopes.

    The client is subclassed rather than given a custom transport so httpx keeps
    its own proxy discovery from the environment.
    """

    try:
        import httpx
    except ImportError:  # pragma: no cover - httpx ships with the openai client
        return None

    class _EnvelopeRepairClient(httpx.Client):
        """Unwrap double-encoded bodies before the OpenAI client parses them."""

        def send(self, request: httpx.Request, *, stream: bool = False, **kwargs: Any) -> httpx.Response:
            response = super().send(request, stream=stream, **kwargs)
            if stream:
                return response
            repaired = unwrap_double_encoded_body(response.content)
            if repaired is None:
                return response
            headers = [
                (key, value)
                for key, value in response.headers.multi_items()
                if key.lower() not in {"content-length", "content-encoding"}
            ]
            unwrapped = httpx.Response(
                response.status_code,
                headers=headers,
                content=repaired,
                request=request,
                extensions=response.extensions,
                history=response.history,
            )
            unwrapped.elapsed = response.elapsed
            return unwrapped

    return _EnvelopeRepairClient()
