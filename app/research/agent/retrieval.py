"""Relevance retrieval over complete conversation turns outside the recent window.

``bounded_recent_turn_history`` answers "what was just said". This module answers
"what earlier turn in this session is about the same thing" without paying for
the whole transcript on every request.

Scoring is lexical, deterministic and dependency-free. Chinese text is
tokenized as character bigrams, while exact identifiers such as ``max_lag`` or
``run-12`` remain strong signals. Retrieval is turn-based so a matching user
question never arrives without the assistant answer that belongs to it.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

DEFAULT_RETRIEVED_TURNS = 3
DEFAULT_RETRIEVAL_CHARACTER_BUDGET = 6_000
DEFAULT_RECENT_TURN_CHARACTER_BUDGET = 24_000
MAX_RETRIEVED_MESSAGE_CHARACTERS = 1_200

_TOKEN_PATTERN = re.compile(r"[一-鿿]+|[A-Za-z0-9_-]+")
_BM25_K1 = 1.5
_BM25_B = 0.75
_COMMON_TERM_DOCUMENT_RATIO = 0.65
_MIN_QUERY_COVERAGE = 0.25
_COMMON_ONLY_QUERY_COVERAGE = 0.75
_LEADING_TRUNCATION_MARKER = "…[前文已截断]"
_TRAILING_TRUNCATION_MARKER = "…[后文已截断]"
_MIDDLE_TRUNCATION_MARKER = "…[中间已截断]"
_SEMANTIC_TERM_PREFIX = "@concept."
_REFERENCE_FOCUS_TERM = "@reference.focus"

# A small, auditable domain lexicon supplies semantic bridges for common research-
# workspace wording without adding a heavyweight embedding runtime to the desktop
# package. These terms supplement BM25; they do not replace exact identifiers.
_SEMANTIC_ALIAS_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        f"{_SEMANTIC_TERM_PREFIX}visual",
        ("图表", "图形", "可视化", "绘图", "chart", "plot", "visualization"),
    ),
    (
        f"{_SEMANTIC_TERM_PREFIX}persist",
        ("导出", "写入", "保存", "输出", "落盘", "export", "save", "write"),
    ),
    (
        f"{_SEMANTIC_TERM_PREFIX}artifact_directory",
        (
            "报告目录",
            "报告文件夹",
            "产物目录",
            "产物文件夹",
            "输出目录",
            "结果目录",
            "artifact directory",
            "artifact folder",
            "report directory",
        ),
    ),
    (
        f"{_SEMANTIC_TERM_PREFIX}contemporaneous",
        ("同期", "同时关系", "当期关系", "同步关系", "contemporaneous"),
    ),
    (
        f"{_SEMANTIC_TERM_PREFIX}lead_lag",
        ("领先滞后", "时滞关系", "延迟关系", "lead-lag", "lead lag", "lagged relationship"),
    ),
    (
        f"{_SEMANTIC_TERM_PREFIX}feature",
        ("外生变量", "驱动变量", "驱动因子", "候选特征", "exogenous variable", "driver feature"),
    ),
)
_REFERENCE_ONLY_CUES = (
    "刚才那个",
    "之前那个",
    "前面那个",
    "上次那个",
    "按刚才",
    "按之前",
    "照刚才",
    "照之前",
    "沿用之前",
    "按原来",
    "照旧",
)
_FOCUS_ACTION_CUES = (
    "设为",
    "设置",
    "调整",
    "改为",
    "要求",
    "限制",
    "目标",
    "方案",
    "参数",
    "分析",
    "导出",
    "保存",
    "使用",
    "采用",
    "窗口",
)


@dataclass(frozen=True)
class RetrievedMessage:
    """One message retained as part of a retrieved conversation turn."""

    message_id: str | None
    turn_id: str | None
    role: str
    content: str
    created_at: str
    episode_id: str | None

    def as_payload(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "turn_id": self.turn_id,
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at,
            "episode_id": self.episode_id,
        }


@dataclass(frozen=True)
class RetrievedTurn:
    """One complete earlier turn selected for the current question."""

    turn_id: str | None
    messages: tuple[RetrievedMessage, ...]
    score: float
    matched_terms: list[str]
    query_coverage: float

    def as_payload(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "messages": [message.as_payload() for message in self.messages],
            "relevance": round(self.score, 4),
            "matched_terms": self.matched_terms,
            "query_coverage": round(self.query_coverage, 4),
        }


@dataclass(frozen=True)
class _TurnCandidate:
    """Internal projection used for scoring without changing stored messages."""

    key: str
    turn_id: str | None
    order: int
    messages: tuple[Any, ...]
    text: str


def _with_content(item: Any, content: str) -> Any:
    if isinstance(item, dict):
        return {**item, "content": content}
    model_copy = getattr(item, "model_copy", None)
    if callable(model_copy):
        return model_copy(update={"content": content})
    raise TypeError(f"unsupported conversation message type: {type(item).__name__}")


def tokenize(text: str) -> list[str]:
    """Split text into comparable terms without a segmentation dictionary."""

    tokens: list[str] = []
    for run in _TOKEN_PATTERN.findall(text.lower()):
        if run[0].isascii() or len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    return tokens


def _semantic_terms(text: str) -> list[str]:
    """Map common paraphrases to stable domain concepts for hybrid retrieval."""

    lowered = text.casefold()
    return [
        concept
        for concept, aliases in _SEMANTIC_ALIAS_GROUPS
        if any(alias in lowered for alias in aliases)
    ]


def _retrieval_terms(text: str) -> list[str]:
    """Return lexical terms plus auditable semantic concept terms."""

    return list(dict.fromkeys([*tokenize(text), *_semantic_terms(text)]))


def _surface_terms(content: str, matched_terms: Sequence[str]) -> list[str]:
    """Resolve synthetic concept matches back to phrases present in one message."""

    lowered = content.casefold()
    surface: list[str] = []
    aliases_by_concept = dict(_SEMANTIC_ALIAS_GROUPS)
    for term in matched_terms:
        if term.startswith(_SEMANTIC_TERM_PREFIX):
            surface.extend(alias for alias in aliases_by_concept.get(term, ()) if alias in lowered)
        elif term == _REFERENCE_FOCUS_TERM:
            continue
        elif term.casefold() in lowered:
            surface.append(term)
    return list(dict.fromkeys(surface))


def _is_reference_only_query(question: str) -> bool:
    """Detect follow-ups whose antecedent is carried entirely by conversation state."""

    normalized = question.casefold().strip()
    return any(cue in normalized for cue in _REFERENCE_ONLY_CUES)


def _field(item: Any, name: str, default: Any = "") -> Any:
    if isinstance(item, dict):
        value = item.get(name, default)
    else:
        value = getattr(item, name, default)
    return default if value is None and default != "" else value


def _identity(item: Any) -> str:
    message_id = _field(item, "message_id", "")
    if message_id:
        return f"id:{message_id}"
    return f"raw:{_field(item, 'role', '')}:{_field(item, 'created_at', '')}:{_field(item, 'content', '')}"


def _group_turns(history: Sequence[Any], roles: tuple[str, ...]) -> list[_TurnCandidate]:
    """Group current messages by durable turn ID with a legacy adjacency fallback."""

    grouped: dict[str, list[Any]] = {}
    metadata: dict[str, tuple[str | None, int]] = {}
    legacy_active_key: str | None = None

    for order, item in enumerate(history):
        role = str(_field(item, "role", ""))
        content = str(_field(item, "content", "")).strip()
        if role not in roles or not content:
            continue
        turn_id = str(_field(item, "turn_id", "") or "") or None
        if turn_id:
            key = f"turn:{turn_id}"
            legacy_active_key = None
        elif role == "user":
            key = f"legacy:{_identity(item)}"
            legacy_active_key = key
        elif role == "assistant" and legacy_active_key is not None:
            key = legacy_active_key
        else:
            key = f"legacy:{_identity(item)}"
            legacy_active_key = None
        if key not in grouped:
            grouped[key] = []
            metadata[key] = (turn_id, order)
        grouped[key].append(item)

    return [
        _TurnCandidate(
            key=key,
            turn_id=metadata[key][0],
            order=metadata[key][1],
            messages=tuple(messages),
            text="\n".join(str(_field(message, "content", "")) for message in messages),
        )
        for key, messages in grouped.items()
    ]


def _truncate_start(content: str, limit: int) -> str:
    """Keep the opening when a paired message has no direct lexical match."""

    if len(content) <= limit:
        return content
    if limit <= len(_TRAILING_TRUNCATION_MARKER):
        return _TRAILING_TRUNCATION_MARKER[:limit]
    return content[: limit - len(_TRAILING_TRUNCATION_MARKER)] + _TRAILING_TRUNCATION_MARKER


def _truncate_recent_end(content: str, limit: int) -> str:
    """Keep the recent end when one complete recent turn exceeds its budget."""

    if len(content) <= limit:
        return content
    if limit <= len(_LEADING_TRUNCATION_MARKER):
        return _LEADING_TRUNCATION_MARKER[-limit:]
    return _LEADING_TRUNCATION_MARKER + content[-(limit - len(_LEADING_TRUNCATION_MARKER)) :]


def _truncate_head_and_tail(content: str, limit: int) -> str:
    """Keep both setup and final conclusion when no lexical anchor is available."""

    if len(content) <= limit:
        return content
    if limit <= len(_MIDDLE_TRUNCATION_MARKER):
        return _MIDDLE_TRUNCATION_MARKER[:limit]
    remaining = limit - len(_MIDDLE_TRUNCATION_MARKER)
    tail = max(1, remaining // 3)
    head = remaining - tail
    return content[:head] + _MIDDLE_TRUNCATION_MARKER + content[-tail:]


def _excerpt_around_matches(content: str, matched_terms: Sequence[str], limit: int) -> str:
    """Keep a bounded excerpt containing the densest available lexical match."""

    if len(content) <= limit:
        return content
    lowered = content.lower()
    positions = sorted(
        position
        for term in matched_terms
        if (position := lowered.find(term.lower())) >= 0
    )
    if not positions:
        return _truncate_start(content, limit)

    content_budget = max(1, limit - len(_LEADING_TRUNCATION_MARKER) - len(_TRAILING_TRUNCATION_MARKER))
    anchor = max(
        positions,
        key=lambda candidate: sum(abs(position - candidate) <= content_budget // 2 for position in positions),
    )
    start = max(0, anchor - content_budget // 3)
    end = min(len(content), start + content_budget)
    start = max(0, end - content_budget)
    excerpt = content[start:end]
    prefix = _LEADING_TRUNCATION_MARKER if start else ""
    suffix = _TRAILING_TRUNCATION_MARKER if end < len(content) else ""
    return f"{prefix}{excerpt}{suffix}"[:limit]


def _excerpt_around_matches_with_tail(
    content: str,
    matched_terms: Sequence[str],
    limit: int,
) -> str:
    """Keep the lexical anchor and the answer tail inside one hard limit."""

    primary = _excerpt_around_matches(content, matched_terms, limit)
    if len(content) <= limit or not primary.endswith(_TRAILING_TRUNCATION_MARKER):
        return primary
    if limit <= len(_MIDDLE_TRUNCATION_MARKER) + 2:
        return primary
    available = limit - len(_MIDDLE_TRUNCATION_MARKER)
    tail_budget = max(1, available // 3)
    anchor_budget = available - tail_budget
    anchor = primary[: -len(_TRAILING_TRUNCATION_MARKER)][:anchor_budget]
    return (anchor + _MIDDLE_TRUNCATION_MARKER + content[-tail_budget:])[:limit]


def _bm25_scores(
    query_terms: Sequence[str],
    documents: Sequence[list[str]],
) -> tuple[list[tuple[float, list[str]]], dict[str, float], Counter[str]]:
    """Score candidates and expose corpus statistics for relevance gating."""

    total = len(documents)
    if not total:
        return [], {}, Counter()
    lengths = [len(document) for document in documents]
    average_length = sum(lengths) / total or 1.0
    frequencies = [Counter(document) for document in documents]
    document_frequency: Counter[str] = Counter()
    for counts in frequencies:
        document_frequency.update(counts.keys())
    idf = {
        term: math.log(1 + (total - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
        for term in query_terms
    }

    scored: list[tuple[float, list[str]]] = []
    for index, counts in enumerate(frequencies):
        score = 0.0
        matched: list[str] = []
        for term in query_terms:
            occurrences = counts.get(term, 0)
            if not occurrences:
                continue
            normalization = 1 - _BM25_B + _BM25_B * (lengths[index] / average_length)
            score += idf[term] * (occurrences * (_BM25_K1 + 1)) / (
                occurrences + _BM25_K1 * normalization
            )
            matched.append(term)
        scored.append((score, matched))
    return scored, idf, document_frequency


def _strong_identifier_terms(query_terms: Sequence[str]) -> set[str]:
    """Return exact technical identifiers that are meaningful by themselves."""

    return {
        term
        for term in query_terms
        if not term.startswith(_SEMANTIC_TERM_PREFIX)
        and term[0].isascii()
        and ("_" in term or "-" in term or any(character.isdigit() for character in term))
    }


def _relevance_gate(
    *,
    query_terms: Sequence[str],
    matched_terms: Sequence[str],
    document_frequency: Counter[str],
    document_count: int,
) -> tuple[bool, float]:
    """Reject weak overlap that consists only of domain-wide vocabulary."""

    if not matched_terms:
        return False, 0.0
    semantic_query = {term for term in query_terms if term.startswith(_SEMANTIC_TERM_PREFIX)}
    semantic_matches = semantic_query.intersection(matched_terms)
    if len(semantic_matches) >= 2 and len(semantic_matches) / len(semantic_query) >= 0.5:
        return True, len(semantic_matches) / len(semantic_query)
    strong_identifiers = _strong_identifier_terms(query_terms)
    if strong_identifiers.intersection(matched_terms):
        return True, 1.0

    matched = set(matched_terms)
    coverage = len(matched) / len(query_terms)
    required_matches = 1 if len(query_terms) <= 2 else 2
    if len(matched) < required_matches or coverage < _MIN_QUERY_COVERAGE:
        return False, coverage

    common_terms = {
        term
        for term in query_terms
        if document_count >= 4 and document_frequency[term] / document_count > _COMMON_TERM_DOCUMENT_RATIO
    }
    if matched.issubset(common_terms) and (
        len(query_terms) <= 2 or coverage < _COMMON_ONLY_QUERY_COVERAGE
    ):
        return False, coverage
    return True, coverage


def _candidate_focus_text(candidate: _TurnCandidate) -> str:
    """Prefer the user's request when deriving a turn-level reference focus."""

    user_text = "\n".join(
        str(_field(message, "content", ""))
        for message in candidate.messages
        if str(_field(message, "role", "")) == "user"
    )
    return user_text or candidate.text


def _reference_focus_candidate(
    candidates: Sequence[_TurnCandidate],
    all_turns: Sequence[_TurnCandidate],
) -> tuple[float, _TurnCandidate] | None:
    """Resolve a pure reference only when one earlier task is clearly salient.

    This is a conservative fallback for phrases such as "继续按刚才那个做". It
    derives a structured focus score from rare terms and task-setting language
    across the complete available transcript. If two topics are similarly
    plausible, retrieval stays empty so the dialogue model can ask for clarity.
    """

    if not candidates:
        return None
    all_documents = [_retrieval_terms(_candidate_focus_text(turn)) for turn in all_turns]
    document_frequency: Counter[str] = Counter()
    for document in all_documents:
        document_frequency.update(set(document))
    document_count = max(1, len(all_documents))

    scored: list[tuple[float, int, _TurnCandidate]] = []
    for candidate in candidates:
        focus_text = _candidate_focus_text(candidate)
        terms = set(_retrieval_terms(focus_text))
        weights = sorted(
            (
                math.log(1 + (document_count - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
                for term in terms
                if not term.startswith(_SEMANTIC_TERM_PREFIX)
            ),
            reverse=True,
        )
        information = sum(weights[:8])
        action = min(3, sum(cue in focus_text.casefold() for cue in _FOCUS_ACTION_CUES)) * 1.25
        scored.append((information + action, candidate.order, candidate))
    scored.sort(key=lambda entry: (-entry[0], -entry[1]))
    best_score, _order, best = scored[0]
    if best_score < 4.0:
        return None
    if len(scored) > 1:
        second_score = scored[1][0]
        if best_score < second_score + 2.0 and best_score < second_score * 1.35:
            return None
    return best_score, best


def _project_turn(
    candidate: _TurnCandidate,
    *,
    score: float,
    matched_terms: list[str],
    query_coverage: float,
    character_budget: int,
) -> RetrievedTurn | None:
    """Project every message in one selected turn inside an atomic budget."""

    if not candidate.messages or character_budget < len(candidate.messages):
        return None
    per_message_limit = min(
        MAX_RETRIEVED_MESSAGE_CHARACTERS,
        max(1, character_budget // len(candidate.messages)),
    )
    projected: list[RetrievedMessage] = []
    for item in candidate.messages:
        content = str(_field(item, "content", ""))
        message_terms = _surface_terms(content, matched_terms)
        excerpt = (
            _excerpt_around_matches_with_tail(content, message_terms, per_message_limit)
            if message_terms
            else _truncate_head_and_tail(content, per_message_limit)
        )
        projected.append(
            RetrievedMessage(
                message_id=_field(item, "message_id", "") or None,
                turn_id=_field(item, "turn_id", "") or None,
                role=str(_field(item, "role", "")),
                content=excerpt,
                created_at=str(_field(item, "created_at", "")),
                episode_id=_field(item, "episode_id", "") or None,
            )
        )
    return RetrievedTurn(
        turn_id=candidate.turn_id,
        messages=tuple(projected),
        score=score,
        matched_terms=matched_terms[:8],
        query_coverage=query_coverage,
    )


def _history_before_current_turn(
    history: Sequence[Any],
    *,
    question: str,
    current_turn_id: str | None,
) -> list[Any]:
    """Remove the current question because the prompt carries it separately."""

    if current_turn_id:
        return [item for item in history if str(_field(item, "turn_id", "") or "") != current_turn_id]
    prior = list(history)
    if (
        prior
        and str(_field(prior[-1], "role", "")) == "user"
        and str(_field(prior[-1], "content", "")).strip() == question.strip()
    ):
        prior.pop()
    return prior


def bounded_recent_turn_history(
    history: Sequence[Any],
    *,
    max_turns: int,
    max_characters: int | None = DEFAULT_RECENT_TURN_CHARACTER_BUDGET,
    max_messages: int | None = None,
    roles: tuple[str, ...] = ("user", "assistant"),
) -> list[Any]:
    """Return recent complete turns without splitting a user/assistant pair."""

    if max_turns < 1 or (max_characters is not None and max_characters < 1):
        return []
    turns = _group_turns(history, roles)
    selected: list[tuple[Any, ...]] = []
    used = 0
    used_messages = 0
    for turn in reversed(turns):
        if len(selected) >= max_turns:
            break
        if max_messages is not None and used_messages and used_messages + len(turn.messages) > max_messages:
            break
        if max_messages is not None and not used_messages and len(turn.messages) > max_messages:
            # A pathological turn larger than the whole window is still safer
            # intact than as a user-less or assistant-less fragment.
            selected.append(turn.messages)
            break
        remaining = max_characters - used if max_characters is not None else None
        if remaining is not None and remaining <= 0:
            break
        turn_characters = sum(len(str(_field(item, "content", ""))) for item in turn.messages)
        if remaining is None or turn_characters <= remaining:
            selected.append(turn.messages)
            used += turn_characters
            used_messages += len(turn.messages)
            continue
        if not selected and turn.messages and remaining >= len(turn.messages):
            per_message_limit = max(1, remaining // len(turn.messages))
            selected.append(
                tuple(
                    _with_content(
                        item,
                        _truncate_recent_end(str(_field(item, "content", "")), per_message_limit),
                    )
                    for item in turn.messages
                )
            )
        break
    return [message for turn in reversed(selected) for message in turn]


def select_persisted_conversation_history(
    history: Sequence[Any],
    *,
    question: str,
    current_turn_id: str | None = None,
    max_messages: int,
    retrieved_turns: int = DEFAULT_RETRIEVED_TURNS,
) -> list[Any]:
    """Select bounded complete turns for the next Graph checkpoint.

    The complete desktop transcript is the candidate source. Relevant earlier
    turns are selected first with the existing lexical retriever, then the
    remaining capacity is filled with the newest complete turns. Selection uses
    bounded retrieval projections for ranking, while the checkpoint retains the
    original messages of every selected turn; prompt construction applies its
    own character budgets later.
    """

    if max_messages < 1:
        return []
    prior = _history_before_current_turn(
        history,
        question=question,
        current_turn_id=current_turn_id,
    )
    if not prior:
        return []

    recent_prompt_window = bounded_recent_turn_history(
        prior,
        max_turns=4,
        max_characters=DEFAULT_RECENT_TURN_CHARACTER_BUDGET,
    )
    retrieved = retrieve_related(
        prior,
        question=question,
        exclude=recent_prompt_window,
        limit=retrieved_turns,
        max_characters=DEFAULT_RETRIEVAL_CHARACTER_BUDGET,
    )
    candidates = _group_turns(prior, ("user", "assistant"))
    candidate_by_key = {candidate.key: candidate for candidate in candidates}
    selected: dict[str, tuple[Any, ...]] = {}
    used_messages = 0

    for turn in retrieved:
        if turn.turn_id:
            key = f"turn:{turn.turn_id}"
        else:
            message_ids = {message.message_id for message in turn.messages if message.message_id}
            key = next(
                (
                    candidate.key
                    for candidate in candidates
                    if any(str(_field(item, "message_id", "")) in message_ids for item in candidate.messages)
                ),
                "",
            )
        if not key or key not in candidate_by_key or key in selected:
            continue
        original_messages = candidate_by_key[key].messages
        if used_messages and used_messages + len(original_messages) > max_messages:
            continue
        if not used_messages and len(original_messages) > max_messages:
            selected[key] = original_messages
            used_messages = len(original_messages)
            break
        selected[key] = original_messages
        used_messages += len(original_messages)

    for candidate in reversed(candidates):
        if candidate.key in selected:
            continue
        if used_messages and used_messages + len(candidate.messages) > max_messages:
            continue
        if not used_messages and len(candidate.messages) > max_messages:
            selected[candidate.key] = candidate.messages
            used_messages = len(candidate.messages)
            break
        selected[candidate.key] = candidate.messages
        used_messages += len(candidate.messages)
        if used_messages >= max_messages:
            break

    return [
        message
        for candidate in candidates
        if candidate.key in selected
        for message in selected[candidate.key]
    ]


def select_conversation_context(
    history: Sequence[Any],
    *,
    question: str,
    current_turn_id: str | None = None,
    recent_turns: int = 4,
    retrieved_turns: int = DEFAULT_RETRIEVED_TURNS,
    recent_max_characters: int = DEFAULT_RECENT_TURN_CHARACTER_BUDGET,
    retrieved_max_characters: int = DEFAULT_RETRIEVAL_CHARACTER_BUDGET,
) -> tuple[list[Any], list[RetrievedTurn]]:
    """Split one history into complete recent turns and relevant earlier turns."""

    prior = _history_before_current_turn(
        history,
        question=question,
        current_turn_id=current_turn_id,
    )
    recent = bounded_recent_turn_history(
        prior,
        max_turns=recent_turns,
        max_characters=recent_max_characters,
    )
    earlier = retrieve_related(
        prior,
        question=question,
        exclude=recent,
        limit=retrieved_turns,
        max_characters=retrieved_max_characters,
    )
    return recent, earlier


def retrieve_related(
    history: Sequence[Any],
    *,
    question: str,
    exclude: Iterable[Any] = (),
    limit: int = DEFAULT_RETRIEVED_TURNS,
    max_characters: int = DEFAULT_RETRIEVAL_CHARACTER_BUDGET,
    roles: tuple[str, ...] = ("user", "assistant"),
) -> list[RetrievedTurn]:
    """Select complete earlier turns most related to ``question``.

    ``exclude`` is normally the recent window already being sent. If any
    message from a turn is recent, the whole turn is excluded so retrieval never
    adds a duplicate half-turn. Weak overlap made only of corpus-wide terms is
    rejected, while exact technical identifiers remain eligible.
    """

    if limit < 1 or max_characters < 1:
        return []
    query_terms = _retrieval_terms(question)
    if not query_terms:
        return []

    excluded = {_identity(item) for item in exclude}
    candidates = [
        candidate
        for candidate in _group_turns(history, roles)
        if not any(_identity(message) in excluded for message in candidate.messages)
    ]
    if not candidates:
        return []

    documents = [_retrieval_terms(candidate.text) for candidate in candidates]
    scored, _idf, document_frequency = _bm25_scores(query_terms, documents)
    ranked: list[tuple[float, float, int, _TurnCandidate, list[str]]] = []
    for candidate, (score, matched) in zip(candidates, scored, strict=True):
        eligible, coverage = _relevance_gate(
            query_terms=query_terms,
            matched_terms=matched,
            document_frequency=document_frequency,
            document_count=len(candidates),
        )
        if eligible and score > 0:
            ranked.append((score, coverage, candidate.order, candidate, matched))
    ranked.sort(key=lambda entry: (-entry[0], -entry[1], -entry[2]))

    if not ranked and _is_reference_only_query(question):
        focus = _reference_focus_candidate(
            candidates,
            _group_turns(history, roles),
        )
        if focus is not None:
            score, candidate = focus
            projected = _project_turn(
                candidate,
                score=score,
                matched_terms=[_REFERENCE_FOCUS_TERM],
                query_coverage=0.0,
                character_budget=max_characters,
            )
            return [projected] if projected is not None else []

    selected: list[tuple[int, RetrievedTurn]] = []
    used = 0
    for score, coverage, order, candidate, matched in ranked:
        if len(selected) >= limit:
            break
        remaining = max_characters - used
        if remaining <= 0:
            break
        projected = _project_turn(
            candidate,
            score=score,
            matched_terms=matched,
            query_coverage=coverage,
            character_budget=remaining,
        )
        if projected is None:
            continue
        used += sum(len(message.content) for message in projected.messages)
        selected.append((order, projected))
    selected.sort(key=lambda entry: entry[0])
    return [turn for _order, turn in selected]
