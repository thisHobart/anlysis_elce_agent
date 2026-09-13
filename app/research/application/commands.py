"""Translate optional text shortcuts into typed interrupt commands.

Buttons and other structured UI actions should call ``ResearchCoordinator.resume``
directly.  This module only handles the small set of unambiguous shortcuts users
may type into the conversation composer; all other text remains a conversational
message and follows the interrupt's contextual default route.
"""

from __future__ import annotations

from collections.abc import Iterable

from app.research.graph.contracts import InterruptKind, ResumeAction

_APPROVE = frozenset(
    {
        "接受",
        "同意",
        "确认",
        "确认执行",
        "立即执行",
        "按这个执行",
        "执行当前方案",
        "直接用你给的方案研究",
        "可以就这么跑吧",
        "可以就这样执行",
        "可以开始",
    }
)
_REJECT = frozenset({"拒绝", "不同意", "不执行", "不要执行", "取消方案", "放弃方案"})
_STOP = frozenset(
    {
        "停止",
        "结束",
        "终止",
        "退出",
        "结束研究",
        "停止研究",
        "接受当前限制",
        "接受这些限制",
        "保留当前结果并结束",
        "不用了",
        "算了",
    }
)
_RETRY = frozenset({"重试", "重新尝试", "再试一次"})
_NEXT_ROUND = frozenset({"下一轮", "开始下一轮", "新一轮", "开始新一轮", "继续下一轮"})


def normalize_command_text(text: str) -> str:
    """Normalize a short command without interpreting arbitrary prose."""

    compact = "".join(text.split()).casefold()
    return compact.translate(str.maketrans("", "", "，,。！？!?"))


def _allowed(action: ResumeAction, choices: set[str]) -> ResumeAction | None:
    return action if action in choices else None


def _explicit_action(
    *,
    kind: InterruptKind,
    normalized: str,
    choices: set[str],
) -> ResumeAction | None:
    if normalized in _APPROVE:
        return _allowed("approve", choices) or (
            _allowed("clarify", choices) if kind == "plan_error" else None
        )
    if normalized in _RETRY:
        return _allowed("retry", choices)
    if normalized in _NEXT_ROUND:
        return _allowed("next_round", choices) or _allowed("followup", choices)
    if normalized in _REJECT:
        return _allowed("reject", choices) or _allowed("stop", choices)
    if normalized in _STOP:
        # A plan-approval interrupt expresses cancellation as ``reject``; all
        # other gates use the global ``stop`` action.
        if kind == "plan_approval":
            return _allowed("reject", choices)
        return _allowed("stop", choices) or _allowed("reject", choices)
    return None


def resolve_text_action(
    *,
    kind: InterruptKind,
    choices: Iterable[ResumeAction],
    message: str,
) -> ResumeAction:
    """Resolve a typed message at an interrupt without guessing its intent.

    Only exact, unambiguous command shortcuts become control actions.  Ordinary
    text is routed back through the model as plan discussion/feedback or result
    follow-up, depending on the active interrupt.
    """

    available = set(choices)
    explicit = _explicit_action(kind=kind, normalized=normalize_command_text(message), choices=available)
    if explicit is not None:
        return explicit

    contextual_defaults: dict[InterruptKind, ResumeAction] = {
        "plan_approval": "followup",
        "plan_error": "modify",
        "result_limitations": "followup",
        "result_rejected": "modify",
        "result": "followup",
        "response_error": "retry",
        "finalization_error": "retry",
    }
    default = contextual_defaults[kind]
    if default in available:
        return default
    if kind == "result_limitations" and "modify" in available:
        # Checkpoints created before result follow-ups were added expose only
        # modify at this gate. Preserve their previous recovery behavior.
        return "modify"
    # Interrupt contracts are validated, but retaining a deterministic fallback
    # keeps older persisted payloads recoverable during schema migration.
    return next(iter(choices))
