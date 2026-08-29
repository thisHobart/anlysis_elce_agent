"""Typed command shortcuts at persistent Graph interrupts."""

from __future__ import annotations

import pytest

from app.research.application.commands import resolve_text_action
from app.research.graph.contracts import InterruptKind, ResumeAction


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("接受", "approve"),
        ("可以，就这么跑吧", "approve"),
        ("拒绝", "reject"),
        ("不同意", "reject"),
        ("停止", "reject"),
        ("为什么选择这个函数？", "followup"),
    ],
)
def test_plan_approval_text_shortcuts_never_turn_rejection_into_modification(
    message: str,
    expected: ResumeAction,
) -> None:
    assert (
        resolve_text_action(
            kind="plan_approval",
            choices=["approve", "modify", "reject", "followup"],
            message=message,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("kind", "choices", "message", "expected"),
    [
        ("result", ["followup", "next_round", "stop"], "结束研究", "stop"),
        ("result", ["followup", "next_round", "stop"], "开始下一轮", "next_round"),
        ("plan_error", ["retry", "modify", "clarify", "stop"], "重试", "retry"),
        ("plan_error", ["retry", "modify", "clarify", "stop"], "接受", "clarify"),
        (
            "result_limitations",
            ["modify", "followup", "stop"],
            "按推荐变量继续深入分析",
            "followup",
        ),
        (
            "result_limitations",
            ["modify", "followup", "stop"],
            "为什么这一轮没有图表？",
            "followup",
        ),
        (
            "result_limitations",
            ["modify", "followup", "stop"],
            "接受当前限制",
            "stop",
        ),
    ],
)
def test_interrupt_shortcuts_respect_the_gate_contract(
    kind: InterruptKind,
    choices: list[ResumeAction],
    message: str,
    expected: ResumeAction,
) -> None:
    assert resolve_text_action(kind=kind, choices=choices, message=message) == expected
