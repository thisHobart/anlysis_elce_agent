"""Translate auditable loop events into the wording shown in the research trace.

The research loop stores observable, deterministic events. This module is the one
place that turns those records into the sentences shown in the desktop timeline,
the run trace, and progress messages. It never invents evidence: every line is
derived from an event that already happened.

Wording follows one convention throughout, so the trace reads as an audit log
rather than a chat transcript:

* verb first, no trailing punctuation, no second person;
* work in flight uses ``分析中 · <名称>``, finished work uses ``已完成 · <名称>``;
* a running step carries the function's technical description, a finished step
  carries only its elapsed time, so the same sentence never appears twice.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.research.tools.catalog import FUNCTION_CATALOG, STAGE_TITLES

STAGE_LABELS: dict[str, str] = {
    "setup": "准备",
    "read": "解析问题",
    "design": "生成方案",
    "confirm": "等待确认",
    "compute": "执行分析",
    "review": "评估结果",
    "answer": "生成结论",
    "issue": "处理异常",
}

INTENT_LABELS: dict[str, str] = {
    "new_plan": "生成新的分析方案",
    "revise_plan": "按反馈修订现有方案",
    "execute_plan": "执行当前方案",
    "discussion": "讨论研究方法",
    "reply": "直接回复",
    "explain_result": "解释已有结果",
    "need_user": "需要补充信息",
}

DECISION_LABELS: dict[str, str] = {
    "accept": "证据充分，本轮结论已收敛",
    "revise": "在授权范围内继续补充证据",
    "need_user": "需要用户决策",
    "reject": "本轮结果不可用",
}

APPROVAL_ACTION_LABELS: dict[str, str] = {
    "approve": "方案已确认",
    "timeout_accept": "确认超时，自动执行方案",
    "modify": "收到修改意见",
    "reject": "方案被拒绝",
    "stop": "研究已终止",
    "retry": "请求重试",
    "followup": "收到追问",
    "clarify": "收到补充说明",
    "accept_limitations": "已接受当前结论限制",
}

FAILURE_TITLES: dict[str, str] = {
    "主 Agent 调用失败": "模型调用失败",
    "Skill 校验失败": "研究方法校验失败",
    "方案生成失败": "方案生成失败",
    "锁定方案失败": "方案锁定失败",
    "准备函数调用失败": "分析准备失败",
    "函数执行失败": "分析执行失败",
    "函数结果校验失败": "结果校验失败",
    "函数结果状态损坏": "结果状态异常",
    "函数重试耗尽": "重试次数耗尽",
    "最终合并失败": "证据汇总失败",
    "结果解释失败": "结论生成失败",
    "研究回复失败": "回复生成失败",
    "评估决策无效": "评估结果无效",
    "拒绝越界审批操作": "拒绝越权操作",
    "拒绝越界用户操作": "拒绝越权操作",
    "拒绝越界结果操作": "拒绝越权操作",
}

_FUNCTION_PATTERN = re.compile(r"^(?P<title>.+?)\s*\[(?P<name>[a-z0-9_]+)\]$")
_CHINESE_PATTERN = re.compile(r"[一-鿿]")


@dataclass(frozen=True)
class ThinkingStep:
    """One line in the Agent's visible research trace."""

    stage: str
    title: str
    detail: str = ""
    status: str = "completed"
    function_name: str | None = None

    @property
    def stage_label(self) -> str:
        return STAGE_LABELS.get(self.stage, "研究")


def _split(name: str) -> tuple[str, str]:
    head, separator, tail = name.partition("：")
    return (head, tail.strip()) if separator else (name, "")


def _function_parts(tail: str) -> tuple[str, str | None, str]:
    """Split ``标题 [function_name] · 附加说明`` into its readable pieces."""

    payload, _separator, extra = tail.partition(" · ")
    match = _FUNCTION_PATTERN.match(payload.strip())
    if match is None:
        return payload.strip(), None, extra.strip()
    return match.group("title"), match.group("name"), extra.strip()


def _duration(details: dict[str, Any]) -> str:
    value = details.get("duration_ms")
    if not isinstance(value, (int, float)):
        return ""
    return f"{value / 1000:.1f} 秒" if value >= 1000 else f"{value:.0f} 毫秒"


def _method(function_name: str | None) -> str:
    """Return the registered technical description of one research function."""

    spec = FUNCTION_CATALOG.get(function_name or "")
    return spec.description.rstrip("。") if spec is not None else ""


def _queue_composition(functions: list[Any], total: Any) -> str:
    """Name a locked batch by the research stages it covers, not just its size."""

    grouped: dict[str, int] = {}
    for function_name in functions:
        spec = FUNCTION_CATALOG.get(str(function_name))
        if spec is None:
            continue
        label = STAGE_TITLES.get(spec.stage, spec.stage)
        grouped[label] = grouped.get(label, 0) + 1
    if not grouped:
        return f"共 {total} 项分析" if total else ""
    breakdown = "、".join(f"{label} {count} 项" for label, count in grouped.items())
    return f"{breakdown}，共 {total or sum(grouped.values())} 项"


def _failure(head: str, tail: str, details: dict[str, Any]) -> ThinkingStep:
    title, function_name, extra = _function_parts(tail)
    label = FAILURE_TITLES.get(head, head)
    reason = extra or str(details.get("error", "")) or tail
    if function_name is not None:
        return ThinkingStep("issue", f"{label} · {title}", reason, "failed", function_name)
    return ThinkingStep("issue", label, reason, "failed")


def narrate_event(event: dict[str, Any]) -> ThinkingStep:
    """Return the trace line for one recorded loop event."""

    name = str(event.get("name", ""))
    status = str(event.get("status", "completed"))
    details = event.get("details") if isinstance(event.get("details"), dict) else {}
    head, tail = _split(name)

    if status == "failed" or head.endswith(("失败", "损坏", "耗尽")) or head.startswith("拒绝"):
        return _failure(head, tail, details)

    if head in {"接收研究问题", "提交研究问题"}:
        return ThinkingStep("read", "解析研究问题", tail, status)
    if head == "主 Agent 路由":
        intent = str(details.get("intent", tail))
        return ThinkingStep("read", "识别处理方式", INTENT_LABELS.get(intent, intent), status)
    if head == "开始研究 Episode":
        _identifier, _separator, goal = tail.partition(" · ")
        return ThinkingStep("read", "启动研究目标", goal or tail, status)

    if head.startswith("激活 Skill"):
        skill = str(details.get("skill", ""))
        protocol = details.get("research_protocol") or {}
        order = "按领域协议固定阶段顺序执行" if protocol else "按方法包声明的流程执行"
        return ThinkingStep("design", "加载研究方法", f"{skill or tail}；{order}", status)
    if head == "生成候选方案":
        _identifier, _separator, objective = tail.partition(" · ")
        return ThinkingStep("design", "生成分析方案", objective or "已产出候选方案", status)
    if head == "生成修订方案":
        revision = details.get("revision")
        functions = details.get("functions") or []
        detail = f"第 {revision} 版，共 {len(functions)} 项分析" if revision else tail
        return ThinkingStep("design", "修订分析方案", detail, status)
    if head == "确认已有方案":
        return ThinkingStep("design", "沿用现有方案", "方案内容未变更，直接进入执行", status)
    if head == "方案校验通过":
        return ThinkingStep("design", "方案校验通过", "步骤、变量与参数均在授权范围内", status)
    if head == "锁定函数队列":
        composition = _queue_composition(details.get("functions") or [], details.get("calls"))
        return ThinkingStep("design", "锁定执行计划", composition or tail, status)

    if head == "等待审批":
        return ThinkingStep("confirm", "等待方案确认", "超时未收到修改意见将自动执行", "running")
    if head == "方案操作":
        action = str(details.get("action", tail.split(" · ")[0]))
        return ThinkingStep("confirm", APPROVAL_ACTION_LABELS.get(action, action), "", status)
    if head == "等待用户处理":
        return ThinkingStep("confirm", "等待用户决策", tail.split(" · ")[-1], "running")

    if head in {"准备执行函数", "函数执行完成", "函数结果校验通过", "复用函数结果"}:
        title, function_name, extra = _function_parts(tail)
        if head == "准备执行函数":
            return ThinkingStep(
                "compute", f"分析中 · {title}", extra or _method(function_name), "running", function_name
            )
        if head == "复用函数结果":
            return ThinkingStep(
                "compute", f"已复用 · {title}", "数据与参数未变更，沿用既有结果", "completed", function_name
            )
        elapsed = _duration(details) if head == "函数执行完成" else ""
        return ThinkingStep("compute", f"已完成 · {title}", elapsed, "completed", function_name)

    if head == "评估运行":
        decision = str(details.get("decision", tail.split("→")[-1].strip()))
        return ThinkingStep("review", "评估分析证据", DECISION_LABELS.get(decision, decision), status)
    if head == "评估触发自动修订":
        return ThinkingStep("review", "触发自动修订", "在已授权范围内补充下一轮证据", status)
    if head == "评估拒绝等待用户决策":
        return ThinkingStep("review", "评估未通过", tail, "warning")

    if head == "解释验证结果":
        return ThinkingStep("answer", "生成结论", "仅引用已验证的确定性证据", status)
    if head == "回答用户":
        return ThinkingStep("answer", "生成回复", tail, status)
    if head.startswith("保存"):
        return ThinkingStep("answer", head, tail, status)

    if not _CHINESE_PATTERN.search(head):
        # An internal identifier reached the trace; keep it auditable without showing it as a step.
        return ThinkingStep("review", "系统事件", name, status)
    return ThinkingStep(stage_for_category(str(event.get("category", ""))), head or "研究进行中", tail, status)


NODE_PROGRESS: dict[str, tuple[int, str, str, str]] = {
    "ingest_user": (3, "read", "解析研究问题", ""),
    "main_agent": (8, "read", "识别处理方式", "在讨论、生成方案、修订方案与执行之间路由"),
    "begin_episode": (10, "read", "启动研究目标", ""),
    "resolve_skill": (14, "design", "加载研究方法", "从已注册的研究方法包中选择"),
    "eda_subagent": (24, "design", "生成分析方案", "选取回答该问题所需的最小分析集合"),
    "revise_user_plan": (26, "design", "修订分析方案", ""),
    "confirm_existing_plan": (30, "design", "沿用现有方案", ""),
    "validate_plan": (34, "design", "校验分析方案", "核对步骤、变量、参数与数据指纹"),
    "prepare_plan_repair": (32, "design", "修复分析方案", "在已授权范围内调整"),
    "prepare_approval": (40, "confirm", "准备方案确认", ""),
    "approval_interrupt": (40, "confirm", "等待方案确认", "超时未收到修改意见将自动执行"),
    "lock_plan": (45, "design", "锁定执行计划", "执行期间不再变更"),
    "execute_tool": (62, "compute", "执行分析函数", ""),
    "validate_tool_result": (72, "compute", "校验分析结果", ""),
    "finalize_iteration": (84, "review", "汇总证据并生成报告", ""),
    "prepare_invalid_evaluation": (85, "review", "重新评估本轮结果", ""),
    "prepare_rejected_result": (86, "review", "本轮结果不可用", "需要用户决定后续调整"),
    "prepare_evaluation_revision": (88, "review", "触发自动修订", ""),
    "prepare_need_user": (98, "confirm", "等待用户决策", ""),
    "explain_result": (96, "answer", "生成结论", ""),
    "reply": (96, "answer", "生成回复", ""),
    "user_interrupt": (98, "confirm", "等待用户决策", ""),
    "result_interrupt": (100, "answer", "本轮研究完成", "可继续追问，或开启新的研究目标"),
    "persist_stop": (100, "answer", "研究已终止", ""),
    "persist_failure": (100, "issue", "本轮研究未完成", ""),
}

# Bookkeeping nodes that move the cursor without doing anything a user needs to see.
SILENT_NODES = frozenset({"advance_tool", "mark_tool_running"})


def narrate_node(node_name: str) -> tuple[int, ThinkingStep]:
    """Return the coarse progress used when a node emits no event of its own."""

    value, stage, title, detail = NODE_PROGRESS.get(node_name, (50, "", "研究进行中", ""))
    return value, ThinkingStep(stage, title, detail, "running")


TRACE_CATEGORIES: dict[str, str] = {
    "setup": "session",
    "read": "agent",
    "design": "plan",
    "confirm": "plan",
    "compute": "tool",
    "review": "evaluation",
    "answer": "agent",
    "issue": "error",
}


CATEGORY_STAGES: dict[str, str] = {
    "session": "setup",
    "input": "setup",
    "user": "read",
    "agent": "read",
    "plan": "design",
    "tool": "compute",
    "evaluation": "review",
    "artifact": "answer",
    "error": "issue",
}


def trace_category(stage: str) -> str:
    """Map a narration stage onto the trace filter category it belongs to."""

    return TRACE_CATEGORIES.get(stage, "agent")


def stage_for_category(category: str) -> str:
    """Map a recorded trace category back onto a narration stage."""

    return CATEGORY_STAGES.get(category, "review")


def progress_message(step: ThinkingStep, source_event: str = "") -> str:
    """Encode one progress update, keeping the raw event name for the audit trace.

    Layout: ``event<TAB>stage<TAB>function<TAB>title<NEWLINE>detail``. The desktop
    records ``event`` verbatim so the run trace narrates exactly once, and reuses the
    decoded step for the live thinking card.
    """

    return f"{source_event}\t{step.stage}\t{step.function_name or ''}\t{step.title}\n{step.detail}"


def split_progress_message(message: str) -> tuple[str, ThinkingStep]:
    """Decode a progress update into its raw event name and its narrated step."""

    head, _newline, detail = message.partition("\n")
    fields = head.split("\t")
    if len(fields) < 4:
        return "", ThinkingStep("", head, detail, "running")
    source_event, stage, function_name = fields[0], fields[1], fields[2]
    title = "\t".join(fields[3:])
    return source_event, ThinkingStep(stage, title, detail, "running", function_name or None)
