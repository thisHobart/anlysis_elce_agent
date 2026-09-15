"""Typed widgets rendered inside one continuous research conversation."""

from __future__ import annotations

import json
from html import escape
from typing import Any

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.desktop.report_view import open_report
from app.research.agent.schemas import AgentRunResult, EDAPlan, EDAResearchScope
from app.research.data.sources.summary import DataSummary, parse_summary
from app.research.forecasting.contracts import ForecastPlan
from app.research.tools.catalog import FUNCTION_CATALOG

STATUS_MARK = {
    "running": "◐",
    "completed": "✓",
    "warning": "!",
    "failed": "✕",
    "stopped": "■",
    "info": "·",
}


class TextMessageWidget(QFrame):
    """Natural-language user, Agent, or system message."""

    def __init__(self, *, role: str, content: str) -> None:
        super().__init__()
        self.setObjectName("userMessage" if role == "user" else "agentMessage")
        self.setMaximumWidth(680)
        if role == "user":
            self.setMinimumWidth(220)
        else:
            self.setMinimumWidth(520)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 9, 12, 9)
        layout.setSpacing(4)
        role_label = QLabel("你" if role == "user" else "研究助手")
        role_label.setObjectName("messageRole")
        content_label = QLabel(content)
        content_label.setObjectName("messageContent")
        content_label.setWordWrap(True)
        content_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(role_label)
        layout.addWidget(content_label)


class NoticeMessageWidget(QFrame):
    """Quiet timeline notice for input changes, cancellation, or errors."""

    def __init__(self, content: str, *, error: bool = False) -> None:
        super().__init__()
        self.setObjectName("errorNotice" if error else "noticeMessage")
        self.setMaximumWidth(680)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 9, 12, 9)
        label = QLabel(content)
        label.setWordWrap(True)
        layout.addWidget(label)


class P2ReviewMessageWidget(QFrame):
    """Persistent P2 gate card backed by an authoritative review-summary payload."""

    open_requested = Signal()
    continue_requested = Signal(str)

    def __init__(self, payload: dict[str, Any], *, read_only: bool = False) -> None:
        super().__init__()
        self.setObjectName("p2ReviewMessage")
        self.setMaximumWidth(680)
        self.setMinimumWidth(560)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)
        title = QLabel("P2 新闻复核")
        title.setObjectName("resultTitle")
        layout.addWidget(title)
        pending = int(payload.get("required_pending", 0))
        resolved = int(payload.get("required_resolved", 0))
        optional = int(payload.get("optional_unreviewed", 0))
        summary = QLabel(f"必审未处理 {pending} 条 · 已处理 {resolved} 条 · 可选抽查 {optional} 条")
        summary.setWordWrap(True)
        layout.addWidget(summary)
        failures = [str(item) for item in payload.get("quality_failures", [])]
        quality = QLabel("确定性质量检查：" + ("通过" if not failures else "；".join(failures)))
        quality.setWordWrap(True)
        quality.setObjectName("errorNotice" if failures else "messageContent")
        layout.addWidget(quality)
        actions = QHBoxLayout()
        active = payload.get("phase") == "p2_needs_review"
        open_button = QPushButton("打开复核")
        open_button.setEnabled(not read_only and active)
        open_button.clicked.connect(self.open_requested)
        actions.addWidget(open_button)
        report_button = QPushButton("查看 P2 报告")
        report_path = str(payload.get("report_path") or "")
        report_button.setEnabled(bool(report_path))
        report_button.clicked.connect(lambda: open_report(report_path))
        actions.addWidget(report_button)
        actions.addStretch(1)
        continue_button = QPushButton("校验并继续")
        continue_button.setObjectName("primaryButton")
        continue_button.setEnabled(not read_only and active and pending == 0)
        continue_button.clicked.connect(
            lambda: self.continue_requested.emit(str(payload.get("revision") or ""))
        )
        actions.addWidget(continue_button)
        layout.addLayout(actions)


class ThinkingStepRow(QWidget):
    """One visible step of the Agent's working process."""

    def __init__(self, *, stage: str, title: str, detail: str, status: str) -> None:
        super().__init__()
        self.setObjectName("thinkingStep")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 3, 0, 3)
        layout.setSpacing(9)
        self.mark = QLabel(STATUS_MARK.get(status, "·"))
        self.mark.setObjectName("stepMark")
        self.mark.setFixedWidth(14)
        self.mark.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.mark)
        text = QVBoxLayout()
        text.setSpacing(1)
        headline = QHBoxLayout()
        headline.setSpacing(7)
        self.stage_label = QLabel(stage)
        self.stage_label.setObjectName("stepStage")
        headline.addWidget(self.stage_label)
        self.title_label = QLabel(title)
        self.title_label.setObjectName("stepTitle")
        self.title_label.setWordWrap(True)
        self.title_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        headline.addWidget(self.title_label, 1)
        text.addLayout(headline)
        # Parented on construction: setVisible on a parentless widget would flash it as a window.
        self.detail_label = QLabel(detail, self)
        self.detail_label.setObjectName("stepDetail")
        self.detail_label.setWordWrap(True)
        self.detail_label.setVisible(bool(detail))
        text.addWidget(self.detail_label)
        layout.addLayout(text, 1)
        self.set_status(status)

    def set_status(self, status: str) -> None:
        self.mark.setText(STATUS_MARK.get(status, "·"))
        for widget in (self.mark, self.title_label):
            widget.setProperty("traceStatus", status)
            widget.style().unpolish(widget)
            widget.style().polish(widget)

    def update_text(self, *, title: str | None = None, detail: str | None = None) -> None:
        if title is not None:
            self.title_label.setText(title)
        if detail is not None:
            self.detail_label.setText(detail)
            self.detail_label.setVisible(bool(detail))


class ThinkingMessageWidget(QFrame):
    """Collapsible, live view of what the Agent is doing and why."""

    def __init__(
        self,
        *,
        steps: list[dict[str, Any]] | None = None,
        state: str = "running",
        mode: str = "research",
    ) -> None:
        super().__init__()
        self.setObjectName("thinkingMessage")
        self.setMaximumWidth(680)
        self.setMinimumWidth(520)
        self._steps: list[dict[str, Any]] = []
        self._rows: list[ThinkingStepRow] = []
        self._collapsed = False
        self._mode = mode
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 11, 14, 11)
        layout.setSpacing(8)

        header = QHBoxLayout()
        header.setSpacing(8)
        self.toggle_button = QPushButton("▾")
        self.toggle_button.setObjectName("thinkingToggle")
        self.toggle_button.setFixedWidth(22)
        self.toggle_button.clicked.connect(self._toggle)
        header.addWidget(self.toggle_button)
        self.title_label = QLabel("正在思考…" if mode == "dialogue" else "正在研究…")
        self.title_label.setObjectName("thinkingTitle")
        header.addWidget(self.title_label, 1)
        self.status_label = QLabel("")
        self.status_label.setObjectName("thinkingStatus")
        header.addWidget(self.status_label)
        layout.addLayout(header)

        self.steps_container = QWidget()
        self.steps_layout = QVBoxLayout(self.steps_container)
        self.steps_layout.setContentsMargins(2, 0, 0, 0)
        self.steps_layout.setSpacing(2)
        layout.addWidget(self.steps_container)

        for step in steps or []:
            self.add_step(
                stage=str(step.get("stage", "研究")),
                title=str(step.get("title", "")),
                detail=str(step.get("detail", "")),
                status=str(step.get("status", "completed")),
                function_name=step.get("function_name"),
            )
        if state != "running":
            self.finish(state=state)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ThinkingMessageWidget:
        return cls(
            steps=list(payload.get("steps", [])),
            state=str(payload.get("state", "running")),
            mode=str(payload.get("mode", "research")),
        )

    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        self._mode = mode
        self.title_label.setText("正在思考…" if mode == "dialogue" else "正在研究…")

    @property
    def steps(self) -> list[dict[str, Any]]:
        return list(self._steps)

    def replace_current(
        self,
        *,
        stage: str,
        title: str,
        detail: str = "",
        status: str = "running",
        function_name: str | None = None,
        keep_detail: bool = False,
    ) -> None:
        """Update the current step in place instead of adding a near-duplicate line."""

        if not self._rows:
            self.add_step(stage=stage, title=title, detail=detail, status=status, function_name=function_name)
            return
        kept_detail = detail or (str(self._steps[-1].get("detail", "")) if keep_detail else "")
        self._rows[-1].stage_label.setText(stage)
        self._rows[-1].update_text(title=title, detail=kept_detail)
        self._rows[-1].set_status(status)
        self._steps[-1] = {
            "stage": stage,
            "title": title,
            "detail": kept_detail,
            "status": status,
            "function_name": function_name,
        }

    def add_step(
        self,
        *,
        stage: str,
        title: str,
        detail: str = "",
        status: str = "running",
        function_name: str | None = None,
    ) -> None:
        for row in self._rows:
            if row.mark.text() == STATUS_MARK["running"]:
                row.set_status("completed")
        for step in self._steps:
            if step.get("status") == "running":
                step["status"] = "completed"
        row = ThinkingStepRow(stage=stage, title=title, detail=detail, status=status)
        self._rows.append(row)
        self._steps.append(
            {
                "stage": stage,
                "title": title,
                "detail": detail,
                "status": status,
                "function_name": function_name,
            }
        )
        self.steps_layout.addWidget(row)
        self._refresh_status()

    def update_current(self, *, detail: str | None = None, status: str | None = None) -> None:
        if not self._rows:
            return
        self._rows[-1].update_text(detail=detail)
        if detail is not None:
            self._steps[-1]["detail"] = detail
        if status is not None:
            self._rows[-1].set_status(status)
            self._steps[-1]["status"] = status

    def finish(self, *, state: str = "completed", summary: str = "") -> None:
        for row in self._rows:
            if row.mark.text() == STATUS_MARK["running"]:
                row.set_status("completed" if state == "completed" else state)
        for step in self._steps:
            if step.get("status") == "running":
                step["status"] = "completed" if state == "completed" else state
        completed_title = "思考过程" if self._mode == "dialogue" else "研究过程"
        titles = {
            "completed": completed_title,
            "failed": "研究中断",
            "stopped": "已终止",
        }
        self.title_label.setText(titles.get(state, completed_title))
        self.status_label.setText(summary or f"{len(self._steps)} 步")
        self.set_collapsed(True)

    def set_collapsed(self, collapsed: bool) -> None:
        self._collapsed = collapsed
        self.steps_container.setVisible(not collapsed)
        self.toggle_button.setText("▸" if collapsed else "▾")

    def _toggle(self) -> None:
        self.set_collapsed(not self._collapsed)

    def _refresh_status(self) -> None:
        self.status_label.setText(f"{len(self._steps)} 步")


class ToolMessageWidget(QFrame):
    """Compatibility card for conversations saved before the thinking view existed."""

    def __init__(self, title: str, detail: str, *, status: str = "running") -> None:
        super().__init__()
        self.setObjectName("toolMessage")
        self.setMaximumWidth(560)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        text_layout = QVBoxLayout()
        text_layout.setSpacing(2)
        self.title_label = QLabel(title)
        self.title_label.setObjectName("toolTitle")
        self.detail_label = QLabel(detail)
        self.detail_label.setObjectName("toolDetail")
        self.detail_label.setWordWrap(True)
        text_layout.addWidget(self.title_label)
        text_layout.addWidget(self.detail_label)
        layout.addLayout(text_layout, 1)
        self.status_label = QLabel()
        layout.addWidget(self.status_label)
        self.set_status(status)

    def set_status(self, status: str, detail: str | None = None) -> None:
        labels = {
            "running": "进行中",
            "completed": "已完成",
            "warning": "注意",
            "failed": "失败",
            "stopped": "已终止",
        }
        self.status_label.setText(labels.get(status, status))
        self.status_label.setProperty("traceStatus", status)
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)
        if detail is not None:
            self.detail_label.setText(detail)


def summary_fields(summary: DataSummary | None) -> list[tuple[str, str, str]]:
    """Render the dataset as the panel renders it: same source, same wording.

    Returns ``(field name, rich-text value, quiet sub-line)`` so the confirmation
    card and the right-hand panel cannot drift apart.
    """

    summary = summary or DataSummary()
    variables = "、".join(
        escape(item.display)
        if item.resolved
        else f'<span style="color:#B45309">{escape(item.display)}</span>'
        for item in summary.variables
    )
    window = f"{summary.start_date} — {summary.end_date}" if summary.start_date else ""
    sub = summary.granularity_text
    if sub and summary.gap_text:
        sub = f"{sub}，{summary.gap_text}"
    return [
        ("电价", escape(summary.price_label), ""),
        ("影响因素", variables, ""),
        ("时间范围", escape(window), sub),
    ]


def as_summary(value: DataSummary | dict[str, Any] | None) -> DataSummary | None:
    """Accept a live summary or the dict a saved conversation stores."""

    if value is None or isinstance(value, DataSummary):
        return value
    return parse_summary(value)


class DataPlanMessageWidget(QFrame):
    """One confirmation covering both the data and the analysis it will feed.

    Splitting the two would create a state where the data is approved and the
    analysis is not, which is not a thing the analyst ever meant to say.
    """

    run_requested = Signal(object)
    reject_requested = Signal()
    revise_requested = Signal()
    details_requested = Signal()

    TITLE = "开始之前，跟你确认一下"
    DATA_SECTION = "要用的数据"
    ACTION_SECTION = "要做的事"
    FOOTNOTE = "点「可以开始」之后，这批数据会先固定下来。后面数据库再更新，也不会影响这一轮的结论。"

    def __init__(
        self,
        plan: EDAPlan | EDAResearchScope,
        *,
        summary: DataSummary | dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.setObjectName("planMessage")
        self.setMinimumWidth(560)
        self.setMaximumWidth(720)
        self.plan = plan
        self.summary = as_summary(summary)
        self.step_checks: dict[str, QWidget] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel(self.TITLE)
        title.setObjectName("planTitle")
        header.addWidget(title)
        header.addStretch(1)
        self.status_label = QLabel("待确认")
        self.status_label.setObjectName("planStatus")
        header.addWidget(self.status_label)
        layout.addLayout(header)

        data_heading = QLabel(self.DATA_SECTION)
        data_heading.setObjectName("planStage")
        layout.addWidget(data_heading)
        for key, value, sub in summary_fields(self.summary):
            row = QLabel(f"{key}　{value}" if value else f"{key}　待定")
            row.setObjectName("planStep")
            row.setTextFormat(Qt.TextFormat.RichText)
            row.setWordWrap(True)
            layout.addWidget(row)
            if sub:
                sub_row = QLabel(sub)
                sub_row.setObjectName("planHint")
                sub_row.setWordWrap(True)
                layout.addWidget(sub_row)

        action_heading = QLabel(self.ACTION_SECTION)
        action_heading.setObjectName("planStage")
        layout.addWidget(action_heading)
        if isinstance(plan, EDAResearchScope):
            objective = QLabel(f"研究目标　{escape(plan.objective)}")
            objective.setObjectName("planStep")
            objective.setWordWrap(True)
            layout.addWidget(objective)
            for index, strategy in enumerate(plan.initial_strategy, start=1):
                row = QLabel(f"{index}. {escape(strategy)}")
                row.setObjectName("planStep")
                row.setWordWrap(True)
                self.step_checks[f"strategy-{index}"] = row
                layout.addWidget(row)
            categories = sorted(
                {
                    {
                        "price": "电价自身规律",
                        "exogenous": "影响因素质量",
                        "relationship": "电价与因素关系",
                    }.get(FUNCTION_CATALOG[name].category, "数据核验")
                    for name in plan.authorized_functions
                }
            )
            scope_row = QLabel(
                f"可用方法　{'、'.join(categories) or '仅系统数据质量核验'}；"
                f"最多 {plan.max_model_rounds} 轮决策、"
                f"{plan.max_tool_calls} 次基础调用；"
                f"每次最多尝试 {plan.max_attempts_per_call} 次"
            )
            scope_row.setObjectName("planHint")
            scope_row.setWordWrap(True)
            layout.addWidget(scope_row)
            variables = "、".join(plan.authorized_variables[:8]) or "不使用外生变量"
            if len(plan.authorized_variables) > 8:
                variables += f" 等 {len(plan.authorized_variables)} 项"
            variable_row = QLabel(f"变量范围　{escape(variables)}")
            variable_row.setObjectName("planHint")
            variable_row.setWordWrap(True)
            layout.addWidget(variable_row)
        else:
            for step in plan.enabled_steps:
                row = QLabel(plan.step_text(step))
                row.setObjectName("planStep")
                row.setWordWrap(True)
                row.setToolTip(step.description)
                self.step_checks[step.step_id] = row
                layout.addWidget(row)

        self.feedback_hint = QLabel(self.FOOTNOTE)
        self.feedback_hint.setObjectName("planHint")
        self.feedback_hint.setWordWrap(True)
        layout.addWidget(self.feedback_hint)

        actions = QHBoxLayout()
        self.details_button = QPushButton("查看取数细节")
        self.details_button.setObjectName("dataQuietLink")
        self.details_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.details_button.clicked.connect(self.details_requested)
        actions.addWidget(self.details_button)
        actions.addStretch(1)
        self.reject_button = QPushButton("先不做")
        self.reject_button.clicked.connect(self.reject_requested)
        actions.addWidget(self.reject_button)
        self.revise_button = QPushButton("我想改改")
        self.revise_button.clicked.connect(self.revise_requested)
        actions.addWidget(self.revise_button)
        self.run_button = QPushButton("可以开始")
        self.run_button.setObjectName("primaryButton")
        self.run_button.clicked.connect(lambda: self.run_requested.emit(self.plan))
        actions.addWidget(self.run_button)
        layout.addLayout(actions)

    def approved_plan(self) -> EDAPlan | EDAResearchScope:
        return self.plan

    def set_feedback_countdown(self, seconds: int) -> None:
        minutes, remainder = divmod(max(0, int(seconds)), 60)
        self.status_label.setText(f"{minutes} 分 {remainder} 秒后自动开始")
        self._show_actions(True)

    def set_explicit_approval(self) -> None:
        self.status_label.setText("等待你确认")
        self._show_actions(True)

    def set_feedback_paused(self, text: str = "正在接收修改意见") -> None:
        self.status_label.setText(text)

    def set_running(self) -> None:
        self.status_label.setText("执行中")
        self._show_actions(False)

    def set_finished(self, status: str = "已完成") -> None:
        self.status_label.setText(status)
        self._show_actions(False)

    def _show_actions(self, visible: bool) -> None:
        for button in (self.run_button, self.revise_button, self.reject_button):
            button.setVisible(visible)


class ForecastPlanMessageWidget(QFrame):
    """Explicit approval card for the fixed, read-only Shandong P3 forecast."""

    run_requested = Signal(object)
    reject_requested = Signal()

    def __init__(self, plan: ForecastPlan) -> None:
        super().__init__()
        self.setObjectName("planMessage")
        self.setMinimumWidth(560)
        self.setMaximumWidth(720)
        self.plan = plan
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel("开始预测之前，跟你确认一下")
        title.setObjectName("planTitle")
        header.addWidget(title)
        header.addStretch(1)
        self.status_label = QLabel("等待你确认")
        self.status_label.setObjectName("planStatus")
        header.addWidget(self.status_label)
        layout.addLayout(header)

        rows = [
            ("目标", "山东省级实时电价"),
            ("预测范围", f"{plan.forecast_start:%Y-%m-%d 00:00} — 23:45（96点）"),
            ("算法", f"CTM-Base + 多因素相似日（{plan.algorithm_version}）"),
            (
                "历史回测",
                "、".join(f"{item.target_start:%Y-%m-%d}" for item in plan.snapshots if item.role == "backtest"),
            ),
            ("运行方式", "CPU确定性训练；3折完成后预测次日"),
        ]
        for key, value in rows:
            row = QLabel(f"{key}　{value}")
            row.setObjectName("planStep")
            row.setWordWrap(True)
            layout.addWidget(row)
        hint = QLabel("只读取已冻结的4份输入数据，不写业务数据库；此方案不会自动开始。")
        hint.setObjectName("planHint")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        actions = QHBoxLayout()
        actions.addStretch(1)
        self.reject_button = QPushButton("先不预测")
        self.reject_button.clicked.connect(self.reject_requested)
        actions.addWidget(self.reject_button)
        self.run_button = QPushButton("确认并开始")
        self.run_button.setObjectName("primaryButton")
        self.run_button.clicked.connect(lambda: self.run_requested.emit(self.plan))
        actions.addWidget(self.run_button)
        layout.addLayout(actions)

    def set_explicit_approval(self) -> None:
        self.status_label.setText("等待你确认")

    def set_feedback_paused(self, text: str = "等待你确认") -> None:
        self.status_label.setText(text)

    def set_running(self) -> None:
        self.status_label.setText("执行中")
        self.run_button.hide()
        self.reject_button.hide()

    def set_finished(self, status: str = "已完成") -> None:
        self.status_label.setText(status)
        self.run_button.hide()
        self.reject_button.hide()


class ForecastResultMessageWidget(QFrame):
    """Compact metrics and artifact links for one completed forecast run."""

    def __init__(self, payload: dict[str, Any]) -> None:
        super().__init__()
        self.setObjectName("resultMessage")
        self.setMinimumWidth(560)
        self.setMaximumWidth(720)
        aggregate = payload.get("aggregate") or {}
        model = aggregate.get("model") or {}
        persistence = aggregate.get("persistence") or {}
        day = aggregate.get("day_naive") or {}
        week = aggregate.get("week_naive") or {}
        warnings = [str(item) for item in payload.get("warnings", [])]
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)
        title = QLabel("山东次日实时电价预测完成")
        title.setObjectName("resultTitle")
        layout.addWidget(title)
        metrics = QLabel(
            "三折 MAE："
            f"模型 {float(model.get('mae', 0)):.2f}；"
            f"持续法 {float(persistence.get('mae', 0)):.2f}；"
            f"前一日同刻 {float(day.get('mae', 0)):.2f}；"
            f"周前朴素 {float(week.get('mae', 0)):.2f}"
        )
        metrics.setObjectName("resultSummary")
        metrics.setWordWrap(True)
        layout.addWidget(metrics)
        diagnostics = payload.get("diagnostics") or {}
        if diagnostics.get("leakage_gate") == "passed":
            diagnostic_label = QLabel("✓ 防泄漏门禁通过；业务数据库保持只读")
            diagnostic_label.setObjectName("resultMeta")
            layout.addWidget(diagnostic_label)
        for warning in warnings:
            label = QLabel(f"! {warning}")
            label.setObjectName("resultWarning")
            label.setWordWrap(True)
            layout.addWidget(label)
        if not warnings:
            label = QLabel("✓ 三折聚合结果优于日/周朴素基线")
            label.setObjectName("resultMeta")
            layout.addWidget(label)
        actions = QHBoxLayout()
        figure_paths = payload.get("figure_paths") or {}
        for title_text, path, primary in (
            ("查看完整报告", str(payload.get("report_path", "")), True),
            ("查看预测曲线", str(figure_paths.get("forecast", "")), False),
            ("打开预测 CSV", str(payload.get("prediction_path", "")), False),
            ("打开结果文件夹", str(payload.get("artifact_directory", "")), False),
        ):
            button = QPushButton(title_text)
            if primary:
                button.setObjectName("primaryButton")
            button.setEnabled(bool(path))
            if title_text == "查看完整报告":
                button.clicked.connect(lambda _checked=False, value=path: open_report(value, self))
            else:
                button.clicked.connect(
                    lambda _checked=False, value=path: QDesktopServices.openUrl(QUrl.fromLocalFile(value))
                )
            actions.addWidget(button)
        actions.addStretch(1)
        layout.addLayout(actions)


class DataDetailsDialog(QDialog):
    """Where the data actually came from, in database words.

    This is the one screen the terminology rules do not cover: it exists so an
    operator can chase down an odd result, and it takes two clicks to reach.
    """

    TITLE = "取数细节"
    EMPTY_TEXT = "这一轮还没有记录取数细节。"

    def __init__(self, details: dict[str, Any] | None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.TITLE)
        self.setMinimumWidth(560)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)
        body = QPlainTextEdit()
        body.setObjectName("dataDetails")
        body.setReadOnly(True)
        body.setPlainText(self._render(details))
        layout.addWidget(body, 1)
        actions = QHBoxLayout()
        actions.addStretch(1)
        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.accept)
        actions.addWidget(close_button)
        layout.addLayout(actions)

    def _render(self, details: dict[str, Any] | None) -> str:
        if not details:
            return self.EMPTY_TEXT
        lines: list[str] = []
        for key, value in details.items():
            if isinstance(value, (list, tuple)):
                value = "\n  ".join(str(item) for item in value)
                lines.append(f"{key}:\n  {value}")
            elif isinstance(value, dict):
                rendered = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
                lines.append(f"{key}:\n{rendered}")
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines)


class ResultMessageWidget(QFrame):
    """One screen of conclusions, plus the way into the full report.

    Limitations, hypothesis verdicts and validity checks are deliberately absent:
    they are written in full to ``report.md`` and ``methods.md``, and repeating
    them here turns the conversation into a second, worse report.
    """

    MAXIMUM_FINDINGS = 3

    def __init__(self, result: AgentRunResult | None = None, *, payload: dict | None = None) -> None:
        super().__init__()
        if result is None and payload is None:
            raise ValueError("result or payload is required")
        if result is not None:
            evaluation = result.evaluation.model_dump(mode="json")
            figure_count = len(result.figure_paths)
            report_path = str(result.report_path)
            artifact_directory = str(result.artifact_directory)
        else:
            data = payload or {}
            evaluation = dict(data.get("evaluation", {}))
            figure_count = int(data.get("figure_count", 0))
            report_path = str(data.get("report_path", ""))
            artifact_directory = str(data.get("artifact_directory", ""))
        evaluation_summary = str(evaluation.get("summary", "研究已完成"))
        findings = [str(item) for item in evaluation.get("findings", [])]
        warnings = [str(item) for item in evaluation.get("warnings", [])]
        followups = [str(item) for item in evaluation.get("suggested_followups", [])]
        variable_recommendations = evaluation.get("variable_recommendations") or {}
        decision = str(evaluation.get("decision", "accept"))

        self.setObjectName("resultMessage")
        self.setMinimumWidth(560)
        self.setMaximumWidth(720)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)
        title = QLabel("变量筛查完成" if variable_recommendations else "分析完成")
        title.setObjectName("resultTitle")
        layout.addWidget(title)
        summary = QLabel(evaluation_summary)
        summary.setObjectName("resultSummary")
        summary.setWordWrap(True)
        layout.addWidget(summary)
        for finding in findings[: self.MAXIMUM_FINDINGS]:
            label = QLabel(f"· {finding}")
            label.setWordWrap(True)
            layout.addWidget(label)
        recommended_variables = [
            str(item) for item in variable_recommendations.get("recommended_variables", [])
        ]
        if variable_recommendations:
            recommendation_label = QLabel(
                "推荐深入分析：" + ("、".join(recommended_variables) if recommended_variables else "暂无优先变量")
            )
            recommendation_label.setObjectName("resultWarning")
            recommendation_label.setWordWrap(True)
            layout.addWidget(recommendation_label)
        pending = self._pending_items(evaluation) if decision in {"need_user", "reject"} else []
        if pending:
            decision_label = QLabel("需要你决定：" + "；".join(pending[:3]))
            decision_label.setObjectName("resultWarning")
            decision_label.setWordWrap(True)
            layout.addWidget(decision_label)
        meta = QLabel(self._meta_text(figure_count, len(warnings), len(followups)))
        meta.setObjectName("resultMeta")
        meta.setWordWrap(True)
        layout.addWidget(meta)
        actions = QHBoxLayout()
        report_button = QPushButton("查看完整报告")
        report_button.setObjectName("primaryButton")
        report_button.setEnabled(bool(report_path))
        report_button.clicked.connect(lambda: open_report(report_path, self))
        package_button = QPushButton("打开结果文件夹")
        package_button.setEnabled(bool(artifact_directory))
        package_button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(artifact_directory)))
        actions.addWidget(report_button)
        actions.addWidget(package_button)
        actions.addStretch(1)
        layout.addLayout(actions)

    @staticmethod
    def _pending_items(evaluation: dict[str, Any]) -> list[str]:
        """Name the open items a decision depends on, without restating their evidence."""

        names = [
            str(check.get("name", ""))
            for check in evaluation.get("checks", [])
            if check.get("status") != "pass" and check.get("scope") not in {"inherent", "within_envelope"}
        ]
        names.extend(
            str(item.get("hypothesis", ""))
            for item in evaluation.get("hypothesis_assessments", [])
            if item.get("status") in {"not_tested", "inconclusive"}
            and item.get("scope") not in {"inherent", "within_envelope"}
        )
        return [name for name in dict.fromkeys(names) if name]

    @staticmethod
    def _meta_text(figure_count: int, warning_count: int, followup_count: int) -> str:
        parts = [f"报告含 {figure_count} 张图表"]
        if warning_count:
            parts.append(f"{warning_count} 条适用边界")
        if followup_count:
            parts.append(f"{followup_count} 条下一步建议")
        return "，".join(parts) + "，可随时打开查看或分享。"

    @classmethod
    def from_payload(cls, payload: dict) -> ResultMessageWidget:
        return cls(payload=payload)
