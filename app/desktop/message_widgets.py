"""Typed widgets rendered inside one continuous research conversation."""

from __future__ import annotations

import json
from html import escape
from typing import Any

from PySide6.QtCore import Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QPainter, QPen
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
from app.research.graph.process_events import ProcessEvent
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


class ThinkingSpinner(QWidget):
    """Small animated activity mark used instead of a thinking card."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("thinkingSpinner")
        self.setFixedSize(16, 16)
        self.setAccessibleName("正在思考")
        self._angle = 0
        self._timer = QTimer(self)
        self._timer.setInterval(80)
        self._timer.timeout.connect(self._advance)
        self._timer.start()

    def _advance(self) -> None:
        self._angle = (self._angle - 30) % 360
        self.update()

    def set_running(self, running: bool) -> None:
        if running:
            self._timer.start()
            self.show()
        else:
            self._timer.stop()
            self.hide()

    def paintEvent(self, _event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor("#3F51B5"), 2.0)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawArc(self.rect().adjusted(2, 2, -2, -2), self._angle * 16, 250 * 16)


class ProcessActionRow(QWidget):
    """One action and its validated outcome inside a decision step."""

    def __init__(self, event: ProcessEvent) -> None:
        super().__init__()
        self.action_id = str(event.action_id)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 3, 0, 5)
        layout.setSpacing(2)
        headline = QHBoxLayout()
        headline.setSpacing(6)
        self.mark = QLabel("◌")
        self.mark.setObjectName("processActionMark")
        headline.addWidget(self.mark)
        self.title = QLabel(event.title or event.tool_name or "执行行动")
        self.title.setObjectName("processActionTitle")
        headline.addWidget(self.title, 1)
        self.status = QLabel("执行中")
        self.status.setObjectName("processActionStatus")
        headline.addWidget(self.status)
        layout.addLayout(headline)
        arguments = json.dumps(event.arguments, ensure_ascii=False, separators=(", ", ": "))
        self.arguments = QLabel(f"参数：{arguments}" if event.arguments else "")
        self.arguments.setObjectName("processActionDetail")
        self.arguments.setWordWrap(True)
        self.arguments.setVisible(bool(event.arguments))
        layout.addWidget(self.arguments)
        self.result_heading = QLabel("结果")
        self.result_heading.setObjectName("processSectionTitle")
        self.result_heading.hide()
        layout.addWidget(self.result_heading)
        self.result = QLabel("")
        self.result.setObjectName("processResult")
        self.result.setWordWrap(True)
        self.result.hide()
        layout.addWidget(self.result)
        self.open_result = QPushButton("打开结果")
        self.open_result.setObjectName("quietButton")
        self.open_result.hide()
        layout.addWidget(self.open_result, 0, Qt.AlignmentFlag.AlignLeft)

    def finish(self, event: ProcessEvent) -> None:
        failed = event.event_type == "action_failed"
        self.mark.setText("✕" if failed else "✓")
        self.status.setText("失败" if failed else "已完成")
        state = "failed" if failed else "completed"
        for widget in (self.mark, self.status):
            widget.setProperty("traceStatus", state)
            widget.style().unpolish(widget)
            widget.style().polish(widget)
        self.result.setText(event.result or ("执行失败" if failed else "已完成"))
        self.result_heading.show()
        self.result.show()
        if event.report_path:
            try:
                self.open_result.clicked.disconnect()
            except RuntimeError:
                pass
            self.open_result.clicked.connect(
                lambda _checked=False, path=event.report_path: open_report(path)
            )
            self.open_result.show()

    def interrupt(self, state: str, detail: str) -> None:
        if self.result.isVisible():
            return
        self.mark.setText("■" if state == "stopped" else "✕")
        self.status.setText("已终止" if state == "stopped" else "失败")
        self.result.setText(detail)
        self.result_heading.show()
        self.result.show()


class ProcessStepWidget(QFrame):
    """One model decision containing its thought, actions, and results."""

    def __init__(self, event: ProcessEvent, *, display_number: int) -> None:
        super().__init__()
        self.setObjectName("processStep")
        self.step_id = event.step_id
        self._collapsed = False
        self._actions: dict[str, ProcessActionRow] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 5, 0, 6)
        layout.setSpacing(5)
        header = QHBoxLayout()
        header.setSpacing(7)
        self.toggle = QPushButton("收起")
        self.toggle.setObjectName("processStepToggle")
        self.toggle.setFixedWidth(34)
        self.toggle.clicked.connect(self._toggle)
        self.spinner = ThinkingSpinner()
        header.addWidget(self.spinner)
        self.title = QLabel(f"步骤 {display_number} · 正在思考…")
        self.title.setObjectName("processStepTitle")
        header.addWidget(self.title, 1)
        self.status = QLabel("")
        self.status.setObjectName("processStepStatus")
        header.addWidget(self.status)
        header.addWidget(self.toggle)
        layout.addLayout(header)

        self.body = QWidget()
        body = QVBoxLayout(self.body)
        body.setContentsMargins(43, 0, 0, 0)
        body.setSpacing(3)
        self.thought_heading = QLabel("思考过程")
        self.thought_heading.setObjectName("processSectionTitle")
        body.addWidget(self.thought_heading)
        self.thought = QLabel(event.content or "正在根据已有证据选择下一步行动…")
        self.thought.setObjectName("processThought")
        self.thought.setWordWrap(True)
        body.addWidget(self.thought)
        self.action_heading = QLabel("行动")
        self.action_heading.setObjectName("processSectionTitle")
        self.action_heading.hide()
        body.addWidget(self.action_heading)
        self.actions_widget = QWidget()
        self.actions_layout = QVBoxLayout(self.actions_widget)
        self.actions_layout.setContentsMargins(0, 0, 0, 0)
        self.actions_layout.setSpacing(1)
        self.actions_widget.hide()
        body.addWidget(self.actions_widget)
        layout.addWidget(self.body)

    @property
    def actions(self) -> dict[str, ProcessActionRow]:
        return dict(self._actions)

    def apply_event(self, event: ProcessEvent) -> None:
        if event.event_type == "thinking_started":
            self.spinner.set_running(True)
            self.title.setText(f"步骤 {event.round or 1} · 正在思考…")
            self.thought_heading.setText("思考过程")
            self.thought.setText(event.content or "正在思考…")
            self.status.setText("")
        elif event.event_type == "thinking_ready":
            self.spinner.set_running(False)
            self.title.setText(f"步骤 {event.round or 1}")
            self.thought_heading.setText("执行依据" if event.source == "system" else "思考过程")
            self.thought.setText(event.content or "本轮未提供思考过程说明")
        elif event.event_type == "action_started":
            self.spinner.set_running(True)
            self.title.setText(f"步骤 {event.round or 1} · 正在行动…")
            action_id = str(event.action_id or event.event_id)
            if action_id not in self._actions:
                row = ProcessActionRow(event)
                self._actions[action_id] = row
                self.actions_layout.addWidget(row)
            self.action_heading.show()
            self.actions_widget.show()
            self.status.setText(f"{len(self._actions)} 个行动")
        elif event.event_type in {"action_completed", "action_failed"}:
            action_id = str(event.action_id or event.event_id)
            row = self._actions.get(action_id)
            if row is None:
                row = ProcessActionRow(event)
                self._actions[action_id] = row
                self.actions_layout.addWidget(row)
            row.finish(event)
            self.action_heading.show()
            self.actions_widget.show()
            completed = sum(action.result.isVisible() for action in self._actions.values())
            self.status.setText(f"{completed}/{len(self._actions)} 已完成")
        elif event.event_type == "step_completed":
            self.spinner.set_running(False)
            failed = any(action.status.text() == "失败" for action in self._actions.values())
            if not self._actions and self.thought.text().startswith("正在"):
                self.thought_heading.setText("处理结果")
                self.thought.setText(event.result or "本轮没有生成可执行行动")
            self.title.setText(f"步骤 {event.round or 1} · {'存在失败' if failed else '已完成'}")
            self.status.setText(event.result or f"{len(self._actions)} 个行动")
            self.status.setProperty("traceStatus", "failed" if failed else "completed")
            self.status.style().unpolish(self.status)
            self.status.style().polish(self.status)

    def set_collapsed(self, collapsed: bool) -> None:
        self._collapsed = collapsed
        self.body.setVisible(not collapsed)
        self.toggle.setText("展开" if collapsed else "收起")

    def interrupt(self, state: str, detail: str) -> None:
        self.spinner.set_running(False)
        for action in self._actions.values():
            action.interrupt(state, detail)
        self.title.setText("步骤中断" if state == "stopped" else "步骤失败")
        self.status.setText(detail)

    def _toggle(self) -> None:
        self.set_collapsed(not self._collapsed)


class ThinkingMessageWidget(QFrame):
    """Lightweight live view of observable Agent stages and checks."""

    def __init__(
        self,
        *,
        steps: list[dict[str, Any]] | None = None,
        process_events: list[dict[str, Any]] | None = None,
        state: str = "running",
        mode: str = "research",
    ) -> None:
        super().__init__()
        self.setObjectName("thinkingMessage")
        self.setMaximumWidth(680)
        self._steps: list[dict[str, Any]] = []
        self._rows: list[ThinkingStepRow] = []
        self._process_events: list[dict[str, Any]] = []
        self._process_event_ids: set[str] = set()
        self._process_rows: dict[str, ProcessStepWidget] = {}
        self._collapsed = False
        self._mode = mode
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(5)

        header = QHBoxLayout()
        header.setSpacing(7)
        self.spinner = ThinkingSpinner()
        header.addWidget(self.spinner)
        self.title_label = QLabel("正在思考…" if mode == "dialogue" else "正在研究…")
        self.title_label.setObjectName("thinkingTitle")
        header.addWidget(self.title_label)
        self.status_label = QLabel("")
        self.status_label.setObjectName("thinkingStatus")
        header.addWidget(self.status_label)
        header.addStretch(1)
        self.toggle_button = QPushButton("收起")
        self.toggle_button.setObjectName("thinkingToggle")
        self.toggle_button.setFixedWidth(34)
        self.toggle_button.clicked.connect(self._toggle)
        header.addWidget(self.toggle_button)
        layout.addLayout(header)

        self.steps_container = QWidget()
        self.steps_layout = QVBoxLayout(self.steps_container)
        self.steps_layout.setContentsMargins(23, 0, 0, 0)
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
        for payload in process_events or []:
            self.apply_process_event(ProcessEvent.model_validate(payload))
        if state != "running":
            self.finish(state=state)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ThinkingMessageWidget:
        return cls(
            steps=list(payload.get("steps", [])),
            process_events=list(payload.get("process_events", [])),
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

    @property
    def process_events(self) -> list[dict[str, Any]]:
        return list(self._process_events)

    @property
    def process_rows(self) -> dict[str, ProcessStepWidget]:
        return dict(self._process_rows)

    def apply_process_event(self, event: ProcessEvent) -> bool:
        """Apply an idempotent event and keep only the current step expanded."""

        if event.event_id in self._process_event_ids:
            return False
        self._process_event_ids.add(event.event_id)
        self._process_events.append(event.model_dump(mode="json"))
        row = self._process_rows.get(event.step_id)
        if row is None:
            for existing in self._process_rows.values():
                existing.set_collapsed(True)
            row = ProcessStepWidget(event, display_number=len(self._process_rows) + 1)
            self._process_rows[event.step_id] = row
            self.steps_layout.addWidget(row)
        row.set_collapsed(False)
        row.apply_event(event)
        self.spinner.set_running(False)
        self.toggle_button.hide()
        self.title_label.setText("研究过程")
        self.status_label.setText(f"{len(self._process_rows)} 步")
        return True

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
        self.spinner.set_running(False)
        if self._process_rows:
            rows = list(self._process_rows.values())
            if state in {"failed", "stopped"}:
                rows[-1].interrupt(state, summary or titles.get(state, completed_title))
            for row in rows[:-1]:
                row.set_collapsed(True)
            rows[-1].set_collapsed(False)
        self.set_collapsed(False)

    def set_collapsed(self, collapsed: bool) -> None:
        self._collapsed = collapsed
        self.steps_container.setVisible(not collapsed)
        self.toggle_button.setText("展开" if collapsed else "收起")

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
