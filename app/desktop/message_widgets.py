"""Typed widgets rendered inside one continuous research conversation."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.desktop.report_view import open_report
from app.research.agent.schemas import AgentRunResult, EDAPlan
from app.research.tools.catalog import FUNCTION_CATALOG, STAGE_TITLES

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

    def __init__(self, *, steps: list[dict[str, Any]] | None = None, state: str = "running") -> None:
        super().__init__()
        self.setObjectName("thinkingMessage")
        self.setMaximumWidth(680)
        self.setMinimumWidth(520)
        self._steps: list[dict[str, Any]] = []
        self._rows: list[ThinkingStepRow] = []
        self._collapsed = False
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
        self.title_label = QLabel("正在研究…")
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
        return cls(steps=list(payload.get("steps", [])), state=str(payload.get("state", "running")))

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
        titles = {
            "completed": "研究过程",
            "failed": "研究中断",
            "stopped": "已终止",
        }
        self.title_label.setText(titles.get(state, "研究过程"))
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


class PlanMessageWidget(QFrame):
    """Read-only analysis plan; changes are requested in plain language."""

    run_requested = Signal(object)
    reject_requested = Signal()

    def __init__(self, plan: EDAPlan) -> None:
        super().__init__()
        self.setObjectName("planMessage")
        self.setMinimumWidth(560)
        self.setMaximumWidth(720)
        self.plan = plan
        self.step_checks: dict[str, QWidget] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel("建议的分析方案" if plan.revision <= 1 else f"修订后的分析方案（第 {plan.revision} 版）")
        title.setObjectName("planTitle")
        header.addWidget(title)
        header.addStretch(1)
        self.status_label = QLabel("待确认")
        self.status_label.setObjectName("planStatus")
        header.addWidget(self.status_label)
        layout.addLayout(header)

        objective = QLabel(plan.objective)
        objective.setWordWrap(True)
        objective.setObjectName("planObjective")
        layout.addWidget(objective)

        if plan.selected_variables:
            variables = QLabel("纳入分析的影响因素：" + "、".join(plan.selected_variables))
            variables.setWordWrap(True)
            variables.setObjectName("planObjective")
            layout.addWidget(variables)

        grouped: dict[str, list[Any]] = {}
        for step in plan.enabled_steps:
            grouped.setdefault(FUNCTION_CATALOG[step.function].stage, []).append(step)
        for stage, steps in grouped.items():
            group = QLabel(STAGE_TITLES.get(stage, stage))
            group.setObjectName("planStage")
            layout.addWidget(group)
            for step in steps:
                row = QLabel(f"· {step.title}｜{FUNCTION_CATALOG[step.function].answers}")
                row.setObjectName("planStep")
                row.setWordWrap(True)
                row.setToolTip(f"{step.description}\n选择理由：{step.rationale}")
                self.step_checks[step.step_id] = row
                layout.addWidget(row)

        if plan.hypotheses:
            hypotheses = QLabel("想验证的判断\n" + "\n".join(f"· {item}" for item in plan.hypotheses[:5]))
            hypotheses.setWordWrap(True)
            hypotheses.setObjectName("planObjective")
            layout.addWidget(hypotheses)

        self.feedback_hint = QLabel("想改或想先了解方案，可以直接在下面说；只有明确确认后才会开始执行。")
        self.feedback_hint.setWordWrap(True)
        self.feedback_hint.setObjectName("planHint")
        layout.addWidget(self.feedback_hint)

        actions = QHBoxLayout()
        actions.addStretch(1)
        self.reject_button = QPushButton("拒绝方案")
        self.reject_button.clicked.connect(self.reject_requested)
        actions.addWidget(self.reject_button)
        self.run_button = QPushButton("立即开始")
        self.run_button.setObjectName("primaryButton")
        self.run_button.clicked.connect(self._emit_plan)
        actions.addWidget(self.run_button)
        layout.addLayout(actions)

    def approved_plan(self) -> EDAPlan:
        return self.plan

    def _emit_plan(self) -> None:
        self.run_requested.emit(self.plan)

    def set_feedback_countdown(self, seconds: int) -> None:
        self.status_label.setText(f"{seconds} 秒后自动执行")
        self.feedback_hint.setText("想改就直接在下面说；输入期间倒计时会暂停。")
        self.run_button.show()
        self.reject_button.show()

    def set_explicit_approval(self) -> None:
        self.status_label.setText("等待明确确认")
        self.feedback_hint.setText("想改或想先了解方案，可以直接在下面说；只有明确确认后才会开始执行。")
        self.run_button.show()
        self.reject_button.show()

    def set_feedback_paused(self, text: str = "正在接收修改意见") -> None:
        self.status_label.setText(text)

    def set_running(self) -> None:
        self.status_label.setText("执行中")
        self.run_button.hide()
        self.reject_button.hide()

    def set_finished(self, status: str = "已完成") -> None:
        self.status_label.setText(status)
        self.run_button.hide()
        self.reject_button.hide()


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
