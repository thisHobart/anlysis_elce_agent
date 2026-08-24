"""Typed widgets rendered inside one continuous research conversation."""

from __future__ import annotations

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.research.agent.schemas import AgentRunResult, EDAPlan
from app.research.tools.catalog import TOOL_CATALOG


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
        role_label = QLabel("你" if role == "user" else "研究 Agent")
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
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 7, 10, 7)
        label = QLabel(content)
        label.setWordWrap(True)
        layout.addWidget(label)


class ToolMessageWidget(QFrame):
    """Observable tool status card mirrored in the Agent trace."""

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
        text_layout.addWidget(self.title_label)
        text_layout.addWidget(self.detail_label)
        layout.addLayout(text_layout, 1)
        self.status_label = QLabel()
        layout.addWidget(self.status_label)
        self.set_status(status)

    def set_status(self, status: str, detail: str | None = None) -> None:
        labels = {
            "running": "运行中",
            "completed": "完成",
            "warning": "警告",
            "failed": "失败",
            "stopped": "已停止",
        }
        self.status_label.setText(labels.get(status, status))
        self.status_label.setProperty("traceStatus", status)
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)
        if detail is not None:
            self.detail_label.setText(detail)


class PlanMessageWidget(QFrame):
    """Read-only model plan; revisions are requested through the conversation."""

    run_requested = Signal(object)

    def __init__(self, plan: EDAPlan, available_variables: list[str]) -> None:
        super().__init__()
        self.setObjectName("planMessage")
        self.setMaximumWidth(720)
        self.plan = plan
        self.step_checks: dict[str, QCheckBox] = {}
        self.method_items: dict[tuple[str, str], QListWidgetItem] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel(f"Agent 推荐的 EDA 方案 · v{plan.revision}")
        title.setObjectName("planTitle")
        header.addWidget(title)
        header.addStretch(1)
        self.status_label = QLabel("等待反馈")
        self.status_label.setObjectName("planStatus")
        header.addWidget(self.status_label)
        layout.addLayout(header)

        skill_label = QLabel(f"Skill：{plan.skill_name}@{plan.skill_version}")
        skill_label.setObjectName("planObjective")
        layout.addWidget(skill_label)

        objective = QLabel(plan.objective)
        objective.setWordWrap(True)
        objective.setObjectName("planObjective")
        layout.addWidget(objective)
        if plan.hypotheses:
            hypotheses = QLabel("待验证假设\n" + "\n".join(f"• {item}" for item in plan.hypotheses[:5]))
            hypotheses.setWordWrap(True)
            hypotheses.setObjectName("planObjective")
            layout.addWidget(hypotheses)
        feedback_hint = QLabel("如需修改，请在下方对话框中提出建议；没有反馈时方案将在倒计时结束后自动执行。")
        feedback_hint.setWordWrap(True)
        feedback_hint.setObjectName("planObjective")
        layout.addWidget(feedback_hint)

        for step in plan.steps:
            checkbox = QCheckBox(step.title)
            checkbox.setChecked(step.enabled)
            checkbox.setEnabled(False)
            checkbox.setToolTip(f"{step.description}\nAgent 理由：{step.rationale}")
            self.step_checks[step.step_id] = checkbox
            layout.addWidget(checkbox)

        self.editor_toggle = QToolButton()
        self.editor_toggle.setText("变量与参数")
        self.editor_toggle.setCheckable(True)
        self.editor_toggle.setChecked(True)
        self.editor_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        layout.addWidget(self.editor_toggle)

        self.editor = QWidget()
        editor_layout = QHBoxLayout(self.editor)
        editor_layout.setContentsMargins(0, 0, 0, 0)
        self.variable_list = QListWidget()
        self.variable_list.setMaximumHeight(118)
        selected = set(plan.selected_variables)
        for name in available_variables:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if name in selected else Qt.CheckState.Unchecked)
            self.variable_list.addItem(item)
        editor_layout.addWidget(self.variable_list, 1)

        method_layout = QVBoxLayout()
        method_layout.setSpacing(3)
        method_layout.addWidget(QLabel("分析方法"))
        self.method_list = QListWidget()
        self.method_list.setMaximumHeight(118)
        for step in plan.steps:
            catalog = TOOL_CATALOG[step.tool]
            configured_methods = step.parameters.get("methods")
            selected_methods = (
                set(configured_methods) if configured_methods is not None else {method.key for method in catalog.methods}
            )
            for method in catalog.methods:
                item = QListWidgetItem(f"{step.title} · {method.label}")
                item.setToolTip(f"{method.description}\n实现：{method.implementation_id} · v{method.version}")
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(
                    Qt.CheckState.Checked if method.key in selected_methods else Qt.CheckState.Unchecked
                )
                self.method_list.addItem(item)
                self.method_items[(step.tool, method.key)] = item
        method_layout.addWidget(self.method_list)
        editor_layout.addLayout(method_layout, 1)

        parameter_layout = QVBoxLayout()
        parameter_layout.addWidget(QLabel("最大滞后"))
        self.max_lag = QSpinBox()
        lag_limit = next(
            (int(step.parameters["max_lag_limit"]) for step in plan.steps if "max_lag_limit" in step.parameters),
            24 * 31,
        )
        self.max_lag.setRange(0, lag_limit)
        self.max_lag.setSuffix(" 个间隔")
        self.max_lag.setValue(
            next((int(step.parameters["max_lag"]) for step in plan.steps if "max_lag" in step.parameters), 0)
        )
        self.variable_list.setEnabled(False)
        self.method_list.setEnabled(False)
        self.max_lag.setEnabled(False)
        parameter_layout.addWidget(self.max_lag)
        parameter_layout.addStretch(1)
        editor_layout.addLayout(parameter_layout)
        layout.addWidget(self.editor)
        self.editor_toggle.toggled.connect(self.editor.setVisible)

        actions = QHBoxLayout()
        actions.addStretch(1)
        self.run_button = QPushButton("立即执行")
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
        self.run_button.show()

    def set_feedback_paused(self, text: str = "正在输入修改意见") -> None:
        self.status_label.setText(text)

    def set_running(self) -> None:
        self.status_label.setText("运行中")
        self.run_button.hide()
        self.editor_toggle.setChecked(False)
        self.editor_toggle.setEnabled(False)
        self.editor.setEnabled(False)
        for checkbox in self.step_checks.values():
            checkbox.setEnabled(False)

    def set_finished(self, status: str = "已完成") -> None:
        self.status_label.setText(status)
        self.run_button.hide()
        self.editor_toggle.setChecked(False)
        self.editor_toggle.setEnabled(False)
        self.editor.setEnabled(False)


class ResultMessageWidget(QFrame):
    """Final evaluation, findings, and reproducibility actions."""

    def __init__(self, result: AgentRunResult | None = None, *, payload: dict | None = None) -> None:
        super().__init__()
        if result is None and payload is None:
            raise ValueError("result or payload is required")
        if result is not None:
            evaluation_summary = result.evaluation.summary
            findings = result.evaluation.findings
            warnings = result.evaluation.warnings
            hypothesis_assessments = result.evaluation.hypothesis_assessments
            run_id = result.run_id
            figure_count = len(result.figure_paths)
            report_path = str(result.report_path)
            artifact_directory = str(result.artifact_directory)
        else:
            data = payload or {}
            evaluation = data.get("evaluation", {})
            evaluation_summary = str(evaluation.get("summary", "研究已完成"))
            findings = list(evaluation.get("findings", []))
            warnings = list(evaluation.get("warnings", []))
            hypothesis_assessments = list(evaluation.get("hypothesis_assessments", []))
            run_id = str(data.get("run_id", ""))
            figure_count = int(data.get("figure_count", 0))
            report_path = str(data.get("report_path", ""))
            artifact_directory = str(data.get("artifact_directory", ""))
        self.setObjectName("resultMessage")
        self.setMaximumWidth(720)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)
        title = QLabel(evaluation_summary)
        title.setObjectName("resultTitle")
        title.setWordWrap(True)
        layout.addWidget(title)
        for finding in findings:
            label = QLabel(f"• {finding}")
            label.setWordWrap(True)
            layout.addWidget(label)
        if hypothesis_assessments:
            status_labels = {
                "candidate_support": "候选支持",
                "not_supported": "暂不支持",
                "inconclusive": "证据不足",
                "not_tested": "未检验",
            }
            hypothesis_title = QLabel("假设验收")
            hypothesis_title.setObjectName("resultMeta")
            layout.addWidget(hypothesis_title)
            for assessment in hypothesis_assessments[:4]:
                if hasattr(assessment, "model_dump"):
                    assessment = assessment.model_dump(mode="json")
                status = status_labels.get(str(assessment.get("status", "")), str(assessment.get("status", "")))
                label = QLabel(f"• [{status}] {assessment.get('hypothesis', '')} — {assessment.get('evidence', '')}")
                label.setWordWrap(True)
                layout.addWidget(label)
        if warnings:
            warning = QLabel("风险：" + "；".join(warnings[:3]))
            warning.setObjectName("resultWarning")
            warning.setWordWrap(True)
            layout.addWidget(warning)
        meta = QLabel(f"运行编号 {run_id} · {figure_count} 张图表")
        meta.setObjectName("resultMeta")
        layout.addWidget(meta)
        actions = QHBoxLayout()
        report_button = QPushButton("打开报告")
        report_button.setEnabled(bool(report_path))
        report_button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(report_path)))
        package_button = QPushButton("打开研究包")
        package_button.setEnabled(bool(artifact_directory))
        package_button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(artifact_directory)))
        actions.addWidget(report_button)
        actions.addWidget(package_button)
        actions.addStretch(1)
        layout.addLayout(actions)

    @classmethod
    def from_payload(cls, payload: dict) -> ResultMessageWidget:
        return cls(payload=payload)
