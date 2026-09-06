"""Three-pane desktop shell for continuous Agent-led research conversations."""

from __future__ import annotations

from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import QDialog, QLabel, QMainWindow, QMessageBox, QToolBar

from app.config import get_settings
from app.desktop.model_config import ModelConfigDialog
from app.desktop.session import SessionStore
from app.desktop.workspace import ResearchWorkspace
from app.research.application.coordinator import ResearchCoordinator


class MainWindow(QMainWindow):
    """Application window intentionally limited to the first-stage workflow."""

    def __init__(
        self,
        *,
        agent: ResearchCoordinator | None = None,
        session_store: SessionStore | None = None,
        plan_feedback_seconds: int = 30,
        auto_execute_plan: bool = False,
    ) -> None:
        super().__init__()
        self.setObjectName("mainWindow")
        self.setWindowTitle("电价与外生变量研究 Agent")
        self.resize(1400, 900)
        self.setMinimumSize(1100, 720)

        self.workspace = ResearchWorkspace(
            agent=agent,
            store=session_store,
            plan_feedback_seconds=plan_feedback_seconds,
            auto_execute_plan=auto_execute_plan,
        )
        toolbar = QToolBar("应用工具", self)
        toolbar.setObjectName("applicationToolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        model_action = QAction("大模型配置", self)
        model_action.setObjectName("modelConfigAction")
        model_action.triggered.connect(self.open_model_config)
        toolbar.addAction(model_action)
        toolbar.addSeparator()
        self.model_status = QLabel()
        self.model_status.setObjectName("modelStatusLabel")
        toolbar.addWidget(self.model_status)
        self._refresh_model_status()
        self.setCentralWidget(self.workspace)
        self.statusBar().showMessage("就绪")
        self.workspace.status_changed.connect(self.statusBar().showMessage)

    def open_model_config(self) -> None:
        if self.workspace.is_busy:
            QMessageBox.information(self, "研究正在运行", "请等待当前任务完成后再修改模型配置。")
            return
        self.workspace.pause_plan_feedback_window()
        try:
            dialog = ModelConfigDialog(self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.workspace.refresh_agent()
                self._refresh_model_status()
                self.statusBar().showMessage("大模型配置已保存，后续方案规划将使用新配置")
        finally:
            self.workspace.resume_plan_feedback_window()

    def _refresh_model_status(self) -> None:
        settings = get_settings()
        configured = bool(settings.llm_model) and (
            settings.llm_provider == "gemini" or bool(settings.llm_base_url)
        )
        if configured:
            provider = {
                "deepseek": "DeepSeek",
                "qwen": "Qwen",
                "gemini": "Gemini 原生",
                "custom": "自定义",
            }[settings.llm_provider]
            api_style = (
                "Gemini API"
                if settings.llm_provider == "gemini"
                else ("Responses" if settings.llm_api_style == "responses" else "Chat")
            )
            self.model_status.setText(f"模型：{provider} · {api_style} · {settings.llm_model}")
        else:
            self.model_status.setText("模型：未配置（研究规划不可用）")

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.workspace.is_busy:
            QMessageBox.information(self, "研究正在运行", "请等待当前规划或分析完成后再关闭应用。")
            event.ignore()
            return
        if self.workspace._owns_agent:
            self.workspace.agent.close()
        super().closeEvent(event)
