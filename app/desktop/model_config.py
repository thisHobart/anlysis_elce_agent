"""Desktop dialog for configuring the OpenAI-compatible research model."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlparse

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QSpinBox,
    QVBoxLayout,
)

from app.config import get_settings, runtime_env_file

MODEL_PRESETS: tuple[tuple[str, str, str], ...] = (
    ("DeepSeek", "https://api.deepseek.com/v1", "deepseek-chat"),
    ("自定义 OpenAI 兼容接口", "", ""),
)


def _env_value(value: object) -> str:
    """Serialize one UI value for the simple KEY=value .env format."""

    return str(value).replace("\r", "").replace("\n", "")


def update_env_values(path: str | Path, values: Mapping[str, object]) -> None:
    """Update selected environment keys while preserving other .env content."""

    env_path = Path(path)
    original = env_path.read_text(encoding="utf-8") if env_path.is_file() else ""
    lines = original.splitlines()
    missing: list[str] = []

    for key in values:
        pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")
        replacement = f"{key}={_env_value(values[key])}"
        for index, line in enumerate(lines):
            if not line.lstrip().startswith("#") and pattern.match(line):
                lines[index] = replacement
                break
        else:
            missing.append(replacement)

    if missing:
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend(missing)

    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _provider_index(base_url: str) -> int:
    for index, (_label, preset_url, _model) in enumerate(MODEL_PRESETS):
        if preset_url and base_url.rstrip("/") == preset_url.rstrip("/"):
            return index
    return len(MODEL_PRESETS) - 1


class ModelConfigDialog(QDialog):
    """Edit the model settings consumed by the research planner."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("modelConfigDialog")
        self.setWindowTitle("大模型配置")
        self.setMinimumWidth(560)

        settings = get_settings()
        self.provider = QComboBox()
        for label, base_url, model in MODEL_PRESETS:
            self.provider.addItem(label, (base_url, model))
        self.provider.setCurrentIndex(_provider_index(settings.llm_base_url))
        self.provider.currentIndexChanged.connect(self._apply_provider_preset)

        self.base_url = QLineEdit(settings.llm_base_url)
        self.base_url.setClearButtonEnabled(True)

        self.api_key = QLineEdit(self._display_api_key(settings.llm_api_key.get_secret_value()))
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key.setClearButtonEnabled(True)
        self.show_api_key = QCheckBox("显示")
        self.show_api_key.toggled.connect(self._toggle_api_key_visibility)
        api_key_row = QHBoxLayout()
        api_key_row.addWidget(self.api_key, 1)
        api_key_row.addWidget(self.show_api_key)

        self.model = QLineEdit(settings.llm_model)
        self.model.setClearButtonEnabled(True)

        self.timeout = QSpinBox()
        self.timeout.setRange(1, 600)
        self.timeout.setValue(round(settings.llm_timeout_seconds))
        self.timeout.setSuffix(" 秒")

        self.retries = QSpinBox()
        self.retries.setRange(0, 10)
        self.retries.setValue(settings.llm_max_retries)
        self.retries.setSuffix(" 次")

        self.structured_mode = QComboBox()
        self.structured_mode.addItem("Function Calling（推荐）", "native")
        self.structured_mode.addItem("JSON 提示词（兼容模式）", "json_prompt")
        mode_index = self.structured_mode.findData(settings.llm_structured_mode)
        self.structured_mode.setCurrentIndex(max(mode_index, 0))

        self.history_messages = QSpinBox()
        self.history_messages.setRange(1, 100)
        self.history_messages.setValue(settings.llm_history_messages)
        self.history_messages.setSuffix(" 条")

        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form.addRow("接口预设", self.provider)
        form.addRow("Base URL", self.base_url)
        form.addRow("API Key", api_key_row)
        form.addRow("模型名称", self.model)
        form.addRow("请求超时", self.timeout)
        form.addRow("失败重试", self.retries)
        form.addRow("结构化输出", self.structured_mode)
        form.addRow("历史消息", self.history_messages)

        help_label = QLabel(
            "配置保存到项目 .env。DeepSeek 使用官方 Base URL；自定义接口需要填写完整的 OpenAI 兼容 Base URL。"
            "研究规划和方案修订必须调用大模型；模型不可用时任务会暂停并提示重试。"
        )
        help_label.setWordWrap(True)
        help_label.setObjectName("modelConfigHelp")

        self.status_label = QLabel(f"配置文件：{runtime_env_file()}")
        self.status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.status_label.setObjectName("modelConfigPath")

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel,
        )
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(help_label)
        layout.addWidget(self.status_label)
        layout.addWidget(buttons)

    @staticmethod
    def _display_api_key(value: str) -> str:
        return "" if value.strip().casefold() in {"none", "null"} else value

    def _toggle_api_key_visibility(self, visible: bool) -> None:
        self.api_key.setEchoMode(
            QLineEdit.EchoMode.Normal if visible else QLineEdit.EchoMode.Password
        )

    def _apply_provider_preset(self, index: int) -> None:
        if index < 0 or index >= len(MODEL_PRESETS):
            return
        base_url, model = self.provider.itemData(index)
        if not base_url and not model:
            return
        self.base_url.setText(base_url)
        self.model.setText(model)

    def _validate(self) -> str | None:
        base_url = self.base_url.text().strip()
        model = self.model.text().strip()
        if not base_url:
            return "必须填写大模型 Base URL。"
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return "Base URL 必须是完整的 http:// 或 https:// 地址。"
        if not model:
            return "必须填写模型名称。"
        return None

    def _save(self) -> None:
        error = self._validate()
        if error:
            QMessageBox.warning(self, "配置不完整", error)
            return
        values = {
            "VPP_LLM_ENABLED": "true",
            "VPP_LLM_BASE_URL": self.base_url.text().strip(),
            "VPP_LLM_API_KEY": self.api_key.text(),
            "VPP_LLM_MODEL": self.model.text().strip(),
            "VPP_LLM_TIMEOUT_SECONDS": self.timeout.value(),
            "VPP_LLM_MAX_RETRIES": self.retries.value(),
            "VPP_LLM_STRUCTURED_MODE": self.structured_mode.currentData(),
            "VPP_LLM_HISTORY_MESSAGES": self.history_messages.value(),
        }
        try:
            update_env_values(runtime_env_file(), values)
            get_settings.cache_clear()
        except OSError as exc:
            QMessageBox.critical(self, "保存失败", f"无法保存模型配置：{exc}")
            return
        self.accept()
