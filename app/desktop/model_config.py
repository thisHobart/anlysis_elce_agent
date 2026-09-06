"""Desktop dialog for configuring the research model protocol."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlparse

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
    QVBoxLayout,
)

from app.config import get_settings, runtime_env_file

MODEL_PRESETS: tuple[tuple[str, str, str, str], ...] = (
    ("deepseek", "DeepSeek", "https://api.deepseek.com/v1", "deepseek-chat"),
    ("qwen", "Qwen", "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-plus"),
    ("gemini", "Gemini（原生 API）", "", "gemini-2.5-flash"),
    ("custom", "自定义接口", "", ""),
)

API_STYLE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("chat", "Chat Completions"),
    ("responses", "Responses"),
)

STRUCTURED_OUTPUT_OPTIONS: tuple[tuple[str, str], ...] = (
    ("function_calling", "Function Calling（原生）"),
    ("json_schema", "JSON Schema（原生）"),
    ("json_mode", "JSON Object（仅格式）"),
    ("prompt_json", "Cherry 兼容 JSON（本地校验）"),
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


def _option_index(options: tuple[tuple[str, ...], ...], value: str, *, default: int = 0) -> int:
    for index, option in enumerate(options):
        if option[0] == value:
            return index
    return default


class ModelConfigDialog(QDialog):
    """Edit the model settings consumed by the research planner."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("modelConfigDialog")
        self.setWindowTitle("大模型配置")
        self.setMinimumWidth(500)

        settings = get_settings()
        self.provider = QComboBox()
        for provider, label, _base_url, _model in MODEL_PRESETS:
            self.provider.addItem(label, provider)
        self.provider.setCurrentIndex(
            _option_index(MODEL_PRESETS, settings.llm_provider, default=len(MODEL_PRESETS) - 1)
        )
        self.provider.setToolTip("服务商只决定已知的请求参数，不替代模型能力检测。")
        self.provider.currentIndexChanged.connect(self._apply_provider_preset)

        self.api_style = QComboBox()
        for style, label in API_STYLE_OPTIONS:
            self.api_style.addItem(label, style)
        self.api_style.setCurrentIndex(_option_index(API_STYLE_OPTIONS, settings.llm_api_style))
        self.api_style.setToolTip("必须与服务端实际支持的消息协议一致，不会自动切换。")

        self.structured_method = QComboBox()
        for method, label in STRUCTURED_OUTPUT_OPTIONS:
            self.structured_method.addItem(label, method)
        self.structured_method.setCurrentIndex(
            _option_index(
                STRUCTURED_OUTPUT_OPTIONS,
                settings.llm_structured_output_method,
            )
        )
        self.structured_method.setToolTip(
            "Cherry 转发 Gemini 时请选择兼容 JSON；结果仍由本地 schema 严格校验。"
        )

        self.base_url = QLineEdit(settings.llm_base_url)
        self.base_url.setClearButtonEnabled(True)
        self.base_url.setPlaceholderText("https://api.example.com/v1")

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
        self.model.setPlaceholderText("请输入模型名称")

        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form.addRow("服务商", self.provider)
        form.addRow("API 形式", self.api_style)
        form.addRow("结构化输出", self.structured_method)
        form.addRow("接口地址", self.base_url)
        form.addRow("API 密钥", api_key_row)
        form.addRow("模型", self.model)

        protocol_notice = QLabel(
            "原生协议优先。Cherry 未转发 Gemini schema 时，可显式选择兼容 JSON；"
            "程序会把 schema 随提示发送，并在本地严格校验，失败结果不会进入分析。"
        )
        protocol_notice.setObjectName("modelProtocolNotice")
        protocol_notice.setWordWrap(True)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel,
        )
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(protocol_notice)
        layout.addWidget(buttons)
        self._sync_provider_fields()

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
        provider, _label, base_url, model = MODEL_PRESETS[index]
        if provider == "gemini":
            self.base_url.clear()
            self.model.setText(model)
            self.structured_method.setCurrentIndex(
                self.structured_method.findData("json_schema")
            )
        elif provider in {"deepseek", "qwen"}:
            self.base_url.setText(base_url)
            self.model.setText(model)
            self.structured_method.setCurrentIndex(
                self.structured_method.findData("function_calling")
            )
        elif base_url or model:
            self.base_url.setText(base_url)
            self.model.setText(model)
        self._sync_provider_fields()

    def _sync_provider_fields(self) -> None:
        native_gemini = self.provider.currentData() == "gemini"
        self.api_style.setEnabled(not native_gemini)
        self.base_url.setEnabled(not native_gemini)
        self.structured_method.setEnabled(not native_gemini)
        self.base_url.setPlaceholderText(
            "Gemini 原生适配器不经过 OpenAI 代理"
            if native_gemini
            else "https://api.example.com/v1"
        )

    def _validate(self) -> str | None:
        base_url = self.base_url.text().strip()
        model = self.model.text().strip()
        native_gemini = self.provider.currentData() == "gemini"
        if not native_gemini and not base_url:
            return "必须填写接口地址。"
        parsed = urlparse(base_url) if base_url else None
        if parsed is not None and (
            parsed.scheme not in {"http", "https"} or not parsed.netloc
        ):
            return "接口地址必须是完整的 http:// 或 https:// 地址。"
        if not model:
            return "必须填写模型名称。"
        return None

    def _save(self) -> None:
        error = self._validate()
        if error:
            QMessageBox.warning(self, "配置不完整", error)
            return
        provider = self.provider.currentData()
        if provider == "gemini":
            structured_output_method = "json_schema"
        elif provider in {"deepseek", "qwen"}:
            structured_output_method = "function_calling"
        else:
            structured_output_method = self.structured_method.currentData()
        values = {
            "VPP_LLM_ENABLED": "true",
            "VPP_LLM_PROVIDER": provider,
            "VPP_LLM_API_STYLE": (
                "chat" if provider == "gemini" else self.api_style.currentData()
            ),
            "VPP_LLM_STRUCTURED_OUTPUT_METHOD": structured_output_method,
            "VPP_LLM_BASE_URL": (
                "" if provider == "gemini" else self.base_url.text().strip()
            ),
            "VPP_LLM_API_KEY": self.api_key.text(),
            "VPP_LLM_MODEL": self.model.text().strip(),
        }
        try:
            update_env_values(runtime_env_file(), values)
            get_settings.cache_clear()
        except OSError as exc:
            QMessageBox.critical(self, "保存失败", f"无法保存模型配置：{exc}")
            return
        self.accept()
