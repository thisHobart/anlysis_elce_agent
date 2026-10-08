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
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QSpinBox,
    QVBoxLayout,
)

from app.config import get_application_settings, get_settings, runtime_env_file
from app.llm.model_profiles import ModelCapabilities, ModelProfile, resolve_model_profile, upsert_user_model_profile
from app.llm.runtime_settings import save_application_settings

MODEL_PRESETS: tuple[tuple[str, str, str, str], ...] = (
    ("deepseek", "DeepSeek", "https://api.deepseek.com/v1", "deepseek-v4-flash"),
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
    ("prompt_json", "Cherry 兼容 JSON/函数选择（本地校验）"),
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
            "Cherry 不转发 schema 或 tools 时请选择兼容模式；结构化结果、函数白名单和参数仍在本地校验。"
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
        self.base_url.textChanged.connect(self._refresh_profile)
        self.model.textChanged.connect(self._refresh_profile)

        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form.addRow("服务商", self.provider)
        form.addRow("接口地址", self.base_url)
        form.addRow("API 密钥", api_key_row)
        form.addRow("模型", self.model)

        self.profile_status = QLabel()
        self.profile_status.setObjectName("modelProfileStatus")
        self.profile_status.setWordWrap(True)
        self.context_window = QLineEdit()
        self.context_window.setPlaceholderText("例如 128000")
        self.max_output = QLineEdit()
        self.max_output.setPlaceholderText("例如 16384")
        profile_form = QFormLayout()
        profile_form.addRow("画像状态", self.profile_status)
        profile_form.addRow("API 形式", self.api_style)
        profile_form.addRow("结构化输出", self.structured_method)
        profile_form.addRow("上下文窗口", self.context_window)
        profile_form.addRow("最大输出", self.max_output)
        profile_group = QGroupBox("模型画像（未识别时一次性补全）")
        profile_group.setLayout(profile_form)

        application_settings = get_application_settings()
        runtime = application_settings.llm
        self.timeout = QDoubleSpinBox()
        self.timeout.setRange(1, 3600)
        self.timeout.setValue(runtime.timeout_seconds)
        self.timeout.setSuffix(" 秒")
        self.retries = QSpinBox()
        self.retries.setRange(0, 10)
        self.retries.setValue(runtime.max_retries)
        self.history_messages = QSpinBox()
        self.history_messages.setRange(1, 100)
        self.history_messages.setValue(runtime.history_messages)
        self.retrieved_turns = QSpinBox()
        self.retrieved_turns.setRange(0, 10)
        self.retrieved_turns.setValue(runtime.retrieved_turns)
        self.reasoning_effort = QComboBox()
        for value in ("", "none", "minimal", "low", "medium", "high", "xhigh", "max"):
            self.reasoning_effort.addItem("默认" if not value else value, value)
        self.reasoning_effort.setCurrentIndex(max(0, self.reasoning_effort.findData(runtime.reasoning_effort)))
        self.thinking_policy = QComboBox()
        self.thinking_policy.addItem("丢弃思考内容", "strip")
        self.thinking_policy.addItem("发现思考内容即拒绝", "reject")
        self.thinking_policy.setCurrentIndex(max(0, self.thinking_policy.findData(runtime.thinking_policy)))
        self.google_vertexai = QCheckBox("使用 Vertex AI（ADC）")
        self.google_vertexai.setChecked(runtime.gemini.vertexai)
        self.google_vertexai.toggled.connect(self._sync_vertex_fields)
        self.google_vertexai.toggled.connect(self._refresh_profile)
        self.google_project = QLineEdit(runtime.gemini.project)
        self.google_project.setPlaceholderText("Google Cloud project ID")
        self.google_project.textChanged.connect(self._refresh_profile)
        self.google_location = QLineEdit(runtime.gemini.location)
        self.google_location.setPlaceholderText("例如 us-central1")
        self.google_location.textChanged.connect(self._refresh_profile)
        runtime_form = QFormLayout()
        runtime_form.addRow("请求超时", self.timeout)
        runtime_form.addRow("端点重试", self.retries)
        runtime_form.addRow("历史消息", self.history_messages)
        runtime_form.addRow("相关历史轮次", self.retrieved_turns)
        runtime_form.addRow("推理强度", self.reasoning_effort)
        runtime_form.addRow("思考内容策略", self.thinking_policy)
        runtime_form.addRow("Gemini 连接", self.google_vertexai)
        runtime_form.addRow("Vertex 项目", self.google_project)
        runtime_form.addRow("Vertex 地区", self.google_location)
        runtime_group = QGroupBox("运行设置")
        runtime_group.setCheckable(True)
        runtime_group.setChecked(False)
        runtime_group.setLayout(runtime_form)

        protocol_notice = QLabel(
            "原生协议优先。Cherry 未转发 Gemini schema 或 tools 时，可显式选择兼容模式；"
            "程序会把 schema 和函数白名单随提示发送，并在本地严格校验，失败结果不会进入分析。"
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
        layout.addWidget(profile_group)
        layout.addWidget(runtime_group)
        layout.addWidget(protocol_notice)
        layout.addWidget(buttons)
        self._sync_provider_fields()
        self._refresh_profile()

    @staticmethod
    def _display_api_key(value: str) -> str:
        return "" if value.strip().casefold() in {"none", "null"} else value

    def _toggle_api_key_visibility(self, visible: bool) -> None:
        self.api_key.setEchoMode(QLineEdit.EchoMode.Normal if visible else QLineEdit.EchoMode.Password)

    def _apply_provider_preset(self, index: int) -> None:
        if index < 0 or index >= len(MODEL_PRESETS):
            return
        provider, _label, base_url, model = MODEL_PRESETS[index]
        if provider == "gemini":
            self.base_url.clear()
            self.model.setText(model)
            self.structured_method.setCurrentIndex(self.structured_method.findData("json_schema"))
        elif provider in {"deepseek", "qwen"}:
            self.base_url.setText(base_url)
            self.model.setText(model)
            self.structured_method.setCurrentIndex(self.structured_method.findData("function_calling"))
        elif base_url or model:
            self.base_url.setText(base_url)
            self.model.setText(model)
        self._sync_provider_fields()

    def _sync_provider_fields(self) -> None:
        native_gemini = self.provider.currentData() == "gemini"
        self.api_style.setEnabled(not native_gemini)
        self.base_url.setEnabled(not native_gemini)
        self.structured_method.setEnabled(not native_gemini)
        self.google_vertexai.setEnabled(native_gemini)
        self._sync_vertex_fields()
        self.base_url.setPlaceholderText(
            "Gemini 原生适配器不经过 OpenAI 代理" if native_gemini else "https://api.example.com/v1"
        )

    def _sync_vertex_fields(self) -> None:
        enabled = self.provider.currentData() == "gemini" and self.google_vertexai.isChecked()
        self.google_project.setEnabled(enabled)
        self.google_location.setEnabled(enabled)

    def _route_url(self) -> str:
        if self.provider.currentData() != "gemini":
            return self.base_url.text().strip()
        if self.google_vertexai.isChecked():
            return f"vertex://{self.google_project.text().strip()}/{self.google_location.text().strip()}"
        return ""

    def _refresh_profile(self) -> None:
        model = self.model.text().strip()
        if not model:
            self.profile_status.setText("模型能力未验证：填写模型名称后可匹配画像。")
            self.context_window.clear()
            self.max_output.clear()
            return
        resolved = resolve_model_profile(
            provider=self.provider.currentData(),
            base_url=self._route_url(),
            model=model,
        )
        if resolved.profile is None:
            self.profile_status.setText("模型能力未验证；可补全容量和协议后保存用户画像。")
            self.context_window.clear()
            self.max_output.clear()
            return
        profile = resolved.profile
        self.profile_status.setText(
            f"已解析：{resolved.source}；上下文 {profile.context_window_tokens}，最大输出 {profile.max_output_tokens}。"
        )
        self.context_window.setText(str(profile.context_window_tokens))
        self.max_output.setText(str(profile.max_output_tokens))
        self.api_style.setCurrentIndex(max(0, self.api_style.findData(profile.api_style)))
        self.structured_method.setCurrentIndex(
            max(0, self.structured_method.findData(profile.structured_output_method))
        )

    def _validate(self) -> str | None:
        base_url = self.base_url.text().strip()
        model = self.model.text().strip()
        native_gemini = self.provider.currentData() == "gemini"
        if not native_gemini and not base_url:
            return "必须填写接口地址。"
        parsed = urlparse(base_url) if base_url else None
        if parsed is not None and (parsed.scheme not in {"http", "https"} or not parsed.netloc):
            return "接口地址必须是完整的 http:// 或 https:// 地址。"
        if not model:
            return "必须填写模型名称。"
        if (
            native_gemini
            and self.google_vertexai.isChecked()
            and (not self.google_project.text().strip() or not self.google_location.text().strip())
        ):
            return "使用 Vertex AI 时必须填写项目和地区。"
        context_value = self.context_window.text().strip()
        output_value = self.max_output.text().strip()
        if bool(context_value) != bool(output_value):
            return "补全模型画像时必须同时填写上下文窗口和最大输出。"
        if context_value:
            try:
                if int(context_value) <= 0 or int(output_value) < 256:
                    raise ValueError
            except ValueError:
                return "上下文窗口必须为正整数，最大输出不得小于 256。"
        return None

    def _save(self) -> None:
        error = self._validate()
        if error:
            QMessageBox.warning(self, "配置不完整", error)
            return
        provider = self.provider.currentData()
        values = {
            "VPP_LLM_PROVIDER": provider,
            "VPP_LLM_BASE_URL": ("" if provider == "gemini" else self.base_url.text().strip()),
            "VPP_LLM_API_KEY": self.api_key.text(),
            "VPP_LLM_MODEL": self.model.text().strip(),
        }
        try:
            update_env_values(runtime_env_file(), values)
            application = get_application_settings()
            current_runtime = application.llm
            gemini = (
                current_runtime.gemini.model_copy(
                    update={
                        "vertexai": self.google_vertexai.isChecked(),
                        "project": self.google_project.text().strip(),
                        "location": self.google_location.text().strip(),
                    }
                )
                if provider == "gemini"
                else current_runtime.gemini
            )
            runtime = current_runtime.model_copy(
                update={
                    "timeout_seconds": self.timeout.value(),
                    "max_retries": self.retries.value(),
                    "history_messages": self.history_messages.value(),
                    "retrieved_turns": self.retrieved_turns.value(),
                    "reasoning_effort": self.reasoning_effort.currentData(),
                    "thinking_policy": self.thinking_policy.currentData(),
                    "unverified_api_style": "chat" if provider == "gemini" else self.api_style.currentData(),
                    "unverified_structured_output_method": (
                        "json_schema" if provider == "gemini" else self.structured_method.currentData()
                    ),
                    "gemini": gemini,
                }
            )
            save_application_settings(application.model_copy(update={"llm": runtime}))
            if self.context_window.text().strip():
                method = "json_schema" if provider == "gemini" else self.structured_method.currentData()
                upsert_user_model_profile(
                    ModelProfile(
                        provider=provider,
                        base_url=self._route_url(),
                        model=self.model.text().strip(),
                        context_window_tokens=int(self.context_window.text()),
                        max_output_tokens=int(self.max_output.text()),
                        api_style="chat" if provider == "gemini" else self.api_style.currentData(),
                        structured_output_method=method,
                        capabilities=ModelCapabilities(
                            function_calling=method == "function_calling",
                            json_schema=method == "json_schema",
                            json_mode=method in {"json_mode", "prompt_json"},
                            parallel_tool_calls=method == "function_calling",
                        ),
                    )
                )
            clear_settings = getattr(get_settings, "cache_clear", None)
            if callable(clear_settings):
                clear_settings()
            clear_application = getattr(get_application_settings, "cache_clear", None)
            if callable(clear_application):
                clear_application()
        except OSError as exc:
            QMessageBox.critical(self, "保存失败", f"无法保存模型配置：{exc}")
            return
        self.accept()
