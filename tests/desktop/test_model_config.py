"""Tests for the desktop model configuration persistence boundary."""

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLabel

from app.config import Settings
from app.desktop.model_config import ModelConfigDialog, update_env_values


def test_update_env_values_preserves_comments_and_unknown_keys(tmp_path: Path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "# keep this comment\n"
        "VPP_LLM_ENABLED=false\n"
        "VPP_LLM_MODEL=old-model\n"
        "OTHER_SETTING=keep\n",
        encoding="utf-8",
    )

    update_env_values(
        env_path,
        {
            "VPP_LLM_ENABLED": "true",
            "VPP_LLM_MODEL": "new-model",
            "VPP_LLM_BASE_URL": "http://127.0.0.1:11434/v1",
        },
    )

    content = env_path.read_text(encoding="utf-8")
    assert "# keep this comment" in content
    assert "VPP_LLM_ENABLED=true" in content
    assert "VPP_LLM_MODEL=new-model" in content
    assert "VPP_LLM_BASE_URL=http://127.0.0.1:11434/v1" in content
    assert "OTHER_SETTING=keep" in content


def test_model_configuration_exposes_only_connection_fields():
    app = QApplication.instance() or QApplication([])
    dialog = ModelConfigDialog()
    try:
        assert not hasattr(dialog, "enabled")
        assert not hasattr(dialog, "timeout")
        assert not hasattr(dialog, "retries")
        assert not hasattr(dialog, "structured_mode")
        assert not hasattr(dialog, "thinking_policy")
        assert not hasattr(dialog, "history_messages")
        assert not hasattr(dialog, "reasoning_effort")
        assert not hasattr(dialog, "status_label")
        assert [dialog.provider.itemData(index) for index in range(dialog.provider.count())] == [
            "deepseek",
            "qwen",
            "custom",
        ]
        assert [dialog.api_style.itemData(index) for index in range(dialog.api_style.count())] == [
            "chat",
            "responses",
        ]
        label_text = " ".join(label.text() for label in dialog.findChildren(QLabel))
        assert "服务商" in label_text
        assert "API 形式" in label_text
        assert "接口地址" in label_text
        assert "API 密钥" in label_text
        assert "请求超时" not in label_text
        assert "历史消息" not in label_text
        assert "Function Calling" in label_text
        assert "不会用提示词 JSON 代替" in label_text
        assert "配置保存到" not in label_text
    finally:
        dialog.close()
        app.processEvents()


def test_model_configuration_save_preserves_backend_policy(
    monkeypatch,
    tmp_path: Path,
):
    app = QApplication.instance() or QApplication([])
    env_path = tmp_path / ".env"
    env_path.write_text(
        "VPP_LLM_TIMEOUT_SECONDS=180\n"
        "VPP_LLM_MAX_RETRIES=3\n"
        "VPP_LLM_THINKING_POLICY=reject\n"
        "VPP_LLM_REASONING_EFFORT=none\n"
        "VPP_LLM_HISTORY_MESSAGES=12\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("app.desktop.model_config.runtime_env_file", lambda: env_path)
    dialog = ModelConfigDialog()
    try:
        dialog.provider.setCurrentIndex(dialog.provider.findData("qwen"))
        dialog.api_style.setCurrentIndex(dialog.api_style.findData("responses"))
        dialog.base_url.setText("https://api.example.com/v1")
        dialog.api_key.setText("secret")
        dialog.model.setText("example-model")

        dialog._save()

        content = env_path.read_text(encoding="utf-8")
        assert "VPP_LLM_BASE_URL=https://api.example.com/v1" in content
        assert "VPP_LLM_API_KEY=secret" in content
        assert "VPP_LLM_MODEL=example-model" in content
        assert "VPP_LLM_PROVIDER=qwen" in content
        assert "VPP_LLM_API_STYLE=responses" in content
        assert "VPP_LLM_TIMEOUT_SECONDS=180" in content
        assert "VPP_LLM_MAX_RETRIES=3" in content
        assert "VPP_LLM_THINKING_POLICY=reject" in content
        assert "VPP_LLM_REASONING_EFFORT=none" in content
        assert "VPP_LLM_HISTORY_MESSAGES=12" in content
    finally:
        dialog.close()
        app.processEvents()


def test_old_environment_defaults_to_custom_chat_without_url_inference():
    settings = Settings(
        _env_file=None,
        llm_base_url="https://api.deepseek.com/v1",
        llm_model="legacy-model",
    )

    assert settings.llm_provider == "custom"
    assert settings.llm_api_style == "chat"


def test_backend_environment_can_set_hidden_reasoning_effort(tmp_path: Path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "VPP_LLM_PROVIDER=qwen\n"
        "VPP_LLM_API_STYLE=responses\n"
        "VPP_LLM_REASONING_EFFORT=none\n",
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_path)

    assert settings.llm_provider == "qwen"
    assert settings.llm_api_style == "responses"
    assert settings.llm_reasoning_effort == "none"
