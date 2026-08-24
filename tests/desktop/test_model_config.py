"""Tests for the desktop model configuration persistence boundary."""

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLabel

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


def test_model_configuration_has_no_enable_planner_selector():
    app = QApplication.instance() or QApplication([])
    dialog = ModelConfigDialog()
    try:
        assert not hasattr(dialog, "enabled")
        label_text = " ".join(label.text() for label in dialog.findChildren(QLabel))
        assert "研究规划和方案修订必须调用大模型" in label_text
    finally:
        dialog.close()
        app.processEvents()
