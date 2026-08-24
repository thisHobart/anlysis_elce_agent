"""PySide6 desktop application bootstrap."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from app.desktop.main_window import MainWindow
from app.desktop.theme import APP_STYLE


def main(argv: Sequence[str] | None = None) -> int:
    """Create and run the desktop UI directly."""

    app = QApplication.instance() or QApplication(list(argv) if argv is not None else sys.argv)
    app.setApplicationName("电价与外生变量研究 Agent")
    app.setOrganizationName("Price Research")
    app.setFont(QFont("Microsoft YaHei UI", 10))
    app.setStyleSheet(APP_STYLE)
    window = MainWindow()
    window.show()
    return app.exec()
