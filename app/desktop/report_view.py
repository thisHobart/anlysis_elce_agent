"""Read a Markdown research report, with its SVG figures, without leaving the app."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QRectF, Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QImage, QPainter, QTextDocument
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

MINIMUM_FIGURE_WIDTH = 320
FIGURE_MARGIN = 48
REPORT_STYLE = """
QTextBrowser#reportBody {
    background: #FFFFFF;
    border: none;
    padding: 8px 28px 28px;
    font-size: 14px;
}
QFrame#reportActions { background: #F6F7F9; }
"""


class ReportBrowser(QTextBrowser):
    """Render report figures at the width the reader actually has."""

    def __init__(self, base_directory: Path) -> None:
        super().__init__()
        self._base_directory = base_directory
        self.setObjectName("reportBody")
        self.setOpenExternalLinks(False)
        self.setOpenLinks(False)
        self.setSearchPaths([str(base_directory)])
        self.document().setBaseUrl(QUrl.fromLocalFile(base_directory.as_posix() + "/"))

    def loadResource(self, resource_type: int, url: QUrl) -> object:
        """Rasterise SVG figures to fit the viewport instead of overflowing it."""

        if resource_type == QTextDocument.ResourceType.ImageResource:
            path = self._resolve(url)
            if path is not None and path.suffix.lower() == ".svg":
                image = self._render_svg(path)
                if image is not None:
                    return image
        return super().loadResource(resource_type, url)

    def _resolve(self, url: QUrl) -> Path | None:
        candidate = Path(url.toLocalFile() if url.isLocalFile() else url.toString())
        if not candidate.is_absolute():
            candidate = self._base_directory / candidate
        return candidate if candidate.is_file() else None

    def _render_svg(self, path: Path) -> QImage | None:
        renderer = QSvgRenderer(str(path))
        if not renderer.isValid():
            return None
        natural = renderer.defaultSize()
        if natural.width() <= 0 or natural.height() <= 0:
            return None
        available = max(MINIMUM_FIGURE_WIDTH, self.viewport().width() - FIGURE_MARGIN)
        width = min(natural.width(), available)
        height = round(natural.height() * width / natural.width())
        ratio = self.devicePixelRatioF()
        image = QImage(round(width * ratio), round(height * ratio), QImage.Format.Format_ARGB32_Premultiplied)
        image.setDevicePixelRatio(ratio)
        image.fill(Qt.GlobalColor.white)
        painter = QPainter(image)
        # QSvgRenderer otherwise uses the QImage's physical pixel bounds. On a
        # high-DPI screen that applies the device ratio twice, which can crop
        # the chart and corrupt fallback-font glyph spacing.
        renderer.render(painter, QRectF(0, 0, width, height))
        painter.end()
        return image


class ReportWindow(QDialog):
    """Show one report package's `report.md` with figures resolved from `figures/`."""

    def __init__(self, report_path: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._report_path = report_path
        self.setWindowTitle(f"研究报告 · {report_path.parent.name}")
        self.setStyleSheet(REPORT_STYLE)
        self.setMinimumSize(640, 480)
        self.resize(1000, 780)

        self.browser = ReportBrowser(report_path.parent)
        self.browser.anchorClicked.connect(self._open_anchor)

        # A full path in a plain QLabel sets the dialog's minimum width, so keep it short.
        heading = QLabel(f"{report_path.parent.name} / {report_path.name}")
        heading.setObjectName("reportPath")
        heading.setToolTip(str(report_path))
        heading.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        heading.setStyleSheet("color:#667085;font-size:12px;padding:10px 28px 0;")

        open_externally = QPushButton("用系统程序打开")
        open_externally.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(report_path)))
        )
        open_folder = QPushButton("打开结果文件夹")
        open_folder.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(report_path.parent)))
        )
        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.accept)

        actions = QHBoxLayout()
        actions.setContentsMargins(28, 8, 28, 12)
        actions.addWidget(open_externally)
        actions.addWidget(open_folder)
        actions.addStretch(1)
        actions.addWidget(close_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(heading)
        layout.addWidget(self.browser, 1)
        layout.addLayout(actions)

        # Figures are rasterised against the viewport, so a resize needs a re-render.
        self._relayout = QTimer(self)
        self._relayout.setSingleShot(True)
        self._relayout.setInterval(180)
        self._relayout.timeout.connect(self._render)
        self._render()

    def _render(self) -> None:
        try:
            markdown = self._report_path.read_text(encoding="utf-8")
        except OSError as error:
            self.browser.setPlainText(f"无法读取报告文件：{error}")
            return
        offset = self.browser.verticalScrollBar().value()
        self.browser.document().clear()
        self.browser.setMarkdown(markdown)
        self.browser.verticalScrollBar().setValue(offset)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._relayout.start()

    def _open_anchor(self, url: QUrl) -> None:
        """Follow a sibling Markdown file in place; send everything else to the OS."""

        target = Path(url.toLocalFile()) if url.isLocalFile() else self._report_path.parent / url.toString()
        if target.suffix.lower() == ".md" and target.is_file():
            self._report_path = target
            self.setWindowTitle(f"研究报告 · {target.name}")
            self._render()
            return
        QDesktopServices.openUrl(url if url.isLocalFile() else QUrl.fromLocalFile(str(target)))


def open_report(report_path: str, parent: QWidget | None = None) -> None:
    """Open Markdown reports in the app; hand anything else to the operating system."""

    path = Path(report_path)
    if path.suffix.lower() == ".md" and path.is_file():
        ReportWindow(path, parent).exec()
        return
    QDesktopServices.openUrl(QUrl.fromLocalFile(report_path))
