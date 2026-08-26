"""History, conversation, and context panes for the desktop workspace."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QAction, QDragEnterEvent, QDropEvent, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.desktop.input_config import FILE_FILTERS, ROLE_LABELS
from app.desktop.message_widgets import (
    NoticeMessageWidget,
    PlanMessageWidget,
    ResultMessageWidget,
    TextMessageWidget,
    ThinkingMessageWidget,
    ToolMessageWidget,
)
from app.desktop.session import InputRole, ResearchSession, SessionInputFile, SessionMessage, TraceEvent
from app.research.agent.schemas import EDAPlan
from app.research.graph.narration import narrate_event

STATUS_LABELS = {
    "idle": "空闲",
    "understanding": "解析问题中",
    "inspecting_data": "检查数据中",
    "awaiting_plan_approval": "待确认方案",
    "awaiting_user": "待用户决策",
    "running": "分析中",
    "evaluating": "评估结果中",
    "completed": "已完成",
    "failed": "执行失败",
    "stopped": "已停止",
}


class HistoryPane(QFrame):
    """Searchable persisted conversation list."""

    new_requested = Signal()
    session_selected = Signal(str)
    rename_requested = Signal(str, str)
    delete_requested = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("historyPane")
        self._sessions: list[ResearchSession] = []
        self._selected_id: str | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 12, 10, 10)
        layout.setSpacing(8)

        actions = QHBoxLayout()
        self.new_button = QPushButton("＋  新的研究")
        self.new_button.setObjectName("newResearchButton")
        self.new_button.clicked.connect(self.new_requested)
        actions.addWidget(self.new_button, 1)
        self.delete_button = QPushButton("删除")
        self.delete_button.setObjectName("deleteSessionButton")
        self.delete_button.setToolTip("删除选中的研究记录（已生成的报告文件不会被删除）")
        self.delete_button.setEnabled(False)
        self.delete_button.clicked.connect(self._delete_selected)
        actions.addWidget(self.delete_button)
        layout.addLayout(actions)
        self.search = QLineEdit()
        self.search.setPlaceholderText("搜索历史研究…")
        self.search.textChanged.connect(self._refresh)
        layout.addWidget(self.search)
        self.list = QListWidget()
        self.list.setObjectName("sessionList")
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.list.setWordWrap(False)
        self.list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self._context_menu)
        self.list.currentItemChanged.connect(self._selection_changed)
        layout.addWidget(self.list, 1)

    def set_sessions(self, sessions: list[ResearchSession], selected_id: str | None) -> None:
        self._sessions = list(sessions)
        self._selected_id = selected_id
        self.delete_button.setEnabled(any(item.session_id == selected_id for item in self._sessions))
        self._refresh()

    def set_busy(self, busy: bool) -> None:
        self.new_button.setEnabled(not busy)
        self.delete_button.setEnabled(not busy and any(item.session_id == self._selected_id for item in self._sessions))
        self.list.setEnabled(not busy)

    def _delete_selected(self) -> None:
        if self._selected_id:
            self.delete_requested.emit(self._selected_id)

    def _group_name(self, value: str) -> str:
        try:
            updated = datetime.fromisoformat(value).astimezone()
            days = (datetime.now().astimezone().date() - updated.date()).days
        except ValueError:
            return "更早"
        if days == 0:
            return "今天"
        if days <= 7:
            return "过去 7 天"
        return "更早"

    def _refresh(self) -> None:
        query = self.search.text().strip().casefold()
        self.list.blockSignals(True)
        self.list.clear()
        grouped: dict[str, list[ResearchSession]] = {"今天": [], "过去 7 天": [], "更早": []}
        for session in sorted(self._sessions, key=lambda item: item.updated_at, reverse=True):
            if query and query not in session.title.casefold():
                continue
            grouped[self._group_name(session.updated_at)].append(session)
        selected_item: QListWidgetItem | None = None
        for group_name, sessions in grouped.items():
            if not sessions:
                continue
            header = QListWidgetItem(group_name)
            header.setFlags(Qt.ItemFlag.NoItemFlags)
            header.setData(Qt.ItemDataRole.UserRole, "")
            self.list.addItem(header)
            for session in sessions:
                status = STATUS_LABELS.get(session.status, session.status)
                item = QListWidgetItem(f"{session.title}\n{status}")
                item.setData(Qt.ItemDataRole.UserRole, session.session_id)
                item.setToolTip(session.title)
                self.list.addItem(item)
                if session.session_id == self._selected_id:
                    selected_item = item
        if selected_item is not None:
            self.list.setCurrentItem(selected_item)
        self.list.blockSignals(False)

    def _selection_changed(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        if current is None:
            return
        session_id = str(current.data(Qt.ItemDataRole.UserRole) or "")
        if session_id:
            self.session_selected.emit(session_id)

    def _context_menu(self, position: Any) -> None:
        item = self.list.itemAt(position)
        if item is None:
            return
        session_id = str(item.data(Qt.ItemDataRole.UserRole) or "")
        if not session_id:
            return
        menu = QMenu(self)
        rename = QAction("重命名", self)
        delete = QAction("删除", self)
        menu.addAction(rename)
        menu.addAction(delete)
        selected = menu.exec(self.list.mapToGlobal(position))
        if selected is rename:
            current_title = item.text().splitlines()[0]
            title, accepted = QInputDialog.getText(self, "重命名", "研究名称", text=current_title)
            if accepted and title.strip():
                self.rename_requested.emit(session_id, title.strip())
        elif selected is delete:
            self.delete_requested.emit(session_id)


class ConversationPane(QFrame):
    """One continuous typed-message timeline and fixed composer."""

    send_requested = Signal(str)
    cancel_requested = Signal()
    plan_run_requested = Signal(object)
    draft_changed = Signal(bool)

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("conversationPane")
        self._running = False
        self._message_widgets: dict[str, QWidget] = {}
        self.current_plan_widget: PlanMessageWidget | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QFrame()
        header.setObjectName("conversationHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(18, 9, 18, 9)
        self.title_label = QLabel("新研究")
        self.title_label.setObjectName("sessionTitle")
        header_layout.addWidget(self.title_label, 1)
        self.status_label = QLabel("等待输入")
        self.status_label.setObjectName("sessionStatus")
        header_layout.addWidget(self.status_label)
        layout.addWidget(header)

        self.scroll = QScrollArea()
        self.scroll.setObjectName("conversationScroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.timeline = QWidget()
        self.timeline_layout = QVBoxLayout(self.timeline)
        self.timeline_layout.setContentsMargins(24, 20, 24, 20)
        self.timeline_layout.setSpacing(14)
        self.timeline_layout.addStretch(1)
        self.scroll.setWidget(self.timeline)
        layout.addWidget(self.scroll, 1)

        composer = QFrame()
        composer.setObjectName("composer")
        composer_outer = QVBoxLayout(composer)
        composer_outer.setContentsMargins(18, 10, 18, 12)
        composer_outer.setSpacing(6)
        self.input = QPlainTextEdit()
        self.input.setObjectName("composerInput")
        self.input.setPlaceholderText("说说你想研究什么，例如“负荷对实时电价的影响有多大”“峰谷价差在夏天有什么不同”…")
        self.input.setMaximumHeight(92)
        self.input.textChanged.connect(lambda: self.draft_changed.emit(bool(self.input.toPlainText().strip())))
        composer_outer.addWidget(self.input)
        actions = QHBoxLayout()
        actions.addStretch(1)
        self.send_button = QPushButton("发送")
        self.send_button.setObjectName("primaryButton")
        self.send_button.clicked.connect(self._submit_or_cancel)
        actions.addWidget(self.send_button)
        composer_outer.addLayout(actions)
        layout.addWidget(composer)
        QShortcut(QKeySequence("Ctrl+Return"), self.input, activated=self._submit_or_cancel)

    def set_session(self, session: ResearchSession) -> None:
        self.title_label.setText(session.title)
        self.status_label.setText(STATUS_LABELS.get(session.status, session.status))
        self.clear_messages()
        for message in session.messages:
            self.render_message(message)
        self.scroll_to_bottom()

    def set_status(self, status: str) -> None:
        self.status_label.setText(STATUS_LABELS.get(status, status))

    def set_running(self, running: bool) -> None:
        self._running = running
        self.send_button.setText("停止" if running else "发送")
        self.send_button.setToolTip("停止当前分析" if running else "发送消息（Ctrl+Enter）")
        self.input.setEnabled(not running)

    def clear_messages(self) -> None:
        while self.timeline_layout.count() > 1:
            item = self.timeline_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.hide()
                widget.deleteLater()
        self._message_widgets.clear()
        self.current_plan_widget = None

    def render_message(self, message: SessionMessage) -> QWidget:
        if message.kind == "text":
            widget: QWidget = TextMessageWidget(role=message.role, content=message.content)
        elif message.kind == "thinking":
            widget = ThinkingMessageWidget.from_payload(message.payload)
        elif message.kind == "tool":
            widget = ToolMessageWidget(
                message.payload.get("title", "分析进行中"),
                message.payload.get("detail", message.content),
                status=message.payload.get("status", "running"),
            )
        elif message.kind == "plan" and message.payload.get("plan"):
            widget = PlanMessageWidget(
                EDAPlan.model_validate(message.payload["plan"]),
            )
            widget.run_requested.connect(self.plan_run_requested)
            plan_state = message.payload.get("state", "awaiting")
            if plan_state == "running":
                widget.set_running()
            elif plan_state in {"completed", "failed", "stopped", "stale"}:
                labels = {"completed": "已完成", "failed": "执行失败", "stopped": "已停止", "stale": "已作废"}
                widget.set_finished(labels[plan_state])
            self.current_plan_widget = widget
        elif message.kind == "result":
            widget = ResultMessageWidget.from_payload(message.payload)
        else:
            widget = NoticeMessageWidget(message.content, error=message.kind == "error")
        self._add_timeline_widget(widget, user_aligned=message.role == "user")
        self._message_widgets[message.message_id] = widget
        return widget

    def append_message(self, message: SessionMessage) -> QWidget:
        widget = self.render_message(message)
        self.scroll_to_bottom()
        return widget

    def _add_timeline_widget(self, widget: QWidget, *, user_aligned: bool) -> None:
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        if user_aligned:
            row_layout.addStretch(1)
            row_layout.addWidget(widget)
        else:
            row_layout.addWidget(widget)
            row_layout.addStretch(1)
        self.timeline_layout.insertWidget(self.timeline_layout.count() - 1, row)

    def scroll_to_bottom(self) -> None:
        QTimer.singleShot(
            0, lambda: self.scroll.verticalScrollBar().setValue(self.scroll.verticalScrollBar().maximum())
        )

    def scroll_to_plan(self) -> None:
        if self.current_plan_widget is not None:
            self.scroll.ensureWidgetVisible(self.current_plan_widget, 20, 40)

    def _submit_or_cancel(self) -> None:
        if self._running:
            self.cancel_requested.emit()
            return
        text = self.input.toPlainText().strip()
        if not text:
            return
        self.input.clear()
        self.send_requested.emit(text)


class FileSlotRow(QFrame):
    """Compact user-editable file slot; validation evidence remains Agent-internal."""

    selected = Signal(str, str)
    cleared = Signal(str)

    def __init__(self, role: InputRole, label: str) -> None:
        super().__init__()
        self.role = role
        self.base_label = label
        self.setAcceptDrops(True)
        self.setObjectName("fileSlot")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 7, 8, 7)
        layout.setSpacing(6)
        text = QVBoxLayout()
        text.setSpacing(1)
        self.name_label = QLabel("尚未选择")
        self.name_label.setObjectName("fileName")
        self.name_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.role_label = QLabel(label)
        self.role_label.setObjectName("fileRole")
        text.addWidget(self.name_label)
        text.addWidget(self.role_label)
        layout.addLayout(text, 1)
        self.clear_button = QPushButton("×")
        self.clear_button.setObjectName("quietButton")
        self.clear_button.setToolTip("移除这个文件")
        self.clear_button.setMaximumWidth(28)
        self.clear_button.clicked.connect(lambda: self.cleared.emit(self.role))
        self.clear_button.hide()
        layout.addWidget(self.clear_button)
        self.choose_button = QPushButton("选择")
        self.choose_button.setObjectName("quietButton")
        self.choose_button.setMaximumWidth(52)
        self.choose_button.clicked.connect(self.choose)
        layout.addWidget(self.choose_button)

    def choose(self) -> None:
        selected, _filter = QFileDialog.getOpenFileName(
            self,
            f"选择{self.base_label}",
            "",
            FILE_FILTERS[self.role],
        )
        if selected:
            self.selected.emit(self.role, selected)

    def set_file(self, item: SessionInputFile) -> None:
        self.name_label.setText(Path(item.path).name if item.path else "尚未选择")
        self.name_label.setToolTip(item.path)
        self.role_label.setText(self.base_label)
        self.choose_button.setText("更换" if item.path else "选择")
        self.clear_button.setVisible(bool(item.path))

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        urls = event.mimeData().urls()
        if urls:
            self.selected.emit(self.role, urls[0].toLocalFile())
            event.acceptProposedAction()


class InputFilesPanel(QFrame):
    """The only entry point for the three supported research data files."""

    file_selected = Signal(str, str)
    file_cleared = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("inputFilesPanel")
        self.rows: dict[InputRole, FileSlotRow] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        header = QHBoxLayout()
        self.title_label = QLabel("数据文件")
        self.title_label.setObjectName("contextTitle")
        header.addWidget(self.title_label)
        header.addStretch(1)
        hint = QLabel("可拖入文件")
        hint.setObjectName("contextHint")
        header.addWidget(hint)
        layout.addLayout(header)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        content.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(5)
        for role in ("target", "actuals", "forecasts"):
            row = FileSlotRow(role, ROLE_LABELS[role])
            row.selected.connect(self.file_selected)
            row.cleared.connect(self.file_cleared)
            self.rows[role] = row
            content_layout.addWidget(row)
        content_layout.addStretch(1)
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)

    def set_session(self, session: ResearchSession) -> None:
        for role, row in self.rows.items():
            row.set_file(session.inputs[role])

    def set_busy(self, busy: bool) -> None:
        for row in self.rows.values():
            row.choose_button.setEnabled(not busy)
            row.clear_button.setEnabled(not busy)


class TracePanel(QFrame):
    """Filterable, independently scrollable observable Agent run trace."""

    maximize_requested = Signal(bool)

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("tracePanel")
        self._events: list[TraceEvent] = []
        self._auto_follow = True
        self._maximized = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(5)
        header = QHBoxLayout()
        title = QLabel("研究过程")
        title.setObjectName("contextTitle")
        header.addWidget(title)
        header.addStretch(1)
        self.filter = QComboBox()
        self.filter.addItem("全部", "all")
        for label, value in (("方案", "plan"), ("分析", "tool"), ("评估", "evaluation"), ("异常", "error")):
            self.filter.addItem(label, value)
        self.filter.currentIndexChanged.connect(self._refresh)
        header.addWidget(self.filter)
        self.maximize_button = QPushButton("放大")
        self.maximize_button.setObjectName("quietButton")
        self.maximize_button.setCheckable(True)
        self.maximize_button.setToolTip("把研究过程铺满窗口")
        self.maximize_button.toggled.connect(self._toggle_maximized)
        header.addWidget(self.maximize_button)
        layout.addLayout(header)
        self.tree = QTreeWidget()
        self.tree.setObjectName("traceTree")
        self.tree.setRootIsDecorated(False)
        self.tree.setAlternatingRowColors(False)
        self.tree.setHeaderLabels(["时间", "阶段", "事件描述"])
        self.tree.header().resizeSection(0, 58)
        self.tree.header().resizeSection(1, 58)
        self.tree.header().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.tree.verticalScrollBar().sliderPressed.connect(self._pause_follow)
        layout.addWidget(self.tree, 1)
        self.latest_button = QPushButton("回到最新")
        self.latest_button.setObjectName("quietButton")
        self.latest_button.hide()
        self.latest_button.clicked.connect(self.follow_latest)
        layout.addWidget(self.latest_button)

    def set_events(self, events: list[TraceEvent]) -> None:
        self._events = list(events)
        self._refresh()

    def add_event(self, event: TraceEvent) -> None:
        self._events.append(event)
        self._refresh()

    def _refresh(self) -> None:
        category = self.filter.currentData() or "all"
        self.tree.clear()
        previous_title = ""
        for event in self._events:
            if category != "all" and event.category != category:
                continue
            try:
                time_text = datetime.fromisoformat(event.created_at).astimezone().strftime("%H:%M:%S")
            except ValueError:
                time_text = "--:--"
            step = narrate_event(
                {
                    "name": event.name,
                    "status": event.status,
                    "details": event.details,
                    "category": event.category,
                }
            )
            detail = step.detail or event.summary
            headline = step.title if not detail else f"{step.title} — {detail}"
            # Several loop events narrate to one title (finish plus verify, local plus graph copy);
            # the first row carries the richest detail, so later repeats are dropped.
            if step.title == previous_title:
                continue
            previous_title = step.title
            item = QTreeWidgetItem([time_text, step.stage_label, headline])
            item.setToolTip(2, f"{event.name}\n{event.summary}" if event.summary else event.name)
            self.tree.addTopLevelItem(item)
        if self._auto_follow:
            self.tree.scrollToBottom()

    def _pause_follow(self) -> None:
        self._auto_follow = False
        self.latest_button.show()

    def follow_latest(self) -> None:
        self._auto_follow = True
        self.latest_button.hide()
        self.tree.scrollToBottom()

    def _toggle_maximized(self, maximized: bool) -> None:
        self._maximized = maximized
        self.maximize_button.setText("还原" if maximized else "放大")
        self.maximize_requested.emit(maximized)

    def set_maximized(self, maximized: bool) -> None:
        if self.maximize_button.isChecked() != maximized:
            self.maximize_button.setChecked(maximized)


class ContextPane(QSplitter):
    """User inputs and a maximizable, scrollable Agent trace."""

    file_selected = Signal(str, str)
    file_cleared = Signal(str)
    trace_maximized = Signal(bool)

    def __init__(self) -> None:
        super().__init__(Qt.Orientation.Vertical)
        self.setObjectName("contextPane")
        self.inputs = InputFilesPanel()
        self.trace = TracePanel()
        self.inputs.file_selected.connect(self.file_selected)
        self.inputs.file_cleared.connect(self.file_cleared)
        self.trace.maximize_requested.connect(self._set_trace_maximized)
        self.addWidget(self.inputs)
        self.addWidget(self.trace)
        self.setSizes([290, 510])
        self.setStretchFactor(0, 0)
        self.setStretchFactor(1, 1)

    def set_session(self, session: ResearchSession) -> None:
        self.inputs.set_session(session)
        self.trace.set_events(session.trace)

    def set_busy(self, busy: bool) -> None:
        self.inputs.set_busy(busy)

    def _set_trace_maximized(self, maximized: bool) -> None:
        self.inputs.setVisible(not maximized)
        if maximized:
            self.setSizes([0, max(self.height(), 1)])
        else:
            self.setSizes([290, 510])
        self.trace_maximized.emit(maximized)
