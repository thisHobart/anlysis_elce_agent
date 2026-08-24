"""History, conversation, and context panes for the desktop workspace."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QAction, QDragEnterEvent, QDropEvent, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
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
    ToolMessageWidget,
)
from app.desktop.session import InputRole, ResearchSession, SessionInputFile, SessionMessage, TraceEvent
from app.research.agent.schemas import EDAPlan

STATUS_LABELS = {
    "idle": "等待输入",
    "understanding": "理解问题",
    "inspecting_data": "检查数据",
    "awaiting_plan_approval": "等待修改",
    "awaiting_user": "等待决策",
    "running": "运行中",
    "evaluating": "评估中",
    "completed": "已完成",
    "failed": "失败",
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
        self.new_button = QPushButton("＋  新建会话")
        self.new_button.setObjectName("newResearchButton")
        self.new_button.clicked.connect(self.new_requested)
        actions.addWidget(self.new_button, 1)
        self.delete_button = QPushButton("删除")
        self.delete_button.setObjectName("deleteSessionButton")
        self.delete_button.setToolTip("删除当前选中的会话（研究产物不会删除）")
        self.delete_button.setEnabled(False)
        self.delete_button.clicked.connect(self._delete_selected)
        actions.addWidget(self.delete_button)
        layout.addLayout(actions)
        self.search = QLineEdit()
        self.search.setPlaceholderText("搜索会话…")
        self.search.textChanged.connect(self._refresh)
        layout.addWidget(self.search)
        self.list = QListWidget()
        self.list.setObjectName("sessionList")
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
            title, accepted = QInputDialog.getText(self, "重命名会话", "会话标题", text=current_title)
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
        title_layout = QVBoxLayout()
        title_layout.setSpacing(1)
        self.title_label = QLabel("新研究")
        self.title_label.setObjectName("sessionTitle")
        self.config_label = QLabel("未使用 YAML 配置（可选）")
        self.config_label.setObjectName("sessionMeta")
        title_layout.addWidget(self.title_label)
        title_layout.addWidget(self.config_label)
        header_layout.addLayout(title_layout, 1)
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
        self.input.setPlaceholderText("提出电价与外生变量研究问题，或继续追问当前结果…")
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
        config = session.inputs["config"]
        self.config_label.setText(Path(config.path).name if config.path else "未使用 YAML 配置（可选）")
        self.clear_messages()
        for message in session.messages:
            self.render_message(message)
        self.scroll_to_bottom()

    def set_status(self, status: str) -> None:
        self.status_label.setText(STATUS_LABELS.get(status, status))

    def set_running(self, running: bool) -> None:
        self._running = running
        self.send_button.setText("停止" if running else "发送")
        self.input.setEnabled(not running)

    def clear_messages(self) -> None:
        while self.timeline_layout.count() > 1:
            item = self.timeline_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._message_widgets.clear()
        self.current_plan_widget = None

    def render_message(self, message: SessionMessage) -> QWidget:
        if message.kind == "text":
            widget: QWidget = TextMessageWidget(role=message.role, content=message.content)
        elif message.kind == "tool":
            widget = ToolMessageWidget(
                message.payload.get("title", "工具调用"),
                message.payload.get("detail", message.content),
                status=message.payload.get("status", "running"),
            )
        elif message.kind == "plan" and message.payload.get("plan"):
            widget = PlanMessageWidget(
                EDAPlan.model_validate(message.payload["plan"]),
                list(message.payload.get("available_variables", [])),
            )
            widget.run_requested.connect(self.plan_run_requested)
            plan_state = message.payload.get("state", "awaiting")
            if plan_state == "running":
                widget.set_running()
            elif plan_state in {"completed", "failed", "stopped", "stale"}:
                labels = {"completed": "已完成", "failed": "失败", "stopped": "已停止", "stale": "已过期"}
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
    """User-editable file slot with expandable variable evidence."""

    selected = Signal(str, str)
    cleared = Signal(str)

    def __init__(self, role: InputRole, label: str) -> None:
        super().__init__()
        self.role = role
        self.base_label = label
        self.setAcceptDrops(True)
        self.setObjectName("fileSlot")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 7, 8, 7)
        layout.setSpacing(5)
        top = QHBoxLayout()
        text = QVBoxLayout()
        text.setSpacing(1)
        self.name_label = QLabel("未选择文件")
        self.name_label.setObjectName("fileName")
        self.name_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.role_label = QLabel(label)
        self.role_label.setObjectName("fileRole")
        text.addWidget(self.name_label)
        text.addWidget(self.role_label)
        top.addLayout(text, 1)
        self.status_label = QLabel("未选择")
        self.status_label.setObjectName("fileStatus")
        self.status_label.setMaximumWidth(45)
        top.addWidget(self.status_label)
        self.details_button = QPushButton("变量")
        self.details_button.setObjectName("quietButton")
        self.details_button.setCheckable(True)
        self.details_button.hide()
        self.details_button.setMaximumWidth(58)
        top.addWidget(self.details_button)
        self.clear_button = QPushButton("×")
        self.clear_button.setObjectName("quietButton")
        self.clear_button.setToolTip("清除文件")
        self.clear_button.setMaximumWidth(28)
        self.clear_button.clicked.connect(lambda: self.cleared.emit(self.role))
        self.clear_button.hide()
        top.addWidget(self.clear_button)
        self.choose_button = QPushButton("选择")
        self.choose_button.setObjectName("quietButton")
        self.choose_button.setMaximumWidth(52)
        self.choose_button.clicked.connect(self.choose)
        top.addWidget(self.choose_button)
        layout.addLayout(top)
        self.variables = QTreeWidget()
        self.variables.setHeaderLabels(["变量", "覆盖率", "状态"])
        self.variables.setRootIsDecorated(False)
        self.variables.setMaximumHeight(112)
        self.variables.setVisible(False)
        self.details_button.toggled.connect(self.variables.setVisible)
        layout.addWidget(self.variables)

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
        self.name_label.setText(Path(item.path).name if item.path else "未选择文件")
        self.name_label.setToolTip(item.path)
        variable_suffix = f" · {len(item.variables)} 个变量" if item.variables else ""
        self.role_label.setText(f"{self.base_label}{variable_suffix}")
        labels = {
            "empty": "未选择",
            "selected": "待校验",
            "loading": "读取中",
            "ready": "就绪",
            "warning": "警告",
            "failed": "失败",
            "changed": "已变更",
        }
        self.status_label.setText(labels.get(item.status, item.status))
        self.status_label.setProperty("inputStatus", item.status)
        self.status_label.setToolTip(item.detail)
        self.choose_button.setText("更换" if item.path else "选择")
        self.clear_button.setVisible(bool(item.path))
        self.variables.clear()
        for variable in item.variables:
            coverage = "—" if variable.coverage_rate is None else f"{variable.coverage_rate:.1%}"
            row = QTreeWidgetItem([variable.name, coverage, "警告" if variable.status == "warning" else "就绪"])
            row.setToolTip(0, variable.detail)
            self.variables.addTopLevelItem(row)
        self.details_button.setText(f"变量 {len(item.variables)}")
        self.details_button.setVisible(bool(item.variables))
        self.variables.setVisible(bool(item.variables) and self.details_button.isChecked())

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        urls = event.mimeData().urls()
        if urls:
            self.selected.emit(self.role, urls[0].toLocalFile())
            event.acceptProposedAction()


class InputFilesPanel(QFrame):
    """The only entry point for the four supported research input files."""

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
        self.title_label = QLabel("文件输入")
        self.title_label.setObjectName("contextTitle")
        header.addWidget(self.title_label)
        header.addStretch(1)
        hint = QLabel("拖到对应位置或选择")
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
        for role in ("config", "target", "actuals", "forecasts"):
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


class PlanStatusPanel(QFrame):
    """Compact read-only mirror of the plan edited in the conversation."""

    plan_activated = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("planStatusPanel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(5)
        header = QHBoxLayout()
        title = QLabel("当前方案")
        title.setObjectName("contextTitle")
        header.addWidget(title)
        header.addStretch(1)
        self.summary = QLabel("尚未生成")
        self.summary.setObjectName("contextHint")
        header.addWidget(self.summary)
        layout.addLayout(header)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setRootIsDecorated(False)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.tree.itemClicked.connect(lambda _item, _column: self.plan_activated.emit())
        layout.addWidget(self.tree, 1)

    def set_plan(self, plan: EDAPlan | None, *, state: str = "pending") -> None:
        self.tree.clear()
        if plan is None:
            self.summary.setText("尚未生成")
            return
        enabled = plan.enabled_steps
        completed = len(enabled) if state == "completed" else 0
        self.summary.setText(f"{completed}/{len(enabled)}")
        markers = {"pending": "○", "running": "◉", "completed": "●", "failed": "!", "stopped": "■"}
        for step in enabled:
            self.tree.addTopLevelItem(QTreeWidgetItem([f"{markers.get(state, '○')}  {step.title}"]))


class TracePanel(QFrame):
    """Filterable, independently scrollable observable Agent run trace."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("tracePanel")
        self._events: list[TraceEvent] = []
        self._auto_follow = True
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(5)
        header = QHBoxLayout()
        title = QLabel("Agent 运行轨迹")
        title.setObjectName("contextTitle")
        header.addWidget(title)
        header.addStretch(1)
        self.filter = QComboBox()
        self.filter.addItem("全部", "all")
        for label, value in (("计划", "plan"), ("工具", "tool"), ("评估", "evaluation"), ("错误", "error")):
            self.filter.addItem(label, value)
        self.filter.currentIndexChanged.connect(self._refresh)
        header.addWidget(self.filter)
        layout.addLayout(header)
        self.tree = QTreeWidget()
        self.tree.setObjectName("traceTree")
        self.tree.setHeaderLabels(["时间", "类型", "事件", "状态"])
        self.tree.header().resizeSection(0, 62)
        self.tree.header().resizeSection(1, 58)
        self.tree.header().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.tree.header().resizeSection(3, 58)
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
        category_labels = {
            "session": "会话",
            "user": "用户",
            "agent": "Agent",
            "input": "输入",
            "plan": "计划",
            "tool": "工具",
            "evaluation": "评估",
            "artifact": "产物",
            "error": "错误",
        }
        status_labels = {
            "info": "信息",
            "running": "运行中",
            "completed": "完成",
            "warning": "警告",
            "failed": "失败",
            "stopped": "停止",
        }
        self.tree.clear()
        for event in self._events:
            if category != "all" and event.category != category:
                continue
            try:
                time_text = datetime.fromisoformat(event.created_at).astimezone().strftime("%H:%M:%S")
            except ValueError:
                time_text = "--:--:--"
            duration = f" · {event.duration_ms:.0f}ms" if event.duration_ms is not None else ""
            item = QTreeWidgetItem(
                [
                    time_text,
                    category_labels.get(event.category, event.category),
                    event.name,
                    f"{status_labels.get(event.status, event.status)}{duration}",
                ]
            )
            item.setToolTip(2, event.summary or str(event.details))
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


class ContextPane(QSplitter):
    """User inputs, compact plan state, and scrollable Agent trace."""

    file_selected = Signal(str, str)
    file_cleared = Signal(str)
    plan_activated = Signal()

    def __init__(self) -> None:
        super().__init__(Qt.Orientation.Vertical)
        self.setObjectName("contextPane")
        self.inputs = InputFilesPanel()
        self.plan = PlanStatusPanel()
        self.trace = TracePanel()
        self.inputs.file_selected.connect(self.file_selected)
        self.inputs.file_cleared.connect(self.file_cleared)
        self.plan.plan_activated.connect(self.plan_activated)
        self.addWidget(self.inputs)
        self.addWidget(self.plan)
        self.addWidget(self.trace)
        self.setSizes([300, 175, 330])
        self.setStretchFactor(0, 0)
        self.setStretchFactor(1, 0)
        self.setStretchFactor(2, 1)

    def set_session(self, session: ResearchSession) -> None:
        self.inputs.set_session(session)
        plan = EDAPlan.model_validate(session.current_plan) if session.current_plan else None
        plan_state = (
            "completed" if session.status == "completed" else "running" if session.status == "running" else "pending"
        )
        self.plan.set_plan(plan, state=plan_state)
        self.trace.set_events(session.trace)

    def set_busy(self, busy: bool) -> None:
        self.inputs.set_busy(busy)
