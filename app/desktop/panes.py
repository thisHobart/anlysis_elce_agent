"""History, conversation, and context panes for the desktop workspace."""

from __future__ import annotations

from datetime import datetime
from html import escape
from typing import Any, ClassVar

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
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
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.desktop.message_widgets import (
    DataPlanMessageWidget,
    ForecastPlanMessageWidget,
    ForecastResultMessageWidget,
    NoticeMessageWidget,
    P2ReviewMessageWidget,
    ResultMessageWidget,
    TextMessageWidget,
    ThinkingMessageWidget,
    ToolMessageWidget,
)
from app.desktop.session import DataPanelState, ResearchSession, SessionMessage, TraceEvent
from app.research.agent.schemas import EDAPlan
from app.research.data.sources.summary import DataSummary, VariableLabel
from app.research.forecasting.contracts import ForecastPlan
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
    plan_reject_requested = Signal()
    plan_revise_requested = Signal()
    data_details_requested = Signal()
    end_research_requested = Signal()
    draft_changed = Signal(bool)
    p2_review_open_requested = Signal()
    p2_revalidate_requested = Signal(str)

    DEFAULT_INPUT_PLACEHOLDER = (
        "说说你想研究什么，例如“负荷对实时电价的影响有多大”“峰谷价差在夏天有什么不同”…"
    )
    CONTINUE_RESEARCH_PLACEHOLDER = (
        "输入下一步研究要求，例如：按推荐变量继续；删除 fcst_water；加入 actual_load 后继续…"
    )

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("conversationPane")
        self._running = False
        self._read_only = False
        self._interaction_kind: str | None = None
        self._message_widgets: dict[str, QWidget] = {}
        self.current_plan_widget: DataPlanMessageWidget | ForecastPlanMessageWidget | None = None
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
        self._bottom_scroll_timer = QTimer(self)
        self._bottom_scroll_timer.setSingleShot(True)
        self._bottom_scroll_timer.setInterval(40)
        self._bottom_scroll_timer.timeout.connect(self._scroll_to_bottom_now)
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
        self.input.setPlaceholderText(self.DEFAULT_INPUT_PLACEHOLDER)
        self.input.setMaximumHeight(92)
        self.input.textChanged.connect(lambda: self.draft_changed.emit(bool(self.input.toPlainText().strip())))
        composer_outer.addWidget(self.input)
        actions = QHBoxLayout()
        actions.addStretch(1)
        self.end_research_button = QPushButton("结束本轮研究")
        self.end_research_button.setObjectName("endResearchButton")
        self.end_research_button.setToolTip("保留当前结果并停止继续分析")
        self.end_research_button.clicked.connect(self.end_research_requested)
        self.end_research_button.hide()
        actions.addWidget(self.end_research_button)
        self.send_button = QPushButton("发送")
        self.send_button.setObjectName("primaryButton")
        self.send_button.clicked.connect(self._submit_or_cancel)
        actions.addWidget(self.send_button)
        composer_outer.addLayout(actions)
        layout.addWidget(composer)
        QShortcut(QKeySequence("Ctrl+Return"), self.input, activated=self._submit_or_cancel)

    def set_session(self, session: ResearchSession) -> None:
        self._read_only = session.read_only
        self.set_interaction_context(None)
        self.title_label.setText(session.title)
        self.status_label.setText(STATUS_LABELS.get(session.status, session.status))
        self.clear_messages()
        for message in session.messages:
            self.render_message(message)
        self.scroll_to_bottom()
        self._refresh_interaction_controls()

    def set_status(self, status: str) -> None:
        self.status_label.setText(STATUS_LABELS.get(status, status))

    def set_running(self, running: bool) -> None:
        self._running = running
        self.send_button.setText("停止" if running else "发送")
        self.send_button.setToolTip("停止当前分析" if running else "发送消息（Ctrl+Enter）")
        self._refresh_interaction_controls()

    def set_interaction_context(self, kind: str | None) -> None:
        """Adapt the composer to the current Graph interaction gate."""

        self._interaction_kind = kind
        self._refresh_interaction_controls()

    def _refresh_interaction_controls(self) -> None:
        continue_research = self._interaction_kind == "result_limitations"
        placeholder = self.CONTINUE_RESEARCH_PLACEHOLDER if continue_research else self.DEFAULT_INPUT_PLACEHOLDER
        if self._read_only:
            placeholder = "历史会话仅供查看；请使用“新的研究”继续。"
        self.input.setPlaceholderText(placeholder)
        self.input.setEnabled(not self._running and not self._read_only)
        self.send_button.setEnabled(not self._read_only)
        self.end_research_button.setVisible(continue_research and not self._running and not self._read_only)

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
        elif message.kind in {"plan", "data_plan"} and message.payload.get("plan"):
            plan = EDAPlan.model_validate(message.payload["plan"])
            # A conversation saved before data and analysis were confirmed together
            # has no stored dataset description, so those rows read 「待定」.
            widget = DataPlanMessageWidget(plan, summary=message.payload.get("data_summary"))
            widget.details_requested.connect(self.data_details_requested)
            widget.revise_requested.connect(self.plan_revise_requested)
            widget.run_requested.connect(self.plan_run_requested)
            widget.reject_requested.connect(self.plan_reject_requested)
            plan_state = message.payload.get("state", "awaiting")
            if plan_state == "running":
                widget.set_running()
            elif plan_state in {"completed", "failed", "stopped", "stale"}:
                labels = {"completed": "已完成", "failed": "执行失败", "stopped": "已停止", "stale": "已作废"}
                widget.set_finished(labels[plan_state])
            if self._read_only:
                widget.set_finished("历史记录")
            self.current_plan_widget = widget
        elif message.kind == "forecast_plan" and message.payload.get("plan"):
            plan = ForecastPlan.model_validate(message.payload["plan"])
            widget = ForecastPlanMessageWidget(plan)
            widget.run_requested.connect(self.plan_run_requested)
            widget.reject_requested.connect(self.plan_reject_requested)
            plan_state = message.payload.get("state", "awaiting")
            if plan_state == "running":
                widget.set_running()
            elif plan_state in {"completed", "failed", "stopped", "stale"}:
                labels = {
                    "completed": "已完成",
                    "failed": "执行失败",
                    "stopped": "已停止",
                    "stale": "已作废",
                }
                widget.set_finished(labels[plan_state])
            if self._read_only:
                widget.set_finished("历史记录")
            self.current_plan_widget = widget
        elif message.kind == "result":
            widget = ResultMessageWidget.from_payload(message.payload)
        elif message.kind == "forecast_result":
            widget = ForecastResultMessageWidget(message.payload)
        elif message.kind == "p2_review":
            widget = P2ReviewMessageWidget(message.payload, read_only=self._read_only)
            widget.open_requested.connect(self.p2_review_open_requested)
            widget.continue_requested.connect(self.p2_revalidate_requested)
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
        # A message can add nested, word-wrapped widgets whose final height is
        # only known after another layout pass. Apply immediately, after the
        # current event, and once more after delayed size hints settle. The
        # single timer is restarted by bursts of progress updates, so they do
        # not queue an unbounded number of callbacks.
        self.timeline_layout.activate()
        self._scroll_to_bottom_now()
        self._bottom_scroll_timer.start()

    def _scroll_to_bottom_now(self) -> None:
        scrollbar = self.scroll.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _submit_or_cancel(self) -> None:
        if self._running:
            self.cancel_requested.emit()
            return
        text = self.input.toPlainText().strip()
        if not text:
            return
        self.input.clear()
        self.send_requested.emit(text)
        self.scroll_to_bottom()


DATA_FIELD_KEYS: tuple[str, ...] = ("电价", "影响因素", "时间范围")


def variables_markup(variables: list[VariableLabel]) -> str:
    """Join variable names, marking the ones still shown under their column name."""

    return "、".join(
        escape(item.display)
        if item.resolved
        else f'<span style="color:#B45309">{escape(item.display)}</span>'
        for item in variables
    )


def factor_kind(variable: VariableLabel) -> str:
    """Classify old and new summaries without guessing when source metadata exists."""

    if variable.kind in {"actual", "forecast"}:
        return variable.kind
    folded = variable.display.casefold()
    forecast_markers = ("预测", "forecast", "fcst", "prediction", "predicted")
    return "forecast" if any(marker in folded for marker in forecast_markers) else "actual"


def factor_groups(variables: list[VariableLabel]) -> tuple[list[VariableLabel], list[VariableLabel]]:
    """Return actual and forecast factors in their original, reproducible order."""

    actual = [item for item in variables if factor_kind(item) == "actual"]
    forecast = [item for item in variables if factor_kind(item) == "forecast"]
    return actual, forecast


def variables_summary_markup(variables: list[VariableLabel]) -> str:
    """Compact factor content for the narrow context panel."""

    actual, forecast = factor_groups(variables)
    parts = []
    if actual:
        parts.append(f"实际值 {len(actual)} 项")
    if forecast:
        parts.append(f"预测值 {len(forecast)} 项")
    return "、".join(parts)


class DataRow(QWidget):
    """One field of the dataset description: its name, its value, and how settled it is."""

    PENDING_TEXT = "待定"

    def __init__(self, key: str) -> None:
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 9, 10, 9)
        layout.setSpacing(8)
        self.mark = QLabel("")
        self.mark.setObjectName("dataMark")
        self.mark.setFixedWidth(12)
        self.mark.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.mark)
        text = QVBoxLayout()
        text.setSpacing(2)
        self.key_label = QLabel(key)
        self.key_label.setObjectName("dataKey")
        text.addWidget(self.key_label)
        self.value_label = QLabel(self.PENDING_TEXT)
        self.value_label.setObjectName("dataValue")
        self.value_label.setWordWrap(True)
        self.value_label.setTextFormat(Qt.TextFormat.RichText)
        self.value_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        text.addWidget(self.value_label)
        # Parented on construction: setVisible on a parentless widget flashes it as a window.
        self.sub_label = QLabel("", self)
        self.sub_label.setObjectName("dataSub")
        self.sub_label.setWordWrap(True)
        self.sub_label.setVisible(False)
        text.addWidget(self.sub_label)
        layout.addLayout(text, 1)

    def set_value(self, markup: str, *, sub: str = "", state: str = "settled", tooltip: str = "") -> None:
        """Render one field; `state` is settled, pending, or waiting."""

        marks = {"settled": "✓", "pending": "", "waiting": "○"}
        self.mark.setText(marks.get(state, ""))
        self.value_label.setText(markup or self.PENDING_TEXT)
        self.value_label.setToolTip(tooltip)
        self.sub_label.setText(sub)
        self.sub_label.setVisible(bool(sub))
        for widget in (self.mark, self.value_label):
            widget.setProperty("dataState", state if markup else "waiting")
            widget.style().unpolish(widget)
            widget.style().polish(widget)


class DataCard(QFrame):
    """White card holding the dataset fields, separated by hairlines."""

    def __init__(self, keys: tuple[str, ...] = DATA_FIELD_KEYS) -> None:
        super().__init__()
        self.setObjectName("dataCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.rows: dict[str, DataRow] = {}
        for index, key in enumerate(keys):
            if index:
                divider = QFrame()
                divider.setObjectName("dataDivider")
                divider.setFixedHeight(1)
                layout.addWidget(divider)
            row = DataRow(key)
            self.rows[key] = row
            layout.addWidget(row)

    def apply(
        self,
        summary: DataSummary | None,
        *,
        exploring: bool = False,
        max_variables: int | None = None,
        summarize_variables: bool = False,
    ) -> None:
        """Fill the card from the one description the UI is allowed to read."""

        summary = summary or DataSummary()
        unsettled = "waiting" if exploring else "settled"
        self.rows["电价"].set_value(
            escape(summary.price_label),
            state="settled" if summary.price_label else unsettled,
        )
        variables = summary.variables
        shown_variables = variables[:max_variables] if max_variables else variables
        if shown_variables and exploring:
            markup = "正在核对：" + "、".join(escape(item.display) for item in shown_variables)
            state = "pending"
        elif variables and summarize_variables:
            markup = variables_summary_markup(variables)
            state = "settled"
        else:
            markup = variables_markup(shown_variables)
            state = "settled" if variables else unsettled
        if not summarize_variables and max_variables and len(variables) > max_variables:
            markup = f"{markup}…（共 {len(variables)} 项）"
        tooltip_parts = []
        if summarize_variables or len(shown_variables) < len(variables):
            tooltip_parts.append("、".join(item.display for item in variables))
        if any(not item.resolved for item in variables):
            tooltip_parts.append("这项数据还没配中文名")
        self.rows["影响因素"].set_value(
            markup,
            state=state,
            tooltip="\n".join(tooltip_parts),
        )
        window = f"{summary.start_date} — {summary.end_date}" if summary.start_date else ""
        sub = summary.granularity_text
        if sub and summary.gap_text:
            sub = f"{sub}，{summary.gap_text}"
        self.rows["时间范围"].set_value(
            escape(window),
            sub=sub,
            state="settled" if window else unsettled,
        )


class FactorListDialog(QDialog):
    """Resizable, searchable view of every factor in the current dataset."""

    def __init__(self, variables: list[VariableLabel], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("全部影响因素")
        self.setMinimumSize(440, 360)
        self.resize(560, 520)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)

        self.summary_label = QLabel(f"共 {len(variables)} 项；可搜索，或展开分组查看。")
        self.summary_label.setObjectName("dataSub")
        layout.addWidget(self.summary_label)
        self.search = QLineEdit(self)
        self.search.setPlaceholderText("搜索影响因素")
        self.search.setClearButtonEnabled(True)
        layout.addWidget(self.search)
        self.tree = QTreeWidget(self)
        self.tree.setObjectName("factorTree")
        self.tree.setHeaderHidden(True)
        self.tree.setRootIsDecorated(True)
        self.tree.setAlternatingRowColors(True)
        layout.addWidget(self.tree, 1)

        self._groups: list[QTreeWidgetItem] = []
        actual, forecast = factor_groups(variables)
        grouped = (("实际值", actual), ("预测值", forecast))
        for title, items in grouped:
            if not items:
                continue
            group = QTreeWidgetItem([f"{title}（{len(items)}）"])
            group.setFlags(group.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self.tree.addTopLevelItem(group)
            self._groups.append(group)
            for variable in items:
                child = QTreeWidgetItem([variable.display])
                if not variable.resolved:
                    child.setToolTip(0, "这项数据还没配中文名")
                group.addChild(child)
            group.setExpanded(True)
        self.search.textChanged.connect(self._filter)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _filter(self, text: str) -> None:
        wanted = text.strip().casefold()
        for group in self._groups:
            visible_children = 0
            for index in range(group.childCount()):
                child = group.child(index)
                visible = not wanted or wanted in child.text(0).casefold()
                child.setHidden(not visible)
                visible_children += int(visible)
            group.setHidden(visible_children == 0)


class DataPanel(QFrame):
    """What data this research runs on, in the words an analyst already uses.

    The panel never reads a database contract; it renders one `DataSummary` that
    was formatted once, upstream, which is why it and the confirmation card can
    never word the same dataset differently.
    """

    refetch_requested = Signal()
    reselect_requested = Signal()
    details_requested = Signal()
    retry_requested = Signal()
    local_file_requested = Signal()
    region_selected = Signal(str)

    EMPTY_TEXT = "还没取数。请从上方选择地区；也可以直接提问讨论研究方法。"
    EXPLORING_TEXT = "正在看有哪些数据能用…"
    EXPLORING_NOTE = "现在只是在看有什么数据，还没开始取。找完会先给你确认。"
    UNAVAILABLE_TITLE = "现在取不到数据"
    UNAVAILABLE_BODY = "和数据服务器连不上。稍等一下再试；一直不行就找运维看看，或者先用本地文件继续。"
    UNAVAILABLE_HISTORY_TITLE = "上次用的数据"
    UNAVAILABLE_NOTE = "上次取的数据还在，可以直接接着分析，只是不是最新的。"
    STATUS_TEXT: ClassVar[dict[str, str]] = {"ready": "已就绪", "exploring": "正在找数据", "unavailable": "取不到"}
    SPINNER_FRAMES = ("◐", "◓", "◑", "◒")

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("dataPanel")
        self._state: DataPanelState = "empty"
        self._summary: DataSummary | None = None
        self._busy = False
        self._spinner_index = 0
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        header = QHBoxLayout()
        self.title_label = QLabel("数据")
        self.title_label.setObjectName("contextTitle")
        header.addWidget(self.title_label)
        header.addStretch(1)
        self.region_button = QPushButton("地区：待选择")
        self.region_button.setObjectName("quietButton")
        self.region_button.setToolTip("选择本次会话使用的地区；同一地区已有数据时直接复用，更新请点“取最新的”")
        self.region_menu = QMenu(self.region_button)
        self.region_button.setMenu(self.region_menu)
        header.addWidget(self.region_button)
        self.status_label = QLabel("")
        self.status_label.setObjectName("dataStatus")
        header.addWidget(self.status_label)
        layout.addLayout(header)

        self.empty_view = self._build_empty_view()
        self.exploring_view = self._build_exploring_view()
        self.ready_view = self._build_ready_view()
        self.unavailable_view = self._build_unavailable_view()
        for view in (self.empty_view, self.exploring_view, self.ready_view, self.unavailable_view):
            layout.addWidget(view)
        layout.addStretch(1)

        self._spinner = QTimer(self)
        self._spinner.setInterval(220)
        self._spinner.timeout.connect(self._advance_spinner)
        self.set_state("empty", None)

    def _build_empty_view(self) -> QWidget:
        view = QWidget(self)
        layout = QVBoxLayout(view)
        layout.setContentsMargins(2, 4, 2, 0)
        self.empty_label = QLabel(self.EMPTY_TEXT)
        self.empty_label.setObjectName("dataSub")
        self.empty_label.setWordWrap(True)
        layout.addWidget(self.empty_label)
        return view

    def _build_exploring_view(self) -> QWidget:
        view = QWidget(self)
        layout = QVBoxLayout(view)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        heading = QLabel(self.EXPLORING_TEXT)
        heading.setObjectName("dataSub")
        heading.setWordWrap(True)
        layout.addWidget(heading)
        self.progress = QProgressBar()
        self.progress.setObjectName("dataProgress")
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(4)
        layout.addWidget(self.progress)
        self.exploring_card = DataCard()
        layout.addWidget(self.exploring_card)
        note = QLabel(self.EXPLORING_NOTE)
        note.setObjectName("dataSub")
        note.setWordWrap(True)
        layout.addWidget(note)
        return view

    def _build_ready_view(self) -> QWidget:
        view = QWidget(self)
        layout = QVBoxLayout(view)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.ready_card = DataCard()
        layout.addWidget(self.ready_card)
        self.variables_button = QPushButton("查看全部影响因素", view)
        self.variables_button.setObjectName("dataQuietLink")
        self.variables_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.variables_button.clicked.connect(self._show_variables_dialog)
        self.variables_button.setVisible(False)
        layout.addWidget(self.variables_button, 0, Qt.AlignmentFlag.AlignLeft)
        bottom = QHBoxLayout()
        bottom.setSpacing(6)
        self.fetched_label = QLabel("")
        self.fetched_label.setObjectName("dataKey")
        self.fetched_label.setWordWrap(True)
        bottom.addWidget(self.fetched_label, 1)
        self.refetch_button = QPushButton("取最新的")
        self.refetch_button.setObjectName("quietButton")
        self.refetch_button.setToolTip("按同一口径重新取一次数据")
        self.refetch_button.clicked.connect(self.refetch_requested)
        bottom.addWidget(self.refetch_button)
        layout.addLayout(bottom)
        links = QHBoxLayout()
        links.setSpacing(4)
        self.reselect_button = QPushButton("换一批数据")
        self.reselect_button.setObjectName("dataQuietLink")
        self.reselect_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.reselect_button.clicked.connect(self.reselect_requested)
        links.addWidget(self.reselect_button)
        separator = QLabel("·")
        separator.setObjectName("dataSub")
        links.addWidget(separator)
        self.details_button = QPushButton("查看取数细节")
        self.details_button.setObjectName("dataQuietLink")
        self.details_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.details_button.clicked.connect(self.details_requested)
        links.addWidget(self.details_button)
        links.addStretch(1)
        layout.addLayout(links)
        return view

    def _build_unavailable_view(self) -> QWidget:
        view = QWidget(self)
        layout = QVBoxLayout(view)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        title = QLabel(self.UNAVAILABLE_TITLE)
        title.setObjectName("dataValue")
        title.setWordWrap(True)
        layout.addWidget(title)
        body = QLabel(self.UNAVAILABLE_BODY)
        body.setObjectName("dataSub")
        body.setWordWrap(True)
        layout.addWidget(body)
        actions = QHBoxLayout()
        actions.setSpacing(6)
        self.retry_button = QPushButton("再试一次")
        self.retry_button.setObjectName("primaryButton")
        self.retry_button.clicked.connect(self.retry_requested)
        actions.addWidget(self.retry_button)
        self.local_file_button = QPushButton("用本地文件")
        self.local_file_button.setObjectName("quietButton")
        self.local_file_button.clicked.connect(self.local_file_requested)
        actions.addWidget(self.local_file_button)
        actions.addStretch(1)
        layout.addLayout(actions)
        # Parented on construction: setVisible on a parentless widget flashes it as a window.
        self.history_container = QWidget(view)
        history_layout = QVBoxLayout(self.history_container)
        history_layout.setContentsMargins(0, 0, 0, 0)
        history_layout.setSpacing(4)
        history_title = QLabel(self.UNAVAILABLE_HISTORY_TITLE)
        history_title.setObjectName("dataKey")
        history_layout.addWidget(history_title)
        self.history_card = DataCard()
        history_layout.addWidget(self.history_card)
        history_note = QLabel(self.UNAVAILABLE_NOTE)
        history_note.setObjectName("dataSub")
        history_note.setWordWrap(True)
        history_layout.addWidget(history_note)
        layout.addWidget(self.history_container)
        return view

    @property
    def state(self) -> DataPanelState:
        return self._state

    @property
    def summary(self) -> DataSummary | None:
        return self._summary

    def set_state(self, state: DataPanelState, summary: DataSummary | None) -> None:
        self._state = state
        self._summary = summary
        self.empty_view.setVisible(state == "empty")
        self.exploring_view.setVisible(state == "exploring")
        self.ready_view.setVisible(state == "ready")
        self.unavailable_view.setVisible(state == "unavailable")
        if state == "exploring":
            self.exploring_card.apply(summary, exploring=True, max_variables=4)
            self._spinner_index = 0
            self._spinner.start()
        else:
            self._spinner.stop()
        if state == "ready":
            self.ready_card.apply(summary, summarize_variables=True)
            self._apply_variables_menu(summary)
            self.fetched_label.setText(f"数据取自{summary.fetched_at_text}" if summary else "")
        if state == "unavailable":
            # A first-ever failure has no earlier dataset to fall back on.
            has_history = summary is not None and bool(summary.price_label)
            self.history_container.setVisible(has_history)
            if has_history:
                self.history_card.apply(summary)
        self._refresh_status()
        self._apply_busy()

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        self._apply_busy()

    def set_session(self, session: ResearchSession) -> None:
        self.set_state(session.data_state, session.data_summary)

    def set_regions(self, regions: list[tuple[str, str]], selected_region_id: str) -> None:
        """Populate the one session-scoped region control from the safe catalog."""

        self.region_menu.clear()
        selected_label = "待选择"
        for region_id, label in regions:
            action = self.region_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(region_id == selected_region_id)
            action.triggered.connect(
                lambda _checked=False, value=region_id: self.region_selected.emit(value)
            )
            if region_id == selected_region_id:
                selected_label = label
        self.region_button.setText(f"地区：{selected_label} ▾")
        self.region_button.setEnabled(bool(regions) and not self._busy)

    def open_region_menu(self) -> None:
        if self.region_button.isEnabled():
            self.region_button.showMenu()

    def _apply_busy(self) -> None:
        for button in (
            self.region_button,
            self.refetch_button,
            self.reselect_button,
            self.details_button,
            self.variables_button,
            self.retry_button,
            self.local_file_button,
        ):
            button.setEnabled(not self._busy)

    def _apply_variables_menu(self, summary: DataSummary | None) -> None:
        """Expose long factor lists without making the context panel excessively tall."""

        variables = summary.variables if summary else []
        self.variables_button.setVisible(bool(variables))
        if not variables:
            return
        self.variables_button.setText(f"查看全部 {len(variables)} 项影响因素")

    def _show_variables_dialog(self) -> None:
        FactorListDialog(self._summary.variables if self._summary else [], self).exec()

    def _advance_spinner(self) -> None:
        self._spinner_index = (self._spinner_index + 1) % len(self.SPINNER_FRAMES)
        self._refresh_status()

    def _refresh_status(self) -> None:
        marks = {"ready": "✓", "exploring": self.SPINNER_FRAMES[self._spinner_index], "unavailable": "⚠"}
        text = self.STATUS_TEXT.get(self._state, "")
        self.status_label.setText(f"{marks[self._state]} {text}" if text else "")
        self.status_label.setProperty("dataState", self._state)
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)


class TracePanel(QFrame):
    """Filterable, independently scrollable observable Agent run trace."""

    maximize_requested = Signal(bool)

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("tracePanel")
        self._events: list[TraceEvent] = []
        self._auto_follow = True
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
        previous_function: str | None = None
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
            # Execution completion and result validation intentionally narrate
            # to the same completed-function title. Collapse only that lifecycle
            # pair. Other repeated titles are separate audit events and must stay
            # visible, however similar they read.
            duplicate_function_completion = bool(
                step.function_name
                and step.function_name == previous_function
                and step.title == previous_title
                and step.title.startswith("已完成 · ")
            )
            if duplicate_function_completion:
                continue
            previous_title = step.title
            previous_function = step.function_name
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
        self.maximize_button.setText("还原" if maximized else "放大")
        self.maximize_requested.emit(maximized)


class ContextPane(QSplitter):
    """The dataset in use and a maximizable, scrollable Agent trace."""

    refetch_requested = Signal()
    reselect_requested = Signal()
    details_requested = Signal()
    retry_requested = Signal()
    local_file_requested = Signal()
    region_selected = Signal(str)
    trace_maximized = Signal(bool)

    PANEL_SIZES: ClassVar[list[int]] = [262, 538]

    def __init__(self) -> None:
        super().__init__(Qt.Orientation.Vertical)
        self.setObjectName("contextPane")
        self.data_panel = DataPanel()
        self.trace = TracePanel()
        self.data_panel.refetch_requested.connect(self.refetch_requested)
        self.data_panel.reselect_requested.connect(self.reselect_requested)
        self.data_panel.details_requested.connect(self.details_requested)
        self.data_panel.retry_requested.connect(self.retry_requested)
        self.data_panel.local_file_requested.connect(self.local_file_requested)
        self.data_panel.region_selected.connect(self.region_selected)
        self.trace.maximize_requested.connect(self._set_trace_maximized)
        self.addWidget(self.data_panel)
        self.addWidget(self.trace)
        self.setSizes(self.PANEL_SIZES)
        self.setStretchFactor(0, 0)
        self.setStretchFactor(1, 1)

    def set_session(self, session: ResearchSession) -> None:
        self.data_panel.set_session(session)
        self.trace.set_events(session.trace)

    def set_busy(self, busy: bool) -> None:
        self.data_panel.set_busy(busy)

    def _set_trace_maximized(self, maximized: bool) -> None:
        self.data_panel.setVisible(not maximized)
        if maximized:
            self.setSizes([0, max(self.height(), 1)])
        else:
            self.setSizes(self.PANEL_SIZES)
        self.trace_maximized.emit(maximized)
