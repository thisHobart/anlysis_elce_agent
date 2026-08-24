"""Small Qt style sheet matching the approved neutral desktop design."""

APP_STYLE = """
QWidget {
    color: #202123;
    font-size: 13px;
}
QMainWindow, QFrame#conversationPane, QScrollArea#conversationScroll {
    background: #FFFFFF;
}
QFrame#historyPane, QSplitter#contextPane, QFrame#inputFilesPanel,
QFrame#planStatusPanel, QFrame#tracePanel {
    background: #F4F4F5;
}
QFrame#historyPane {
    border-right: 1px solid #E5E7EB;
}
QSplitter#researchWorkspace::handle, QSplitter#contextPane::handle {
    background: #E5E7EB;
    width: 1px;
    height: 1px;
}
QFrame#conversationHeader {
    background: #FFFFFF;
    border-bottom: 1px solid #E5E7EB;
}
QLabel#sessionTitle {
    font-size: 15px;
    font-weight: 600;
}
QLabel#sessionMeta, QLabel#contextHint, QLabel#fileRole, QLabel#resultMeta,
QLabel#toolDetail, QLabel#planObjective {
    color: #6B7280;
    font-size: 12px;
}
QLabel#sessionStatus, QLabel#planStatus {
    background: #ECEBFF;
    color: #4F46E5;
    border-radius: 4px;
    padding: 3px 7px;
}
QLabel#contextTitle, QLabel#planTitle, QLabel#resultTitle {
    font-weight: 600;
    font-size: 13px;
}
QPushButton {
    min-height: 28px;
    padding: 2px 10px;
    border: 1px solid #DADDE2;
    border-radius: 6px;
    background: #FFFFFF;
}
QPushButton:hover { background: #F7F7F8; }
QPushButton:disabled { color: #9CA3AF; background: #F4F4F5; }
QPushButton#primaryButton, QPushButton#newResearchButton {
    background: #4F46E5;
    color: white;
    border-color: #4F46E5;
    font-weight: 500;
}
QPushButton#primaryButton:hover, QPushButton#newResearchButton:hover { background: #4338CA; }
QPushButton#primaryButton:disabled, QPushButton#newResearchButton:disabled {
    background: #D1D5DB;
    color: #6B7280;
    border-color: #D1D5DB;
}
QPushButton#quietButton { background: transparent; }
QLineEdit, QPlainTextEdit, QComboBox, QSpinBox, QListWidget, QTreeWidget {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 6px;
    selection-background-color: #ECEBFF;
    selection-color: #202123;
}
QLineEdit, QComboBox, QSpinBox { min-height: 27px; padding: 1px 6px; }
QListWidget#sessionList {
    background: transparent;
    border: none;
    outline: none;
}
QListWidget#sessionList::item { padding: 7px 8px; border-radius: 6px; }
QListWidget#sessionList::item:selected { background: #ECEBFF; color: #4F46E5; }
QFrame#composer {
    background: #FFFFFF;
    border-top: 1px solid #E5E7EB;
}
QPlainTextEdit#composerInput {
    padding: 8px;
    border-radius: 10px;
}
QFrame#userMessage {
    background: #F2F2F3;
    border-radius: 10px;
}
QFrame#agentMessage { background: transparent; }
QLabel#messageRole { color: #4F46E5; font-weight: 600; font-size: 12px; }
QLabel#messageContent { font-size: 14px; }
QFrame#noticeMessage, QFrame#errorNotice {
    background: #F7F7F8;
    border: 1px solid #E5E7EB;
    border-radius: 6px;
}
QFrame#errorNotice { border-color: #F3B5AE; color: #B42318; }
QFrame#toolMessage {
    background: #F7F7F8;
    border: 1px solid #E5E7EB;
    border-radius: 6px;
}
QLabel#toolTitle { font-weight: 500; }
QLabel[traceStatus="running"] { color: #2563EB; }
QLabel[traceStatus="completed"] { color: #16803C; }
QLabel[traceStatus="warning"] { color: #B45309; }
QLabel[traceStatus="failed"] { color: #B42318; }
QFrame#planMessage, QFrame#resultMessage {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
}
QLabel#resultWarning { color: #B45309; }
QFrame#fileSlot {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 6px;
}
QLabel#fileName { font-family: Consolas; font-weight: 500; }
QLabel[inputStatus="ready"] { color: #16803C; }
QLabel[inputStatus="warning"] { color: #B45309; }
QLabel[inputStatus="failed"] { color: #B42318; }
QLabel[inputStatus="loading"], QLabel[inputStatus="selected"] { color: #2563EB; }
QTreeWidget#traceTree {
    background: #202123;
    color: #E5E7EB;
    border: none;
    font-family: Consolas;
    font-size: 11px;
}
QTreeWidget#traceTree::item:selected { background: #374151; color: white; }
QHeaderView::section {
    background: #F4F4F5;
    border: none;
    border-bottom: 1px solid #E5E7EB;
    padding: 4px;
    font-size: 11px;
}
QScrollBar:vertical { width: 9px; background: transparent; }
QScrollBar::handle:vertical { background: #C9CDD4; border-radius: 4px; min-height: 28px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
"""
