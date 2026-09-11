"""Qt style sheet for the electricity-price research workspace."""

APP_STYLE = """
QWidget {
    color: #1F2430;
    font-size: 13px;
}
QMainWindow, QFrame#conversationPane, QScrollArea#conversationScroll {
    background: #FFFFFF;
}
QFrame#historyPane, QSplitter#contextPane, QFrame#dataPanel,
QFrame#tracePanel {
    background: #F6F7F9;
}
QFrame#historyPane {
    border-right: 1px solid #E7EAEF;
}
QSplitter#researchWorkspace::handle, QSplitter#contextPane::handle {
    background: #E7EAEF;
    width: 1px;
    height: 1px;
}
QFrame#conversationHeader {
    background: #FFFFFF;
    border-bottom: 1px solid #E7EAEF;
}
QLabel#sessionTitle {
    font-size: 15px;
    font-weight: 600;
}
QLabel#dataKey, QLabel#resultMeta,
QLabel#toolDetail, QLabel#planHint, QLabel#stepDetail {
    color: #667085;
    font-size: 12px;
}
QLabel#planHint {
    color: #7A8194;
}
QLabel#sessionStatus, QLabel#planStatus {
    background: #EEF0FB;
    color: #3F51B5;
    border-radius: 4px;
    padding: 3px 8px;
}
QLabel#contextTitle, QLabel#planTitle, QLabel#resultTitle, QLabel#thinkingTitle {
    font-weight: 600;
    font-size: 13.5px;
}
QLabel#planStage {
    color: #3F51B5;
    font-size: 12px;
    font-weight: 600;
    margin-top: 4px;
}
QLabel#planStep {
    color: #3C4254;
    font-size: 12.5px;
    padding-left: 4px;
}
QLabel#resultSummary {
    color: #3C4254;
}
QPushButton {
    min-height: 28px;
    padding: 2px 12px;
    border: 1px solid #DADDE2;
    border-radius: 6px;
    background: #FFFFFF;
}
QPushButton:hover { background: #F4F5F8; }
QPushButton:disabled { color: #9CA3AF; background: #F4F4F5; }
QPushButton#primaryButton, QPushButton#newResearchButton {
    background: #3F51B5;
    color: white;
    border-color: #3F51B5;
    font-weight: 500;
}
QPushButton#primaryButton:hover, QPushButton#newResearchButton:hover { background: #35459B; }
QPushButton#primaryButton:disabled, QPushButton#newResearchButton:disabled {
    background: #D1D5DB;
    color: #6B7280;
    border-color: #D1D5DB;
}
QPushButton#quietButton { background: transparent; }
QPushButton#thinkingToggle {
    background: transparent;
    border: none;
    color: #667085;
    min-height: 18px;
    padding: 0;
}
QLineEdit, QPlainTextEdit, QComboBox, QSpinBox, QListWidget, QTreeWidget {
    background: #FFFFFF;
    border: 1px solid #E7EAEF;
    border-radius: 6px;
    selection-background-color: #EEF0FB;
    selection-color: #1F2430;
}
QLineEdit, QComboBox, QSpinBox { min-height: 27px; padding: 1px 6px; }
QListWidget#sessionList {
    background: transparent;
    border: none;
    outline: none;
}
QListWidget#sessionList::item { padding: 7px 8px; border-radius: 6px; }
QListWidget#sessionList::item:selected { background: #EEF0FB; color: #3F51B5; }
QFrame#composer {
    background: #FFFFFF;
    border-top: 1px solid #E7EAEF;
}
QPlainTextEdit#composerInput {
    padding: 8px;
    border-radius: 10px;
}
QFrame#userMessage {
    background: #F2F3F5;
    border-radius: 10px;
}
QFrame#agentMessage { background: transparent; }
QLabel#messageRole { color: #3F51B5; font-weight: 600; font-size: 12px; }
QLabel#messageContent { font-size: 14px; }
QFrame#noticeMessage, QFrame#errorNotice {
    background: #F8F9FB;
    border: 1px solid #E7EAEF;
    border-radius: 8px;
}
QFrame#errorNotice { border-color: #F3B5AE; color: #B42318; }
QFrame#toolMessage {
    background: #F8F9FB;
    border: 1px solid #E7EAEF;
    border-radius: 8px;
}
QFrame#thinkingMessage {
    background: #FAFBFC;
    border: 1px solid #E7EAEF;
    border-radius: 10px;
}
QLabel#thinkingStatus {
    color: #7A8194;
    font-size: 12px;
}
QLabel#stepStage {
    color: #667085;
    background: #EDEFF3;
    border-radius: 3px;
    padding: 0px 5px;
    font-size: 11px;
}
QLabel#stepTitle { font-size: 12.5px; color: #2A3040; }
QLabel#stepMark { font-size: 12px; color: #98A2B3; }
QLabel#toolTitle { font-weight: 500; }
QLabel[traceStatus="running"] { color: #3F51B5; }
QLabel[traceStatus="completed"] { color: #0F766E; }
QLabel[traceStatus="warning"] { color: #B45309; }
QLabel[traceStatus="failed"] { color: #BE123C; }
QLabel[traceStatus="stopped"] { color: #667085; }
QFrame#planMessage, QFrame#resultMessage {
    background: #FFFFFF;
    border: 1px solid #E7EAEF;
    border-radius: 10px;
}
QLabel#resultWarning { color: #B45309; }
QFrame#dataCard {
    background: #FFFFFF;
    border: 1px solid #E7EAEF;
    border-radius: 6px;
}
QFrame#dataDivider { background: #EEF0F3; border: none; }
QLabel#dataValue { font-size: 13px; line-height: 1.55; }
QLabel#dataSub, QLabel#dataMark { color: #98A2B3; font-size: 12px; }
QLabel#dataStatus { font-size: 12px; }
QLabel#dataStatus[dataState="ready"] { color: #0F766E; }
QLabel#dataStatus[dataState="exploring"] { color: #3F51B5; }
QLabel#dataStatus[dataState="unavailable"] { color: #BE123C; }
QLabel#dataMark[dataState="settled"] { color: #0F766E; }
QLabel#dataMark[dataState="waiting"] { color: #C9CDD4; }
QLabel#dataValue[dataState="pending"], QLabel#dataValue[dataState="waiting"] { color: #98A2B3; }
QPushButton#dataQuietLink {
    background: transparent;
    border: none;
    color: #98A2B3;
    font-size: 12px;
    min-height: 18px;
    padding: 0 2px;
}
QPushButton#dataQuietLink:hover { color: #667085; }
QPushButton#dataQuietLink:disabled { color: #C9CDD4; }
QProgressBar#dataProgress {
    background: #EEF0F3;
    border: none;
    border-radius: 2px;
}
QProgressBar#dataProgress::chunk { background: #3F51B5; border-radius: 2px; }
QPlainTextEdit#dataDetails { font-family: Consolas, "Courier New", monospace; font-size: 12px; }
QTreeWidget#traceTree {
    background: #FFFFFF;
    color: #2A3040;
    border: 1px solid #E7EAEF;
    border-radius: 8px;
    font-size: 12px;
    outline: none;
}
QTreeWidget#traceTree::item { padding: 3px 2px; }
QTreeWidget#traceTree::item:selected { background: #EEF0FB; color: #1F2430; }
QHeaderView::section {
    background: #F6F7F9;
    border: none;
    border-bottom: 1px solid #E7EAEF;
    padding: 5px;
    font-size: 11px;
    color: #667085;
}
QScrollBar:vertical { width: 9px; background: transparent; }
QScrollBar::handle:vertical { background: #C9CDD4; border-radius: 4px; min-height: 28px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
"""
