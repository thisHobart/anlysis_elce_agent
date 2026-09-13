"""Desktop P2 review window; users edit typed fields, never raw review JSON."""

from __future__ import annotations

import json
from collections.abc import Callable
from html import escape
from typing import Any
from uuid import uuid4

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app.research.full_flow.contracts import P2ReviewDecision, P2ReviewItem, P2ReviewSummary
from app.research.news.model_extraction import ModelNewsExtraction

_EVIDENCE_FIELDS = {
    "relevance",
    "event_type",
    "status",
    "physical_effect",
    "affected_regions",
    "affected_assets",
    "asset_groups",
    "magnitude",
    "effective_start_at",
    "effective_end_at",
}
_EFFECTS = (
    "supply_up",
    "supply_down",
    "demand_up",
    "demand_down",
    "transfer_up",
    "transfer_down",
    "mixed",
    "unknown",
)
_STATUSES = ("occurred", "restored", "planned", "forecast", "corrected", "cancelled", "unknown")
_PRECISIONS = ("instant", "hour", "day", "month", "range", "vague", "unknown")
_BASES = ("stated_absolute", "stated_components", "derived_from_publication")
_QUANTITY_UNITS = ("MW", "GW", "kW", "万千瓦", "亿千瓦")
_QUANTITY_SEMANTICS = (
    "capacity_level",
    "capacity_change",
    "generation_loss",
    "generation_restore",
    "demand_level",
    "demand_change",
    "output_level",
    "supply_change",
    "transfer_change",
    "unknown",
)
_QUANTITY_DIRECTIONS = ("increase", "decrease", "mixed", "unknown")


def _comma_values(text: str) -> list[str]:
    return [part.strip() for part in text.replace("，", ",").split(",") if part.strip()]


def _result_as_model_candidate(result: dict[str, Any]) -> dict[str, Any] | None:
    """Convert an already validated domain candidate back to the human-review input schema."""

    converted: list[dict[str, Any]] = []
    for event in result.get("events") or ():
        if not isinstance(event, dict) or event.get("event_type") in {"unknown", "irrelevant"}:
            continue
        evidence = [
            {
                "field_name": span.get("field_name"),
                "text_field": span.get("text_field"),
                "quote": span.get("quote"),
                "occurrence_index": 0,
            }
            for span in event.get("evidence") or ()
            if isinstance(span, dict)
            and span.get("field_name") in _EVIDENCE_FIELDS
            and span.get("text_field") in {"title", "body"}
            and span.get("quote")
        ]
        if not evidence:
            continue
        resolution = event.get("entity_resolution") or {}
        asset_groups = [
            str(item.get("value"))
            for item in resolution.get("asset_groups") or ()
            if isinstance(item, dict) and item.get("value")
        ]
        magnitude = event.get("magnitude")
        quantity = None
        if isinstance(magnitude, dict):
            quantity = {
                "value": magnitude.get("raw_value"),
                "unit": magnitude.get("raw_unit"),
                "semantic": magnitude.get("semantic"),
                "direction": magnitude.get("direction", "unknown"),
                "raw_text": magnitude.get("raw_text"),
            }
        resolution_time = event.get("time_resolution") or {}
        precision = str(resolution_time.get("precision") or "unknown")
        basis = str(resolution_time.get("basis") or "stated_absolute")
        instant = None
        end_instant = None
        if event.get("effective_start_at") and precision in {"instant", "hour"}:
            instant = {"iso": event["effective_start_at"], "basis": basis}
        if event.get("effective_end_at") and instant is not None:
            end_instant = {"iso": event["effective_end_at"], "basis": basis}
        converted.append(
            {
                "relevance": event.get("relevance", "short_term"),
                "event_type": event["event_type"],
                "status": event.get("status", "unknown"),
                "physical_effect": event.get("physical_effect", "unknown"),
                "affected_regions": list(event.get("affected_regions") or ()),
                "affected_assets": list(event.get("affected_assets") or ()),
                "asset_groups": asset_groups,
                "quantity": quantity,
                "time_precision": precision,
                "time_text": resolution_time.get("stated_text"),
                "event_instant": instant,
                "event_end_instant": end_instant,
                "confidence": event.get("confidence") if event.get("confidence") is not None else 1.0,
                "evidence": evidence,
                "cleared_fields": list(event.get("cleared_fields") or ()),
            }
        )
    if not converted:
        return None
    return ModelNewsExtraction.model_validate(
        {"schema_version": "1.4.0", "disposition": "event", "events": converted}
    ).model_dump(mode="json")


class P2ReviewDialog(QDialog):
    """Review blocking and optional extraction records against their source evidence."""

    summary_changed = Signal(object)
    reviewer_changed = Signal(str)

    def __init__(
        self,
        summary: P2ReviewSummary,
        *,
        submit: Callable[[P2ReviewDecision], P2ReviewSummary],
        reviewer: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("P2 新闻抽取复核")
        self.resize(1040, 720)
        self.summary = summary
        self._submit = submit
        self._current_candidates: list[dict[str, Any]] = []
        self._working: dict[str, Any] | None = None

        root = QVBoxLayout(self)
        self.summary_label = QLabel()
        root.addWidget(self.summary_label)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.items = QListWidget()
        self.items.setMinimumWidth(280)
        self.items.currentItemChanged.connect(self._item_changed)
        splitter.addWidget(self.items)

        detail = QWidget()
        detail_layout = QVBoxLayout(detail)
        self.source = QLabel()
        self.source.setWordWrap(True)
        self.source.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        detail_layout.addWidget(self.source)
        self.body = QPlainTextEdit()
        self.body.setReadOnly(True)
        self.body.setMaximumHeight(150)
        detail_layout.addWidget(self.body)

        selection = QFormLayout()
        self.disposition = QComboBox()
        self.disposition.addItem("接受并用于分析", "accept")
        self.disposition.addItem("修正后用于分析", "correct")
        self.disposition.addItem("拒绝并排除", "reject")
        self.disposition.addItem("接受并仅作背景", "background")
        self.disposition.currentIndexChanged.connect(self._update_editor_state)
        selection.addRow("处置", self.disposition)
        self.candidate = QComboBox()
        self.candidate.currentIndexChanged.connect(self._candidate_changed)
        selection.addRow("抽取候选", self.candidate)
        self.event_index = QComboBox()
        self.event_index.currentIndexChanged.connect(self._event_changed)
        selection.addRow("候选中的事件", self.event_index)
        self.reviewer = QLineEdit(reviewer)
        selection.addRow("复核人", self.reviewer)
        self.reason = QPlainTextEdit()
        self.reason.setMaximumHeight(72)
        selection.addRow("处置原因", self.reason)
        detail_layout.addLayout(selection)

        editor = QFormLayout()
        self.physical_effect = QComboBox()
        self.physical_effect.addItems(_EFFECTS)
        editor.addRow("物理影响", self.physical_effect)
        self.event_status = QComboBox()
        self.event_status.addItems(_STATUSES)
        editor.addRow("事件状态", self.event_status)
        self.regions = QLineEdit()
        editor.addRow("影响地区（逗号分隔）", self.regions)
        self.assets = QLineEdit()
        editor.addRow("影响资产（逗号分隔）", self.assets)
        self.quantity_value = QLineEdit()
        editor.addRow("数量", self.quantity_value)
        self.quantity_unit = QComboBox()
        self.quantity_unit.addItems(_QUANTITY_UNITS)
        editor.addRow("数量单位", self.quantity_unit)
        self.quantity_semantic = QComboBox()
        self.quantity_semantic.addItems(_QUANTITY_SEMANTICS)
        editor.addRow("数量含义", self.quantity_semantic)
        self.quantity_direction = QComboBox()
        self.quantity_direction.addItems(_QUANTITY_DIRECTIONS)
        editor.addRow("数量方向", self.quantity_direction)
        self.quantity_raw_text = QLineEdit()
        editor.addRow("数量证据原文", self.quantity_raw_text)
        self.time_precision = QComboBox()
        self.time_precision.addItems(_PRECISIONS)
        editor.addRow("时间精度", self.time_precision)
        self.time_text = QLineEdit()
        editor.addRow("原文时间短语", self.time_text)
        self.event_instant = QLineEdit()
        self.event_instant.setPlaceholderText("仅 instant/hour：2026-08-03T20:00:00+08:00")
        editor.addRow("事件时间 ISO-8601", self.event_instant)
        self.time_basis = QComboBox()
        self.time_basis.addItems(_BASES)
        editor.addRow("时间依据", self.time_basis)
        self.time_evidence_field = QComboBox()
        self.time_evidence_field.addItems(("body", "title"))
        editor.addRow("时间证据位置", self.time_evidence_field)
        self.time_evidence_quote = QLineEdit()
        editor.addRow("时间证据原文", self.time_evidence_quote)
        detail_layout.addLayout(editor)
        self.blocking = QLabel()
        self.blocking.setWordWrap(True)
        detail_layout.addWidget(self.blocking)
        detail_layout.addStretch(1)
        splitter.addWidget(detail)
        splitter.setSizes([310, 700])
        root.addWidget(splitter, 1)

        actions = QHBoxLayout()
        actions.addStretch(1)
        save = QPushButton("提交本条复核")
        save.setObjectName("primaryButton")
        save.clicked.connect(self._save)
        actions.addWidget(save)
        close = QPushButton("关闭")
        close.clicked.connect(self.accept)
        actions.addWidget(close)
        root.addLayout(actions)
        self._populate()
        self._update_editor_state()

    def _populate(self, preferred_key: str | None = None) -> None:
        self.summary_label.setText(
            f"必审未处理 {self.summary.required_pending} 条；已处理 {self.summary.required_resolved} 条；"
            f"可选抽查 {self.summary.optional_unreviewed} 条。保存单条不会自动推进流程。"
        )
        self.items.clear()
        preferred_row = 0
        for index, item in enumerate(self.summary.items):
            marker = "必审" if item.required else "抽查"
            status = {
                "unreviewed": "未处理",
                "accepted": "已接受",
                "corrected": "已修正",
                "rejected": "已拒绝",
            }[item.review_status]
            row = QListWidgetItem(f"[{marker}/{status}] {item.title}")
            row.setData(Qt.ItemDataRole.UserRole, item.cache_key)
            self.items.addItem(row)
            if item.cache_key == preferred_key:
                preferred_row = index
        if self.items.count():
            self.items.setCurrentRow(preferred_row)

    def _selected_item(self) -> P2ReviewItem | None:
        row = self.items.currentItem()
        key = str(row.data(Qt.ItemDataRole.UserRole)) if row else ""
        return next((item for item in self.summary.items if item.cache_key == key), None)

    def _item_changed(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        item = self._selected_item()
        if item is None:
            return
        self.source.setText(
            f"<b>{escape(item.title)}</b><br>{escape(item.source_name)} · {escape(item.source_ref)}"
        )
        self.body.setPlainText(item.body)
        self.blocking.setText(
            "阻断原因：" + ("；".join(item.blocking_reasons) if item.blocking_reasons else "无；本条为可选抽查")
        )
        self.disposition.model().item(3).setEnabled(item.allows_background_only)
        if self.disposition.currentData() == "background" and not item.allows_background_only:
            self.disposition.setCurrentIndex(0)
        candidates: list[tuple[str, dict[str, Any]]] = []
        original = _result_as_model_candidate(item.result)
        if original is not None:
            candidates.append(("当前抽取", original))
        for index, result in enumerate(item.result.get("pass_results") or (), 1):
            if isinstance(result, dict):
                candidate = _result_as_model_candidate(result)
                if candidate is not None:
                    candidates.append((f"独立抽取候选 {index}", candidate))
        self._current_candidates = [value for _label, value in candidates]
        self.candidate.clear()
        for label, _value in candidates:
            self.candidate.addItem(label)
        self._candidate_changed(0)

    def _candidate_changed(self, index: int) -> None:
        if index < 0 or index >= len(self._current_candidates):
            self._working = None
            self.event_index.clear()
            return
        self._working = json.loads(json.dumps(self._current_candidates[index], ensure_ascii=False))
        self.event_index.clear()
        for number, event in enumerate(self._working.get("events") or (), 1):
            self.event_index.addItem(f"事件 {number}：{event.get('event_type', '未知')}", number - 1)
        if self.event_index.count():
            self.event_index.setCurrentIndex(0)
            self._event_changed(0)

    @staticmethod
    def _set_combo(combo: QComboBox, value: str) -> None:
        index = combo.findText(value)
        combo.setCurrentIndex(max(0, index))

    def _event_changed(self, _index: int) -> None:
        event = self._current_event()
        if event is None:
            return
        self._set_combo(self.physical_effect, str(event.get("physical_effect") or "unknown"))
        self._set_combo(self.event_status, str(event.get("status") or "unknown"))
        self.regions.setText("，".join(str(value) for value in event.get("affected_regions") or ()))
        self.assets.setText("，".join(str(value) for value in event.get("affected_assets") or ()))
        quantity = event.get("quantity") or {}
        self.quantity_value.setText(str(quantity.get("value") or ""))
        self._set_combo(self.quantity_unit, str(quantity.get("unit") or "MW"))
        self._set_combo(self.quantity_semantic, str(quantity.get("semantic") or "unknown"))
        self._set_combo(self.quantity_direction, str(quantity.get("direction") or "unknown"))
        self.quantity_raw_text.setText(str(quantity.get("raw_text") or ""))
        self._set_combo(self.time_precision, str(event.get("time_precision") or "unknown"))
        self.time_text.setText(str(event.get("time_text") or ""))
        instant = event.get("event_instant") or {}
        self.event_instant.setText(str(instant.get("iso") or ""))
        self._set_combo(self.time_basis, str(instant.get("basis") or "stated_absolute"))
        evidence = next(
            (claim for claim in event.get("evidence") or () if claim.get("field_name") == "effective_start_at"),
            {},
        )
        self._set_combo(self.time_evidence_field, str(evidence.get("text_field") or "body"))
        self.time_evidence_quote.setText(str(evidence.get("quote") or ""))

    def _current_event(self) -> dict[str, Any] | None:
        if self._working is None:
            return None
        index = self.event_index.currentData()
        events = self._working.get("events") or ()
        return events[index] if isinstance(index, int) and 0 <= index < len(events) else None

    def _apply_editor(self) -> None:
        event = self._current_event()
        if event is None:
            raise ValueError("当前抽取没有可修正的事件候选")
        event["physical_effect"] = self.physical_effect.currentText()
        event["status"] = self.event_status.currentText()
        event["affected_regions"] = _comma_values(self.regions.text())
        event["affected_assets"] = _comma_values(self.assets.text())
        quantity_value = self.quantity_value.text().strip()
        quantity_text = self.quantity_raw_text.text().strip()
        if quantity_value or quantity_text:
            if not quantity_value or not quantity_text:
                raise ValueError("填写数量时，数量值和数量证据原文必须同时填写")
            event["quantity"] = {
                "value": float(quantity_value),
                "unit": self.quantity_unit.currentText(),
                "semantic": self.quantity_semantic.currentText(),
                "direction": self.quantity_direction.currentText(),
                "raw_text": quantity_text,
            }
        else:
            event["quantity"] = None
        precision = self.time_precision.currentText()
        event["time_precision"] = precision
        event["time_text"] = self.time_text.text().strip() or None
        instant = self.event_instant.text().strip()
        event["event_instant"] = (
            {"iso": instant, "basis": self.time_basis.currentText()} if instant else None
        )
        if instant and precision not in {"instant", "hour"}:
            raise ValueError("填写精确事件时间时，时间精度必须是 instant 或 hour")
        quote = self.time_evidence_quote.text().strip()
        evidence = [
            claim for claim in event.get("evidence") or () if claim.get("field_name") != "effective_start_at"
        ]
        if quote:
            evidence.append(
                {
                    "field_name": "effective_start_at",
                    "text_field": self.time_evidence_field.currentText(),
                    "quote": quote,
                    "occurrence_index": 0,
                }
            )
        event["evidence"] = evidence

    def _update_editor_state(self) -> None:
        correcting = self.disposition.currentData() == "correct"
        for widget in (
            self.candidate,
            self.event_index,
            self.physical_effect,
            self.event_status,
            self.regions,
            self.assets,
            self.quantity_value,
            self.quantity_unit,
            self.quantity_semantic,
            self.quantity_direction,
            self.quantity_raw_text,
            self.time_precision,
            self.time_text,
            self.event_instant,
            self.time_basis,
            self.time_evidence_field,
            self.time_evidence_quote,
        ):
            widget.setEnabled(correcting)

    def _save(self) -> None:
        item = self._selected_item()
        if item is None:
            return
        reviewer = self.reviewer.text().strip()
        reason = self.reason.toPlainText().strip()
        if not reviewer or not reason:
            QMessageBox.warning(self, "复核未保存", "复核人和处置原因都必须填写。")
            return
        action = str(self.disposition.currentData())
        decision = "accepted"
        use = "analysis"
        corrected = None
        if action == "correct":
            decision = "corrected"
            try:
                self._apply_editor()
                corrected = ModelNewsExtraction.model_validate(self._working).model_dump(mode="json")
            except (TypeError, ValueError) as exc:
                QMessageBox.warning(self, "修正内容无效", str(exc))
                return
        elif action == "reject":
            decision = "rejected"
        elif action == "background":
            use = "background_only"
        try:
            self.summary = self._submit(
                P2ReviewDecision(
                    flow_id=self.summary.flow_id,
                    cache_key=item.cache_key,
                    expected_revision=item.revision,
                    decision=decision,
                    use=use,
                    reviewer=reviewer,
                    reason=reason,
                    request_id=uuid4().hex,
                    corrected=corrected,
                )
            )
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "复核未保存", str(exc))
            return
        self.reviewer_changed.emit(reviewer)
        self.summary_changed.emit(self.summary)
        self.reason.clear()
        self._populate(preferred_key=item.cache_key)
        QMessageBox.information(self, "复核已保存", "本条决定已写入复核数据库。")
