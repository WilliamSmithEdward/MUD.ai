"""Memory panel: editable view of the permanent MemoryStore."""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..llm.memory import MemoryStore
from .widgets import mono_font


class MemoryPanel(QWidget):
    """List view + add/edit/delete buttons over a MemoryStore."""

    entry_added = pyqtSignal(str)

    def __init__(self, store: MemoryStore) -> None:
        super().__init__()
        self.store = store
        self.list = QListWidget()
        self.list.setFont(mono_font())
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list.itemDoubleClicked.connect(self._edit_selected)
        self.list.itemChanged.connect(self._on_item_changed)
        self.list.installEventFilter(self)
        self.list.setStyleSheet(
            "QListWidget { background-color: #11141a; color: #d8d8e0; }"
        )

        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self._on_add)
        edit_btn = QPushButton("Edit")
        edit_btn.clicked.connect(self._edit_selected)
        del_btn = QPushButton("Delete")
        del_btn.clicked.connect(self._delete_selected)
        clear_btn = QPushButton("Clear all")
        clear_btn.clicked.connect(self._on_clear_all)

        button_row = QHBoxLayout()
        button_row.addWidget(add_btn)
        button_row.addWidget(edit_btn)
        button_row.addWidget(del_btn)
        button_row.addStretch(1)
        button_row.addWidget(clear_btn)

        header = QLabel(
            "Permanent memory: durable facts injected into every prompt.\n"
            "Toggle the checkbox to disable an entry without deleting it."
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(header)
        layout.addWidget(self.list, 1)
        layout.addLayout(button_row)

        store.subscribe(self.refresh)
        self.refresh()

    # ----- rendering ---------------------------------------------------------
    def refresh(self) -> None:
        self.list.blockSignals(True)
        self.list.clear()
        for e in self.store.entries:
            item = QListWidgetItem(self._format_entry(e))
            item.setData(Qt.ItemDataRole.UserRole, e.id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if e.enabled else Qt.CheckState.Unchecked
            )
            self.list.addItem(item)
        self.list.blockSignals(False)

    @staticmethod
    def _format_entry(e: object) -> str:
        text = getattr(e, "text", "")
        source = getattr(e, "source", "")
        ts = getattr(e, "ts", "")[:19].replace("T", " ")
        prefix = {
            "operator": "OP ",
            "agent_decision": "AGD",
            "agent_chat": "AGC",
        }.get(source, "???")
        return f"[{prefix} {ts}] {text}"

    # ----- mutations ---------------------------------------------------------
    def _on_add(self) -> None:
        text, ok = QInputDialog.getMultiLineText(
            self, "Add memory entry",
            "Durable fact to remember on every decision:", "",
        )
        if ok and text.strip():
            self.store.add(text.strip(), source="operator")

    def _edit_selected(self) -> None:
        item = self.list.currentItem()
        if item is None:
            return
        entry_id = item.data(Qt.ItemDataRole.UserRole)
        current = next(
            (e.text for e in self.store.entries if e.id == entry_id), ""
        )
        text, ok = QInputDialog.getMultiLineText(
            self, "Edit memory entry", "Text:", current,
        )
        if ok:
            self.store.update(entry_id, text=text)

    def _delete_selected(self) -> None:
        item = self.list.currentItem()
        if item is None:
            return
        entry_id = item.data(Qt.ItemDataRole.UserRole)
        self.store.delete(entry_id)

    def _on_clear_all(self) -> None:
        if not self.store.entries:
            return
        reply = QMessageBox.question(
            self, "Clear memory",
            f"Delete all {len(self.store.entries)} memory entries? "
            "The file on disk will be cleared.",
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.store.clear()

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        entry_id = item.data(Qt.ItemDataRole.UserRole)
        enabled = item.checkState() == Qt.CheckState.Checked
        self.store.update(entry_id, enabled=enabled)

    # ----- key handling ------------------------------------------------------
    def eventFilter(self, obj: object, ev: object) -> bool:  # type: ignore[override]
        if obj is self.list and isinstance(ev, QKeyEvent) and ev.type() == QKeyEvent.Type.KeyPress:
            if ev.key() == Qt.Key.Key_Delete:
                self._delete_selected()
                return True
        return super().eventFilter(obj, ev)  # type: ignore[arg-type]
