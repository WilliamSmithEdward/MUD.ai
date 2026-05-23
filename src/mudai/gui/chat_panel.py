"""Side-conversation chat panel.

The bottom is a text input + send button; the top is a streaming transcript of
the operator/agent conversation. The widget itself owns no LLM logic - it only
emits a signal when the user submits a message and exposes append methods that
the main window calls from worker callbacks.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QKeyEvent, QTextCursor
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from .widgets import mono_font


class _Input(QPlainTextEdit):
    """Multi-line input. Enter submits; Shift+Enter inserts newline."""

    submitted = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.setFont(mono_font())
        self.setPlaceholderText(
            "Talk to the agent (Enter to send, Shift+Enter for newline). "
            "Anything you say is treated as authoritative steering."
        )
        self.setFixedHeight(90)

    def keyPressEvent(self, e: QKeyEvent | None) -> None:  # type: ignore[override]
        if e is not None and e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if e.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                super().keyPressEvent(e)
                return
            self.submitted.emit()
            return
        super().keyPressEvent(e)


class ChatPanel(QWidget):
    """Operator <-> agent side conversation."""

    message_submitted = pyqtSignal(str)   # user text
    clear_requested = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.view = QTextBrowser()
        self.view.setFont(mono_font())
        self.view.setReadOnly(True)
        self.view.setUndoRedoEnabled(False)
        self.view.document().setMaximumBlockCount(4000)
        self.view.setStyleSheet(
            "QTextBrowser { background-color: #11141a; color: #d8d8e0; }"
        )

        self.input = _Input()
        self.input.submitted.connect(self._on_submit)
        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self._on_submit)
        self.clear_btn = QPushButton("Clear chat")
        self.clear_btn.clicked.connect(self.clear_requested.emit)

        button_row = QHBoxLayout()
        button_row.addWidget(self.send_btn)
        button_row.addWidget(self.clear_btn)
        button_row.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(QLabel("Side conversation with the agent"))
        layout.addWidget(self.view, 1)
        layout.addWidget(self.input)
        layout.addLayout(button_row)

        self._streaming = False

    # ----- API for main_window ----------------------------------------------
    def append_user(self, text: str) -> None:
        self._append_block("you", text, color="#ffcc66")

    def begin_assistant(self) -> None:
        self._append_block("agent", "", color="#8fd3ff", trailing_newline=False)
        self._streaming = True

    def append_assistant_delta(self, delta: str) -> None:
        if not self._streaming:
            self.begin_assistant()
        cursor = self.view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        # insertText (not insertHtml) preserves whitespace across streamed
        # token deltas. Apply colour via char format.
        fmt = cursor.charFormat()
        fmt.setForeground(QColor("#8fd3ff"))
        cursor.setCharFormat(fmt)
        cursor.insertText(delta)
        self.view.setTextCursor(cursor)
        sb = self.view.verticalScrollBar()
        if sb is not None:
            sb.setValue(sb.maximum())

    def end_assistant(self) -> None:
        if not self._streaming:
            return
        cursor = self.view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertHtml("<br><br>")
        self.view.setTextCursor(cursor)
        self._streaming = False

    def append_system_note(self, text: str, color: str = "#888") -> None:
        cursor = self.view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        safe = text.replace("<", "&lt;").replace(">", "&gt;")
        cursor.insertHtml(
            f'<span style="color:{color}; font-style:italic">[{safe}]</span><br>'
        )
        self.view.setTextCursor(cursor)

    def clear(self) -> None:
        self.view.clear()
        self._streaming = False

    def focus_input(self) -> None:
        self.input.setFocus()

    # ----- internal ----------------------------------------------------------
    def _append_block(
        self,
        label: str,
        text: str,
        color: str,
        trailing_newline: bool = True,
    ) -> None:
        cursor = self.view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        safe = (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
        )
        cursor.insertHtml(
            f'<b style="color:{color}">{label}:</b> '
            f'<span style="color:{color}">{safe}</span>'
            + ("<br><br>" if trailing_newline else "")
        )
        self.view.setTextCursor(cursor)
        sb = self.view.verticalScrollBar()
        if sb is not None:
            sb.setValue(sb.maximum())

    def _on_submit(self) -> None:
        text = self.input.toPlainText().strip()
        if not text:
            return
        self.input.clear()
        self.message_submitted.emit(text)
