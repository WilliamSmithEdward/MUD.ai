"""Custom widgets: ANSI-aware MUD output view, streaming reasoning view."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QTextCursor
from PyQt6.QtWidgets import QTextBrowser

from ..mud.ansi import ansi_to_html


_MONO: QFont | None = None


def mono_font() -> QFont:
    """Lazily construct the monospace font (requires QApplication to exist)."""
    global _MONO
    if _MONO is None:
        f = QFont("Cascadia Mono", 10)
        if not f.exactMatch():
            f = QFont("Consolas", 10)
        _MONO = f
    return _MONO


class MudOutputView(QTextBrowser):
    """Append-only ANSI-rendered MUD output."""

    def __init__(self) -> None:
        super().__init__()
        self.setFont(mono_font())
        self.setOpenExternalLinks(False)
        self.setReadOnly(True)
        self.setUndoRedoEnabled(False)
        self.document().setMaximumBlockCount(5000)
        self.setStyleSheet(
            "QTextBrowser { background-color: #0b0b0f; color: #d6d6d6; }"
        )

    def append_mud(self, raw_text: str) -> None:
        html = ansi_to_html(raw_text)
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertHtml(html)
        self.setTextCursor(cursor)
        sb = self.verticalScrollBar()
        if sb is not None:
            sb.setValue(sb.maximum())

    def append_local(self, text: str, color: str = "#ffcc66") -> None:
        """Echo a locally-sent line (yours or the agent's) into the transcript."""
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        safe = text.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        cursor.insertHtml(f'<span style="color:{color}">&gt; {safe}</span><br>')
        self.setTextCursor(cursor)
        sb = self.verticalScrollBar()
        if sb is not None:
            sb.setValue(sb.maximum())


class ReasoningView(QTextBrowser):
    """Streaming text widget for the LLM's chain-of-thought."""

    def __init__(self) -> None:
        super().__init__()
        self.setFont(mono_font())
        self.setReadOnly(True)
        self.setUndoRedoEnabled(False)
        self.document().setMaximumBlockCount(2000)
        self.setStyleSheet(
            "QTextBrowser { background-color: #0f1014; color: #b8e0b8; }"
        )

    def start_new(self, header: str) -> None:
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertHtml(
            f'<br><span style="color:#888"># {header}</span><br>'
        )
        self.setTextCursor(cursor)

    def append_delta(self, delta: str) -> None:
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        # Use insertText, not insertHtml: HTML collapses whitespace between
        # consecutive fragments, which mangles streamed token deltas.
        cursor.insertText(delta)
        self.setTextCursor(cursor)
        sb = self.verticalScrollBar()
        if sb is not None:
            sb.setValue(sb.maximum())

    def append_note(self, note: str, color: str = "#888") -> None:
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertHtml(f'<br><span style="color:{color}">{note}</span><br>')
        self.setTextCursor(cursor)


__all__ = ["MudOutputView", "ReasoningView", "mono_font"]
del Qt  # silence unused-import on some pyright versions
