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
        # Try a curated list of terminal-grade fonts; fall back to the system
        # monospace via StyleHint + fixed-pitch if none are installed.
        for family in ("Cascadia Mono", "Consolas", "Courier New", "Monospace"):
            f = QFont(family, 10)
            if f.exactMatch():
                f.setStyleHint(QFont.StyleHint.Monospace)
                f.setFixedPitch(True)
                _MONO = f
                break
        else:
            f = QFont()
            f.setStyleHint(QFont.StyleHint.Monospace)
            f.setFamily("monospace")
            f.setFixedPitch(True)
            f.setPointSize(10)
            _MONO = f
    return _MONO


class MudOutputView(QTextBrowser):
    """Append-only ANSI-rendered MUD output."""

    def __init__(self) -> None:
        super().__init__()
        f = mono_font()
        self.setFont(f)
        # Ensure the document itself defaults to the monospace font so any
        # whitespace runs (rendered as &nbsp;) keep terminal alignment.
        self.document().setDefaultFont(f)
        self.setOpenExternalLinks(False)
        self.setReadOnly(True)
        self.setUndoRedoEnabled(False)
        self.document().setMaximumBlockCount(5000)
        # Tighten line spacing so multi-line ASCII art looks like a terminal.
        self.setStyleSheet(
            "QTextBrowser {"
            " background-color: #0b0b0f;"
            " color: #d6d6d6;"
            " font-family: 'Cascadia Mono','Consolas','Courier New',monospace;"
            " }"
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
        safe = (
            text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace(" ", "&nbsp;")
                .replace("\n", "<br>")
        )
        cursor.insertHtml(f'<span style="color:{color}">&gt;&nbsp;{safe}</span><br>')
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
