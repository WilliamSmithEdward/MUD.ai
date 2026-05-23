"""ANSI SGR (color) to Qt HTML translator. Handles a useful subset for MUDs."""
from __future__ import annotations

import re
from html import escape


_ANSI_RE = re.compile(r"\x1b\[([0-9;]*)m")
# Strip other common control sequences we will not render.
_OTHER_CSI = re.compile(r"\x1b\[[0-9;?]*[A-HJKSTfnsulh]")

_FG: dict[int, str] = {
    30: "#000000", 31: "#cd0000", 32: "#00cd00", 33: "#cdcd00",
    34: "#1e90ff", 35: "#cd00cd", 36: "#00cdcd", 37: "#e5e5e5",
    90: "#7f7f7f", 91: "#ff0000", 92: "#00ff00", 93: "#ffff00",
    94: "#5c5cff", 95: "#ff00ff", 96: "#00ffff", 97: "#ffffff",
}
_BG: dict[int, str] = {k + 10: v for k, v in _FG.items()}


def ansi_to_html(text: str) -> str:
    """Convert a text fragment containing ANSI SGR escapes into HTML spans.

    Output uses inline styles so it works in any QTextEdit / QTextBrowser.
    Non-SGR control sequences are stripped.
    """
    text = _OTHER_CSI.sub("", text)
    out: list[str] = []
    pos = 0
    fg: str | None = None
    bg: str | None = None
    bold = False
    open_span = False

    def open_tag() -> str:
        styles: list[str] = []
        if fg:
            styles.append(f"color:{fg}")
        if bg:
            styles.append(f"background-color:{bg}")
        if bold:
            styles.append("font-weight:bold")
        if not styles:
            return ""
        return f'<span style="{";".join(styles)}">'

    def close_tag() -> str:
        return "</span>" if open_span else ""

    for m in _ANSI_RE.finditer(text):
        # Append text since last match, escaped and with \n -> <br>
        segment = text[pos:m.start()]
        if segment:
            out.append(escape(segment).replace("\n", "<br>"))
        codes_str = m.group(1)
        codes = [int(c) for c in codes_str.split(";") if c != ""] or [0]
        # State change: close current span, mutate, reopen.
        out.append(close_tag())
        open_span = False
        for code in codes:
            if code == 0:
                fg = None; bg = None; bold = False
            elif code == 1:
                bold = True
            elif code == 22:
                bold = False
            elif code in _FG:
                fg = _FG[code]
            elif code in _BG:
                bg = _BG[code]
            elif code == 39:
                fg = None
            elif code == 49:
                bg = None
        tag = open_tag()
        if tag:
            out.append(tag)
            open_span = True
        pos = m.end()
    tail = text[pos:]
    if tail:
        out.append(escape(tail).replace("\n", "<br>"))
    out.append(close_tag())
    return "".join(out)


def strip_ansi(text: str) -> str:
    """Return text with all ANSI / CSI sequences removed."""
    return _OTHER_CSI.sub("", _ANSI_RE.sub("", text))
