"""ANSI SGR (color) to Qt HTML translator. Handles a useful subset for MUDs.

Supports:
  * Standard 16 foreground/background colors (30-37, 40-47, 90-97, 100-107)
  * 256-color palette via 38;5;N and 48;5;N
  * 24-bit truecolor via 38;2;R;G;B and 48;2;R;G;B
  * Bold (1), faint (2), italic (3), underline (4), reverse video (7)
  * Their resets: 22, 23, 24, 27, 39, 49, and full reset 0
Whitespace is preserved (spaces, tabs, newlines) so MUD room layouts and
ASCII art render correctly inside Qt rich-text views.
"""
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


def _xterm256(n: int) -> str:
    """Map an xterm 256-color index to a #rrggbb string."""
    if n < 0:
        n = 0
    if n > 255:
        n = 255
    if n < 16:
        # First 16 mirror the standard palette (use bright variants for 8-15).
        base = (30 + n) if n < 8 else (90 + (n - 8))
        return _FG[base]
    if n < 232:
        # 6x6x6 color cube.
        i = n - 16
        r = (i // 36) % 6
        g = (i // 6) % 6
        b = i % 6
        steps = [0, 95, 135, 175, 215, 255]
        return f"#{steps[r]:02x}{steps[g]:02x}{steps[b]:02x}"
    # Grayscale ramp.
    level = 8 + (n - 232) * 10
    return f"#{level:02x}{level:02x}{level:02x}"


def _escape_preserve_ws(segment: str) -> str:
    """HTML-escape and preserve whitespace (spaces, tabs, newlines)."""
    s = escape(segment)
    s = s.replace("\t", "&nbsp;" * 8)
    s = s.replace(" ", "&nbsp;")
    s = s.replace("\n", "<br>")
    return s


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
    faint = False
    italic = False
    underline = False
    reverse = False
    open_span = False

    def open_tag() -> str:
        eff_fg, eff_bg = fg, bg
        if reverse:
            # Swap, with sensible defaults if either side is unset.
            eff_fg, eff_bg = (bg or "#0b0b0f"), (fg or "#d6d6d6")
        styles: list[str] = []
        if eff_fg:
            styles.append(f"color:{eff_fg}")
        if eff_bg:
            styles.append(f"background-color:{eff_bg}")
        if bold:
            styles.append("font-weight:bold")
        if faint:
            styles.append("opacity:0.6")
        if italic:
            styles.append("font-style:italic")
        if underline:
            styles.append("text-decoration:underline")
        if not styles:
            return ""
        return f'<span style="{";".join(styles)}">'

    def close_tag() -> str:
        return "</span>" if open_span else ""

    def apply_codes(codes: list[int]) -> None:
        nonlocal fg, bg, bold, faint, italic, underline, reverse
        i = 0
        while i < len(codes):
            code = codes[i]
            if code == 0:
                fg = None; bg = None
                bold = faint = italic = underline = reverse = False
            elif code == 1:
                bold = True
            elif code == 2:
                faint = True
            elif code == 3:
                italic = True
            elif code == 4:
                underline = True
            elif code == 7:
                reverse = True
            elif code == 22:
                bold = False; faint = False
            elif code == 23:
                italic = False
            elif code == 24:
                underline = False
            elif code == 27:
                reverse = False
            elif code == 39:
                fg = None
            elif code == 49:
                bg = None
            elif code in _FG:
                fg = _FG[code]
            elif code in _BG:
                bg = _BG[code]
            elif code in (38, 48):
                # Extended color: 38;5;N (indexed) or 38;2;R;G;B (rgb).
                target_is_fg = (code == 38)
                if i + 1 < len(codes):
                    mode = codes[i + 1]
                    if mode == 5 and i + 2 < len(codes):
                        color = _xterm256(codes[i + 2])
                        if target_is_fg:
                            fg = color
                        else:
                            bg = color
                        i += 2
                    elif mode == 2 and i + 4 < len(codes):
                        r, g, b = codes[i + 2], codes[i + 3], codes[i + 4]
                        color = (
                            f"#{max(0, min(255, r)):02x}"
                            f"{max(0, min(255, g)):02x}"
                            f"{max(0, min(255, b)):02x}"
                        )
                        if target_is_fg:
                            fg = color
                        else:
                            bg = color
                        i += 4
                    else:
                        i += 1
            # Unknown codes silently ignored.
            i += 1

    for m in _ANSI_RE.finditer(text):
        segment = text[pos:m.start()]
        if segment:
            out.append(_escape_preserve_ws(segment))
        codes_str = m.group(1)
        codes = [int(c) for c in codes_str.split(";") if c != ""] or [0]
        # State change: close current span, mutate, reopen.
        out.append(close_tag())
        open_span = False
        apply_codes(codes)
        tag = open_tag()
        if tag:
            out.append(tag)
            open_span = True
        pos = m.end()
    tail = text[pos:]
    if tail:
        out.append(_escape_preserve_ws(tail))
    out.append(close_tag())
    return "".join(out)


def strip_ansi(text: str) -> str:
    """Return text with all ANSI / CSI sequences removed."""
    return _OTHER_CSI.sub("", _ANSI_RE.sub("", text))
