from __future__ import annotations

from mudai.mud.ansi import ansi_to_html, strip_ansi


def test_strip_ansi_removes_color_codes() -> None:
    s = "\x1b[31mred\x1b[0m and \x1b[1;32mbold green\x1b[0m"
    assert strip_ansi(s) == "red and bold green"


def test_strip_ansi_removes_cursor_codes() -> None:
    s = "before\x1b[2Jcleared\x1b[10;5Hpositioned"
    assert strip_ansi(s) == "beforeclearedpositioned"


def test_ansi_to_html_escapes_html() -> None:
    out = ansi_to_html("a <b> & c")
    assert "&lt;b&gt;" in out
    assert "&amp;" in out


def test_ansi_to_html_renders_color_span() -> None:
    out = ansi_to_html("\x1b[31mhello\x1b[0m")
    assert "color:#cd0000" in out
    assert "hello" in out


def test_ansi_to_html_newline_to_br() -> None:
    out = ansi_to_html("line1\nline2")
    assert "<br>" in out


def test_ansi_to_html_handles_reset_between_segments() -> None:
    out = ansi_to_html("\x1b[31mA\x1b[0mB\x1b[32mC\x1b[0m")
    # B should not be inside a color span
    assert ">B<" in out or ">B" in out
    assert "color:#cd0000" in out
    assert "color:#00cd00" in out


def test_ansi_to_html_empty_string() -> None:
    assert ansi_to_html("") == ""
