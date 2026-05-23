from __future__ import annotations

import json
from pathlib import Path

from mudai.llm.experience import TraceIndex


def _write_trace(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )


def test_reload_indexes_only_approved(tmp_path: Path) -> None:
    _write_trace(tmp_path / "s1.jsonl", [
        {"approved": True, "command": "north",
         "transcript": "[MUD] A dim cave. Exits: north."},
        {"approved": False, "command": "south",
         "transcript": "[MUD] A bright meadow."},
        {"approved": True, "command": "look",
         "transcript": "[MUD] You see an altar.", "outcome": "An altar of stone."},
    ])
    idx = TraceIndex(tmp_path)
    n = idx.reload()
    assert n == 2
    cmds = sorted(e.command for e in idx.entries)
    assert cmds == ["look", "north"]


def test_retrieve_prefers_topical_match(tmp_path: Path) -> None:
    _write_trace(tmp_path / "s.jsonl", [
        {"approved": True, "command": "kill orc",
         "transcript": "[MUD] An orc growls at you in the cave.",
         "outcome": "You attack the orc."},
        {"approved": True, "command": "buy bread",
         "transcript": "[MUD] The baker smiles at you in the shop.",
         "outcome": "You buy bread."},
        {"approved": True, "command": "drink water",
         "transcript": "[MUD] A clear fountain stands here.",
         "outcome": "You feel refreshed."},
    ])
    idx = TraceIndex(tmp_path)
    idx.reload()
    hits = idx.retrieve("orc growls in the cave", k=1)
    assert len(hits) == 1
    assert hits[0].command == "kill orc"


def test_retrieve_empty_when_no_matches(tmp_path: Path) -> None:
    _write_trace(tmp_path / "s.jsonl", [
        {"approved": True, "command": "north",
         "transcript": "[MUD] A dim cave."},
    ])
    idx = TraceIndex(tmp_path)
    idx.reload()
    assert idx.retrieve("", k=3) == []
    # Query with only stopwords and unmatched terms.
    assert idx.retrieve("xyzzy plugh quux", k=3) == []


def test_add_row_appends_to_index(tmp_path: Path) -> None:
    idx = TraceIndex(tmp_path)
    idx.reload()
    assert idx.entries == []
    e = idx.add_row({
        "approved": True, "command": "look",
        "transcript": "[MUD] You stand in a hall.",
    })
    assert e is not None
    assert len(idx.entries) == 1
    hits = idx.retrieve("hall", k=1)
    assert hits and hits[0].command == "look"


def test_render_examples_includes_situation_and_command(tmp_path: Path) -> None:
    _write_trace(tmp_path / "s.jsonl", [
        {"approved": True, "command": "score",
         "transcript": "[MUD] You feel weak.",
         "outcome": "Hp 5/100"},
    ])
    idx = TraceIndex(tmp_path)
    idx.reload()
    text = TraceIndex.render_examples(idx.entries)
    assert "Command sent: score" in text
    assert "You feel weak." in text
    assert "Hp 5/100" in text
