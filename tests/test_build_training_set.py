from __future__ import annotations

import json
from pathlib import Path

from mudai.scripts.build_training_set import build


def _write_trace(p: Path, rows: list[dict]) -> None:
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_build_filters_rejected_by_default(tmp_path: Path) -> None:
    traces = tmp_path / "traces"
    traces.mkdir()
    _write_trace(
        traces / "s.jsonl",
        [
            {
                "system": "SYS",
                "transcript": [{"role": "mud", "text": "You see a rat."}],
                "raw_response": "It's a rat.\nCOMMAND: kill rat",
                "command": "kill rat",
                "approved": True,
            },
            {
                "system": "SYS",
                "transcript": [{"role": "mud", "text": "Boom"}],
                "raw_response": "bad\nCOMMAND: flee",
                "command": "flee",
                "approved": False,
            },
        ],
    )
    out = tmp_path / "training.jsonl"
    n = build(traces, out)
    assert n == 1
    line = json.loads(out.read_text("utf-8").splitlines()[0])
    msgs = line["messages"]
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "[MUD] You see a rat." in msgs[1]["content"]
    assert msgs[2]["content"].endswith("COMMAND: kill rat")


def test_build_include_rejected(tmp_path: Path) -> None:
    traces = tmp_path / "t"
    traces.mkdir()
    _write_trace(
        traces / "s.jsonl",
        [{"system": "S", "transcript": [], "command": "x", "approved": False}],
    )
    out = tmp_path / "out.jsonl"
    assert build(traces, out, include_rejected=True) == 1


def test_build_synthesizes_assistant_when_raw_missing(tmp_path: Path) -> None:
    traces = tmp_path / "t"
    traces.mkdir()
    _write_trace(
        traces / "s.jsonl",
        [
            {
                "system": "S",
                "transcript": [],
                "reasoning": "go north",
                "command": "north",
                "approved": True,
            }
        ],
    )
    out = tmp_path / "out.jsonl"
    build(traces, out)
    line = json.loads(out.read_text("utf-8").splitlines()[0])
    assert "go north" in line["messages"][2]["content"]
    assert "COMMAND: north" in line["messages"][2]["content"]


def test_build_includes_chat_rows_regardless_of_approval(tmp_path: Path) -> None:
    traces = tmp_path / "t"
    traces.mkdir()
    _write_trace(
        traces / "s.jsonl",
        [
            {
                "type": "chat",
                "system": "CHAT SYS",
                "user": "Should I fight the troll?",
                "assistant": "No, flee south first.",
            },
            {
                "type": "chat",
                "system": "CHAT SYS",
                "user": "",
                "assistant": "ignored - empty user",
            },
        ],
    )
    out = tmp_path / "out.jsonl"
    n = build(traces, out)
    assert n == 1
    line = json.loads(out.read_text("utf-8").splitlines()[0])
    assert line["messages"][1]["content"] == "Should I fight the troll?"
    assert line["messages"][2]["content"] == "No, flee south first."
