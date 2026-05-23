from __future__ import annotations

import json
from pathlib import Path

from mudai.config import AgentConfig, LLMConfig
from mudai.llm.agent import Agent
from mudai.llm.backend import LlamaBackend
from mudai.llm.memory import MemoryStore


def _store(tmp_path: Path) -> MemoryStore:
    return MemoryStore.load(tmp_path / "memory.jsonl")


def test_add_persists_atomically_to_disk(tmp_path: Path) -> None:
    s = _store(tmp_path)
    e = s.add("the troll guards the bridge", source="operator")
    assert e is not None
    reloaded = MemoryStore.load(tmp_path / "memory.jsonl")
    assert len(reloaded.entries) == 1
    assert reloaded.entries[0].text == "the troll guards the bridge"


def test_add_dedupes_case_insensitive_and_reenables(tmp_path: Path) -> None:
    s = _store(tmp_path)
    e1 = s.add("Always rest before fighting", source="operator")
    assert e1 is not None
    s.update(e1.id, enabled=False)
    e2 = s.add("ALWAYS REST BEFORE FIGHTING", source="agent_chat")
    assert e2 is not None
    assert e2.id == e1.id
    assert e2.enabled is True
    assert len(s.entries) == 1


def test_capture_from_text_picks_up_remember_lines(tmp_path: Path) -> None:
    s = _store(tmp_path)
    text = (
        "thinking out loud...\n"
        "REMEMBER: never attack the temple guards\n"
        "more reasoning\n"
        "remember:   the inn is north of the fountain\n"
        "COMMAND: look\n"
    )
    captured = s.capture_from_text(text, source="agent_decision")
    assert len(captured) == 2
    facts = [e.text for e in s.entries]
    assert "never attack the temple guards" in facts
    assert "the inn is north of the fountain" in facts


def test_delete_and_clear(tmp_path: Path) -> None:
    s = _store(tmp_path)
    e = s.add("fact one")
    s.add("fact two")
    assert e is not None
    assert s.delete(e.id) is True
    assert s.delete("nonexistent") is False
    assert len(s.entries) == 1
    s.clear()
    assert s.entries == []
    # Persistence check: file is now empty.
    assert (tmp_path / "memory.jsonl").read_text("utf-8") == ""


def test_render_for_prompt_omits_disabled(tmp_path: Path) -> None:
    s = _store(tmp_path)
    a = s.add("enabled fact")
    b = s.add("disabled fact")
    assert b is not None
    s.update(b.id, enabled=False)
    rendered = s.render_for_prompt()
    assert "enabled fact" in rendered
    assert "disabled fact" not in rendered
    assert rendered.startswith("- ")
    # Empty store renders to "".
    s.clear()
    assert s.render_for_prompt() == ""
    assert a is not None


def test_agent_system_prompt_includes_memory(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.add("the dragon is immune to fire")
    s.add("kobolds drop silver pieces")
    backend = LlamaBackend(Path("does-not-exist.gguf"), LLMConfig())
    agent = Agent(backend, AgentConfig(), LLMConfig(), memory=s)
    msg = agent.system_message()
    assert "PERMANENT MEMORY" in msg
    assert "the dragon is immune to fire" in msg
    assert "kobolds drop silver pieces" in msg


def test_subscribe_fires_on_mutation(tmp_path: Path) -> None:
    s = _store(tmp_path)
    calls: list[int] = []
    s.subscribe(lambda: calls.append(1))
    s.add("x")
    s.add("y")
    e = s.entries[-1]
    s.update(e.id, text="y2")
    s.delete(e.id)
    assert len(calls) == 4


def test_corrupt_lines_skipped_on_load(tmp_path: Path) -> None:
    p = tmp_path / "memory.jsonl"
    p.write_text(
        "not json\n"
        + json.dumps({
            "id": "abc", "text": "good", "source": "operator",
            "ts": "2026-01-01T00:00:00+00:00", "enabled": True, "tag": "",
        })
        + "\n",
        encoding="utf-8",
    )
    s = MemoryStore.load(p)
    assert len(s.entries) == 1
    assert s.entries[0].text == "good"
