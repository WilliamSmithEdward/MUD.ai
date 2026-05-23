"""Persistent memory store: durable facts the LLM should always know.

Lives between sessions as JSONL on disk so the operator can edit it with any
text editor. Both the decision-loop agent and the side-chat session inject the
current enabled memory entries into their system prompts as a bullet list.

Either the operator OR the LLM may add entries:
  * Operator: clicks "Save to memory" or "Add" in the Memory panel.
  * LLM: emits a line of the form ``REMEMBER: <fact>`` in any response;
    ``capture_from_text`` extracts and stores it.
"""
from __future__ import annotations

import json
import re
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


_REMEMBER_RE = re.compile(r"(?im)^\s*REMEMBER\s*:\s*(.+?)\s*$")


@dataclass
class MemoryEntry:
    id: str
    text: str
    source: str           # "operator" | "agent_decision" | "agent_chat"
    ts: str
    enabled: bool = True
    tag: str = ""

    @staticmethod
    def new(text: str, source: str, tag: str = "") -> "MemoryEntry":
        return MemoryEntry(
            id=uuid.uuid4().hex[:12],
            text=text.strip(),
            source=source,
            ts=datetime.now(timezone.utc).isoformat(),
            tag=tag.strip(),
        )


@dataclass
class MemoryStore:
    """Thread-safe JSONL-backed list of memory entries."""

    path: Path
    entries: list[MemoryEntry] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _listeners: list[Callable[[], None]] = field(default_factory=list, repr=False)

    # ----- persistence -------------------------------------------------------
    @classmethod
    def load(cls, path: Path) -> "MemoryStore":
        store = cls(path=path)
        if path.exists():
            for line in path.read_text("utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    store.entries.append(MemoryEntry(**data))
                except (json.JSONDecodeError, TypeError):
                    continue
        return store

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                for e in self.entries:
                    f.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")
            tmp.replace(self.path)

    # ----- listeners ---------------------------------------------------------
    def subscribe(self, fn: Callable[[], None]) -> None:
        self._listeners.append(fn)

    def _notify(self) -> None:
        for fn in list(self._listeners):
            try:
                fn()
            except Exception:
                pass

    # ----- mutations ---------------------------------------------------------
    def add(self, text: str, source: str = "operator", tag: str = "") -> MemoryEntry | None:
        text = (text or "").strip()
        if not text:
            return None
        with self._lock:
            # Dedupe on exact text (case-insensitive).
            for existing in self.entries:
                if existing.text.lower() == text.lower():
                    if not existing.enabled:
                        existing.enabled = True
                        self.save()
                        self._notify()
                    return existing
            entry = MemoryEntry.new(text, source, tag)
            self.entries.append(entry)
            self.save()
        self._notify()
        return entry

    def update(self, entry_id: str, *, text: str | None = None,
               enabled: bool | None = None, tag: str | None = None) -> bool:
        with self._lock:
            for e in self.entries:
                if e.id == entry_id:
                    if text is not None:
                        e.text = text.strip()
                    if enabled is not None:
                        e.enabled = enabled
                    if tag is not None:
                        e.tag = tag.strip()
                    self.save()
                    self._notify()
                    return True
        return False

    def delete(self, entry_id: str) -> bool:
        with self._lock:
            before = len(self.entries)
            self.entries = [e for e in self.entries if e.id != entry_id]
            if len(self.entries) == before:
                return False
            self.save()
        self._notify()
        return True

    def clear(self) -> None:
        with self._lock:
            self.entries = []
            self.save()
        self._notify()

    # ----- query / render ----------------------------------------------------
    def enabled_entries(self) -> list[MemoryEntry]:
        with self._lock:
            return [e for e in self.entries if e.enabled]

    def render_for_prompt(self, max_entries: int = 50) -> str:
        """Return a bullet-list rendering of enabled entries, newest last."""
        entries = self.enabled_entries()
        if not entries:
            return ""
        # Newest last so recency bias helps; cap to max_entries.
        if len(entries) > max_entries:
            entries = entries[-max_entries:]
        lines = [f"- {e.text}" for e in entries]
        return "\n".join(lines)

    # ----- auto-capture ------------------------------------------------------
    def capture_from_text(self, text: str, source: str) -> list[MemoryEntry]:
        """Scan ``text`` for REMEMBER: lines; persist each as an entry."""
        captured: list[MemoryEntry] = []
        for match in _REMEMBER_RE.finditer(text or ""):
            entry = self.add(match.group(1).strip(), source=source)
            if entry is not None:
                captured.append(entry)
        return captured
