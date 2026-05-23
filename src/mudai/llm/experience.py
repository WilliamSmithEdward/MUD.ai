"""In-context experience replay.

Indexes operator-approved decisions from past session traces and retrieves the
most relevant prior (situation -> command -> outcome) examples for the agent's
current context. Injecting these into the prompt is the main mechanism by which
the agent gets better at playing the MUD over time without weight updates.

Index format: one entry per approved decision row in ``logs/traces/*.jsonl``:

    {
        "context": <last few lines of the transcript at decision time>,
        "command": <command that was sent>,
        "reasoning": <short reasoning summary, if present>,
        "outcome": <MUD text that followed within the trace window>,
        "tokens": <bag-of-word frequency dict for retrieval>,
    }

Retrieval uses a BM25-lite score over whitespace tokens (lowercased, stripped
of ANSI residue and punctuation). No external deps; computed lazily.
"""
from __future__ import annotations

import json
import math
import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from ..mud.ansi import strip_ansi


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'_-]{1,}")

# Tiny English stopword list — keep small so MUD-specific words dominate.
_STOPWORDS: frozenset[str] = frozenset(
    """
    a an the and or but if then so of in on at to from with by for is are was
    were be been being it its this that these those you your yours i me my we
    our us he she him her his they them their as not no yes do does did doing
    have has had having will would can could should may might must about into
    over under out up down here there now also more most some any all just
    very really only than too own same other another some such own one two
    three said say says new old very much many few
    """.split()
)


@dataclass
class ExperienceEntry:
    context: str
    command: str
    reasoning: str
    outcome: str
    tokens: Counter = field(default_factory=Counter)


def _tokenize(text: str) -> list[str]:
    text = strip_ansi(text or "")
    return [
        t for t in (m.group(0).lower() for m in _TOKEN_RE.finditer(text))
        if t not in _STOPWORDS and len(t) < 32
    ]


def _last_n_lines(text: str, n: int = 12) -> str:
    """Return the last ``n`` non-empty lines of ``text`` joined with newlines."""
    if not text:
        return ""
    lines = [ln for ln in strip_ansi(text).splitlines() if ln.strip()]
    return "\n".join(lines[-n:])


class TraceIndex:
    """Loads approved decisions from trace JSONL files; supports retrieval."""

    def __init__(self, traces_dir: Path) -> None:
        self.traces_dir = traces_dir
        self.entries: list[ExperienceEntry] = []
        self._df: Counter = Counter()
        self._avgdl: float = 1.0
        self._lock = threading.RLock()

    # ----- loading -----------------------------------------------------------
    def reload(self) -> int:
        """Re-scan the traces dir; rebuild the index. Returns entry count."""
        with self._lock:
            entries: list[ExperienceEntry] = []
            for path in sorted(self.traces_dir.glob("*.jsonl")):
                try:
                    entries.extend(self._load_file(path))
                except OSError:
                    continue
            self.entries = entries
            self._rebuild_stats()
            return len(self.entries)

    def add_row(self, row: dict) -> ExperienceEntry | None:
        """Incrementally add a freshly-finalized trace row to the index."""
        entry = self._row_to_entry(row)
        if entry is None:
            return None
        with self._lock:
            self.entries.append(entry)
            self._rebuild_stats()
        return entry

    def _load_file(self, path: Path) -> Iterable[ExperienceEntry]:
        out: list[ExperienceEntry] = []
        for line in path.read_text("utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            entry = self._row_to_entry(row)
            if entry is not None:
                out.append(entry)
        return out

    @staticmethod
    def _row_to_entry(row: dict) -> ExperienceEntry | None:
        if not isinstance(row, dict):
            return None
        if not row.get("approved"):
            return None
        command = (row.get("command") or "").strip()
        if not command:
            return None
        # Pull the transcript-shaped fields the agent already logs.
        context = _last_n_lines(
            row.get("transcript") or row.get("last_mud") or "",
            n=12,
        )
        reasoning = (row.get("reasoning") or "").strip()
        # Truncate reasoning for prompt density.
        if len(reasoning) > 400:
            reasoning = reasoning[:400].rstrip() + "..."
        outcome = _last_n_lines(row.get("outcome") or "", n=8)
        if len(outcome) > 600:
            outcome = outcome[:600].rstrip() + "..."
        tokens = Counter(_tokenize(context + " " + outcome))
        return ExperienceEntry(
            context=context,
            command=command,
            reasoning=reasoning,
            outcome=outcome,
            tokens=tokens,
        )

    # ----- retrieval ---------------------------------------------------------
    def _rebuild_stats(self) -> None:
        self._df = Counter()
        total_len = 0
        for e in self.entries:
            total_len += sum(e.tokens.values())
            for term in e.tokens:
                self._df[term] += 1
        n = max(1, len(self.entries))
        self._avgdl = (total_len / n) if n else 1.0

    def retrieve(self, query: str, k: int = 3) -> list[ExperienceEntry]:
        """Return up to ``k`` entries best matching ``query`` (BM25-lite)."""
        if k <= 0:
            return []
        with self._lock:
            if not self.entries:
                return []
            q_tokens = _tokenize(query)
            if not q_tokens:
                return []
            n = len(self.entries)
            k1, b = 1.5, 0.75
            scored: list[tuple[float, ExperienceEntry]] = []
            for entry in self.entries:
                dl = sum(entry.tokens.values()) or 1
                score = 0.0
                for term in set(q_tokens):
                    tf = entry.tokens.get(term, 0)
                    if tf == 0:
                        continue
                    df = self._df.get(term, 0) or 1
                    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                    denom = tf + k1 * (1 - b + b * dl / self._avgdl)
                    score += idf * (tf * (k1 + 1)) / denom
                if score > 0:
                    scored.append((score, entry))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [e for _, e in scored[:k]]

    # ----- prompt rendering --------------------------------------------------
    @staticmethod
    def render_examples(entries: list[ExperienceEntry]) -> str:
        if not entries:
            return ""
        blocks: list[str] = []
        for i, e in enumerate(entries, 1):
            parts = [f"Example {i}:"]
            if e.context:
                parts.append("Situation:\n" + e.context)
            if e.reasoning:
                parts.append("Reasoning: " + e.reasoning.replace("\n", " "))
            parts.append("Command sent: " + e.command)
            if e.outcome:
                parts.append("Outcome:\n" + e.outcome)
            blocks.append("\n".join(parts))
        return "\n\n".join(blocks)


__all__ = ["TraceIndex", "ExperienceEntry"]
