"""The agent: builds prompts, manages context window, parses LLM output into a command."""
from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from ..config import AgentConfig, LLMConfig
from .backend import LlamaBackend
from .memory import MemoryStore


@dataclass
class TranscriptEntry:
    role: str           # "mud" | "you" | "agent"
    text: str
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class Decision:
    reasoning: str       # visible reasoning shown to the operator
    command: str
    raw: str             # full LLM response (including <think> if present)
    thinking: str = ""   # contents of Qwen3-style <think>...</think> blocks


# Matches `COMMAND: <text>` on its own line (case-insensitive).
_COMMAND_RE = re.compile(r"(?im)^\s*COMMAND\s*:\s*(.+?)\s*$")
# Matches Qwen3-style chain-of-thought blocks (may be unterminated mid-stream).
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


class Agent:
    """Owns the rolling transcript, the LLM, and the prompt assembly."""

    def __init__(
        self,
        backend: LlamaBackend,
        agent_cfg: AgentConfig,
        llm_cfg: LLMConfig,
        memory: MemoryStore | None = None,
    ) -> None:
        self.backend = backend
        self.agent_cfg = agent_cfg
        self.llm_cfg = llm_cfg
        self.memory = memory
        # Bounded deque so memory cannot grow unbounded across long sessions.
        self.transcript: deque[TranscriptEntry] = deque(maxlen=2000)

    # ----- transcript --------------------------------------------------------
    def add_mud(self, text: str) -> None:
        if text.strip():
            self.transcript.append(TranscriptEntry("mud", text))

    def add_user(self, text: str) -> None:
        if text.strip():
            self.transcript.append(TranscriptEntry("you", text))

    def add_agent(self, text: str) -> None:
        if text.strip():
            self.transcript.append(TranscriptEntry("agent", text))

    def add_operator_note(self, text: str) -> None:
        """One-shot operator context. Surfaces in the prompt as [OPERATOR]."""
        if text.strip():
            self.transcript.append(TranscriptEntry("operator", text))

    # ----- prompt assembly ---------------------------------------------------
    def system_message(self) -> str:
        """Public accessor for the assembled system prompt (system + steering)."""
        return self._system_message()

    def _system_message(self) -> str:
        steering = (self.agent_cfg.steering_notes or "").strip()
        base = self.agent_cfg.system_prompt.rstrip()
        parts = [base]
        mem_text = self.memory.render_for_prompt() if self.memory is not None else ""
        if mem_text:
            parts.append(
                "PERMANENT MEMORY (durable facts you have learned; treat as"
                " authoritative):\n" + mem_text
            )
        if steering:
            parts.append(
                "STEERING NOTES (operator, authoritative):\n" + steering
            )
        return "\n\n".join(parts)

    def _transcript_text(self) -> str:
        """Render newest-last transcript trimmed to the token budget."""
        lines: list[str] = []
        budget = self.llm_cfg.transcript_token_budget
        used = 0
        # Walk newest-first, accumulate, then reverse.
        for entry in reversed(self.transcript):
            label = {
                "mud": "[MUD]",
                "you": "[YOU]",
                "agent": "[AGENT]",
                "operator": "[OPERATOR]",
            }.get(entry.role, "[?]")
            chunk = f"{label} {entry.text.rstrip()}"
            tok = self.backend.count_tokens(chunk)
            if used + tok > budget:
                break
            lines.append(chunk)
            used += tok
        lines.reverse()
        return "\n".join(lines)

    def transcript_text(self) -> str:
        """Public accessor for the trimmed transcript rendering."""
        return self._transcript_text()

    def build_messages(self) -> list[dict[str, str]]:
        user_block = (
            "Recent MUD transcript (oldest first):\n"
            "------------------------------------\n"
            f"{self._transcript_text()}\n"
            "------------------------------------\n\n"
            "Decide the next command. Reply with brief reasoning, then a final line:\n"
            "COMMAND: <text>\n"
        )
        return [
            {"role": "system", "content": self._system_message()},
            {"role": "user", "content": user_block},
        ]

    # ----- decision ----------------------------------------------------------
    @staticmethod
    def parse_decision(raw: str) -> Decision:
        # 1. Pull out any Qwen3-style <think>...</think> blocks first.
        think_parts = _THINK_RE.findall(raw)
        visible = _THINK_RE.sub("", raw)
        # Handle an unterminated <think> tail (can happen if max_tokens hit).
        if "<think>" in visible.lower() and "</think>" not in visible.lower():
            idx = visible.lower().rfind("<think>")
            think_parts.append(visible[idx + len("<think>") :])
            visible = visible[:idx]
        thinking = "\n".join(p.strip() for p in think_parts if p.strip())

        # 2. Find COMMAND: marker in the visible portion.
        match = _COMMAND_RE.search(visible)
        if match:
            command = match.group(1).strip()
            reasoning = visible[: match.start()].strip()
        else:
            non_empty = [ln.strip() for ln in visible.splitlines() if ln.strip()]
            command = non_empty[-1] if non_empty else ""
            reasoning = "\n".join(non_empty[:-1]) if len(non_empty) > 1 else ""
        # Strip surrounding quotes/backticks the model sometimes adds.
        command = command.strip("`\"' ")
        return Decision(
            reasoning=reasoning, command=command, raw=raw, thinking=thinking
        )

    def decide_stream(
        self, on_delta: Callable[[str], None]
    ) -> Decision:
        """Run one decision; stream raw text deltas to on_delta; return parsed Decision."""
        messages = self.build_messages()
        parts: list[str] = []
        for delta in self.backend.stream_chat(messages):
            parts.append(delta)
            on_delta(delta)
        raw = "".join(parts)
        return self.parse_decision(raw)
