"""The agent: builds prompts, manages context window, parses LLM output into a command."""
from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from ..config import AgentConfig, LLMConfig
from .backend import LlamaBackend
from .experience import TraceIndex
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
        experience: TraceIndex | None = None,
    ) -> None:
        self.backend = backend
        self.agent_cfg = agent_cfg
        self.llm_cfg = llm_cfg
        self.memory = memory
        self.experience = experience
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
        transcript = self._transcript_text()
        examples_block = ""
        k = getattr(self.agent_cfg, "experience_examples_k", 0) or 0
        if k > 0 and self.experience is not None:
            # Use the tail of the live transcript as the retrieval query.
            tail = "\n".join(transcript.splitlines()[-20:])
            examples = self.experience.retrieve(tail, k=k)
            rendered = self.experience.render_examples(examples)
            if rendered:
                examples_block = (
                    "PAST SUCCESSFUL EXAMPLES (operator-approved decisions from"
                    " earlier sessions; use as guidance, do not copy verbatim if"
                    " the situation differs):\n"
                    "========================================\n"
                    f"{rendered}\n"
                    "========================================\n\n"
                )
        user_block = (
            f"{examples_block}"
            "Recent MUD transcript (oldest first):\n"
            "------------------------------------\n"
            f"{transcript}\n"
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

        # 2. Find ALL COMMAND: markers in the visible portion. The model may
        # emit several to chain actions in a single decision (e.g.
        #   COMMAND: open door
        #   COMMAND: n
        #   COMMAND: look
        # ). We join them with " ; " so downstream code can split and send
        # them in order with the usual min-command-interval spacing.
        matches = list(_COMMAND_RE.finditer(visible))
        if matches:
            cmds = [m.group(1).strip().strip("`\"' ") for m in matches]
            cmds = [c for c in cmds if c]
            command = " ; ".join(cmds)
            reasoning = visible[: matches[0].start()].strip()
        else:
            non_empty = [ln.strip() for ln in visible.splitlines() if ln.strip()]
            command = (non_empty[-1] if non_empty else "").strip("`\"' ")
            reasoning = "\n".join(non_empty[:-1]) if len(non_empty) > 1 else ""
        return Decision(
            reasoning=reasoning, command=command, raw=raw, thinking=thinking
        )

    @staticmethod
    def split_commands(text: str) -> list[str]:
        """Split a (possibly multi-command) proposal into individual commands.

        Honours both `;` and embedded newlines as separators. Empty fragments
        are dropped. Each fragment is stripped of surrounding whitespace and
        wrapping quotes/backticks the model sometimes adds.
        """
        if not text:
            return []
        # Normalize newlines to ';' so we can split once.
        normalized = text.replace("\r\n", "\n").replace("\n", " ; ")
        parts = [p.strip().strip("`\"' ") for p in normalized.split(";")]
        return [p for p in parts if p]

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

    # ----- reflection --------------------------------------------------------
    def build_reflection_messages(
        self, recent_examples: list[str], max_lessons: int = 5
    ) -> list[dict[str, str]]:
        """Build a prompt that asks the model to distill durable lessons.

        ``recent_examples`` is a list of pre-rendered text blocks describing
        recent (situation, command, outcome) tuples. The model is instructed to
        reply ONLY with REMEMBER: lines, which the memory system will capture.
        """
        sys = (
            "You are reviewing your own recent play of a text-based MUD to"
            " extract durable lessons that will help you play better in the"
            " future. You will see several (situation, command you chose,"
            " outcome) tuples. Identify patterns: what worked, what failed,"
            " hazards to avoid, syntax that the MUD actually accepts, names of"
            " important places or NPCs.\n\n"
            f"Reply with AT MOST {max_lessons} lines, each in the exact form:\n"
            "REMEMBER: <one concise, generally-applicable fact or rule>\n\n"
            "Rules:\n"
            "  - Each REMEMBER must be actionable and broadly useful (not"
            " trivia about a single moment).\n"
            "  - No prose, no headers, no numbering. Only REMEMBER: lines.\n"
            "  - If nothing new was learned, output a single line: NONE"
        )
        body = "\n\n".join(recent_examples) if recent_examples else "(no recent decisions)"
        user = (
            "Recent decisions to learn from:\n"
            "===============================\n"
            f"{body}\n"
            "===============================\n"
        )
        return [
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ]

    def reflect(self, recent_examples: list[str], max_lessons: int = 5) -> str:
        """Run a reflection pass; returns the raw LLM text (REMEMBER: lines).

        Caller is expected to feed the returned text through ``MemoryStore``
        ``capture_from_text`` to persist the lessons.
        """
        messages = self.build_reflection_messages(recent_examples, max_lessons)
        parts: list[str] = []
        for delta in self.backend.stream_chat(messages):
            parts.append(delta)
        return "".join(parts)
