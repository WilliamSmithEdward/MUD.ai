"""Side-channel chat between the operator and the LLM.

This is independent from the autonomous decision loop:
  * The decision loop reads the rolling MUD transcript and emits ONE command.
  * The chat session is a free-form conversation. It sees a snapshot of the
    same MUD transcript (so it can reason about the current game state) plus
    its own message history.

Both flows share a single ``LlamaBackend``; the backend serializes calls with
an internal lock so neither can corrupt the other's KV cache.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from ..config import AgentConfig, LLMConfig
from .agent import Agent
from .backend import LlamaBackend


CHAT_SYSTEM_PROMPT = (
    "You are the same AI that is autonomously playing a text-based MUD for the"
    " operator. Right now you are NOT issuing a game command - you are having a"
    " side conversation with the operator about the game.\n"
    "Rules:\n"
    "  1. Be concise. Plain prose. No COMMAND: marker. Do not send anything to"
    " the MUD here.\n"
    "  2. You may ask the operator clarifying questions when you are uncertain"
    " (objectives, unfamiliar mobs, unclear room descriptions, what loot to"
    " keep, when to flee, etc.).\n"
    "  3. Anything the operator tells you here is authoritative steering for"
    " your future decisions. Treat it like a STEERING NOTE update.\n"
    "  4. You can reference the latest MUD transcript shown below for context.\n"
    "  5. If the operator teaches you a DURABLE fact (a rule, a name, a"
    " location, a danger) you should always remember, add a line of the form"
    " 'REMEMBER: <one concise fact>' anywhere in your reply. It will be saved"
    " to permanent memory and shown to you on every future decision. Use"
    " sparingly - one fact per line, and only for things worth keeping forever.\n"
)


@dataclass
class ChatTurn:
    role: str   # "user" | "assistant"
    text: str
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ChatSession:
    """Holds chat history; renders prompts that include the live MUD context."""

    def __init__(
        self,
        backend: LlamaBackend,
        agent: Agent,
        agent_cfg: AgentConfig,
        llm_cfg: LLMConfig,
        history_limit: int = 200,
    ) -> None:
        self.backend = backend
        self.agent = agent
        self.agent_cfg = agent_cfg
        self.llm_cfg = llm_cfg
        self.history: deque[ChatTurn] = deque(maxlen=history_limit)

    # ----- history -----------------------------------------------------------
    def add_user(self, text: str) -> ChatTurn:
        turn = ChatTurn("user", text)
        self.history.append(turn)
        return turn

    def add_assistant(self, text: str) -> ChatTurn:
        turn = ChatTurn("assistant", text)
        self.history.append(turn)
        return turn

    def clear(self) -> None:
        self.history.clear()

    # ----- prompt assembly ---------------------------------------------------
    def _system_with_context(self) -> str:
        steering = (self.agent_cfg.steering_notes or "").strip()
        transcript = self.agent.transcript_text()
        memory = self.agent.memory.render_for_prompt() if self.agent.memory else ""
        parts = [CHAT_SYSTEM_PROMPT.rstrip()]
        if memory:
            parts.append(
                "PERMANENT MEMORY (durable facts you have learned):\n" + memory
            )
        if steering:
            parts.append(
                "STEERING NOTES (operator, authoritative for game decisions):\n"
                f"{steering}"
            )
        if transcript:
            parts.append(
                "RECENT MUD TRANSCRIPT (snapshot, for your situational"
                " awareness):\n"
                "------------------------------------\n"
                f"{transcript}\n"
                "------------------------------------"
            )
        return "\n\n".join(parts)

    def build_messages(self) -> list[dict[str, str]]:
        msgs: list[dict[str, str]] = [
            {"role": "system", "content": self._system_with_context()}
        ]
        for turn in self.history:
            msgs.append({"role": turn.role, "content": turn.text})
        return msgs

    # ----- generation --------------------------------------------------------
    def stream_reply(self, on_delta: Callable[[str], None]) -> str:
        """Stream a chat reply; append it to history; return the full text."""
        messages = self.build_messages()
        parts: list[str] = []
        iterator: Iterator[str] = self.backend.stream_chat(
            messages, max_tokens=self.llm_cfg.max_chat_tokens
        )
        for delta in iterator:
            parts.append(delta)
            on_delta(delta)
        full = "".join(parts)
        self.add_assistant(full)
        return full
