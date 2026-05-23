from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from mudai.config import AgentConfig, LLMConfig
from mudai.llm.agent import Agent
from mudai.llm.backend import LlamaBackend
from mudai.llm.chat import ChatSession


class _FakeBackend(LlamaBackend):
    """In-memory backend that yields a scripted reply, no real model needed."""

    def __init__(self, reply: str = "Hi there, what's the plan?") -> None:
        super().__init__(Path("does-not-exist.gguf"), LLMConfig())
        self._reply = reply
        self.captured_messages: list[dict[str, str]] | None = None
        self.captured_max_tokens: int | None = None

    @property
    def loaded(self) -> bool:  # type: ignore[override]
        return True

    def stream_chat(  # type: ignore[override]
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        self.captured_messages = messages
        self.captured_max_tokens = max_tokens
        for ch in self._reply:
            yield ch


def _make() -> tuple[Agent, ChatSession, _FakeBackend]:
    backend = _FakeBackend()
    agent = Agent(backend, AgentConfig(), LLMConfig(transcript_token_budget=500))
    chat = ChatSession(backend, agent, agent.agent_cfg, agent.llm_cfg)
    return agent, chat, backend


def test_chat_system_includes_steering_and_transcript() -> None:
    agent, chat, _ = _make()
    agent.agent_cfg.steering_notes = "BE CAREFUL OF TROLLS"
    agent.add_mud("You see a troll.")
    sys = chat._system_with_context()
    assert "BE CAREFUL OF TROLLS" in sys
    assert "troll" in sys.lower()
    # Chat system prompt forbids the COMMAND marker (it's only for the game loop).
    assert "No COMMAND" in sys


def test_chat_streams_and_records_assistant() -> None:
    _, chat, backend = _make()
    chat.add_user("What should we do next?")
    collected: list[str] = []
    full = chat.stream_reply(collected.append)
    assert "".join(collected) == full
    assert full == backend._reply
    # History now has user + assistant.
    roles = [t.role for t in chat.history]
    assert roles == ["user", "assistant"]
    assert chat.history[-1].text == full


def test_chat_uses_max_chat_tokens_not_decision_tokens() -> None:
    _, chat, backend = _make()
    chat.llm_cfg.max_chat_tokens = 999
    chat.llm_cfg.max_decision_tokens = 32
    chat.add_user("hi")
    chat.stream_reply(lambda _d: None)
    assert backend.captured_max_tokens == 999


def test_chat_messages_alternate_after_history() -> None:
    _, chat, backend = _make()
    chat.add_user("first")
    chat.add_assistant("ok")
    chat.add_user("second")
    chat.stream_reply(lambda _d: None)
    msgs = backend.captured_messages
    assert msgs is not None
    assert msgs[0]["role"] == "system"
    assert [m["role"] for m in msgs[1:]] == ["user", "assistant", "user"]
