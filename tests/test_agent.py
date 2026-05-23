from __future__ import annotations

from mudai.llm.agent import Agent
from mudai.llm.backend import LlamaBackend


def test_parse_decision_extracts_command_after_marker() -> None:
    raw = (
        "The room is dark. I should look around for exits first.\n"
        "Then pick a direction.\n"
        "COMMAND: look\n"
    )
    d = Agent.parse_decision(raw)
    assert d.command == "look"
    assert "dark" in d.reasoning


def test_parse_decision_strips_quotes_and_backticks() -> None:
    raw = "thinking...\nCOMMAND: `north`"
    assert Agent.parse_decision(raw).command == "north"
    raw2 = 'plan...\nCOMMAND: "say hello"'
    assert Agent.parse_decision(raw2).command == "say hello"


def test_parse_decision_case_insensitive_marker() -> None:
    raw = "reasoning\ncommand: kill rat"
    assert Agent.parse_decision(raw).command == "kill rat"


def test_parse_decision_fallback_when_no_marker() -> None:
    raw = "I have no idea what to do.\nlook"
    d = Agent.parse_decision(raw)
    assert d.command == "look"
    assert "no idea" in d.reasoning


def test_parse_decision_empty_when_no_content() -> None:
    d = Agent.parse_decision("")
    assert d.command == ""


def _make_agent() -> Agent:
    from pathlib import Path

    from mudai.config import AgentConfig, LLMConfig

    # Backend is never loaded; count_tokens falls back to chars/4.
    backend = LlamaBackend(Path("does-not-exist.gguf"), LLMConfig())
    return Agent(backend, AgentConfig(), LLMConfig(transcript_token_budget=50))


def test_transcript_trimmed_to_token_budget() -> None:
    agent = _make_agent()
    # Each entry is ~40 chars -> ~10 tokens. Add many.
    for i in range(30):
        agent.add_mud(f"line {i:02d} aaaaaaaaaaaaaaaaaaaaaaaaaa")
    rendered = agent._transcript_text()
    # Should keep only the most recent entries (newer numbers).
    assert "line 29" in rendered
    assert "line 00" not in rendered


def test_system_message_includes_steering() -> None:
    agent = _make_agent()
    agent.agent_cfg.steering_notes = "DO NOT ATTACK ANYONE"
    msg = agent._system_message()
    assert "DO NOT ATTACK ANYONE" in msg
    assert "STEERING NOTES" in msg


def test_build_messages_shape() -> None:
    agent = _make_agent()
    agent.add_mud("You are in a small room.")
    msgs = agent.build_messages()
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "small room" in msgs[1]["content"]
    assert "COMMAND:" in msgs[1]["content"]
