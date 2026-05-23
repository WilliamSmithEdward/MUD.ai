from __future__ import annotations

from mudai.llm.agent import Agent


def test_qwen3_think_block_extracted_and_stripped() -> None:
    raw = (
        "<think>The room has a rat. I should attack it carefully.</think>\n"
        "I'll attack the rat.\nCOMMAND: kill rat\n"
    )
    d = Agent.parse_decision(raw)
    assert d.command == "kill rat"
    assert "rat" in d.thinking
    assert "<think>" not in d.reasoning
    assert "attack the rat" in d.reasoning


def test_unterminated_think_block_still_recovered() -> None:
    # Model hit max_tokens mid-think.
    raw = "Some text.\n<think>I am pondering deeply and never finishing"
    d = Agent.parse_decision(raw)
    assert "pondering" in d.thinking
    # No COMMAND: marker => fallback picks last non-empty visible line.
    assert d.command == "Some text."


def test_no_think_block_works_as_before() -> None:
    raw = "reasoning\nCOMMAND: look"
    d = Agent.parse_decision(raw)
    assert d.command == "look"
    assert d.thinking == ""


def test_operator_note_appears_in_transcript() -> None:
    from pathlib import Path

    from mudai.config import AgentConfig, LLMConfig
    from mudai.llm.backend import LlamaBackend

    backend = LlamaBackend(Path("does-not-exist.gguf"), LLMConfig())
    agent = Agent(backend, AgentConfig(), LLMConfig(transcript_token_budget=500))
    agent.add_mud("You are in a room.")
    agent.add_operator_note("DANGER: hostile mob ahead, flee south.")
    rendered = agent._transcript_text()
    assert "[OPERATOR]" in rendered
    assert "flee south" in rendered
