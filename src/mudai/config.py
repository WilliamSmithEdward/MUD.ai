"""Application configuration: persisted JSON, dataclass-style via pydantic."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
LOGS_DIR = REPO_ROOT / "logs"
TRACES_DIR = LOGS_DIR / "traces"
MEMORY_PATH = REPO_ROOT / "memory.jsonl"
CONFIG_PATH = REPO_ROOT / "config.local.json"

for _d in (MODELS_DIR, LOGS_DIR, TRACES_DIR):
    _d.mkdir(parents=True, exist_ok=True)


class LLMConfig(BaseModel):
    model_file: str = "Qwen3-14B-Q6_K.gguf"
    n_ctx: int = 40960            # context window tokens (Qwen3-14B native)
    n_gpu_layers: int = -1        # -1 = all on GPU
    n_threads: int = 8
    # Performance knobs (RTX 5090 friendly defaults)
    n_batch: int = 2048           # prompt-eval batch size
    n_ubatch: int = 512           # physical micro-batch
    flash_attn: bool = True       # fused attention kernel (faster, less VRAM)
    kv_cache_type: str = "q8_0"   # "f16" | "q8_0" | "q4_0" - halves KV bandwidth
    offload_kqv: bool = True      # keep KV cache on GPU
    temperature: float = 0.6
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.05
    # Qwen3 emits long <think> blocks before the COMMAND: line; budget must
    # accommodate that or the response will be truncated mid-think and
    # produce an empty parsed command (the loop then stalls on empties).
    max_decision_tokens: int = 2048
    max_chat_tokens: int = 1024
    # Recent transcript budget (tokens). Older lines are dropped from prompt.
    transcript_token_budget: int = 6000
    # Chat template name; "auto" = use template embedded in GGUF metadata.
    chat_template: str = "auto"


class MudConfig(BaseModel):
    host: str = "mud.arctic.org"
    port: int = 2700
    encoding: str = "utf-8"
    # Idle ms after last MUD output before the agent is allowed to act.
    # Acts as a debounce window: every new MUD chunk restarts the timer.
    decision_idle_ms: int = 2500
    auto_connect_on_start: bool = False


class AgentConfig(BaseModel):
    auto_send: bool = False
    auto_load_model_on_start: bool = False
    # If True (default), the agent schedules its own next decision after each
    # command send/reject so the loop is self-sustaining even when the MUD is
    # silent. If False, decisions only fire after MUD output (reactive mode).
    proactive_decisions: bool = True
    # Hard cap: even on auto-send, never issue commands faster than this (ms).
    min_command_interval_ms: int = 1500
    # Number of past successful decisions to inject into the prompt as in-context
    # examples. 0 disables experience replay. This is the primary mechanism by
    # which the agent gets better at playing the MUD as you accumulate sessions.
    experience_examples_k: int = 3
    # Auto-reflect every N approved decisions (LLM reads recent traces and
    # distills durable lessons into permanent memory). 0 disables auto-reflect.
    reflect_every_n_decisions: int = 30
    # Also reflect automatically when the MUD connection closes.
    reflect_on_disconnect: bool = True
    system_prompt: str = (
        "You are an autonomous player of a text-based multi-user dungeon (MUD).\n"
        "You read the most recent MUD output and decide what command(s) to send next.\n"
        "Rules:\n"
        "  1. Think briefly (1-4 short sentences) about the current room, your goals,"
        " threats, and what to do next.\n"
        "  2. Then on a NEW line output exactly: COMMAND: <the raw text to send>\n"
        "  3. You MAY chain a short sequence (up to 4) by emitting multiple"
        " 'COMMAND: <cmd>' lines, one per line, in the order you want them"
        " sent. Each is a separate command sent to the MUD with a small delay"
        " between them; do NOT combine multiple commands into one line.\n"
        "  4. Each command must be a single MUD command. No quotes, no markdown.\n"
        "  5. Only chain commands when the result of the first is predictable"
        " (e.g. 'open door' then 'north', or 'get all' then 'inventory'). If"
        " you need to SEE what happened before deciding, emit only ONE command.\n"
        "  6. If unsure, prefer safe info commands like 'look', 'score', 'inventory',"
        " or 'help <topic>'.\n"
        "  7. Never invent room contents that were not in the MUD output.\n"
        "  8. Follow any STEERING NOTES from the operator strictly; they override"
        " your own plans.\n"
        "  9. If you are genuinely uncertain and need operator guidance, you MAY"
        " add ONE extra line of the form 'QUESTION: <your question>' in addition"
        " to (not instead of) the COMMAND line. The operator will see it in a"
        " side-chat panel and may answer.\n"
        "  10. If you discover a DURABLE fact you should always remember (a rule,"
        " a name, a location, a danger), you MAY add a line of the form"
        " 'REMEMBER: <one concise fact>'. It will be saved to permanent memory"
        " and shown to you on every future decision. Use sparingly.\n"
    )
    steering_notes: str = (
        "Goals: explore the starting area, map exits, do not engage hostile mobs"
        " until I say so. Always 'look' after entering a new room."
    )


class AppConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    mud: MudConfig = Field(default_factory=MudConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)

    @classmethod
    def load(cls) -> "AppConfig":
        if CONFIG_PATH.exists():
            try:
                data: dict[str, Any] = json.loads(CONFIG_PATH.read_text("utf-8"))
                return cls.model_validate(data)
            except (json.JSONDecodeError, ValueError):
                pass
        cfg = cls()
        cfg.save()
        return cfg

    def save(self) -> None:
        CONFIG_PATH.write_text(
            json.dumps(self.model_dump(), indent=2), encoding="utf-8"
        )

    def model_path(self) -> Path:
        return MODELS_DIR / self.llm.model_file
