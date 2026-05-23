"""Catalog of curated GGUF models tuned for an RTX 5090.

Each entry is a Hugging Face repo + filename. The download script and the
Settings dialog use this list. Add more freely.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelEntry:
    key: str
    label: str
    repo_id: str
    filename: str
    approx_vram_gb: float
    notes: str


CATALOG: tuple[ModelEntry, ...] = (
    ModelEntry(
        key="qwen3-14b-q6",
        label="Qwen3-14B Q6_K  (default, ~12 GB, fast + smart)",
        repo_id="unsloth/Qwen3-14B-GGUF",
        filename="Qwen3-14B-Q6_K.gguf",
        approx_vram_gb=12.0,
        notes="Best default balance for agentic text-game play on a 5090. "
              "Qwen3 emits <think>...</think> reasoning before its answer.",
    ),
    ModelEntry(
        key="qwen3-32b-q4",
        label="Qwen3-32B Q4_K_M  (~20 GB, smarter, slower)",
        repo_id="bartowski/Qwen_Qwen3-32B-GGUF",
        filename="Qwen_Qwen3-32B-Q4_K_M.gguf",
        approx_vram_gb=20.0,
        notes="Use when reasoning quality > latency.",
    ),
    ModelEntry(
        key="mistral-small-24b-q5",
        label="Mistral-Small-24B-Instruct-2501 Q5_K_M (~17 GB)",
        repo_id="bartowski/Mistral-Small-24B-Instruct-2501-GGUF",
        filename="Mistral-Small-24B-Instruct-2501-Q5_K_M.gguf",
        approx_vram_gb=17.0,
        notes="Strong non-Qwen middle ground.",
    ),
    ModelEntry(
        key="qwen3-8b-q6",
        label="Qwen3-8B Q6_K  (~7 GB, fastest with good reasoning)",
        repo_id="unsloth/Qwen3-8B-GGUF",
        filename="Qwen3-8B-Q6_K.gguf",
        approx_vram_gb=7.0,
        notes="Lowest latency option, still uses Qwen3 thinking mode.",
    ),
)

DEFAULT_KEY = "qwen3-14b-q6"


def by_key(key: str) -> ModelEntry | None:
    for e in CATALOG:
        if e.key == key:
            return e
    return None


def by_filename(filename: str) -> ModelEntry | None:
    for e in CATALOG:
        if e.filename == filename:
            return e
    return None
