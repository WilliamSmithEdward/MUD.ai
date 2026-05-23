"""LLM backend wrapping llama-cpp-python with streaming chat completion."""
from __future__ import annotations

import os
import site
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

from ..config import LLMConfig


def _register_cuda_dll_dirs() -> None:
    """On Windows, register pip-installed NVIDIA runtime DLL folders so the
    CUDA build of llama.dll can find cudart/cublas/nvrtc at load time.

    llama_cpp loads its DLLs with ``winmode=RTLD_GLOBAL``, which ignores
    ``add_dll_directory`` registrations; we also prepend to ``PATH``.
    """
    if sys.platform != "win32":
        return
    candidates: list[Path] = []
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        nv = Path(sp) / "nvidia"
        if nv.is_dir():
            candidates.extend(p for p in nv.glob("*/bin") if p.is_dir())
    seen: set[str] = set()
    for d in candidates:
        key = str(d).lower()
        if key in seen:
            continue
        seen.add(key)
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(str(d))
            except (OSError, FileNotFoundError):
                pass
        os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")


_register_cuda_dll_dirs()


class LlamaBackend:
    """Thin wrapper. Imports llama_cpp lazily so the GUI can boot without GPU.

    A single Llama instance is NOT safe to call concurrently. The decision loop
    and the side-chat session share this backend, so all completions are
    serialized through ``_call_lock``. Calls queue; the GUI thread never blocks.
    """

    def __init__(self, model_path: Path, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self.model_path = model_path
        self._llm: Any | None = None
        self._call_lock = threading.Lock()

    def load(self) -> None:
        if self._llm is not None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}. "
                "Run `python -m mudai.scripts.download_model` first."
            )
        # Import here so the GUI can launch even if llama-cpp-python is missing.
        from llama_cpp import Llama  # type: ignore[import-not-found]

        chat_format: str | None = None
        if self.cfg.chat_template != "auto":
            chat_format = self.cfg.chat_template

        # Map KV cache dtype string -> GGML type id.
        # 0 = f32, 1 = f16, 8 = q8_0, 2 = q4_0
        kv_type_map = {"f16": 1, "q8_0": 8, "q4_0": 2, "f32": 0}
        type_kv = kv_type_map.get(self.cfg.kv_cache_type, 1)

        self._llm = Llama(
            model_path=str(self.model_path),
            n_ctx=self.cfg.n_ctx,
            n_gpu_layers=self.cfg.n_gpu_layers,
            n_threads=self.cfg.n_threads,
            n_batch=self.cfg.n_batch,
            n_ubatch=self.cfg.n_ubatch,
            flash_attn=self.cfg.flash_attn,
            type_k=type_kv,
            type_v=type_kv,
            offload_kqv=self.cfg.offload_kqv,
            chat_format=chat_format,
            verbose=False,
        )

    def unload(self) -> None:
        self._llm = None

    @property
    def loaded(self) -> bool:
        return self._llm is not None

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Yield text deltas for a chat completion. Caller assembles the full string.

        Serialized with ``_call_lock`` so the game-loop and side-chat never call
        the underlying Llama instance concurrently.
        """
        if self._llm is None:
            self.load()
        assert self._llm is not None
        with self._call_lock:
            stream = self._llm.create_chat_completion(
                messages=cast(Any, messages),
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                top_k=self.cfg.top_k,
                repeat_penalty=self.cfg.repeat_penalty,
                max_tokens=max_tokens or self.cfg.max_decision_tokens,
                stream=True,
            )
            for chunk in stream:  # type: ignore[union-attr]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                text = delta.get("content")
                if text:
                    yield str(text)

    def count_tokens(self, text: str) -> int:
        """Approximate token count. Loads the model if necessary; falls back to chars/4."""
        if self._llm is None:
            return max(1, len(text) // 4)
        try:
            toks = self._llm.tokenize(text.encode("utf-8"), add_bos=False)
            return len(toks)
        except Exception:
            return max(1, len(text) // 4)
