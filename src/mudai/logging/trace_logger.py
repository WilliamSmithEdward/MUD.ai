"""JSONL trace logger: one row per agent decision, suitable for LoRA training."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import TRACES_DIR


class TraceLogger:
    def __init__(self, session_name: str | None = None) -> None:
        if session_name is None:
            session_name = datetime.now().strftime("session_%Y%m%d_%H%M%S")
        self.path: Path = TRACES_DIR / f"{session_name}.jsonl"
        # Touch so the file exists even if no decisions are written.
        self.path.touch(exist_ok=True)

    def log(self, row: dict[str, Any]) -> None:
        row = {"ts": datetime.now(timezone.utc).isoformat(), **row}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
