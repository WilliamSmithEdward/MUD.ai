"""Convert approved trace rows in logs/traces/*.jsonl into an SFT dataset.

Output is a single JSONL where each line is `{"messages": [...]}` using the
exact same prompt format the live agent uses, so the LoRA learns to imitate the
same call signature.

Usage:
    python -m mudai.scripts.build_training_set
    python -m mudai.scripts.build_training_set --include-rejected --out custom.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import REPO_ROOT, TRACES_DIR


def _rebuild_user_block(transcript: list[dict[str, str]]) -> str:
    label_map = {
        "mud": "[MUD]",
        "you": "[YOU]",
        "agent": "[AGENT]",
        "operator": "[OPERATOR]",
    }
    lines = [
        f"{label_map.get(e.get('role', '?'), '[?]')} {e.get('text', '').rstrip()}"
        for e in transcript
    ]
    return (
        "Recent MUD transcript (oldest first):\n"
        "------------------------------------\n"
        f"{chr(10).join(lines)}\n"
        "------------------------------------\n\n"
        "Decide the next command. Reply with brief reasoning, then a final line:\n"
        "COMMAND: <text>\n"
    )


def build(
    traces_dir: Path, out_path: Path, include_rejected: bool = False
) -> int:
    n = 0
    with out_path.open("w", encoding="utf-8") as out:
        for fp in sorted(traces_dir.glob("*.jsonl")):
            for raw_line in fp.read_text("utf-8").splitlines():
                if not raw_line.strip():
                    continue
                try:
                    row = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                # Chat (side-conversation) rows ----------------------------
                if row.get("type") == "chat":
                    system = row.get("system") or ""
                    user = row.get("user") or ""
                    assistant = row.get("assistant") or ""
                    if not user.strip() or not assistant.strip():
                        continue
                    out.write(
                        json.dumps(
                            {
                                "messages": [
                                    {"role": "system", "content": system},
                                    {"role": "user", "content": user},
                                    {"role": "assistant", "content": assistant},
                                ]
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    n += 1
                    continue
                # Decision rows --------------------------------------------
                if not include_rejected and not row.get("approved"):
                    continue
                system = row.get("system") or ""
                transcript = row.get("transcript") or []
                user = _rebuild_user_block(transcript)
                # Prefer the model's exact raw response when available so the
                # trained model reproduces the same shape (incl. <think>).
                assistant = row.get("raw_response") or ""
                if not assistant:
                    reasoning = (row.get("reasoning") or "").strip()
                    cmd = (row.get("command") or "").strip()
                    assistant = (reasoning + "\n" if reasoning else "") + f"COMMAND: {cmd}"
                out.write(
                    json.dumps(
                        {
                            "messages": [
                                {"role": "system", "content": system},
                                {"role": "user", "content": user},
                                {"role": "assistant", "content": assistant},
                            ]
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traces", type=Path, default=TRACES_DIR)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "training.jsonl")
    ap.add_argument("--include-rejected", action="store_true")
    args = ap.parse_args()
    n = build(args.traces, args.out, args.include_rejected)
    print(f"wrote {n} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
