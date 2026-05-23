"""Download a GGUF model into ./models/ using huggingface_hub.

Usage:
    python -m mudai.scripts.download_model              # default model
    python -m mudai.scripts.download_model --list
    python -m mudai.scripts.download_model --model qwen3-32b-q4
    python -m mudai.scripts.download_model --repo bartowski/Foo-GGUF --file Foo-Q5.gguf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..config import MODELS_DIR
from ..llm import models_catalog


def download(repo_id: str, filename: str, dest_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id} :: {filename} -> {dest_dir}")
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(dest_dir),
    )
    print(f"OK: {path}")
    return Path(path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Download a GGUF model into ./models/")
    ap.add_argument("--list", action="store_true", help="List known models and exit.")
    ap.add_argument("--model", default=models_catalog.DEFAULT_KEY,
                    help="Catalog key (see --list). Default: %(default)s.")
    ap.add_argument("--repo", default=None,
                    help="Override: full HF repo id, e.g. bartowski/Foo-GGUF.")
    ap.add_argument("--file", default=None,
                    help="Override: filename inside the repo.")
    args = ap.parse_args(argv)

    if args.list:
        for e in models_catalog.CATALOG:
            mark = " (default)" if e.key == models_catalog.DEFAULT_KEY else ""
            print(f"  {e.key}{mark}")
            print(f"      {e.label}")
            print(f"      repo:   {e.repo_id}")
            print(f"      file:   {e.filename}")
            print(f"      ~VRAM:  {e.approx_vram_gb} GB")
            print(f"      notes:  {e.notes}")
        return 0

    if args.repo and args.file:
        repo_id, filename = args.repo, args.file
    else:
        entry = models_catalog.by_key(args.model)
        if entry is None:
            print(f"Unknown model key: {args.model!r}. Use --list.", file=sys.stderr)
            return 2
        repo_id, filename = entry.repo_id, entry.filename

    try:
        download(repo_id, filename, MODELS_DIR)
    except Exception as e:
        print(f"Download failed: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
