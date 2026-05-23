# MUD.ai

A PyQt6 desktop app that lets a local LLM (running on your GPU via `llama-cpp-python`)
play a Telnet MUD autonomously, with a human-in-the-loop steering pane and full
training-data logging for later LoRA fine-tuning.

Hardware target: RTX 5090 (32 GB VRAM) + Ryzen 9800X3D + 32 GB RAM.
Default model: **Qwen3-14B Instruct, Q6_K GGUF** (fits comfortably, ~60-90 tok/s,
strong reasoning for text-game planning). Other models are hot-swappable from the UI.

Test target: `mud.arctic.org:2700`.

---

## 1. Install

```powershell
# from repo root
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip wheel

# llama-cpp-python with CUDA 12.4 prebuilt wheels (works for 5090 / Blackwell via PTX JIT).
# If a CUDA 12.6 wheel index is available when you read this, use that instead.
pip install llama-cpp-python ^
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

pip install -r requirements.txt
```

If `llama-cpp-python` fails to load the GPU at runtime, rebuild from source with
CUDA toolkit installed:

```powershell
$env:CMAKE_ARGS="-DGGML_CUDA=on"
pip install --no-binary llama-cpp-python --force-reinstall llama-cpp-python
```

## 2. Download a model

```powershell
python -m mudai.scripts.download_model           # downloads default Qwen3-14B Q6_K
python -m mudai.scripts.download_model --list    # show known models
python -m mudai.scripts.download_model --model qwen3-32b-q4
```

Models land in `./models/`. The app's Settings dialog also has a "Download / Switch"
button so you do not have to leave the GUI.

## 3. Run

```powershell
python -m mudai
```

Default connection: `mud.arctic.org:2700`. Change in Settings.

## 4. Layout

- **Left pane**: live MUD output (ANSI colored, monospace).
- **Right top**: LLM reasoning stream (chain-of-thought from the model).
- **Right middle**: Steering notes - free-form text you edit any time; injected
  into the system prompt on every decision. This is your "context steering".
- **Bottom bar**: proposed command + Send / Edit / Reject. Toggle Auto-send for
  full autonomy (default OFF).
- **Settings**: host/port, model file, context window (n_ctx), GPU layers,
  temperature, top_p, max decision tokens, autonomy default, polling cadence.

## 5. Training data

Every decision is logged as one JSONL row in `logs/traces/<session>.jsonl`:

```json
{
  "ts": "2026-05-23T12:34:56Z",
  "system": "...",
  "steering": "...",
  "transcript": [{"role": "mud", "text": "..."}, ...],
  "reasoning": "...",
  "command": "north",
  "approved": true,
  "outcome": "You walk north.\n..."
}
```

Curate with any JSONL editor, then train a LoRA later (see
`scripts/train_lora.md`). The app can hot-load a GGUF that has the LoRA merged in.

## 6. Safety

- Default autonomy is **manual approval**. Auto-send must be enabled per session
  and shows a large STOP button while active.
- All sent/received bytes are logged.
- Telnet only; no auto-credentials. The app prompts you for character login.
