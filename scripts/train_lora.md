# Fine-tuning your MUD agent

## 1. Collect data

Play a few sessions with the app (manual mode or auto with steering). Every
decision lands in `logs/traces/session_*.jsonl`. Each row already has the exact
system prompt, steering notes, transcript window, the model's reasoning, the
final command, whether you approved it, and the MUD's outcome.

Curate: open the JSONL, delete rows where the command was rejected or the
outcome was bad; lightly edit reasoning/commands for the rows you want as
"golden" examples. A small high-quality set (a few hundred rows) beats a noisy
large set.

## 2. Convert to a training-ready dataset

Make a `training.jsonl` where each row is:

```json
{
  "messages": [
    {"role": "system",    "content": "<system + steering>"},
    {"role": "user",      "content": "Recent MUD transcript ...\nCOMMAND request"},
    {"role": "assistant", "content": "<reasoning>\nCOMMAND: <command>"}
  ]
}
```

A trivial converter:

```python
import json, pathlib
out = pathlib.Path("training.jsonl").open("w", encoding="utf-8")
for fp in pathlib.Path("logs/traces").glob("*.jsonl"):
    for line in fp.read_text("utf-8").splitlines():
        row = json.loads(line)
        if not row.get("approved"):
            continue
        sys_msg = row["system"]
        # Rebuild the user block exactly as the app does:
        transcript_text = "\n".join(
            f"[{e['role'].upper()}] {e['text'].rstrip()}" for e in row["transcript"]
        )
        user_msg = (
            "Recent MUD transcript (oldest first):\n"
            "------------------------------------\n"
            f"{transcript_text}\n"
            "------------------------------------\n\n"
            "Decide the next command. Reply with brief reasoning, then a final line:\n"
            "COMMAND: <text>\n"
        )
        assistant_msg = row.get("raw_response") or (
            (row.get("reasoning", "") + "\n" if row.get("reasoning") else "")
            + f"COMMAND: {row['command']}"
        )
        out.write(json.dumps({"messages": [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]}) + "\n")
out.close()
```

## 3. Train a LoRA with Unsloth (WSL2 recommended on Windows)

```bash
# inside WSL2 Ubuntu
pip install "unsloth[cu124] @ git+https://github.com/unslothai/unsloth.git"
pip install trl datasets accelerate

python - <<'PY'
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

model, tok = FastLanguageModel.from_pretrained(
    "Qwen/Qwen3-14B-Instruct",      # or whichever HF id matches your GGUF
    max_seq_length=16384,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model, r=32, lora_alpha=32, lora_dropout=0.0,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
)

ds = load_dataset("json", data_files="training.jsonl", split="train")

trainer = SFTTrainer(
    model=model, tokenizer=tok,
    train_dataset=ds,
    args=SFTConfig(
        output_dir="mudai-lora",
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        num_train_epochs=2, learning_rate=2e-4,
        bf16=True, logging_steps=10, save_steps=200,
        max_seq_length=16384, packing=False,
    ),
)
trainer.train()

# Merge + export GGUF for llama.cpp:
model.save_pretrained_merged("mudai-merged", tok, save_method="merged_16bit")
# Then quantize with llama.cpp's convert + quantize tools to Q6_K.
PY
```

## 4. Drop the new GGUF into `./models/` and select it in Settings.

The app reloads the backend automatically when you change the model file.
