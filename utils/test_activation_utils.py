#!/usr/bin/env python
"""Standalone sanity-check for utils.activation_utils.

Creates a tiny JSONL dataset with two conversations, then runs
extract_dataset_activations with all three variants in debug mode so you
can visually inspect the masking / pooling behaviour.

Usage:
    python test_activation_utils_debug.py
(The first run will download a small model ~30 MB.)
"""
from pathlib import Path
import json, tempfile
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from utils.activation_utils import extract_dataset_activations

# -------------------------------------------------------------
# 1. Prepare a minimal model & tokenizer (tiny for speed)
# -------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"  # Compact but instruction-tuned model

print("Loading tiny model …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID,
                                             torch_dtype=torch.float32).eval()

# -------------------------------------------------------------
# 2. Build a synthetic dataset with 2 rows of different length
# -------------------------------------------------------------
rows = [
    {
        "messages": [
            {"role": "user", "content": "Hello AAA"},
            {"role": "assistant", "content": "Response BBB."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Longer prompt CCC DDD EEE FFF."},
            {"role": "assistant", "content": "Another answer GGG HHH."}
        ]
    },
]

tmp_dir = tempfile.mkdtemp()
jsonl_path = Path(tmp_dir) / "mini.jsonl"
with jsonl_path.open("w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"Dataset written to {jsonl_path}")

# -------------------------------------------------------------
# 3. Run extraction with all variants and debug=True
# -------------------------------------------------------------
print("Running extract_dataset_activations in debug mode …")
acts = extract_dataset_activations(
    dataset_path=str(jsonl_path),
    tokenizer=tokenizer,
    model=model,
    layer=0,
    variant=("prompt_t1", "prompt_avg", "response_avg"),
    batch_size=2,
    device="cpu",
    debug=True,
    include_eos=True,
)

print("Returned type:", type(acts))
print("Shapes:")
for name, ten in zip(("prompt_t1", "prompt_avg", "response_avg"), acts):
    print(f"  {name}: {ten.shape}")

print("✅  Debug run complete. Inspect logs above for masking details.") 