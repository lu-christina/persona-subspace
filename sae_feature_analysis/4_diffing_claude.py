#!/usr/bin/env python
"""
4_diffing_claude.py – Optimized version of 4_diffing.py with batched SAE encoding
Compare SAE‑encoded feature statistics for **Llama‑3.1‑8B** *chat* vs *base* models
with performance improvements: batched SAE encoding and reduced GPU transfers.
"""

# -----------------------------------------------------------------------------
# 0 · Imports & reproducibility -------------------------------------------------
# -----------------------------------------------------------------------------

from __future__ import annotations

import os, random, time, json
from pathlib import Path
from typing import Dict, List, Tuple

import torch, numpy as np, pandas as pd
from tqdm.auto import tqdm
from datasets import load_dataset  # HF datasets
from transformers import AutoTokenizer, AutoModelForCausalLM

# SAE loader (sae‑lens)
from sae_lens import SAE  # type: ignore

# ---------------- Determinism --------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------------------------------------------
# 1 · User configuration (unchanged) -------------------------------------------
# -----------------------------------------------------------------------------

MODEL_TYPE = "llama"   # Options: "qwen" or "llama"
SAE_LAYER  = 15
SAE_TRAINER = "32x"
N_PROMPTS = 10_000

# Token‑type offsets exactly as before
TOKEN_OFFSETS: Dict[str, int] = {"asst": -2, "endheader": -1, "newline": 0}

# Data & model paths (network volume)
SAE_BASE_PATH = "/workspace/sae/llama-3.1-8b/saes"
PROMPTS_HF    = "lmsys/lmsys-chat-1m"
PROMPTS_PATH  = f"/workspace/data/{PROMPTS_HF.split('/')[-1]}/chat_{N_PROMPTS}.jsonl"

# Ensure directories exist
Path(PROMPTS_PATH).parent.mkdir(parents=True, exist_ok=True)

# Model names
BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B"
CHAT_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

MODEL_VARIANTS = {"chat": CHAT_MODEL_NAME, "base": BASE_MODEL_NAME}

# Processing params
BATCH_SIZE  = 32   # fits in 80 GB H100 with bf16 activations
MAX_LENGTH  = 512

# SAE release mapping (unchanged)
SAE_RELEASE = "fnlp/Llama3_1-8B-Base-LXR-32x"
SAE_PATH    = f"{SAE_BASE_PATH}/resid_post_layer_{SAE_LAYER}/trainer_{SAE_TRAINER}"
LAYER_INDEX = SAE_LAYER

# -----------------------------------------------------------------------------
# 2 · Prompt loader (auto‑download) -------------------------------------------
# -----------------------------------------------------------------------------

def load_lmsys_prompts(prompts_path: str, prompts_hf: str, n_prompts: int, seed: int) -> pd.DataFrame:
    """Load or download lmsys-chat‑1m prompts to local .jsonl then return DataFrame."""
    if os.path.exists(prompts_path):
        print(f"✓ Prompts found at {prompts_path}")
        return pd.read_json(prompts_path, lines=True)

    print(f"⏬ Downloading {n_prompts} prompts from {prompts_hf} …")
    ds = load_dataset(prompts_hf)["train"].shuffle(seed=seed).select(range(n_prompts))
    df = ds.to_pandas()
    df["prompt"] = df["conversation"].apply(lambda x: x[0]["content"])
    df = df[["conversation_id", "prompt", "redacted", "language"]]
    df.to_json(prompts_path, orient="records", lines=True)
    print(f"✓ Saved prompts → {prompts_path}")
    return df

prompts_df = load_lmsys_prompts(PROMPTS_PATH, PROMPTS_HF, N_PROMPTS, SEED)
print(f"Loaded {prompts_df.shape[0]} prompts")

# -----------------------------------------------------------------------------
# 3 · SAE loader (auto‑download) ----------------------------------------------
# -----------------------------------------------------------------------------

def load_llamascope_sae(sae_path: str, sae_layer: int, sae_trainer: str) -> SAE:
    """Load SAE from disk or pull from HF using sae‑lens."""
    ae_file = Path(sae_path) / "sae_weights.safetensors"
    if ae_file.exists():
        print(f"✓ SAE cache hit at {ae_file.parent}")
        return SAE.load_from_disk(sae_path)

    print("⏬ Downloading SAE via sae_lens…")
    ae_file.parent.mkdir(parents=True, exist_ok=True)
    sae, _, sparsity = SAE.from_pretrained(
        release=f"llama_scope_lxr_{sae_trainer}", sae_id=f"l{sae_layer}r_{sae_trainer}", device="cuda"
    )
    sae.save_model(sae_path, sparsity)
    return sae

sae = load_llamascope_sae(SAE_PATH, SAE_LAYER, SAE_TRAINER).to(DEVICE).eval()
print(f"SAE loaded with {sae.cfg.d_sae} features → device {DEVICE}")

# -----------------------------------------------------------------------------
# 4 · Tokenizer (always chat variant) -----------------------------------------
# -----------------------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL_NAME, use_fast=True, padding_side="left")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# -----------------------------------------------------------------------------
# 5 · Utility: hook-based activation extraction --------------------------------
# -----------------------------------------------------------------------------

class StopForward(Exception):
    """Exception to stop forward pass after target layer."""
    pass

ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>"

def find_assistant_position(input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                          assistant_header: str, token_offset: int, tokenizer, device) -> int:
    """Find the position of the assistant token based on the given offset."""
    # Find assistant header position
    assistant_tokens = tokenizer.encode(assistant_header, add_special_tokens=False)
    
    # Find where assistant section starts
    assistant_pos = None
    for k in range(len(input_ids) - len(assistant_tokens) + 1):
        if torch.equal(input_ids[k:k+len(assistant_tokens)], torch.tensor(assistant_tokens).to(device)):
            assistant_pos = k + len(assistant_tokens) + token_offset
            break
    
    if assistant_pos is None:
        # Fallback to last non-padding token
        assistant_pos = attention_mask.sum().item() - 1
    
    # Ensure position is within bounds
    max_pos = attention_mask.sum().item() - 1
    assistant_pos = min(assistant_pos, max_pos)
    assistant_pos = max(assistant_pos, 0)
    
    return int(assistant_pos)

# -----------------------------------------------------------------------------
# 6 · Vectorised per‑feature statistics ---------------------------------------
# -----------------------------------------------------------------------------

def compute_stats(x: torch.Tensor) -> Dict[str, torch.Tensor]:
    x = x.float()  # fp32 for safety
    n = x.shape[0]

    all_mean = x.mean(0).to(torch.float64)
    all_std  = x.std(0, unbiased=False).to(torch.float64)
    max_v    = x.max(0).values

    active = x > 0
    num_active = active.sum(0)
    sparsity = num_active / n

    inf = torch.finfo(x.dtype).max
    active_mean = (x * active).sum(0).to(torch.float64) / num_active.clamp(min=1)
    active_min  = torch.where(active, x, inf).amin(0)
    var_active  = ((x - active_mean) ** 2) * active
    active_std  = (var_active.sum(0).to(torch.float64) / num_active.clamp(min=2).sub_(1)).sqrt()

    p90, p95, p99 = [torch.quantile(x, q, dim=0, interpolation="linear") for q in (0.9, 0.95, 0.99)]

    return {k: v.cpu() for k, v in {
        "all_mean": all_mean, "all_std": all_std, "max": max_v, "num_active": num_active,
        "sparsity": sparsity, "active_mean": active_mean, "active_min": active_min,
        "active_std": active_std, "p90": p90, "p95": p95, "p99": p99,
    }.items()}

# -----------------------------------------------------------------------------
# 7 · Optimized SAE encoding function -----------------------------------------
# -----------------------------------------------------------------------------

def batch_encode_sae(feats: Dict[str, List[torch.Tensor]], sae: SAE, device: str) -> Dict[str, torch.Tensor]:
    """Encode all activations in a single batch for efficiency."""
    print("🚀 Batching SAE encoding across all token types...")
    
    # Collect all activations and track token type boundaries
    all_activations = []
    token_type_indices = {}
    start_idx = 0
    
    for tk, lst in feats.items():
        token_type_indices[tk] = (start_idx, start_idx + len(lst))
        all_activations.extend(lst)
        start_idx += len(lst)
        print(f"  • {tk}: {len(lst)} activations")
    
    total_activations = len(all_activations)
    print(f"  • Total: {total_activations} activations")
    
    # Single large SAE encoding
    print("  • Stacking activations...")
    all_acts = torch.stack(all_activations).to(device)
    print(f"  • Shape: {all_acts.shape}")
    
    print("  • Running SAE encoding...")
    with torch.no_grad():
        all_encoded = sae(all_acts.to(torch.bfloat16)).cpu()
    
    print("  • Computing statistics per token type...")
    # Split back by token type and compute stats
    stats_out = {}
    for tk, (start, end) in token_type_indices.items():
        token_encoded = all_encoded[start:end]
        stats_out[tk] = compute_stats(token_encoded)
        print(f"    • {tk}: {token_encoded.shape}")
    
    return stats_out

# -----------------------------------------------------------------------------
# 8 · Core pipeline for one MODEL_VER (optimized) ----------------------------
# -----------------------------------------------------------------------------

def process_variant(model_ver: str):
    # Dynamic pieces that depend on MODEL_VER
    model_name = CHAT_MODEL_NAME if model_ver == "chat" else BASE_MODEL_NAME
    output_file = f"/workspace/results/4_diffing/{MODEL_TYPE}_trainer{SAE_TRAINER}_layer{SAE_LAYER}/{N_PROMPTS}_prompts/gpu_{model_ver}_claude.pt"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {model_ver.upper()} | loading {model_name} ===")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    
    # Get target layer for hooking
    target_layer = model.model.layers[LAYER_INDEX]

    # Per‑token‑type feature tensors
    feats: Dict[str, List[torch.Tensor]] = {k: [] for k in TOKEN_OFFSETS}

    print("🔄 Extracting activations...")
    for i in tqdm(range(0, len(prompts_df), BATCH_SIZE), desc="Batches"):
        batch = prompts_df.prompt.iloc[i : i + BATCH_SIZE].tolist()
        # Each prompt needs a *list* of messages; pass one‑element list to the template
        templated = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in batch
        ]
        tok = tokenizer(templated, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH).to(DEVICE)
        
        # Hook to capture activations
        activations = None
        
        def hook_fn(module, input, output):
            nonlocal activations
            activations = output[0] if isinstance(output, tuple) else output
            raise StopForward()
        
        # Register hook
        handle = target_layer.register_forward_hook(hook_fn)
        
        try:
            with torch.no_grad():
                _ = model(**tok)
        except StopForward:
            pass
        finally:
            handle.remove()

        # Extract activations for each token type
        for j in range(len(batch)):
            ids, mask = tok["input_ids"][j], tok["attention_mask"][j]
            for tk, offs in TOKEN_OFFSETS.items():
                pos = find_assistant_position(ids, mask, ASSISTANT_HEADER, offs, tokenizer, DEVICE)
                feats[tk].append(activations[j, pos].cpu())

    # Optimized: batch SAE encoding across all token types
    stats_out = batch_encode_sae(feats, sae, DEVICE)

    torch.save(stats_out, output_file, pickle_protocol=5, _use_new_zipfile_serialization=False)
    print(f"✓ Saved ⇒ {output_file}")

# -----------------------------------------------------------------------------
# 9 · Main loop ----------------------------------------------------------------
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()
    for ver in ("chat", "base"):
        process_variant(ver)
    print(f"🎉 All done in {time.time() - t0:,.1f} s")