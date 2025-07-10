# %% [markdown]
# # Feature Analysis: Diffing Base and Instruct
# 
# This notebook analyzes which SAE features increase in activations between base and chat models.

# %%
import csv
import json
import torch
import os
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from datasets import load_dataset
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download
from sae_lens import SAE
from tqdm.auto import tqdm
import random

# ✨ Determinism & precision controls
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

# Allow TF32 ( ≤ 1 e‑6 error ) – flip to False for a golden fp32 pass
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% [markdown]
# ## Configs

# %%
# =============================================================================
# MODEL SELECTION - Change this to switch between models
# =============================================================================
MODEL_TYPE = "llama"  # Options: "qwen" or "llama"
MODEL_VER = "base"
SAE_LAYER = 15
SAE_TRAINER = "32x"
N_PROMPTS = 10000

# =============================================================================
# OUTPUT FILE CONFIGURATION
# =============================================================================
OUTPUT_FILE = f"/workspace/results/4_diffing/{MODEL_TYPE}_trainer{SAE_TRAINER}_layer{SAE_LAYER}/{N_PROMPTS}_prompts/gpu_{MODEL_VER}.pt"
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# =============================================================================
# AUTO-CONFIGURED SETTINGS BASED ON MODEL TYPE
# =============================================================================

if MODEL_TYPE == "llama":
    BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B"
    CHAT_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    if MODEL_VER == "chat":
        MODEL_NAME = CHAT_MODEL_NAME
    elif MODEL_VER == "base":
        MODEL_NAME = BASE_MODEL_NAME
    else:
        raise ValueError(f"Unknown MODEL_VER: {MODEL_VER}. Use 'chat' or 'base'")

    SAE_RELEASE = "fnlp/Llama3_1-8B-Base-LXR-32x"
    ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>"
    TOKEN_OFFSETS = {"asst": -2, "endheader": -1, "newline": 0}
    SAE_BASE_PATH = "/workspace/sae/llama-3.1-8b/saes"
    
else:
    raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}. Use 'qwen' or 'llama'")

# =============================================================================
# DERIVED CONFIGURATIONS
# =============================================================================
SAE_PATH = f"{SAE_BASE_PATH}/resid_post_layer_{SAE_LAYER}/trainer_{SAE_TRAINER}"
LAYER_INDEX = SAE_LAYER

# Data paths
PROMPTS_HF = "lmsys/lmsys-chat-1m"
SEED = 42
PROMPTS_PATH = f"/workspace/data/{PROMPTS_HF.split('/')[-1]}/chat_{N_PROMPTS}.jsonl"
os.makedirs(os.path.dirname(PROMPTS_PATH), exist_ok=True)

# Processing parameters
# Fits in 80 GB H100 with bf16 activations
BATCH_SIZE = 32
MAX_LENGTH = 512

# =============================================================================
# SUMMARY
# =============================================================================
print(f"Configuration Summary:")
print(f"  Model: {MODEL_NAME}")
print(f"  SAE: {SAE_RELEASE}")
print(f"  SAE Layer: {SAE_LAYER}, Trainer: {SAE_TRAINER}")
print(f"  Available token types: {list(TOKEN_OFFSETS.keys())}")
print(f"  Assistant header: {ASSISTANT_HEADER}")
print(f"  Output file: {OUTPUT_FILE}")

# %% [markdown]
# ## Load Data

# %%
def load_lmsys_prompts(prompts_path: str, prompts_hf: str, n_prompts: int, seed: int) -> pd.DataFrame:
    # Check if prompts_path exists
    if os.path.exists(prompts_path):
        print(f"Prompts already exist at {prompts_path}")
        return pd.read_json(prompts_path, lines=True)
    else:
        print(f"Prompts do not exist at {prompts_path}. Loading from {prompts_hf}...")
        dataset = load_dataset(prompts_hf)
        dataset = dataset['train'].shuffle(seed=seed).select(range(n_prompts))
        df = dataset.to_pandas()

        # Extract the prompt from the first conversation item
        df['prompt'] = df['conversation'].apply(lambda x: x[0]['content'])

        # Only keep some columns
        df = df[['conversation_id', 'prompt', 'redacted', 'language']]

        # Save to .jsonl file
        df.to_json(prompts_path, orient='records', lines=True)
        return df

prompts_df = load_lmsys_prompts(PROMPTS_PATH, PROMPTS_HF, N_PROMPTS, SEED)
print(f"Loaded {prompts_df.shape[0]} prompts")
print(f"Prompt keys: {prompts_df.keys()}")


# %% [markdown]
# ## Load Model and SAE

# %%
# Load tokenizer (from chat model)
tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")

# %%
# Load model
device_map_value = device.index if device.type == 'cuda' and device.index is not None else str(device)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map={"": device_map_value}
)
model.eval()

print(f"Model loaded: {model.__class__.__name__}")
print(f"Model device: {next(model.parameters()).device}")

# %%
# Load SAE
def load_llamascope_sae(SAE_PATH, SAE_LAYER, SAE_TRAINER):
    """Load llamaScope SAE from Hugging Face."""

    # Check if SAE file exist locally
    ae_file_path = os.path.join(SAE_PATH, "sae_weights.safetensors")

    if os.path.exists(ae_file_path):
        print(f"✓ Found SAE files at: {os.path.dirname(ae_file_path)}")
        sae = SAE.load_from_disk(SAE_PATH)
    else:
        print(f"SAE not found locally, downloading from HF via sae_lens...")
        os.makedirs(os.path.dirname(SAE_PATH), exist_ok=True)

        sae, _, sparsity = SAE.from_pretrained(
            release = f"llama_scope_lxr_{SAE_TRAINER}", # see other options in sae_lens/pretrained_saes.yaml
            sae_id = f"l{SAE_LAYER}r_{SAE_TRAINER}", # won't always be a hook point
            device = "cuda"
        )
        sae.save_model(SAE_PATH, sparsity)

    return sae

sae = load_llamascope_sae(SAE_PATH, SAE_LAYER, SAE_TRAINER)
sae = sae.to(device)  # Move SAE to GPU
print(f"SAE loaded with {sae.cfg.d_sae} features")
print(f"SAE device: {next(sae.parameters()).device}")

# %% [markdown]
# ## Activation Extraction Functions

# %%
# ---------------------------------------------------------------------------
# 🐛 Replace StopForward hook with a clean partial‑forward wrapper
# ---------------------------------------------------------------------------

class UpToLayer(torch.nn.Module):
    """Runs the Transformer up to and including `layer_idx` and returns the
    layer‑normed hidden states (B, S, H)."""

    def __init__(self, model: torch.nn.Module, layer_idx: int):
        super().__init__()
        self.embed_tokens = model.model.embed_tokens
        self.layers = torch.nn.ModuleList(model.model.layers[: layer_idx + 1])
        self.norm = model.model.norm

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask, **kwargs)
        return self.norm(x)

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

# ---------------------------------------------------------------------------
# 🐛 Vectorised, deterministic extraction using UpToLayer
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_activations_and_metadata(prompts: List[str], layer_idx: int):
    partial = UpToLayer(model, layer_idx).eval()

    all_acts, all_meta, tmplts = [], [], []
    for i in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[i : i + BATCH_SIZE]
        messages = [{"role": "user", "content": p} for p in batch]
        formatted = [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages
        ]
        tmplts.extend(formatted)

        tok = tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        ).to(device)

        h = partial(**tok)  # (B, S, H)
        for j, _ in enumerate(batch):
            amask, ids = tok["attention_mask"][j], tok["input_ids"][j]
            pos = {
                t: find_assistant_position(ids, amask, ASSISTANT_HEADER, o, tokenizer, device)
                for t, o in TOKEN_OFFSETS.items()
            }
            all_acts.append(h[j].cpu())
            all_meta.append({"prompt_idx": i + j, "positions": pos, "attention_mask": amask.cpu(), "input_ids": ids.cpu()})

    max_len = max(a.shape[0] for a in all_acts)
    hid = all_acts[0].shape[1]
    padded = [torch.cat([a, a.new_zeros(max_len - a.size(0), hid)], 0) for a in all_acts]
    return torch.stack(padded), all_meta, tmplts

@torch.no_grad()
def extract_token_activations(full_activations: torch.Tensor, metadata: List[Dict]) -> Dict[str, torch.Tensor]:
    """Extract activations for specific token positions from full sequence activations."""
    results = {}
    
    # Initialize results for each token type
    for token_type in TOKEN_OFFSETS.keys():
        results[token_type] = []
    
    # Extract activations for each token type
    for i, meta in enumerate(metadata):
        for token_type, position in meta['positions'].items():
            # Extract activation at the specific position
            activation = full_activations[i, position, :]  # [hidden_dim]
            results[token_type].append(activation)
    
    # Convert lists to tensors
    for token_type in TOKEN_OFFSETS.keys():
        results[token_type] = torch.stack(results[token_type], dim=0)
    
    return results

print("Activation extraction functions defined")

# %% [markdown]
# ## Extract Activations

# %%
# Extract activations for all positions first, then extract specific token positions
print("Extracting activations for all positions...")
full_activations, metadata, formatted_prompts = extract_activations_and_metadata(prompts_df['prompt'].tolist(), LAYER_INDEX)
print(f"Full activations shape: {full_activations.shape}")

# Extract activations for all token types
print("\nExtracting activations for specific token positions...")
token_activations = extract_token_activations(full_activations, metadata)

for token_type, activations in token_activations.items():
    print(f"Token type '{token_type}' activations shape: {activations.shape}")

# %% [markdown]
# ## Apply SAE to Get Feature Activations

# %%
@torch.no_grad()
def get_sae_features_batched(activations: torch.Tensor) -> torch.Tensor:
    """Apply SAE to get feature activations with proper batching."""
    activations = activations.to(device)
    
    # Process in batches to avoid memory issues
    feature_activations = []
    
    for i in range(0, activations.shape[0], BATCH_SIZE):
        batch = activations[i:i+BATCH_SIZE]
        features = sae.encode(batch)  # [batch, num_features]
        feature_activations.append(features.cpu())
    
    return torch.cat(feature_activations, dim=0)

# Get SAE feature activations for specific token positions
print("Computing SAE features for specific token positions...")
token_features = {}

for token_type, activations in token_activations.items():
    print(f"Processing SAE features for token type '{token_type}'...")
    features = get_sae_features_batched(activations)
    token_features[token_type] = features
    print(f"Features shape for '{token_type}': {features.shape}")

print(f"\nCompleted SAE feature extraction for {len(token_features)} token types")



# ---------------------------------------------------------------------------
# 🐛 Device‑agnostic, vectorised stats & save
# ---------------------------------------------------------------------------

def compute_stats(t: torch.Tensor):
    t = t.float()
    n = t.shape[0]

    all_mean = t.mean(0, dtype=torch.float64)
    all_std = t.std(0, unbiased=False, dtype=torch.float64)
    max_v = t.max(0).values

    active = t > 0
    num_active = active.sum(0)
    sparsity = num_active / n

    inf = torch.finfo(t.dtype).max
    active_mean = (t * active).sum(0, dtype=torch.float64) / num_active.clamp(min=1)
    active_min = torch.where(active, t, inf).amin(0)
    diff2 = ((t - active_mean) ** 2) * active
    active_std = (diff2.sum(0, dtype=torch.float64) / num_active.clamp(min=2).sub_(1)).sqrt()

    p90 = torch.quantile(t, 0.9, dim=0, interpolation="linear")
    p95 = torch.quantile(t, 0.95, dim=0, interpolation="linear")
    p99 = torch.quantile(t, 0.99, dim=0, interpolation="linear")

    return {
        "all_mean": all_mean.cpu(),
        "all_std": all_std.cpu(),
        "active_mean": active_mean.cpu(),
        "active_min": active_min.cpu(),
        "active_std": active_std.cpu(),
        "max": max_v.cpu(),
        "num_active": num_active.cpu(),
        "sparsity": sparsity.cpu(),
        "p90": p90.cpu(),
        "p95": p95.cpu(),
        "p99": p99.cpu(),
    }


def save_as_pt(in_gpu: bool = True):
    src = f"{MODEL_TYPE}_trainer{SAE_TRAINER}_layer{SAE_LAYER}_{MODEL_VER}"
    out = {"metadata": {"source": src, "model_type": MODEL_TYPE, "model_ver": MODEL_VER, "sae_layer": SAE_LAYER, "sae_trainer": SAE_TRAINER, "num_prompts": N_PROMPTS, "token_types": list(TOKEN_OFFSETS)}}

    for tk in TOKEN_OFFSETS:
        feats = token_features[tk].to("cuda" if in_gpu else "cpu")
        out[tk] = compute_stats(feats)
        print(f"✓ {tk} done ({'GPU' if in_gpu else 'CPU'})")

    return out

results_dict = save_as_pt(in_gpu=True)

# %%
# Save results
print("Saving results...")

# Save as PyTorch file (much faster and more efficient)
pt_output_file = OUTPUT_FILE
torch.save(results_dict, pt_output_file)
print(f"PyTorch results saved to: {pt_output_file}")

# Show preview of PyTorch data structure
print(f"\nPyTorch file structure:")
print(f"Keys: {list(results_dict.keys())}")
print(f"Metadata: {results_dict['metadata']}")

for token_type in TOKEN_OFFSETS.keys():
    print(f"\n{token_type} statistics shapes:")
    for stat_name, tensor in results_dict[token_type].items():
        print(f"  {stat_name}: {tensor.shape}")
    
    # Show some sample statistics
    print(f"\n{token_type} sample statistics:")
    print(f"  all_mean - min: {results_dict[token_type]['all_mean'].min():.6f}, max: {results_dict[token_type]['all_mean'].max():.6f}")
    print(f"  sparsity - min: {results_dict[token_type]['sparsity'].min():.6f}, max: {results_dict[token_type]['sparsity'].max():.6f}")
    print(f"  num_active - min: {results_dict[token_type]['num_active'].min():.0f}, max: {results_dict[token_type]['num_active'].max():.0f}")

print(f"\n✓ Analysis complete! PyTorch file size is much smaller and loads faster than CSV.")


