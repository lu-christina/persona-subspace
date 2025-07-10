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
OUTPUT_FILE = f"/workspace/results/4_diffing/{MODEL_TYPE}_trainer{SAE_TRAINER}_layer{SAE_LAYER}/{N_PROMPTS}_prompts/{MODEL_VER}.pt"
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# =============================================================================
# FEATURE DASHBOARD URL - Global variable for links
# =============================================================================
LLAMA_BASE_URL = f"https://www.neuronpedia.org/llama3.1-8b/{SAE_LAYER}-llamascope-res-131k/"

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
    BASE_URL = LLAMA_BASE_URL
    
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
BATCH_SIZE = 8
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
class StopForward(Exception):
    """Exception to stop forward pass after target layer."""
    pass

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

@torch.no_grad()
def extract_activations_and_metadata(prompts: List[str], layer_idx: int) -> Tuple[torch.Tensor, List[Dict], List[str]]:
    """Extract activations and prepare metadata for all prompts."""
    all_activations = []
    all_metadata = []
    formatted_prompts_list = []
    
    # Get target layer
    target_layer = model.model.layers[layer_idx]
    
    # Process in batches
    for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="Processing batches"):
        batch_prompts = prompts[i:i+BATCH_SIZE]
        
        # Format prompts as chat messages
        formatted_prompts = []
        for prompt in batch_prompts:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            formatted_prompts.append(formatted_prompt)
        
        formatted_prompts_list.extend(formatted_prompts)
        
        # Tokenize batch
        batch_inputs = tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH
        )
        
        # Move to device
        batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
        
        # Hook to capture activations
        activations = None
        
        def hook_fn(module, input, output):
            nonlocal activations
            activations = output[0] if isinstance(output, tuple) else output
            raise StopForward()
        
        # Register hook
        handle = target_layer.register_forward_hook(hook_fn)
        
        try:
            _ = model(**batch_inputs)
        except StopForward:
            pass
        finally:
            handle.remove()
        
        # For each prompt in the batch, calculate positions for all token types
        for j, formatted_prompt in enumerate(formatted_prompts):
            attention_mask = batch_inputs["attention_mask"][j]
            input_ids = batch_inputs["input_ids"][j]
            
            # Calculate positions for all token types
            positions = {}
            for token_type, token_offset in TOKEN_OFFSETS.items():
                positions[token_type] = find_assistant_position(
                    input_ids, attention_mask, ASSISTANT_HEADER, token_offset, tokenizer, device
                )
            
            # Store the full activation sequence and metadata
            all_activations.append(activations[j].cpu())  # [seq_len, hidden_dim]
            all_metadata.append({
                'prompt_idx': i + j,
                'positions': positions,
                'attention_mask': attention_mask.cpu(),
                'input_ids': input_ids.cpu()
            })
    
    # Find the maximum sequence length across all activations
    max_seq_len = max(act.shape[0] for act in all_activations)
    hidden_dim = all_activations[0].shape[1]
    
    # Pad all activations to the same length
    padded_activations = []
    for act in all_activations:
        if act.shape[0] < max_seq_len:
            padding = torch.zeros(max_seq_len - act.shape[0], hidden_dim)
            padded_act = torch.cat([act, padding], dim=0)
        else:
            padded_act = act
        padded_activations.append(padded_act)
    
    return torch.stack(padded_activations, dim=0), all_metadata, formatted_prompts_list

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

@torch.no_grad()
def get_sae_features_all_positions(full_activations: torch.Tensor) -> torch.Tensor:
    """Pre-compute SAE features for ALL positions at once for optimization."""
    print(f"Processing {full_activations.shape[0]} prompts with max {full_activations.shape[1]} tokens each...")
    
    # Reshape to [total_positions, hidden_dim]
    total_positions = full_activations.shape[0] * full_activations.shape[1]
    reshaped_activations = full_activations.view(total_positions, -1)
    
    # Apply SAE to all positions
    full_sae_features = get_sae_features_batched(reshaped_activations)
    
    # Reshape back to [num_prompts, seq_len, num_features]
    full_sae_features = full_sae_features.view(full_activations.shape[0], full_activations.shape[1], -1)
    
    print(f"Full SAE features shape: {full_sae_features.shape}")
    print(f"✓ SAE features pre-computed for all positions")
    
    return full_sae_features

# Get SAE feature activations for specific token positions
print("Computing SAE features for specific token positions...")
token_features = {}

for token_type, activations in token_activations.items():
    print(f"Processing SAE features for token type '{token_type}'...")
    features = get_sae_features_batched(activations)
    token_features[token_type] = features
    print(f"Features shape for '{token_type}': {features.shape}")

print(f"\nCompleted SAE feature extraction for {len(token_features)} token types")

# Uncomment the lines below if you need all-position features for optimization
# print("\nOptimization: Pre-computing SAE features for all positions...")
# full_sae_features = get_sae_features_all_positions(full_activations)

# %% [markdown]
# ## Analysis and Save Results

# %%
def save_as_csv():
    """Save results as CSV format (slower but human readable)"""
    csv_results = []
    source_name = f"{MODEL_TYPE}_trainer{SAE_TRAINER}_layer{SAE_LAYER}_{MODEL_VER}"
    
    print(f"Processing results for CSV format, source: {source_name}")
    
    # Process each token type
    for token_type in TOKEN_OFFSETS.keys():
        print(f"\nProcessing token type: {token_type}")
        
        # Get features tensor for this token type: [num_prompts, num_features]
        features_tensor = token_features[token_type]
        
        # Convert to numpy for easier processing (handle BFloat16)
        features_np = features_tensor.float().numpy()
        
        print(f"Processing all {features_np.shape[1]} features for token_type='{token_type}'")
        
        # Process ALL features (not just active ones)
        for feature_idx in range(features_np.shape[1]):
            feature_activations = features_np[:, feature_idx]  # [num_prompts]
            
            # Split into active and inactive
            active_mask = feature_activations > 0
            active_activations = feature_activations[active_mask]
            
            # Calculate comprehensive statistics
            all_mean = float(feature_activations.mean())
            all_std = float(feature_activations.std())
            max_activation = float(feature_activations.max())  # same whether active or all
            
            # Active-only statistics
            if len(active_activations) > 0:
                active_mean = float(active_activations.mean())
                active_min = float(active_activations.min())
                active_std = float(active_activations.std())
            else:
                active_mean = active_min = active_std = 0.0
            
            # Sparsity statistics
            num_active = len(active_activations)
            sparsity = num_active / len(feature_activations)  # fraction of prompts where feature is active
            
            # Percentiles (useful for understanding distribution)
            p90 = float(np.percentile(feature_activations, 90))
            p95 = float(np.percentile(feature_activations, 95))
            p99 = float(np.percentile(feature_activations, 99))
            
            # Add to results
            csv_result = {
                'feature_id': int(feature_idx),
                'all_mean': all_mean,
                'all_std': all_std,
                'active_mean': active_mean,
                'active_min': active_min,
                'active_std': active_std,
                'max': max_activation,
                'num_active': num_active,
                'sparsity': sparsity,
                'p90': p90,
                'p95': p95,
                'p99': p99,
                'source': source_name,
                'token': token_type,
            }
            csv_results.append(csv_result)
        
        print(f"Processed all {features_np.shape[1]} features for token_type='{token_type}'")
    
    print(f"\nTotal feature records: {len(csv_results)}")
    return csv_results

def save_as_pt_cpu():
    """Save results as PyTorch tensors using CPU computation (most accurate)"""
    source_name = f"{MODEL_TYPE}_trainer{SAE_TRAINER}_layer{SAE_LAYER}_{MODEL_VER}"
    
    print(f"Processing results for PyTorch format using CPU, source: {source_name}")
    
    # Store results as tensors for each token type
    results_dict = {}
    
    # Process each token type
    for token_type in TOKEN_OFFSETS.keys():
        print(f"\nProcessing token type: {token_type}")
        
        # Get features tensor for this token type: [num_prompts, num_features]
        features_tensor = token_features[token_type].float()  # Convert to float32 on CPU
        
        print(f"Processing all {features_tensor.shape[1]} features for token_type='{token_type}' on CPU")
        
        # Calculate statistics vectorized across all features
        # features_tensor shape: [num_prompts, num_features]
        
        # All statistics (including zeros)
        all_mean = features_tensor.mean(dim=0)  # [num_features]
        all_std = features_tensor.std(dim=0)    # [num_features]
        max_vals = features_tensor.max(dim=0)[0]  # [num_features]
        
        # Active statistics (only non-zero values)
        active_mask = features_tensor > 0  # [num_prompts, num_features]
        num_active = active_mask.sum(dim=0)  # [num_features]
        sparsity = num_active.float() / features_tensor.shape[0]  # [num_features]
        
        # For active mean/std/min, we need to handle features with no active values
        active_mean = torch.zeros_like(all_mean)
        active_std = torch.zeros_like(all_std)
        active_min = torch.zeros_like(all_mean)
        
        # Percentiles
        p90 = torch.quantile(features_tensor, 0.9, dim=0)
        p95 = torch.quantile(features_tensor, 0.95, dim=0)
        p99 = torch.quantile(features_tensor, 0.99, dim=0)
        
        # Calculate active stats only for features that have active values
        for feat_idx in range(features_tensor.shape[1]):
            if num_active[feat_idx] > 0:
                active_vals = features_tensor[:, feat_idx][active_mask[:, feat_idx]]
                active_mean[feat_idx] = active_vals.mean()
                active_std[feat_idx] = active_vals.std()
                active_min[feat_idx] = active_vals.min()
        
        # Store all statistics as tensors
        results_dict[token_type] = {
            'all_mean': all_mean,
            'all_std': all_std,
            'active_mean': active_mean,
            'active_min': active_min,
            'active_std': active_std,
            'max': max_vals,
            'num_active': num_active,
            'sparsity': sparsity,
            'p90': p90,
            'p95': p95,
            'p99': p99,
        }
        
        print(f"Processed all {features_tensor.shape[1]} features for token_type='{token_type}'")
    
    # Add metadata
    results_dict['metadata'] = {
        'source': source_name,
        'model_type': MODEL_TYPE,
        'model_ver': MODEL_VER,
        'sae_layer': SAE_LAYER,
        'sae_trainer': SAE_TRAINER,
        'num_prompts': features_tensor.shape[0],
        'num_features': features_tensor.shape[1],
        'token_types': list(TOKEN_OFFSETS.keys())
    }
    
    print(f"\nTotal token types processed: {len(results_dict) - 1}")  # -1 for metadata
    return results_dict

def save_as_pt_gpu():
    """Save results as PyTorch tensors using GPU computation (faster but potentially less accurate)"""
    source_name = f"{MODEL_TYPE}_trainer{SAE_TRAINER}_layer{SAE_LAYER}_{MODEL_VER}"
    
    print(f"Processing results for PyTorch format using GPU, source: {source_name}")
    
    # Store results as tensors for each token type
    results_dict = {}
    
    # Process each token type
    for token_type in TOKEN_OFFSETS.keys():
        print(f"\nProcessing token type: {token_type}")
        
        # Get features tensor for this token type: [num_prompts, num_features]
        # Keep on GPU for faster computation and ensure float dtype
        features_tensor = token_features[token_type].to(device).float()
        
        print(f"Processing all {features_tensor.shape[1]} features for token_type='{token_type}' on GPU")
        print(f"Features tensor dtype: {features_tensor.dtype}")
        
        # Calculate statistics vectorized across all features on GPU
        # features_tensor shape: [num_prompts, num_features]
        
        # All statistics (including zeros)
        all_mean = features_tensor.mean(dim=0)  # [num_features]
        all_std = features_tensor.std(dim=0)    # [num_features]
        max_vals = features_tensor.max(dim=0)[0]  # [num_features]
        
        # Active statistics (only non-zero values)
        active_mask = features_tensor > 0  # [num_prompts, num_features]
        num_active = active_mask.sum(dim=0)  # [num_features]
        sparsity = num_active.float() / features_tensor.shape[0]  # [num_features]
        
        # Percentiles - compute on GPU
        p90 = torch.quantile(features_tensor, 0.9, dim=0)
        p95 = torch.quantile(features_tensor, 0.95, dim=0)
        p99 = torch.quantile(features_tensor, 0.99, dim=0)
        
        # For active mean/std/min, we need to handle features with no active values
        # Use masked operations for better GPU performance
        active_mean = torch.zeros_like(all_mean)
        active_std = torch.zeros_like(all_std)
        active_min = torch.zeros_like(all_mean)
        
        # Find features that have active values
        has_active = num_active > 0
        
        if has_active.any():
            # Use broadcasting to compute active stats efficiently
            # For each feature with active values, compute mean/std/min
            features_with_active = features_tensor[:, has_active]  # [num_prompts, num_active_features]
            mask_with_active = active_mask[:, has_active]  # [num_prompts, num_active_features]
            
            # Set inactive values to 0 for mean calculation
            active_values = features_with_active * mask_with_active
            
            # Calculate active means
            active_sums = active_values.sum(dim=0)  # [num_active_features]
            active_counts = mask_with_active.sum(dim=0)  # [num_active_features]
            active_means_subset = active_sums / active_counts  # [num_active_features]
            active_mean[has_active] = active_means_subset
            
            # Calculate active mins (set inactive to large value first)
            large_value = features_tensor.max() + 1
            features_for_min = features_with_active.clone()
            features_for_min[~mask_with_active] = large_value
            active_min[has_active] = features_for_min.min(dim=0)[0]
            
            # Calculate active stds
            # For each feature, compute std of only active values
            for i, feat_idx in enumerate(torch.where(has_active)[0]):
                active_vals = features_tensor[:, feat_idx][active_mask[:, feat_idx]]
                if len(active_vals) > 1:  # Need at least 2 values for std
                    active_std[feat_idx] = active_vals.std()
        
        # Store all statistics as tensors (move to CPU for storage)
        results_dict[token_type] = {
            'all_mean': all_mean.cpu(),
            'all_std': all_std.cpu(),
            'active_mean': active_mean.cpu(),
            'active_min': active_min.cpu(),
            'active_std': active_std.cpu(),
            'max': max_vals.cpu(),
            'num_active': num_active.cpu(),
            'sparsity': sparsity.cpu(),
            'p90': p90.cpu(),
            'p95': p95.cpu(),
            'p99': p99.cpu(),
        }
        
        print(f"Processed all {features_tensor.shape[1]} features for token_type='{token_type}'")
    
    # Add metadata
    results_dict['metadata'] = {
        'source': source_name,
        'model_type': MODEL_TYPE,
        'model_ver': MODEL_VER,
        'sae_layer': SAE_LAYER,
        'sae_trainer': SAE_TRAINER,
        'num_prompts': features_tensor.shape[0],
        'num_features': features_tensor.shape[1],
        'token_types': list(TOKEN_OFFSETS.keys())
    }
    
    print(f"\nTotal token types processed: {len(results_dict) - 1}")  # -1 for metadata
    return results_dict

# Choose your approach:
# results_dict = save_as_pt_cpu()    # Most accurate, slower
# results_dict = save_as_pt_gpu()    # Faster, potentially less accurate

# Use CPU version by default for accuracy
print("Using CPU version for maximum accuracy...")
results_dict = save_as_pt_cpu()

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


