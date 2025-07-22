# %% [markdown]
# # Feature Analysis: Finding features active on particular tokens


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
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

@dataclass
class ModelConfig:
    """Configuration for model-specific settings"""
    base_model_name: str
    chat_model_name: str
    hf_release: str  # Reference only - actual loading uses saelens_release/sae_id
    assistant_header: str
    token_offsets: Dict[str, int]
    sae_base_path: str
    saelens_release: str  # Template for sae_lens release parameter
    sae_id_template: str  # Template for sae_lens sae_id parameter
    
    def get_sae_params(self, sae_layer: int, sae_trainer: str) -> Tuple[str, str]:
        """
        Generate SAE lens release and sae_id parameters.
        
        Args:
            sae_layer: Layer number for the SAE
            sae_trainer: Trainer identifier for the SAE
            
        Returns:
            Tuple of (release, sae_id) for sae_lens.SAE.from_pretrained()
        """
        if self.saelens_release == "llama_scope_lxr_{trainer}":
            release = self.saelens_release.format(trainer=sae_trainer)
            sae_id = self.sae_id_template.format(layer=sae_layer, trainer=sae_trainer)
        elif self.saelens_release == "gemma-scope-9b-pt-res":
            # Parse SAE_TRAINER "131k-l0-34" into components for Gemma
            parts = sae_trainer.split("-")
            width = parts[0]  # "131k"
            l0_value = parts[2]  # "34"
            
            release = self.saelens_release
            sae_id = self.sae_id_template.format(layer=sae_layer, width=width, l0=l0_value)
        elif self.saelens_release == "gemma-scope-9b-pt-res-canonical":
            # Parse SAE_TRAINER "131k-l0-34" into components for Gemma
            parts = sae_trainer.split("-")
            width = parts[0]  # "131k"

            release = self.saelens_release
            sae_id = self.sae_id_template.format(layer=sae_layer, width=width)
        else:
            raise ValueError(f"Unknown SAE lens release template: {self.saelens_release}")
        
        return release, sae_id

# Model configurations
MODEL_CONFIGS = {
    "llama": ModelConfig(
        base_model_name="meta-llama/Llama-3.1-8B",
        chat_model_name="meta-llama/Llama-3.1-8B-Instruct",
        hf_release="fnlp/Llama3_1-8B-Base-LXR-32x",
        assistant_header="<|start_header_id|>assistant<|end_header_id|>",
        token_offsets={"asst": -2, "endheader": -1, "newline": 0},
        sae_base_path="/workspace/sae/llama-3.1-8b/saes",
        saelens_release="llama_scope_lxr_{trainer}",
        sae_id_template="l{layer}r_{trainer}"
    ),
    "gemma": ModelConfig(
        base_model_name="google/gemma-2-9b",
        chat_model_name="google/gemma-2-9b-it",
        hf_release="google/gemma-scope-9b-pt-res/layer_{layer}/width_{width}/average_l0_{l0}",
        assistant_header="<start_of_turn>model",
        token_offsets={"model": -1, "newline": 0},
        sae_base_path="/workspace/sae/gemma-2-9b/saes",
        saelens_release="gemma-scope-9b-pt-res-canonical",
        sae_id_template="layer_{layer}/width_{width}/canonical"
    )
}

# =============================================================================
# AUTOMATION CONFIGURATION
# =============================================================================
MODEL_TYPE = "gemma"  # Options: "gemma" or "llama"
SAE_TRAINER = "131k-l0-114"
N_PROMPTS = 1000

# Define layers and model versions to process
LAYERS_TO_PROCESS = [20]  # Common layers for analysis
MODEL_VERSIONS = ["chat"]  # Process both base and chat versions

# Target tokens to analyze (will find all occurrences of these tokens)
TARGET_TOKENS = ["you"]  # We'll use existing header tokens for model/newline

OUTPUT_FILE = f"./results/8_active_tokens/{MODEL_TYPE}_trainer{SAE_TRAINER}_layer{LAYERS_TO_PROCESS[0]}/2_model_{TARGET_TOKENS[0]}.csv"
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# =============================================================================
# CONFIGURATION SETUP
# =============================================================================
if MODEL_TYPE not in MODEL_CONFIGS:
    raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}. Available: {list(MODEL_CONFIGS.keys())}")

config = MODEL_CONFIGS[MODEL_TYPE]

# Always use chat model for tokenizer (has chat template)
CHAT_MODEL_NAME = config.chat_model_name

# Set up derived configurations
ASSISTANT_HEADER = config.assistant_header
TOKEN_OFFSETS = config.token_offsets
SAE_BASE_PATH = config.sae_base_path

# Data paths
PROMPTS_HF = "lmsys/lmsys-chat-1m"
SEED = 42
PROMPTS_PATH = f"/workspace/data/{PROMPTS_HF.split('/')[-1]}/chat_{N_PROMPTS}.jsonl"
# PROMPTS_PATH = "./prompts/personal_40/personal.jsonl"
os.makedirs(os.path.dirname(PROMPTS_PATH), exist_ok=True)

# Processing parameters
BATCH_SIZE = 32
MAX_LENGTH = 512

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

def load_prompts_from_jsonl(file_path: str) -> pd.DataFrame:
    """Load prompts from a JSONL file. Expects each line to have a 'content' field."""
    prompts = []
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            prompts.append(data['content'])
    return pd.DataFrame(prompts, columns=['prompt'])

# Note: Data loading, model loading, and SAE loading are now handled in the automation functions

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

def find_target_token_positions(input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                              target_token: str, tokenizer, device) -> List[int]:
    """Find all positions where a target token appears in the input (case-insensitive)."""
    # Get all possible tokenizations of the target token (both original case and variations)
    target_token_ids = []
    
    # Generate case variations
    case_variants = [target_token.lower(), target_token.upper(), target_token.capitalize()]
    
    for variant in case_variants:
        # Try different tokenization contexts for each case variant
        # 1. Standalone token
        standalone_tokens = tokenizer.encode(variant, add_special_tokens=False)
        target_token_ids.extend(standalone_tokens)
        
        # 2. Token with space prefix (common in middle of sentences)
        spaced_tokens = tokenizer.encode(" " + variant, add_special_tokens=False)
        target_token_ids.extend(spaced_tokens)
        
        # 3. Token with various punctuation/context
        context_variants = [f" {variant}?", f" {variant}.", f" {variant},", f" {variant}!", f"{variant}"]
        for ctx_variant in context_variants:
            variant_tokens = tokenizer.encode(ctx_variant, add_special_tokens=False)
            target_token_ids.extend(variant_tokens)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_target_token_ids = []
    for token_id in target_token_ids:
        if token_id not in seen:
            seen.add(token_id)
            unique_target_token_ids.append(token_id)
    
    # Find all positions where any of these tokens appear
    positions = []
    max_pos = attention_mask.sum().item() - 1
    
    for i in range(len(input_ids)):
        if i > max_pos:  # Don't check padding tokens
            break
        
        if input_ids[i].item() in unique_target_token_ids:
            # Double-check by decoding to make sure it's actually our target token (case-insensitive)
            decoded = tokenizer.decode([input_ids[i].item()]).strip().lower()
            if target_token.lower() in decoded:
                positions.append(i)
    
    return positions

@torch.no_grad()
def extract_activations_and_metadata(prompts: List[str], layer_idx: int, model) -> Tuple[torch.Tensor, List[Dict], List[str]]:
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
            
            # Calculate positions for original token types (header-based)
            positions = {}
            for token_type, token_offset in TOKEN_OFFSETS.items():
                positions[token_type] = find_assistant_position(
                    input_ids, attention_mask, ASSISTANT_HEADER, token_offset, tokenizer, device
                )
            
            # Find positions for target tokens
            target_token_positions = {}
            for target_token in TARGET_TOKENS:
                target_positions = find_target_token_positions(
                    input_ids, attention_mask, target_token, tokenizer, device
                )
                target_token_positions[target_token] = target_positions
                
            
            # Store the full activation sequence and metadata
            all_activations.append(activations[j].cpu())  # [seq_len, hidden_dim]
            all_metadata.append({
                'prompt_idx': i + j,
                'positions': positions,
                'target_token_positions': target_token_positions,
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
    
    # Initialize results for each original token type
    for token_type in TOKEN_OFFSETS.keys():
        results[token_type] = []
    
    # Initialize results for each target token type
    for target_token in TARGET_TOKENS:
        results[target_token] = []
    
    # Extract activations for each token type
    for i, meta in enumerate(metadata):
        # Original token types (single position per prompt)
        for token_type, position in meta['positions'].items():
            activation = full_activations[i, position, :]  # [hidden_dim]
            results[token_type].append(activation)
        
        # Target tokens (potentially multiple positions per prompt)
        for target_token in TARGET_TOKENS:
            if target_token in meta['target_token_positions']:
                target_positions = meta['target_token_positions'][target_token]
                if target_positions:  # If any positions found
                    for position in target_positions:
                        activation = full_activations[i, position, :]
                        results[target_token].append(activation)
    
    # Convert lists to tensors
    for token_type in TOKEN_OFFSETS.keys():
        if results[token_type]:  # Only if we have data
            results[token_type] = torch.stack(results[token_type], dim=0)
        else:
            # Create empty tensor if no activations found
            results[token_type] = torch.empty(0, full_activations.shape[-1])
    
    for target_token in TARGET_TOKENS:
        if results[target_token]:  # Only if we have data
            results[target_token] = torch.stack(results[target_token], dim=0)
        else:
            # Create empty tensor if no activations found
            results[target_token] = torch.empty(0, full_activations.shape[-1])
    
    return results

print("Activation extraction functions defined")

# %% [markdown]
# ## SAE Feature Processing Functions

# %%
@torch.no_grad()
def get_sae_features_batched(activations: torch.Tensor, sae) -> torch.Tensor:
    """Apply SAE to get feature activations with proper batching."""
    activations = activations.to(device)
    
    # Process in batches to avoid memory issues
    feature_activations = []
    
    for i in range(0, activations.shape[0], BATCH_SIZE):
        batch = activations[i:i+BATCH_SIZE]
        features = sae.encode(batch)  # [batch, num_features]
        feature_activations.append(features.cpu())
    
    return torch.cat(feature_activations, dim=0)

def create_feature_comparison_csv(token_features, model_type, sae_trainer, sae_layer, model_ver, token_offsets):
    """Create CSV comparing features active on both the main header token (model/endheader) and 'you' tokens."""
    print("\nCreating feature comparison CSV...")
    
    # Use only the main header token (model for Gemma, endheader for Llama)
    if model_type == "gemma":
        header_token = "model"  # For Gemma: <start_of_turn>model
    else:  # llama
        header_token = "endheader"  # For Llama: <|start_header_id|>assistant<|end_header_id|>
    
    # Get features for the main header token only
    if header_token not in token_features or token_features[header_token].numel() == 0:
        print(f"No '{header_token}' token features found!")
        return
    
    header_features = token_features[header_token]
    
    # Get 'you' token features
    if "you" not in token_features or token_features["you"].numel() == 0:
        print("No 'you' token features found!")
        return
    
    you_features = token_features["you"]
    
    print(f"'{header_token}' token instances: {header_features.shape[0]}")
    print(f"'You' token instances: {you_features.shape[0]}")
    print(f"Number of features: {header_features.shape[1]}")
    
    # Debug: Check if we have any activations at all
    print(f"DEBUG: Header features non-zero count: {(header_features > 0).sum().item()}")
    print(f"DEBUG: You features non-zero count: {(you_features > 0).sum().item()}")
    if you_features.shape[0] > 0:
        print(f"DEBUG: First few you feature 45426 activations: {you_features[:5, 45426].tolist()}")
    if header_features.shape[0] > 0:
        print(f"DEBUG: First few header feature 45426 activations: {header_features[:5, 45426].tolist()}")
    
    # Calculate mean activations for each feature across instances
    header_mean_activations = header_features.mean(dim=0)  # [num_features]
    you_mean_activations = you_features.mean(dim=0)  # [num_features]
    
    # Calculate number of active prompts for each feature
    header_active_prompts = (header_features > 0).sum(dim=0)  # [num_features]
    you_active_prompts = (you_features > 0).sum(dim=0)  # [num_features]
    
    # Find features that are active on BOTH token types (non-zero mean)
    header_active_mask = header_mean_activations > 0
    you_active_mask = you_mean_activations > 0
    both_active_mask = header_active_mask & you_active_mask
    
    # Debug feature 45426 specifically
    if len(header_mean_activations) > 45426:
        print(f"DEBUG Feature 45426:")
        print(f"  Header mean: {header_mean_activations[45426].item():.6f}")
        print(f"  You mean: {you_mean_activations[45426].item():.6f}")
        print(f"  Header > 0: {header_active_mask[45426].item()}")
        print(f"  You > 0: {you_active_mask[45426].item()}")
        print(f"  Both > 0: {both_active_mask[45426].item()}")
    
    # Filter to features active on both header tokens and 'you' tokens
    active_feature_indices = torch.where(both_active_mask)[0]
    
    if len(active_feature_indices) == 0:
        print("No features found that are active on both header tokens and 'you' tokens!")
        return
    
    print(f"Found {len(active_feature_indices)} features active on both token types")
    
    # Create data for CSV
    csv_data = []
    for idx in active_feature_indices:
        feature_id = idx.item()
        header_mean = header_mean_activations[idx].item()
        you_mean = you_mean_activations[idx].item()
        combined_mean = (header_mean + you_mean) / 2
        header_active_count = header_active_prompts[idx].item()
        you_active_count = you_active_prompts[idx].item()
        
        csv_data.append({
            'feature_id': feature_id,
            'header_mean_activation': header_mean,
            'you_mean_activation': you_mean,
            'combined_mean': combined_mean,
            'header_active_prompts': header_active_count,
            'you_active_prompts': you_active_count
        })
    
    # Sort by combined mean (descending)
    csv_data.sort(key=lambda x: x['combined_mean'], reverse=True)
    
    # Create output directory and filename
    
    csv_filename = OUTPUT_FILE
    
    # Write CSV
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['feature_id', 'header_mean_activation', 'you_mean_activation', 'combined_mean', 'header_active_prompts', 'you_active_prompts']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"\nCSV saved to: {csv_filename}")
    print(f"Total features in CSV: {len(csv_data)}")
    print(f"Top 5 features by combined mean:")
    for i, row in enumerate(csv_data[:5]):
        print(f"  {i+1}. Feature {row['feature_id']}: header={row['header_mean_activation']:.4f} ({row['header_active_prompts']} prompts), you={row['you_mean_activation']:.4f} ({row['you_active_prompts']} prompts), combined={row['combined_mean']:.4f}")
    
    return csv_filename

def save_as_pt_cpu(token_features, model_type, sae_trainer, sae_layer, model_ver, token_offsets):
    """Save results as PyTorch tensors using CPU computation (most accurate)"""
    source_name = f"{model_type}_trainer{sae_trainer}_layer{sae_layer}_{model_ver}"
    
    print(f"Processing results for PyTorch format using CPU, source: {source_name}")
    
    # Store results as tensors for each token type
    results_dict = {}
    
    # Process each token type
    for token_type in token_offsets.keys():
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
        
        # Store essential statistics as tensors
        results_dict[token_type] = {
            'all_mean': all_mean,
            'all_std': all_std,
            'max': max_vals,
            'num_active': num_active,
            'sparsity': sparsity,
        }
        
        print(f"Processed all {features_tensor.shape[1]} features for token_type='{token_type}'")
    
    # Add metadata
    results_dict['metadata'] = {
        'source': source_name,
        'model_type': model_type,
        'model_ver': model_ver,
        'sae_layer': sae_layer,
        'sae_trainer': sae_trainer,
        'num_prompts': features_tensor.shape[0],
        'num_features': features_tensor.shape[1],
        'token_types': list(token_offsets.keys())
    }
    
    print(f"\nTotal token types processed: {len(results_dict) - 1}")  # -1 for metadata
    return results_dict

def load_sae(config: ModelConfig, sae_path: str, sae_layer: int, sae_trainer: str) -> SAE:
    """
    Unified SAE loading function that handles both Llama and Gemma models.
    
    Args:
        config: ModelConfig object containing model-specific settings
        sae_path: Local path to store/load SAE files
        sae_layer: Layer number for the SAE
        sae_trainer: Trainer identifier for the SAE
    
    Returns:
        SAE: Loaded SAE model
    """
    # Check if SAE file exists locally
    ae_file_path = os.path.join(sae_path, "sae_weights.safetensors")
    
    if os.path.exists(ae_file_path):
        print(f"✓ Found SAE files at: {os.path.dirname(ae_file_path)}")
        sae = SAE.load_from_disk(sae_path)
        return sae
    
    print(f"SAE not found locally, downloading from HF via sae_lens...")
    os.makedirs(os.path.dirname(sae_path), exist_ok=True)
    
    # Get SAE parameters from config
    release, sae_id = config.get_sae_params(sae_layer, sae_trainer)
    print(f"Loading SAE with release='{release}', sae_id='{sae_id}'")
    
    # Load the SAE using sae_lens
    sae, _, sparsity = SAE.from_pretrained(
        release=release,
        sae_id=sae_id,
        device="cuda" # Hardcoded because it wants a string
    )
    
    # Save the SAE locally for future use
    sae.save_model(sae_path, sparsity)
    return sae

# %% [markdown]
# ## Main Processing Function and Automation Loop

# %%
def process_layer_and_version(sae_layer: int, model_ver: str, sae):
    """Process a single layer and model version combination with pre-loaded SAE."""
    print(f"\n{'='*60}")
    print(f"PROCESSING: Layer {sae_layer}, Model Version: {model_ver}")
    print(f"{'='*60}")
    
    # Set model name based on version
    if model_ver == "chat":
        model_name = config.chat_model_name
    elif model_ver == "base":
        model_name = config.base_model_name
    else:
        raise ValueError(f"Unknown model_ver: {model_ver}. Use 'chat' or 'base'")
    
    # Set up paths
    # output_file = f"/workspace/results/4_diffing/{MODEL_TYPE}_trainer{SAE_TRAINER}_layer{sae_layer}/{N_PROMPTS}_prompts/{model_ver}.pt"
    # os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # # Check if results already exist
    # if os.path.exists(output_file):
    #     print(f"✓ Results already exist at: {output_file}")
    #     print("Skipping this combination...")
    #     return
    
    layer_index = sae_layer
    
    print(f"Configuration:")
    print(f"  Model to load: {model_name}")
    print(f"  SAE Layer: {sae_layer}, Trainer: {SAE_TRAINER}")
    print(f"  Output file: {OUTPUT_FILE}")
    
    # Load model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": 0}
    )
    model.eval()
    print(f"Model loaded: {model.__class__.__name__}")
    
    # Extract activations
    print("\nExtracting activations for all positions...")
    full_activations, metadata, formatted_prompts = extract_activations_and_metadata(
        prompts_df['prompt'].tolist(), layer_index, model
    )
    print(f"Full activations shape: {full_activations.shape}")
    
    # Extract token-specific activations
    print("\nExtracting activations for specific token positions...")
    token_activations = extract_token_activations(full_activations, metadata)
    
    # Get SAE features
    print("\nComputing SAE features for specific token positions...")
    token_features = {}
    for token_type, activations in token_activations.items():
        print(f"Processing SAE features for token type '{token_type}'...")
        features = get_sae_features_batched(activations, sae)
        token_features[token_type] = features
        print(f"Features shape for '{token_type}': {features.shape}")
    
    # Create CSV comparison
    print("\nCreating CSV comparison...")
    csv_file = create_feature_comparison_csv(token_features, MODEL_TYPE, SAE_TRAINER, sae_layer, model_ver, TOKEN_OFFSETS)
    
    # Process and save results
    # print("\nProcessing results...")
    # results_dict = save_as_pt_cpu(token_features, MODEL_TYPE, SAE_TRAINER, sae_layer, model_ver, TOKEN_OFFSETS)
    
    # Save results
    # print("Saving results...")
    # torch.save(results_dict, output_file)
    # print(f"Results saved to: {output_file}")
    if csv_file:
        print(f"CSV comparison saved to: {csv_file}")
    
    # Clean up memory (but not SAE - that's handled by the caller)
    del model
    del full_activations
    del token_activations
    del token_features
    # del results_dict
    torch.cuda.empty_cache()
    print("Memory cleaned up")

def run_automation():
    """Run the automated processing for all layer and model version combinations."""
    print(f"\n{'='*80}")
    print("STARTING AUTOMATED PROCESSING")
    print(f"{'='*80}")
    print(f"Model Type: {MODEL_TYPE}")
    print(f"Layers to process: {LAYERS_TO_PROCESS}")
    print(f"Model versions: {MODEL_VERSIONS}")
    print(f"Total combinations: {len(LAYERS_TO_PROCESS) * len(MODEL_VERSIONS)}")
    
    # Load prompts once (they're the same for all combinations)
    print("\nLoading prompts...")
    global prompts_df
    prompts_df = load_lmsys_prompts(PROMPTS_PATH, PROMPTS_HF, N_PROMPTS, SEED)
    print(f"Loaded {prompts_df.shape[0]} prompts")
    
    # Load tokenizer once (same for all combinations)
    print("\nLoading tokenizer...")
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")
    
    # Process each layer (load SAE once per layer)
    for layer_idx, sae_layer in enumerate(LAYERS_TO_PROCESS):
        print(f"\n{'='*80}")
        print(f"PROCESSING LAYER {sae_layer} ({layer_idx + 1}/{len(LAYERS_TO_PROCESS)})")
        print(f"{'='*80}")
        
        # Load SAE once for this layer
        sae_path = f"{SAE_BASE_PATH}/resid_post_layer_{sae_layer}/trainer_{SAE_TRAINER}"
        
        print(f"Loading SAE for layer {sae_layer}...")
        try:
            sae = load_sae(config, sae_path, sae_layer, SAE_TRAINER)
            sae = sae.to(device)
            print(f"✓ SAE loaded with {sae.cfg.d_sae} features")
        except Exception as e:
            print(f"❌ Error loading SAE for layer {sae_layer}: {str(e)}")
            print("Skipping this layer...")
            continue
        
        # Process both model versions for this layer
        for model_ver in MODEL_VERSIONS:
            print(f"\n{'='*60}")
            print(f"LAYER {sae_layer}, MODEL VERSION: {model_ver}")
            print(f"{'='*60}")
            
            try:
                process_layer_and_version(sae_layer, model_ver, sae)
                print(f"✓ Successfully processed layer {sae_layer}, version {model_ver}")
            except Exception as e:
                print(f"❌ Error processing layer {sae_layer}, version {model_ver}: {str(e)}")
                print("Continuing to next model version...")
                continue
        
        # Clean up SAE for this layer
        print(f"\nCleaning up SAE for layer {sae_layer}...")
        del sae
        torch.cuda.empty_cache()
        print("✓ SAE memory cleaned up")
    
    print(f"\n{'='*80}")
    print("AUTOMATION COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    run_automation()


