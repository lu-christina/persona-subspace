import torch
import pandas as pd
import numpy as np
import os
from pathlib import Path

# Configuration
# MODEL_TYPE = "llama"
# SAE_LAYER = 15
# SAE_TRAINER = "32x"
# TOKEN_OFFSETS = {"asst": -2, "endheader": -1, "newline": 0}
# N_PROMPTS = 1000
MODEL_TYPE = "gemma"
SAE_LAYER = 20
SAE_TRAINER = "131k-l0-114"
TOKEN_OFFSETS = {"model": -1, "newline": 0}

# File paths
BASE_FILE = f"/workspace/results/5_diffing_personal/{MODEL_TYPE}_trainer{SAE_TRAINER}_layer{SAE_LAYER}/personal_40/base.pt"
CHAT_FILE = f"/workspace/results/5_diffing_personal/{MODEL_TYPE}_trainer{SAE_TRAINER}_layer{SAE_LAYER}/personal_40/chat.pt"

# Output directory
OUTPUT_FILE = Path(f"{MODEL_TYPE}_trainer{SAE_TRAINER}_layer{SAE_LAYER}/personal_40/target_features.csv")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Link
LLAMA_LINK_FORMAT = f"https://www.neuronpedia.org/llama3.1-8b/{SAE_LAYER}-llamascope-res-131k/"
GEMMA_LINK_FORMAT = f"https://www.neuronpedia.org/gemma-2-9b/{SAE_LAYER}-gemmascope-res-131k/"

print(f"Loading base model data from: {BASE_FILE}")
print(f"Loading chat model data from: {CHAT_FILE}")
print(f"Output file: {OUTPUT_FILE}")

# Load the PyTorch files
base_data = torch.load(BASE_FILE)
chat_data = torch.load(CHAT_FILE)

print(f"\nBase data keys: {list(base_data.keys())}")
print(f"Chat data keys: {list(chat_data.keys())}")
print(f"Base metadata: {base_data['metadata']}")
print(f"Chat metadata: {chat_data['metadata']}")

# Verify token types match
base_tokens = [k for k in base_data.keys() if k != 'metadata']
chat_tokens = [k for k in chat_data.keys() if k != 'metadata']
print(f"\nBase token types: {base_tokens}")
print(f"Chat token types: {chat_tokens}")
assert base_tokens == chat_tokens, "Token types don't match between base and chat!"

token_types = base_tokens
print(f"Processing {len(token_types)} token types: {token_types}")

# Extract feature IDs that are target-exclusive for at least one model
print("\nExtracting target-exclusive feature IDs...")

# Initialize combined mask for target-exclusive features
num_features = base_data['metadata']['num_features']
target_exclusive_mask = torch.zeros(num_features, dtype=torch.bool)

# For each token type, find target-exclusive features
for token_type in token_types:
    print(f"Processing token type: {token_type}")
    
    # Get target and control num_active tensors for base and chat
    base_target_active = base_data[token_type]['target_num_active']
    base_control_active = base_data[token_type]['control_num_active']
    chat_target_active = chat_data[token_type]['target_num_active']
    chat_control_active = chat_data[token_type]['control_num_active']
    
    # Create boolean masks for target-exclusive features (target > 0 AND control == 0)
    base_target_exclusive = (base_target_active > 0) & (base_control_active == 0)
    chat_target_exclusive = (chat_target_active > 0) & (chat_control_active == 0)
    
    # Combine with OR operation - feature is target-exclusive if exclusive in base OR chat
    token_target_exclusive = base_target_exclusive | chat_target_exclusive
    
    # Update combined mask
    target_exclusive_mask = target_exclusive_mask | token_target_exclusive
    
    print(f"  Base target-exclusive features: {base_target_exclusive.sum().item()}")
    print(f"  Chat target-exclusive features: {chat_target_exclusive.sum().item()}")
    print(f"  Combined target-exclusive features for this token: {token_target_exclusive.sum().item()}")

# Get target-exclusive feature IDs
target_exclusive_features = torch.nonzero(target_exclusive_mask).squeeze(-1).tolist()
target_exclusive_features.sort()
print(f"\nTotal target-exclusive features: {len(target_exclusive_features)}")

# Generate links based on model type
if MODEL_TYPE == "llama":
    link_format = LLAMA_LINK_FORMAT
elif MODEL_TYPE == "gemma":
    link_format = GEMMA_LINK_FORMAT
else:
    raise ValueError(f"Unknown model type: {MODEL_TYPE}")

# Create DataFrame with feature IDs and links
df_results = pd.DataFrame({
    'feature_id': target_exclusive_features,
    'link': [f"{link_format}{feature_id}" for feature_id in target_exclusive_features]
})

# Save to CSV
df_results.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved {len(df_results)} target-exclusive features to: {OUTPUT_FILE}")
print(f"Preview of first 5 entries:")
print(df_results.head())
