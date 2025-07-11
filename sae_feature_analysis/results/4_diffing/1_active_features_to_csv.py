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
SAE_TRAINER = "131k-l0-34"
TOKEN_OFFSETS = {"model": -1, "newline": 0}
N_PROMPTS = 1000

PERCENT_ACTIVE = 1

# File paths
BASE_FILE = f"/workspace/results/4_diffing/{MODEL_TYPE}_trainer{SAE_TRAINER}_layer{SAE_LAYER}/{N_PROMPTS}_prompts/base.pt"
CHAT_FILE = f"/workspace/results/4_diffing/{MODEL_TYPE}_trainer{SAE_TRAINER}_layer{SAE_LAYER}/{N_PROMPTS}_prompts/chat.pt"

# Output directory
OUTPUT_FILE = Path(f"{MODEL_TYPE}_trainer{SAE_TRAINER}_layer{SAE_LAYER}/{N_PROMPTS}_prompts/explanations_{PERCENT_ACTIVE}percent.csv")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Link
LLAMA_LINK_FORMAT = f"https://www.neuronpedia.org/llama3.1-8b/{SAE_LAYER}-llamascope-res-131k/"
GEMMA_LINK_FORMAT = f"https://www.neuronpedia.org/gemma-2-9b/{SAE_LAYER}-gemmascope-res-131k-l0_32plus/"

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

# Extract unique feature IDs that are active in at least one condition
print("\nExtracting unique active feature IDs using boolean masks...")

# Initialize combined mask for all features
num_features = base_data['metadata']['num_features']
combined_mask = torch.zeros(num_features, dtype=torch.bool)

# For each token type, combine masks for features with sparsity > 0
for token_type in token_types:
    print(f"Processing token type: {token_type}")
    
    # Get num_active tensors for base and chat
    base_num_active = base_data[token_type]['num_active']
    chat_num_active = chat_data[token_type]['num_active']
    
    # Create boolean masks for active features (num_active > 0)
    base_active_mask = base_num_active > int(N_PROMPTS * PERCENT_ACTIVE / 100)
    chat_active_mask = chat_num_active > int(N_PROMPTS * PERCENT_ACTIVE / 100)
    
    # Combine with OR operation - feature is active if active in base OR chat
    token_active_mask = base_active_mask | chat_active_mask
    
    # Update combined mask
    combined_mask = combined_mask | token_active_mask
    
    print(f"  Base active features: {base_active_mask.sum().item()}")
    print(f"  Chat active features: {chat_active_mask.sum().item()}")
    print(f"  Combined active features for this token: {token_active_mask.sum().item()}")

# Get unique active feature IDs
unique_active_features = torch.nonzero(combined_mask).squeeze(-1).tolist()
unique_active_features.sort()
print(f"\nTotal unique active features: {len(unique_active_features)}")

# Generate links based on model type
if MODEL_TYPE == "llama":
    link_format = LLAMA_LINK_FORMAT
elif MODEL_TYPE == "gemma":
    link_format = GEMMA_LINK_FORMAT
else:
    raise ValueError(f"Unknown model type: {MODEL_TYPE}")

# Create DataFrame with feature IDs and links
df_results = pd.DataFrame({
    'feature_id': unique_active_features,
    'link': [f"{link_format}{feature_id}" for feature_id in unique_active_features]
})

# Save to CSV
df_results.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved {len(df_results)} unique active features to: {OUTPUT_FILE}")
print(f"Preview of first 5 entries:")
print(df_results.head())
