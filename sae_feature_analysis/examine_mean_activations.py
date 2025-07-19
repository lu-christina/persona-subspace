#!/usr/bin/env python3
"""
Script to examine the structure of mean activations file for compatibility
with mean_ablation steerer.
"""

import torch
import os

def examine_mean_activations_file(file_path: str):
    """Examine the structure of a mean activations .pt file."""
    
    print(f"Examining file: {file_path}")
    print(f"File size: {os.path.getsize(file_path) / (1024**2):.2f} MB")
    print("=" * 60)
    
    # Load the file
    print("Loading file...")
    data = torch.load(file_path, map_location='cpu')
    
    # Check type and structure
    print(f"Data type: {type(data)}")
    
    if isinstance(data, dict):
        print(f"Dictionary keys: {list(data.keys())}")
        
        for key, value in data.items():
            print(f"\nKey: '{key}'")
            print(f"  Type: {type(value)}")
            
            if torch.is_tensor(value):
                print(f"  Shape: {value.shape}")
                print(f"  Dtype: {value.dtype}")
                print(f"  Device: {value.device}")
                print(f"  Memory usage: {value.numel() * value.element_size() / (1024**2):.2f} MB")
                
                # Show basic statistics (but don't print values for large tensors)
                if value.numel() > 0:
                    print(f"  Min: {value.min().item():.6f}")
                    print(f"  Max: {value.max().item():.6f}")
                    print(f"  Mean: {value.mean().item():.6f}")
                    print(f"  Std: {value.std().item():.6f}")
                    
                    # Show a few sample values if tensor is small
                    if value.numel() <= 10:
                        print(f"  Values: {value.tolist()}")
                    elif value.ndim == 1 and value.numel() <= 20:
                        print(f"  First 5 values: {value[:5].tolist()}")
                        print(f"  Last 5 values: {value[-5:].tolist()}")
                        
            elif isinstance(value, list):
                print(f"  Length: {len(value)}")
                if len(value) > 0:
                    print(f"  First element type: {type(value[0])}")
                    if len(value) <= 10:
                        print(f"  Values: {value}")
                    else:
                        print(f"  First 5 values: {value[:5]}")
                        print(f"  Last 5 values: {value[-5:]}")
                        
            elif isinstance(value, dict):
                print(f"  Nested dict keys: {list(value.keys())}")
                for nested_key, nested_value in value.items():
                    print(f"    {nested_key}: {type(nested_value)} = {nested_value}")
            else:
                print(f"  Value: {value}")
    
    elif torch.is_tensor(data):
        print(f"Tensor shape: {data.shape}")
        print(f"Tensor dtype: {data.dtype}")
        print(f"Memory usage: {data.numel() * data.element_size() / (1024**2):.2f} MB")
        
        if data.numel() > 0:
            print(f"Min: {data.min().item():.6f}")
            print(f"Max: {data.max().item():.6f}")
            print(f"Mean: {data.mean().item():.6f}")
            print(f"Std: {data.std().item():.6f}")
    
    else:
        print(f"Unexpected data type: {type(data)}")
        print(f"Data: {data}")
    
    print("\n" + "=" * 60)
    print("COMPATIBILITY ANALYSIS")
    print("=" * 60)
    
    if isinstance(data, dict) and 'mean_activations' in data:
        mean_acts = data['mean_activations']
        feature_ids = data.get('feature_ids', [])
        
        print(f"✓ Found 'mean_activations' tensor with shape: {mean_acts.shape}")
        print(f"✓ Found {len(feature_ids)} feature IDs")
        
        if mean_acts.ndim == 1:
            print("\n⚠️  COMPATIBILITY ISSUE:")
            print(f"   - Current format: mean_activations has shape {mean_acts.shape} (scalar projections)")
            print(f"   - Required format: List[torch.Tensor] where each tensor has shape (d_model,)")
            print(f"   - This file contains mean PROJECTIONS onto feature directions, not mean ACTIVATIONS")
            print(f"   - For mean_ablation steerer, you need mean activation vectors (d_model dims)")
            print(f"\n   To use with mean_ablation steerer, you would need:")
            print(f"   1. The original SAE decoder vectors (feature directions)")
            print(f"   2. Multiply: mean_projection * feature_direction to get mean activation vector")
            print(f"   3. Or extract mean activation vectors directly from raw activations")
            
        elif mean_acts.ndim == 2:
            d_model = mean_acts.shape[1]
            print(f"\n✓ COMPATIBLE FORMAT:")
            print(f"   - Shape {mean_acts.shape} suggests [num_features, d_model] format")
            print(f"   - Each row is a mean activation vector of size {d_model}")
            print(f"   - This can be converted to List[torch.Tensor] for the steerer")
            print(f"   - Conversion: [mean_acts[i] for i in range(mean_acts.shape[0])]")
        else:
            print(f"\n❌ UNEXPECTED FORMAT:")
            print(f"   - mean_activations has {mean_acts.ndim} dimensions: {mean_acts.shape}")
            print(f"   - Expected either 1D (projections) or 2D (activation vectors)")
    
    else:
        print("❌ File does not contain expected 'mean_activations' key in dictionary format")

if __name__ == "__main__":
    file_path = "/workspace/sae/gemma-2-9b/mean_activations/gemma_trainer131k-l0-114_layer20.pt"
    examine_mean_activations_file(file_path)