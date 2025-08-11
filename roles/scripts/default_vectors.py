#!/usr/bin/env python3
"""
default_activation.py - Calculate mean default activation across roles

This script calculates mean default activations for both 'role' and 'role_240' directories:

For 'role':
- pos_1: Mean activations where keys start with 'pos_' and label == 1
- default_1: Mean activations where keys start with 'default_' and label == 1  
- all_1: Mean of all activations labeled 1 from all role files

For 'role_240':
- pos_1: Mean activations where keys start with 'pos_' and label == 1
- default_1: SPECIAL CASE - Mean ALL activations from 0_default.pt and 1_default.pt
- all_1: Mean of all activations labeled 1 from role files + the two default files

Key format: {pos|default}_q{question_index}_p{prompt_index}

Output: /workspace/{dir}/default_activations.pt containing dict with:
- 3 keys (pos_1, default_1, all_1) -> tensors of shape (num_layers, hidden_dims)
- metadata with sample counts
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import torch
from tqdm import tqdm
import re


def parse_key_prefix(key: str) -> Optional[str]:
    """Extract pos/default prefix from keys like 'pos_q1_p2' or 'default_q3_p1'."""
    match = re.match(r'^(pos|default)_', key)
    return match.group(1) if match else None


def load_labels_and_activations(labels_dir: str, activations_dir: str) -> Tuple[Dict[str, Dict[str, Union[int, str]]], Dict[str, Dict[str, torch.Tensor]]]:
    """Load all label and activation files from directories."""
    labels_path = Path(labels_dir)
    activations_path = Path(activations_dir)
    
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
    if not activations_path.exists():
        raise FileNotFoundError(f"Activations directory not found: {activations_dir}")
    
    # Load all label files
    all_labels = {}
    for label_file in labels_path.glob("*.json"):
        role_name = label_file.stem
        try:
            with open(label_file, 'r') as f:
                all_labels[role_name] = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load labels for {role_name}: {e}")
    
    # Load all activation files (including those without corresponding labels)
    all_activations = {}
    for activation_file in activations_path.glob("*.pt"):
        role_name = activation_file.stem
        try:
            all_activations[role_name] = torch.load(activation_file, map_location='cpu')
        except Exception as e:
            print(f"Warning: Failed to load activations for {role_name}: {e}")
    
    return all_labels, all_activations


def filter_activations_by_label(
    labels: Dict[str, Union[int, str]], 
    activations: Dict[str, torch.Tensor], 
    prefix: str, 
    target_label: int = 1
) -> List[torch.Tensor]:
    """Filter activations by prefix and label score."""
    filtered = []
    
    for key, activation in activations.items():
        key_prefix = parse_key_prefix(key)
        if key_prefix == prefix and key in labels:
            label_score = labels[key]
            if label_score == target_label:
                filtered.append(activation)
    
    return filtered


def calculate_mean_activations_role(all_labels: Dict[str, Dict], all_activations: Dict[str, Dict]) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    """Calculate mean activations for 'role' directory."""
    pos_activations = []
    default_activations = []
    all_activations_list = []
    
    # Process each role
    for role_name in all_labels.keys():
        if role_name not in all_activations:
            print(f"Warning: No activations found for role {role_name}")
            continue
            
        labels = all_labels[role_name]
        activations = all_activations[role_name]
        
        # Collect pos activations with label == 1
        pos_acts = filter_activations_by_label(labels, activations, 'pos', 1)
        pos_activations.extend(pos_acts)
        all_activations_list.extend(pos_acts)
        
        # Collect default activations with label == 1
        default_acts = filter_activations_by_label(labels, activations, 'default', 1)
        default_activations.extend(default_acts)
        all_activations_list.extend(default_acts)
    
    # Calculate means
    result = {}
    counts = {}
    
    if pos_activations:
        result['pos_1'] = torch.stack(pos_activations).mean(dim=0)
        counts['pos_1'] = len(pos_activations)
    else:
        print("Warning: No pos activations with label 1 found")
        
    if default_activations:
        result['default_1'] = torch.stack(default_activations).mean(dim=0)
        counts['default_1'] = len(default_activations)
    else:
        print("Warning: No default activations with label 1 found")
        
    if all_activations_list:
        result['all_1'] = torch.stack(all_activations_list).mean(dim=0)
        counts['all_1'] = len(all_activations_list)
    else:
        print("Warning: No activations with label 1 found")
    
    return result, counts


def calculate_mean_activations_role_240(all_labels: Dict[str, Dict], all_activations: Dict[str, Dict]) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    """Calculate mean activations for 'role_240' directory with special default handling."""
    pos_activations = []
    all_activations_list = []
    
    # Process role files (same as role directory for pos and all)
    for role_name in all_labels.keys():
        if role_name in ['0_default', '1_default']:
            continue  # Handle these separately
            
        if role_name not in all_activations:
            print(f"Warning: No activations found for role {role_name}")
            continue
            
        labels = all_labels[role_name]
        activations = all_activations[role_name]
        
        # Collect pos activations with label == 1
        pos_acts = filter_activations_by_label(labels, activations, 'pos', 1)
        pos_activations.extend(pos_acts)
        all_activations_list.extend(pos_acts)
        
        # Collect default activations with label == 1 for all_1
        default_acts = filter_activations_by_label(labels, activations, 'default', 1)
        all_activations_list.extend(default_acts)
    
    # Special handling for default_1: mean ALL activations from 0_default.pt and 1_default.pt
    # These files don't have labels, so we add ALL their activations
    default_activations = []
    for default_file in ['0_default', '1_default']:
        if default_file in all_activations:
            # Add ALL activations from these files (no label filtering)
            activations = all_activations[default_file]
            for key, activation in activations.items():
                default_activations.append(activation)
                # Also add to all_1
                all_activations_list.append(activation)
        else:
            print(f"Warning: Default file {default_file} not found in activations")
    
    # Calculate means
    result = {}
    counts = {}
    
    if pos_activations:
        result['pos_1'] = torch.stack(pos_activations).mean(dim=0)
        counts['pos_1'] = len(pos_activations)
    else:
        print("Warning: No pos activations with label 1 found")
        
    if default_activations:
        result['default_1'] = torch.stack(default_activations).mean(dim=0)
        counts['default_1'] = len(default_activations)
    else:
        print("Warning: No default activations found in 0_default.pt and 1_default.pt")
        
    if all_activations_list:
        result['all_1'] = torch.stack(all_activations_list).mean(dim=0)
        counts['all_1'] = len(all_activations_list)
    else:
        print("Warning: No activations found for all_1")
    
    return result, counts


def process_directory(dir_name: str) -> None:
    """Process a single directory (role or role_240)."""
    print(f"\n=== Processing {dir_name} directory ===")
    
    labels_dir = f"/workspace/{dir_name}/extract_labels"
    activations_dir = f"/workspace/{dir_name}/response_activations"
    output_path = f"/workspace/{dir_name}/default_activations.pt"
    
    try:
        # Load data
        print("Loading labels and activations...")
        all_labels, all_activations = load_labels_and_activations(labels_dir, activations_dir)
        
        print(f"Loaded {len(all_labels)} label files and {len(all_activations)} activation files")
        
        # Calculate means based on directory type
        if dir_name == 'role_240':
            result, counts = calculate_mean_activations_role_240(all_labels, all_activations)
        else:
            result, counts = calculate_mean_activations_role(all_labels, all_activations)
        
        # Prepare output with metadata
        output = {
            'activations': result,
            'metadata': {
                'counts': counts,
                'directory': dir_name,
                'total_files_processed': {
                    'labels': len(all_labels),
                    'activations': len(all_activations)
                }
            }
        }
        
        # Save results
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(output, output_path)
        
        print(f"Results saved to: {output_path}")
        print("Sample counts:")
        for key, count in counts.items():
            tensor_shape = result[key].shape if key in result else "N/A"
            print(f"  {key}: {count} samples, shape: {tensor_shape}")
            
    except Exception as e:
        print(f"Error processing {dir_name}: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Calculate mean default activations across roles')
    parser.add_argument('--dirs', nargs='+', default=['roles', 'roles_240'], 
                       help='Directories to process (default: roles roles_240)')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    print("=== Default Activation Calculator ===")
    print(f"Processing directories: {args.dirs}")
    
    for dir_name in args.dirs:
        try:
            process_directory(dir_name)
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            break
        except Exception as e:
            print(f"Failed to process {dir_name}: {e}")
            continue
    
    print("\n=== Processing complete ===")


if __name__ == "__main__":
    main()