#!/usr/bin/env python3
"""
default_vectors.py - Calculate mean default activation across roles

This script calculates mean default activations for roles directories.
Processes files from any input directory and saves results to any output directory.

For 'roles' (detected by substring in input path):
- pos_1: Mean activations where keys start with 'pos_' and score == 1
- default_1: Mean activations where keys start with 'default_' and score == 1  
- all_1: Mean of all activations scored 1 from all role files

For 'roles_240' (detected by substring in input path):
- pos_1: Mean activations where keys start with 'pos_' and score == 1
- default_1: SPECIAL CASE - Mean ALL activations from {int}_default.pt files (e.g., 0_default.pt, 1_default.pt, 2_default.pt, etc.)
- all_1: Mean of all activations scored 1 from role files + all default files

Key format: {pos|default}_q{question_index}_p{prompt_index}

Output: {output_dir}/default_activations.pt containing dict with:
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


def load_scores_and_activations(scores_dir: str, activations_dir: str) -> Tuple[Dict[str, Dict[str, Union[int, str]]], Dict[str, Dict[str, torch.Tensor]]]:
    """Load all score and activation files from directories."""
    scores_path = Path(scores_dir)
    activations_path = Path(activations_dir)
    
    if not scores_path.exists():
        raise FileNotFoundError(f"Scores directory not found: {scores_dir}")
    if not activations_path.exists():
        raise FileNotFoundError(f"Activations directory not found: {activations_dir}")
    
    # Load all score files
    all_scores = {}
    for score_file in scores_path.glob("*.json"):
        role_name = score_file.stem
        try:
            with open(score_file, 'r') as f:
                all_scores[role_name] = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load scores for {role_name}: {e}")
    
    # Load all activation files (including those without corresponding scores)
    all_activations = {}
    for activation_file in activations_path.glob("*.pt"):
        role_name = activation_file.stem
        try:
            all_activations[role_name] = torch.load(activation_file, map_location='cpu')
        except Exception as e:
            print(f"Warning: Failed to load activations for {role_name}: {e}")
    
    return all_scores, all_activations


def filter_activations_by_score(
    scores: Dict[str, Union[int, str]], 
    activations: Dict[str, torch.Tensor], 
    prefix: str, 
    target_score: int = 1
) -> List[torch.Tensor]:
    """Filter activations by prefix and score value."""
    filtered = []
    
    for key, activation in activations.items():
        key_prefix = parse_key_prefix(key)
        if key_prefix == prefix and key in scores:
            score_value = scores[key]
            if score_value == target_score:
                filtered.append(activation)
    
    return filtered


def calculate_mean_activations_roles(all_scores: Dict[str, Dict], all_activations: Dict[str, Dict]) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    """Calculate mean activations for 'roles' directory."""
    pos_activations = []
    default_activations = []
    all_activations_list = []
    
    # Process each role
    for role_name in all_scores.keys():
        if role_name not in all_activations:
            print(f"Warning: No activations found for role {role_name}")
            continue
            
        scores = all_scores[role_name]
        activations = all_activations[role_name]
        
        # Collect pos activations with score == 1
        pos_acts = filter_activations_by_score(scores, activations, 'pos', 1)
        pos_activations.extend(pos_acts)
        all_activations_list.extend(pos_acts)
        
        # Collect default activations with score == 1
        default_acts = filter_activations_by_score(scores, activations, 'default', 1)
        default_activations.extend(default_acts)
        all_activations_list.extend(default_acts)
    
    # Calculate means
    result = {}
    counts = {}
    
    if pos_activations:
        result['pos_1'] = torch.stack(pos_activations).mean(dim=0)
        counts['pos_1'] = len(pos_activations)
    else:
        print("Warning: No pos activations with score 1 found")
        
    if default_activations:
        result['default_1'] = torch.stack(default_activations).mean(dim=0)
        counts['default_1'] = len(default_activations)
    else:
        print("Warning: No default activations with score 1 found")
        
    if all_activations_list:
        result['all_1'] = torch.stack(all_activations_list).mean(dim=0)
        counts['all_1'] = len(all_activations_list)
    else:
        print("Warning: No activations with score 1 found")
    
    return result, counts


def calculate_mean_activations_roles_240(all_scores: Dict[str, Dict], all_activations: Dict[str, Dict]) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    """Calculate mean activations for 'roles_240' directory with special default handling."""
    pos_activations = []
    all_activations_list = []
    
    # Find all default files with pattern {int}_default
    default_file_pattern = re.compile(r'^\d+_default$')
    default_files = [name for name in all_activations.keys() if default_file_pattern.match(name)]
    
    # Process role files (same as roles directory for pos and all)
    for role_name in all_scores.keys():
        if role_name in default_files:
            continue  # Handle these separately
            
        if role_name not in all_activations:
            print(f"Warning: No activations found for role {role_name}")
            continue
            
        scores = all_scores[role_name]
        activations = all_activations[role_name]
        
        # Collect pos activations with score == 1
        pos_acts = filter_activations_by_score(scores, activations, 'pos', 1)
        pos_activations.extend(pos_acts)
        all_activations_list.extend(pos_acts)
        
        # Collect default activations with score == 1 for all_1
        default_acts = filter_activations_by_score(scores, activations, 'default', 1)
        all_activations_list.extend(default_acts)
    
    # Special handling for default_1: mean ALL activations from {int}_default.pt files
    # These files don't have scores, so we add ALL their activations
    default_activations = []
    for default_file in default_files:
        if default_file in all_activations:
            # Add ALL activations from these files (no score filtering)
            activations = all_activations[default_file]
            for key, activation in activations.items():
                default_activations.append(activation)
                # Also add to all_1
                all_activations_list.append(activation)
        else:
            print(f"Warning: Default file {default_file} not found in activations")
    
    if default_files:
        print(f"Found and processed {len(default_files)} default files: {default_files}")
    else:
        print("Warning: No default files with pattern {int}_default.pt found")
    
    # Calculate means
    result = {}
    counts = {}
    
    if pos_activations:
        result['pos_1'] = torch.stack(pos_activations).mean(dim=0)
        counts['pos_1'] = len(pos_activations)
    else:
        print("Warning: No pos activations with score 1 found")
        
    if default_activations:
        result['default_1'] = torch.stack(default_activations).mean(dim=0)
        counts['default_1'] = len(default_activations)
    else:
        print("Warning: No default activations found in {int}_default.pt files")
        
    if all_activations_list:
        result['all_1'] = torch.stack(all_activations_list).mean(dim=0)
        counts['all_1'] = len(all_activations_list)
    else:
        print("Warning: No activations found for all_1")
    
    return result, counts


def process_directory(input_dir: str, output_dir: str) -> None:
    """Process a directory with scores and activations."""
    print(f"\n=== Processing {input_dir} ===")
    
    scores_dir = os.path.join(input_dir, "extract_scores")
    activations_dir = os.path.join(input_dir, "response_activations")
    output_path = os.path.join(output_dir, "default_vectors.pt")
    
    # Determine processing type based on directory name
    is_roles_240 = "roles_240" in input_dir
    dir_type = "roles_240" if is_roles_240 else "roles"
    
    try:
        # Load data
        print("Loading scores and activations...")
        all_scores, all_activations = load_scores_and_activations(scores_dir, activations_dir)
        
        print(f"Loaded {len(all_scores)} score files and {len(all_activations)} activation files")
        
        # Calculate means based on directory type
        if is_roles_240:
            result, counts = calculate_mean_activations_roles_240(all_scores, all_activations)
        else:
            result, counts = calculate_mean_activations_roles(all_scores, all_activations)
        
        # Prepare output with metadata
        output = {
            'activations': result,
            'metadata': {
                'counts': counts,
                'directory_type': dir_type,
                'input_dir': input_dir,
                'output_dir': output_dir,
                'total_files_processed': {
                    'scores': len(all_scores),
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
        print(f"Error processing {input_dir}: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Calculate mean default activations across roles')
    parser.add_argument('--input-dir', type=str, default='/workspace/roles', 
                       help='Input directory containing extract_scores and response_activations (default: /workspace/roles)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for saving results (default: same as input-dir)')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Use input directory as output directory if not specified
    output_dir = args.output_dir if args.output_dir else args.input_dir
    
    print("=== Default Vector Calculator ===")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {output_dir}")
    
    try:
        process_directory(args.input_dir, output_dir)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Failed to process {args.input_dir}: {e}")
    
    print("\n=== Processing complete ===")


if __name__ == "__main__":
    main()