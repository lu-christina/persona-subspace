#!/usr/bin/env python3
"""
5_vectors.py - Generate role vectors from response activations and labels

Creates mean activation vectors for each role based on 0-3 labeling system:
1. pos_0, pos_1, pos_2, pos_3: mean activations for pos prompt type by label
2. default_0, default_1, default_2, default_3: mean activations for default prompt type by label  
3. pos_all: mean of all pos activations regardless of label
4. default_all: mean of all default activations regardless of label

Only vectors with >= min_count_threshold activations are included in output.
Each vector has shape (n_layers, hidden_dim) = (46, 4608)
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Union
import torch
from tqdm import tqdm


def load_data(role: str, activations_base_path: str, scores_base_path: str) -> Tuple[Optional[Dict[str, torch.Tensor]], Optional[Dict[str, Union[int, str]]]]:
    """Load activation tensors and scores for a given role."""
    activations_path = f"{activations_base_path}/{role}.pt"
    scores_path = f"{scores_base_path}/{role}.json"
    
    try:
        activations = torch.load(activations_path, map_location='cpu')
        with open(scores_path, 'r') as f:
            scores = json.load(f)
        return activations, scores
    except FileNotFoundError as e:
        print(f"Warning: Missing file for {role}: {e}")
        return None, None
    except Exception as e:
        print(f"Error loading data for {role}: {e}")
        return None, None


def compute_vectors(
    activations: Dict[str, torch.Tensor], 
    scores: Dict[str, Union[int, str]],
    min_count_threshold: int
) -> Dict[str, torch.Tensor]:
    """Compute mean activation vectors grouped by prompt type and label."""
    
    # Collect activations by prompt type and label
    groups = {
        'pos': {0: [], 1: [], 2: [], 3: []},
        'default': {0: [], 1: [], 2: [], 3: []}
    }
    
    # Iterate through all activations and group by prompt type and label
    for key, activation in activations.items():
        if key not in scores:
            continue
            
        score = scores[key]
        
        # Skip REFUSAL scores
        if score == "REFUSAL":
            continue
            
        # Parse key to get prompt type
        if key.startswith('pos_'):
            prompt_type = 'pos'
        elif key.startswith('default_'):
            prompt_type = 'default'
        else:
            continue
            
        # Ensure score is valid (0-3)
        if score not in [0, 1, 2, 3]:
            continue
            
        groups[prompt_type][score].append(activation)
    
    # Compute mean vectors for groups that meet threshold
    def compute_mean_activation(activation_list: List[torch.Tensor]) -> torch.Tensor:
        if not activation_list:
            return torch.zeros(46, 4608)
        return torch.stack(activation_list).mean(dim=0)
    
    # Build result dictionary
    result = {}
    
    # Add vectors for each prompt type and label
    for prompt_type in ['pos', 'default']:
        # Individual label vectors
        for label in [0, 1, 2, 3]:
            activation_list = groups[prompt_type][label]
            if len(activation_list) >= min_count_threshold:
                vector_key = f"{prompt_type}_{label}"
                result[vector_key] = compute_mean_activation(activation_list)
        
        # "All" vector for this prompt type
        all_activations = []
        for label_activations in groups[prompt_type].values():
            all_activations.extend(label_activations)
        
        if len(all_activations) >= min_count_threshold:
            vector_key = f"{prompt_type}_all"
            result[vector_key] = compute_mean_activation(all_activations)
    
    return result


def process_role(role: str, min_count_threshold: int, activations_base_path: str, scores_base_path: str, output_base_path: str) -> bool:
    """Process a single role and save vectors."""
    activations, scores = load_data(role, activations_base_path, scores_base_path)
    
    if activations is None or scores is None:
        return False
    
    vectors = compute_vectors(activations, scores, min_count_threshold)
    
    # Save vectors
    output_path = f"{output_base_path}/{role}.pt"
    torch.save(vectors, output_path)
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate role vectors from activations and scores")
    parser.add_argument("--min_count_threshold", type=int, default=50,
                       help="Minimum number of activations needed to create a vector (default: 50)")
    parser.add_argument("--activations_path", type=str, default="/workspace/roles/response_activations",
                       help="Path to directory containing activation tensors (default: /workspace/roles/response_activations)")
    parser.add_argument("--scores_path", type=str, default="/workspace/roles/extract_scores", 
                       help="Path to directory containing score labels (default: /workspace/roles/extract_scores)")
    parser.add_argument("--output_path", type=str, default="/workspace/roles/vectors",
                       help="Path to directory for saving vectors (default: /workspace/roles/vectors)")
    parser.add_argument("--roles", nargs="+", 
                       help="Specific roles to process (default: all roles)")
    parser.add_argument("--list_roles", action="store_true",
                       help="List available roles and exit")
    
    args = parser.parse_args()
    
    # Get list of available roles from activations directory
    activations_dir = Path(args.activations_path)
    available_roles = [f.stem for f in activations_dir.glob("*.pt")]
    available_roles.sort()
    
    if args.list_roles:
        print(f"Available roles ({len(available_roles)}):")
        for role in available_roles:
            print(f"  {role}")
        return
    
    # Determine which roles to process
    if args.roles:
        roles_to_process = [t for t in args.roles if t in available_roles]
        if len(roles_to_process) != len(args.roles):
            missing = set(args.roles) - set(roles_to_process)
            print(f"Warning: Roles not found: {missing}")
    else:
        roles_to_process = available_roles
    
    print(f"Processing {len(roles_to_process)} roles with settings:")
    print(f"  min_count_threshold: {args.min_count_threshold}")
    print(f"  activations_path: {args.activations_path}")
    print(f"  scores_path: {args.scores_path}")
    print(f"  output_path: {args.output_path}")
    
    # Ensure output directory exists
    os.makedirs(args.output_path, exist_ok=True)
    
    # Process roles with progress bar
    successful = 0
    failed = 0
    
    for role in tqdm(roles_to_process, desc="Processing roles"):
        if process_role(role, args.min_count_threshold, args.activations_path, args.scores_path, args.output_path):
            successful += 1
        else:
            failed += 1
    
    print(f"\nCompleted: {successful} successful, {failed} failed")
    print(f"Vectors saved to {args.output_path}")


if __name__ == "__main__":
    main()