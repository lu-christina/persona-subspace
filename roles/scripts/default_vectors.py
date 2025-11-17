#!/usr/bin/env python3
"""
default_vectors.py - Calculate mean default activation across roles

This script calculates mean default activations from separate scores and activations directories.

Usage:
    python default_vectors.py --scores-dir <path> --activations-dir <path> [--output-dir <path>] [--processing-type roles_240]

Arguments:
    --scores-dir: Directory containing score JSON files (default: ./extract_scores)
    --activations-dir: Directory containing activation .pt files (default: ./response_activations)
    --output-dir: Output directory for results (default: current directory)
    --processing-type: Either 'roles' or 'roles_240' (default: roles_240)

For 'roles' processing:
- pos_1: Mean activations where keys start with 'pos_' and score == 1
- default_1: Mean activations where keys start with 'default_' and score == 1
- all_1: Mean of all activations scored 1 from all role files

For 'roles_240' processing (default):
- pos_1: Mean activations where keys start with 'pos_' and score == 1
- default_1: SPECIAL CASE - Mean ALL activations from {int}_default.pt files (e.g., 0_default.pt, 1_default.pt, 2_default.pt, etc.)
- all_1: Mean of all activations scored 1 from role files + all default files

Key format: {pos|default}_q{question_index}_p{prompt_index}

Output: {output_dir}/default_vectors.pt containing dict with:
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
import gc


def update_running_mean(running_mean: Optional[torch.Tensor], new_value: torch.Tensor, count: int) -> Tuple[torch.Tensor, int]:
    """
    Update running mean using Welford's online algorithm.

    Args:
        running_mean: Current running mean (None if first value)
        new_value: New tensor to incorporate
        count: Current count of values

    Returns:
        Tuple of (updated_mean, updated_count)
    """
    if running_mean is None:
        return new_value.clone().float(), 1
    else:
        count += 1
        running_mean = running_mean + (new_value.float() - running_mean) / count
        return running_mean, count


def parse_key_prefix(key: str) -> Optional[str]:
    """Extract pos/default prefix from keys like 'pos_q1_p2' or 'default_q3_p1'."""
    match = re.match(r'^(pos|default)_', key)
    return match.group(1) if match else None


def load_scores_and_activations(scores_dir: str, activations_dir: str) -> Tuple[Dict[str, Dict[str, Union[int, str]]], Dict[str, Path]]:
    """Load all score files and get activation file paths (without loading tensors)."""
    scores_path = Path(scores_dir)
    activations_path = Path(activations_dir)

    if not scores_path.exists():
        raise FileNotFoundError(f"Scores directory not found: {scores_dir}")
    if not activations_path.exists():
        raise FileNotFoundError(f"Activations directory not found: {activations_dir}")

    # Load all score files (these are small JSON files, ok to keep in memory)
    all_scores = {}
    for score_file in scores_path.glob("*.json"):
        role_name = score_file.stem
        try:
            with open(score_file, 'r') as f:
                all_scores[role_name] = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load scores for {role_name}: {e}")

    # Get activation file paths (don't load the tensors yet - streaming approach)
    activation_file_paths = {}
    for activation_file in activations_path.glob("*.pt"):
        role_name = activation_file.stem
        activation_file_paths[role_name] = activation_file

    return all_scores, activation_file_paths


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


def calculate_mean_activations_roles(all_scores: Dict[str, Dict], activation_file_paths: Dict[str, Path]) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    """Calculate mean activations for 'roles' directory with streaming."""
    # Initialize running means
    pos_mean = None
    pos_count = 0
    default_mean = None
    default_count = 0
    all_mean = None
    all_count = 0

    # Process each role with streaming
    print("Processing role files...")
    for role_name in tqdm(all_scores.keys(), desc="Role files"):
        if role_name not in activation_file_paths:
            print(f"Warning: No activations found for role {role_name}")
            continue

        # Load activations for this file only
        try:
            activations = torch.load(activation_file_paths[role_name], map_location='cpu')
        except Exception as e:
            print(f"Warning: Failed to load activations for {role_name}: {e}")
            continue

        scores = all_scores[role_name]

        # Process activations with streaming
        for key, activation in activations.items():
            key_prefix = parse_key_prefix(key)
            if key_prefix is None or key not in scores or scores[key] != 1:
                continue

            # Update running means
            if key_prefix == 'pos':
                pos_mean, pos_count = update_running_mean(pos_mean, activation, pos_count)
                all_mean, all_count = update_running_mean(all_mean, activation, all_count)
            elif key_prefix == 'default':
                default_mean, default_count = update_running_mean(default_mean, activation, default_count)
                all_mean, all_count = update_running_mean(all_mean, activation, all_count)

        # Clear memory
        del activations
        gc.collect()

    # Prepare results
    result = {}
    counts = {}

    if pos_mean is not None:
        result['pos_1'] = pos_mean
        counts['pos_1'] = pos_count
    else:
        print("Warning: No pos activations with score 1 found")

    if default_mean is not None:
        result['default_1'] = default_mean
        counts['default_1'] = default_count
    else:
        print("Warning: No default activations with score 1 found")

    if all_mean is not None:
        result['all_1'] = all_mean
        counts['all_1'] = all_count
    else:
        print("Warning: No activations with score 1 found")

    return result, counts


def calculate_mean_activations_roles_240(all_scores: Dict[str, Dict], activation_file_paths: Dict[str, Path]) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    """Calculate mean activations for 'roles_240' directory with streaming and special default handling."""
    # Initialize running means
    pos_mean = None
    pos_count = 0
    default_mean = None
    default_count = 0
    all_mean = None
    all_count = 0

    # Find all default files with pattern {int}_default
    default_file_pattern = re.compile(r'^\d+_default$')
    default_files = [name for name in activation_file_paths.keys() if default_file_pattern.match(name)]

    # Process role files (same as roles directory for pos and all) with streaming
    print("Processing role files...")
    for role_name in tqdm(all_scores.keys(), desc="Role files"):
        if role_name in default_files:
            continue  # Handle these separately

        if role_name not in activation_file_paths:
            print(f"Warning: No activations found for role {role_name}")
            continue

        # Load activations for this file only
        try:
            activations = torch.load(activation_file_paths[role_name], map_location='cpu')
        except Exception as e:
            print(f"Warning: Failed to load activations for {role_name}: {e}")
            continue

        scores = all_scores[role_name]

        # Process pos activations with score == 1
        for key, activation in activations.items():
            key_prefix = parse_key_prefix(key)
            if key_prefix == 'pos' and key in scores and scores[key] == 1:
                pos_mean, pos_count = update_running_mean(pos_mean, activation, pos_count)
                all_mean, all_count = update_running_mean(all_mean, activation, all_count)

            # Process default activations with score == 1 for all_1
            elif key_prefix == 'default' and key in scores and scores[key] == 1:
                all_mean, all_count = update_running_mean(all_mean, activation, all_count)

        # Clear memory
        del activations
        gc.collect()

    # Special handling for default_1: mean ALL activations from {int}_default.pt files
    # These files don't have scores, so we add ALL their activations
    if default_files:
        print(f"Found {len(default_files)} default files, processing with streaming...")
        for default_file in tqdm(default_files, desc="Default files"):
            if default_file not in activation_file_paths:
                print(f"Warning: Default file {default_file} not found in activations")
                continue

            # Load activations for this file only
            try:
                activations = torch.load(activation_file_paths[default_file], map_location='cpu')
            except Exception as e:
                print(f"Warning: Failed to load activations for {default_file}: {e}")
                continue

            # Add ALL activations from these files (no score filtering)
            for key, activation in activations.items():
                default_mean, default_count = update_running_mean(default_mean, activation, default_count)
                # Also add to all_1
                all_mean, all_count = update_running_mean(all_mean, activation, all_count)

            # Clear memory after each file
            del activations
            gc.collect()
    else:
        print("Warning: No default files with pattern {int}_default.pt found")

    # Prepare results
    result = {}
    counts = {}

    if pos_mean is not None:
        result['pos_1'] = pos_mean
        counts['pos_1'] = pos_count
    else:
        print("Warning: No pos activations with score 1 found")

    if default_mean is not None:
        result['default_1'] = default_mean
        counts['default_1'] = default_count
    else:
        print("Warning: No default activations found in {int}_default.pt files")

    if all_mean is not None:
        result['all_1'] = all_mean
        counts['all_1'] = all_count
    else:
        print("Warning: No activations found for all_1")

    return result, counts


def process_directory(scores_dir: str, activations_dir: str, output_dir: str, processing_type: str = 'roles_240') -> None:
    """Process scores and activations directories.

    Args:
        scores_dir: Directory containing score JSON files
        activations_dir: Directory containing activation .pt files
        output_dir: Directory to save results
        processing_type: Either 'roles' or 'roles_240' (default: 'roles_240')
    """
    print(f"\n=== Processing with {processing_type} method ===")

    output_path = os.path.join(output_dir, "default_vectors.pt")

    try:
        # Load data (streaming approach - only get file paths)
        print("Loading scores and getting activation file paths...")
        all_scores, activation_file_paths = load_scores_and_activations(scores_dir, activations_dir)

        print(f"Loaded {len(all_scores)} score files and found {len(activation_file_paths)} activation files")

        # Calculate means based on processing type (with streaming)
        if processing_type == 'roles_240':
            result, counts = calculate_mean_activations_roles_240(all_scores, activation_file_paths)
        else:
            result, counts = calculate_mean_activations_roles(all_scores, activation_file_paths)

        # Prepare output with metadata
        output = {
            'activations': result,
            'metadata': {
                'counts': counts,
                'directory_type': processing_type,
                'scores_dir': scores_dir,
                'activations_dir': activations_dir,
                'output_dir': output_dir,
                'total_files_processed': {
                    'scores': len(all_scores),
                    'activations': len(activation_file_paths)
                }
            }
        }

        # Save results
        os.makedirs(output_dir, exist_ok=True)
        torch.save(output, output_path)

        print(f"Results saved to: {output_path}")
        print("Sample counts:")
        for key, count in counts.items():
            tensor_shape = result[key].shape if key in result else "N/A"
            print(f"  {key}: {count} samples, shape: {tensor_shape}")

    except Exception as e:
        print(f"Error processing directories: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Calculate mean default activations across roles')
    parser.add_argument('--scores-dir', type=str, default='./extract_scores',
                       help='Directory containing score JSON files (default: ./extract_scores)')
    parser.add_argument('--activations-dir', type=str, default='./response_activations',
                       help='Directory containing activation .pt files (default: ./response_activations)')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Output directory for saving results (default: current directory)')
    parser.add_argument('--processing-type', type=str, default='roles_240', choices=['roles', 'roles_240'],
                       help='Processing type: "roles" or "roles_240" (default: roles_240)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')

    args = parser.parse_args()

    print("=== Default Vector Calculator ===")
    print(f"Scores directory: {args.scores_dir}")
    print(f"Activations directory: {args.activations_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Processing type: {args.processing_type}")

    try:
        process_directory(args.scores_dir, args.activations_dir, args.output_dir, args.processing_type)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Failed to process directories: {e}")

    print("\n=== Processing complete ===")


if __name__ == "__main__":
    main()