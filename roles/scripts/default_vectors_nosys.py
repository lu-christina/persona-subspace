#!/usr/bin/env python3
"""
default_vectors_nosys.py - Calculate mean default activation for no-system-prompt keys only

This script calculates mean activations from {int}_default.pt files using only
keys matching 'default_p0_q*' (the no-system-prompt variant), and adds the result
to an existing default_vectors.pt file.

Usage:
    python default_vectors_nosys.py --activations-dir <path> --output-dir <path>

Arguments:
    --activations-dir: Directory containing activation .pt files (default: ./response_activations)
    --output-dir: Directory containing existing default_vectors.pt (default: current directory)

Processing:
- Finds all {int}_default.pt files (e.g., 0_default.pt, 1_default.pt, etc.)
- For each file, only processes keys matching 'default_p0_q*'
- Calculates running mean across all matching activations
- Loads existing default_vectors.pt and adds 'default_nosys' key
- Preserves all existing values (pos_1, default_1, all_1)

Output: Updates {output_dir}/default_vectors.pt with:
- 'default_nosys' -> tensor of shape (num_layers, hidden_dims)
- Updated metadata with sample counts
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Tuple, Optional
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


def is_nosys_key(key: str) -> bool:
    """Check if key is a no-system-prompt key (default_p0_q*)."""
    return key.startswith('default_p0_')


def get_activation_file_paths(activations_dir: str) -> Dict[str, Path]:
    """Get paths to all activation files."""
    activations_path = Path(activations_dir)

    if not activations_path.exists():
        raise FileNotFoundError(f"Activations directory not found: {activations_dir}")

    activation_file_paths = {}
    for activation_file in activations_path.glob("*.pt"):
        role_name = activation_file.stem
        activation_file_paths[role_name] = activation_file

    return activation_file_paths


def calculate_mean_activations_nosys(activation_file_paths: Dict[str, Path]) -> Tuple[Optional[torch.Tensor], int]:
    """
    Calculate mean activations from {int}_default.pt files using only default_p0_q* keys.

    Args:
        activation_file_paths: Dict mapping file names to paths

    Returns:
        Tuple of (mean_tensor, count)
    """
    # Initialize running mean
    nosys_mean = None
    nosys_count = 0

    # Find all default files with pattern {int}_default
    default_file_pattern = re.compile(r'^\d+_default$')
    default_files = [name for name in activation_file_paths.keys() if default_file_pattern.match(name)]

    if not default_files:
        print("Warning: No default files with pattern {int}_default.pt found")
        return None, 0

    print(f"Found {len(default_files)} default files, processing with streaming...")
    for default_file in tqdm(sorted(default_files), desc="Default files"):
        if default_file not in activation_file_paths:
            print(f"Warning: Default file {default_file} not found in activations")
            continue

        # Load activations for this file only
        try:
            activations = torch.load(activation_file_paths[default_file], map_location='cpu', weights_only=False)
        except Exception as e:
            print(f"Warning: Failed to load activations for {default_file}: {e}")
            continue

        # Only process keys matching default_p0_q*
        for key, activation in activations.items():
            if is_nosys_key(key):
                nosys_mean, nosys_count = update_running_mean(nosys_mean, activation, nosys_count)

        # Clear memory after each file
        del activations
        gc.collect()

    return nosys_mean, nosys_count


def process_directory(activations_dir: str, output_dir: str) -> None:
    """
    Process activations directory and update default_vectors.pt with nosys results.

    Args:
        activations_dir: Directory containing activation .pt files
        output_dir: Directory containing existing default_vectors.pt
    """
    print("\n=== Processing nosys activations ===")

    output_path = os.path.join(output_dir, "default_vectors.pt")

    # Check if existing file exists
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Existing default_vectors.pt not found at: {output_path}")

    try:
        # Get activation file paths
        print("Getting activation file paths...")
        activation_file_paths = get_activation_file_paths(activations_dir)
        print(f"Found {len(activation_file_paths)} activation files")

        # Calculate nosys mean
        nosys_mean, nosys_count = calculate_mean_activations_nosys(activation_file_paths)

        if nosys_mean is None:
            print("Error: No nosys activations found")
            return

        print(f"Calculated mean from {nosys_count} nosys activations")

        # Load existing default_vectors.pt
        print(f"Loading existing file: {output_path}")
        existing_data = torch.load(output_path, map_location='cpu', weights_only=False)

        # Add nosys results
        existing_data['activations']['default_nosys'] = nosys_mean
        existing_data['metadata']['counts']['default_nosys'] = nosys_count

        # Save back
        torch.save(existing_data, output_path)

        print(f"Results saved to: {output_path}")
        print(f"Added 'default_nosys': {nosys_count} samples, shape: {nosys_mean.shape}")

    except Exception as e:
        print(f"Error processing directories: {e}")
        raise


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Calculate mean nosys default activations')
    parser.add_argument('--activations-dir', type=str, default='./response_activations',
                       help='Directory containing activation .pt files (default: ./response_activations)')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Directory containing existing default_vectors.pt (default: current directory)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')

    args = parser.parse_args()

    print("=== Default Vector Calculator (nosys only) ===")
    print(f"Activations directory: {args.activations_dir}")
    print(f"Output directory: {args.output_dir}")

    try:
        process_directory(args.activations_dir, args.output_dir)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Failed to process directories: {e}")

    print("\n=== Processing complete ===")


if __name__ == "__main__":
    main()
