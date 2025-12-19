#!/usr/bin/env python3
"""
Add Contrast Vector Projections to Prefills JSONL

This script calculates contrast vector projections for each entry in all_roles.jsonl
by loading activations one role at a time and computing dot products with the
contrast vector at a specified layer.

Usage:
    uv run dynamics/scripts/all_roles_contrast.py \
        --input_file dynamics/results/qwen-3-32b/prefills/all_roles.jsonl \
        --activations_dir /workspace/qwen-3-32b/roles_240/response_activations \
        --contrast_vectors /workspace/qwen-3-32b/roles_240/contrast_vectors.pt \
        --layer 32
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Add contrast vector projections to prefills JSONL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to input all_roles.jsonl file"
    )

    parser.add_argument(
        "--activations_dir",
        type=str,
        required=True,
        help="Path to response_activations directory"
    )

    parser.add_argument(
        "--contrast_vectors",
        type=str,
        required=True,
        help="Path to contrast_vectors.pt file"
    )

    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Layer number for projection"
    )

    parser.add_argument(
        "--field_name",
        type=str,
        default="contrast",
        help="Name of the output field to add"
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file path (default: overwrite input file)"
    )

    return parser.parse_args()


def load_entries(input_file: str) -> list:
    """Load all entries from JSONL file."""
    entries = []
    with open(input_file, 'r') as f:
        for line in f:
            entries.append(json.loads(line.strip()))
    return entries


def group_by_role(entries: list) -> dict:
    """Group entry indices by role for efficient processing."""
    role_to_indices = defaultdict(list)
    for idx, entry in enumerate(entries):
        role_to_indices[entry['role']].append(idx)
    return role_to_indices


def main():
    args = parse_arguments()

    # Validate paths
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        return 1

    if not os.path.exists(args.activations_dir):
        print(f"Error: Activations directory not found: {args.activations_dir}")
        return 1

    if not os.path.exists(args.contrast_vectors):
        print(f"Error: Contrast vectors file not found: {args.contrast_vectors}")
        return 1

    output_file = args.output_file or args.input_file

    # Load contrast vectors and extract the layer we need
    print(f"Loading contrast vectors from {args.contrast_vectors}")
    contrast_vectors = torch.load(args.contrast_vectors, weights_only=False)
    cv = F.normalize(contrast_vectors[args.layer].float(), dim=0)
    print(f"Using layer {args.layer} contrast vector, shape: {cv.shape}")

    # Load all entries
    print(f"Loading entries from {args.input_file}")
    entries = load_entries(args.input_file)
    print(f"Loaded {len(entries)} entries")

    # Group by role
    role_to_indices = group_by_role(entries)
    print(f"Found {len(role_to_indices)} unique roles")

    # Process each role
    processed = 0
    missing = 0

    for role_idx, (role, indices) in enumerate(sorted(role_to_indices.items())):
        activation_file = os.path.join(args.activations_dir, f"{role}.pt")

        if not os.path.exists(activation_file):
            print(f"Warning: Activation file not found for role '{role}': {activation_file}")
            missing += len(indices)
            continue

        # Load activations for this role
        acts = torch.load(activation_file, weights_only=False)

        # Process each entry for this role
        for idx in indices:
            entry = entries[idx]
            label = entry['label']

            if label not in acts:
                print(f"Warning: Label '{label}' not found in activations for role '{role}'")
                missing += 1
                continue

            # Get activation at the specified layer and compute projection
            activation = acts[label][args.layer, :].float()
            projection = float(activation @ cv)

            # Add to entry
            entry[args.field_name] = projection
            processed += 1

        # Free memory
        del acts

        print(f"Processed {role_idx + 1}/{len(role_to_indices)} roles...")

    print(f"Processed {processed} entries, {missing} missing")

    # Write output
    print(f"Writing to {output_file}")
    with open(output_file, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')

    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
