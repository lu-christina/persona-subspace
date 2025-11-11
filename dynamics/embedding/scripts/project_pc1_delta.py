#!/usr/bin/env python3
"""
PC1 Delta Extraction Script

This script processes conversation activations from multiple auditor models, projects them
onto role PC1 space, and computes PC1 deltas between consecutive assistant responses for
each user turn. Results are saved to a parquet file.

Usage:
    python embedding/scripts/project_pc1_delta.py \
        --base-dir /workspace/qwen-3-32b/dynamics \
        --auditor-models gpt-5,sonnet-4.5,kimi-k2 \
        --short-model qwen-3-32b \
        --pca-file /workspace/qwen-3-32b/roles_240/pca/layer32_pos23.pt \
        --layer 32 \
        --output-dir /root/git/persona-subspace/dynamics/results/qwen-3-32b
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import torch
import numpy as np
from glob import glob
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# Import PCA utilities
from utils.pca_utils import MeanScaler, L2MeanScaler


def pc_projection(mean_acts_per_turn, pca_results, n_pcs=1):
    """
    Project activations into PCA space.
    Based on the function from anomalies.ipynb.

    Args:
        mean_acts_per_turn: Activations tensor or list of tensors
        pca_results: Dictionary containing 'scaler' and 'pca' objects
        n_pcs: Number of principal components to return

    Returns:
        Projected activations of shape (n_samples, n_pcs)
    """
    if isinstance(mean_acts_per_turn, list):
        stacked_acts = torch.stack(mean_acts_per_turn)
    else:
        stacked_acts = mean_acts_per_turn
    stacked_acts = stacked_acts.float().numpy()
    scaled_acts = pca_results['scaler'].transform(stacked_acts)
    projected_acts = pca_results['pca'].transform(scaled_acts)
    return projected_acts[:, :n_pcs]


def load_pca_results(pca_file: str):
    """Load PCA results from file."""
    if not os.path.exists(pca_file):
        raise FileNotFoundError(f"PCA file not found: {pca_file}")

    try:
        pca_results = torch.load(pca_file, weights_only=False)
        print(f"Loaded PCA results from {pca_file}")
        return pca_results
    except Exception as e:
        raise RuntimeError(f"Error loading PCA results: {e}")


def process_conversation(
    activation_file: Path,
    transcript_file: Path,
    pca_results: Dict,
    layer: int,
    short_model: str,
    short_auditor_model: str
) -> List[Dict]:
    """
    Process a single conversation file and extract PC1 deltas.

    Args:
        activation_file: Path to .pt activation file
        transcript_file: Path to .json transcript file
        pca_results: PCA model dictionary
        layer: Layer number to extract
        short_model: Short model name
        short_auditor_model: Short auditor model name

    Returns:
        List of dictionaries containing PC1 delta data for each user turn
    """
    # Load activation file
    try:
        act_data = torch.load(activation_file, weights_only=False, map_location='cpu')
    except Exception as e:
        print(f"Error loading activation file {activation_file}: {e}")
        return []

    # Load transcript file
    try:
        with open(transcript_file, 'r') as f:
            transcript_data = json.load(f)
    except Exception as e:
        print(f"Error loading transcript file {transcript_file}: {e}")
        return []

    # Get conversation
    conversation = transcript_data.get('conversation', [])

    # Extract activations for the specified layer
    activations = act_data.get('activations')
    if activations is None:
        print(f"No activations found in {activation_file}")
        return []

    # Use minimum length if there's a mismatch
    n_turns = min(activations.shape[0], len(conversation))

    # Skip if usable length is too short (need at least 4 turns)
    if n_turns < 4:
        return []

    # Trim to usable length
    activations = activations[:n_turns]
    conversation = conversation[:n_turns]

    # Extract assistant turn activations (odd indices: 1, 3, 5, ...)
    assistant_indices = list(range(1, len(conversation), 2))
    assistant_activations = activations[assistant_indices, layer, :]

    # Project to PC1
    pc1_values = pc_projection(assistant_activations, pca_results, n_pcs=1).squeeze()

    # Ensure pc1_values is 1D array
    if pc1_values.ndim == 0:
        pc1_values = np.array([float(pc1_values)])

    # Extract metadata from transcript
    model = transcript_data.get('model', '')
    auditor_model = transcript_data.get('auditor_model', '')
    domain = transcript_data.get('domain', '')
    persona_id = transcript_data.get('persona_id', -1)
    topic_id = transcript_data.get('topic_id', -1)

    # Compute PC1 deltas for each user turn
    results = []

    # Iterate through user turns (even indices, excluding first and last)
    # User turns are at indices 0, 2, 4, 6, ...
    # We skip index 0 (no previous assistant) and need at least one more turn after
    for i in range(2, len(conversation) - 1, 2):
        # i is the user turn index
        # Previous assistant is at i-1, next assistant is at i+1
        prev_assistant_idx = (i - 1) // 2  # Convert to assistant list index
        next_assistant_idx = (i + 1) // 2  # Convert to assistant list index

        # Skip if indices are out of bounds
        if next_assistant_idx >= len(pc1_values):
            break

        prev_pc1 = float(pc1_values[prev_assistant_idx])
        next_pc1 = float(pc1_values[next_assistant_idx])
        pc1_delta = next_pc1 - prev_pc1

        results.append({
            'short_model': short_model,
            'short_auditor_model': short_auditor_model,
            'model': model,
            'auditor_model': auditor_model,
            'domain': domain,
            'persona_id': persona_id,
            'topic_id': topic_id,
            'response_id': i,  # User turn index
            'prev_pc1': prev_pc1,
            'next_pc1': next_pc1,
            'pc1_delta': pc1_delta
        })

    return results


def process_auditor_model(
    base_dir: Path,
    auditor_model: str,
    pca_results: Dict,
    layer: int,
    short_model: str
) -> List[Dict]:
    """
    Process all conversations for a single auditor model.

    Args:
        base_dir: Base directory containing auditor model subdirectories
        auditor_model: Name of auditor model
        pca_results: PCA model dictionary
        layer: Layer number to extract
        short_model: Short model name

    Returns:
        List of all PC1 delta records for this auditor model
    """
    short_auditor_model = auditor_model

    # Build paths
    activations_dir = base_dir / auditor_model / "default" / "activations"
    transcripts_dir = base_dir / auditor_model / "default" / "transcripts"

    # Check directories exist
    if not activations_dir.exists():
        print(f"Warning: Activations directory not found: {activations_dir}")
        return []

    if not transcripts_dir.exists():
        print(f"Warning: Transcripts directory not found: {transcripts_dir}")
        return []

    # Get all activation files
    activation_files = sorted(activations_dir.glob("*.pt"))

    if not activation_files:
        print(f"Warning: No .pt files found in {activations_dir}")
        return []

    print(f"\nProcessing {auditor_model}: {len(activation_files)} files")

    # Process all files
    all_results = []

    for activation_file in tqdm(activation_files, desc=f"Processing {auditor_model}"):
        # Find matching transcript file
        transcript_file = transcripts_dir / f"{activation_file.stem}.json"

        if not transcript_file.exists():
            print(f"Warning: No matching transcript for {activation_file.name}")
            continue

        # Process conversation
        results = process_conversation(
            activation_file,
            transcript_file,
            pca_results,
            layer,
            short_model,
            short_auditor_model
        )

        all_results.extend(results)

    print(f"Extracted {len(all_results)} PC1 delta records from {auditor_model}")

    return all_results


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Extract PC1 deltas from conversation activations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process all default auditor models
    python embedding/scripts/project_pc1_delta.py \\
        --base-dir /workspace/qwen-3-32b/dynamics \\
        --short-model qwen-3-32b \\
        --pca-file /workspace/qwen-3-32b/roles_240/pca/layer32_pos23.pt \\
        --output-dir /root/git/persona-subspace/dynamics/results/qwen-3-32b

    # Process specific auditor models
    python embedding/scripts/project_pc1_delta.py \\
        --base-dir /workspace/qwen-3-32b/dynamics \\
        --auditor-models gpt-5,kimi-k2 \\
        --short-model qwen-3-32b \\
        --pca-file /workspace/qwen-3-32b/roles_240/pca/layer32_pos23.pt \\
        --layer 32 \\
        --output-dir ./results
        """
    )

    parser.add_argument('--base-dir', type=str, required=True,
                       help='Base directory containing auditor model subdirectories')
    parser.add_argument('--auditor-models', type=str, default='gpt-5,sonnet-4.5,kimi-k2',
                       help='Comma-separated list of auditor models (default: gpt-5,sonnet-4.5,kimi-k2)')
    parser.add_argument('--short-model', type=str, required=True,
                       help='Short model name (e.g., qwen-3-32b)')
    parser.add_argument('--pca-file', type=str, required=True,
                       help='Path to PCA file')
    parser.add_argument('--layer', type=int, default=32,
                       help='Layer number to extract (default: 32)')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for parquet file')

    args = parser.parse_args()

    # Parse auditor models
    auditor_models = [m.strip() for m in args.auditor_models.split(',')]

    # Validate paths
    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        print(f"Error: Base directory does not exist: {base_dir}")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load PCA results
    print(f"Loading PCA results from {args.pca_file}")
    try:
        pca_results = load_pca_results(args.pca_file)
    except Exception as e:
        print(f"Error: {e}")
        return 1

    # Process each auditor model
    all_results = []

    for auditor_model in auditor_models:
        results = process_auditor_model(
            base_dir,
            auditor_model,
            pca_results,
            args.layer,
            args.short_model
        )
        all_results.extend(results)

    # Create DataFrame
    if not all_results:
        print("Warning: No results to save")
        return 1

    df = pd.DataFrame(all_results)

    # Save to parquet
    output_file = output_dir / "pc1_deltas.parquet"
    df.to_parquet(output_file, index=False)

    print(f"\nSaved {len(df)} records to {output_file}")
    print(f"\nSummary by auditor model:")
    print(df.groupby('short_auditor_model').size())

    return 0


if __name__ == "__main__":
    sys.exit(main())
