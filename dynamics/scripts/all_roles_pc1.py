#!/usr/bin/env python3
"""
All Roles Prefill Script

This script takes the first 10 questions from roles_pc1_questions.csv, iterates through all
role response files to find matching questions with prompt_id=0, calculates PC1 projections
using PCA results, and outputs to a JSONL file with incremental writing.

Usage:
    uv run scripts/all_roles_prefill.py \
        --basedir /workspace/qwen-3-32b/roles_240 \
        --layer 32 \
        --output_file /root/git/persona-subspace/dynamics/results/qwen-3-32b/prefills/role_pc1_prefills_q132.jsonl
"""

import argparse
import json
import os
import sys
from pathlib import Path
import pandas as pd
import torch
import numpy as np
from glob import glob

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))


def pc_projection(mean_acts_per_turn, pca_results, n_pcs=8):
    """
    Project activations into PCA space.
    Based on the function from anomalies.ipynb.
    """
    if isinstance(mean_acts_per_turn, list):
        stacked_acts = torch.stack(mean_acts_per_turn)
    else:
        stacked_acts = mean_acts_per_turn
    stacked_acts = stacked_acts.float().numpy()
    scaled_acts = pca_results['scaler'].transform(stacked_acts)
    projected_acts = pca_results['pca'].transform(scaled_acts)
    return projected_acts[:, :n_pcs]


def load_first_10_questions(csv_path):
    """Load the first 10 questions from the CSV file."""
    try:
        df = pd.read_csv(csv_path)
        first_10_questions = df.head(10)['question_id'].tolist()
        print(f"Loaded first 10 questions: {first_10_questions}")
        return first_10_questions
    except Exception as e:
        print(f"Error loading CSV file {csv_path}: {e}")
        return None


def get_all_roles(basedir):
    """Get all role names from response files."""
    response_dir = os.path.join(basedir, "responses")
    if not os.path.exists(response_dir):
        print(f"Response directory not found: {response_dir}")
        return []

    role_files = glob(os.path.join(response_dir, "*.jsonl"))
    all_roles = [os.path.splitext(os.path.basename(f))[0] for f in role_files]

    # Filter out roles containing "default"
    roles = [role for role in all_roles if "default" not in role]
    print(f"Found {len(roles)} roles (filtered out {len(all_roles) - len(roles)} default roles)")
    return sorted(roles)


def load_pca_results(basedir, layer):
    """Load PCA results for the specified layer."""
    pca_path = os.path.join(basedir, "pca", f"layer{layer}_pos23.pt")
    if not os.path.exists(pca_path):
        print(f"PCA file not found: {pca_path}")
        return None

    try:
        pca_results = torch.load(pca_path, weights_only=False)
        print(f"Loaded PCA results from {pca_path}")
        return pca_results
    except Exception as e:
        print(f"Error loading PCA results: {e}")
        return None


def process_role(role, question_ids, basedir, layer, pca_results, current_id):
    """Process a single role and return list of entries for JSONL output."""
    entries = []

    # Load response file
    response_path = os.path.join(basedir, "responses", f"{role}.jsonl")
    if not os.path.exists(response_path):
        print(f"Warning: Response file not found for role {role}: {response_path}")
        return entries, current_id

    # Load activation file
    activation_path = os.path.join(basedir, "response_activations", f"{role}.pt")
    if not os.path.exists(activation_path):
        print(f"Warning: Activation file not found for role {role}: {activation_path}")
        return entries, current_id

    try:
        # Load activations
        acts = torch.load(activation_path, weights_only=False)

        # Load responses
        responses = {}
        with open(response_path, 'r') as f:
            for line in f:
                obj = json.loads(line)
                key = (obj['prompt_index'], obj['question_index'])
                responses[key] = obj

        # Collect all valid activations and metadata for batch processing
        activations_to_project = []
        valid_entries = []

        # First pass: validate and collect valid data
        for question_id in question_ids:
            key = (0, question_id)  # prompt_index=0, question_index from CSV
            label = f"pos_p0_q{question_id}"

            if key not in responses:
                print(f"Warning: No response found for {role}, prompt_id=0, question_id={question_id}")
                continue

            if label not in acts:
                print(f"Warning: No activation found for {role}, label={label}")
                continue

            # Collect activation and metadata for batch processing
            activations_to_project.append(acts[label][layer, :])
            valid_entries.append((question_id, label, responses[key]['conversation']))

        # Batch projection (single PCA call for all activations)
        if activations_to_project:
            stacked_activations = torch.stack(activations_to_project)
            projections = pc_projection(stacked_activations, pca_results, 1)

            # Create entries using pre-computed projections
            for i, (question_id, label, conversation) in enumerate(valid_entries):
                pc1_value = float(projections[i].squeeze())

                entry = {
                    'id': current_id,
                    'role': role,
                    'question_index': question_id,
                    'prompt_index': 0,
                    'label': label,
                    'pc1': pc1_value,
                    'conversation': conversation
                }

                entries.append(entry)
                current_id += 1

        print(f"Processed {len(entries)} entries for role: {role}")

    except Exception as e:
        print(f"Error processing role {role}: {e}")

    return entries, current_id


def main():
    parser = argparse.ArgumentParser(description="Generate role prefills with PC1 projections")
    parser.add_argument("--basedir", default="/workspace/qwen-3-32b/roles_240",
                       help="Path to roles_240 directory containing role data")
    parser.add_argument("--layer", type=int, default=32,
                       help="Layer number for PCA analysis")
    parser.add_argument("--questions_ranked_file",
                       default="/root/git/persona-subspace/dynamics/results/qwen-3-32b/prefills/roles_pc1_questions.csv",
                       help="CSV file containing ranked questions")
    parser.add_argument("--output_file", required=True,
                       help="Output JSONL file path")

    args = parser.parse_args()

    # Validate base directory
    if not os.path.exists(args.basedir):
        print(f"Error: Base directory does not exist: {args.basedir}")
        return 1

    # Load first 10 questions from CSV
    question_ids = load_first_10_questions(args.questions_ranked_file)
    if question_ids is None:
        return 1

    # Load PCA results
    pca_results = load_pca_results(args.basedir, args.layer)
    if pca_results is None:
        return 1

    # Get all roles
    roles = get_all_roles(args.basedir)
    if not roles:
        print("No roles found!")
        return 1

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Process roles and write incrementally
    current_id = 0
    total_entries = 0

    with open(args.output_file, 'w') as output_file:
        for i, role in enumerate(roles):
            print(f"Processing role {i+1}/{len(roles)}: {role}")
            entries, current_id = process_role(role, question_ids, args.basedir, args.layer, pca_results, current_id)

            # Write entries for this role immediately
            for entry in entries:
                output_file.write(json.dumps(entry) + '\n')
                output_file.flush()  # Ensure data is written to disk

            total_entries += len(entries)

    print(f"\nCompleted! Processed {len(roles)} roles and wrote {total_entries} entries to {args.output_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())