#!/usr/bin/env python3
"""
All Roles PCA Script

This script takes the first 10 questions from roles_pc1_questions.csv, iterates through all
role response files to find matching questions with prompt_id=0, calculates full PCA projections
for both role and trait PCAs, and outputs individual .pt files with incremental IDs.

Usage:
    uv run scripts/all_roles_pca.py \
        --roles-basedir /workspace/qwen-3-32b/roles_240 \
        --traits-basedir /workspace/qwen-3-32b/traits_240 \
        --layer 32 \
        --output-dir /root/git/persona-subspace/dynamics/results/qwen-3-32b/pca_projections
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

    If n_pcs is None, returns the full transformed vector.
    """
    if isinstance(mean_acts_per_turn, list):
        stacked_acts = torch.stack(mean_acts_per_turn)
    else:
        stacked_acts = mean_acts_per_turn
    stacked_acts = stacked_acts.float().numpy()
    scaled_acts = pca_results['scaler'].transform(stacked_acts)
    projected_acts = pca_results['pca'].transform(scaled_acts)

    if n_pcs is None:
        return projected_acts
    else:
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


def get_all_roles(roles_basedir):
    """Get all role names from response files."""
    response_dir = os.path.join(roles_basedir, "responses")
    if not os.path.exists(response_dir):
        print(f"Response directory not found: {response_dir}")
        return []

    role_files = glob(os.path.join(response_dir, "*.jsonl"))
    all_roles = [os.path.splitext(os.path.basename(f))[0] for f in role_files]

    # Filter out roles containing "default"
    roles = [role for role in all_roles if "default" not in role]
    print(f"Found {len(roles)} roles (filtered out {len(all_roles) - len(roles)} default roles)")
    return sorted(roles)


def load_pca_results(basedir, layer, pca_type):
    """Load PCA results for the specified layer and type."""
    if pca_type == "role":
        pca_path = os.path.join(basedir, "pca", f"layer{layer}_pos23.pt")
    elif pca_type == "trait":
        pca_path = os.path.join(basedir, "pca", f"layer{layer}_pos-neg50.pt")
    else:
        raise ValueError(f"Unknown PCA type: {pca_type}")

    if not os.path.exists(pca_path):
        print(f"PCA file not found: {pca_path}")
        return None

    try:
        pca_results = torch.load(pca_path, weights_only=False)
        print(f"Loaded {pca_type} PCA results from {pca_path}")
        return pca_results
    except Exception as e:
        print(f"Error loading {pca_type} PCA results: {e}")
        return None


def process_role(role, question_ids, roles_basedir, layer, role_pca_results, trait_pca_results, output_dir, current_id):
    """Process a single role and save individual .pt files for each entry."""
    entries_saved = 0

    # Load response file
    response_path = os.path.join(roles_basedir, "responses", f"{role}.jsonl")
    if not os.path.exists(response_path):
        print(f"Warning: Response file not found for role {role}: {response_path}")
        return entries_saved, current_id

    # Load activation file
    activation_path = os.path.join(roles_basedir, "response_activations", f"{role}.pt")
    if not os.path.exists(activation_path):
        print(f"Warning: Activation file not found for role {role}: {activation_path}")
        return entries_saved, current_id

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
            role_projections = pc_projection(stacked_activations, role_pca_results, n_pcs=None)
            trait_projections = pc_projection(stacked_activations, trait_pca_results, n_pcs=None)

            # Create and save individual .pt files using pre-computed projections
            for i, (question_id, label, conversation) in enumerate(valid_entries):
                role_pca_transformed = role_projections[i]
                trait_pca_transformed = trait_projections[i]

                entry = {
                    'id': current_id,
                    'role': role,
                    'question_index': question_id,
                    'prompt_index': 0,
                    'label': label,
                    'role_pca_transformed': role_pca_transformed,
                    'trait_pca_transformed': trait_pca_transformed,
                    'conversation': conversation
                }

                # Save individual .pt file
                output_path = os.path.join(output_dir, f"{current_id}.pt")
                torch.save(entry, output_path)

                entries_saved += 1
                current_id += 1

        print(f"Processed and saved {entries_saved} entries for role: {role}")

    except Exception as e:
        print(f"Error processing role {role}: {e}")

    return entries_saved, current_id


def main():
    parser = argparse.ArgumentParser(description="Generate role projections with full PCA transformations")
    parser.add_argument("--roles-basedir", default="/workspace/qwen-3-32b/roles_240",
                       help="Path to roles_240 directory containing role data")
    parser.add_argument("--traits-basedir", default="/workspace/qwen-3-32b/traits_240",
                       help="Path to traits_240 directory containing trait PCA data")
    parser.add_argument("--layer", type=int, default=32,
                       help="Layer number for PCA analysis")
    parser.add_argument("--questions-ranked-file",
                       default="/root/git/persona-subspace/dynamics/results/qwen-3-32b/prefills/roles_pc1_questions.csv",
                       help="CSV file containing ranked questions")
    parser.add_argument("--output-dir", required=True,
                       help="Output directory for .pt files")

    args = parser.parse_args()

    # Validate directories
    if not os.path.exists(args.roles_basedir):
        print(f"Error: Roles base directory does not exist: {args.roles_basedir}")
        return 1

    if not os.path.exists(args.traits_basedir):
        print(f"Error: Traits base directory does not exist: {args.traits_basedir}")
        return 1

    # Load first 10 questions from CSV
    question_ids = load_first_10_questions(args.questions_ranked_file)
    if question_ids is None:
        return 1

    # Load PCA results
    role_pca_results = load_pca_results(args.roles_basedir, args.layer, "role")
    if role_pca_results is None:
        return 1

    trait_pca_results = load_pca_results(args.traits_basedir, args.layer, "trait")
    if trait_pca_results is None:
        return 1

    # Get all roles
    roles = get_all_roles(args.roles_basedir)
    if not roles:
        print("No roles found!")
        return 1

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Process roles and save individual .pt files
    current_id = 0
    total_entries = 0

    for i, role in enumerate(roles):
        print(f"Processing role {i+1}/{len(roles)}: {role}")
        entries_saved, current_id = process_role(
            role, question_ids, args.roles_basedir, args.layer,
            role_pca_results, trait_pca_results, args.output_dir, current_id
        )
        total_entries += entries_saved

    print(f"\nCompleted! Processed {len(roles)} roles and saved {total_entries} .pt files to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())