#!/usr/bin/env python3
"""
Compute Between-Class Scatter (LDA) analysis on role-playing activations.

This script loads activations from different roles and scores, filters by minimum
samples, and computes the LDA subspace that maximizes between-class variance.
"""

import os
import sys
import json
import torch
import numpy as np
import argparse
import re
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

sys.path.append('.')
sys.path.append('..')

from utils.pca_utils import L2MeanScaler, MeanScaler


def select_samples_by_priority(keys, scores, target_score, max_samples=None):
    """
    Select samples prioritizing question diversity over prompt diversity.

    Parameters:
    -----------
    keys : list
        All keys from activation file
    scores : dict
        Mapping from key to score
    target_score : int
        Score to filter for (2 or 3)
    max_samples : int or None
        Maximum number of samples to select

    Returns:
    --------
    selected_keys : list
        Keys selected based on priority (question_id first, then prompt_id)
    """
    # Filter by target score
    filtered_keys = [k for k in keys if scores.get(k) == target_score]

    if not filtered_keys:
        return []

    # Parse and sort by (question_id, prompt_id)
    # Format: pos_p{prompt_id}_q{question_id}
    parsed = []
    for key in filtered_keys:
        match = re.match(r'pos_p(\d+)_q(\d+)', key)
        if match:
            prompt_id = int(match.group(1))
            question_id = int(match.group(2))
            parsed.append((question_id, prompt_id, key))

    # Sort by question_id first, then prompt_id
    parsed.sort(key=lambda x: (x[0], x[1]))

    # Apply max_samples limit
    if max_samples is not None:
        parsed = parsed[:max_samples]

    return [item[2] for item in parsed]


def prepare_class_data(acts2, acts3, min_samples=10):
    """
    Combine acts2 and acts3 dictionaries into class-based structure.

    Parameters:
    -----------
    acts2 : dict
        Dictionary mapping role names to tensors of shape [n_samples, N] (single layer)
    acts3 : dict
        Dictionary mapping role names to tensors of shape [n_samples, N] (single layer)
    min_samples : int
        Minimum number of samples required to include a class

    Returns:
    --------
    class_data : dict
        Dictionary mapping class names (e.g., "graduate_pos_2") to arrays of shape [n_samples, N]
    """
    class_data = {}

    # Process pos_2 samples
    for role, activations in acts2.items():
        class_name = f"{role}_pos_2"
        # Convert to numpy
        layer_acts = activations.cpu().float().numpy()
        if len(layer_acts) >= min_samples:
            class_data[class_name] = layer_acts

    # Process pos_3 samples
    for role, activations in acts3.items():
        class_name = f"{role}_pos_3"
        # Convert to numpy
        layer_acts = activations.cpu().float().numpy()
        if len(layer_acts) >= min_samples:
            class_data[class_name] = layer_acts

    return class_data


def find_role_variance_subspace(class_data, scaler=None):
    """
    Find subspace maximizing between-class variance using sklearn's LDA.

    Parameters:
    -----------
    class_data : dict
        Dictionary mapping class names to arrays of shape [n_samples, N]
    scaler : sklearn-compatible scaler or None
        Optional scaler (StandardScaler, L2MeanScaler, MeanScaler, etc.)

    Returns:
    --------
    projection_matrix : array of shape [N, n_components]
        Columns are the optimal projection directions (all components)
    explained_variance_ratio : array
        Percentage of variance explained by each component
    projected_data : dict
        Dictionary mapping class names to projected data of shape [n_samples, n_components]
    fitted_scaler : scaler object or None
        The fitted scaler (for later use on mean vectors)
    lda : LinearDiscriminantAnalysis
        The fitted LDA object
    """
    # Gather all data and class information
    class_names = list(class_data.keys())
    class_samples = [class_data[name] for name in class_names]
    class_sizes = [len(samples) for samples in class_samples]

    print(f"Computing LDA on {len(class_names)} classes with {sum(class_sizes)} total samples")

    # Concatenate all data and create labels
    X = np.vstack(class_samples)  # shape: [total_samples, N]
    y = np.concatenate([[i] * size for i, size in enumerate(class_sizes)])

    # Apply scaler if provided
    if scaler is not None:
        print(f"Applying scaler: {type(scaler).__name__}")
        X = scaler.fit_transform(X)

    # Fit sklearn LDA
    print("Fitting sklearn LDA...")
    n_components = min(len(class_names) - 1, X.shape[1])
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    lda.fit(X, y)

    print(f"Fitted LDA with {lda.n_components} components (max possible: {len(class_names) - 1})")

    # Get projection matrix (scalings_ in sklearn)
    projection_matrix = lda.scalings_

    # Get explained variance ratio
    explained_variance_ratio = lda.explained_variance_ratio_

    print("Projecting data onto LDA subspace...")
    # Project data onto subspace
    projected_data = {}
    start_idx = 0
    for name, size in zip(class_names, class_sizes):
        projected_data[name] = lda.transform(X[start_idx:start_idx + size])
        start_idx += size

    return projection_matrix, explained_variance_ratio, projected_data, scaler, lda


def compute_lda_for_scaler(scaler_name, scaler, class_data, pca_results, args):
    """
    Compute LDA for a specific scaler configuration.
    """
    print("\n" + "=" * 70)
    print(f"COMPUTING LDA WITH SCALER: {scaler_name.upper()}")
    print("=" * 70)

    # Compute LDA
    projection_matrix, variance_ratios, projected_data, fitted_scaler, lda = find_role_variance_subspace(
        class_data,
        scaler=scaler
    )

    print(f"\nProjection matrix shape: {projection_matrix.shape}")
    print(f"Number of components: {len(variance_ratios)}")

    # Display results
    print("\nRESULTS:")
    n_show = min(10, len(variance_ratios))
    print(f"\nTop {n_show} variance ratios (explained variance):")
    for i in range(n_show):
        print(f"  Component {i+1}: {variance_ratios[i]:.4f} ({variance_ratios[i]*100:.2f}%)")

    cumsum = np.cumsum(variance_ratios)
    print(f"\nCumulative variance explained (top {n_show}):")
    for i in range(n_show):
        print(f"  Components 1-{i+1}: {cumsum[i]:.4f} ({cumsum[i]*100:.2f}%)")

    # Project mean vectors if available
    combined_vectors_projected = None
    if pca_results is not None:
        print("\nProjecting mean vectors...")
        pos_2_vectors = torch.stack(pca_results['vectors']['pos_2'])[:, args.layer, :].float().numpy()
        pos_3_vectors = torch.stack(pca_results['vectors']['pos_3'])[:, args.layer, :].float().numpy()
        combined_vectors = np.vstack([pos_2_vectors, pos_3_vectors])

        print(f"  Pos_2 vectors: {pos_2_vectors.shape}")
        print(f"  Pos_3 vectors: {pos_3_vectors.shape}")

        # Apply scaler and project
        if fitted_scaler is not None:
            combined_vectors_scaled = fitted_scaler.transform(combined_vectors)
        else:
            combined_vectors_scaled = combined_vectors

        combined_vectors_projected = combined_vectors_scaled @ projection_matrix
        print(f"  Projected shape: {combined_vectors_projected.shape}")

    # Prepare results dictionary
    results = {
        'layer': args.layer,
        'min_samples': args.min_samples,
        'max_samples': args.max_samples,
        'scaler_type': scaler_name,
        'projection_matrix': projection_matrix,
        'variance_explained': variance_ratios,
        'projected_data': projected_data,
        'class_names': list(class_data.keys()),
        'class_sizes': {name: len(data) for name, data in class_data.items()},
        'scaler': fitted_scaler,
        'lda': lda,
        'n_components': len(variance_ratios),
    }

    # Add mean vector projections if available
    if pca_results is not None and combined_vectors_projected is not None:
        results['mean_vectors_projected'] = combined_vectors_projected
        results['roles'] = {
            'pos_2': pca_results['roles']['pos_2'],
            'pos_3': pca_results['roles']['pos_3']
        }

    return results


def main():
    parser = argparse.ArgumentParser(description='Compute LDA on role-playing activations')
    parser.add_argument('--base_dir', type=str, required=True,
                        help='Base directory containing the data (e.g., /workspace/gemma-2-27b)')
    parser.add_argument('--type', type=str, default='roles_240',
                        help='Type of data (e.g., roles_240)')
    parser.add_argument('--layer', type=int, required=True,
                        help='Which layer to analyze')
    parser.add_argument('--min_samples', type=int, default=10,
                        help='Minimum number of samples per class')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples per role per score (prioritizes question diversity). Default: use all samples.')
    parser.add_argument('--scaler', type=str, default='all',
                        choices=['all', 'standard', 'l2mean', 'mean', 'none'],
                        help='Which scaler(s) to use. "all" runs all 4 options.')
    parser.add_argument('--pca_file', type=str, default=None,
                        help='Path to PCA results file (for mean vectors). If not provided, will use {base_dir}/{type}/pca/layer{layer}_pos23.pt')

    args = parser.parse_args()

    # Setup paths
    dir_path = f"{args.base_dir}/{args.type}"
    lda_dir = f"{dir_path}/lda"
    os.makedirs(lda_dir, exist_ok=True)

    print("=" * 70)
    print("BETWEEN-CLASS SCATTER (LDA) ANALYSIS")
    print("=" * 70)
    print(f"Base directory: {args.base_dir}")
    print(f"Type: {args.type}")
    print(f"Layer: {args.layer}")
    print(f"Min samples per class: {args.min_samples}")
    print(f"Max samples per role: {args.max_samples if args.max_samples else 'All'}")
    print(f"Scaler(s): {args.scaler}")
    print(f"Output directory: {lda_dir}")
    print()

    # Load scores
    print("Loading scores...")
    scores = {}
    scores_dir = f"{dir_path}/extract_scores"
    for file in os.listdir(scores_dir):
        if file.endswith('.json'):
            with open(f"{scores_dir}/{file}") as f:
                scores[file.replace('.json', '')] = json.load(f)
    print(f"Loaded {len(scores)} score files")

    # Load activations with smart sampling (only target layer for memory efficiency)
    print(f"\nLoading activations (layer {args.layer} only)...")
    if args.max_samples:
        print(f"Sampling strategy: Prioritize question diversity, max {args.max_samples} samples per role")
    acts3 = {}
    acts2 = {}
    acts_dir = f"{dir_path}/response_activations"

    for file in os.listdir(acts_dir):
        if file.endswith('.pt') and 'default' not in file:
            role_name = file.replace('.pt', '')
            role_scores = scores[role_name]
            obj = torch.load(f"{acts_dir}/{file}")

            # Select keys for pos_2 and pos_3 with smart sampling
            keys_pos2 = select_samples_by_priority(
                list(obj.keys()),
                role_scores,
                target_score=2,
                max_samples=args.max_samples
            )
            keys_pos3 = select_samples_by_priority(
                list(obj.keys()),
                role_scores,
                target_score=3,
                max_samples=args.max_samples
            )

            # Load only selected keys for target layer
            if keys_pos2:
                role_acts2 = [obj[key][args.layer] for key in keys_pos2]
                acts2[role_name] = torch.stack(role_acts2)

            if keys_pos3:
                role_acts3 = [obj[key][args.layer] for key in keys_pos3]
                acts3[role_name] = torch.stack(role_acts3)

    print(f"Loaded {len(acts3)} roles with pos_3 samples")
    print(f"Loaded {len(acts2)} roles with pos_2 samples")

    # Print sample counts
    if acts2 or acts3:
        total_pos2 = sum(len(v) for v in acts2.values())
        total_pos3 = sum(len(v) for v in acts3.values())
        print(f"Total pos_2 samples: {total_pos2}")
        print(f"Total pos_3 samples: {total_pos3}")

    # Prepare class data
    print(f"\nPreparing class data (min_samples={args.min_samples})...")
    class_data = prepare_class_data(acts2, acts3, args.min_samples)

    print(f"Number of classes after filtering: {len(class_data)}")
    total_samples = sum(len(data) for data in class_data.values())
    print(f"Total samples: {total_samples}")

    # Load PCA results for mean vectors
    if args.pca_file:
        pca_file = args.pca_file
    else:
        pca_file = f"{dir_path}/pca/layer{args.layer}_pos23.pt"

    print(f"\nLooking for PCA results at: {pca_file}")
    try:
        pca_results = torch.load(pca_file, weights_only=False)
        print("PCA results loaded successfully")
    except FileNotFoundError:
        print(f"Warning: PCA file not found. Skipping mean vector projection")
        pca_results = None

    # Determine which scalers to run
    if args.scaler == 'all':
        scalers_to_run = [
            ('standard', StandardScaler()),
            ('l2mean', L2MeanScaler()),
            ('mean', MeanScaler()),
            ('none', None)
        ]
    else:
        if args.scaler == 'standard':
            scaler_obj = StandardScaler()
        elif args.scaler == 'l2mean':
            scaler_obj = L2MeanScaler()
        elif args.scaler == 'mean':
            scaler_obj = MeanScaler()
        else:  # none
            scaler_obj = None
        scalers_to_run = [(args.scaler, scaler_obj)]

    # Compute LDA for each scaler
    for scaler_name, scaler_obj in scalers_to_run:
        results = compute_lda_for_scaler(scaler_name, scaler_obj, class_data, pca_results, args)

        # Save results
        output_file = f"{lda_dir}/layer{args.layer}_{scaler_name}_pos23.pt"
        torch.save(results, output_file)
        print(f"\n✓ Results saved to: {output_file}")

    print("\n" + "=" * 70)
    print("ALL DONE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
