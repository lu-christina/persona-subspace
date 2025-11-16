#!/usr/bin/env env python3
"""
Streaming variance analysis on raw activations.

Processes all role sample activations in chunks to compute conditional variance
metrics based on PC1 position without loading the entire activation tensor into memory.
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm

sys.path.append('.')
sys.path.append('..')


def load_pca_results(pca_path):
    """Load PCA results for projecting activations."""
    print(f"Loading PCA results from {pca_path}")
    return torch.load(pca_path, weights_only=False)


def get_role_files(response_activations_dir):
    """Get sorted list of role activation files."""
    files = [f for f in os.listdir(response_activations_dir) if f.endswith('.pt')]
    return sorted(files)


def project_to_pc_space(activations, pca_results):
    """Project activations to PC space using scaler and PCA."""
    scaled = pca_results['scaler'].transform(activations.float().numpy())
    projected = pca_results['pca'].transform(scaled)
    return projected


def compute_global_mean_streaming(response_activations_dir, role_files, layer, chunk_size=20):
    """Compute global mean of raw activations using streaming."""
    print("Pass 1: Computing global mean and PC1 values...")

    running_sum = None
    total_samples = 0
    pc1_values = []

    # Process in chunks
    for i in tqdm(range(0, len(role_files), chunk_size), desc="Loading chunks"):
        chunk_files = role_files[i:i + chunk_size]
        chunk_activations = []

        for file in chunk_files:
            file_path = os.path.join(response_activations_dir, file)
            obj = torch.load(file_path, weights_only=False)

            # Extract all samples for this role at target layer
            for key in obj:
                act = obj[key][layer, :]  # Shape: [4608]
                chunk_activations.append(act)

        # Stack chunk
        chunk_tensor = torch.stack(chunk_activations).float()  # Shape: [n_samples, 4608]

        # Update running sum for mean
        if running_sum is None:
            running_sum = chunk_tensor.sum(dim=0)
        else:
            running_sum += chunk_tensor.sum(dim=0)

        total_samples += chunk_tensor.shape[0]

    global_mean = running_sum / total_samples
    print(f"Total samples: {total_samples}")
    print(f"Global mean shape: {global_mean.shape}")

    return global_mean, total_samples


def build_pc1_array(response_activations_dir, role_files, layer, pca_results, chunk_size=20):
    """Build PC1 array by streaming through all role files."""
    print("Pass 1: Building PC1 array...")

    pc1_values = []
    total_samples = 0

    # Process in chunks
    for i in tqdm(range(0, len(role_files), chunk_size), desc="Projecting to PC space"):
        chunk_files = role_files[i:i + chunk_size]
        chunk_activations = []

        for file in chunk_files:
            file_path = os.path.join(response_activations_dir, file)
            obj = torch.load(file_path, weights_only=False)

            # Extract all samples for this role at target layer
            for key in obj:
                act = obj[key][layer, :]  # Shape: [4608]
                chunk_activations.append(act)

        # Stack chunk and project to PC space
        chunk_tensor = torch.stack(chunk_activations).float()  # Shape: [n_samples, 4608]
        projected = project_to_pc_space(chunk_tensor, pca_results)

        # Extract PC1 values
        pc1_chunk = projected[:, 0]
        pc1_values.append(pc1_chunk)

        total_samples += len(pc1_chunk)

    # Concatenate all PC1 values
    pc1_array = np.concatenate(pc1_values)
    print(f"Total samples: {total_samples}")
    print(f"PC1 array shape: {pc1_array.shape}")

    return pc1_array, total_samples


def accumulate_groups(response_activations_dir, role_files, layer, pc1_array,
                       global_mean, pca_results, threshold, quintile_edges,
                       model_name, chunk_size=20):
    """Accumulate samples for each group (assistant, roleplay, quintiles)."""
    print("\nPass 2: Accumulating samples per group...")

    # Initialize group lists
    assistant_samples = []
    roleplay_samples = []
    quintile_samples = [[] for _ in range(5)]

    # Lists for PC1-removed versions
    assistant_samples_no_pc1 = []
    roleplay_samples_no_pc1 = []
    quintile_samples_no_pc1 = [[] for _ in range(5)]

    # Get PC1 direction for projection removal
    pc1_direction = torch.from_numpy(pca_results['pca'].components_[0]).float()

    # Lists for distance computation
    all_samples = []
    all_samples_no_pc1 = []

    sample_idx = 0

    # Process in chunks
    for i in tqdm(range(0, len(role_files), chunk_size), desc="Accumulating groups"):
        chunk_files = role_files[i:i + chunk_size]
        chunk_activations = []

        for file in chunk_files:
            file_path = os.path.join(response_activations_dir, file)
            obj = torch.load(file_path, weights_only=False)

            # Extract all samples for this role at target layer
            for key in obj:
                act = obj[key][layer, :]  # Shape: [4608]
                chunk_activations.append(act)

        # Stack chunk
        chunk_tensor = torch.stack(chunk_activations).float()  # Shape: [n_samples, 4608]

        # Project out PC1
        pc1_loadings = (chunk_tensor @ pc1_direction).unsqueeze(1)
        pc1_projections = pc1_loadings * pc1_direction.unsqueeze(0)
        chunk_no_pc1 = chunk_tensor - pc1_projections

        # Get PC1 values for this chunk
        chunk_size_actual = chunk_tensor.shape[0]
        chunk_pc1 = pc1_array[sample_idx:sample_idx + chunk_size_actual]

        # Determine masks based on model
        if model_name == "gemma-2-27b":
            assistant_mask = chunk_pc1 > threshold
            roleplay_mask = chunk_pc1 <= threshold
        else:
            assistant_mask = chunk_pc1 < threshold
            roleplay_mask = chunk_pc1 >= threshold

        # Accumulate threshold groups
        if assistant_mask.sum() > 0:
            assistant_samples.append(chunk_tensor[assistant_mask])
            assistant_samples_no_pc1.append(chunk_no_pc1[assistant_mask])

        if roleplay_mask.sum() > 0:
            roleplay_samples.append(chunk_tensor[roleplay_mask])
            roleplay_samples_no_pc1.append(chunk_no_pc1[roleplay_mask])

        # Accumulate quintile groups
        for q_idx in range(5):
            if q_idx == 0:
                mask = (chunk_pc1 >= quintile_edges[q_idx]) & (chunk_pc1 <= quintile_edges[q_idx + 1])
            else:
                mask = (chunk_pc1 > quintile_edges[q_idx]) & (chunk_pc1 <= quintile_edges[q_idx + 1])

            if mask.sum() > 0:
                quintile_samples[q_idx].append(chunk_tensor[mask])
                quintile_samples_no_pc1[q_idx].append(chunk_no_pc1[mask])

        # Accumulate all samples for distance computation
        all_samples.append(chunk_tensor)
        all_samples_no_pc1.append(chunk_no_pc1)

        sample_idx += chunk_size_actual

    # Concatenate all groups
    print("\nConcatenating groups...")
    assistant_tensor = torch.cat(assistant_samples, dim=0)
    roleplay_tensor = torch.cat(roleplay_samples, dim=0)
    quintile_tensors = [torch.cat(q_samples, dim=0) for q_samples in quintile_samples]

    assistant_tensor_no_pc1 = torch.cat(assistant_samples_no_pc1, dim=0)
    roleplay_tensor_no_pc1 = torch.cat(roleplay_samples_no_pc1, dim=0)
    quintile_tensors_no_pc1 = [torch.cat(q_samples, dim=0) for q_samples in quintile_samples_no_pc1]

    all_tensor = torch.cat(all_samples, dim=0)
    all_tensor_no_pc1 = torch.cat(all_samples_no_pc1, dim=0)

    print(f"Assistant samples: {assistant_tensor.shape[0]}")
    print(f"Roleplay samples: {roleplay_tensor.shape[0]}")
    for q_idx, q_tensor in enumerate(quintile_tensors):
        print(f"Quintile {q_idx + 1} samples: {q_tensor.shape[0]}")

    return {
        'assistant': assistant_tensor,
        'roleplay': roleplay_tensor,
        'quintiles': quintile_tensors,
        'assistant_no_pc1': assistant_tensor_no_pc1,
        'roleplay_no_pc1': roleplay_tensor_no_pc1,
        'quintiles_no_pc1': quintile_tensors_no_pc1,
        'all': all_tensor,
        'all_no_pc1': all_tensor_no_pc1,
    }


def compute_variance_metrics(groups, global_mean, threshold, quintile_edges, pc1_array, model_name):
    """Compute all variance metrics from accumulated groups."""
    print("\nComputing variance metrics...")

    # Threshold analysis
    var_assistant = torch.var(groups['assistant'], dim=0).mean().item()
    var_roleplay = torch.var(groups['roleplay'], dim=0).mean().item()
    var_ratio = var_assistant / var_roleplay

    var_assistant_no_pc1 = torch.var(groups['assistant_no_pc1'], dim=0).mean().item()
    var_roleplay_no_pc1 = torch.var(groups['roleplay_no_pc1'], dim=0).mean().item()
    var_ratio_no_pc1 = var_assistant_no_pc1 / var_roleplay_no_pc1

    # Quintile analysis
    quintile_variances = []
    quintile_variances_no_pc1 = []
    quintile_sizes = []

    for q_idx, q_tensor in enumerate(groups['quintiles']):
        var_q = torch.var(q_tensor, dim=0).mean().item()
        var_q_no_pc1 = torch.var(groups['quintiles_no_pc1'][q_idx], dim=0).mean().item()

        quintile_variances.append(var_q)
        quintile_variances_no_pc1.append(var_q_no_pc1)
        quintile_sizes.append(q_tensor.shape[0])

    # Quintile ratio
    if model_name == "gemma-2-27b":
        quintile_ratio = quintile_variances[0] / quintile_variances[-1]
        quintile_ratio_no_pc1 = quintile_variances_no_pc1[0] / quintile_variances_no_pc1[-1]
    else:
        quintile_ratio = quintile_variances[-1] / quintile_variances[0]
        quintile_ratio_no_pc1 = quintile_variances_no_pc1[-1] / quintile_variances_no_pc1[0]

    # Distance correlation
    all_tensor = groups['all']
    all_tensor_no_pc1 = groups['all_no_pc1']

    # Compute means
    all_mean = all_tensor.mean(dim=0)
    all_mean_no_pc1 = all_tensor_no_pc1.mean(dim=0)

    # Compute L2 distances
    distances = torch.norm(all_tensor - all_mean, p=2, dim=1).numpy()
    distances_no_pc1 = torch.norm(all_tensor_no_pc1 - all_mean_no_pc1, p=2, dim=1).numpy()

    # Correlations
    corr, p_value = pearsonr(pc1_array, distances)
    corr_no_pc1, p_value_no_pc1 = pearsonr(pc1_array, distances_no_pc1)

    # Determine mask descriptions based on model
    if model_name == "gemma-2-27b":
        assistant_mask_desc = f"pc1 > {threshold}"
        roleplay_mask_desc = f"pc1 <= {threshold}"
    else:
        assistant_mask_desc = f"pc1 < {threshold}"
        roleplay_mask_desc = f"pc1 >= {threshold}"

    return {
        'threshold_analysis': {
            'pc1_threshold': threshold,
            'assistant_like': {
                'mask': assistant_mask_desc,
                'n_samples': int(groups['assistant'].shape[0]),
                'variance_raw': float(var_assistant),
                'variance_raw_pc1_removed': float(var_assistant_no_pc1)
            },
            'roleplay': {
                'mask': roleplay_mask_desc,
                'n_samples': int(groups['roleplay'].shape[0]),
                'variance_raw': float(var_roleplay),
                'variance_raw_pc1_removed': float(var_roleplay_no_pc1)
            },
            'variance_ratio_raw': float(var_ratio),
            'variance_ratio_raw_pc1_removed': float(var_ratio_no_pc1)
        },
        'quintile_analysis': {
            'n_quintiles': 5,
            'quintiles': [
                {
                    'quintile': i + 1,
                    'pc1_range': [float(quintile_edges[i]), float(quintile_edges[i + 1])],
                    'n_samples': int(quintile_sizes[i]),
                    'variance_full': float(quintile_variances[i]),
                    'variance_pc1_removed': float(quintile_variances_no_pc1[i])
                }
                for i in range(5)
            ],
            'variance_ratio_first_to_last_full': float(quintile_ratio),
            'variance_ratio_first_to_last_pc1_removed': float(quintile_ratio_no_pc1)
        },
        'distance_correlation': {
            'full_space': {
                'correlation': float(corr),
                'p_value': float(p_value),
                'significant': bool(p_value < 0.05)
            },
            'pc1_removed': {
                'correlation': float(corr_no_pc1),
                'p_value': float(p_value_no_pc1),
                'significant': bool(p_value_no_pc1 < 0.05)
            }
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Compute streaming variance analysis on raw activations')
    parser.add_argument('--model', type=str, required=True, help='Model name (e.g., gemma-2-27b)')
    parser.add_argument('--layer', type=int, required=True, help='Target layer for analysis')
    parser.add_argument('--chunk-size', type=int, default=20, help='Number of roles to process per chunk')
    parser.add_argument('--threshold', type=int, default=None, help='PC1 threshold (auto-detect if not provided)')
    args = parser.parse_args()

    # Setup paths
    base_dir = f"/workspace/{args.model}"
    type_dir = "roles_240"
    dir_path = f"{base_dir}/{type_dir}"
    response_activations_dir = f"{dir_path}/response_activations"
    pca_path = f"{dir_path}/pca/layer{args.layer}_pos23.pt"

    # Auto-detect threshold if not provided
    if args.threshold is None:
        if args.model == "gemma-2-27b":
            threshold = 25
        else:
            threshold = -25
    else:
        threshold = args.threshold

    print(f"Model: {args.model}")
    print(f"Layer: {args.layer}")
    print(f"Threshold: {threshold}")
    print(f"Chunk size: {args.chunk_size}")

    # Load PCA results
    pca_results = load_pca_results(pca_path)

    # Get role files
    role_files = get_role_files(response_activations_dir)
    print(f"Found {len(role_files)} role files")

    # Pass 1: Compute global mean and build PC1 array
    global_mean, total_samples = compute_global_mean_streaming(
        response_activations_dir, role_files, args.layer, args.chunk_size
    )

    pc1_array, _ = build_pc1_array(
        response_activations_dir, role_files, args.layer, pca_results, args.chunk_size
    )

    # Compute quintile edges
    quintile_edges = np.quantile(pc1_array, np.linspace(0, 1, 6))
    print(f"\nQuintile edges: {quintile_edges}")

    # Pass 2: Accumulate groups
    groups = accumulate_groups(
        response_activations_dir, role_files, args.layer, pc1_array,
        global_mean, pca_results, threshold, quintile_edges,
        args.model, args.chunk_size
    )

    # Compute metrics
    metrics = compute_variance_metrics(
        groups, global_mean, threshold, quintile_edges, pc1_array, args.model
    )

    # Build output JSON
    timestamp = datetime.now().isoformat()
    output_data = {
        "model_name": args.model,
        "layer": args.layer,
        "n_samples": int(total_samples),
        "timestamp": timestamp,
        "analysis_version": "1.0_streaming",
        "conditional_var_roles": {
            "description": "Conditional variance analysis for raw role activations based on PC1 position",
            "n_samples": int(total_samples),
            **metrics
        }
    }

    # Save output
    outdir = Path("./results") / args.model.lower()
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f"variance_layer{args.layer}_streaming.json"

    with open(outfile, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Saved results to: {outfile}")
    print("\nSummary:")
    print(f"  Threshold variance ratio: {metrics['threshold_analysis']['variance_ratio_raw']:.4f}")
    print(f"  Threshold variance ratio (PC1 removed): {metrics['threshold_analysis']['variance_ratio_raw_pc1_removed']:.4f}")
    print(f"  Quintile variance ratio: {metrics['quintile_analysis']['variance_ratio_first_to_last_full']:.2f}x")
    print(f"  Distance correlation: r={metrics['distance_correlation']['full_space']['correlation']:.4f}")


if __name__ == "__main__":
    main()
