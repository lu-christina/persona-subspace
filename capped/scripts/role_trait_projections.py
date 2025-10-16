#!/usr/bin/env python3
"""
Role Projections Script

Computes projections of role-playing response activations onto target vectors.
Reads pre-computed activations from .pt files, scores from .json files, and
projects onto specified vectors at their designated layers.

Usage:
        uv run role_projections.py \
            --base_dir /workspace/gemma-2-27b/roles_240 \
            --target_vectors /workspace/gemma-2-27b/evals/multi_contrast_vectors.pt \
            --output_jsonl /workspace/gemma-2-27b/roles_240/role_projections.jsonl
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import torch
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute projections of role activations onto target vectors",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Base directory containing response_activations and extract_scores subdirectories"
    )

    parser.add_argument(
        "--target_vectors",
        type=str,
        required=True,
        help="Path to target vectors file (.pt format)"
    )

    parser.add_argument(
        "--output_jsonl",
        type=str,
        required=True,
        help="Path to output JSONL file"
    )

    return parser.parse_args()


def load_target_vectors(filepath: str) -> List[Dict[str, Any]]:
    """
    Load target vectors from file.

    Returns:
        List of dicts with keys: 'name', 'vector', 'layer'
    """
    logger.info(f"Loading target vectors from {filepath}")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Target vectors file not found: {filepath}")

    try:
        vectors_data = torch.load(filepath, weights_only=False)
    except Exception as e:
        raise ValueError(f"Failed to load target vectors: {e}")

    # Handle two formats: list of vectors or dict with 'vectors' key
    if isinstance(vectors_data, dict) and 'vectors' in vectors_data:
        vectors = vectors_data['vectors']
    elif isinstance(vectors_data, list):
        vectors = vectors_data
    else:
        raise ValueError("Target vectors file must be a list or dict with 'vectors' key")

    # Convert numpy arrays to torch tensors if needed, preserve bfloat16 dtype
    import numpy as np
    for vec_info in vectors:
        if 'vector' in vec_info:
            if isinstance(vec_info['vector'], np.ndarray):
                vec_info['vector'] = torch.from_numpy(vec_info['vector']).bfloat16()
            elif isinstance(vec_info['vector'], torch.Tensor):
                # Keep original dtype (should be bfloat16)
                pass
            else:
                raise ValueError(f"Vector '{vec_info.get('name', '?')}' must be numpy array or torch tensor")

    logger.info(f"Loaded {len(vectors)} target vectors")
    return vectors


def get_role_files(base_dir: str) -> List[Tuple[str, Path, Optional[Path]]]:
    """
    Get list of role files with their corresponding score files.

    Returns:
        List of (role_name, activation_path, score_path) tuples.
        score_path may be None if the file doesn't exist.
    """
    activations_dir = Path(base_dir) / "response_activations"
    scores_dir = Path(base_dir) / "extract_scores"

    if not activations_dir.exists():
        raise FileNotFoundError(f"Activations directory not found: {activations_dir}")

    activation_files = sorted(activations_dir.glob("*.pt"))
    role_files = []

    for act_file in activation_files:
        role_name = act_file.stem
        score_file = scores_dir / f"{role_name}.json"

        # Check if score file exists
        if score_file.exists():
            role_files.append((role_name, act_file, score_file))
        else:
            logger.warning(f"No score file found for role '{role_name}', will set scores to None")
            role_files.append((role_name, act_file, None))

    logger.info(f"Found {len(role_files)} role files")
    return role_files


def load_role_data(role_name: str, activation_path: Path, score_path: Optional[Path]) -> Tuple[Dict[str, torch.Tensor], Dict[str, Optional[int]]]:
    """
    Load activation and score data for a single role.

    Returns:
        (activations_dict, scores_dict)
    """
    # Load activations
    try:
        activations = torch.load(activation_path, map_location='cpu', weights_only=False)
    except Exception as e:
        logger.error(f"Error loading activations for {role_name}: {e}")
        raise

    # Load scores if available
    scores = {}
    if score_path is not None:
        try:
            with open(score_path, 'r') as f:
                scores = json.load(f)
        except Exception as e:
            logger.error(f"Error loading scores for {role_name}: {e}")
            raise

    return activations, scores


def compute_projections_for_role(
    role_name: str,
    activations: Dict[str, torch.Tensor],
    scores: Dict[str, Optional[int]],
    target_vectors: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Compute projections for all responses of a given role.

    Uses efficient batch operations by stacking activations.

    Args:
        role_name: Name of the role
        activations: Dict mapping response_id to activation tensor [num_layers, hidden_dims]
        scores: Dict mapping response_id to score (may be empty)
        target_vectors: List of target vector dicts

    Returns:
        List of result dicts for JSONL output
    """
    if len(activations) == 0:
        logger.warning(f"No activations found for role {role_name}")
        return []

    # Stack activations into a single tensor
    response_ids = sorted(activations.keys())
    stacked_activations = torch.stack([activations[rid] for rid in response_ids])
    # Shape: [num_responses, num_layers, hidden_dims]

    num_responses = len(response_ids)
    logger.debug(f"Processing {num_responses} responses for role {role_name}")

    # Prepare results structure
    results = []
    for rid in response_ids:
        # Parse prompt_label from response_id (e.g., "pos_p0_q0" -> "pos")
        prompt_label = rid.split('_')[0]

        # Get score (or None if not available)
        score = scores.get(rid, None)

        result = {
            'role': role_name,
            'prompt_label': prompt_label,
            'response_id': rid,
            'score': score,
            'projections': {}
        }
        results.append(result)

    # Compute projections for each target vector
    for vec_info in target_vectors:
        vec_name = vec_info['name']
        target_vector = vec_info['vector']  # Shape: [hidden_dims]
        layer_idx = vec_info['layer']

        # Extract activations at the specified layer
        # Shape: [num_responses, hidden_dims]
        activations_at_layer = stacked_activations[:, layer_idx, :]

        # Compute projections: (h · v) / ||v||
        # Broadcasting: [num_responses, hidden_dims] @ [hidden_dims] = [num_responses]
        vector_norm = torch.norm(target_vector)
        if vector_norm == 0:
            projections = torch.zeros(num_responses)
        else:
            projections = torch.matmul(activations_at_layer, target_vector) / vector_norm

        # Convert to list and assign to results
        projections_list = projections.tolist()
        for i, result in enumerate(results):
            result['projections'][vec_name] = projections_list[i]

    return results


def main():
    """Main function."""
    args = parse_arguments()

    logger.info("="*60)
    logger.info("Role Projections Analysis")
    logger.info("="*60)
    logger.info(f"Base directory: {args.base_dir}")
    logger.info(f"Target vectors: {args.target_vectors}")
    logger.info(f"Output JSONL: {args.output_jsonl}")

    # Load target vectors
    target_vectors = load_target_vectors(args.target_vectors)

    # Get list of role files
    role_files = get_role_files(args.base_dir)

    if not role_files:
        logger.error("No role files found")
        return

    # Create output directory
    output_dir = os.path.dirname(args.output_jsonl)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Process each role
    all_results = []

    with tqdm(role_files, desc="Processing roles", unit="role") as pbar:
        for role_name, activation_path, score_path in pbar:
            pbar.set_postfix(role=role_name, refresh=True)

            try:
                # Load data for this role
                activations, scores = load_role_data(role_name, activation_path, score_path)

                # Compute projections
                role_results = compute_projections_for_role(
                    role_name=role_name,
                    activations=activations,
                    scores=scores,
                    target_vectors=target_vectors
                )

                all_results.extend(role_results)

            except Exception as e:
                logger.error(f"Error processing role {role_name}: {e}")
                continue

    # Write output
    logger.info(f"Writing {len(all_results)} results to {args.output_jsonl}")
    with open(args.output_jsonl, 'w', encoding='utf-8') as f:
        for result in all_results:
            f.write(json.dumps(result) + '\n')

    logger.info("Processing completed successfully!")
    logger.info(f"Total results: {len(all_results)}")


if __name__ == "__main__":
    main()
