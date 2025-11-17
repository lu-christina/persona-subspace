#!/usr/bin/env python3
"""
Dataset Activation Analysis Script

Collects activation statistics from a HuggingFace chat dataset and computes
explained variance ratios for persona subspaces (roles/traits).

Usage:
    uv run scripts/dataset_activation.py \
        --hf_dataset lmsys/lmsys-chat-1m \
        --model_name Qwen/Qwen2.5-32B-Instruct \
        --target_layer 32 \
        --output_dir /workspace/qwen-3-32b/dataset_analysis/ \
        --pca_roles /workspace/qwen-3-32b/roles_240/pca/layer32_pos23.pt \
        --pca_traits /workspace/qwen-3-32b/traits_240/pca/layer32_pos-neg50.pt \
        --samples 10000
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'utils'))
sys.path.append('.')
sys.path.append('..')

from utils.internals import ProbingModel, process_batch_conversations

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze dataset activations and compute PCA explained variance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--hf_dataset",
        type=str,
        required=True,
        help="HuggingFace dataset name (e.g., lmsys/lmsys-chat-1m)"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="HuggingFace model name"
    )

    parser.add_argument(
        "--target_layer",
        type=int,
        required=True,
        help="Target layer for PCA projections and variance computation"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save output files"
    )

    parser.add_argument(
        "--pca_roles",
        type=str,
        default=None,
        help="Path to roles PCA configuration file (.pt format)"
    )

    parser.add_argument(
        "--pca_traits",
        type=str,
        default=None,
        help="Path to traits PCA configuration file (.pt format)"
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=10000,
        help="Number of conversations to sample"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for processing"
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=40960,
        help="Maximum sequence length"
    )

    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=100,
        help="Save checkpoint every N conversations"
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
        help="Resume from checkpoint if available"
    )

    return parser.parse_args()


def load_pca_config(filepath: str) -> Tuple[Any, Any]:
    """
    Load PCA configuration from file.

    Args:
        filepath: Path to PCA config file

    Returns:
        Tuple of (scaler, pca) objects
    """
    logger.info(f"Loading PCA config from {filepath}")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"PCA config file not found: {filepath}")

    try:
        pca_results = torch.load(filepath, weights_only=False)
    except Exception as e:
        raise ValueError(f"Failed to load PCA config: {e}")

    if 'scaler' not in pca_results or 'pca' not in pca_results:
        raise ValueError("PCA config must contain 'scaler' and 'pca' keys")

    scaler = pca_results['scaler']
    pca = pca_results['pca']

    logger.info(f"Loaded PCA with {pca.n_components_} components")

    return scaler, pca


def load_and_sample_dataset(
    dataset_name: str,
    n_samples: int,
    seed: int
) -> List[Dict[str, Any]]:
    """
    Load HuggingFace dataset, filter for English, and sample N conversations.

    Returns:
        List of sampled conversation records
    """
    logger.info(f"Loading dataset: {dataset_name}")

    # Load dataset
    dataset = load_dataset(dataset_name, split='train', streaming=True)

    # Filter for English and collect into list
    logger.info("Filtering for English conversations...")
    english_convos = []

    for record in tqdm(dataset, desc="Filtering dataset"):
        if record.get('language') == 'English':
            english_convos.append(record)

            # Stop once we have enough candidates (collect more than needed for sampling)
            if len(english_convos) >= n_samples * 2:
                break

    logger.info(f"Found {len(english_convos)} English conversations")

    # Sample N conversations with fixed seed
    random.seed(seed)
    if len(english_convos) > n_samples:
        sampled = random.sample(english_convos, n_samples)
    else:
        sampled = english_convos
        logger.warning(f"Only {len(sampled)} conversations available, using all")

    logger.info(f"Sampled {len(sampled)} conversations")
    return sampled


def bucket_conversations_by_length(
    conversations: List[List[Dict[str, str]]],
    tokenizer,
    max_length: int = None
) -> List[int]:
    """
    Sort conversations by tokenized length and return sorted indices.
    Optionally filter out conversations that exceed max_length.

    Args:
        conversations: List of conversation lists
        tokenizer: Tokenizer for length calculation
        max_length: Optional maximum length - conversations exceeding this will be marked for exclusion

    Returns:
        List of indices sorted by conversation length (excludes conversations that are too long)
    """
    logger.info("Bucketing conversations by length...")

    conv_lengths = []
    excluded_count = 0

    for i, conv in enumerate(conversations):
        try:
            tokenized = tokenizer.apply_chat_template(
                conv,
                tokenize=True,
                add_generation_prompt=False
            )
            length = len(tokenized)

            # Exclude conversations that are too long
            if max_length is not None and length > max_length:
                excluded_count += 1
                logger.debug(f"Excluding conversation {i} with length {length} > {max_length}")
                continue

        except Exception as e:
            logger.warning(f"Error tokenizing conversation {i}: {e}")
            continue

        conv_lengths.append((i, length))

    # Sort by length
    conv_lengths.sort(key=lambda x: x[1])
    sorted_indices = [idx for idx, _ in conv_lengths]

    lengths = [length for _, length in conv_lengths]
    if lengths:
        logger.info(f"Length bucketing complete: min={min(lengths)}, max={max(lengths)}, "
                   f"mean={sum(lengths)/len(lengths):.1f} tokens")
        if excluded_count > 0:
            logger.warning(f"Excluded {excluded_count} conversations that exceeded max_length={max_length}")

    return sorted_indices


class LayerStreamingStats:
    """Compute streaming statistics per layer using Welford's algorithm."""

    def __init__(self, n_layers: int):
        self.n_layers = n_layers
        self.stats = {}
        for layer_idx in range(n_layers):
            self.stats[layer_idx] = {
                'token_count': 0,
                'token_mean': 0.0,
                'token_M2': 0.0,
                'token_min': float('inf'),
                'token_max': float('-inf'),
                'response_count': 0,
                'response_mean': 0.0,
                'response_M2': 0.0,
                'response_min': float('inf'),
                'response_max': float('-inf'),
            }

    def update_token_stats(self, layer_idx: int, value: float):
        """Update token-level statistics."""
        s = self.stats[layer_idx]
        s['token_count'] += 1
        delta = value - s['token_mean']
        s['token_mean'] += delta / s['token_count']
        s['token_M2'] += delta * (value - s['token_mean'])
        s['token_min'] = min(s['token_min'], value)
        s['token_max'] = max(s['token_max'], value)

    def update_response_stats(self, layer_idx: int, value: float):
        """Update response-level statistics."""
        s = self.stats[layer_idx]
        s['response_count'] += 1
        delta = value - s['response_mean']
        s['response_mean'] += delta / s['response_count']
        s['response_M2'] += delta * (value - s['response_mean'])
        s['response_min'] = min(s['response_min'], value)
        s['response_max'] = max(s['response_max'], value)

    def get_stats(self) -> Dict[int, Dict[str, Any]]:
        """Get current statistics."""
        results = {}
        for layer_idx, s in self.stats.items():
            results[layer_idx] = {
                'token_level_norms': {
                    'count': s['token_count'],
                    'mean': s['token_mean'],
                    'std': np.sqrt(s['token_M2'] / s['token_count']) if s['token_count'] > 1 else 0.0,
                    'min': s['token_min'] if s['token_min'] != float('inf') else 0.0,
                    'max': s['token_max'] if s['token_max'] != float('-inf') else 0.0,
                },
                'response_level_norms': {
                    'count': s['response_count'],
                    'mean': s['response_mean'],
                    'std': np.sqrt(s['response_M2'] / s['response_count']) if s['response_count'] > 1 else 0.0,
                    'min': s['response_min'] if s['response_min'] != float('inf') else 0.0,
                    'max': s['response_max'] if s['response_max'] != float('-inf') else 0.0,
                }
            }
        return results

    def to_dict(self) -> Dict:
        """Serialize for checkpointing."""
        return {'n_layers': self.n_layers, 'stats': self.stats}

    @classmethod
    def from_dict(cls, data: Dict):
        """Deserialize from checkpoint."""
        obj = cls(data['n_layers'])
        obj.stats = data['stats']
        return obj


def save_checkpoint(
    checkpoint_path: str,
    metadata: Dict,
    last_processed_idx: int,
    mean_activations: List[torch.Tensor],
    streaming_stats: LayerStreamingStats
):
    """Save checkpoint for resumability."""
    checkpoint = {
        'metadata': metadata,
        'last_processed_idx': last_processed_idx,
        'mean_activations': [act.cpu() for act in mean_activations],
        'streaming_stats': streaming_stats.to_dict()
    }

    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved: {last_processed_idx} conversations processed")


def load_checkpoint(checkpoint_path: str) -> Tuple[Dict, int, List[torch.Tensor], LayerStreamingStats]:
    """Load checkpoint to resume processing."""
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, weights_only=False)

    metadata = checkpoint['metadata']
    last_processed_idx = checkpoint['last_processed_idx']
    mean_activations = checkpoint['mean_activations']
    streaming_stats = LayerStreamingStats.from_dict(checkpoint['streaming_stats'])

    logger.info(f"Resuming from conversation {last_processed_idx}")
    logger.info(f"Already collected {len(mean_activations)} assistant responses")

    return metadata, last_processed_idx, mean_activations, streaming_stats


def extract_mean_activations_and_compute_norms(
    conv_activations: torch.Tensor,
    streaming_stats: LayerStreamingStats
) -> torch.Tensor:
    """
    Extract mean activations per assistant turn and update norm statistics.

    Args:
        conv_activations: Tensor of shape [num_turns, num_layers, hidden_size]
        streaming_stats: Statistics object to update

    Returns:
        Tensor of shape [num_assistant_turns, num_layers, hidden_size] with mean activations
    """
    num_turns = conv_activations.shape[0]
    num_layers = conv_activations.shape[1]

    # Extract assistant turns (assumes alternating user/assistant, starting with user)
    # So assistant turns are at indices 1, 3, 5, ...
    assistant_indices = list(range(1, num_turns, 2))

    if len(assistant_indices) == 0:
        return torch.empty(0, num_layers, conv_activations.shape[2])

    assistant_turn_means = []

    for turn_idx in assistant_indices:
        # Get activation for this turn: [num_layers, hidden_size]
        turn_activation = conv_activations[turn_idx]

        # Compute norms per layer for token-level statistics
        for layer_idx in range(num_layers):
            layer_activation = turn_activation[layer_idx]  # [hidden_size]
            token_norm = torch.norm(layer_activation).item()
            streaming_stats.update_token_stats(layer_idx, token_norm)

        assistant_turn_means.append(turn_activation)

    # Stack into tensor: [num_assistant_turns, num_layers, hidden_size]
    assistant_means = torch.stack(assistant_turn_means)

    # Update response-level norm statistics
    for layer_idx in range(num_layers):
        for turn_mean in assistant_means:
            response_norm = torch.norm(turn_mean[layer_idx]).item()
            streaming_stats.update_response_stats(layer_idx, response_norm)

    return assistant_means


def process_conversations(
    conversations: List[List[Dict[str, str]]],
    sorted_indices: List[int],
    probing_model,
    batch_size: int,
    max_length: int,
    streaming_stats: LayerStreamingStats,
    all_mean_activations: List[torch.Tensor],
    start_idx: int = 0,
    checkpoint_path: Optional[str] = None,
    checkpoint_interval: int = 100,
    metadata: Optional[Dict] = None
) -> int:
    """
    Process conversations in batches and collect mean activations.

    Returns:
        Total number of assistant responses processed
    """
    total_assistant_responses = len(all_mean_activations)

    # Create batches from sorted indices
    batches = []
    for i in range(start_idx, len(sorted_indices), batch_size):
        batch_indices = sorted_indices[i:i + batch_size]
        batches.append(batch_indices)

    with tqdm(batches, desc="Processing conversations", initial=start_idx // batch_size) as pbar:
        for batch_num, batch_indices in enumerate(pbar):
            try:
                # Get conversations for this batch, skip missing indices
                batch_convs = []
                for idx in batch_indices:
                    if idx < len(conversations):
                        batch_convs.append(conversations[idx])
                    else:
                        logger.warning(f"Skipping conversation index {idx} (out of range)")

                if not batch_convs:
                    logger.warning(f"No valid conversations in batch {batch_num}, skipping")
                    continue

                # Extract activations
                batch_activations = process_batch_conversations(
                    probing_model=probing_model,
                    conversations=batch_convs,
                    max_length=max_length
                )

                # Process each conversation in batch
                for conv_activations in batch_activations:
                    if conv_activations.numel() == 0:
                        continue

                    # Extract mean activations and compute norms
                    assistant_means = extract_mean_activations_and_compute_norms(
                        conv_activations,
                        streaming_stats
                    )

                    # Store mean activations for each assistant turn
                    for turn_mean in assistant_means:
                        all_mean_activations.append(turn_mean.cpu())
                        total_assistant_responses += 1

                # Update progress bar
                pbar.set_postfix(assistant_responses=total_assistant_responses, refresh=True)

                # Save checkpoint periodically
                if checkpoint_path and (batch_num + 1) % checkpoint_interval == 0:
                    last_idx = start_idx + (batch_num + 1) * batch_size
                    save_checkpoint(
                        checkpoint_path,
                        metadata,
                        last_idx,
                        all_mean_activations,
                        streaming_stats
                    )

            except Exception as e:
                logger.error(f"Error processing batch {batch_num}: {e}")
                # Save checkpoint on error
                if checkpoint_path:
                    last_idx = start_idx + batch_num * batch_size
                    save_checkpoint(
                        checkpoint_path,
                        metadata,
                        last_idx,
                        all_mean_activations,
                        streaming_stats
                    )
                raise

            # Clear GPU cache periodically
            if torch.cuda.is_available() and batch_num % 5 == 0:
                torch.cuda.empty_cache()

    return total_assistant_responses


def compute_explained_variance(
    mean_activations: torch.Tensor,
    scaler,
    pca
) -> Tuple[np.ndarray, float]:
    """
    Compute PCA projections and explained variance ratio.

    Args:
        mean_activations: Tensor of shape [n_responses, hidden_dim]
        scaler: sklearn StandardScaler
        pca: sklearn PCA

    Returns:
        Tuple of (projected data, explained variance ratio)
    """
    # Convert to numpy and scale
    # Convert bfloat16 to float32 first since numpy doesn't support bfloat16
    if mean_activations.dtype == torch.bfloat16:
        mean_activations = mean_activations.float()
    activations_np = mean_activations.cpu().numpy()
    scaled = scaler.transform(activations_np)

    # Project
    projected = pca.transform(scaled)

    # Compute explained variance ratio
    # var(projected) / var(original)
    original_var = np.var(activations_np)
    projected_var = np.var(projected)

    explained_var_ratio = projected_var / original_var if original_var > 0 else 0.0

    return projected, explained_var_ratio


def compute_percentiles(values: List[float]) -> Dict[str, float]:
    """Compute percentiles for a list of values."""
    if len(values) == 0:
        return {}

    arr = np.array(values)
    return {
        str(p): float(np.percentile(arr, p))
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
    }


def save_modular_outputs(
    output_dir: str,
    metadata: Dict,
    mean_activations: List[torch.Tensor],
    streaming_stats: LayerStreamingStats,
    target_layer: int,
    pca_configs: Dict[str, Tuple[Any, Any, str]]
):
    """
    Save outputs to separate files.

    Args:
        output_dir: Directory to save files
        metadata: Metadata dict
        mean_activations: List of tensors [num_layers, hidden_dim]
        streaming_stats: Statistics object
        target_layer: Target layer for projections
        pca_configs: Dict mapping name -> (scaler, pca, filepath)
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Save metadata
    logger.info("Saving metadata...")
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # 2. Save mean activations
    logger.info("Saving mean activations...")
    activations_stacked = torch.stack(mean_activations)  # [n_responses, n_layers, hidden_dim]
    activations_path = os.path.join(output_dir, 'mean_activations.pt')
    torch.save({
        'activations': activations_stacked,
        'target_layer': target_layer
    }, activations_path)
    logger.info(f"Saved {activations_stacked.shape[0]} response activations with shape {activations_stacked.shape}")

    # 3. Save activation statistics
    logger.info("Saving activation statistics...")
    stats = streaming_stats.get_stats()

    # Add percentiles for token and response norms
    for layer_idx in stats:
        # Collect all norms for percentile computation (if needed in future)
        # For now, we just have the streaming stats
        stats[layer_idx]['token_level_norms']['percentiles'] = {}
        stats[layer_idx]['response_level_norms']['percentiles'] = {}

    stats_output = {
        'per_layer_stats': {str(k): v for k, v in stats.items()},
        'target_layer': target_layer
    }

    stats_path = os.path.join(output_dir, 'activation_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats_output, f, indent=2)

    # 4. Save PCA projections
    target_layer_activations = activations_stacked[:, target_layer, :]  # [n_responses, hidden_dim]

    for pca_name, (scaler, pca, filepath) in pca_configs.items():
        logger.info(f"Computing and saving {pca_name} projections...")

        projected, explained_var_ratio = compute_explained_variance(
            target_layer_activations,
            scaler,
            pca
        )

        projection_output = {
            'projected': torch.from_numpy(projected),
            'explained_variance_ratio': explained_var_ratio,
            'pca_n_components': pca.n_components_,
            'pca_explained_variance_from_fit': pca.explained_variance_ratio_.tolist(),
            'target_layer': target_layer,
            'pca_config_path': filepath
        }

        projection_path = os.path.join(output_dir, f'{pca_name}_projections.pt')
        torch.save(projection_output, projection_path)

        logger.info(f"{pca_name}: explained variance ratio = {explained_var_ratio:.4f}")


def main():
    """Main function."""
    args = parse_arguments()

    logger.info("="*60)
    logger.info("Dataset Activation Analysis")
    logger.info("="*60)
    logger.info(f"Dataset: {args.hf_dataset}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Target layer: {args.target_layer}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"PCA roles: {args.pca_roles}")
    logger.info(f"PCA traits: {args.pca_traits}")
    logger.info(f"Samples: {args.samples}")
    logger.info(f"Seed: {args.seed}")

    # Validate that at least one PCA config is provided
    if args.pca_roles is None and args.pca_traits is None:
        logger.warning("No PCA configs provided. Will only compute activation statistics.")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Checkpoint path
    checkpoint_path = os.path.join(args.output_dir, 'checkpoint.pt')

    # Load model first to get number of layers
    logger.info("Loading model and tokenizer...")
    pm = ProbingModel(args.model_name)
    model = pm.model
    tokenizer = pm.tokenizer
    model.eval()

    # Get number of layers
    n_layers = len(pm.get_layers())
    logger.info(f"Model has {n_layers} layers")

    # Validate target layer
    if args.target_layer >= n_layers or args.target_layer < 0:
        raise ValueError(f"Target layer {args.target_layer} is invalid. Model has {n_layers} layers (0-{n_layers-1})")

    # Initialize or resume state
    if args.resume_from_checkpoint and os.path.exists(checkpoint_path):
        metadata, start_idx, all_mean_activations, streaming_stats = load_checkpoint(checkpoint_path)
    else:
        metadata = {
            'dataset': args.hf_dataset,
            'model': args.model_name,
            'target_layer': args.target_layer,
            'n_layers': n_layers,
            'n_conversations_requested': args.samples,
            'seed': args.seed,
            'language_filter': 'English',
            'pca_configs': {}
        }

        if args.pca_roles:
            metadata['pca_configs']['roles'] = args.pca_roles
        if args.pca_traits:
            metadata['pca_configs']['traits'] = args.pca_traits

        start_idx = 0
        all_mean_activations = []
        streaming_stats = LayerStreamingStats(n_layers)

    # Load and sample dataset
    try:
        sampled_records = load_and_sample_dataset(
            args.hf_dataset,
            args.samples,
            args.seed
        )
        conversations = [record['conversation'] for record in sampled_records]

        if start_idx == 0:
            metadata['n_conversations_sampled'] = len(conversations)
        else:
            logger.info(f"Re-sampled {len(conversations)} conversations with same seed for resumption")
            if start_idx >= len(conversations):
                logger.warning(f"Checkpoint start_idx ({start_idx}) >= available conversations ({len(conversations)})")
                logger.warning("Starting from available conversations")
                start_idx = len(conversations) - 1
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        if start_idx > 0:
            logger.error("Cannot resume without dataset. Please check dataset availability.")
        raise

    # Sort conversations by length for efficient batching
    # Pass max_length to filter out conversations that are too long
    sorted_indices = bucket_conversations_by_length(conversations, tokenizer, max_length=args.max_length)

    # Process conversations
    try:
        total_assistant_responses = process_conversations(
            conversations=conversations,
            sorted_indices=sorted_indices,
            probing_model=pm,
            batch_size=args.batch_size,
            max_length=args.max_length,
            streaming_stats=streaming_stats,
            all_mean_activations=all_mean_activations,
            start_idx=start_idx,
            checkpoint_path=checkpoint_path,
            checkpoint_interval=args.checkpoint_interval,
            metadata=metadata
        )

        metadata['n_assistant_responses'] = total_assistant_responses

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        logger.info("Checkpoint saved, can resume with --resume_from_checkpoint")
        return

    # Load PCA configs
    pca_configs = {}
    if args.pca_roles:
        scaler, pca = load_pca_config(args.pca_roles)
        pca_configs['roles'] = (scaler, pca, args.pca_roles)

    if args.pca_traits:
        scaler, pca = load_pca_config(args.pca_traits)
        pca_configs['traits'] = (scaler, pca, args.pca_traits)

    # Save outputs
    save_modular_outputs(
        output_dir=args.output_dir,
        metadata=metadata,
        mean_activations=all_mean_activations,
        streaming_stats=streaming_stats,
        target_layer=args.target_layer,
        pca_configs=pca_configs
    )

    # Clean up checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        logger.info("Checkpoint cleaned up")

    logger.info("Processing completed successfully!")
    logger.info(f"Processed {metadata['n_conversations_sampled']} conversations")
    logger.info(f"Collected {total_assistant_responses} assistant responses")
    logger.info(f"Output saved to {args.output_dir}")


if __name__ == "__main__":
    main()
