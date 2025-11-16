#!/usr/bin/env python3
"""
Chat Dataset Projections Script

Collects activations from a HuggingFace chat dataset and computes projections
onto target vectors to determine projection distributions for cap selection.

Usage:
    uv run evals/scripts/chat_projections.py \
        --hf_dataset lmsys/lmsys-chat-1m \
        --model_name google/gemma-2-27b-it \
        --target_vectors /workspace/gemma-2-27b/evals/multi_contrast_vectors.pt \
        --output_json /workspace/gemma-2-27b/chat_projections_stats.json \
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
        description="Compute projection statistics from chat dataset",
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
        "--target_vectors",
        type=str,
        required=True,
        help="Path to target vectors file (.pt format)"
    )

    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="Path to output JSON statistics file"
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
        default=4096,
        help="Maximum sequence length"
    )

    parser.add_argument(
        "--histogram_bins",
        type=int,
        default=100,
        help="Number of histogram bins"
    )

    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=100,
        help="Save checkpoint every N conversations"
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file to resume from"
    )

    return parser.parse_args()


def load_target_vectors(filepath: str) -> List[Dict[str, Any]]:
    """Load target vectors from file."""
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

    # Convert numpy arrays to torch tensors if needed
    import numpy as np
    for vec_info in vectors:
        if 'vector' in vec_info:
            if isinstance(vec_info['vector'], np.ndarray):
                vec_info['vector'] = torch.from_numpy(vec_info['vector']).bfloat16()
            elif isinstance(vec_info['vector'], torch.Tensor):
                # Keep original dtype
                pass
            else:
                raise ValueError(f"Vector '{vec_info.get('name', '?')}' must be numpy array or torch tensor")

    logger.info(f"Loaded {len(vectors)} target vectors")
    return vectors


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
    tokenizer
) -> List[int]:
    """
    Sort conversations by tokenized length and return sorted indices.

    Args:
        conversations: List of conversation lists
        tokenizer: Tokenizer for length calculation

    Returns:
        List of indices sorted by conversation length
    """
    logger.info("Bucketing conversations by length...")

    conv_lengths = []
    for i, conv in enumerate(conversations):
        try:
            tokenized = tokenizer.apply_chat_template(
                conv,
                tokenize=True,
                add_generation_prompt=False
            )
            length = len(tokenized)
        except Exception as e:
            logger.warning(f"Error tokenizing conversation {i}: {e}")
            length = 999999  # Put at end if tokenization fails

        conv_lengths.append((i, length))

    # Sort by length
    conv_lengths.sort(key=lambda x: x[1])
    sorted_indices = [idx for idx, _ in conv_lengths]

    lengths = [length for _, length in conv_lengths if length < 999999]
    if lengths:
        logger.info(f"Length bucketing complete: min={min(lengths)}, max={max(lengths)}, "
                   f"mean={sum(lengths)/len(lengths):.1f} tokens")

    return sorted_indices


class StreamingStats:
    """Compute streaming statistics using Welford's algorithm."""

    def __init__(self, vector_names: List[str]):
        self.stats = {}
        for name in vector_names:
            self.stats[name] = {
                'count': 0,
                'mean': 0.0,
                'M2': 0.0,  # For variance
                'min': float('inf'),
                'max': float('-inf')
            }

    def update(self, vector_name: str, value: float):
        """Update statistics with a single value."""
        s = self.stats[vector_name]
        s['count'] += 1
        delta = value - s['mean']
        s['mean'] += delta / s['count']
        s['M2'] += delta * (value - s['mean'])
        s['min'] = min(s['min'], value)
        s['max'] = max(s['max'], value)

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get current statistics."""
        results = {}
        for name, s in self.stats.items():
            results[name] = {
                'count': s['count'],
                'mean': s['mean'],
                'std': np.sqrt(s['M2'] / s['count']) if s['count'] > 1 else 0.0,
                'min': s['min'],
                'max': s['max']
            }
        return results

    def to_dict(self) -> Dict:
        """Serialize for checkpointing."""
        return self.stats

    @classmethod
    def from_dict(cls, data: Dict, vector_names: List[str]):
        """Deserialize from checkpoint."""
        obj = cls(vector_names)
        obj.stats = data
        return obj


def save_checkpoint(
    checkpoint_path: str,
    metadata: Dict,
    last_processed_idx: int,
    projections: Dict[str, List[float]],
    streaming_stats: StreamingStats
):
    """Save checkpoint for resumability."""
    checkpoint = {
        'metadata': metadata,
        'last_processed_idx': last_processed_idx,
        'projections': projections,
        'streaming_stats': streaming_stats.to_dict()
    }

    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f)

    logger.info(f"Checkpoint saved: {last_processed_idx} conversations processed")


def load_checkpoint(checkpoint_path: str, vector_names: List[str]) -> Tuple[Dict, int, Dict, StreamingStats]:
    """Load checkpoint to resume processing."""
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    with open(checkpoint_path, 'r') as f:
        checkpoint = json.load(f)

    metadata = checkpoint['metadata']
    last_processed_idx = checkpoint['last_processed_idx']
    projections = checkpoint['projections']
    streaming_stats = StreamingStats.from_dict(checkpoint['streaming_stats'], vector_names)

    logger.info(f"Resuming from conversation {last_processed_idx}")
    logger.info(f"Already collected {streaming_stats.stats[vector_names[0]]['count']} assistant responses")

    return metadata, last_processed_idx, projections, streaming_stats


def compute_projections_from_activations(
    activations: torch.Tensor,
    target_vectors: List[Dict[str, Any]]
) -> Dict[str, List[float]]:
    """
    Compute projections for all assistant turns in a conversation.

    Args:
        activations: Tensor of shape [num_turns, num_layers, hidden_size]
        target_vectors: List of target vector dicts

    Returns:
        Dict mapping vector name to list of projections (one per assistant turn)
    """
    num_turns = activations.shape[0]

    # Count assistant turns (assumes alternating user/assistant, starting with user)
    # So assistant turns are at indices 1, 3, 5, ...
    assistant_indices = list(range(1, num_turns, 2))

    if len(assistant_indices) == 0:
        return {vec['name']: [] for vec in target_vectors}

    projections = {vec['name']: [] for vec in target_vectors}

    # Compute projections for each vector
    for vec_info in target_vectors:
        vec_name = vec_info['name']
        target_vector = vec_info['vector']
        layer_idx = vec_info['layer']

        # Get activations for all assistant turns at this layer
        assistant_activations = activations[assistant_indices, layer_idx, :]
        # Shape: [num_assistant_turns, hidden_size]

        # Ensure target vector is on same device as activations
        target_vector = target_vector.to(assistant_activations.device)

        # Compute projections: (h · v) / ||v||
        vector_norm = torch.norm(target_vector)
        if vector_norm == 0:
            projs = torch.zeros(len(assistant_indices), device=assistant_activations.device)
        else:
            projs = torch.matmul(assistant_activations, target_vector) / vector_norm

        projections[vec_name] = projs.tolist()

    return projections


def process_conversations(
    conversations: List[List[Dict[str, str]]],
    sorted_indices: List[int],
    probing_model: ProbingModel,
    target_vectors: List[Dict[str, Any]],
    batch_size: int,
    max_length: int,
    streaming_stats: StreamingStats,
    all_projections: Dict[str, List[float]],
    start_idx: int = 0,
    checkpoint_path: Optional[str] = None,
    checkpoint_interval: int = 100,
    metadata: Optional[Dict] = None
) -> int:
    """
    Process conversations in batches and collect projections.

    Returns:
        Total number of assistant responses processed
    """

    # Process in batches
    total_assistant_responses = streaming_stats.stats[target_vectors[0]['name']]['count']

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

                    # Compute projections for all assistant turns
                    conv_projections = compute_projections_from_activations(
                        conv_activations,
                        target_vectors
                    )

                    # Update statistics and store projections
                    for vec_name, proj_list in conv_projections.items():
                        for proj_value in proj_list:
                            streaming_stats.update(vec_name, proj_value)
                            all_projections[vec_name].append(proj_value)
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
                        all_projections,
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
                        all_projections,
                        streaming_stats
                    )
                raise

            # Clear GPU cache periodically
            if torch.cuda.is_available() and batch_num % 5 == 0:
                torch.cuda.empty_cache()

    return total_assistant_responses


def compute_final_statistics(
    all_projections: Dict[str, List[float]],
    streaming_stats: StreamingStats,
    histogram_bins: int
) -> Dict[str, Dict[str, Any]]:
    """Compute final statistics including percentiles and histograms."""
    logger.info("Computing final statistics...")

    results = {}
    basic_stats = streaming_stats.get_stats()

    for vec_name, proj_values in all_projections.items():
        if len(proj_values) == 0:
            continue

        proj_array = np.array(proj_values)

        # Compute percentiles
        percentiles = {
            str(p): float(np.percentile(proj_array, p))
            for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
        }

        # Compute histogram
        counts, bin_edges = np.histogram(proj_array, bins=histogram_bins)

        results[vec_name] = {
            **basic_stats[vec_name],
            'percentiles': percentiles,
            'histogram': {
                'bins': bin_edges.tolist(),
                'counts': counts.tolist()
            }
        }

    return results


def main():
    """Main function."""
    args = parse_arguments()

    logger.info("="*60)
    logger.info("Chat Dataset Projection Analysis")
    logger.info("="*60)
    logger.info(f"Dataset: {args.hf_dataset}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Target vectors: {args.target_vectors}")
    logger.info(f"Output: {args.output_json}")
    logger.info(f"Samples: {args.samples}")
    logger.info(f"Seed: {args.seed}")

    # Create output directory
    output_dir = os.path.dirname(args.output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Checkpoint path
    checkpoint_path = args.output_json.replace('.json', '_checkpoint.json')

    # Load target vectors
    target_vectors = load_target_vectors(args.target_vectors)
    vector_names = [vec['name'] for vec in target_vectors]

    # Initialize or resume state
    if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        metadata, start_idx, all_projections, streaming_stats = load_checkpoint(
            args.resume_from_checkpoint,
            vector_names
        )
    else:
        metadata = {
            'dataset': args.hf_dataset,
            'model': args.model_name,
            'n_conversations_requested': args.samples,
            'seed': args.seed,
            'language_filter': 'English'
        }
        start_idx = 0
        all_projections = {name: [] for name in vector_names}
        streaming_stats = StreamingStats(vector_names)

    # Load and sample dataset (same seed = same conversations)
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
            # Verify we have enough conversations to resume from checkpoint
            if start_idx >= len(conversations):
                logger.warning(f"Checkpoint start_idx ({start_idx}) >= available conversations ({len(conversations)})")
                logger.warning("Starting from available conversations")
                start_idx = len(conversations) - 1
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        if start_idx > 0:
            logger.error("Cannot resume without dataset. Please check dataset availability.")
        raise

    # Load model
    logger.info("Loading model and tokenizer...")
    pm = ProbingModel(args.model_name)
    pm.model.eval()
    logger.info("Model loaded successfully")

    # Sort conversations by length for efficient batching
    sorted_indices = bucket_conversations_by_length(conversations, pm.tokenizer)

    # Process conversations
    try:
        total_assistant_responses = process_conversations(
            conversations=conversations,
            sorted_indices=sorted_indices,
            probing_model=pm,
            target_vectors=target_vectors,
            batch_size=args.batch_size,
            max_length=args.max_length,
            streaming_stats=streaming_stats,
            all_projections=all_projections,
            start_idx=start_idx,
            checkpoint_path=checkpoint_path,
            checkpoint_interval=args.checkpoint_interval,
            metadata=metadata
        )

        metadata['n_assistant_turns'] = total_assistant_responses

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        logger.info("Checkpoint saved, can resume with --resume_from_checkpoint")
        return

    # Compute final statistics
    final_stats = compute_final_statistics(
        all_projections,
        streaming_stats,
        args.histogram_bins
    )

    # Save output
    output = {
        'metadata': metadata,
        'per_vector_stats': final_stats
    }

    logger.info(f"Writing statistics to {args.output_json}")
    with open(args.output_json, 'w') as f:
        json.dump(output, f, indent=2)

    # Clean up checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        logger.info("Checkpoint cleaned up")

    logger.info("Processing completed successfully!")
    logger.info(f"Processed {metadata['n_conversations_sampled']} conversations")
    logger.info(f"Collected {total_assistant_responses} assistant responses")


if __name__ == "__main__":
    main()
