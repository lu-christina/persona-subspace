#!/usr/bin/env python3
"""
Token Norm Analysis Script

Computes per-token L2 norms of activations across all layers for a dataset.

Usage:
    uv run scripts/token_norms.py \
        --hf_dataset lmsys/lmsys-chat-1m \
        --model_name Qwen/Qwen2.5-32B-Instruct \
        --output_dir /workspace/qwen-3-32b/token_norms/ \
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

from utils.internals import ProbingModel, ConversationEncoder, ActivationExtractor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute per-token L2 norms of activations across all layers",
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
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save output files"
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
        "--checkpoint_interval",
        type=int,
        default=100,
        help="Save checkpoint every N batches"
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
        help="Resume from checkpoint if available"
    )

    return parser.parse_args()


class LayerNormStats:
    """Compute streaming statistics per layer using Welford's algorithm."""

    def __init__(self, n_layers: int):
        self.n_layers = n_layers
        self.stats = {}
        for layer_idx in range(n_layers):
            self.stats[layer_idx] = {
                'count': 0,
                'mean': 0.0,
                'M2': 0.0,
                'min': float('inf'),
                'max': float('-inf'),
            }

    def update(self, layer_idx: int, value: float):
        """Update statistics for a layer with a new value."""
        s = self.stats[layer_idx]
        s['count'] += 1
        delta = value - s['mean']
        s['mean'] += delta / s['count']
        s['M2'] += delta * (value - s['mean'])
        s['min'] = min(s['min'], value)
        s['max'] = max(s['max'], value)

    def update_batch(self, layer_idx: int, values: torch.Tensor):
        """Update statistics for a layer with a batch of values."""
        for val in values.tolist():
            self.update(layer_idx, val)

    def get_stats(self) -> Dict[int, Dict[str, Any]]:
        """Get current statistics."""
        results = {}
        for layer_idx, s in self.stats.items():
            results[layer_idx] = {
                'count': s['count'],
                'mean': s['mean'],
                'std': np.sqrt(s['M2'] / s['count']) if s['count'] > 1 else 0.0,
                'min': s['min'] if s['min'] != float('inf') else 0.0,
                'max': s['max'] if s['max'] != float('-inf') else 0.0,
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


def load_and_sample_dataset(
    dataset_name: str,
    n_samples: int,
    seed: int
) -> List[Dict[str, Any]]:
    """
    Load HuggingFace dataset, filter for English, and sample N conversations.
    """
    logger.info(f"Loading dataset: {dataset_name}")

    dataset = load_dataset(dataset_name, split='train', streaming=True)

    logger.info("Filtering for English conversations...")
    english_convos = []

    for record in tqdm(dataset, desc="Filtering dataset"):
        if record.get('language') == 'English':
            english_convos.append(record)
            if len(english_convos) >= n_samples * 2:
                break

    logger.info(f"Found {len(english_convos)} English conversations")

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
    """Sort conversations by tokenized length and return sorted indices."""
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

            if max_length is not None and length > max_length:
                excluded_count += 1
                continue

        except Exception as e:
            logger.warning(f"Error tokenizing conversation {i}: {e}")
            continue

        conv_lengths.append((i, length))

    conv_lengths.sort(key=lambda x: x[1])
    sorted_indices = [idx for idx, _ in conv_lengths]

    lengths = [length for _, length in conv_lengths]
    if lengths:
        logger.info(f"Length bucketing complete: min={min(lengths)}, max={max(lengths)}, "
                   f"mean={sum(lengths)/len(lengths):.1f} tokens")
        if excluded_count > 0:
            logger.warning(f"Excluded {excluded_count} conversations that exceeded max_length={max_length}")

    return sorted_indices


def save_checkpoint(
    checkpoint_path: str,
    metadata: Dict,
    last_batch_idx: int,
    streaming_stats: LayerNormStats
):
    """Save checkpoint for resumability."""
    checkpoint = {
        'metadata': metadata,
        'last_batch_idx': last_batch_idx,
        'streaming_stats': streaming_stats.to_dict()
    }
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved: {last_batch_idx} batches processed")


def load_checkpoint(checkpoint_path: str) -> Tuple[Dict, int, LayerNormStats]:
    """Load checkpoint to resume processing."""
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, weights_only=False)

    metadata = checkpoint['metadata']
    last_batch_idx = checkpoint['last_batch_idx']
    streaming_stats = LayerNormStats.from_dict(checkpoint['streaming_stats'])

    logger.info(f"Resuming from batch {last_batch_idx}")

    return metadata, last_batch_idx, streaming_stats


def process_conversations(
    conversations: List[List[Dict[str, str]]],
    sorted_indices: List[int],
    extractor: ActivationExtractor,
    n_layers: int,
    batch_size: int,
    max_length: int,
    streaming_stats: LayerNormStats,
    start_batch_idx: int = 0,
    checkpoint_path: Optional[str] = None,
    checkpoint_interval: int = 100,
    metadata: Optional[Dict] = None
) -> int:
    """Process conversations in batches and compute per-token norms."""
    total_tokens = streaming_stats.stats[0]['count'] if streaming_stats.stats else 0

    # Create batches from sorted indices
    batches = []
    for i in range(0, len(sorted_indices), batch_size):
        batch_indices = sorted_indices[i:i + batch_size]
        batches.append(batch_indices)

    with tqdm(batches, desc="Processing conversations", initial=start_batch_idx) as pbar:
        for batch_num, batch_indices in enumerate(pbar):
            if batch_num < start_batch_idx:
                continue

            try:
                # Get conversations for this batch
                batch_convs = []
                for idx in batch_indices:
                    if idx < len(conversations):
                        batch_convs.append(conversations[idx])

                if not batch_convs:
                    continue

                # Extract activations: (num_layers, batch_size, max_seq_len, hidden_size)
                activations, metadata_batch = extractor.batch_conversations(
                    conversations=batch_convs,
                    layer=None,  # all layers
                    max_length=max_length
                )

                # Compute norms: (num_layers, batch_size, max_seq_len)
                norms = torch.norm(activations.float(), dim=-1)

                # Update stats only for non-padding tokens
                truncated_lengths = metadata_batch['truncated_lengths']
                batch_size_actual = len(truncated_lengths)

                for batch_idx in range(batch_size_actual):
                    valid_length = truncated_lengths[batch_idx]
                    for layer_idx in range(n_layers):
                        token_norms = norms[layer_idx, batch_idx, :valid_length]
                        streaming_stats.update_batch(layer_idx, token_norms)
                        total_tokens = streaming_stats.stats[layer_idx]['count']

                pbar.set_postfix(tokens=total_tokens, refresh=True)

                # Save checkpoint periodically
                if checkpoint_path and (batch_num + 1) % checkpoint_interval == 0:
                    save_checkpoint(
                        checkpoint_path,
                        metadata,
                        batch_num + 1,
                        streaming_stats
                    )

            except Exception as e:
                logger.error(f"Error processing batch {batch_num}: {e}")
                if checkpoint_path:
                    save_checkpoint(
                        checkpoint_path,
                        metadata,
                        batch_num,
                        streaming_stats
                    )
                raise

            # Clear GPU cache periodically
            if torch.cuda.is_available() and batch_num % 5 == 0:
                torch.cuda.empty_cache()

    return total_tokens


def save_results(
    output_dir: str,
    metadata: Dict,
    streaming_stats: LayerNormStats
):
    """Save results to output directory."""
    os.makedirs(output_dir, exist_ok=True)

    # Save metadata
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Save statistics
    stats = streaming_stats.get_stats()
    stats_output = {str(k): v for k, v in stats.items()}

    stats_path = os.path.join(output_dir, 'token_norm_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats_output, f, indent=2)

    logger.info(f"Results saved to {output_dir}")


def main():
    """Main function."""
    args = parse_arguments()

    logger.info("=" * 60)
    logger.info("Token Norm Analysis")
    logger.info("=" * 60)
    logger.info(f"Dataset: {args.hf_dataset}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Samples: {args.samples}")
    logger.info(f"Seed: {args.seed}")

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, 'checkpoint.pt')

    # Load model
    logger.info("Loading model and tokenizer...")
    pm = ProbingModel(args.model_name)
    encoder = ConversationEncoder(pm.tokenizer)
    extractor = ActivationExtractor(pm, encoder)
    pm.model.eval()

    n_layers = len(pm.get_layers())
    logger.info(f"Model has {n_layers} layers")

    # Initialize or resume state
    if args.resume_from_checkpoint and os.path.exists(checkpoint_path):
        metadata, start_batch_idx, streaming_stats = load_checkpoint(checkpoint_path)
    else:
        metadata = {
            'dataset': args.hf_dataset,
            'model': args.model_name,
            'n_layers': n_layers,
            'n_conversations_requested': args.samples,
            'seed': args.seed,
            'language_filter': 'English',
            'max_length': args.max_length,
        }
        start_batch_idx = 0
        streaming_stats = LayerNormStats(n_layers)

    # Load and sample dataset
    sampled_records = load_and_sample_dataset(
        args.hf_dataset,
        args.samples,
        args.seed
    )
    conversations = [record['conversation'] for record in sampled_records]
    metadata['n_conversations_sampled'] = len(conversations)

    # Sort by length for efficient batching
    sorted_indices = bucket_conversations_by_length(
        conversations, pm.tokenizer, max_length=args.max_length
    )

    # Process conversations
    try:
        total_tokens = process_conversations(
            conversations=conversations,
            sorted_indices=sorted_indices,
            extractor=extractor,
            n_layers=n_layers,
            batch_size=args.batch_size,
            max_length=args.max_length,
            streaming_stats=streaming_stats,
            start_batch_idx=start_batch_idx,
            checkpoint_path=checkpoint_path,
            checkpoint_interval=args.checkpoint_interval,
            metadata=metadata
        )
        metadata['total_tokens'] = total_tokens

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        logger.info("Checkpoint saved, can resume with --resume_from_checkpoint")
        return

    # Save results
    save_results(
        output_dir=args.output_dir,
        metadata=metadata,
        streaming_stats=streaming_stats
    )

    # Clean up checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        logger.info("Checkpoint cleaned up")

    logger.info("Processing completed successfully!")
    logger.info(f"Processed {metadata['n_conversations_sampled']} conversations")
    logger.info(f"Computed norms for {total_tokens} tokens")
    logger.info(f"Output saved to {args.output_dir}")


if __name__ == "__main__":
    main()
