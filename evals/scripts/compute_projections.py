#!/usr/bin/env python3
"""
Compute Projection Analysis Script

This script processes JSONL files containing model responses and computes projections
onto target vectors (e.g., PCA components) for each response.

Usage:
    uv run evals/compute_projections.py \
        --input_jsonl /workspace/qwen-3-32b/evals/unsteered/unsteered_scores.jsonl \
        --target_vectors /path/to/target_vectors.pt \
        --model_name Qwen/Qwen2.5-32B-Instruct \
        --output_jsonl /workspace/qwen-3-32b/evals/unsteered/unsteered_projections.jsonl \
        --batch_size 4
"""

import argparse
import json
import logging
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
import torch
from tqdm import tqdm

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'utils'))
sys.path.append('.')
sys.path.append('..')

from utils.probing_utils import load_model, process_batch_conversations, is_gemma_model
from utils.pca_utils import L2MeanScaler, MeanScaler

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute projections of model responses onto target vectors",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--input_jsonl",
        type=str,
        required=True,
        help="Path to input JSONL file with responses"
    )

    parser.add_argument(
        "--target_vectors",
        type=str,
        required=True,
        help="Path to target vectors file (.pt format)"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="HuggingFace model name"
    )

    parser.add_argument(
        "--output_jsonl",
        type=str,
        required=True,
        help="Path to output JSONL file"
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
        "--thinking",
        type=lambda x: x.lower() in ['true', '1', 'yes'],
        default=False,
        help="Enable thinking mode for chat templates (default: False)"
    )

    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip records that already have projections"
    )

    return parser.parse_args()


def load_target_vectors(filepath: str) -> Dict[str, Any]:
    """Load target vectors from file."""
    logger.info(f"Loading target vectors from {filepath}")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Target vectors file not found: {filepath}")

    try:
        vectors_data = torch.load(filepath, weights_only=False)
    except Exception as e:
        raise ValueError(f"Failed to load target vectors: {e}")

    # Handle two formats:
    # 1. Dictionary with 'vectors' key (new format from create_target_vectors_example.py)
    # 2. List of vectors (format from 0_capping_prep.ipynb)
    if isinstance(vectors_data, list):
        # Convert list format to dict format
        vectors_data = {'vectors': vectors_data}
    elif 'vectors' not in vectors_data:
        raise ValueError("Target vectors file must contain 'vectors' key or be a list of vectors")

    # Convert numpy arrays to torch tensors if needed
    import numpy as np
    for vec_info in vectors_data['vectors']:
        if 'vector' in vec_info:
            if isinstance(vec_info['vector'], np.ndarray):
                vec_info['vector'] = torch.from_numpy(vec_info['vector']).float()
            elif isinstance(vec_info['vector'], torch.Tensor):
                vec_info['vector'] = vec_info['vector'].float()
            else:
                raise ValueError(f"Vector '{vec_info.get('name', '?')}' must be numpy array or torch tensor")

    logger.info(f"Loaded {len(vectors_data['vectors'])} target vectors")

    return vectors_data


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Load records from JSONL file."""
    logger.info(f"Loading JSONL from {filepath}")

    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line.strip())
                records.append(record)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")

    logger.info(f"Loaded {len(records)} records")
    return records


def reconstruct_conversation(record: Dict[str, Any], is_gemma: bool) -> List[Dict[str, str]]:
    """
    Reconstruct conversation from JSONL record.

    Args:
        record: Dictionary with 'prompt', 'question', and 'response' fields
        is_gemma: Whether this is a Gemma model (no system prompt support)

    Returns:
        List of conversation messages
    """
    prompt = record.get('prompt', '')
    question = record.get('question', '')
    response = record.get('response', '')

    # Extract system prompt by removing question from prompt
    # The prompt field contains both system prompt and question
    if question and prompt.endswith(question):
        system_prompt = prompt[:-len(question)].strip()
    else:
        # Fallback: try to find question in prompt
        if question in prompt:
            idx = prompt.rfind(question)
            system_prompt = prompt[:idx].strip()
        else:
            system_prompt = prompt

    # Build conversation
    if is_gemma or not system_prompt:
        # Gemma doesn't support system prompts, so combine system + question
        combined_content = f"{system_prompt} {question}".strip() if system_prompt else question
        conversation = [
            {"role": "user", "content": combined_content},
            {"role": "assistant", "content": response}
        ]
    else:
        # Standard format with system prompt
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
            {"role": "assistant", "content": response}
        ]

    return conversation


def compute_projection(activation: torch.Tensor, target_vector: torch.Tensor, scaler=None) -> tuple:
    """
    Compute scalar projection of activation onto target vector, with optional scaling.

    Args:
        activation: Tensor of shape (hidden_size,)
        target_vector: Tensor of shape (hidden_size,)
        scaler: Optional scaler object (L2MeanScaler, MeanScaler, or None)

    Returns:
        Tuple of (raw_projection, scaled_projection)
        - raw_projection: projection without scaling
        - scaled_projection: projection with scaler applied (None if no scaler)
    """
    import numpy as np

    # Ensure same device and dtype
    activation = activation.to(target_vector.device).float()
    target_vector = target_vector.float()

    # Compute raw projection: (h · v) / ||v||
    vector_norm = torch.norm(target_vector)
    if vector_norm == 0:
        raw_projection = 0.0
    else:
        raw_projection = torch.dot(activation, target_vector) / vector_norm
        raw_projection = raw_projection.item()

    # Compute scaled projection if scaler is provided
    scaled_projection = None
    if scaler is not None:
        try:
            # Convert activation to numpy for scaler
            activation_np = activation.cpu().numpy()

            # Apply scaler transform
            # Scalers expect (n_samples, hidden_size), so add batch dimension
            activation_scaled = scaler.transform(activation_np[np.newaxis, :])[0]  # Remove batch dim

            # Convert back to torch
            activation_scaled_torch = torch.from_numpy(activation_scaled).float().to(target_vector.device)

            # Compute scaled projection
            scaled_norm = torch.norm(target_vector)
            if scaled_norm > 0:
                scaled_projection = torch.dot(activation_scaled_torch, target_vector) / scaled_norm
                scaled_projection = scaled_projection.item()
            else:
                scaled_projection = 0.0
        except Exception as e:
            logger.warning(f"Error computing scaled projection: {e}")
            scaled_projection = None

    return raw_projection, scaled_projection


def process_batch(
    records: List[Dict[str, Any]],
    model,
    tokenizer,
    target_vectors_data: Dict[str, Any],
    is_gemma: bool,
    max_length: int,
    thinking: bool
) -> List[Dict[str, Any]]:
    """
    Process a batch of records and compute projections.

    Args:
        records: List of JSONL records
        model: Loaded model
        tokenizer: Tokenizer
        target_vectors_data: Dictionary with target vectors
        is_gemma: Whether this is a Gemma model
        max_length: Maximum sequence length
        thinking: Enable thinking mode

    Returns:
        List of records with added 'projections' field
    """
    # Reconstruct conversations
    conversations = []
    for record in records:
        conv = reconstruct_conversation(record, is_gemma)
        conversations.append(conv)

    # Set up chat kwargs
    chat_kwargs = {}
    if thinking and 'qwen' in tokenizer.name_or_path.lower():
        chat_kwargs['enable_thinking'] = thinking

    # Extract activations for all conversations
    try:
        batch_activations = process_batch_conversations(
            model=model,
            tokenizer=tokenizer,
            conversations=conversations,
            max_length=max_length,
            **chat_kwargs
        )
    except Exception as e:
        logger.error(f"Error extracting activations: {e}")
        # Return records without projections
        return records

    # Compute projections for each record
    results = []
    for i, (record, conv_activations) in enumerate(zip(records, batch_activations)):
        # conv_activations has shape (num_turns, num_layers, hidden_size)
        # We want the last turn (assistant response)
        if conv_activations.numel() == 0 or conv_activations.shape[0] == 0:
            logger.warning(f"No activations for record {record.get('id', i)}")
            result = record.copy()
            result['projections'] = {}
            results.append(result)
            continue

        # Get assistant response activation (last turn)
        assistant_turn_idx = -1
        assistant_activation = conv_activations[assistant_turn_idx, :, :]  # (num_layers, hidden_size)

        # Compute projections for each target vector
        projections = {}
        projections_scaled = {}

        for vec_info in target_vectors_data['vectors']:
            vec_name = vec_info['name']
            target_vector = vec_info['vector']
            layer_idx = vec_info['layer']
            scaler = vec_info.get('scaler', None)

            # Get activation at the specified layer
            activation_at_layer = assistant_activation[layer_idx, :]  # (hidden_size,)

            # Compute projection (raw and scaled)
            raw_proj, scaled_proj = compute_projection(activation_at_layer, target_vector, scaler)

            projections[vec_name] = raw_proj
            if scaled_proj is not None:
                projections_scaled[vec_name] = scaled_proj

        # Add projections to record
        result = record.copy()
        result['projections'] = projections
        if projections_scaled:
            result['projections_scaled'] = projections_scaled
        results.append(result)

    return results


def bucket_records_by_length(
    records: List[Dict[str, Any]],
    tokenizer,
    is_gemma: bool,
    thinking: bool
) -> List[Dict[str, Any]]:
    """
    Sort records by tokenized conversation length for efficient batching.

    Args:
        records: List of JSONL records
        tokenizer: Tokenizer for length calculation
        is_gemma: Whether this is a Gemma model
        thinking: Enable thinking mode

    Returns:
        List of records sorted by conversation length (shortest first)
    """
    logger.info("Bucketing records by conversation length...")

    # Set up chat kwargs
    chat_kwargs = {}
    if thinking and 'qwen' in tokenizer.name_or_path.lower():
        chat_kwargs['enable_thinking'] = thinking

    record_lengths = []
    for record in records:
        try:
            # Reconstruct conversation
            conversation = reconstruct_conversation(record, is_gemma)

            # Calculate tokenized length
            tokenized = tokenizer.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=False,
                **chat_kwargs
            )
            length = len(tokenized)
        except Exception as e:
            logger.warning(f"Error tokenizing record {record.get('id', '?')}: {e}")
            length = 999999  # Put at end if tokenization fails

        record_lengths.append((record, length))

    # Sort by length (shortest first)
    record_lengths.sort(key=lambda x: x[1])
    sorted_records = [r for r, _ in record_lengths]

    # Log length distribution
    lengths = [length for _, length in record_lengths if length < 999999]
    if lengths:
        logger.info(f"Length bucketing complete: min={min(lengths)}, max={max(lengths)}, "
                   f"mean={sum(lengths)/len(lengths):.1f} tokens")

    return sorted_records


def main():
    """Main function."""
    args = parse_arguments()

    logger.info("="*60)
    logger.info("Compute Projection Analysis")
    logger.info("="*60)
    logger.info(f"Input JSONL: {args.input_jsonl}")
    logger.info(f"Target vectors: {args.target_vectors}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Output JSONL: {args.output_jsonl}")
    logger.info(f"Batch size: {args.batch_size}")

    # Load target vectors
    target_vectors_data = load_target_vectors(args.target_vectors)

    # Load input records
    records = load_jsonl(args.input_jsonl)

    # Filter records if skip_existing is enabled
    if args.skip_existing:
        records_to_process = [r for r in records if 'projections' not in r]
        logger.info(f"Skipping {len(records) - len(records_to_process)} records with existing projections")
        logger.info(f"Processing {len(records_to_process)} records")
    else:
        records_to_process = records

    if not records_to_process:
        logger.info("No records to process")
        return

    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model(args.model_name)
    model.eval()
    logger.info("Model loaded successfully")

    # Check if this is a Gemma model
    is_gemma = is_gemma_model(args.model_name)
    logger.info(f"Is Gemma model: {is_gemma}")

    # Sort records by length for efficient batching
    records_to_process = bucket_records_by_length(
        records_to_process,
        tokenizer,
        is_gemma,
        args.thinking
    )

    # Create output directory
    output_dir = os.path.dirname(args.output_jsonl)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Process in batches
    processed_records = []
    batches = [records_to_process[i:i + args.batch_size]
               for i in range(0, len(records_to_process), args.batch_size)]

    with tqdm(batches, desc="Processing batches", unit="batch") as pbar:
        for batch_idx, batch in enumerate(pbar):
            pbar.set_postfix(batch=f"{batch_idx+1}/{len(batches)}", refresh=True)

            try:
                batch_results = process_batch(
                    records=batch,
                    model=model,
                    tokenizer=tokenizer,
                    target_vectors_data=target_vectors_data,
                    is_gemma=is_gemma,
                    max_length=args.max_length,
                    thinking=args.thinking
                )
                processed_records.extend(batch_results)

            except Exception as e:
                logger.error(f"Error processing batch {batch_idx+1}: {e}")
                # Add records without projections
                for record in batch:
                    result = record.copy()
                    result['projections'] = {}
                    processed_records.append(result)

            # Clear GPU cache periodically
            if torch.cuda.is_available() and batch_idx % 5 == 0:
                torch.cuda.empty_cache()

    # Write output
    logger.info(f"Writing output to {args.output_jsonl}")
    with open(args.output_jsonl, 'w', encoding='utf-8') as f:
        for record in processed_records:
            f.write(json.dumps(record) + '\n')

    logger.info("Processing completed successfully!")
    logger.info(f"Processed {len(processed_records)} records")


if __name__ == "__main__":
    main()
