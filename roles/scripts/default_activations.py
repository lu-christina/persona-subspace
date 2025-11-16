#!/usr/bin/env python3
"""
default_activations.py - Process default response JSONL files and compute mean activations

Takes a folder containing {integer}_default.jsonl files, loads a model,
computes mean activations per response, and projects them into PC space.

Each response gets a mean activation (averaged over response tokens only).
Outputs:
  - Per-file .pt files with dict {label}_p{prompt_index}_q{question_index} -> tensor
  - Overall mean activation across all responses (raw + PC projected)
"""

import argparse
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import torch
from tqdm import tqdm
import jsonlines

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.internals import ProbingModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Process default response files and compute mean activations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing {int}_default.jsonl files"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="HuggingFace model name or path"
    )
    parser.add_argument(
        "--pca-path",
        type=str,
        required=True,
        help="Path to PCA results .pt file containing 'scaler' and 'pca' keys"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save output .pt files"
    )
    parser.add_argument(
        "--target-layer",
        type=int,
        required=True,
        help="Target layer index for PCA projection"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for processing responses"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run model on"
    )

    return parser.parse_args()


def load_pca_config(filepath: str) -> Tuple[Any, Any]:
    """Load PCA scaler and PCA object from .pt file."""
    logger.info(f"Loading PCA config from {filepath}")
    pca_results = torch.load(filepath, weights_only=False)

    if 'scaler' not in pca_results or 'pca' not in pca_results:
        raise ValueError("PCA config must contain 'scaler' and 'pca' keys")

    return pca_results['scaler'], pca_results['pca']


def find_default_files(input_dir: str) -> List[Tuple[int, Path]]:
    """Find all {int}_default.jsonl files and return sorted by integer."""
    pattern = re.compile(r'^(\d+)_default\.jsonl$')
    files = []

    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    for file in input_path.glob("*_default.jsonl"):
        match = pattern.match(file.name)
        if match:
            idx = int(match.group(1))
            files.append((idx, file))

    files.sort(key=lambda x: x[0])
    logger.info(f"Found {len(files)} default files in {input_dir}")
    return files


def load_responses_from_jsonl(filepath: Path) -> List[Dict]:
    """Load response data from a default JSONL file."""
    responses = []
    with jsonlines.open(filepath, mode='r') as reader:
        for item in reader:
            responses.append(item)
    return responses


def supports_system_prompt(model_name: str) -> bool:
    """
    Check if the model supports system prompts.

    Gemma models don't support system prompts properly and require
    concatenating the system prompt with the first user message.

    Args:
        model_name: HuggingFace model identifier

    Returns:
        True if model supports system prompts, False otherwise
    """
    # Gemma models don't support system prompts
    if "gemma-2" in model_name.lower() or model_name.startswith("google/gemma-2"):
        return False
    # All other models (Qwen, Llama, etc.) support system prompts
    return True


def preprocess_conversation(conversation: List[Dict[str, str]], model_name: str) -> List[Dict[str, str]]:
    """
    Preprocess conversation to handle models that don't support system prompts.

    For Gemma models, if a system prompt exists, it's prepended to the first user message.

    Args:
        conversation: List of message dicts with 'role' and 'content' keys
        model_name: HuggingFace model identifier

    Returns:
        Processed conversation suitable for the model
    """
    if supports_system_prompt(model_name):
        # Model supports system prompts, return as-is
        return conversation

    # Model doesn't support system prompts (e.g., Gemma)
    # Check if conversation has a system prompt
    if not conversation or conversation[0].get("role") != "system":
        return conversation

    # Extract system prompt and prepend to first user message
    system_content = conversation[0]["content"]
    remaining_messages = conversation[1:]

    # Find first user message
    for i, msg in enumerate(remaining_messages):
        if msg.get("role") == "user":
            # Prepend system prompt to user message
            new_user_content = f"{system_content}\n\n{msg['content']}"
            processed = remaining_messages.copy()
            processed[i] = {"role": "user", "content": new_user_content}
            return processed

    # No user message found, just return without system message
    return remaining_messages


def extract_response_activations(
    conversations: List[List[Dict]],
    model,
    tokenizer,
    n_layers: int,
    batch_size: int,
    max_length: int,
    device: str,
    model_name: str
) -> List[torch.Tensor]:
    """
    Extract mean activations for response tokens only.

    Returns list of tensors, each of shape [n_layers, hidden_dim]
    """
    all_activations = []

    # Storage for layer activations during forward pass
    layer_activations = [[] for _ in range(n_layers)]

    def make_hook(layer_idx):
        def hook(module, input, output):
            # Extract hidden states (first element of output tuple for most models)
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            layer_activations[layer_idx].append(hidden_states.detach())
        return hook

    # Register hooks
    pm_temp = ProbingModel.from_existing(model, None)
    layers = pm_temp.get_layers()
    hooks = []
    for layer_idx, layer in enumerate(layers):
        hook = layer.register_forward_hook(make_hook(layer_idx))
        hooks.append(hook)

    try:
        # Process in batches
        for batch_start in range(0, len(conversations), batch_size):
            batch_end = min(batch_start + batch_size, len(conversations))
            batch_conversations = conversations[batch_start:batch_end]

            # Clear layer activations
            for i in range(n_layers):
                layer_activations[i].clear()

            # Format conversations (preprocess for models without system prompt support)
            formatted_prompts = []
            for conv in batch_conversations:
                # Preprocess conversation to handle system prompts
                processed_conv = preprocess_conversation(conv, model_name)
                formatted = tokenizer.apply_chat_template(
                    processed_conv,
                    tokenize=False,
                    add_generation_prompt=False
                )
                formatted_prompts.append(formatted)

            # Tokenize
            batch_tokens = tokenizer(
                formatted_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(device)

            # Find response token positions for each item in batch
            response_masks = []
            for conv in batch_conversations:
                # Tokenize without assistant response to find where it starts
                conv_without_response = conv[:-1]
                # Preprocess conversation for models without system prompt support
                processed_conv_without_response = preprocess_conversation(conv_without_response, model_name)
                prompt_formatted = tokenizer.apply_chat_template(
                    processed_conv_without_response,
                    tokenize=False,
                    add_generation_prompt=True
                )
                prompt_tokens = tokenizer(
                    prompt_formatted,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length
                )
                prompt_len = prompt_tokens['input_ids'].shape[1]
                response_masks.append(prompt_len)

            # Forward pass
            with torch.no_grad():
                _ = model(**batch_tokens)

            # Extract mean activations for response tokens only
            attention_mask = batch_tokens['attention_mask']

            for batch_idx in range(len(batch_conversations)):
                response_start = response_masks[batch_idx]
                response_end = attention_mask[batch_idx].sum().item()

                if response_end <= response_start:
                    logger.warning(f"Empty response at batch_idx={batch_idx}, skipping")
                    continue

                # Collect activations for this response across all layers
                response_acts = []
                for layer_idx in range(n_layers):
                    layer_act = layer_activations[layer_idx][0][batch_idx]  # [seq_len, hidden_dim]
                    response_act = layer_act[response_start:response_end]  # [response_len, hidden_dim]
                    mean_act = response_act.mean(dim=0)  # [hidden_dim]
                    # Move to CPU immediately to avoid device mismatch when stacking
                    response_acts.append(mean_act.cpu())

                # Stack to [n_layers, hidden_dim]
                stacked = torch.stack(response_acts)
                all_activations.append(stacked)

    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()

    return all_activations


def project_activation_to_pc(
    activation: torch.Tensor,
    scaler,
    pca,
    layer_idx: int
) -> torch.Tensor:
    """Project activation from specific layer into PC space."""
    layer_activation = activation[layer_idx]  # [hidden_dim]

    # Convert to float32 if needed
    if layer_activation.dtype == torch.bfloat16:
        layer_activation = layer_activation.float()

    # Convert to numpy and apply PCA
    act_np = layer_activation.cpu().numpy().reshape(1, -1)
    scaled = scaler.transform(act_np)
    projected = pca.transform(scaled)

    return torch.from_numpy(projected).squeeze()


def main():
    args = parse_arguments()

    # Setup
    logger.info("=" * 80)
    logger.info("Default Activations Processing")
    logger.info("=" * 80)
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"PCA path: {args.pca_path}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Target layer: {args.target_layer}")
    logger.info(f"Device: {args.device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    logger.info("Loading model...")
    pm = ProbingModel(args.model_name, device=args.device)
    model = pm.model
    tokenizer = pm.tokenizer
    model.eval()
    n_layers = len(pm.get_layers())
    logger.info(f"Model loaded with {n_layers} layers")

    # Load PCA
    scaler, pca = load_pca_config(args.pca_path)

    # Find all default files
    default_files = find_default_files(args.input_dir)

    if not default_files:
        logger.error(f"No default files found in {args.input_dir}")
        return

    # Process each file
    all_activations_for_mean = []

    for file_idx, filepath in tqdm(default_files, desc="Processing files"):
        logger.info(f"\nProcessing {filepath.name}...")

        # Load responses
        responses = load_responses_from_jsonl(filepath)
        logger.info(f"  Loaded {len(responses)} responses")

        # Extract conversations
        conversations = [resp['conversation'] for resp in responses]

        # Get activations
        activations = extract_response_activations(
            conversations,
            model,
            tokenizer,
            n_layers,
            args.batch_size,
            args.max_length,
            args.device,
            args.model_name
        )

        # Build dictionary with proper keys
        activations_dict = {}
        for resp, act in zip(responses, activations):
            label = resp['label']
            prompt_idx = resp['prompt_index']
            question_idx = resp['question_index']
            key = f"{label}_p{prompt_idx}_q{question_idx}"
            activations_dict[key] = act
            all_activations_for_mean.append(act)

        # Save per-file activations
        output_path = os.path.join(args.output_dir, f"{file_idx}_default.pt")
        torch.save(activations_dict, output_path)
        logger.info(f"  Saved {len(activations_dict)} activations to {output_path}")

    # Compute overall mean
    logger.info("\nComputing overall statistics...")
    logger.info(f"Total responses: {len(all_activations_for_mean)}")

    if all_activations_for_mean:
        overall_mean = torch.stack(all_activations_for_mean).mean(dim=0)  # [n_layers, hidden_dim]

        # Project to PC space
        overall_mean_projected = project_activation_to_pc(
            overall_mean,
            scaler,
            pca,
            args.target_layer
        )

        # Save overall mean
        overall_output = {
            'mean_activation': overall_mean,
            'mean_activation_projected': overall_mean_projected,
            'target_layer': args.target_layer,
            'n_responses': len(all_activations_for_mean)
        }

        overall_path = os.path.join(args.output_dir, "overall_mean.pt")
        torch.save(overall_output, overall_path)
        logger.info(f"Saved overall mean to {overall_path}")
        logger.info(f"  Mean activation shape: {overall_mean.shape}")
        logger.info(f"  Projected shape: {overall_mean_projected.shape}")
    else:
        logger.warning("No activations collected, skipping overall mean computation")

    logger.info("\n" + "=" * 80)
    logger.info("Processing complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
