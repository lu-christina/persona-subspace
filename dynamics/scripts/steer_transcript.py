#!/usr/bin/env python3
"""
Replay Transcript with Steering

This script replays user turns from an existing transcript with a steered model using projection capping.
It maintains multi-turn conversation context and applies steering from experiment configurations.

Example usage:
uv run steer_transcript.py \
    --transcript /root/git/persona-subspace/dynamics/results/qwen-3-32b/interactive/philosophy.json \
    --config /workspace/qwen-3-32b/capped/configs/contrast/role_trait_sliding_config.pt \
    --experiment_id "layers_54:58-p0.25" \
    --output_file /root/git/persona-subspace/dynamics/results/qwen-3-32b/steered/philosophy.json \
    --model_name "Qwen/Qwen3-32B"
"""

import argparse
import json
import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'utils'))

from utils.steering_utils import create_projection_cap_steerer
from utils.probing_utils import load_model

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_multi_config(config_filepath: str) -> Tuple[Dict[str, Dict], List[Dict]]:
    """Load and validate multi-capping configuration file.

    Returns:
        Tuple of (vectors_dict, experiments_list)
        vectors_dict: {'vec_name': {'vector': tensor, 'layer': int}, ...}
        experiments_list: [{'id': str, 'interventions': [{'vector': str, 'cap': float}, ...]}, ...]
    """
    logger.info(f"Loading config from {config_filepath}")

    if not os.path.exists(config_filepath):
        raise FileNotFoundError(f"Config file not found: {config_filepath}")

    try:
        config = torch.load(config_filepath, weights_only=False)
    except Exception as e:
        raise ValueError(f"Failed to load config: {e}")

    # Validate config structure
    if not isinstance(config, dict):
        raise ValueError("Config file must contain a dictionary")

    if 'vectors' not in config or 'experiments' not in config:
        raise ValueError("Config must contain 'vectors' and 'experiments' keys")

    vectors = config['vectors']
    experiments = config['experiments']

    if not isinstance(vectors, dict):
        raise ValueError("'vectors' must be a dictionary")

    if not isinstance(experiments, list):
        raise ValueError("'experiments' must be a list")

    # Validate vectors
    for vec_name, vec_data in vectors.items():
        if not isinstance(vec_data, dict):
            raise ValueError(f"Vector '{vec_name}' data must be a dictionary")
        if 'vector' not in vec_data or 'layer' not in vec_data:
            raise ValueError(f"Vector '{vec_name}' must have 'vector' and 'layer' keys")

    # Validate experiments
    for i, exp in enumerate(experiments):
        if not isinstance(exp, dict):
            raise ValueError(f"Experiment {i} must be a dictionary")
        if 'id' not in exp or 'interventions' not in exp:
            raise ValueError(f"Experiment {i} must have 'id' and 'interventions' keys")

        exp_id = exp['id']
        interventions = exp['interventions']

        if not isinstance(interventions, list):
            raise ValueError(f"Experiment '{exp_id}' interventions must be a list")

        for j, interv in enumerate(interventions):
            if not isinstance(interv, dict):
                raise ValueError(f"Experiment '{exp_id}' intervention {j} must be a dictionary")
            if 'vector' not in interv or 'cap' not in interv:
                raise ValueError(f"Experiment '{exp_id}' intervention {j} must have 'vector' and 'cap' keys")

            # Validate that referenced vector exists
            vec_ref = interv['vector']
            if vec_ref not in vectors:
                raise ValueError(f"Experiment '{exp_id}' references unknown vector '{vec_ref}'")

    logger.info(f"Loaded {len(vectors)} vectors and {len(experiments)} experiments")

    return vectors, experiments


def load_transcript(transcript_path: str) -> Dict[str, Any]:
    """Load transcript JSON file.

    Returns:
        Dictionary with 'model', 'turns', 'conversation' keys
    """
    logger.info(f"Loading transcript from {transcript_path}")

    if not os.path.exists(transcript_path):
        raise FileNotFoundError(f"Transcript file not found: {transcript_path}")

    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load transcript: {e}")

    # Validate transcript structure
    if not isinstance(transcript, dict):
        raise ValueError("Transcript must be a dictionary")

    if 'conversation' not in transcript:
        raise ValueError("Transcript must contain 'conversation' key")

    conversation = transcript['conversation']
    if not isinstance(conversation, list):
        raise ValueError("'conversation' must be a list")

    # Count user turns
    user_turns = sum(1 for msg in conversation if msg.get('role') == 'user')

    logger.info(f"Loaded transcript with {len(conversation)} messages ({user_turns} user turns)")

    return transcript


def save_results(output_path: str, results: Dict[str, Any]):
    """Save results to JSON file."""
    logger.info(f"Saving results to {output_path}")

    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved successfully")
    except Exception as e:
        raise ValueError(f"Failed to save results: {e}")


def replay_with_steering(
    transcript: Dict[str, Any],
    model,
    tokenizer,
    steering_vectors: List[torch.Tensor],
    cap_thresholds: List[float],
    layer_indices: List[int],
    experiment_id: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    thinking: bool = False
) -> Dict[str, Any]:
    """Replay transcript with steering applied.

    Args:
        transcript: Original transcript dictionary
        model: Language model
        tokenizer: Tokenizer
        steering_vectors: List of steering vector tensors
        cap_thresholds: List of cap values for each vector
        layer_indices: List of layer indices for each vector
        experiment_id: Experiment identifier
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        thinking: Enable thinking mode (False for Qwen)

    Returns:
        Dictionary with results
    """
    conversation = transcript['conversation']

    # Extract user turns from original conversation
    user_turns = [msg for msg in conversation if msg.get('role') == 'user']

    logger.info(f"Replaying {len(user_turns)} user turns with steering")
    logger.info(f"Experiment: {experiment_id}")
    logger.info(f"Steering: {len(steering_vectors)} vectors across layers {sorted(set(layer_indices))}")

    # Initialize conversation history for multi-turn context
    conversation_history = []
    new_conversation = []

    # Apply steering for all turns
    with create_projection_cap_steerer(
        model=model,
        feature_directions=steering_vectors,
        cap_thresholds=cap_thresholds,
        layer_indices=layer_indices,
        positions="all"
    ) as steerer:

        for turn_idx, user_msg in enumerate(user_turns, 1):
            user_content = user_msg['content']

            logger.info(f"Turn {turn_idx}/{len(user_turns)}: generating response...")

            # Add user message to conversation history
            conversation_history.append({"role": "user", "content": user_content})
            new_conversation.append({"role": "user", "content": user_content})

            # Format conversation with chat template
            try:
                formatted_prompt = tokenizer.apply_chat_template(
                    conversation_history,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=thinking
                )
            except Exception as e:
                logger.error(f"Error formatting chat template: {e}")
                # Fallback to simple formatting
                formatted_prompt = "\n".join([
                    f"{msg['role']}: {msg['content']}" for msg in conversation_history
                ]) + "\nassistant: "

            # Tokenize
            inputs = tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=40960
            ).to(model.device)

            # Generate response with steering
            try:
                with torch.inference_mode():
                    outputs = model.generate(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=True if temperature > 0 else False,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=True
                    )

                # Decode response
                input_length = inputs.input_ids.shape[1]
                generated_tokens = outputs[0][input_length:]
                response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                response = response.strip()

                logger.info(f"Turn {turn_idx}/{len(user_turns)}: generated {len(generated_tokens)} tokens")

            except Exception as e:
                logger.error(f"Error generating response for turn {turn_idx}: {e}")
                response = ""

            # Add assistant response to conversation history
            conversation_history.append({"role": "assistant", "content": response})
            new_conversation.append({"role": "assistant", "content": response})

    # Prepare results
    results = {
        "model": model.config.name_or_path if hasattr(model.config, 'name_or_path') else "unknown",
        "turns": len(user_turns),
        "steering_config": {
            "experiment_id": experiment_id,
            "num_interventions": len(steering_vectors),
            "layers": sorted(set(layer_indices))
        },
        "conversation": new_conversation
    }

    return results


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Replay transcript with steering applied",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--transcript",
        type=str,
        required=True,
        help="Path to transcript JSON file"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to steering config file (.pt format)"
    )

    parser.add_argument(
        "--experiment_id",
        type=str,
        required=True,
        help="Experiment ID to use from config (e.g., 'layers_32:36-p0.01')"
    )

    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to output JSON file"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-32B",
        help="Model name"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)"
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Maximum new tokens to generate"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for text generation"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    logger.info("="*60)
    logger.info("Replay Transcript with Steering")
    logger.info("="*60)

    # Load inputs
    transcript = load_transcript(args.transcript)
    vectors_dict, experiments_list = load_multi_config(args.config)

    # Find experiment by ID
    experiments_by_id = {exp['id']: exp for exp in experiments_list}

    if args.experiment_id not in experiments_by_id:
        available_ids = list(experiments_by_id.keys())
        raise ValueError(
            f"Experiment ID '{args.experiment_id}' not found in config. "
            f"Available IDs: {available_ids[:10]}{'...' if len(available_ids) > 10 else ''}"
        )

    experiment = experiments_by_id[args.experiment_id]
    interventions = experiment['interventions']

    logger.info(f"Using experiment '{args.experiment_id}' with {len(interventions)} interventions")

    # Load model
    logger.info(f"Loading model {args.model_name} on {args.device}")
    model, tokenizer = load_model(args.model_name, device=args.device)
    model.eval()

    # Detect if Qwen model (disable thinking)
    is_qwen = 'qwen' in args.model_name.lower()
    thinking = not is_qwen

    if is_qwen:
        logger.info("Detected Qwen model, disabling thinking mode")

    # Prepare steering vectors
    steering_vectors = []
    cap_thresholds = []
    layer_indices = []

    for interv in interventions:
        vec_name = interv['vector']
        cap_value = interv['cap']

        # Get vector tensor and layer
        vector_tensor = torch.as_tensor(
            vectors_dict[vec_name]['vector'],
            dtype=model.dtype,
            device=model.device
        )
        layer = vectors_dict[vec_name]['layer']

        steering_vectors.append(vector_tensor)
        cap_thresholds.append(float(cap_value))
        layer_indices.append(layer)

    logger.info(f"Prepared {len(steering_vectors)} steering vectors across layers {sorted(set(layer_indices))}")

    # Replay transcript with steering
    results = replay_with_steering(
        transcript=transcript,
        model=model,
        tokenizer=tokenizer,
        steering_vectors=steering_vectors,
        cap_thresholds=cap_thresholds,
        layer_indices=layer_indices,
        experiment_id=args.experiment_id,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        thinking=thinking
    )

    # Save results
    save_results(args.output_file, results)

    logger.info("="*60)
    logger.info("Replay completed successfully!")
    logger.info(f"Results saved to: {args.output_file}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
