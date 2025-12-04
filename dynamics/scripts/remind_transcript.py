#!/usr/bin/env python3
"""
Replay Transcript with Reminder Injection

This script replays user turns from an existing transcript and injects a reminder
message when activation projections exceed configured thresholds.

Unlike steer_transcript.py which applies projection capping during generation,
this script:
1. Generates responses without steering
2. After each response, checks if mean response activation projections exceed caps
3. If ALL projections exceed their thresholds, injects a reminder before the next turn
4. The reminder is transient - not persisted in conversation history

Example usage:
uv run remind_transcript.py \
    --transcript /root/git/persona-subspace/dynamics/results/qwen-3-32b/interactive/philosophy.json \
    --config /workspace/qwen-3-32b/capped/configs/contrast/role_trait_sliding_config.pt \
    --experiment_id "layers_54:58-p0.25" \
    --output_file /root/git/persona-subspace/dynamics/results/qwen-3-32b/reminded/philosophy.json \
    --model_name "Qwen/Qwen3-32B"
"""

import argparse
import json
import os
import re
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from utils.internals import ProbingModel, ConversationEncoder

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def strip_internal_state(message: str) -> str:
    """
    Remove <INTERNAL_STATE>...</INTERNAL_STATE> tags from a message.

    Args:
        message: The message potentially containing internal state tags

    Returns:
        The message with internal state tags and their content removed
    """
    # Remove internal state tags and everything between them
    pattern = r'<INTERNAL_STATE>.*?</INTERNAL_STATE>'
    cleaned = re.sub(pattern, '', message, flags=re.DOTALL)
    # Clean up any extra whitespace that might be left
    cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)
    return cleaned.strip()


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


def load_reminder_text(reminder_path: str) -> str:
    """Load reminder text from file.

    Args:
        reminder_path: Path to the reminder text file

    Returns:
        The reminder text content
    """
    logger.info(f"Loading reminder text from {reminder_path}")

    if not os.path.exists(reminder_path):
        raise FileNotFoundError(f"Reminder file not found: {reminder_path}")

    with open(reminder_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def extract_layer_activations(
    model,
    tokenizer,
    conversation_history: List[Dict],
    layer_idx: int,
    thinking: bool = False
) -> torch.Tensor:
    """Extract activations at a specific layer for the conversation.

    Args:
        model: The language model
        tokenizer: Model tokenizer
        conversation_history: Conversation to extract activations for
        layer_idx: Layer index to extract from
        thinking: Whether thinking mode is enabled

    Returns:
        Tensor of shape (seq_len, hidden_size)
    """
    try:
        formatted = tokenizer.apply_chat_template(
            conversation_history,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=thinking
        )
    except TypeError:
        # Some tokenizers don't support enable_thinking
        formatted = tokenizer.apply_chat_template(
            conversation_history,
            tokenize=False,
            add_generation_prompt=False
        )

    tokens = tokenizer(formatted, return_tensors="pt", add_special_tokens=False)
    input_ids = tokens["input_ids"].to(model.device)

    activations = []

    def hook_fn(module, input, output):
        act = output[0] if isinstance(output, tuple) else output
        activations.append(act[0, :, :].detach())  # (seq_len, hidden_size)

    # Get model layers
    layers = model.model.layers
    handle = layers[layer_idx].register_forward_hook(hook_fn)

    try:
        with torch.inference_mode():
            model(input_ids)
    finally:
        handle.remove()

    return activations[0]  # (seq_len, hidden_size)


def compute_response_projection(
    model,
    tokenizer,
    encoder: ConversationEncoder,
    conversation_history: List[Dict],
    steering_vector: torch.Tensor,
    layer_index: int,
    thinking: bool = False
) -> float:
    """
    Compute mean projection of last assistant response tokens onto steering vector.

    Args:
        model: The language model
        tokenizer: Model tokenizer
        encoder: ConversationEncoder instance
        conversation_history: Full conversation including latest assistant response
        steering_vector: Direction vector to project onto
        layer_index: Layer to extract activations from
        thinking: Whether thinking mode is enabled

    Returns:
        Scalar projection value (unnormalized dot product)
    """
    # Get per-turn response indices
    try:
        turn_indices = encoder.response_indices(
            conversation_history,
            per_turn=True,
            enable_thinking=thinking
        )
    except TypeError:
        # Some encoders don't support enable_thinking
        turn_indices = encoder.response_indices(
            conversation_history,
            per_turn=True
        )

    if not turn_indices:
        logger.warning("No response indices found")
        return 0.0

    # Get last assistant turn's indices
    last_turn_indices = turn_indices[-1]

    if not last_turn_indices:
        logger.warning("Last turn has no indices")
        return 0.0

    # Extract activations
    activations = extract_layer_activations(
        model, tokenizer, conversation_history, layer_index, thinking
    )

    # Select activations for last turn
    # Ensure indices are within bounds
    max_idx = activations.shape[0]
    valid_indices = [idx for idx in last_turn_indices if idx < max_idx]

    if not valid_indices:
        logger.warning(f"No valid indices (max={max_idx}, indices={last_turn_indices[:5]}...)")
        return 0.0

    response_acts = activations[valid_indices]  # (num_response_tokens, hidden_size)
    mean_act = response_acts.mean(dim=0)  # (hidden_size,)

    # Ensure tensors are on the same device
    mean_act = mean_act.to(steering_vector.device)

    # Unnormalized projection (raw dot product)
    projection = (mean_act @ steering_vector).item()

    return projection


def check_all_projections_exceed_caps(
    model,
    tokenizer,
    encoder: ConversationEncoder,
    conversation_history: List[Dict],
    steering_vectors: List[torch.Tensor],
    cap_thresholds: List[float],
    layer_indices: List[int],
    thinking: bool = False
) -> Tuple[bool, List[float]]:
    """
    Check if ALL vector projections exceed their respective cap thresholds.

    Args:
        model: The language model
        tokenizer: Model tokenizer
        encoder: ConversationEncoder instance
        conversation_history: Full conversation including latest assistant response
        steering_vectors: List of direction vectors to project onto
        cap_thresholds: List of cap thresholds for each vector
        layer_indices: List of layer indices for each vector
        thinking: Whether thinking mode is enabled

    Returns:
        Tuple of (all_exceed: bool, projections: List[float])
    """
    projections = []

    for vec, cap, layer_idx in zip(steering_vectors, cap_thresholds, layer_indices):
        proj = compute_response_projection(
            model, tokenizer, encoder, conversation_history,
            vec, layer_idx, thinking
        )
        projections.append(proj)

    # Check if ALL projections exceed their caps
    all_exceed = all(proj > cap for proj, cap in zip(projections, cap_thresholds))

    return all_exceed, projections


def replay_with_reminder(
    transcript: Dict[str, Any],
    model,
    tokenizer,
    encoder: ConversationEncoder,
    steering_vectors: List[torch.Tensor],
    cap_thresholds: List[float],
    layer_indices: List[int],
    experiment_id: str,
    reminder_text: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    thinking: bool = False
) -> Dict[str, Any]:
    """Replay transcript with reminder injection when projections exceed caps.

    Args:
        transcript: Original transcript dictionary
        model: Language model
        tokenizer: Tokenizer
        encoder: ConversationEncoder instance
        steering_vectors: List of steering vector tensors
        cap_thresholds: List of cap values for each vector
        layer_indices: List of layer indices for each vector
        experiment_id: Experiment identifier
        reminder_text: Text to inject as reminder
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        thinking: Enable thinking mode (False for Qwen)

    Returns:
        Dictionary with results including reminder_indices
    """
    conversation = transcript['conversation']

    # Extract user turns from original conversation
    user_turns = [msg for msg in conversation if msg.get('role') == 'user']

    logger.info(f"Replaying {len(user_turns)} user turns with reminder injection")
    logger.info(f"Experiment: {experiment_id}")
    logger.info(f"Monitoring: {len(steering_vectors)} vectors across layers {sorted(set(layer_indices))}")
    logger.info(f"Cap thresholds: {cap_thresholds}")

    # Initialize conversation histories
    conversation_history = []  # Persistent history (no reminders)
    new_conversation = []      # Output conversation (no reminders)
    reminder_indices = []      # Track which turns had reminders
    insert_reminder_flag = False

    for turn_idx, user_msg in enumerate(user_turns, 1):
        # Strip internal state tags before sending to target model
        user_content = strip_internal_state(user_msg['content'])

        logger.info(f"Turn {turn_idx}/{len(user_turns)}: generating response...")

        # Add user message to persistent history
        conversation_history.append({"role": "user", "content": user_content})
        new_conversation.append({"role": "user", "content": user_content})

        # Build prompt for generation
        if insert_reminder_flag:
            # Create TEMPORARY prompt with reminder injected before user message
            prompt_messages = list(conversation_history[:-1])  # Copy all except current user
            prompt_messages.append({"role": "system", "content": reminder_text})
            prompt_messages.append({"role": "user", "content": user_content})
            reminder_indices.append(turn_idx)
            insert_reminder_flag = False
            logger.info(f"Turn {turn_idx}: injecting reminder (transient)")
        else:
            prompt_messages = conversation_history

        # Format conversation with chat template
        try:
            formatted_prompt = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=thinking
            )
        except TypeError:
            # Fallback if enable_thinking not supported
            formatted_prompt = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            logger.error(f"Error formatting chat template: {e}")
            # Fallback to simple formatting
            formatted_prompt = "\n".join([
                f"{msg['role']}: {msg['content']}" for msg in prompt_messages
            ]) + "\nassistant: "

        # Tokenize
        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=40960
        ).to(model.device)

        # Generate response (no steering)
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

        # Add assistant response to persistent history (reminder NOT included)
        conversation_history.append({"role": "assistant", "content": response})
        new_conversation.append({"role": "assistant", "content": response})

        # Check if reminder needed for next turn (only if not the last turn)
        if turn_idx < len(user_turns):
            all_exceed, projections = check_all_projections_exceed_caps(
                model, tokenizer, encoder, conversation_history,
                steering_vectors, cap_thresholds, layer_indices, thinking
            )

            proj_str = ", ".join([f"{p:.3f}" for p in projections])
            cap_str = ", ".join([f"{c:.3f}" for c in cap_thresholds])
            logger.info(f"Turn {turn_idx}: projections [{proj_str}] vs caps [{cap_str}]")

            if all_exceed:
                insert_reminder_flag = True
                logger.info(f"Turn {turn_idx}: ALL projections exceed caps, will insert reminder next turn")

    # Prepare results
    results = {
        "model": model.config.name_or_path if hasattr(model.config, 'name_or_path') else "unknown",
        "turns": len(user_turns),
        "reminder_config": {
            "experiment_id": experiment_id,
            "num_interventions": len(steering_vectors),
            "layers": sorted(set(layer_indices)),
            "cap_thresholds": cap_thresholds
        },
        "reminder_indices": reminder_indices,
        "conversation": new_conversation
    }

    return results


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Replay transcript with reminder injection based on activation thresholds",
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
        "--reminder",
        type=str,
        default="/root/git/persona-subspace/dynamics/data/conversation_reminder.txt",
        help="Path to reminder text file"
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

    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="If specified, only use vectors from this layer for cap checking (filters experiment interventions)"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    logger.info("="*60)
    logger.info("Replay Transcript with Reminder Injection")
    logger.info("="*60)

    # Load inputs
    transcript = load_transcript(args.transcript)
    vectors_dict, experiments_list = load_multi_config(args.config)
    reminder_text = load_reminder_text(args.reminder)

    logger.info(f"Reminder text loaded ({len(reminder_text)} chars)")

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

    # Filter interventions to single layer if --layer specified
    if args.layer is not None:
        filtered_interventions = []
        for interv in interventions:
            vec_name = interv['vector']
            layer = vectors_dict[vec_name]['layer']
            if layer == args.layer:
                filtered_interventions.append(interv)

        if not filtered_interventions:
            available_layers = sorted(set(vectors_dict[i['vector']]['layer'] for i in interventions))
            raise ValueError(f"No vectors found at layer {args.layer}. Available layers: {available_layers}")

        interventions = filtered_interventions
        logger.info(f"Filtered to {len(interventions)} interventions at layer {args.layer}")

    # Load model
    logger.info(f"Loading model {args.model_name} on {args.device}")
    pm = ProbingModel(args.model_name, device=args.device)
    model = pm.model
    tokenizer = pm.tokenizer
    model.eval()

    # Create encoder
    encoder = ConversationEncoder(tokenizer, args.model_name)

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

    # Replay transcript with reminder injection
    results = replay_with_reminder(
        transcript=transcript,
        model=model,
        tokenizer=tokenizer,
        encoder=encoder,
        steering_vectors=steering_vectors,
        cap_thresholds=cap_thresholds,
        layer_indices=layer_indices,
        experiment_id=args.experiment_id,
        reminder_text=reminder_text,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        thinking=thinking
    )

    # Save results
    save_results(args.output_file, results)

    logger.info("="*60)
    logger.info("Replay completed successfully!")
    logger.info(f"Results saved to: {args.output_file}")
    logger.info(f"Reminders inserted at turns: {results['reminder_indices']}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
