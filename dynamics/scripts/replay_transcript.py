#!/usr/bin/env python3
"""
Replay Transcript without Steering

This script replays user turns from an existing transcript with vLLM without any steering applied.
It maintains multi-turn conversation context and generates new responses.

Example usage:
uv run replay_transcript.py \
    --transcript /root/git/persona-subspace/dynamics/results/qwen-3-32b/interactive/philosophy.json \
    --model "Qwen/Qwen3-32B" \
    --output /root/git/persona-subspace/dynamics/results/qwen-3-32b/replayed/philosophy.json
"""

import argparse
import json
import os
import re
import sys
import logging
from pathlib import Path
from typing import Dict, Any

import torch

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'utils'))

from utils.internals import ProbingModel

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


def replay_transcript(
    transcript: Dict[str, Any],
    model,
    tokenizer,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    thinking: bool = False
) -> Dict[str, Any]:
    """Replay transcript without steering.

    Args:
        transcript: Original transcript dictionary
        model: Language model
        tokenizer: Tokenizer
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        thinking: Enable thinking mode (False for Qwen)

    Returns:
        Dictionary with results
    """
    conversation = transcript['conversation']

    # Extract user turns from original conversation
    user_turns = [msg for msg in conversation if msg.get('role') == 'user']

    logger.info(f"Replaying {len(user_turns)} user turns without steering")

    # Initialize conversation history for multi-turn context
    conversation_history = []
    new_conversation = []

    for turn_idx, user_msg in enumerate(user_turns, 1):
        # Strip internal state tags before sending to model
        user_content = strip_internal_state(user_msg['content'])

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

        # Generate response without steering
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
        "conversation": new_conversation
    }

    return results


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Replay transcript without steering",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--transcript",
        type=str,
        required=True,
        help="Path to transcript JSON file"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name or path (e.g., 'Qwen/Qwen3-32B')"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output JSON file (default: transcript_path with '_replayed' suffix)"
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
    logger.info("Replay Transcript without Steering")
    logger.info("="*60)

    # Load transcript
    transcript = load_transcript(args.transcript)

    # Determine output path
    if args.output is None:
        # Create default output path by adding '_replayed' suffix
        transcript_path = Path(args.transcript)
        output_path = str(transcript_path.parent / f"{transcript_path.stem}_replayed{transcript_path.suffix}")
    else:
        output_path = args.output

    logger.info(f"Output will be saved to: {output_path}")

    # Load model
    logger.info(f"Loading model {args.model} on {args.device}")
    pm = ProbingModel(args.model, device=args.device)
    model = pm.model
    tokenizer = pm.tokenizer
    model.eval()

    # Detect if Qwen model (disable thinking)
    is_qwen = 'qwen' in args.model.lower()
    thinking = not is_qwen

    if is_qwen:
        logger.info("Detected Qwen model, disabling thinking mode")

    # Replay transcript without steering
    results = replay_transcript(
        transcript=transcript,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        thinking=thinking
    )

    # Save results
    save_results(output_path, results)

    logger.info("="*60)
    logger.info("Replay completed successfully!")
    logger.info(f"Results saved to: {output_path}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
