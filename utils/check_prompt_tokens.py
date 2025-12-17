#!/usr/bin/env python3
"""
Check chat prompt format tokens right before the assistant's response.

This script uses the ConversationEncoder utility to find where assistant
responses start and shows the N tokens immediately preceding them.
"""

from typing import List, Dict
from transformers import AutoTokenizer
from internals import ConversationEncoder


def check_tokens_before_response(
    tokenizer_name: str,
    conversation: List[Dict[str, str]],
    n_tokens: int = 3,
) -> None:
    """
    Show the N tokens right before each assistant response.

    Args:
        tokenizer_name: HuggingFace tokenizer/model name
        conversation: List of {"role", "content"} dicts
        n_tokens: Number of tokens to show before response (default: 3)
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    encoder = ConversationEncoder(tokenizer, model_name=tokenizer_name)

    # Tokenize full conversation
    full_ids = tokenizer.apply_chat_template(
        conversation, tokenize=True, add_generation_prompt=False
    )

    # Get response indices per turn
    response_indices = encoder.response_indices(conversation, per_turn=True)

    print(f"Model: {tokenizer_name}")
    print(f"Total tokens: {len(full_ids)}")
    print("-" * 60)

    for turn_idx, indices in enumerate(response_indices):
        if not indices:
            continue

        # Get start of assistant response
        response_start = indices[0]

        # Get N tokens before response
        start_idx = max(0, response_start - n_tokens)
        preceding_indices = list(range(start_idx, response_start))
        preceding_ids = [full_ids[i] for i in preceding_indices]

        # Decode tokens
        preceding_tokens = [tokenizer.decode([tid]) for tid in preceding_ids]

        # Also get first few tokens of response for context
        response_preview_ids = [full_ids[i] for i in indices[:3]]
        response_preview = [tokenizer.decode([tid]) for tid in response_preview_ids]

        print(f"\nAssistant turn {turn_idx + 1}:")
        print(f"  Response starts at index: {response_start}")
        print(f"  {n_tokens} tokens before response:")
        for i, (idx, tid, tok) in enumerate(zip(preceding_indices, preceding_ids, preceding_tokens)):
            print(f"    [{idx}] id={tid:6d}  {repr(tok)}")
        print(f"  First {len(response_preview)} tokens of response:")
        for i, (idx, tid, tok) in enumerate(zip(indices[:3], response_preview_ids, response_preview)):
            print(f"    [{idx}] id={tid:6d}  {repr(tok)}")


def main():
    import sys
    import json

    # Default conversation
    sample_conversation = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
    ]

    # Default models
    models = [
        "Qwen/Qwen3-32B",
        "google/gemma-2-27b-it",
        "meta-llama/Llama-3.3-70B-Instruct",
    ]

    # Parse arguments
    jsonl_file = None
    for arg in sys.argv[1:]:
        if arg.endswith('.jsonl'):
            jsonl_file = arg
        else:
            models = [arg]  # Override model if specified

    # Load conversation from jsonl if provided
    if jsonl_file:
        with open(jsonl_file) as f:
            data = json.loads(f.readline())
            sample_conversation = data['conversation']
            print(f"Loaded conversation from: {jsonl_file}")
            print(f"System prompt: {sample_conversation[0].get('content', '')[:80]}...")

    for model_name in models:
        print("\n" + "=" * 60)
        try:
            check_tokens_before_response(model_name, sample_conversation, n_tokens=7)
        except Exception as e:
            print(f"Error with {model_name}: {e}")
        print()


if __name__ == "__main__":
    main()
