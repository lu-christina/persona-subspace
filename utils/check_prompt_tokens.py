#!/usr/bin/env python3
"""
Check chat prompt format tokens right before the assistant's response.

Compares two approaches:
1. Full conversation with add_generation_prompt=False - tokens before response
2. Truncated conversation (no assistant) with add_generation_prompt=True - final tokens

This verifies they produce identical tokens at the pre-response position.
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
    Compare tokens before response using full vs truncated conversation.

    Args:
        tokenizer_name: HuggingFace tokenizer/model name
        conversation: List of {"role", "content"} dicts (must have assistant turn)
        n_tokens: Number of tokens to compare (default: 3)
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    encoder = ConversationEncoder(tokenizer, model_name=tokenizer_name)

    # Chat template kwargs - don't use enable_thinking to avoid template adding thinking block
    chat_kwargs = {}
    if 'qwen' in tokenizer_name.lower():
        chat_kwargs['enable_thinking'] = False  # Avoid template adding <think> tags

    # Method 1: Full conversation with add_generation_prompt=False
    full_ids = tokenizer.apply_chat_template(
        conversation, tokenize=True, add_generation_prompt=False, **chat_kwargs
    )

    # Method 2: Truncated (system + user only) with add_generation_prompt=True
    truncated_conv = [m for m in conversation if m['role'] != 'assistant']
    truncated_ids = tokenizer.apply_chat_template(
        truncated_conv, tokenize=True, add_generation_prompt=True, **chat_kwargs
    )

    # Show assistant content preview
    assistant_content = next((m['content'] for m in conversation if m['role'] == 'assistant'), '')
    print(f"Model: {tokenizer_name}")
    print(f"Assistant content starts: {repr(assistant_content[:60])}...")
    print(f"Full conversation tokens: {len(full_ids)}")
    print(f"Truncated + gen_prompt tokens: {len(truncated_ids)}")
    print("-" * 60)

    # Get response start index from full conversation
    response_indices = encoder.response_indices(conversation, per_turn=True, **chat_kwargs)
    response_start = response_indices[0][0] if response_indices and response_indices[0] else None

    if response_start is None:
        print("ERROR: Could not find response start")
        return

    # Get N tokens before response from full conversation
    before_start = max(0, response_start - n_tokens)
    full_before_ids = full_ids[before_start:response_start]

    # Get last N tokens from truncated conversation
    trunc_last_ids = truncated_ids[-n_tokens:]

    # Compare
    match = full_before_ids == trunc_last_ids

    print(f"\n{n_tokens} tokens BEFORE response (full conv), response_start={response_start}:")
    for i, tid in enumerate(full_before_ids):
        idx = before_start + i
        print(f"  [{idx}] id={tid:6d}  {repr(tokenizer.decode([tid]))}")

    print(f"\nLast {n_tokens} tokens (truncated + gen_prompt):")
    for i, tid in enumerate(trunc_last_ids):
        idx = len(truncated_ids) - n_tokens + i
        print(f"  [{idx}] id={tid:6d}  {repr(tokenizer.decode([tid]))}")

    print(f"\n✓ MATCH" if match else f"\n✗ MISMATCH")


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
            check_tokens_before_response(model_name, sample_conversation, n_tokens=3)
        except Exception as e:
            print(f"Error with {model_name}: {e}")
        print()


if __name__ == "__main__":
    main()
