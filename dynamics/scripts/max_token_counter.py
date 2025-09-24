#!/usr/bin/env python3
"""
Script to find the maximum token count in conversation files.
Loads the tokenizer once from the model field and counts tokens for each conversation.
"""

import json
import glob
import os
from transformers import AutoTokenizer
import argparse

def load_tokenizer_from_first_file(file_path):
    """Load tokenizer from the model field in the first conversation file."""
    with open(file_path, 'r') as f:
        data = json.load(f)

    model_name = data.get('model', '')
    if not model_name:
        raise ValueError(f"No model field found in {file_path}")

    print(f"Loading tokenizer for model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

def count_conversation_tokens(conversation, tokenizer):
    """Count total tokens in a conversation."""
    total_tokens = 0

    for message in conversation:
        content = message.get('content', '')
        tokens = tokenizer.encode(content)
        total_tokens += len(tokens)

    return total_tokens

def find_max_tokens(directory_path):
    """Find the maximum token count across all conversation files."""

    # Find all JSON files in the directory
    json_files = glob.glob(os.path.join(directory_path, "*.json"))

    if not json_files:
        print(f"No JSON files found in {directory_path}")
        return

    print(f"Found {len(json_files)} JSON files")

    # Load tokenizer from the first file
    tokenizer = load_tokenizer_from_first_file(json_files[0])

    max_tokens = 0
    max_file = ""
    results = []

    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            conversation = data.get('conversation', [])
            if not conversation:
                continue

            token_count = count_conversation_tokens(conversation, tokenizer)
            results.append((os.path.basename(file_path), token_count))

            if token_count > max_tokens:
                max_tokens = token_count
                max_file = file_path

            print(f"{os.path.basename(file_path)}: {token_count:,} tokens")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print(f"\n{'='*50}")
    print(f"MAXIMUM TOKENS: {max_tokens:,}")
    print(f"FILE: {os.path.basename(max_file)}")
    print(f"{'='*50}")

    # Show top 10 files by token count
    results.sort(key=lambda x: x[1], reverse=True)
    print(f"\nTop 10 files by token count:")
    for i, (filename, tokens) in enumerate(results[:10], 1):
        print(f"{i:2d}. {filename}: {tokens:,} tokens")

def main():
    parser = argparse.ArgumentParser(description="Find maximum token count in conversation files")
    parser.add_argument("directory", help="Path to directory containing JSON conversation files")
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a valid directory")
        return

    find_max_tokens(args.directory)

if __name__ == "__main__":
    main()