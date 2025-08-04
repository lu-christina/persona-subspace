#!/usr/bin/env python3
"""
6_control_conversation.py

Takes a role-playing transcript and generates a "control" conversation by replacing
the roleplay instruction with a neutral system prompt "You are an AI assistant."
Uses the same model as the original transcript by default.

Usage:
    python 6_control_conversation.py <transcript_file> [options]
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add utils to path
sys.path.append('.')
sys.path.append('..')
sys.path.append('../utils')

from utils.inference_utils import load_vllm_model, continue_conversation, close_vllm_model


def load_transcript(filepath):
    """
    Load and parse JSON transcript file, extracting model info.
    
    Args:
        filepath: Path to JSON transcript file
        
    Returns:
        dict: Parsed transcript data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file isn't valid JSON
        KeyError: If required fields are missing
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Transcript file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        transcript = json.load(f)
    
    # Validate required fields
    required_fields = ['model', 'role', 'conversation']
    for field in required_fields:
        if field not in transcript:
            raise KeyError(f"Required field '{field}' missing from transcript")
    
    return transcript


def extract_user_messages(conversation):
    """
    Extract user messages starting from turn 2 onwards (skip initial roleplay instruction).
    
    Args:
        conversation: List of conversation turns (dicts with 'role' and 'content')
        
    Returns:
        list: User messages from index 2 onwards
    """
    user_messages = []
    
    # Start from index 2 to skip initial roleplay instruction
    for i in range(2, len(conversation)):
        turn = conversation[i]
        if turn['role'] == 'user':
            user_messages.append(turn['content'])
    
    return user_messages


def run_control_conversation(model_wrapper, user_messages, max_tokens=1000, temperature=0.7):
    """
    Run control conversation with neutral system prompt.
    
    Args:
        model_wrapper: VLLMModelWrapper instance
        user_messages: List of user message strings
        max_tokens: Maximum tokens per response
        temperature: Sampling temperature
        
    Returns:
        list: Complete conversation history with all turns
    """
    # Start with neutral system prompt
    conversation_history = []
    
    # Add the neutral system prompt as first user message
    # (this simulates the roleplay structure but with neutral content)
    first_response, conversation_history = continue_conversation(
        model_wrapper,
        conversation_history,
        "You are an AI assistant.",
        max_tokens=max_tokens,
        temperature=temperature
    )
    
    # Continue with remaining user messages
    for user_message in user_messages:
        response, conversation_history = continue_conversation(
            model_wrapper,
            conversation_history,
            user_message,
            max_tokens=max_tokens,
            temperature=temperature
        )
    
    return conversation_history


def generate_control_filepath(original_filepath, output_dir=None):
    """
    Generate output filepath with "_control" suffix.
    
    Args:
        original_filepath: Path to original transcript file
        output_dir: Optional output directory (default: same as input)
        
    Returns:
        str: Output filepath
    """
    original_path = Path(original_filepath)
    
    if output_dir:
        output_path = Path(output_dir)
    else:
        output_path = original_path.parent
    
    # Add "_control" before file extension
    new_filename = f"{original_path.stem}_control{original_path.suffix}"
    return str(output_path / new_filename)


def save_control_conversation(conversation_data, output_filepath):
    """
    Save control conversation to JSON file.
    
    Args:
        conversation_data: Dict with conversation metadata and history
        output_filepath: Path to save the file
    """
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    
    with open(output_filepath, 'w') as f:
        json.dump(conversation_data, f, indent=2)
    
    print(f"Control conversation saved to: {output_filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate control conversation from roleplay transcript",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        'transcript_file',
        help='Path to input JSON transcript file'
    )
    
    # Optional arguments
    parser.add_argument(
        '--model',
        help='Model name/identifier (default: use model from transcript)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=1000,
        help='Maximum tokens per response'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Sampling temperature'
    )
    parser.add_argument(
        '--max-model-len',
        type=int,
        default=4096,
        help='Maximum model sequence length'
    )
    parser.add_argument(
        '--tensor-parallel-size',
        type=int,
        help='Number of GPUs to use (default: auto-detect)'
    )
    parser.add_argument(
        '--output-dir',
        help='Output directory (default: same as input file)'
    )
    
    args = parser.parse_args()
    
    try:
        # Load transcript and extract model info
        print(f"Loading transcript from: {args.transcript_file}")
        transcript = load_transcript(args.transcript_file)
        
        # Use model from transcript unless overridden
        model_name = args.model if args.model else transcript['model']
        print(f"Using model: {model_name}")
        
        # Extract user messages (skip initial roleplay instruction)
        user_messages = extract_user_messages(transcript['conversation'])
        print(f"Extracted {len(user_messages)} user messages")
        
        # Load VLLM model
        print("Loading VLLM model...")
        model_kwargs = {
            'max_model_len': args.max_model_len
        }
        if args.tensor_parallel_size:
            model_kwargs['tensor_parallel_size'] = args.tensor_parallel_size
            
        model_wrapper = load_vllm_model(model_name, **model_kwargs)
        
        try:
            # Run control conversation
            print("Running control conversation...")
            conversation_history = run_control_conversation(
                model_wrapper,
                user_messages,
                max_tokens=args.max_tokens,
                temperature=args.temperature
            )
            
            # Prepare output data
            control_role = f"{transcript['role']}_control"
            conversation_data = {
                'model': model_name,
                'turns': len(conversation_history),
                'role': control_role,
                'conversation': conversation_history
            }
            
            # Generate output filepath and save
            output_filepath = generate_control_filepath(args.transcript_file, args.output_dir)
            save_control_conversation(conversation_data, output_filepath)
            
            print(f"Successfully generated control conversation:")
            print(f"  Original role: {transcript['role']}")
            print(f"  Control role: {control_role}")
            print(f"  Turns: {len(conversation_history)}")
            
        finally:
            # Clean up model
            print("Cleaning up model...")
            close_vllm_model(model_wrapper)
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()