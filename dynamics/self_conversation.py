#!/usr/bin/env python3
"""
Self-conversation script that uses a model to talk to itself.

This script creates conversations where a single model alternates between two conversation
perspectives, effectively having a conversation with itself across multiple turns.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import re

# Add utils directory to path
utils_path = Path(__file__).parent.parent / "utils"
sys.path.append(str(utils_path))

try:
    from inference_utils import load_vllm_model, chat_conversation, close_vllm_model
    from tqdm import tqdm
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure you're running from the correct directory and have installed dependencies.")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate self-conversations where a model talks to itself"
    )
    
    # Core arguments
    parser.add_argument(
        "--model-name", 
        type=str, 
        default="Qwen/Qwen3-32B",
        help="HuggingFace model identifier"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="/root/git/persona-subspace/dynamics/results/qwen-3-32b",
        help="Output directory for conversation JSON files"
    )
    parser.add_argument(
        "--initial-message", 
        type=str, 
        default="You will be having a conversation with another instance of yourself. Feel free to discuss whatever you like. Do not say goodbye or end the conversation at any point.",
        help="Initial message to start the conversation"
    )
    parser.add_argument(
        "--num-turns", 
        type=int, 
        default=10,
        help="Number of turns in each conversation"
    )
    parser.add_argument(
        "--num-transcripts", 
        type=int, 
        default=1,
        help="Number of conversation transcripts to generate"
    )
    parser.add_argument(
        "--continue-file", 
        type=str, 
        default=None,
        help="Optional path to existing conversation JSON file to continue from"
    )
    
    # vLLM model parameters
    parser.add_argument(
        "--max-model-len", 
        type=int, 
        default=40960,
        help="Maximum sequence length for the model (default: 8192)"
    )
    parser.add_argument(
        "--tensor-parallel-size", 
        type=int, 
        default=None,
        help="Number of GPUs to use for tensor parallelism (default: auto-detect)"
    )
    parser.add_argument(
        "--gpu-memory-utilization", 
        type=float, 
        default=0.9,
        help="GPU memory utilization ratio (default: 0.9)"
    )
    
    # Generation parameters
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--max-tokens", 
        type=int, 
        default=1024,
        help="Maximum tokens per response (default: 1024)"
    )
    parser.add_argument(
        "--top-p", 
        type=float, 
        default=0.9,
        help="Top-p sampling parameter (default: 0.9)"
    )
    
    return parser.parse_args()


def get_next_file_id(output_dir: Path) -> int:
    """Find the next available file ID by checking existing files."""
    if not output_dir.exists():
        return 1
    
    max_id = 0
    pattern = re.compile(r'self_conversation_(\d+)\.json')
    
    for file_path in output_dir.glob("self_conversation_*.json"):
        match = pattern.match(file_path.name)
        if match:
            file_id = int(match.group(1))
            max_id = max(max_id, file_id)
    
    return max_id + 1


def create_output_directory(base_dir: str, num_turns: int) -> Path:
    """Create output directory structure using actual conversation turns."""
    output_path = Path(base_dir) / f"{num_turns}_turns"
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def load_existing_conversation(file_path: str) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """
    Load an existing conversation from a JSON file.
    
    Args:
        file_path: Path to the JSON conversation file
        
    Returns:
        Tuple of (conversation_history, original_metadata)
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is invalid
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'conversation' not in data:
            raise ValueError("Invalid conversation file: missing 'conversation' key")
        
        conversation = data['conversation']
        if not isinstance(conversation, list):
            raise ValueError("Invalid conversation file: 'conversation' must be a list")
        
        # Validate conversation format
        for i, message in enumerate(conversation):
            if not isinstance(message, dict) or 'role' not in message or 'content' not in message:
                raise ValueError(f"Invalid message format at index {i}")
            if message['role'] not in ['user', 'assistant']:
                raise ValueError(f"Invalid role '{message['role']}' at index {i}")
        
        # Extract metadata (everything except conversation)
        metadata = {k: v for k, v in data.items() if k != 'conversation'}
        
        return conversation, metadata
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Continue file not found: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in continue file: {e}")
    except Exception as e:
        raise ValueError(f"Error loading continue file: {e}")


def generate_self_conversation(
    model_wrapper,
    initial_message: str,
    num_turns: int,
    temperature: float = 0.7,
    max_tokens: int = 512,
    top_p: float = 0.9,
    existing_conversation: List[Dict[str, str]] = None
) -> List[Dict[str, str]]:
    """
    Generate a self-conversation where the model talks to itself.
    
    Args:
        model_wrapper: The loaded model wrapper
        initial_message: Initial message (used only if existing_conversation is None)
        num_turns: Target total number of conversation turns (user+assistant pairs)
        temperature: Sampling temperature
        max_tokens: Max tokens per response
        top_p: Top-p sampling
        existing_conversation: Optional existing conversation to continue from
    
    Returns:
        The conversation from perspective A (complete conversation history).
    """
    if existing_conversation is not None:
        # Continue from existing conversation
        conv_a = existing_conversation.copy()
        
        # Count current turns (each turn = user message + assistant response pair)
        user_messages = len([msg for msg in conv_a if msg['role'] == 'user'])
        assistant_messages = len([msg for msg in conv_a if msg['role'] == 'assistant'])
        current_turns = min(user_messages, assistant_messages)
        
        if current_turns >= num_turns:
            # Truncate to desired number of complete turns (user+assistant pairs)
            truncated_conv = []
            turn_count = 0
            for msg in conv_a:
                truncated_conv.append(msg)
                if msg['role'] == 'assistant':
                    turn_count += 1
                    if turn_count >= num_turns:
                        break
            return truncated_conv
        
        # Continue from existing conversation
        conv_a = conv_a.copy()
        
        # Determine which perspective should continue based on last message
        if conv_a[-1]['role'] == 'assistant':
            # Last message was assistant, so perspective B should continue
            conv_b = [{"role": "user", "content": conv_a[-1]['content']}]
        else:
            # Last message was user, need to complete with assistant first
            conv_b = []
    else:
        # Start new conversation
        conv_a = [{"role": "user", "content": initial_message}]
        conv_b = []
    
    # Generate messages until conv_a has num_turns complete turns (num_turns * 2 messages)
    target_messages = num_turns * 2
    
    # Keep track of which conversation we're currently generating for
    current_conv_is_a = True
    
    try:
        while len(conv_a) < target_messages:
            if current_conv_is_a:
                # Generate response for conversation A
                response = chat_conversation(
                    model_wrapper,
                    conv_a,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p
                )
                conv_a.append({"role": "assistant", "content": response})
                
                # Add this response as user message to conversation B (if not the final message)
                if len(conv_a) < target_messages:
                    conv_b.append({"role": "user", "content": response})
            else:
                # Generate response for conversation B
                response = chat_conversation(
                    model_wrapper,
                    conv_b,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p
                )
                conv_b.append({"role": "assistant", "content": response})
                
                # Add this response as user message to conversation A
                conv_a.append({"role": "user", "content": response})
            
            # Swap which conversation we're generating for
            current_conv_is_a = not current_conv_is_a
            
    except Exception as e:
        # Attach the partial conversation to the exception so it can be saved
        e.partial_conversation = conv_a
        raise e
    
    # Return conversation A (the complete conversation)
    return conv_a


def save_conversation(
    conversation: List[Dict[str, str]],
    output_path: Path,
    file_id: int,
    metadata: Dict[str, Any]
) -> None:
    """Save conversation to JSON file."""
    output_data = {
        **metadata,
        "conversation": conversation
    }
    
    filename = f"self_conversation_{file_id}.json"
    filepath = output_path / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


def main():
    """Main execution function."""
    args = parse_args()
    
    # Handle continue file if provided
    existing_conversation = None
    original_metadata = {}
    
    if args.continue_file:
        print(f"Loading existing conversation from: {args.continue_file}")
        try:
            existing_conversation, original_metadata = load_existing_conversation(args.continue_file)
            current_turns = len([msg for msg in existing_conversation if msg['role'] == 'user'])
            print(f"Loaded conversation with {current_turns} turns ({len(existing_conversation)} messages)")
            
            if current_turns >= args.num_turns:
                print(f"Note: Existing conversation has {current_turns} turns, will truncate to {args.num_turns}")
            else:
                print(f"Will continue from {current_turns} to {args.num_turns} turns")
                
        except (FileNotFoundError, ValueError) as e:
            print(f"Error loading continue file: {e}")
            sys.exit(1)
    
    # Create output directory
    output_dir = create_output_directory(args.output_dir, args.num_turns)
    print(f"Output directory: {output_dir}")
    
    # Get starting file ID
    starting_id = get_next_file_id(output_dir)
    print(f"Starting from ID: {starting_id}")
    
    # Prepare metadata (merge original with new parameters)
    metadata = {
        **original_metadata,  # Keep original metadata as base
        "model": args.model_name,  # Override with current parameters
        "turns": args.num_turns,
        "max_model_len": args.max_model_len,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "top_p": args.top_p
    }
    
    # Add continue file info to metadata if continuing
    if args.continue_file:
        metadata["continued_from"] = args.continue_file
    
    # Load model
    print(f"Loading model: {args.model_name}")
    try:
        model_wrapper = load_vllm_model(
            model_name=args.model_name,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization
        )
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    try:
        # Generate conversations
        if existing_conversation:
            print(f"Continuing {args.num_transcripts} conversations from existing base...")
        else:
            print(f"Generating {args.num_transcripts} new conversations with {args.num_turns} turns each...")
        
        for i in tqdm(range(args.num_transcripts), desc="Generating conversations"):
            file_id = starting_id + i
            
            # For multiple transcripts, only use existing_conversation for the first one
            # Subsequent transcripts should start fresh unless explicitly continuing
            use_existing = existing_conversation if i == 0 else None
            
            try:
                # Generate conversation
                conversation = generate_self_conversation(
                    model_wrapper,
                    args.initial_message,
                    args.num_turns,
                    args.temperature,
                    args.max_tokens,
                    args.top_p,
                    existing_conversation=use_existing
                )
                
                # Save conversation with original metadata
                save_conversation(conversation, output_dir, file_id, metadata)
                tqdm.write(f"Saved conversation {file_id} ({len(conversation)} messages)")
                
            except Exception as e:
                tqdm.write(f"Error during generation of conversation {file_id}: {e}")
                
                # If we have a partial conversation, try to save it
                try:
                    # Try to get whatever conversation was generated up to the error point
                    # This will depend on where the error occurred in generate_self_conversation
                    partial_conversation = getattr(e, 'partial_conversation', None)
                    if partial_conversation is None and use_existing:
                        # If we were continuing from existing, at least save that
                        partial_conversation = use_existing
                    
                    if partial_conversation and len(partial_conversation) > 0:
                        # Calculate actual turns completed
                        actual_turns = len([msg for msg in partial_conversation if msg['role'] == 'user'])
                        
                        # Create corrected metadata
                        corrected_metadata = metadata.copy()
                        corrected_metadata["turns"] = actual_turns
                        corrected_metadata["completed_turns"] = actual_turns
                        corrected_metadata["requested_turns"] = args.num_turns
                        corrected_metadata["error"] = str(e)
                        corrected_metadata["status"] = "partial"
                        
                        # Save partial conversation
                        save_conversation(partial_conversation, output_dir, file_id, corrected_metadata)
                        tqdm.write(f"Saved partial conversation {file_id} ({len(partial_conversation)} messages, {actual_turns} turns)")
                    else:
                        tqdm.write(f"No partial conversation available to save for {file_id}")
                        
                except Exception as save_error:
                    tqdm.write(f"Failed to save partial conversation {file_id}: {save_error}")
                
                # Continue with next conversation instead of stopping entirely
                continue
        
        print(f"\nCompleted! Generated {args.num_transcripts} conversations.")
        print(f"Files saved in: {output_dir}")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error during generation: {e}")
    finally:
        # Clean up model
        print("Cleaning up model...")
        close_vllm_model(model_wrapper)
        print("Done!")


if __name__ == "__main__":
    main()