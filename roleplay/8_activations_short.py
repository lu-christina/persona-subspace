#!/usr/bin/env python3
"""
Activation extraction script for single-shot roleplay conversations.

This script takes the single-shot transcript files created by 7_inference_short.py
and extracts full layer activations for each individual conversation using 
HuggingFace transformers and PyTorch hooks.

Usage:
    uv run roleplay/8_activations_short.py
"""

import json
import torch
import gc
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add utils to path for imports
sys.path.append('.')
sys.path.append('..')

from utils.probing_utils import load_model, extract_full_activations


def load_single_shot_transcript(file_path: str) -> Dict[str, Any]:
    """Load and parse a single-shot transcript JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Transcript file not found: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {file_path}: {e}")


def extract_conversations(transcript: Dict[str, Any]) -> List[List[Dict[str, str]]]:
    """
    Extract individual conversations from single-shot transcript.
    
    Args:
        transcript: Loaded transcript data with "conversations" field
        
    Returns:
        List of conversations, where each conversation is a list of message dicts
    """
    conversations = transcript.get("conversations", [])
    
    if not conversations:
        raise ValueError("No conversations found in transcript")
    
    # Validate format
    for i, conversation in enumerate(conversations):
        if not isinstance(conversation, list) or len(conversation) != 2:
            raise ValueError(f"Conversation {i} should have exactly 2 messages (user + assistant)")
        
        if conversation[0]["role"] != "user" or conversation[1]["role"] != "assistant":
            raise ValueError(f"Conversation {i} should have user message followed by assistant message")
    
    return conversations


def extract_activations_for_conversations(
    model, 
    tokenizer, 
    conversations: List[List[Dict[str, str]]]
) -> List[torch.Tensor]:
    """
    Extract full layer activations for each conversation.
    
    Args:
        model: Loaded HuggingFace model
        tokenizer: Loaded HuggingFace tokenizer
        conversations: List of conversation message lists
        
    Returns:
        List of tensors, each with shape (n_layers, n_tokens, hidden_size)
    """
    activations_list = []
    
    for i, conversation in enumerate(conversations):
        print(f"  Processing conversation {i+1}/{len(conversations)}...")
        
        try:
            # Extract full activations for this conversation
            # extract_full_activations expects layer=None for all layers
            activations = extract_full_activations(
                model=model,
                tokenizer=tokenizer, 
                conversation=conversation,
                layer=None  # Extract from all layers
            )
            
            activations_list.append(activations)
            print(f"    ✓ Extracted activations with shape: {activations.shape}")
            
            # Clean up GPU memory after each conversation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"    ✗ Error processing conversation {i+1}: {e}")
            # Continue with remaining conversations rather than failing completely
            continue
    
    return activations_list


def save_activations(activations: List[torch.Tensor], output_path: str) -> None:
    """Save list of activation tensors to file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(activations, output_path)
    print(f"  Saved {len(activations)} activation tensors to {output_path}")


def process_transcript_file(input_path: str, output_path: str, model, tokenizer) -> None:
    """
    Process a single transcript file to extract activations.
    
    Args:
        input_path: Path to input single-shot transcript JSON file
        output_path: Path to output activation tensor file
        model: Loaded HuggingFace model
        tokenizer: Loaded HuggingFace tokenizer
    """
    print(f"Processing {input_path}...")
    
    # Load transcript
    transcript = load_single_shot_transcript(input_path)
    
    # Extract conversations
    conversations = extract_conversations(transcript)
    print(f"  Found {len(conversations)} single-shot conversations")
    
    # Extract activations for all conversations
    activations_list = extract_activations_for_conversations(model, tokenizer, conversations)
    
    if not activations_list:
        print(f"  ⚠️  No activations extracted for {input_path}")
        return
    
    print(f"  Successfully extracted activations for {len(activations_list)} conversations")
    
    # Save activations
    save_activations(activations_list, output_path)


def main():
    """Main function to process all single-shot transcript files."""
    
    # Define input and output paths
    transcript_dir = Path("/root/git/persona-subspace/roleplay/results/gemma-2-27b/role_vectors/transcripts")
    output_dir = Path("/workspace/roleplay/gemma-2-27b")
    
    # List of single-shot transcript files to process
    transcript_files = [
        "anxious_teenager_single.json",
        "anxious_teenager_control_single.json",
        "deep_sea_leviathan_single.json", 
        "deep_sea_leviathan_control_single.json",
        "medieval_bard_single.json",
        "medieval_bard_control_single.json"
    ]
    
    print("=== Single-Shot Roleplay Activation Extraction ===")
    print(f"Processing {len(transcript_files)} transcript files...")
    print(f"Output directory: {output_dir}")
    
    try:
        # Load model and tokenizer
        print("\nLoading model: google/gemma-2-27b-it")
        model, tokenizer = load_model("google/gemma-2-27b-it")
        print("✓ Model loaded successfully!")
        
        # Process each transcript file
        for transcript_file in transcript_files:
            input_path = transcript_dir / transcript_file
            
            # Create output filename by changing extension to .pt
            output_filename = transcript_file.replace(".json", ".pt")
            output_path = output_dir / output_filename
            
            try:
                process_transcript_file(str(input_path), str(output_path), model, tokenizer)
                print(f"✓ Completed {transcript_file}")
            except Exception as e:
                print(f"✗ Error processing {transcript_file}: {e}")
                continue
            
            # Force garbage collection between files
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print(f"\n🎉 All transcript processing completed!")
        print(f"Activation files saved to: {output_dir}")
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1
    
    finally:
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("🧹 Cleaned up GPU memory")
    
    return 0


if __name__ == "__main__":
    exit(main())