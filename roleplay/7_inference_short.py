#!/usr/bin/env python3
"""
Single-shot inference script for role-playing personas using vLLM.

This script takes existing multi-turn roleplay conversations and converts them
to single-shot responses by combining the initial role prompt with each user 
question individually. 

Usage:
    uv run roleplay/7_inference_short.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Add utils to path for imports
sys.path.append('.')
sys.path.append('..')

from utils.inference_utils import load_vllm_model, batch_chat, close_vllm_model, cleanup_all_models


def load_transcript(file_path: str) -> Dict[str, Any]:
    """Load and parse a transcript JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Transcript file not found: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {file_path}: {e}")


def extract_role_and_questions(transcript: Dict[str, Any]) -> Tuple[str, List[str]]:
    """
    Extract the initial role prompt and all user questions from a conversation.
    
    Args:
        transcript: Loaded transcript data
        
    Returns:
        Tuple of (role_prompt, list_of_user_questions)
    """
    conversation = transcript.get("conversation", [])
    
    if not conversation:
        raise ValueError("No conversation found in transcript")
    
    # First message should be the role assignment
    if conversation[0]["role"] != "user":
        raise ValueError("First message should be a user message with role assignment")
    
    role_prompt = conversation[0]["content"]
    
    # Extract all subsequent user questions (skip the first role assignment)
    user_questions = []
    for turn in conversation[1:]:
        if turn["role"] == "user":
            user_questions.append(turn["content"])
    
    return role_prompt, user_questions


def create_single_shot_prompts(role_prompt: str, questions: List[str]) -> List[str]:
    """
    Combine the role prompt with each question to create single-shot prompts.
    
    Args:
        role_prompt: Initial role assignment (e.g., "You are an anxious teenager.")
        questions: List of user questions
        
    Returns:
        List of combined prompts
    """
    single_shot_prompts = []
    for question in questions:
        # Combine role and question
        combined_prompt = f"{role_prompt}\n\n{question}"
        single_shot_prompts.append(combined_prompt)
    
    return single_shot_prompts


def format_single_shot_results(
    role_prompt: str, 
    questions: List[str], 
    responses: List[str],
    original_transcript: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Format single-shot results into the same structure as original transcripts.
    
    Args:
        role_prompt: The initial role assignment
        questions: List of user questions  
        responses: List of model responses
        original_transcript: Original transcript for metadata
        
    Returns:
        Formatted result dictionary
    """
    conversations = []
    
    for question, response in zip(questions, responses):
        conversation = [
            {"role": "user", "content": f"{role_prompt}\n\n{question}"},
            {"role": "assistant", "content": response}
        ]
        conversations.append(conversation)
    
    # Create result structure similar to original
    result = {
        "model": original_transcript.get("model", "google/gemma-2-27b-it"),
        "turns": len(questions) * 2,  # Each single-shot is 2 turns (user + assistant)
        "role": original_transcript.get("role", "unknown"),
        "conversations": conversations,  # Note: plural "conversations" instead of singular "conversation"
        "original_turns": original_transcript.get("turns", 0),
        "conversion_timestamp": datetime.now().isoformat(),
        "conversion_type": "multi_turn_to_single_shot"
    }
    
    return result


def save_single_shot_results(results: Dict[str, Any], output_path: str) -> None:
    """Save single-shot results to JSON file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def process_role_transcript(transcript_path: str, output_path: str, model_wrapper) -> None:
    """
    Process a single role transcript file to create single-shot responses.
    
    Args:
        transcript_path: Path to input transcript JSON file
        output_path: Path to output single-shot JSON file  
        model_wrapper: vLLM model wrapper for inference
    """
    print(f"Processing {transcript_path}...")
    
    # Load transcript
    transcript = load_transcript(transcript_path)
    
    # Extract role and questions
    role_prompt, questions = extract_role_and_questions(transcript)
    
    print(f"  Found role: {role_prompt[:50]}...")
    print(f"  Found {len(questions)} questions")
    
    if not questions:
        print("  No questions found, skipping...")
        return
    
    # Create single-shot prompts
    single_shot_prompts = create_single_shot_prompts(role_prompt, questions)
    
    # Run batch inference
    print(f"  Running batch inference on {len(single_shot_prompts)} prompts...")
    responses = batch_chat(
        model_wrapper=model_wrapper,
        messages=single_shot_prompts,
        temperature=0.7,
        max_tokens=512,
        top_p=0.9,
        progress=True
    )
    
    print(f"  Generated {len(responses)} responses")
    
    # Format results
    results = format_single_shot_results(role_prompt, questions, responses, transcript)
    
    # Save results
    save_single_shot_results(results, output_path)
    print(f"  Saved results to {output_path}")


def main():
    """Main function to process all role transcripts."""
    
    # Define input and output paths
    transcript_dir = Path("/root/git/persona-subspace/roleplay/results/gemma-2-27b/role_vectors/transcripts")
    
    # List of transcript files to process
    transcript_files = [
        "anxious_teenager.json",
        "anxious_teenager_control.json", 
        "deep_sea_leviathan.json",
        "deep_sea_leviathan_control.json",
        "medieval_bard.json",
        "medieval_bard_control.json"
    ]
    
    print("=== Single-Shot Roleplay Inference ===")
    print(f"Processing {len(transcript_files)} transcript files...")
    
    try:
        # Load vLLM model
        print("Loading vLLM model: google/gemma-2-27b-it")
        model_wrapper = load_vllm_model(
            model_name="google/gemma-2-27b-it",
            max_model_len=4096,
            tensor_parallel_size=None,  # Auto-detect
            gpu_memory_utilization=0.9
        )
        
        print("Model loaded successfully!")
        
        # Process each transcript file
        for transcript_file in transcript_files:
            input_path = transcript_dir / transcript_file
            
            # Create output filename by inserting "_single" before file extension
            output_filename = transcript_file.replace(".json", "_single.json")
            output_path = transcript_dir / output_filename
            
            try:
                process_role_transcript(str(input_path), str(output_path), model_wrapper)
            except Exception as e:
                print(f"Error processing {transcript_file}: {e}")
                continue
        
        print("\n✅ All transcript processing completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    finally:
        # Clean up model resources
        if 'model_wrapper' in locals():
            print("Cleaning up model resources...")
            close_vllm_model(model_wrapper)
        
        # Ensure all resources are cleaned up
        print("Performing final cleanup...")
        cleanup_all_models()
    
    return 0


if __name__ == "__main__":
    exit(main())